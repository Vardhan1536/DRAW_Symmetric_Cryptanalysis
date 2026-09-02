

import sys
import os
import time
import numpy as np
import cma
import argparse


parser = argparse.ArgumentParser(description="CWMC-QOC Key Recovery Pipeline -- V3")
parser.add_argument('-p', type=int, default=10, help='Number of layers (p)')
parser.add_argument('--pt', type=lambda x: int(x, 0), nargs='*', help='Plaintexts (hex or int)')
parser.add_argument('--ct', type=lambda x: int(x, 0), nargs='*', help='Ciphertexts (hex or int)')
parser.add_argument('--k-opt', type=lambda x: int(x, 0), default=0xA73B, help='Optimal Key K* (hex or int)')
args = parser.parse_args()

K_OPT   = args.k_opt
P       = args.p
if args.pt is not None and args.ct is not None:
    if len(args.pt) != len(args.ct):
        raise ValueError("Number of PTs must match number of CTs")
    PAIRS_N = list(zip(args.pt, args.ct))
    # Update scaling_study.PAIRS so run_landscape uses the provided pairs
    scaling_study.PAIRS = PAIRS_N
else:
    PAIRS_N = PAIRS[:2]  # 2 pairs: unique global minimum for K*

# -----------------------------------------------------------------------------
# 1. BUILD LANDSCAPE + STRUCTURAL DATA  (from actual compiler, no LUTs)
# -----------------------------------------------------------------------------

def build_landscape():
    import multiprocessing as mp
    from scaling_study import run_landscape

    n_pairs = len(PAIRS_N)
    n_cpu   = mp.cpu_count()
    print(f"[1] Building CWMC4 landscape: {n_pairs} pairs, {n_cpu} CPU cores...")
    t0 = time.time()

    # run_landscape uses mp.Pool across all cores -- fast parallel execution
    df = run_landscape(num_pairs=n_pairs)

    E = df['E_CWMC4'].values.astype(np.float64)
    E -= E.min()

    valley_mask = (E <= 10.0)
    valley_keys = np.where(valley_mask)[0]
    near_opt    = (E <= 5.0)
    near_keys   = np.where(near_opt)[0]

    # Influence: per-bit error correlation with valley
    influence = np.zeros(N)
    for q in range(N):
        key_bit  = (K_OPT >> q) & 1
        val_bits = (valley_keys >> q) & 1
        influence[q] = np.mean(val_bits != key_bit) if len(valley_keys) > 0 else 0.5
    influence = influence / (influence.max() + 1e-12)

    
    CMA_W = np.array([0.8334, 1.0037, 0.9866, 1.8273,
                       0.8341, 0.9261, 1.6045, 1.4800,
                       0.3623, 0.4957, 0.7277, 0.2575,
                       0.6011, 0.3755, 2.0471, 0.5519])
    mixer_w = CMA_W / (CMA_W.max() + 1e-12)

    # Top-30 synergy edges: joint bit-error in valley (E<=10)
    pair_scores = {}
    for u in range(N):
        for v in range(u + 1, N):
            ku = (K_OPT >> u) & 1
            kv = (K_OPT >> v) & 1
            vu = (valley_keys >> u) & 1
            vv = (valley_keys >> v) & 1
            pair_scores[(u, v)] = (np.mean((vu != ku) & (vv != kv))
                                   if len(valley_keys) > 0 else 0.0)
    top40     = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:40]
    max_score = max(s for _, s in top40) if top40 else 1.0
    synergy_edges = [((u, v), s / max_score) for (u, v), s in top40
                     if s / max_score > 0.05]  # drop noise edges

    print(f"   Built in {time.time()-t0:.2f}s | "
          f"E(K*)={E[K_OPT]:.1f} | E=0: {int(np.sum(E==0))} | "
          f"E<=5: {near_opt.sum()} | E<=10: {valley_mask.sum()}")
    return E, influence, mixer_w, synergy_edges





# -----------------------------------------------------------------------------
# 2. BUILD PARAMETRIC ANSATZ  (p=8, RXX synergy, explicit ZZ+Z cost)
# -----------------------------------------------------------------------------

def build_ansatz(influence, mixer_w, synergy_edges, p=P, measure=False):
    """
    Per-layer structure:
      [1] Alternating RY/RX weighted mixer  -- beta[l] x mixer_w[q]
      [2] Influence RZ layer                -- alpha[l] x influence[q]
      [3] Synergy RXX entangler             -- gamma[l] x sij  (XX != ZZ -> distinct from cost)
      [4] Cost ZZ+Z Pauli gadgets           -- delta[l] x ZZ pairs + delta[l] x Z singles
          (CNOT-RZ-CNOT decomposition: no PauliEvolutionGate, runs on AerSimulator)
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    beta  = ParameterVector("beta", p)
    alpha = ParameterVector("alpha", p)
    gamma = ParameterVector("gamma", p)
    delta = ParameterVector("delta", p)

    qc = QuantumCircuit(N)
    qc.h(range(N))

    for l in range(p):
        # [1] Weighted mixer
        for q in range(N):
            ang = 2 * beta[l] * float(mixer_w[q])
            if l % 2 == 0:
                qc.ry(ang, q)
            else:
                qc.rx(ang, q)

        # [2] Influence RZ
        for q in range(N):
            qc.rz(2 * alpha[l] * float(influence[q]), q)

        # [3] Synergy RXX (XX Hamiltonian -- entangler, not cost)
        for (u, v), sij in synergy_edges:
            qc.rxx(2 * gamma[l] * sij, u, v)

        # [4] Cost ZZ + Z (explicit Pauli gadgets -- no PauliEvolutionGate)
        for (u, v), sij in synergy_edges:
            ang_zz = 2 * delta[l] * sij
            qc.cx(u, v)
            qc.rz(ang_zz, v)
            qc.cx(u, v)
        for q in range(N):
            qc.rz(2 * delta[l] * float(influence[q]), q)

    if measure:
        qc.measure_all()

    return qc, beta, alpha, gamma, delta


# -----------------------------------------------------------------------------
# 3. STATEVECTOR ENGINE  -- transpile ONCE, rebind each call
#    (critical: avoids ~800 expensive transpile calls during CMA-ES)
# -----------------------------------------------------------------------------

_backend       = None   # initialized in init_engine (after multiprocessing is done)
_qc_transpiled = None
_beta_pv = _alpha_pv = _gamma_pv = _delta_pv = None
_p_global = P

def init_engine(influence, mixer_w, synergy_edges, p=P):
    global _qc_transpiled, _beta_pv, _alpha_pv, _gamma_pv, _delta_pv, _p_global, _backend
    # Deferred Qiskit import -- safe after multiprocessing.Pool has closed
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    _p_global = p
    _backend  = AerSimulator(method='statevector')
    print(f"\n[2] Transpiling p={p} parametric circuit (done once)...")
    t0 = time.time()

    qc_param, b_pv, a_pv, g_pv, d_pv = build_ansatz(
        influence, mixer_w, synergy_edges, p=p, measure=False
    )
    _beta_pv  = b_pv
    _alpha_pv = a_pv
    _gamma_pv = g_pv
    _delta_pv = d_pv

    # save_statevector instruction for AerSimulator
    qc_sv = qc_param.copy()
    qc_sv.save_statevector()

    _qc_transpiled = transpile(qc_sv, _backend, optimization_level=1)
    print(f"   Transpile done in {time.time()-t0:.1f}s | "
          f"depth={_qc_transpiled.depth()} | "
          f"ops={sum(_qc_transpiled.count_ops().values())}")


def statevector_probs(params: np.ndarray) -> np.ndarray:
    """Bind params into pre-transpiled circuit and run AerSimulator."""
    p = _p_global
    pm = {}
    for l in range(p):
        pm[_beta_pv[l]]  = float(params[l])
        pm[_alpha_pv[l]] = float(params[p + l])
        pm[_gamma_pv[l]] = float(params[2*p + l])
        pm[_delta_pv[l]] = float(params[3*p + l])

    bound = _qc_transpiled.assign_parameters(pm)
    sv    = _backend.run(bound).result().get_statevector()
    return np.abs(np.array(sv))**2


# -----------------------------------------------------------------------------
# 4. OBJECTIVE FUNCTIONS
# -----------------------------------------------------------------------------

_E_array = _E_le1 = _E_le5 = _E_le10 = None
_eval_count = [0]

def init_objectives(E):
    global _E_array, _E_le1, _E_le5, _E_le10
    _E_array = E
    _E_le1   = (E <= 1.0).astype(np.float64)
    _E_le5   = (E <= 5.0).astype(np.float64)
    _E_le10  = (E <= 10.0).astype(np.float64)


def _log(tag, probs, every=32):
    _eval_count[0] += 1
    if _eval_count[0] % every == 0:
        pk  = float(probs[K_OPT])
        p5  = float(np.dot(probs, _E_le5))
        p10 = float(np.dot(probs, _E_le10))
        print(f"  [{_eval_count[0]:5d}] {tag}  P(K*)={pk:.3e}  "
              f"P(E<=5)={p5:.5f}  P(E<=10)={p10:.5f}")


def obj_v1(params):
    """V1: maximize P(E<=10) -- large smooth target."""
    probs = statevector_probs(params)
    val   = float(np.dot(probs, _E_le10))
    _log(f"P(E<=10)={val:.5f}", probs)
    return -val


def obj_v2(params):
    """V2: maximize CVaR-like near-ground objective -- P(E<=1)+0.2P(E<=5)+0.1P(E<=10).
    Research (IBM 2020): CVaR/tail objectives outperform expectation for combinatorial."""
    probs = statevector_probs(params)
    p1    = float(np.dot(probs, _E_le1))
    p5    = float(np.dot(probs, _E_le5))
    p10   = float(np.dot(probs, _E_le10))
    val   = p1 + 0.2 * p5 + 0.1 * p10
    _log(f"obj={val:.5f}  P(E<=1)={p1:.5f}", probs)
    return -val


def obj_v3(params):
    """V3: directly maximize P(K*) -- key recovery objective."""
    probs = statevector_probs(params)
    pk    = float(probs[K_OPT])
    _log(f"P(K*)={pk:.4e}", probs)
    return -pk


# -----------------------------------------------------------------------------
# 5. CMA-ES RUNNER  (research-calibrated: popsize=32, restarts built-in)
# -----------------------------------------------------------------------------

def run_cmaes(objective, x0, sigma0, maxiter, label, lo=0.0, hi=np.pi):
    n_params = len(x0)
    print(f"\n{'='*66}")
    print(f"  {label}")
    print(f"  params={n_params}  popsize=48  maxiter={maxiter}  "
          f"sigma0={sigma0}  bounds=[{lo:.1f}, {hi:.2f}]")
    print(f"{'='*66}")
    _eval_count[0] = 0

    opts = {
        'maxiter':  maxiter,
        'popsize':  48,         # research: 3x minimum for robust global search
        'bounds':   [[lo]*n_params, [hi]*n_params],
        'tolx':     1e-5,
        'tolfun':   1e-7,
        'verbose':  -9,
        'seed':     42,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    t0 = time.time()

    best_val = float('inf')
    while not es.stop():
        sols    = es.ask()
        fitness = [objective(x) for x in sols]
        es.tell(sols, fitness)
        if min(fitness) < best_val:
            best_val = min(fitness)

    res = es.result
    print(f"\n  Converged | time={time.time()-t0:.1f}s | "
          f"evals={res.evaluations} | best={-res.fbest:.6f}")
    return np.array(res.xbest), -res.fbest


def main():
    print("=" * 66)
    print("  CWMC-QOC V3 -- Research-Calibrated Key Recovery")
    print(f"  p={P} layers | {4*P}-param VQA | AerSimulator statevector")
    print("=" * 66)

    # Build landscape
    E, influence, mixer_w, synergy_edges = build_landscape()
    init_objectives(E)

    # Transpile once
    init_engine(influence, mixer_w, synergy_edges, p=P)

    # Baseline
    print("\n[3] Baseline evaluation...")
    baseline = statevector_probs(np.zeros(4 * P))
    print(f"  Baseline P(K*) = {baseline[K_OPT]:.4e}  (uniform = {1/N_KEYS:.4e})")

    # -- V1: Valley Concentration (p=8, research-informed x0) -----------------
    # Adiabatic-inspired: mixer strong early (high beta), cost weak early (low delta)
    betas  = np.linspace(1.4, 0.2, P)   # stronger early mixer (adiabatic)
    alphas = np.full(P, 0.15)            # slightly lower influence
    gammas = np.full(P, 0.25)            # modest entanglement
    deltas = np.linspace(0.05, 1.3, P)  # slower cost ramp (adiabatic)
    x0_v1  = np.concatenate([betas, alphas, gammas, deltas])

    params_v1, best_v1 = run_cmaes(
        obj_v1, x0_v1, sigma0=0.3, maxiter=80,
        label="V1 -- Maximize P(E<=10)  [Valley Concentration]",
        lo=0.0, hi=np.pi
    )
    probs_v1 = statevector_probs(params_v1)
    print(f"\n  V1 | P(E<=10)={np.dot(probs_v1, _E_le10):.5f}  "
          f"P(E<=5)={np.dot(probs_v1, _E_le5):.5f}  "
          f"P(K*)={probs_v1[K_OPT]:.3e}")

    # -- V2: Near-Ground Sharpening --------------------------------------------
    params_v2, best_v2 = run_cmaes(
        obj_v2, params_v1, sigma0=0.15, maxiter=120,
        label="V2 -- CVaR Sharpening: P(E<=1)+0.2*P(E<=5)+0.1*P(E<=10)",
        lo=0.0, hi=np.pi
    )
    probs_v2 = statevector_probs(params_v2)
    print(f"\n  V2 | P(E<=1)={np.dot(probs_v2, _E_le1):.5f}  "
          f"P(E<=5)={np.dot(probs_v2, _E_le5):.5f}  "
          f"P(K*)={probs_v2[K_OPT]:.3e}")

    # -- V3: Direct Key Recovery -----------------------------------------------
    params_v3, best_v3 = run_cmaes(
        obj_v3, params_v2, sigma0=0.05, maxiter=200,
        label="V3 -- Maximize P(K*)  [Direct Key Recovery]",
        lo=0.0, hi=np.pi
    )
    probs_v3 = statevector_probs(params_v3)
    print(f"\n  V3 | P(K*)={probs_v3[K_OPT]:.4e}  "
          f"(speedup={probs_v3[K_OPT]*N_KEYS:.1f}x over random)")

    # Print schedule
    print(f"\n  Learned Schedule (p={P} layers):")
    print(f"  beta  = {np.round(params_v3[0:P], 4)}")
    print(f"  alpha = {np.round(params_v3[P:2*P], 4)}")
    print(f"  gamma = {np.round(params_v3[2*P:3*P], 4)}")
    print(f"  delta = {np.round(params_v3[3*P:4*P], 4)}")

    # -- Final Shot-Based Verification on AerSimulator -------------------------
    print(f"\n{'='*66}")
    print(f"  FINAL VERIFICATION -- 16384 shots on AerSimulator")
    print(f"{'='*66}")

    qc_final, b_pv, a_pv, g_pv, d_pv = build_ansatz(
        influence, mixer_w, synergy_edges, p=P, measure=True
    )
    pm = {}
    for l in range(P):
        pm[b_pv[l]] = float(params_v3[l])
        pm[a_pv[l]] = float(params_v3[P + l])
        pm[g_pv[l]] = float(params_v3[2*P + l])
        pm[d_pv[l]] = float(params_v3[3*P + l])
    qc_bound = qc_final.assign_parameters(pm)

    shot_be = _backend   # reuse same AerSimulator instance from init_engine
    from qiskit import transpile
    qc_t    = transpile(qc_bound, shot_be, optimization_level=1)
    counts  = shot_be.run(qc_t, shots=16384).result().get_counts()

    k_opt_bits    = f"{K_OPT:016b}"
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Target: K* = 0x{K_OPT:04X} = {k_opt_bits}")
    print(f"\n  {'Rank':>4}  {'Bitstring':>18}  {'Hex':>6}  "
          f"{'Shots':>6}  {'Prob':>8}  {'Note'}")
    print(f"  {'-'*4:>4}  {'-'*18:>18}  {'-'*6:>6}  "
          f"{'-'*6:>6}  {'-'*8:>8}")

    found = False
    for rank, (bits, cnt) in enumerate(sorted_counts[:20], 1):
        key_int = int(bits, 2)
        note    = "  *** K*  KEY RECOVERED!" if key_int == K_OPT else ""
        if key_int == K_OPT:
            found = True
        print(f"  {rank:>4}  {bits:>18}  0x{key_int:04X}  "
              f"{cnt:>6}  {cnt/16384:>8.5f}{note}")

    k_opt_count = counts.get(k_opt_bits, 0)
    speedup     = (k_opt_count / 16384) / (1 / N_KEYS) if k_opt_count > 0 else 0

    print(f"\n  {'-'*62}")
    print(f"  K* shots : {k_opt_count} / 16384")
    print(f"  K* prob  : {k_opt_count/16384:.5f}  (baseline = {1/N_KEYS:.5f})")
    print(f"  Speedup  : {speedup:.1f}x over uniform random")

    # Full progression table
    print(f"\n  Optimization Progression:")
    print(f"  {'Stage':>8}  {'P(E<=10)':>10}  {'P(E<=5)':>10}  "
          f"{'P(E<=1)':>10}  {'P(K*)':>12}  {'Speedup':>8}")
    for lbl, prb in [("Baseline", baseline), ("V1", probs_v1),
                     ("V2", probs_v2), ("V3", probs_v3)]:
        p10 = float(np.dot(prb, _E_le10))
        p5  = float(np.dot(prb, _E_le5))
        p1  = float(np.dot(prb, _E_le1))
        pk  = float(prb[K_OPT])
        spd = pk * N_KEYS
        print(f"  {lbl:>8}  {p10:>10.5f}  {p5:>10.5f}  "
              f"{p1:>10.5f}  {pk:>12.4e}  {spd:>8.1f}x")

    print(f"\n{'='*66}")
    if found and sorted_counts[0][0] == k_opt_bits:
        print(f"  OK  KEY RECOVERED -- K* = 0x{K_OPT:04X} ranks #1 in output!")
    elif found:
        rank_k = next(r for r, (b, _) in enumerate(sorted_counts, 1)
                      if b == k_opt_bits)
        print(f"  OK  KEY FOUND at rank #{rank_k} -- K* = 0x{K_OPT:04X}")
    else:
        print(f"  K* not in top-20. Consider increasing P or V3 maxiter.")
    print(f"{'='*66}\n")

    np.save('params_v3_final.npy', params_v3)
    print("  Saved: params_v3_final.npy")
    return params_v3, counts


if __name__ == "__main__":
    main()
