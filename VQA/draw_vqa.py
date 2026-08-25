import numpy as np
import sys
import os
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
import cma

from landscape_geometry_experiments import build_landscape, exp1_bit_influence, exp2_pairwise_synergy
from saes import encrypt
import argparse

# Parse arguments for standalone execution
parser = argparse.ArgumentParser(description="CWMC-QOC VQA")
parser.add_argument("--pt", type=lambda x: int(x, 0), default=0xE445, help="Plaintext (hex or int)")
parser.add_argument("--key", type=lambda x: int(x, 0), default=0x3AB6, help="Target optimal key (hex or int)")
args, _ = parser.parse_known_args()

# ─── 1. Energy Landscape ──────────────────────────────────────────────────────
K_OPT   = args.key
N       = 16
N_KEYS  = 1 << N

ct_val = encrypt(args.pt, args.key)
E_array, W_raw = build_landscape(args.pt, ct_val)
W = W_raw / np.max(W_raw)    # normalized mixer weights, shape (N,)

# ─── 2. Empirical CMA-ES & CWMC Data ──────────────────────────────────────────
I_raw = exp1_bit_influence(E_array)
I_norm = I_raw / np.max(I_raw)   # normalized influence spectrum, shape (N,)

S_matrix = exp2_pairwise_synergy(E_array, I_raw)
edges = []
for u in range(N):
    for v in range(u+1, N):
        edges.append((u, v, S_matrix[u, v]))
edges.sort(key=lambda x: x[2], reverse=True)
SYNERGY_RAW = edges[:10]
max_syn     = max(w for _, _, w in SYNERGY_RAW)
SYNERGY_EDGES = [(u, v, w / max_syn) for u, v, w in SYNERGY_RAW]

# ─── 3. Ansatz Circuit (diagram only — uses opaque custom Gate for H_CWMC) ────
def build_qoc_circuit(beta_vals, alpha_vals, gamma_vals, delta_vals, p=4):
    qc = QuantumCircuit(N)
    qc.h(range(N))
    qc.barrier()
    for l in range(p):
        if l % 2 == 0:
            for q in range(N):
                qc.ry(2 * beta_vals[l] * W[q], q)
        else:
            for q in range(N):
                qc.rx(2 * beta_vals[l] * W[q], q)
        qc.barrier()
        for q in range(N):
            qc.rz(2 * alpha_vals[l] * I_norm[q], q)
        qc.barrier()
        for u, v, Sij in SYNERGY_EDGES:
            qc.rzz(2 * gamma_vals[l] * Sij, u, v)
        qc.barrier()
        qc.append(Gate(name=f"exp(-iδ{l+1}H_CWMC)", num_qubits=N, params=[]),
                  range(N))
        qc.barrier()
    qc.measure_all()
    return qc

def simulate_schrodinger_evolution(beta_vals, alpha_vals, gamma_vals, delta_vals, p=4):
    sv = np.ones(N_KEYS, dtype=np.complex128) / np.sqrt(N_KEYS)
    idx_all = np.arange(N_KEYS)

    for l in range(p):
        # ── Mixer block (alternating RY / RX) ────────────────────────────────
        for q in range(N):
            theta  = beta_vals[l] * W[q]
            cos_t  = np.cos(theta)
            sin_t  = np.sin(theta)
            step   = 1 << q

            if l % 2 == 0:          # RY(2*theta)
                idx0 = idx_all[(idx_all & step) == 0]
                idx1 = idx0 | step
                sv0, sv1   = sv[idx0].copy(), sv[idx1].copy()
                sv[idx0]   = cos_t * sv0 - sin_t * sv1
                sv[idx1]   = sin_t * sv0 + cos_t * sv1
            else:                   # RX(2*theta)
                swapped = idx_all ^ step
                sv = cos_t * sv - 1j * sin_t * sv[swapped]   # new array; no aliasing

        # ── Influence RZ layer ────────────────────────────────────────────────
        for q in range(N):
            theta = alpha_vals[l] * I_norm[q]
            bit_q = (idx_all >> q) & 1
            # RZ(2θ): |0〉→exp(-iθ)|0〉, |1〉→exp(+iθ)|1〉
            signs = np.where(bit_q == 0, -1, 1)
            sv   *= np.exp(1j * theta * signs)

        # ── Synergy RZZ layer ─────────────────────────────────────────────────
        for u, v, Sij in SYNERGY_EDGES:
            theta = gamma_vals[l] * Sij
            bit_u = (idx_all >> u) & 1
            bit_v = (idx_all >> v) & 1
            # RZZ(2θ)=exp(-iθ ZZ): same bits→exp(-iθ), diff bits→exp(+iθ)
            signs = np.where(bit_u == bit_v, -1, 1)
            sv   *= np.exp(1j * theta * signs)

        # ── CWMC cost evolution ───────────────────────────────────────────────
        sv *= np.exp(-1j * delta_vals[l] * E_array)

    return sv


def evaluate_obj(x, p=4):
    beta  = x[0:p]
    alpha = x[p:2*p]
    gamma = x[2*p:3*p]
    delta = x[3*p:4*p]
    sv    = simulate_schrodinger_evolution(beta, alpha, gamma, delta, p)
    probs = np.abs(sv) ** 2
    return -probs[K_OPT], probs


# ─── 5. FWHT helper ───────────────────────────────────────────────────────────
def fwht(a):
    """In-place Fast Walsh-Hadamard Transform."""
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x, y       = a[j], a[j + h]
                a[j]       = x + y
                a[j + h]   = x - y
        h *= 2
    return a



def build_H_CWMC(E_arr):
    coeffs     = fwht(E_arr.copy()) / N_KEYS
    pauli_list = []
    for alpha_idx in range(N_KEYS):
        if abs(coeffs[alpha_idx]) > 1e-8:
            # FIX: format without [::-1]
            bstr  = format(alpha_idx, f'0{N}b')          # MSB first, length N
            p_str = ''.join('Z' if b == '1' else 'I' for b in bstr)
            pauli_list.append((p_str, coeffs[alpha_idx]))
    return SparsePauliOp.from_list(pauli_list)


# ─── 7. Build native Qiskit circuit from optimized parameters ─────────────────
def build_native_qiskit_circuit(best_x, H_CWMC, p=4):
    qc = QuantumCircuit(N)
    qc.h(range(N))
    for l in range(p):
        if l % 2 == 0:
            for q in range(N):
                qc.ry(2 * best_x[l] * W[q], q)
        else:
            for q in range(N):
                qc.rx(2 * best_x[l] * W[q], q)
        for q in range(N):
            qc.rz(2 * best_x[p + l] * I_norm[q], q)
        for u, v, Sij in SYNERGY_EDGES:
            qc.rzz(2 * best_x[2 * p + l] * Sij, u, v)
        # PauliEvolutionGate for diagonal (all-Z) Hamiltonian is EXACT
        # (all terms commute → no Trotter error regardless of step count).
        qc.append(PauliEvolutionGate(H_CWMC, time=best_x[3 * p + l]), range(N))
    qc.measure_all()
    return qc


# ─── 8. Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = 20

    # ── Step 1: circuit diagram ───────────────────────────────────────────────
    print(f"[1] Building CWMC-QOC structural diagram (p={p})...")
    dummy = [0.1] * p
    qc_diag = build_qoc_circuit(dummy, dummy, dummy, dummy, p)
    with open('qoc_vqa_circuit.txt', 'w', encoding='utf-8') as f:
        f.write(f"CWMC-QOC Structural Circuit Diagram (p={p} Layered VQA)\n")
        f.write("=" * 53 + "\n\n")
        f.write(str(qc_diag.draw(output='text', fold=200)))
        f.write("\n\nLayers per repetition:\n")
        f.write("  1. Alternating RY/RX weighted mixers  (beta)\n")
        f.write("  2. RZ influence layer                 (alpha)\n")
        f.write("  3. RZZ synergy layer                  (gamma)\n")
        f.write("  4. Diagonal phase exp(-i delta H_CWMC)(delta)\n")
    print("   -> Diagram saved to 'qoc_vqa_circuit.txt'.")

    # ── Step 2: CMA-ES optimisation ───────────────────────────────────────────
    print("\n[2] Running CMA-ES optimisation (target: P(K*) = P(0xA73B))...")
    np.random.seed(42)

    # Informed schedule: mixer fades out, cost driver ramps up
    x0 = np.concatenate([
        np.linspace(1.0, 0.1, p),  # beta  (decreasing)
        np.full(p, 0.1),           # alpha (low constant)
        np.full(p, 0.1),           # gamma (low constant)
        np.linspace(0.1, 1.0, p),  # delta (increasing)
    ])

    es = cma.CMAEvolutionStrategy(x0, 0.2, {
        'bounds':   [0, np.pi],
        'popsize':  20,
        'maxiter':  100,
        'verbose':  -9,
    })

    generation = 1
    while not es.stop():
        solutions   = es.ask()
        fitnesses   = []
        best_probs  = None
        best_f      = float('inf')

        for x in solutions:
            f_val, probs = evaluate_obj(x, p)
            fitnesses.append(f_val)
            if f_val < best_f:
                best_f, best_probs = f_val, probs

        es.tell(solutions, fitnesses)

        p_k    = best_probs[K_OPT]
        e_mean = float(np.sum(best_probs * E_array))
        p_e5   = float(np.sum(best_probs[E_array <= 5]))
        p_e10  = float(np.sum(best_probs[E_array <= 10]))

        print(f"Gen {generation:2d} | P(K*) = {p_k:.6f} | "
              f"<E> = {e_mean:7.2f} | P(E<=5) = {p_e5:.6f} | P(E<=10) = {p_e10:.6f}", flush=True)
        generation += 1

    best_x = es.result.xbest
    print("\n" + "=" * 52)
    print(" OPTIMISATION COMPLETE")
    print("=" * 52)
    print(f"beta  = {np.round(best_x[0:p],  4)}")
    print(f"alpha = {np.round(best_x[p:2*p],  4)}")
    print(f"gamma = {np.round(best_x[2*p:3*p], 4)}")
    print(f"delta = {np.round(best_x[3*p:4*p],4)}")

    # Statevector ground-truth probability
    sv_final  = simulate_schrodinger_evolution(
        best_x[0:p], best_x[p:2*p], best_x[2*p:3*p], best_x[3*p:4*p], p)
    probs_sv  = np.abs(sv_final) ** 2
    p_k_true  = probs_sv[K_OPT]
    rank_k    = int(np.sum(probs_sv > probs_sv[K_OPT]))
    print(f"\nStatevector P(K*) = {p_k_true:.8f}  |  rank of K* = {rank_k}")

    # ── Step 3: Build H_CWMC and native Qiskit circuit ────────────────────────
    print("\n[3] Building H_CWMC via FWHT (BUG-1 fixed: no bstr reversal)...")
    H_CWMC = build_H_CWMC(E_array)
    print(f"    Pauli terms in H_CWMC: {len(H_CWMC)}")

    # ── Step 4: Transpile and simulate ───────────────────────────────────────
    print("\n[4] Compiling native Qiskit circuit...")
    final_qc = build_native_qiskit_circuit(best_x, H_CWMC, p)
    print("    Circuit depth before transpile:", final_qc.depth())

    sim = AerSimulator(method='statevector')

    print("    Transpiling (optimization_level=1)...")
    transpiled_qc = transpile(final_qc, sim, optimization_level=1)
    print(f"    Gates after transpile: {transpiled_qc.size()}")

    shots = 100_000
    print(f"    Running {shots:,} shots...")
    result = sim.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    # ── FIX (BUG-2): look up binary string, not hex string ────────────────────
    target_bits = format(K_OPT, f'0{N}b')   # '1010011100111011'
    print(f"\n    Target measurement string: '{target_bits}'  (K* = 0x{K_OPT:04X})")
    print(f"    Distinct outcomes observed: {len(counts)}")

    if target_bits in counts:
        shot_count = counts[target_bits]
        p_measured = shot_count / shots
        print(f"\n  [SUCCESS] K* RECOVERED: {shot_count}/{shots} shots  "
              f"(P_meas = {p_measured:.6f},  P_sv = {p_k_true:.6f})")
    else:
        print(f"\n  [FAILED] K* not in top-{len(counts)} outcomes for {shots:,} shots.")
        print(f"    Statevector P(K*) = {p_k_true:.8f}")
        print("    (Low P_sv → increase p or re-run CMA-ES with more iterations.)")

    # Show top-10 outcomes for debugging
    print("\n  Top-10 measurement outcomes:")
    top10 = sorted(counts.items(), key=lambda x: -x[1])[:10]
    for i, (bits, cnt) in enumerate(top10):
        key_int = int(bits, 2)
        marker  = " ← K*" if key_int == K_OPT else ""
        print(f"    #{i+1:2d}  0x{key_int:04X}  ({bits})  count={cnt:6d}"
              f"  P={cnt/shots:.6f}{marker}")