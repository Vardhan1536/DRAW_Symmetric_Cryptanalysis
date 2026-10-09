"""
two_phase_grover_simulation.py  -  Full Qiskit Iteration Simulation

"""

import os, sys, math, argparse
from collections import defaultdict

HERE         = os.path.dirname(os.path.abspath(__file__))
GITHUB2_ROOT = os.path.abspath(os.path.join(HERE, '..'))
SAT_DIR      = os.path.join(GITHUB2_ROOT, 'SAT_formulation')

sys.path.insert(0, HERE)
sys.path.insert(0, SAT_DIR)
sys.path.insert(0, os.path.join(HERE, 'Reversible_Circuit'))

from sat          import SAESMassacciCompiler
from draw_weights import build_draw_weights
from utils        import generate_assignment, check_clauses
from Oracle_A     import build_oracle_A
from Oracle_B     import build_oracle_B

from qiskit     import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Qubit
from qiskit_aer import AerSimulator

PT       = 0x17FD
CT       = 0xA55B
TRUE_KEY = 0x3ABC
N_QUBITS = 16      
N_SPACE  = 65536

CONFIGS = [
    ('W=16',        16.0),
    ('W=16,14',     14.0),
    ('W=16,14,12',  12.0),
]

def optimal_iters(search_space: int, n_solutions: int) -> int:
    """floor(pi/4 x sqrt(N / M))."""
    if n_solutions <= 0:
        return 0
    return max(1, math.floor((math.pi / 4) * math.sqrt(search_space / n_solutions)))


# =============================================================================
# PRUNE: Compact a pooled qubit oracle down to only the qubits actually used
# =============================================================================
def prune_unallocated_ancillae(qc_full: QuantumCircuit, n_key_qubits: int = N_QUBITS):
    """
    Remove idle ancillae that have no quantum operations, compacting the register width.
    Key qubits (0..n_key_qubits-1) are preserved in order.
    """
    all_qubits = qc_full.qubits   

    active_set: set = set()
    for instr in qc_full.data:
        if instr.operation.name != 'barrier':
            for q in instr.qubits:
                active_set.add(q)

    key_qubits = set(all_qubits[:n_key_qubits])
    active_set |= key_qubits

    active_list = sorted(active_set, key=lambda q: all_qubits.index(q))
    n_active    = len(active_list)

    old_to_new: dict = {q: i for i, q in enumerate(active_list)}

    qc_pruned = QuantumCircuit(n_active, name=qc_full.name + '_pruned')
    for instr in qc_full.data:
        if instr.operation.name == 'barrier':
            new_qargs = [old_to_new[q] for q in instr.qubits if q in old_to_new]
            if new_qargs:
                qc_pruned.barrier([qc_pruned.qubits[i] for i in new_qargs])
            continue
        new_qargs = [old_to_new[q] for q in instr.qubits]
        new_cargs = list(instr.clbits)   
        qc_pruned.append(instr.operation, new_qargs, new_cargs)

    return qc_pruned, old_to_new


# =============================================================================
# STEP 1: Classical truth-table sweep to compute exact M1 and M2
# =============================================================================
def classical_verification(c, weights, thresh_A):
    cA_list = [cl for cid, cl in enumerate(c.clauses) if weights[cid] >= thresh_A]
    cB_list = [cl for cid, cl in enumerate(c.clauses) if weights[cid] <  thresh_A]

    print(f"\n  [Classical] Oracle A: {len(cA_list)} heavy clauses (weight >= {thresh_A})")
    print(f"  [Classical] Oracle B: {len(cB_list)} light clauses  (weight <  {thresh_A})")
    print(f"  [Classical] Sweeping all {N_SPACE:,} keys ...")

    surviving_A  = []
    surviving_AB = []
    true_key_in_A  = False
    true_key_in_AB = False

    for key in range(N_SPACE):
        asgn = generate_assignment(PT, CT, key)
        in_A = check_clauses(asgn, cA_list)
        if in_A:
            surviving_A.append(key)
            if key == TRUE_KEY:
                true_key_in_A = True
            in_B = check_clauses(asgn, cB_list)
            if in_B:
                surviving_AB.append(key)
                if key == TRUE_KEY:
                    true_key_in_AB = True

    M1 = len(surviving_A)
    M2 = len(surviving_AB)

    print(f"\n  [Classical] Keys surviving Oracle A (M1): {M1}")
    print(f"  [Classical] True key 0x{TRUE_KEY:04X} in M1: {true_key_in_A}")
    print(f"  [Classical] Keys surviving A + B    (M2): {M2}")
    print(f"  [Classical] True key 0x{TRUE_KEY:04X} in M2: {true_key_in_AB}")

    if not true_key_in_A:
        print(f"\n  [WARNING] True key NOT in Phase-1 survivors! "
              f"Oracle A clauses are too restrictive for threshold={thresh_A}.")
    if not true_key_in_AB:
        print(f"\n  [WARNING] True key NOT in Phase-2 survivors! "
              f"Oracle B further eliminates the true key.")

    return cA_list, cB_list, M1, M2, surviving_A, surviving_AB


# =============================================================================
# STEP 2: Build the 16-qubit diffuser embedded in a pool-size circuit
# =============================================================================
def build_diffuser_circuit(pool_size: int) -> QuantumCircuit:
    diff_small = QuantumCircuit(N_QUBITS)
    diff_small.h(range(N_QUBITS))
    diff_small.x(range(N_QUBITS))
    diff_small.h(N_QUBITS - 1)
    diff_small.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    diff_small.h(N_QUBITS - 1)
    diff_small.x(range(N_QUBITS))
    diff_small.h(range(N_QUBITS))

    diff_full = QuantumCircuit(pool_size)
    diff_full.compose(diff_small, qubits=list(range(N_QUBITS)), inplace=True)
    return diff_full


# =============================================================================
# STEP 3: Assemble the full two-phase Grover QuantumCircuit
# =============================================================================
def build_two_phase_circuit(qc_A, qc_B, m1: int, m2: int) -> QuantumCircuit:
    assert qc_A.num_qubits == qc_B.num_qubits, (
        f"Pruned Oracle A ({qc_A.num_qubits}q) and B ({qc_B.num_qubits}q) must match."
    )
    n_total = qc_A.num_qubits
    full_qc = QuantumCircuit(n_total, N_QUBITS)

    full_qc.h(range(N_QUBITS))
    full_qc.barrier()

    diff_qc  = build_diffuser_circuit(n_total)   
    qc_A_inv = qc_A.inverse()

    print(f"\n  [Circuit] Composing Phase 1:")
    for _ in range(m1):
        full_qc.compose(qc_A, inplace=True)
        full_qc.compose(diff_qc, inplace=True)
    full_qc.barrier()

    print(f"  [Circuit] Composing Phase 2: ")
    for _ in range(m2):
        full_qc.compose(qc_A_inv, inplace=True)
        full_qc.compose(qc_B,    inplace=True)
        full_qc.compose(qc_A,    inplace=True)
        full_qc.compose(diff_qc, inplace=True)
    full_qc.barrier()

    full_qc.measure(range(N_QUBITS), range(N_QUBITS))

    return full_qc


# =============================================================================
# STEP 4: Run on AerSimulator (statevector for memory efficiency)
# =============================================================================
def run_simulation(full_qc: QuantumCircuit, shots: int = 100):
    print(f"\n  [AER] Transpiling {full_qc.num_qubits}-qubit circuit "
          f"(decomposing mcx -> ccx, no coupling_map) ...")

    BASIS = ['u', 'cx', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg',
             'rx', 'ry', 'rz', 'ccx', 'swap', 'measure', 'barrier']
    try:
        t_qc = transpile(full_qc, basis_gates=BASIS, optimization_level=0)
        print(f"  [AER] Transpilation done ({t_qc.num_qubits}q). "
              f"Gate count after decomposition: {t_qc.size():,}")
    except Exception as exc:
        print(f"  [AER] Transpilation warning (running without decomposition): {exc}")
        t_qc = full_qc

    print(f"  [AER] Launching {shots} shots ...")

    sim = AerSimulator(
        method='statevector',
        max_parallel_threads=1,   
    )
    try:
        job    = sim.run(t_qc, shots=shots)
        result = job.result()
        if result.status != 'COMPLETED':
            raise RuntimeError(result.status)
        counts = result.get_counts()
        print(f"  [AER] Simulation complete. Unique bitstrings: {len(counts)}")
        return counts
    except Exception as exc:
        print(f"\n  [AER] Simulation FAILED: {exc}")
        print(f"  If it keeps failing, the circuit depth ({full_qc.depth():,} layers)")
        print(f"  may be creating high entanglement that overwhelms statevector.")
        return None


# =============================================================================
# STEP 5: Decode bitstring counts into ranked key candidates
# =============================================================================
def decode_counts(counts: dict, surviving_A: list, surviving_AB: list, shots: int) -> list:
    set_A  = set(surviving_A)
    set_AB = set(surviving_AB)

    decoded = []
    for bitstr, cnt in counts.items():
        clean = bitstr.replace(' ', '')
        key_int = int(clean[::-1], 2)
        decoded.append({
            'key':         key_int,
            'count':       cnt,
            'probability': cnt / shots,
            'is_true_key': (key_int == TRUE_KEY),
            'in_M1':       (key_int in set_A),
            'in_M2':       (key_int in set_AB),
        })

    decoded.sort(key=lambda x: -x['count'])
    return decoded


def print_results(decoded: list, m1: int, m2: int, M1: int, M2: int,
                  shots: int, config_name: str):
    print(f"\n{'='*65}")
    print(f"  RESULTS  {config_name}  |  m1={m1}, m2={m2}  |  {shots} shots")
    print(f"{'='*65}")
    print(f"  {'Rank':<5}  {'Key':>8}  {'Shots':>6}  {'Prob':>7}  Tags")
    print(f"  {'-'*60}")
    for rank, e in enumerate(decoded[:15], 1):
        tags = []
        if e['is_true_key']: tags.append('<- TRUE KEY (correct!)')
        elif e['in_M2']:     tags.append('[M1+B survivor]')
        elif e['in_M1']:     tags.append('[M1 survivor]')
        print(f"  {rank:<5}  0x{e['key']:04X}  {e['count']:>6}  "
              f"{e['probability']:>6.1%}  {'  '.join(tags)}")

    true_shots = next((e['count'] for e in decoded if e['is_true_key']), 0)
    top1_ok    = decoded[0]['is_true_key'] if decoded else False
    print(f"\n  True key 0x{TRUE_KEY:04X}: {true_shots}/{shots} shots "
          f"({100*true_shots/shots:.1f}%)   Top-1 correct: {top1_ok}")
    print(f"{'='*65}")


# =============================================================================
# MAIN CONFIG RUNNER
# =============================================================================
def run_config(config_name: str, thresh_A: float,
               c, weights, shots: int) -> dict:
    bar = '-' * 65
    print(f"\n{bar}")
    print(f"  CONFIG: {config_name}   (Oracle A threshold = {thresh_A})")
    print(f"{bar}")

    cA_list, cB_list, M1, M2, surviving_A, surviving_AB = classical_verification(
        c, weights, thresh_A
    )

    if M1 == 0:
        print(f"\n  [SKIP] M1=0 — Oracle A eliminates all keys. "
              f"No Phase-1 survivors to amplify.")
        return {'name': config_name, 'M1': M1, 'M2': M2, 'success': False}

    m1          = optimal_iters(N_SPACE, M1)
    m2          = optimal_iters(M1, max(1, M2))
    std_grover_k = optimal_iters(N_SPACE, 1)
    total_q     = m1 + m2

    print(f"\n  [Theory] Standard Grover (N={N_SPACE}, M=1): k = {std_grover_k}")
    print(f"  [Theory] Phase 1 (Oracle A, M1={M1}):  m1 = floor(pi/4 x sqrt({N_SPACE}/{M1})) = {m1}")
    print(f"  [Theory] Phase 2 (Oracle B, M2={M2}):  m2 = floor(pi/4 x sqrt({M1}/{max(1,M2)})) = {m2}")
    print(f"  [Theory] Total two-phase queries: {total_q}  "
          f"| Speedup: {std_grover_k}/{total_q} = {std_grover_k/max(1,total_q):.1f}x")

    print(f"\n  [Qiskit] Building Oracle A ({len(cA_list)} heavy clauses) ...")
    res_A  = build_oracle_A(cA_list, f"OracleA_{config_name}")
    qc_A_full   = res_A[0]
    peak_A      = res_A[1]
    alloc_A     = res_A[5]   
    print(f"           Done. Raw pool: {qc_A_full.num_qubits}q, peak used: {peak_A}q")

    print(f"\n  [Qiskit] Building Oracle B ({len(cB_list)} light clauses) ...")
    res_B  = build_oracle_B(cB_list, f"OracleB_{config_name}")
    qc_B_full   = res_B[0]
    peak_B      = res_B[1]
    alloc_B     = res_B[5]
    print(f"           Done. Raw pool: {qc_B_full.num_qubits}q, peak used: {peak_B}q")

    print(f"\n  [Prune] Compacting Oracle A from {qc_A_full.num_qubits}q -> active only ...")
    qc_A, map_A = prune_unallocated_ancillae(qc_A_full, n_key_qubits=N_QUBITS)
    print(f"          Oracle A pruned: {qc_A_full.num_qubits}q -> {qc_A.num_qubits}q")

    print(f"  [Prune] Compacting Oracle B from {qc_B_full.num_qubits}q -> active only ...")
    qc_B, map_B = prune_unallocated_ancillae(qc_B_full, n_key_qubits=N_QUBITS)
    print(f"          Oracle B pruned: {qc_B_full.num_qubits}q -> {qc_B.num_qubits}q")

    if qc_A.num_qubits != qc_B.num_qubits:
        n_max = max(qc_A.num_qubits, qc_B.num_qubits)
        if qc_A.num_qubits < n_max:
            padded = QuantumCircuit(n_max, name=qc_A.name)
            padded.compose(qc_A, qubits=range(qc_A.num_qubits), inplace=True)
            qc_A = padded
        if qc_B.num_qubits < n_max:
            padded = QuantumCircuit(n_max, name=qc_B.name)
            padded.compose(qc_B, qubits=range(qc_B.num_qubits), inplace=True)
            qc_B = padded
        print(f"  [Prune] Padded to common width: {n_max}q")

    n_pruned = qc_A.num_qubits
    print(f"\n  [Qiskit] Assembling two-phase circuit ({m1}+{m2} iters, {n_pruned}q pruned) ...")
    full_qc = build_two_phase_circuit(qc_A, qc_B, m1, m2)
    print(f"  [Qiskit] Gate count: {full_qc.size():,}   Depth: {full_qc.depth():,}")

    counts = run_simulation(full_qc, shots=shots)
    if counts is None:
        return {'name': config_name, 'M1': M1, 'M2': M2,
                'm1': m1, 'm2': m2, 'success': False, 'counts': None}

    decoded = decode_counts(counts, surviving_A, surviving_AB, shots)
    print_results(decoded, m1, m2, M1, M2, shots, config_name)

    true_shots = next((e['count'] for e in decoded if e['is_true_key']), 0)
    return {
        'name':           config_name,
        'thresh_A':       thresh_A,
        'M1':             M1,
        'M2':             M2,
        'm1':             m1,
        'm2':             m2,
        'total_queries':  total_q,
        'std_grover_k':   std_grover_k,
        'speedup':        std_grover_k / max(1, total_q),
        'true_key_shots': true_shots,
        'true_key_prob':  true_shots / shots,
        'top1_correct':   decoded[0]['is_true_key'] if decoded else False,
        'success':        True,
        'counts':         counts,
        'decoded':        decoded,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Full Qiskit Two-Phase Grover Simulation for S-AES'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--thresh', type=float, default=16.0,
        help='Oracle A weight threshold (default 16.0 -> W=16 config)'
    )
    group.add_argument(
        '--all', action='store_true',
        help='Run all three threshold configurations'
    )
    parser.add_argument(
        '--shots', type=int, default=100,
        help='Measurement shots per simulation (default 100)'
    )
    args = parser.parse_args()

    print('=' * 65)
    print('  TWO-PHASE GROVER SIMULATION  -  S-AES Key Recovery')
    print('  Physical Qiskit oracle circuits (Oracle_A + Oracle_B)')
    print('=' * 65)
    print(f'\n  Plaintext  : 0x{PT:04X}')
    print(f'  Ciphertext : 0x{CT:04X}')
    print(f'  True key   : 0x{TRUE_KEY:04X}')
    print(f'  Key space  : {N_SPACE:,} (16-bit S-AES)')
    print(f'  Shots      : {args.shots}')

    print(f'\n  [SAT] Compiling S-AES SAT clauses ...')
    c = SAESMassacciCompiler()
    c.add_plaintext_ciphertext_pair(PT, CT)
    weights = build_draw_weights(c)
    print(f'  [SAT] {len(c.clauses)} clauses. '
          f'Weight range: [{min(weights):.1f}, {max(weights):.1f}]')

    if args.all:
        configs_to_run = CONFIGS
    else:
        match = next((cfg for cfg in CONFIGS if cfg[1] == args.thresh), None)
        configs_to_run = [match] if match else [(f'W>={args.thresh}', args.thresh)]

    # Run
    all_results = []
    for cfg_name, thresh_A in configs_to_run:
        res = run_config(cfg_name, thresh_A, c, weights, args.shots)
        all_results.append(res)

    if args.all and len(all_results) > 1:
        std_k = optimal_iters(N_SPACE, 1)
        print(f'\n{"="*80}')
        print(f'  FINAL COMPARISON')
        print(f'{"="*80}')
        print(f'  {"Config":<16} {"M1":>6} {"M2":>4} {"m1":>5} {"m2":>5} '
              f'{"Queries":>8} {"Speedup":>8} {"P(key)":>8} {"Top-1":>6}')
        print(f'  {"-"*78}')
        print(f'  {"Std Grover":<16} {N_SPACE:>6} {"1":>4} {"--":>5} {"--":>5} '
              f'{std_k:>8} {"1.0x":>8} {"--":>8} {"--":>6}')
        for r in all_results:
            if not r.get('success'):
                print(f'  {r["name"]:<16}  FAILED (M1={r["M1"]})')
                continue
            print(f'  {r["name"]:<16} {r["M1"]:>6} {r["M2"]:>4} '
                  f'{r["m1"]:>5} {r["m2"]:>5} {r["total_queries"]:>8} '
                  f'{r["speedup"]:>7.1f}x {r["true_key_prob"]:>7.1%} '
                  f'{"YES" if r["top1_correct"] else "NO":>6}')
        print(f'{"="*80}')


if __name__ == '__main__':
    main()
