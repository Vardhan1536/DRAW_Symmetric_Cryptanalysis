import os, sys, math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GITHUB2_ROOT = os.path.abspath(os.path.join(HERE, '..'))
SAT_DIR = os.path.join(GITHUB2_ROOT, 'SAT_formulation')

sys.path.insert(0, HERE)
sys.path.insert(0, SAT_DIR)
sys.path.insert(0, os.path.join(HERE, 'Reversible_Circuit'))

from sat import SAESMassacciCompiler
from draw_weights import build_draw_weights
from qiskit import QuantumCircuit, QuantumRegister
from Reversible_circuit import sub_byte_paper_inplace

PT        = 0x17FD
CT        = 0xA55B
PT_BITS   = [(PT >> (15-i)) & 1 for i in range(16)]
CT_BITS   = [(CT >> (15-i)) & 1 for i in range(16)]
RCON1_VAL = 0x80
RCON2_VAL = 0x30
N_QUBITS  = 16

APPROACHES = [
    ('W=16',       16.0,  None, 69,  24,  6),
    ('W=16,14',    14.0,  None, 24,  41,  3),
    ('W=16,14,12', 12.0,  None,  4, 100,  1),
    ('Std Grover', None,  None,  1, 201,  0),  
]

c = SAESMassacciCompiler()
c.add_plaintext_ciphertext_pair(PT, CT)
weights = build_draw_weights(c)

from Oracle_A import build_oracle_A, build_diffusion, run_oracle_A_iterations
from Oracle_B import build_oracle_B, run_oracle_B_iterations


print("Building diffusion circuit...")
diff_gates, diff_depth, diff_ops = build_diffusion()
print(f"  Diffusion: {diff_gates} gates, depth {diff_depth}")
print(f"  Breakdown: {diff_ops}")
print()

init_gates = N_QUBITS   
init_depth = 1

results = []

for (name, thresh_A, thresh_B, M1, m1, m2) in APPROACHES:
    print(f"\n{'='*60}")
    print(f"  Building: {name}  (M1={M1}, m1={m1}, m2={m2})")
    print(f"{'='*60}")

    if thresh_A is None:
        # Standard Grover
        all_clauses = c.clauses
        print(f"  Standard: full oracle ({len(all_clauses)} clauses)")
        _, peak, total_g, ops, depth = build_oracle_A(all_clauses, "FullOracle")
        print(f"  Full Oracle: {total_g} gates, depth {depth}, peak {peak} qubits")

        # Phase 1 
        ph1_iter_gates = total_g + diff_gates
        ph1_iter_depth = depth + diff_depth
        ph1_total_gates = init_gates + m1 * ph1_iter_gates
        ph1_total_depth = init_depth + m1 * ph1_iter_depth
        ph2_total_gates = 0
        ph2_total_depth = 0
        total_gates = ph1_total_gates
        total_depth = ph1_total_depth

        results.append({
            'name': name, 'M1': M1, 'm1': m1, 'm2': m2,
            'oracle_A_gates': total_g, 'oracle_A_depth': depth,
            'oracle_B_gates': 0,       'oracle_B_depth': 0,
            'ph1_iter_gates': ph1_iter_gates, 'ph1_iter_depth': ph1_iter_depth,
            'ph2_iter_gates': 0,               'ph2_iter_depth': 0,
            'ph1_total_gates': ph1_total_gates, 'ph1_total_depth': ph1_total_depth,
            'ph2_total_gates': ph2_total_gates, 'ph2_total_depth': ph2_total_depth,
            'total_gates': total_gates,         'total_depth': total_depth,
        })

    else:
        # Two-phase
        cA_list = [cl for cid, cl in enumerate(c.clauses) if weights[cid] >= thresh_A]
        cB_list = [cl for cid, cl in enumerate(c.clauses) if weights[cid] < thresh_A]

        print(f"  Oracle A: {len(cA_list)} clauses (w>={thresh_A})")
        _, peakA, gA, opsA, dA = build_oracle_A(cA_list, f"OracleA_{name}")
        print(f"    Gates={gA}, Depth={dA}, Peak={peakA} qubits")

        _, peakB, gB, opsB, dB = build_oracle_B(cB_list, f"OracleB_{name}")
        print(f"    Gates={gB}, Depth={dB}, Peak={peakB} qubits")

        # Phase 1 iteration: Oracle_A + Diffusion_Std
        ph1_iter_gates = gA + diff_gates
        ph1_iter_depth = dA + diff_depth
        ph1_total_gates = init_gates + m1 * ph1_iter_gates
        ph1_total_depth = init_depth + m1 * ph1_iter_depth

        # Phase 2 iteration: Oracle_B + D_QAA
        dqaa_gates = 2 * gA + diff_gates
        dqaa_depth = 2 * dA + diff_depth
        ph2_iter_gates = gB + dqaa_gates
        ph2_iter_depth = dB + dqaa_depth
        ph2_total_gates = m2 * ph2_iter_gates
        ph2_total_depth = m2 * ph2_iter_depth

        total_gates = ph1_total_gates + ph2_total_gates
        total_depth = ph1_total_depth + ph2_total_depth

        results.append({
            'name': name, 'M1': M1, 'm1': m1, 'm2': m2,
            'oracle_A_gates': gA, 'oracle_A_depth': dA,
            'oracle_B_gates': gB, 'oracle_B_depth': dB,
            'ph1_iter_gates': ph1_iter_gates, 'ph1_iter_depth': ph1_iter_depth,
            'ph2_iter_gates': ph2_iter_gates, 'ph2_iter_depth': ph2_iter_depth,
            'ph1_total_gates': ph1_total_gates, 'ph1_total_depth': ph1_total_depth,
            'ph2_total_gates': ph2_total_gates, 'ph2_total_depth': ph2_total_depth,
            'total_gates': total_gates,         'total_depth': total_depth,
        })

print(f"\n\n{'='*100}")
print("  GATE COUNT AND CIRCUIT DEPTH SUMMARY")
print(f"{'='*100}")
print(f"\n  Diffusion (D_std, 16 qubits): {diff_gates} gates, depth {diff_depth}")
print(f"  D_QAA cost per Phase 2 iter = Oracle_A + Oracle_B + 2*Oracle_A + D_std")
print()

print(f"  {'Approach':<14} {'M1':>4} {'m1':>4} {'m2':>3}  "
      f"{'OracleA_G':>10} {'OracleA_D':>10}  "
      f"{'OracleB_G':>10} {'OracleB_D':>10}")
print(f"  {'-'*88}")
for r in results:
    print(f"  {r['name']:<14} {r['M1']:>4} {r['m1']:>4} {r['m2']:>3}  "
          f"{r['oracle_A_gates']:>10,} {r['oracle_A_depth']:>10}  "
          f"{r['oracle_B_gates']:>10,} {r['oracle_B_depth']:>10}")

print()
print(f"  {'Approach':<14} {'Ph1 Iter G':>12} {'Ph1 Iter D':>11}  "
      f"{'Ph2 Iter G':>12} {'Ph2 Iter D':>11}")
print(f"  {'-'*72}")
for r in results:
    print(f"  {r['name']:<14} {r['ph1_iter_gates']:>12,} {r['ph1_iter_depth']:>11}  "
          f"{r['ph2_iter_gates']:>12,} {r['ph2_iter_depth']:>11}")

print()
print(f"  {'Approach':<14} {'Ph1 Total G':>13} {'Ph1 Total D':>12}  "
      f"{'Ph2 Total G':>13} {'Ph2 Total D':>12}  "
      f"{'TOTAL GATES':>13} {'TOTAL DEPTH':>12}")
print(f"  {'-'*100}")
for r in results:
    print(f"  {r['name']:<14} {r['ph1_total_gates']:>13,} {r['ph1_total_depth']:>12,}  "
          f"{r['ph2_total_gates']:>13,} {r['ph2_total_depth']:>12,}  "
          f"{r['total_gates']:>13,} {r['total_depth']:>12,}")
