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
from saes import SBOX_LUT, INV_SBOX_LUT, mult4
from Oracle_A import build_oracle_A, build_diffusion
from Oracle_B import build_oracle_B

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

PT       = 0x17FD
CT       = 0xA55B
TRUE_KEY = 0x3ABC
N        = 65536
N_QUBITS = 16

APPROACHES = [
    ('W=16',       16.0),
    ('W=16,14',    14.0),
    ('W=16,14,12', 12.0),
    ('Std Grover', None)
]

RCON = {1: 0x80, 2: 0x30}

# S-AES Classical Operations (for Truth Table Evaluation)
def inv_sub_bytes(state):
    s0 = (state >> 12) & 0xF; s1 = (state >> 8) & 0xF
    s2 = (state >> 4)  & 0xF; s3 = state & 0xF
    return (INV_SBOX_LUT[s0] << 12) | (INV_SBOX_LUT[s1] << 8) | \
           (INV_SBOX_LUT[s2] << 4)  |  INV_SBOX_LUT[s3]

def shift_rows(state):
    n0 = (state >> 12) & 0xF; n1 = (state >> 8) & 0xF
    n2 = (state >> 4)  & 0xF; n3 = state & 0xF
    return (n0 << 12) | (n3 << 8) | (n2 << 4) | n1

def gf_mult(a, b):
    p = 0
    for _ in range(4):
        if b & 1: p ^= a
        hi_bit_set = a & 8
        a <<= 1
        if hi_bit_set: a ^= 0x13
        b >>= 1
    return p & 0xF

def inv_mix_columns(state):
    n0 = (state >> 12) & 0xF; n1 = (state >> 8) & 0xF
    n2 = (state >> 4)  & 0xF; n3 = state & 0xF
    m0 = gf_mult(9, n0) ^ gf_mult(2, n1)
    m1 = gf_mult(2, n0) ^ gf_mult(9, n1)
    m2 = gf_mult(9, n2) ^ gf_mult(2, n3)
    m3 = gf_mult(2, n2) ^ gf_mult(9, n3)
    return (m0 << 12) | (m1 << 8) | (m2 << 4) | m3

def assign_var(asgn, offset, val, bits):
    for i in range(bits):
        asgn[offset + i] = (val >> ((bits-1)-i)) & 1

def generate_assignment(pt, ct, key):
    asgn = {}
    assign_var(asgn, 1, key, 16)

    W_prev2 = (key >> 8) & 0xFF
    W_prev1 = key & 0xFF

    rcon_1 = RCON[1]
    sb_lower_1 = SBOX_LUT[W_prev1 & 0xF]
    sb_upper_1 = SBOX_LUT[(W_prev1 >> 4) & 0xF]
    W_new0_1 = W_prev2 ^ rcon_1 ^ (sb_lower_1 << 4) ^ sb_upper_1
    W_new1_1 = W_new0_1 ^ W_prev1

    assign_var(asgn, 17, sb_lower_1, 4); assign_var(asgn, 21, sb_upper_1, 4)
    assign_var(asgn, 25, W_new0_1, 8);   assign_var(asgn, 33, W_new1_1, 8)

    rcon_2 = RCON[2]
    sb_lower_2 = SBOX_LUT[W_new1_1 & 0xF]
    sb_upper_2 = SBOX_LUT[(W_new1_1 >> 4) & 0xF]
    W_new0_2 = W_new0_1 ^ rcon_2 ^ (sb_lower_2 << 4) ^ sb_upper_2
    W_new1_2 = W_new0_2 ^ W_new1_1

    assign_var(asgn, 41, sb_lower_2, 4); assign_var(asgn, 45, sb_upper_2, 4)
    assign_var(asgn, 49, W_new0_2, 8);   assign_var(asgn, 57, W_new1_2, 8)

    K1 = (W_new0_1 << 8) | W_new1_1
    K2 = (W_new0_2 << 8) | W_new1_2

    assign_var(asgn, 65, pt, 16); assign_var(asgn, 81, ct, 16)

    s_shift_final = ct ^ K2
    s2s = shift_rows(s_shift_final)
    assign_var(asgn, 177, s2s, 16)

    s2 = inv_sub_bytes(s2s)
    assign_var(asgn, 161, s2, 16)

    s1m = s2 ^ K1
    assign_var(asgn, 129, s1m, 16)

    s1_shift = inv_mix_columns(s1m)
    s1s = shift_rows(s1_shift)
    assign_var(asgn, 113, s1s, 16)

    s0 = inv_sub_bytes(s1s)
    assign_var(asgn, 97, s0, 16)

    n0_0 = (s1_shift >> 12) & 0xF; n1_0 = (s1_shift >> 8) & 0xF
    assign_var(asgn, 145, mult4(n0_0), 4); assign_var(asgn, 149, mult4(n1_0), 4)

    n0_1 = (s1_shift >> 4) & 0xF; n1_1 = s1_shift & 0xF
    assign_var(asgn, 153, mult4(n0_1), 4); assign_var(asgn, 157, mult4(n1_1), 4)

    return asgn

def check_clauses(assignment, clauses):
    for clause in clauses:
        satisfied = False
        for lit in clause:
            v = abs(lit)
            if v not in assignment:
                continue
            val = assignment[v]
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                satisfied = True
                break
        if not satisfied:
            return False
    return True

def optimal_iters(N_space: int, M: int) -> int:
    """Grover / QAA optimal iteration count: floor((pi/4) * sqrt(N/M))."""
    if M == 0: return 0
    return math.floor((math.pi / 4) * math.sqrt(N_space / M))

def run_aer_simulator(qc_oracle, iterations, N_QUBITS=16):
    """
    Constructs the physical QuantumCircuit for multiple Grover iterations and 
    runs it on the AerSimulator. 
    """
    print(f"\n[AER] Building physical Grover Iteration circuit for {iterations} iterations...")
    
    pool_size = qc_oracle.num_qubits
    full_qc = QuantumCircuit(pool_size, N_QUBITS)
    
    full_qc.h(range(N_QUBITS))
    
    diff_qc_small = QuantumCircuit(N_QUBITS)
    diff_qc_small.h(range(N_QUBITS))
    diff_qc_small.x(range(N_QUBITS))
    diff_qc_small.h(N_QUBITS-1)
    diff_qc_small.mcx(list(range(N_QUBITS-1)), N_QUBITS-1)
    diff_qc_small.h(N_QUBITS-1)
    diff_qc_small.x(range(N_QUBITS))
    diff_qc_small.h(range(N_QUBITS))
    
    diff_qc = QuantumCircuit(pool_size)
    diff_qc.compose(diff_qc_small, qubits=list(range(N_QUBITS)), inplace=True)
    
    for _ in range(iterations):
        full_qc.compose(qc_oracle, inplace=True)
        full_qc.compose(diff_qc, inplace=True)
        
    full_qc.measure(range(N_QUBITS), range(N_QUBITS))
    
    print(f"[AER] Circuit built. Commencing simulation on AerSimulator...")
    sim = AerSimulator(method='matrix_product_state') 
    try:
        t_qc = transpile(full_qc, sim)
        job = sim.run(t_qc, shots=100)
        result = job.result()
        counts = result.get_counts()
        print(f"[AER] Simulation Success! Counts: {counts}")
        return counts
    except Exception as e:
        print(f"[AER] Simulation Failed (likely memory limit exceeded): {e}")
        return None

if __name__ == "__main__":
    print("=" * 70)
    print("  UNIFIED TWO-PHASE GROVER EVALUATOR")
    print("  1. Classical Exact-SAT Truth Table Verification")
    print("  2. Optimal Dynamic Iteration Calculation")
    print("  3. Physical Hardware Gate Depth Tracking")
    print("=" * 70)
    
    c = SAESMassacciCompiler()
    c.add_plaintext_ciphertext_pair(PT, CT)
    weights = build_draw_weights(c)
    
    results = []
    
    for name, thresh_A in APPROACHES:
        print(f"\n{'-'*70}")
        print(f"  Configuration: {name}")
        print(f"{'-'*70}")
        
        if thresh_A is None:
            all_clauses = c.clauses
            print(f"  [Classical] Evaluating Standard Oracle ({len(all_clauses)} clauses) over 65536 keys...")
            
            M1 = 0
            true_key_found = False
            for key in range(65536):
                asgn = generate_assignment(PT, CT, key)
                if check_clauses(asgn, all_clauses):
                    M1 += 1
                    if key == TRUE_KEY:
                        true_key_found = True
                        
            print(f"    -> Keys Surviving: {M1}")
            print(f"    -> True Key Survives: {true_key_found}")
            
            m1 = optimal_iters(65536, 1) 
            print(f"  [Theory] Optimal Iterations (m1) = {m1}")
            
            # Fetch Qiskit Depth
            print(f"  [Qiskit] Building and transpiling rigorous physical circuit...")
            qc_A, peak, gates_A, _, depth_A = build_oracle_A(all_clauses, "FullOracle")
            
            diff_g, diff_d, _ = build_diffusion()
            
            results.append({
                'name': name, 'M1': M1, 'm1': m1, 'm2': 0,
                'gate_cost': 16 + m1 * (gates_A + diff_g),
                'depth_cost': 1 + m1 * (depth_A + diff_d)
            })
            
            
        else:
            cA_list = [cl for cid, cl in enumerate(c.clauses) if weights[cid] >= thresh_A]
            cB_list = [cl for cid, cl in enumerate(c.clauses) if weights[cid] < thresh_A]
            
            print(f"  [Classical] Evaluating Oracle A ({len(cA_list)} clauses) over 65536 keys...")
            M1 = 0
            surviving_keys = []
            for key in range(65536):
                asgn = generate_assignment(PT, CT, key)
                if check_clauses(asgn, cA_list):
                    M1 += 1
                    surviving_keys.append(key)
            
            print(f"    -> Keys Surviving Oracle A (M1): {M1}")
            print(f"  [Classical] Evaluating Oracle B ({len(cB_list)} clauses) over {M1} keys...")
            
            M2 = 0
            true_key_found = False
            for key in surviving_keys:
                asgn = generate_assignment(PT, CT, key)
                if check_clauses(asgn, cB_list):
                    M2 += 1
                    if key == TRUE_KEY:
                        true_key_found = True
            
            print(f"    -> Keys Surviving Oracle B (M2): {M2}")
            print(f"    -> True Key Survives: {true_key_found}")
            
            # Dynamically compute m1 and m2
            m1 = optimal_iters(65536, M1)
            m2 = optimal_iters(M1, 1)
            print(f"  [Theory] Optimal Iterations: Phase 1 (m1) = {m1}, Phase 2 (m2) = {m2}")
            
            # Fetch Qiskit Depth
            print(f"  [Qiskit] Building and transpiling rigorous physical circuits...")
            qc_A, peak_A, gates_A, _, depth_A = build_oracle_A(cA_list, f"OracleA_{name}")
            qc_B, peak_B, gates_B, _, depth_B = build_oracle_B(cB_list, f"OracleB_{name}")
            
            diff_g, diff_d, _ = build_diffusion()
            
            cost_ph1 = m1 * (gates_A + diff_g)
            cost_ph2 = m2 * (gates_B + 2*gates_A + diff_g)
            
            depth_ph1 = m1 * (depth_A + diff_d)
            depth_ph2 = m2 * (depth_B + 2*depth_A + diff_d)
            
            results.append({
                'name': name, 'M1': M1, 'm1': m1, 'm2': m2,
                'gate_cost': 16 + cost_ph1 + cost_ph2,
                'depth_cost': 1 + depth_ph1 + depth_ph2
            })
            
    print("\n" + "="*80)
    print("  FINAL UNIFIED RESULTS (Classical Evaluation + Qiskit Rigorous Depth)")
    print("="*80)
    print(f"{'Approach':<15} {'M1':<6} {'m1':<6} {'m2':<6} {'Total Gates':<15} {'Total Depth':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<15} {r['M1']:<6} {r['m1']:<6} {r['m2']:<6} {r['gate_cost']:<15,d} {r['depth_cost']:<15,d}")
