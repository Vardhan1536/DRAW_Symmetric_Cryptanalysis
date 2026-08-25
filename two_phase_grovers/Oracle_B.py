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
from qiskit import QuantumCircuit, QuantumRegister, transpile
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

V_KEY  = list(range(1,   17))
V_SB1  = list(range(17,  25))
V_W2   = list(range(25,  33))
V_W3   = list(range(33,  41))
V_SB2  = list(range(41,  49))
V_W4   = list(range(49,  57))
V_W5   = list(range(57,  65))
V_PT   = set(range(65,   81))
V_CT   = set(range(81,   97))
V_ARK0 = list(range(97,  113))
V_SR   = list(range(113, 129))
V_STX  = list(range(129, 145))
V_STZ1 = list(range(145, 161))
V_STZ  = list(range(161, 177))
V_DEEP = list(range(177, 193))
V_ARK0_set = set(V_ARK0)

STAGE_ORDER = ['KEY','SB1','W2','W3','SB2','W4','W5',
               'PT','CT','ARK0','SR','STX','STZ1','STZ','DEEP']
STAGE_MAP = {}
for v in V_KEY:  STAGE_MAP[v] = 'KEY'
for v in V_SB1:  STAGE_MAP[v] = 'SB1'
for v in V_W2:   STAGE_MAP[v] = 'W2'
for v in V_W3:   STAGE_MAP[v] = 'W3'
for v in V_SB2:  STAGE_MAP[v] = 'SB2'
for v in V_W4:   STAGE_MAP[v] = 'W4'
for v in V_W5:   STAGE_MAP[v] = 'W5'
for v in V_PT:   STAGE_MAP[v] = 'PT'
for v in V_CT:   STAGE_MAP[v] = 'CT'
for v in V_ARK0: STAGE_MAP[v] = 'ARK0'
for v in V_SR:   STAGE_MAP[v] = 'SR'
for v in V_STX:  STAGE_MAP[v] = 'STX'
for v in V_STZ1: STAGE_MAP[v] = 'STZ1'
for v in V_STZ:  STAGE_MAP[v] = 'STZ'
for v in V_DEEP: STAGE_MAP[v] = 'DEEP'

def latest_stage(clause):
    stages = set()
    for l in clause:
        s = STAGE_MAP.get(abs(l), 'UNKNOWN')
        if s not in ('PT','CT'): stages.add(s)
    for s in reversed(STAGE_ORDER):
        if s in stages: return s
    return 'KEY'

# ══════════════════════════════════════════════════════════════════════════════
# ORACLE BUILDER 
# ══════════════════════════════════════════════════════════════════════════════
def build_oracle_B(clause_list, label="OracleB"):
    """
    Builds the full Qiskit circuit for an oracle over the given clauses.
    Uses the pebbling schedule:
      KEY schedule (forward) + ARK0 + SR + STZ1 + STZ + DEEP (backward)
    Returns (QuantumCircuit, peak_qubits, gate_counts_dict, depth)
    """
    stage_clauses = defaultdict(list)
    for cl in clause_list:
        stage_clauses[latest_stage(cl)].append((cl, 'cnf', None))

    POOL_SIZE = 250   
    pool = QuantumRegister(POOL_SIZE, 'q')
    qc   = QuantumCircuit(pool, name=label)

    allocations = {}
    free_q      = list(range(POOL_SIZE))
    peak        = [0]

    def alloc(var):
        if var in allocations: return allocations[var]
        q = free_q.pop(0)
        allocations[var] = q
        used = POOL_SIZE - len(free_q)
        if used > peak[0]: peak[0] = used
        return q

    def free(var):
        if var in allocations:
            q = allocations.pop(var)
            free_q.insert(0, q)
            free_q.sort()

    for v in V_KEY: alloc(v)
    clause_a  = alloc('ca')
    viol_flag = alloc('vf')
    target    = alloc('tgt')

    def compute_sbox(in_vars, out_vars):
        for i in range(8): alloc(out_vars[i])
        for i in range(8): qc.cx(pool[allocations[in_vars[i]]], pool[allocations[out_vars[i]]])
        anc4_vars = [f'anc{i}' for i in range(4)]
        anc4_qs = [alloc(v) for v in anc4_vars]
        q_out = [pool[allocations[v]] for v in out_vars]
        q_anc = [pool[q] for q in anc4_qs]
        sub_byte_paper_inplace(q_out, q_anc, qc, inv=False)
        for v in anc4_vars: free(v)

    def uncompute_sbox(in_vars, out_vars):
        anc4_vars = [f'anc{i}' for i in range(4)]
        anc4_qs = [alloc(v) for v in anc4_vars]
        q_out = [pool[allocations[v]] for v in out_vars]
        q_anc = [pool[q] for q in anc4_qs]
        sub_byte_paper_inplace(q_out, q_anc, qc, inv=True)
        for v in anc4_vars: free(v)
        for i in reversed(range(8)): qc.cx(pool[allocations[in_vars[i]]], pool[allocations[out_vars[i]]])
        for i in range(8): free(out_vars[i])

    def compute_xor(v1, v2, vout, rcon=None):
        for i in range(8): alloc(vout[i])
        for i in range(8):
            qc.cx(pool[allocations[v1[i]]], pool[allocations[vout[i]]])
            if v2 is not None:
                qc.cx(pool[allocations[v2[i]]], pool[allocations[vout[i]]])
        if rcon:
            for i in range(8):
                if (rcon >> (7-i)) & 1: qc.x(pool[allocations[vout[i]]])

    def uncompute_xor(v1, v2, vout, rcon=None):
        if rcon:
            for i in range(8):
                if (rcon >> (7-i)) & 1: qc.x(pool[allocations[vout[i]]])
        for i in reversed(range(8)):
            if v2 is not None:
                qc.cx(pool[allocations[v2[i]]], pool[allocations[vout[i]]])
            qc.cx(pool[allocations[v1[i]]], pool[allocations[vout[i]]])
        for i in range(8): free(vout[i])

    def compute_state16(src_vars, dst_vars):
        for i in range(16): alloc(dst_vars[i])
        for i in range(16):
            qc.cx(pool[allocations[src_vars[i]]], pool[allocations[dst_vars[i]]])

    def uncompute_state16(src_vars, dst_vars):
        for i in reversed(range(16)):
            qc.cx(pool[allocations[src_vars[i]]], pool[allocations[dst_vars[i]]])
        for i in range(16): free(dst_vars[i])

    def inplace_esop8_xor(src_vars, dst_vars):
        anc4_vars = [f'anc{i}' for i in range(4)]
        anc4_qs = [alloc(v) for v in anc4_vars]
        q_dst = [pool[allocations[v]] for v in dst_vars]
        q_anc = [pool[q] for q in anc4_qs]
        for i in range(8): qc.cx(pool[allocations[src_vars[i]]], q_dst[i])
        sub_byte_paper_inplace(q_dst, q_anc, qc, inv=False)
        for i in range(8): qc.cx(pool[allocations[src_vars[i]]], q_dst[i])
        sub_byte_paper_inplace(q_dst, q_anc, qc, inv=True)
        for i in range(8): qc.cx(pool[allocations[src_vars[i]]], q_dst[i])
        for v in anc4_vars: free(v)

    def compute_esop16(src_vars, dst_vars, fold_bits=None):
        for i in range(16): alloc(dst_vars[i])
        for i in range(16):
            if fold_bits and fold_bits[i]: qc.x(pool[allocations[src_vars[i]]])
            qc.cx(pool[allocations[src_vars[i]]], pool[allocations[dst_vars[i]]])
            if fold_bits and fold_bits[i]: qc.x(pool[allocations[src_vars[i]]])
        anc4_vars = [f'anc{i}' for i in range(4)]
        anc4_qs = [alloc(v) for v in anc4_vars]
        q_dst = [pool[allocations[v]] for v in dst_vars]
        q_anc = [pool[q] for q in anc4_qs]
        sub_byte_paper_inplace(q_dst[0:8], q_anc, qc, inv=False)
        sub_byte_paper_inplace(q_dst[8:16], q_anc, qc, inv=False)
        for v in anc4_vars: free(v)

    def uncompute_esop16(src_vars, dst_vars, fold_bits=None):
        anc4_vars = [f'anc{i}' for i in range(4)]
        anc4_qs = [alloc(v) for v in anc4_vars]
        q_dst = [pool[allocations[v]] for v in dst_vars]
        q_anc = [pool[q] for q in anc4_qs]
        sub_byte_paper_inplace(q_dst[8:16], q_anc, qc, inv=True)
        sub_byte_paper_inplace(q_dst[0:8], q_anc, qc, inv=True)
        for v in anc4_vars: free(v)
        for i in reversed(range(16)):
            if fold_bits and fold_bits[i]: qc.x(pool[allocations[src_vars[i]]])
            qc.cx(pool[allocations[src_vars[i]]], pool[allocations[dst_vars[i]]])
            if fold_bits and fold_bits[i]: qc.x(pool[allocations[src_vars[i]]])
        for i in range(16): free(dst_vars[i])

    def derive_K2():
        for i in range(8): alloc(V_W4[i])
        for i in range(8): alloc(V_W5[i])
        for i in range(8): qc.cx(pool[allocations[V_KEY[i]]], pool[allocations[V_W4[i]]])
        inplace_esop8_xor(V_KEY[8:16], V_W4)
        for i in range(8):
            if (RCON1_VAL >> (7-i)) & 1: qc.x(pool[allocations[V_W4[i]]])
        for i in range(8): qc.cx(pool[allocations[V_KEY[8+i]]], pool[allocations[V_W5[i]]])
        for i in range(8): qc.cx(pool[allocations[V_W4[i]]], pool[allocations[V_W5[i]]])
        for i in range(8):
            if (RCON2_VAL >> (7-i)) & 1: qc.x(pool[allocations[V_W4[i]]])
        inplace_esop8_xor(V_W5, V_W4)
        for i in range(8): qc.cx(pool[allocations[V_W4[i]]], pool[allocations[V_W5[i]]])

    def free_K2():
        for i in range(8): qc.cx(pool[allocations[V_W4[i]]], pool[allocations[V_W5[i]]])
        inplace_esop8_xor(V_W5, V_W4)
        for i in range(8):
            if (RCON2_VAL >> (7-i)) & 1: qc.x(pool[allocations[V_W4[i]]])
        for i in range(8): qc.cx(pool[allocations[V_W4[i]]], pool[allocations[V_W5[i]]])
        for i in range(8): qc.cx(pool[allocations[V_KEY[8+i]]], pool[allocations[V_W5[i]]])
        for i in range(8):
            if (RCON1_VAL >> (7-i)) & 1: qc.x(pool[allocations[V_W4[i]]])
        inplace_esop8_xor(V_KEY[8:16], V_W4)
        for i in range(8): qc.cx(pool[allocations[V_KEY[i]]], pool[allocations[V_W4[i]]])
        for i in range(8): free(V_W4[i])
        for i in range(8): free(V_W5[i])

    def eval_clauses(clauses):
        for cl_tuple in clauses:
            clause = cl_tuple[0]
            ctrls = []; trivially_sat = False; skip = False
            for lit in clause:
                v = abs(lit)
                if v in V_PT:
                    bit = PT_BITS[v-65]
                    if (bit==1 and lit>0) or (bit==0 and lit<0): trivially_sat=True; break
                    continue
                if v in V_CT:
                    bit = CT_BITS[v-81]
                    if (bit==1 and lit>0) or (bit==0 and lit<0): trivially_sat=True; break
                    continue
                if v in V_ARK0_set:
                    idx = v-97; v = V_KEY[idx]
                    if PT_BITS[idx]==1: lit = -lit
                if v not in allocations: skip=True; break
                ctrls.append((allocations[v], lit>0))
            if trivially_sat or skip or not ctrls: continue
            seen = {}
            for q, pos in ctrls:
                if q in seen:
                    if seen[q] != pos: trivially_sat=True; break
                else: seen[q] = pos
            if trivially_sat: continue
            ctrls = list(seen.items())
            for q, pos in ctrls:
                if pos: qc.x(pool[q])
            q_ctrls = [pool[q] for q,_ in ctrls]
            
            def apply_clause_check(ctrls, tgt):
                if len(ctrls) == 1:
                    qc.cx(ctrls[0], tgt)
                elif len(ctrls) == 2:
                    qc.ccx(ctrls[0], ctrls[1], tgt)
                elif len(ctrls) == 3:
                    a1 = alloc('mcx_a1')
                    qc.ccx(ctrls[0], ctrls[1], pool[a1])
                    qc.ccx(pool[a1], ctrls[2], tgt)
                    qc.ccx(ctrls[0], ctrls[1], pool[a1])
                    free('mcx_a1')
                elif len(ctrls) == 4:
                    a1 = alloc('mcx_a1')
                    a2 = alloc('mcx_a2')
                    qc.ccx(ctrls[0], ctrls[1], pool[a1])
                    qc.ccx(ctrls[2], ctrls[3], pool[a2])
                    qc.ccx(pool[a1], pool[a2], tgt)
                    qc.ccx(ctrls[2], ctrls[3], pool[a2])
                    qc.ccx(ctrls[0], ctrls[1], pool[a1])
                    free('mcx_a2')
                    free('mcx_a1')
                else:
                    qc.mcx(ctrls, tgt)
                    
            apply_clause_check(q_ctrls, pool[clause_a])
            qc.cx(pool[clause_a], pool[viol_flag])
            apply_clause_check(q_ctrls, pool[clause_a])
            for q, pos in ctrls:
                if pos: qc.x(pool[q])

    # ── Pebbling schedule 
    compute_sbox(V_KEY[8:16], V_SB1)
    eval_clauses(stage_clauses['SB1'])
    compute_xor(V_KEY[0:8], V_SB1, V_W2, rcon=RCON1_VAL)
    eval_clauses(stage_clauses['W2'])

    compute_xor(V_KEY[8:16], V_W2, V_W3)
    eval_clauses(stage_clauses['W3'])

    compute_sbox(V_W3, V_SB2)
    eval_clauses(stage_clauses['SB2'])
    compute_xor(V_W2, V_SB2, V_W4, rcon=RCON2_VAL)
    eval_clauses(stage_clauses['W4'])

    compute_xor(V_W3, V_W4, V_W5)
    eval_clauses(stage_clauses['W5'])

    # ARK0
    compute_state16(V_KEY, V_ARK0)
    for i in range(16):
        if PT_BITS[i]: qc.x(pool[allocations[V_ARK0[i]]])
    eval_clauses(stage_clauses['ARK0'])

    # SR
    compute_esop16(V_KEY, V_SR, fold_bits=PT_BITS)
    eval_clauses(stage_clauses['SR'])

    # STZ1
    compute_esop16(V_W4+V_W5, V_STZ, fold_bits=CT_BITS)
    eval_clauses(stage_clauses['STZ1'])

    # STZ
    eval_clauses(stage_clauses['STZ'])

    # DEEP
    compute_esop16(V_STZ, V_DEEP)
    eval_clauses(stage_clauses['DEEP'])

    # Phase flip
    qc.x(pool[viol_flag])
    qc.cx(pool[viol_flag], pool[target])
    qc.x(pool[viol_flag])
    print(f"      [Transpiling Oracle B ({label}) into basic hardware gates ('u', 'cx') for strict logical depth...]")
    tc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=2)
    ops   = tc.count_ops()
    gates = {k: v for k, v in ops.items()}
    total = sum(v for k, v in gates.items() if k != 'barrier')
    depth = tc.depth()
    return qc, peak[0], total, gates, depth



def run_oracle_B_iterations(clause_list, iterations, label="OracleB"):
    qc_oracle, peak, total_g, gates, depth = build_oracle_B(clause_list, label)
    diff_total_g, diff_depth, diff_ops = build_diffusion()
    
    total_iter_gates = iterations * (total_g + diff_total_g)
    total_iter_depth = iterations * (depth + diff_depth)
    
    return total_iter_gates, total_iter_depth

def build_diffusion():
    n = N_QUBITS
    qr = QuantumRegister(n, "q")
    qc = QuantumCircuit(qr, name="Diffusion")
    qc.h(qr)
    qc.x(qr)
    qc.h(qr[n-1])
    qc.mcx(list(qr[:n-1]), qr[n-1])
    qc.h(qr[n-1])
    qc.x(qr)
    qc.h(qr)
    tc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=2)
    ops   = tc.count_ops()
    total = sum(v for k,v in ops.items() if k != 'barrier')
    depth = tc.depth()
    return total, depth, dict(ops)

def verify_oracle_truth_table(clause_list, pt=PT, ct=CT):
    from utils import generate_assignment, check_clauses
    surviving_keys = []
    for key in range(65536):
        asgn = generate_assignment(pt, ct, key)
        if check_clauses(asgn, clause_list):
            surviving_keys.append(key)
    return len(surviving_keys), surviving_keys

def run_aer_simulator(qc_oracle, iterations, diff_qc=None):
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    print(f"\n[AER] Building physical Grover Iteration circuit for {iterations} iterations...")
    pool_size = qc_oracle.num_qubits
    full_qc = QuantumCircuit(pool_size, 16)
    full_qc.h(range(16))
    
    if diff_qc is None:
        diff_qc_small = QuantumCircuit(16)
        diff_qc_small.h(range(16))
        diff_qc_small.x(range(16))
        diff_qc_small.h(15)
        diff_qc_small.mcx(list(range(15)), 15)
        diff_qc_small.h(15)
        diff_qc_small.x(range(16))
        diff_qc_small.h(range(16))
        diff_qc = QuantumCircuit(pool_size)
        diff_qc.compose(diff_qc_small, qubits=list(range(16)), inplace=True)
        
    for _ in range(iterations):
        full_qc.compose(qc_oracle, inplace=True)
        full_qc.compose(diff_qc, inplace=True)
        
    full_qc.measure(range(16), range(16))
    
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
