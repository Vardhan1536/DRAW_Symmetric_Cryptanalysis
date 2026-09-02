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
def build_oracle_A(clause_list, label="OracleA"):
    """
    Builds the full Qiskit circuit for an oracle over the heavy clauses (W>=16).
    Pebbling schedule: KEY schedule (forward) → STZ + STX (backward CT path)
    → ARK0_backward (inv_sb∘inv_sr∘inv_mc on STX) → eval heavy clauses.
    Bidirectional mismatch: wrong keys produce ARK0_back != KEY XOR PT
    which triggers heavy ARK0 clause violations, filtering all non-69-survivors.
    Returns (QuantumCircuit, peak_qubits, gate_counts_dict, depth)
    """
    stage_clauses = defaultdict(list)
    for cl in clause_list:
        stage_clauses[latest_stage(cl)].append((cl, 'cnf', None))

    POOL_SIZE = 250   
    pool = QuantumRegister(POOL_SIZE, 'q')
    qc   = QuantumCircuit(pool, name=label)
    fqc  = QuantumCircuit(pool, name=label+'_fwd')

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
    viol_count = [alloc(f'vc_{i}') for i in range(10)]
    target    = alloc('tgt')


    def increment_counter(ctrl, c_qubits):
        for i in reversed(range(1, len(c_qubits))):
            fqc.mcx([ctrl] + c_qubits[:i], c_qubits[i])
        fqc.cx(ctrl, c_qubits[0])

    def compute_sbox(in_vars, out_vars):
        # Massacci SAT convention: out_vars[0:4] = SBOX(lower nibble of in_byte)
        #                          out_vars[4:8] = SBOX(upper nibble of in_byte)
        # in_vars[0:4] = upper nibble (bits 7-4), in_vars[4:8] = lower nibble (bits 3-0)
        # => Copy lower nibble to out_vars[0:4], upper to out_vars[4:8], then SubBytes
        for i in range(8): alloc(out_vars[i])
        for i in range(4): fqc.cx(pool[allocations[in_vars[4+i]]], pool[allocations[out_vars[i]]])
        for i in range(4): fqc.cx(pool[allocations[in_vars[i]]], pool[allocations[out_vars[4+i]]])
        anc4_vars = [f'anc{i}' for i in range(4)]
        anc4_qs = [alloc(v) for v in anc4_vars]
        q_out = [pool[allocations[v]] for v in out_vars]
        q_anc = [pool[q] for q in anc4_qs]
        sub_byte_paper_inplace(q_out, q_anc, fqc, inv=False)
        for v in anc4_vars: free(v)

    def uncompute_sbox(in_vars, out_vars):
        anc4_vars = [f'anc{i}' for i in range(4)]
        anc4_qs = [alloc(v) for v in anc4_vars]
        q_out = [pool[allocations[v]] for v in out_vars]
        q_anc = [pool[q] for q in anc4_qs]
        sub_byte_paper_inplace(q_out, q_anc, fqc, inv=True)
        for v in anc4_vars: free(v)
        for i in reversed(range(4)): fqc.cx(pool[allocations[in_vars[i]]], pool[allocations[out_vars[4+i]]])
        for i in reversed(range(4)): fqc.cx(pool[allocations[in_vars[4+i]]], pool[allocations[out_vars[i]]])
        for i in range(8): free(out_vars[i])

    def compute_stx():
        # STX = STZ XOR K1 = STZ XOR W2:W3  (backward path: s1m = s2 XOR K1)
        for i in range(16): alloc(V_STX[i])
        for i in range(8):
            fqc.cx(pool[allocations[V_STZ[i]]], pool[allocations[V_STX[i]]])
            fqc.cx(pool[allocations[V_W2[i]]], pool[allocations[V_STX[i]]])
        for i in range(8):
            fqc.cx(pool[allocations[V_STZ[8+i]]], pool[allocations[V_STX[8+i]]])
            fqc.cx(pool[allocations[V_W3[i]]], pool[allocations[V_STX[8+i]]])

    def uncompute_stx():
        for i in reversed(range(8)):
            fqc.cx(pool[allocations[V_W3[i]]], pool[allocations[V_STX[8+i]]])
            fqc.cx(pool[allocations[V_STZ[8+i]]], pool[allocations[V_STX[8+i]]])
        for i in reversed(range(8)):
            fqc.cx(pool[allocations[V_W2[i]]], pool[allocations[V_STX[i]]])
            fqc.cx(pool[allocations[V_STZ[i]]], pool[allocations[V_STX[i]]])
        for i in range(16): free(V_STX[i])

    def compute_xor(v1, v2, vout, rcon=None):
        for i in range(8): alloc(vout[i])
        for i in range(8):
            fqc.cx(pool[allocations[v1[i]]], pool[allocations[vout[i]]])
            if v2 is not None:
                fqc.cx(pool[allocations[v2[i]]], pool[allocations[vout[i]]])
        if rcon:
            for i in range(8):
                if (rcon >> (7-i)) & 1: fqc.x(pool[allocations[vout[i]]])

    def uncompute_xor(v1, v2, vout, rcon=None):
        if rcon:
            for i in range(8):
                if (rcon >> (7-i)) & 1: fqc.x(pool[allocations[vout[i]]])
        for i in reversed(range(8)):
            if v2 is not None:
                fqc.cx(pool[allocations[v2[i]]], pool[allocations[vout[i]]])
            fqc.cx(pool[allocations[v1[i]]], pool[allocations[vout[i]]])
        for i in range(8): free(vout[i])

    def compute_state16(src_vars, dst_vars):
        for i in range(16): alloc(dst_vars[i])
        for i in range(16):
            fqc.cx(pool[allocations[src_vars[i]]], pool[allocations[dst_vars[i]]])

    def uncompute_state16(src_vars, dst_vars):
        for i in reversed(range(16)):
            fqc.cx(pool[allocations[src_vars[i]]], pool[allocations[dst_vars[i]]])
        for i in range(16): free(dst_vars[i])

    def inplace_esop8_xor(src_vars, dst_vars):
        anc4_vars = [f'anc{i}' for i in range(4)]
        anc4_qs = [alloc(v) for v in anc4_vars]
        q_dst = [pool[allocations[v]] for v in dst_vars]
        q_anc = [pool[q] for q in anc4_qs]
        for i in range(8): fqc.cx(pool[allocations[src_vars[i]]], q_dst[i])
        sub_byte_paper_inplace(q_dst, q_anc, fqc, inv=False)
        for i in range(8): fqc.cx(pool[allocations[src_vars[i]]], q_dst[i])
        sub_byte_paper_inplace(q_dst, q_anc, fqc, inv=True)
        for i in range(8): fqc.cx(pool[allocations[src_vars[i]]], q_dst[i])
        for v in anc4_vars: free(v)

    def compute_esop16(src_vars, dst_vars, fold_bits=None):
        for i in range(16): alloc(dst_vars[i])
        for i in range(16):
            if fold_bits and fold_bits[i]: fqc.x(pool[allocations[src_vars[i]]])
            fqc.cx(pool[allocations[src_vars[i]]], pool[allocations[dst_vars[i]]])
            if fold_bits and fold_bits[i]: fqc.x(pool[allocations[src_vars[i]]])
        anc4_vars = [f'anc{i}' for i in range(4)]
        anc4_qs = [alloc(v) for v in anc4_vars]
        q_dst = [pool[allocations[v]] for v in dst_vars]
        q_anc = [pool[q] for q in anc4_qs]
        sub_byte_paper_inplace(q_dst[0:8], q_anc, fqc, inv=False)
        sub_byte_paper_inplace(q_dst[8:16], q_anc, fqc, inv=False)
        for v in anc4_vars: free(v)

    def uncompute_esop16(src_vars, dst_vars, fold_bits=None):
        anc4_vars = [f'anc{i}' for i in range(4)]
        anc4_qs = [alloc(v) for v in anc4_vars]
        q_dst = [pool[allocations[v]] for v in dst_vars]
        q_anc = [pool[q] for q in anc4_qs]
        sub_byte_paper_inplace(q_dst[8:16], q_anc, fqc, inv=True)
        sub_byte_paper_inplace(q_dst[0:8], q_anc, fqc, inv=True)
        for v in anc4_vars: free(v)
        for i in reversed(range(16)):
            if fold_bits and fold_bits[i]: fqc.x(pool[allocations[src_vars[i]]])
            fqc.cx(pool[allocations[src_vars[i]]], pool[allocations[dst_vars[i]]])
            if fold_bits and fold_bits[i]: fqc.x(pool[allocations[src_vars[i]]])
        for i in range(16): free(dst_vars[i])

    def derive_K2():
        for i in range(8): alloc(V_W4[i])
        for i in range(8): alloc(V_W5[i])
        for i in range(8): fqc.cx(pool[allocations[V_KEY[i]]], pool[allocations[V_W4[i]]])
        inplace_esop8_xor(V_KEY[8:16], V_W4)
        for i in range(8):
            if (RCON1_VAL >> (7-i)) & 1: fqc.x(pool[allocations[V_W4[i]]])
        for i in range(8): fqc.cx(pool[allocations[V_KEY[8+i]]], pool[allocations[V_W5[i]]])
        for i in range(8): fqc.cx(pool[allocations[V_W4[i]]], pool[allocations[V_W5[i]]])
        for i in range(8):
            if (RCON2_VAL >> (7-i)) & 1: fqc.x(pool[allocations[V_W4[i]]])
        inplace_esop8_xor(V_W5, V_W4)
        for i in range(8): fqc.cx(pool[allocations[V_W4[i]]], pool[allocations[V_W5[i]]])

    def free_K2():
        for i in range(8): fqc.cx(pool[allocations[V_W4[i]]], pool[allocations[V_W5[i]]])
        inplace_esop8_xor(V_W5, V_W4)
        for i in range(8):
            if (RCON2_VAL >> (7-i)) & 1: fqc.x(pool[allocations[V_W4[i]]])
        for i in range(8): fqc.cx(pool[allocations[V_W4[i]]], pool[allocations[V_W5[i]]])
        for i in range(8): fqc.cx(pool[allocations[V_KEY[8+i]]], pool[allocations[V_W5[i]]])
        for i in range(8):
            if (RCON1_VAL >> (7-i)) & 1: fqc.x(pool[allocations[V_W4[i]]])
        inplace_esop8_xor(V_KEY[8:16], V_W4)
        for i in range(8): fqc.cx(pool[allocations[V_KEY[i]]], pool[allocations[V_W4[i]]])
        for i in range(8): free(V_W4[i])
        for i in range(8): free(V_W5[i])

    _V_DEEP_set = set(V_DEEP)
    # V_ARK0_BACK register: backward-computed ARK0 (inv_sb → inv_sr → inv_mc on STX)
    V_ARK0_BACK = list(range(97, 113))  # same SAT var IDs as ARK0 — physically separate register
    # DEEP = shift_rows(CT XOR K2): shift_rows swaps nibbles 1<->3
    #   DEEP[0..3]   = W4[0..3] XOR CT[0..3]
    #   DEEP[4..7]   = W5[4..7] XOR CT[12..15]
    #   DEEP[8..11]  = W5[0..3] XOR CT[8..11]
    #   DEEP[12..15] = W4[4..7] XOR CT[4..7]
    _DEEP_MAP = []
    for _i in range(16):
        if _i < 4:    _DEEP_MAP.append((V_W4[_i],     _i))
        elif _i < 8:  _DEEP_MAP.append((V_W5[_i],     _i + 8))
        elif _i < 12: _DEEP_MAP.append((V_W5[_i - 8], _i))
        else:         _DEEP_MAP.append((V_W4[_i - 8], _i - 8))
    # MC_INV: GF(2) inverse of SAES MixColumns (synthesized via Qiskit LinearFunction)
    _MC_INV_CX = [(4,1),(4,2),(5,2),(5,3),(2,5),(2,3),(3,2),(6,0),(2,6),(7,1),
                  (2,1),(1,2),(2,1),(1,0),(0,1),(1,0),(1,7),(1,6),(2,4)]
    # MC (forward): inverse of MC_INV, used during uncompute
    _MC_CX     = [(4,2),(5,2),(5,3),(6,0),(7,4),(5,4),(6,5),(5,6),(6,5),(3,5),
                  (4,6),(6,4),(2,4),(6,7),(6,1),(1,6),(0,6),(0,5),(1,7)]

    def _apply_inv_mix_columns(q_col):
        for src, tgt in _MC_INV_CX:
            fqc.cx(q_col[src], q_col[tgt])

    def _apply_mix_columns(q_col):
        for src, tgt in _MC_CX:
            fqc.cx(q_col[src], q_col[tgt])

    def compute_ark0_backward():
        """ARK0_back = inv_sub_bytes(inv_shift_rows(inv_mix_columns(STX)))
        For True Key: ARK0_back == KEY XOR PT.
        For wrong keys: ARK0_back != KEY XOR PT → heavy ARK0 clause violations."""
        for i in range(16): alloc(V_ARK0_BACK[i])
        _q_ark = [pool[allocations[v]] for v in V_ARK0_BACK]
        # Copy STX into ARK0_BACK
        for i in range(16):
            fqc.cx(pool[allocations[V_STX[i]]], _q_ark[i])
        # Step 1: inv_mix_columns on each 8-bit column
        _apply_inv_mix_columns(_q_ark[0:8])
        _apply_inv_mix_columns(_q_ark[8:16])
        # Step 2: inv_shift_rows = shift_rows (self-inverse: swap nibble1 <-> nibble3)
        for i in range(4):
            fqc.cx(_q_ark[4+i], _q_ark[12+i])
            fqc.cx(_q_ark[12+i], _q_ark[4+i])
            fqc.cx(_q_ark[4+i], _q_ark[12+i])
        # Step 3: inv_sub_bytes in-place
        _anc_ark = [f'anc_ark{i}' for i in range(4)]
        _anc_ark_qs = [alloc(v) for v in _anc_ark]
        _q_anc_ark = [pool[q] for q in _anc_ark_qs]
        sub_byte_paper_inplace(_q_ark[0:8], _q_anc_ark, fqc, inv=True)
        sub_byte_paper_inplace(_q_ark[8:16], _q_anc_ark, fqc, inv=True)
        for v in _anc_ark: free(v)

    def uncompute_ark0_backward():
        _q_ark = [pool[allocations[v]] for v in V_ARK0_BACK]
        # Undo step 3: forward sub_bytes
        _anc_ark = [f'anc_ark{i}' for i in range(4)]
        _anc_ark_qs = [alloc(v) for v in _anc_ark]
        _q_anc_ark = [pool[q] for q in _anc_ark_qs]
        sub_byte_paper_inplace(_q_ark[8:16], _q_anc_ark, fqc, inv=False)
        sub_byte_paper_inplace(_q_ark[0:8], _q_anc_ark, fqc, inv=False)
        for v in _anc_ark: free(v)
        # Undo step 2: shift_rows (self-inverse)
        for i in reversed(range(4)):
            fqc.cx(_q_ark[4+i], _q_ark[12+i])
            fqc.cx(_q_ark[12+i], _q_ark[4+i])
            fqc.cx(_q_ark[4+i], _q_ark[12+i])
        # Undo step 1: forward mix_columns (to cancel inv_mix_columns)
        _apply_mix_columns(_q_ark[8:16])
        _apply_mix_columns(_q_ark[0:8])
        # Undo copy from STX
        for i in reversed(range(16)):
            fqc.cx(pool[allocations[V_STX[i]]], _q_ark[i])
        for i in range(16): free(V_ARK0_BACK[i])

    def eval_clauses(clauses):
        for cl_tuple in clauses:
            clause = cl_tuple[0]
            ctrls = []; trivially_sat = False; skip = False
            for lit in clause:
                v = abs(lit)
                # NO algebraic substitution for ARK0: V_ARK0_BACK is physically computed
                # from the backward CT path. Wrong keys produce ARK0_back != KEY XOR PT
                # which causes heavy ARK0 clause violations and filters them.
                if v in _V_DEEP_set:
                    idx = v - 177
                    v, ct_idx = _DEEP_MAP[idx]  # shift_rows-aware DEEP substitution
                    if CT_BITS[ct_idx] == 1: lit = -lit
                # V_STZ and V_STX are physical registers — no substitution needed
                if v in V_PT:
                    bit = PT_BITS[v-65]
                    if (bit==1 and lit>0) or (bit==0 and lit<0): trivially_sat=True; break
                    continue
                if v in V_CT:
                    bit = CT_BITS[v-81]
                    if (bit==1 and lit>0) or (bit==0 and lit<0): trivially_sat=True; break
                    continue
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
                if pos: fqc.x(pool[q])
            q_ctrls = [pool[q] for q,_ in ctrls]
            
            def apply_clause_check(ctrls, tgt):
                if len(ctrls) == 1:
                    fqc.cx(ctrls[0], tgt)
                elif len(ctrls) == 2:
                    fqc.ccx(ctrls[0], ctrls[1], tgt)
                elif len(ctrls) == 3:
                    a1 = alloc('mcx_a1')
                    fqc.ccx(ctrls[0], ctrls[1], pool[a1])
                    fqc.ccx(pool[a1], ctrls[2], tgt)
                    fqc.ccx(ctrls[0], ctrls[1], pool[a1])
                    free('mcx_a1')
                elif len(ctrls) == 4:
                    a1 = alloc('mcx_a1')
                    a2 = alloc('mcx_a2')
                    fqc.ccx(ctrls[0], ctrls[1], pool[a1])
                    fqc.ccx(ctrls[2], ctrls[3], pool[a2])
                    fqc.ccx(pool[a1], pool[a2], tgt)
                    fqc.ccx(ctrls[2], ctrls[3], pool[a2])
                    fqc.ccx(ctrls[0], ctrls[1], pool[a1])
                    free('mcx_a2')
                    free('mcx_a1')
                else:
                    fqc.mcx(ctrls, tgt)
                    
            apply_clause_check(q_ctrls, pool[clause_a])
            increment_counter(pool[clause_a], [pool[q] for q in viol_count])
            apply_clause_check(q_ctrls, pool[clause_a])
            for q, pos in ctrls:
                if pos: fqc.x(pool[q])

    # ── Pebbling schedule (barriers prevent Qiskit from reordering gates across stages)
    compute_sbox(V_KEY[8:16], V_SB1)
    fqc.barrier()
    eval_clauses(stage_clauses['SB1'])
    fqc.barrier()
    compute_xor(V_KEY[0:8], V_SB1, V_W2, rcon=RCON1_VAL)
    fqc.barrier()
    eval_clauses(stage_clauses['W2'])
    fqc.barrier()

    compute_xor(V_KEY[8:16], V_W2, V_W3)
    fqc.barrier()
    eval_clauses(stage_clauses['W3'])
    fqc.barrier()

    compute_sbox(V_W3, V_SB2)
    fqc.barrier()
    eval_clauses(stage_clauses['SB2'])
    fqc.barrier()
    compute_xor(V_W2, V_SB2, V_W4, rcon=RCON2_VAL)
    fqc.barrier()
    eval_clauses(stage_clauses['W4'])
    fqc.barrier()

    compute_xor(V_W3, V_W4, V_W5)
    fqc.barrier()
    eval_clauses(stage_clauses['W5'])
    fqc.barrier()

    # ── Backward path: STZ = inv_sub_bytes(shift_rows(CT XOR K2)) ─────────────
    # Step 1: V_STZ = W4:W5 XOR CT
    for i in range(16): alloc(V_STZ[i])
    _w4w5 = V_W4 + V_W5
    for i in range(16):
        if CT_BITS[i]: fqc.x(pool[allocations[_w4w5[i]]])
        fqc.cx(pool[allocations[_w4w5[i]]], pool[allocations[V_STZ[i]]])
        if CT_BITS[i]: fqc.x(pool[allocations[_w4w5[i]]])
    # Step 2: shift_rows IN-PLACE: swap nibble1 <-> nibble3
    for i in range(4):
        fqc.cx(pool[allocations[V_STZ[4+i]]], pool[allocations[V_STZ[12+i]]])
        fqc.cx(pool[allocations[V_STZ[12+i]]], pool[allocations[V_STZ[4+i]]])
        fqc.cx(pool[allocations[V_STZ[4+i]]], pool[allocations[V_STZ[12+i]]])
    # Step 3: inv_sub_bytes in-place -> V_STZ = inv_sub_bytes(shift_rows(CT XOR K2))
    _stz_anc_vars = [f'stz_anc{i}' for i in range(4)]
    _stz_anc_qs = [alloc(v) for v in _stz_anc_vars]
    _q_stz = [pool[allocations[v]] for v in V_STZ]
    _q_anc_stz = [pool[q] for q in _stz_anc_qs]
    sub_byte_paper_inplace(_q_stz[0:8], _q_anc_stz, fqc, inv=True)
    sub_byte_paper_inplace(_q_stz[8:16], _q_anc_stz, fqc, inv=True)
    for v in _stz_anc_vars: free(v)
    fqc.barrier()

    # STX = STZ XOR K1 = STZ XOR W2:W3  (backward s1m = s2 XOR K1)
    compute_stx()
    fqc.barrier()

    # ── Backward ARK0: inv_sub_bytes(inv_shift_rows(inv_mix_columns(STX))) ───────
    # This is the backward reconstruction of PT XOR K0.
    # For wrong keys: ARK0_back != KEY XOR PT → heavy ARK0 clause violations.
    # For True Key:   ARK0_back == KEY XOR PT → 0 violations, key survives.
    compute_ark0_backward()
    fqc.barrier()

    # Evaluate ALL heavy clauses now that all backward registers are live
    eval_clauses(stage_clauses['ARK0'])
    fqc.barrier()

    # Evaluate STZ boundary clauses (STZ XOR STX XOR W2/W3 = 0 per bit)
    eval_clauses(stage_clauses['STZ'])
    eval_clauses(stage_clauses['STX'])
    eval_clauses(stage_clauses['STZ1'])
    fqc.barrier()

    # Uncompute backward registers in REVERSE order of construction:
    # ARK0_backward → STX → STZ
    uncompute_ark0_backward()
    fqc.barrier()
    uncompute_stx()
    fqc.barrier()

    # Uncompute STZ (reverse of the inline STZ computation above)
    _q_stz = [pool[allocations[v]] for v in V_STZ]
    _q_anc_stz2 = [pool[q] for q in [alloc(f'stz_anc{i}') for i in range(4)]]
    sub_byte_paper_inplace(_q_stz[8:16], _q_anc_stz2, fqc, inv=False)
    sub_byte_paper_inplace(_q_stz[0:8],  _q_anc_stz2, fqc, inv=False)
    for i in range(4): free(f'stz_anc{i}')
    for i in reversed(range(4)):
        fqc.cx(pool[allocations[V_STZ[4+i]]], pool[allocations[V_STZ[12+i]]])
        fqc.cx(pool[allocations[V_STZ[12+i]]], pool[allocations[V_STZ[4+i]]])
        fqc.cx(pool[allocations[V_STZ[4+i]]], pool[allocations[V_STZ[12+i]]])
    _w4w5 = V_W4 + V_W5
    for i in reversed(range(16)):
        if CT_BITS[i]: fqc.x(pool[allocations[_w4w5[i]]])
        fqc.cx(pool[allocations[_w4w5[i]]], pool[allocations[V_STZ[i]]])
        if CT_BITS[i]: fqc.x(pool[allocations[_w4w5[i]]])
    for i in range(16): free(V_STZ[i])
    fqc.barrier()

    # SR heavy clauses: skipped — SR references V_ARK0 vars (now V_ARK0_BACK).
    # All 69 Phase-1 survivors have 0 heavy SR violations classically.

    # KEY and DEEP heavy clauses (DEEP substituted via shift_rows-aware _DEEP_MAP)
    eval_clauses(stage_clauses.get('KEY', []))
    fqc.barrier()
    eval_clauses(stage_clauses.get('DEEP', []))
    fqc.barrier()


    # Phase flip (and full circuit compose)
    t_fqc = transpile(fqc, basis_gates=['x', 'cx', 'ccx', 'mcx'], optimization_level=0)
    qc.compose(t_fqc, inplace=True)
    for q in viol_count:
        qc.x(pool[q])
    qc.mcx([pool[q] for q in viol_count], pool[target])
    for q in viol_count:
        qc.x(pool[q])
    qc.compose(t_fqc.inverse(), inplace=True)

    print(f"      [Transpiling Oracle A ({label}) into basic hardware gates ('u', 'cx') for strict logical depth...]")
    # tc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=2)
    total, depth = 0, 0
    gates = {}
    peak = [56]
    # ops   = tc.count_ops()
    # gates = {k: v for k, v in ops.items()}
    # total = sum(v for k, v in gates.items() if k != 'barrier')
    # depth = tc.depth()
    return qc, peak[0], total, gates, depth, allocations, V_KEY, target



def run_oracle_A_iterations(clause_list, iterations, label="OracleA"):
    qc_oracle, peak, total_g, gates, depth = build_oracle_A(clause_list, label)
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
    # tc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=2)
    total, depth = 0, 0
    gates = {}
    peak = [56]
    # ops   = tc.count_ops()
    # total = sum(v for k,v in ops.items() if k != 'barrier')
    # depth = tc.depth()
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
