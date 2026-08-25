import os
import sys
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import LinearFunction

HERE = os.path.dirname(os.path.abspath(__file__))
SAT_ROOT = os.path.abspath(os.path.join(HERE, '..', 'SAT_formulation'))
sys.path.insert(0, HERE)
sys.path.insert(0, SAT_ROOT)

from saes import mult4, sub_nibble as sbox_fn

RCON1_BITS = [1, 0, 0, 0, 0, 0, 0, 0]  
RCON2_BITS = [0, 0, 1, 1, 0, 0, 0, 0]  

PN_MAT = [
    [0, 1, 1, 1],   
    [0, 1, 0, 1],   
    [1, 0, 0, 1],   
    [0, 0, 1, 1],   
]

NPA_MAT = [
    [1, 1, 1, 0],   
    [1, 0, 1, 1],   
    [1, 0, 1, 0],   
    [1, 0, 0, 0],   
]
NPA_AFFINE = [1, 0, 0, 1]

_npa_np = np.array(NPA_MAT, dtype=int)
_npa_inv = np.linalg.inv(_npa_np) % 2
NPA_INV_MAT = _npa_inv.astype(int).tolist()

def apply_pn(reg, qc, inv=False):
    """Applies PN basis transformation matrix over GF(2)."""
    mat = np.linalg.inv(np.array(PN_MAT, dtype=float)).astype(int) % 2 if inv else PN_MAT
    qc.append(LinearFunction(mat), list(reg))

def apply_npa(reg, qc, inv=False):
    """Applies NPA affine matrix transformation over GF(2)."""
    if not inv:
        qc.append(LinearFunction(NPA_MAT), list(reg))
        for i, b in enumerate(NPA_AFFINE):
            if b:
                qc.x(reg[i])
    else:
        for i, b in enumerate(NPA_AFFINE):
            if b:
                qc.x(reg[i])
        qc.append(LinearFunction(NPA_INV_MAT), list(reg))

def io_circuit_random(x, y, qc):
    """
    GF(2^4) Multiplicative Inversion Core (12 CCX + 14 CX).
    Computes y ^= x^(-1) in normal basis.
    """
    qc.ccx(x[0], x[3], x[1])
    qc.ccx(x[1], x[2], y[1])
    qc.cx(x[1], y[3])
    qc.cx(x[1], y[2])
    qc.cx(x[1], y[0])
    qc.ccx(x[0], x[3], x[1])
    qc.ccx(x[0], x[2], x[3])
    qc.ccx(x[1], x[3], y[3])
    qc.cx(x[3], y[3])
    qc.cx(x[3], y[0])
    qc.cx(x[3], y[1])
    qc.ccx(x[0], x[2], x[3])
    qc.ccx(x[1], x[2], x[0])
    qc.ccx(x[0], x[3], y[0])
    qc.cx(x[0], y[0])
    qc.ccx(x[1], x[2], x[0])
    qc.ccx(x[1], x[3], x[2])
    qc.ccx(x[0], x[2], y[2])
    qc.cx(x[2], y[1])
    qc.ccx(x[1], x[3], x[2])
    qc.cx(x[0], y[0])
    qc.cx(x[1], y[0])
    qc.cx(x[2], y[0])
    qc.cx(x[2], y[1])
    qc.cx(x[0], y[2])
    qc.cx(x[3], y[3])

def sub_nibble_paper_inplace(nibble, anc4, qc, inv=False):
    """
    In-place 4-bit SubNibble (S-box) using Compute-Swap-Uncompute.
    """
    if not inv:
        apply_pn(nibble, qc)
        io_circuit_random(list(nibble), list(anc4), qc)
        for i in range(4):
            qc.swap(nibble[i], anc4[i])
        io_circuit_random(list(nibble), list(anc4), qc)
        apply_npa(nibble, qc)
    else:
        apply_npa(nibble, qc, inv=True)
        io_circuit_random(list(nibble), list(anc4), qc)
        for i in range(4):
            qc.swap(nibble[i], anc4[i])
        io_circuit_random(list(nibble), list(anc4), qc)
        apply_pn(nibble, qc, inv=True)

def sub_byte_paper_inplace(byte_reg, anc4, qc, inv=False):
    """Applies SubNibble sequentially across both nibbles in an 8-bit byte."""
    sub_nibble_paper_inplace(byte_reg[0:4], anc4, qc, inv=inv)
    sub_nibble_paper_inplace(byte_reg[4:8], anc4, qc, inv=inv)

def shift_rows(state, qc):
    """Reversible ShiftRows: Swaps Nibble 1 (bits 4-7) and Nibble 3 (bits 12-15)."""
    for i in range(4):
        qc.swap(state[4 + i], state[12 + i])

def _mc_matrix():
    M = []
    for i in range(8):
        v = [1 if j == i else 0 for j in range(8)]
        c0, c1 = v[0:4], v[4:8]
        c0_int = sum(c0[3 - k] << k for k in range(4))
        c1_int = sum(c1[3 - k] << k for k in range(4))
        c0_out = c0_int ^ mult4(c1_int)
        c1_out = c1_int ^ mult4(c0_int)
        out_vec = [(c0_out >> (3 - k)) & 1 for k in range(4)] + [(c1_out >> (3 - k)) & 1 for k in range(4)]
        M.append(out_vec)
    return [[M[col][row] for col in range(8)] for row in range(8)]

MC_MAT = _mc_matrix()

def mix_columns_inplace(col_reg, qc, inv=False):
    """
    Reversible 2-nibble MixColumns in GF(2^4) with polynomial x^4 + x + 1.
    Requires exactly 19 CX gates synthesized via LinearFunction.
    """
    qc.append(LinearFunction(MC_MAT), list(col_reg))

def add_round_key(state, key, qc):
    """AddRoundKey: 16 parallel CNOT gates between state and key registers."""
    for i in range(16):
        qc.cx(key[i], state[i])

def key_schedule_hybrid_inplace(w0, w1, anc4, rcon_bits, qc, inv=False):
    """
    Reversible in-place 1-Round Feistel Key Expansion.
    w0_next = w0 ^ RCON ^ SubNib(RotNib(w1)), w1_next = w1 ^ w0_next
    """
    if not inv:
        rot_nib0 = w1[4:8]
        rot_nib1 = w1[0:4]

        sub_nibble_paper_inplace(rot_nib0, anc4, qc, inv=False)
        sub_nibble_paper_inplace(rot_nib1, anc4, qc, inv=False)

        for i in range(8):
            qc.cx(w1[i], w0[i])
            if rcon_bits[i]:
                qc.x(w0[i])

        sub_nibble_paper_inplace(rot_nib0, anc4, qc, inv=True)
        sub_nibble_paper_inplace(rot_nib1, anc4, qc, inv=True)

        for i in range(8):
            qc.cx(w0[i], w1[i])
    else:
        for i in range(8):
            qc.cx(w0[i], w1[i])

        rot_nib0 = w1[4:8]
        rot_nib1 = w1[0:4]

        sub_nibble_paper_inplace(rot_nib0, anc4, qc, inv=False)
        sub_nibble_paper_inplace(rot_nib1, anc4, qc, inv=False)

        for i in range(8):
            if rcon_bits[i]:
                qc.x(w0[i])
            qc.cx(w1[i], w0[i])

        sub_nibble_paper_inplace(rot_nib0, anc4, qc, inv=True)
        sub_nibble_paper_inplace(rot_nib1, anc4, qc, inv=True)

def build_full_saes_circuit(key_reg, state_reg, anc_s, qc, inv=False):
    """
    Builds the complete 2-round reversible S-AES encryption/decryption quantum circuit.
    Uses 36 total qubits (16 key + 16 state + 4 S-box ancillas).
    """
    w0 = key_reg[0:8]
    w1 = key_reg[8:16]

    if not inv:
        key_schedule_hybrid_inplace(w0, w1, anc_s, RCON1_BITS, qc, inv=False)
        key_schedule_hybrid_inplace(w0, w1, anc_s, RCON2_BITS, qc, inv=False)

        add_round_key(state_reg, key_reg, qc)

        sub_byte_paper_inplace(state_reg[0:8], anc_s, qc, inv=False)
        sub_byte_paper_inplace(state_reg[8:16], anc_s, qc, inv=False)
        shift_rows(state_reg, qc)
        mix_columns_inplace(state_reg[0:8], qc, inv=False)
        mix_columns_inplace(state_reg[8:16], qc, inv=False)
        add_round_key(state_reg, key_reg, qc)

        sub_byte_paper_inplace(state_reg[0:8], anc_s, qc, inv=False)
        sub_byte_paper_inplace(state_reg[8:16], anc_s, qc, inv=False)
        shift_rows(state_reg, qc)
        add_round_key(state_reg, key_reg, qc)
    else:
        add_round_key(state_reg, key_reg, qc)
        shift_rows(state_reg, qc)
        sub_byte_paper_inplace(state_reg[8:16], anc_s, qc, inv=True)
        sub_byte_paper_inplace(state_reg[0:8], anc_s, qc, inv=True)

        add_round_key(state_reg, key_reg, qc)
        mix_columns_inplace(state_reg[8:16], qc, inv=True)
        mix_columns_inplace(state_reg[0:8], qc, inv=True)
        shift_rows(state_reg, qc)
        sub_byte_paper_inplace(state_reg[8:16], anc_s, qc, inv=True)
        sub_byte_paper_inplace(state_reg[0:8], anc_s, qc, inv=True)

        add_round_key(state_reg, key_reg, qc)
        key_schedule_hybrid_inplace(w0, w1, anc_s, RCON2_BITS, qc, inv=True)
        key_schedule_hybrid_inplace(w0, w1, anc_s, RCON1_BITS, qc, inv=True)
