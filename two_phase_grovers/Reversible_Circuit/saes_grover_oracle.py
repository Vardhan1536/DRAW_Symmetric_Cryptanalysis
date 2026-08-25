import os, sys
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import LinearFunction

HERE = os.path.dirname(os.path.abspath(__file__))
GROVERS_ROOT = os.path.abspath(os.path.join(HERE, '..'))
GITHUB_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

sys.path.insert(0, HERE)
sys.path.insert(0, GROVERS_ROOT)
sys.path.insert(0, os.path.join(GITHUB_ROOT, 'SAT'))
sys.path.insert(0, os.path.join(GITHUB_ROOT, 'SAT', 'DRAW'))

# Add path for Reversible_circuit.py dependencies
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'SAT_formulation')))

from Reversible_circuit import (
    RCON1_BITS,
    RCON2_BITS,
    add_round_key,
    shift_rows,
    sub_byte_paper_inplace,
    mix_columns_inplace,
    key_schedule_hybrid_inplace
)

def build_oracle_hybrid(pt: int, ct: int) -> QuantumCircuit:
    """
    37-qubit hybrid Grover oracle for S-AES.
    """
    key_reg = QuantumRegister(16, "key")
    state   = QuantumRegister(16, "state")
    anc_s   = QuantumRegister(4,  "anc_s")
    target  = QuantumRegister(1,  "target")
    qc = QuantumCircuit(key_reg, state, anc_s, target)

    w0 = key_reg[0:8]
    w1 = key_reg[8:16]

    # Initialize state with PT
    for i in range(16):
        if (pt >> (15 - i)) & 1:
            qc.x(state[i])

    # ---- FORWARD PASS ----
    add_round_key(state, key_reg, qc)
    key_schedule_hybrid_inplace(w0, w1, anc_s, RCON1_BITS, qc)

    sub_byte_paper_inplace(state[0:8],  anc_s, qc)
    sub_byte_paper_inplace(state[8:16], anc_s, qc)
    shift_rows(state, qc)
    mix_columns_inplace(state[0:8], qc)
    mix_columns_inplace(state[8:16], qc)
    add_round_key(state, key_reg, qc)
    key_schedule_hybrid_inplace(w0, w1, anc_s, RCON2_BITS, qc)

    sub_byte_paper_inplace(state[0:8],  anc_s, qc)
    sub_byte_paper_inplace(state[8:16], anc_s, qc)
    shift_rows(state, qc)
    add_round_key(state, key_reg, qc)

    # ---- CHECK CIPHERTEXT ----
    for i in range(16):
        if not ((ct >> (15 - i)) & 1):
            qc.x(state[i])
    qc.mcx(list(state), target[0])
    for i in range(16):
        if not ((ct >> (15 - i)) & 1):
            qc.x(state[i])

    # ---- REVERSE PASS ----
    add_round_key(state, key_reg, qc)
    shift_rows(state, qc)
    sub_byte_paper_inplace(state[8:16], anc_s, qc, inv=True)
    sub_byte_paper_inplace(state[0:8],  anc_s, qc, inv=True)
    key_schedule_hybrid_inplace(w0, w1, anc_s, RCON2_BITS, qc, inv=True)

    add_round_key(state, key_reg, qc)
    mix_columns_inplace(state[8:16], qc, inv=True)
    mix_columns_inplace(state[0:8],  qc, inv=True)
    shift_rows(state, qc)
    sub_byte_paper_inplace(state[8:16], anc_s, qc, inv=True)
    sub_byte_paper_inplace(state[0:8],  anc_s, qc, inv=True)
    key_schedule_hybrid_inplace(w0, w1, anc_s, RCON1_BITS, qc, inv=True)

    add_round_key(state, key_reg, qc)

    # Clear PT
    for i in range(16):
        if (pt >> (15 - i)) & 1:
            qc.x(state[i])

    return qc


if __name__ == "__main__":
    import sys, os
    from qiskit import transpile, QuantumRegister, QuantumCircuit

    # Ensure the script can find the dependencies in SAT_formulation
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'SAT_formulation')))
    from Reversible_circuit import build_full_saes_circuit

    print("======================================================")
    print("1. ISOLATED FORWARD PASS (Encryption Only)")
    print("======================================================")
    print("This matches what Wang et al. measured in their table.")
    
    key_reg = QuantumRegister(16, 'key')
    state_reg = QuantumRegister(16, 'state')
    anc_s = QuantumRegister(4, 'anc')
    qc_forward = QuantumCircuit(key_reg, state_reg, anc_s)
    
    # Build only the forward pass
    build_full_saes_circuit(key_reg, state_reg, anc_s, qc_forward, inv=False)
    
    raw_fwd = dict(qc_forward.count_ops())
    print("\nPre-Transpilation Gate Counts (Forward Pass):")
    for gate, count in raw_fwd.items():
        print(f"  {gate.ljust(15)}: {count}")
        
    qc_fwd_transpiled = transpile(qc_forward, basis_gates=['u', 'cx'], optimization_level=3)
    final_fwd = dict(qc_fwd_transpiled.count_ops())
    print(f"\nFinal Transpiled Logical Depth : {qc_fwd_transpiled.depth()}")
    print(f"Final Transpiled Total Gates : {sum(final_fwd.values())}")

    print("\n\n======================================================")
    print("2. FULL GROVER ORACLE (Compute + MCX Check + Uncompute)")
    print("======================================================")
    print("This matches the honest 'Ours' measurement in your paper.")
    
    qc_full = build_oracle_hybrid(pt=0x1234, ct=0x5678)
    
    raw_full = dict(qc_full.count_ops())
    print("\nPre-Transpilation Gate Counts (Full Oracle):")
    for gate, count in raw_full.items():
        print(f"  {gate.ljust(15)}: {count}")
    
    print("\nTranspiling full oracle (This might take a minute due to the 16-control MCX)...")
    qc_full_transpiled = transpile(qc_full, basis_gates=['u', 'cx'], optimization_level=3)
    final_full = dict(qc_full_transpiled.count_ops())
    
    print(f"\nFinal Transpiled Logical Depth : {qc_full_transpiled.depth()}")
    print(f"Final Transpiled Total Gates : {sum(final_full.values())}")
    print("\nFinal Gate Breakdown (Full Oracle):")
    for gate, count in final_full.items():
        print(f"  {gate.ljust(15)}: {count}")

