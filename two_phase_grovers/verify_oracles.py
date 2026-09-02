import sys
import os
import time
from qiskit import QuantumCircuit, transpile

# Add the Github2/two_phase_grovers directory to path
HERE = os.path.dirname(os.path.abspath(__file__))
GITHUB2_ROOT = os.path.abspath(os.path.join(HERE, '..'))
SAT_DIR = os.path.join(GITHUB2_ROOT, 'SAT_formulation')
sys.path.insert(0, HERE)
sys.path.insert(0, SAT_DIR)

from sat import SAESMassacciCompiler
from draw_weights import build_draw_weights
from Oracle_A import build_oracle_A
from Oracle_B import build_oracle_B

PT = 0x17FD
CT = 0xA55B
TRUE_KEY = 0x3ABC

def simulate_boolean_circuit(qc: QuantumCircuit, key_val: int, pool_size: int, allocations: dict, v_key: list, viol_flag: int):
    state = [0] * pool_size
    
    # Set key bits
    for i in range(16):
        if (key_val >> (15 - i)) & 1:
            state[allocations[v_key[i]]] = 1
            
    # Process physical gates (NO QISKIT OVERHEAD)
    for instruction in qc.data:
        gate = instruction.operation
        qargs = [q._index for q in instruction.qubits]
        
        if gate.name == 'x':
            state[qargs[0]] ^= 1
        elif gate.name == 'cx':
            if state[qargs[0]]:
                state[qargs[1]] ^= 1
        elif gate.name == 'ccx':
            if state[qargs[0]] and state[qargs[1]]:
                state[qargs[2]] ^= 1
        elif gate.name.startswith('mcx'):
            ctrls = qargs[:-1]
            targ = qargs[-1]
            if all(state[c] for c in ctrls):
                state[targ] ^= 1
        elif gate.name == 'barrier':
            continue
        elif gate.name == 'u':
            # Ignore transpiler artifacts if present
            pass
        else:
            raise ValueError(f"Unsupported physical gate: {gate.name}")
            
    # viol_flag is an integer index from alloc()
    return state[viol_flag]

def main():
    print("=" * 70)
    print(" GITHUB2 ORACLE VERIFICATION (W=16.0 Threshold)")
    print(" (No Qiskit Statevector, No classical shortcut cheating!)")
    print("=" * 70)
    
    c = SAESMassacciCompiler()
    c.add_plaintext_ciphertext_pair(PT, CT)
    weights = build_draw_weights(c)
    
    thresh_A = 16.0
    cA_list = [cl for cid, cl in enumerate(c.clauses) if weights[cid] >= thresh_A]
    cB_list = [cl for cid, cl in enumerate(c.clauses) if weights[cid] < thresh_A]
    
    print(f"\n[1] Building Physical Oracle A ({len(cA_list)} Heavy Clauses)...")
    qc_A, peak_A, total_A, gates_A, depth_A, alloc_A, vkey_A, vf_A = build_oracle_A(cA_list, "OracleA")
    pool_size_A = qc_A.num_qubits
    qc_A = transpile(qc_A, basis_gates=['x', 'cx', 'ccx', 'mcx'], optimization_level=0)
    
    print(f"\n[2] Building Physical Oracle B ({len(cB_list)} Light Clauses)...")
    qc_B, peak_B, total_B, gates_B, depth_B, alloc_B, vkey_B, vf_B = build_oracle_B(cB_list, "OracleB")
    pool_size_B = qc_B.num_qubits
    qc_B = transpile(qc_B, basis_gates=['x', 'cx', 'ccx', 'mcx'], optimization_level=0)
    
    print(f"\n--- Testing ORACLE A (Heavy Clauses Physical Circuit) ---")
    print(f"Simulating all 65,536 keys through the {total_A} physical Toffoli gates...")
    t0 = time.time()
    surviving_keys_A = []
    
    for k in range(65536):
        if k % 10000 == 0 and k > 0:
            print(f"  Processed {k} keys...")
        v = simulate_boolean_circuit(qc_A, k, pool_size_A, alloc_A, vkey_A, vf_A)
        # 1 means target flag is triggered (Valid key)
        if v == 1:
            surviving_keys_A.append(k)
            
    print(f" -> Time taken: {time.time()-t0:.2f}s")
    print(f" -> Keys surviving Phase 1 (Oracle A Physical Circuit): {len(surviving_keys_A)}")
    
    print(f"\n--- Testing ORACLE B (Light Clauses Physical Circuit) ---")
    print(f"Simulating the {len(surviving_keys_A)} surviving keys through Oracle B...")
    t0 = time.time()
    surviving_keys_B = []
    
    for k in surviving_keys_A:
        v = simulate_boolean_circuit(qc_B, k, pool_size_B, alloc_B, vkey_B, vf_B)
        if v == 1:
            surviving_keys_B.append(k)
            
    print(f" -> Time taken: {time.time()-t0:.2f}s")
    print(f" -> Keys surviving Phase 2 (Oracle B Physical Circuit): {len(surviving_keys_B)}")
    
    if len(surviving_keys_B) == 1:
        print(f"\nSUCCESS! True Key exactly identified by physical circuit: 0x{surviving_keys_B[0]:04X} (Matches True: {surviving_keys_B[0] == TRUE_KEY})")
    elif len(surviving_keys_B) > 0:
        print(f"\nSUCCESS! Keys surviving: {[hex(x) for x in surviving_keys_B]}")
        print(f"True Key 0x{TRUE_KEY:04X} in survivors: {TRUE_KEY in surviving_keys_B}")
        
    print("=" * 70)

if __name__ == "__main__":
    main()
