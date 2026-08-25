# Two-Phase Grover's Algorithm

This directory contains the core implementation of our Two-Phase Grover's cryptanalysis approach for S-AES. It compiles the classical cryptography into weighted SAT clauses and splits the quantum search into two distinct phases (heavy constraints and light constraints).

## File Overview

- **`two_phase_grovers.py`**: This the main script. It dynamically performs classical truth-table verification over the entire key space (65,536 keys) to compute exact surviving keys ($M_1, M_2$) and the optimal iteration counts ($m_1, m_2$). *(Note: We explicitly bypass the Qiskit Aer simulator because executing a 51 qubit circuit would exceed memory limits and crash. Instead, this script uses mathematically exact classical Boolean validation to scale the evaluation.)*
- **`gate_depth_analysis_new.py`**: This file computes and prints the circuit depths and gate counts for the four threshold strategies shown in our paper. **The script explicitly constructs the quantum circuits and uses Qiskit's compiler (`transpile` with `optimization_level=2`) to extract the exact logical depth and gate counts.** The results yielded by this script represent the resource requirements for a circuit with a **forward pass only** (ignoring uncomputation depths).
- **`Oracle_A.py` / `Oracle_B.py`**: Constructs the physical Qiskit quantum circuits of Oracles for Phase 1 (heavy constraints) and Phase 2 (light constraints).
- **`utils.py`**: Shared utility logic for fast exact-SAT classical evaluations.
- **`Reversible_Circuit/`**: Contains the quantum reversible circuit components for S-AES (like the IO-based S-Box and MixColumns) and hardware execution scripts.

## How to Run

**Parameters:**
To accurately reproduce the numerical results exactly as they appear in the paper, ensure the following problem constants are configured at the top of the scripts:
- **Plaintext (`PT`)**: `0x17FD`
- **Ciphertext (`CT`)**: `0xA55B`
- **True Key**: `0x3ABC`
To run the unified exact-SAT evaluator and physical Qiskit compiler:
```bash
python two_phase_grovers.py
```

To strictly run the theoretical analytical gate-depth computation:
```bash
python gate_depth_analysis_new.py
```
