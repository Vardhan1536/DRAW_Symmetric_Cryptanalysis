# DRAW-Symmetric Cryptanalysis Framework

This repository contains the official implementation of the DRAW framework applied to the S-AES cryptographic cipher. The codebase explores multiple quantum and hybrid-quantum cryptanalysis strategies, organized into three distinct sub-projects.

---

## 1. Two-Phase Grover's Algorithm (`two_phase_grovers/`)

This module implements a Two-Phase Grover's search approach that compiles the classical cryptography into weighted SAT clauses (WCNF). It splits the physical quantum search into heavy and light constraints to optimize hardware resources.

- **`two_phase_grovers/two_phase_grovers.py`**: The unified script that dynamically calculates surviving keys and optimal iteration counts via mathematically exact classical Boolean validation. *(Note: We explicitly bypass the Qiskit Aer simulator because executing a 51+ qubit uncompressed oracle would exceed memory limits and crash. Instead, this script uses mathematically exact classical Boolean validation to scale the evaluation.)*
- **`two_phase_grovers/gate_depth_analysis_new.py`**: A theoretical analysis script that explicitly constructs physical quantum circuits and uses Qiskit's compiler to extract the logical circuit depths and gate counts for forward pass only.
- **`two_phase_grovers/Reversible_Circuit/`**: Contains the quantum reversible circuit components for S-AES (like the IO-based S-Box and MixColumns) and hardware execution scripts.

**How to Run:**
```bash
python two_phase_grovers/two_phase_grovers.py
python two_phase_grovers/gate_depth_analysis_new.py
```
*(Ensure `PT = 0x17FD`, `CT = 0xA55B`, `True Key = 0x3ABC` are configured in the scripts to reproduce exact paper results).*

---

## 2. Differential Cryptanalysis Guided Simulated Annealing (`dc_guided_annealing/`)

This module implements a Differential-Informed Simulated Annealing framework. It uses classical Differential Cryptanalysis (DC) to compute log-odds ratio biases, which are injected into a Hybrid Bounded-Penalty QUBO model.

- **`dc_guided_annealing/hybrid_qubo_compiler.py`**: Does the SAT-to-QUBO compilation. It transforms the DRAW logic into a Hybrid Bounded-Penalty QUBO model.
- **`dc_guided_annealing/dc_annealing.py`**: The execution script. It computes DC biases from the cipher's MixColumns operation and executes a Two-Phase Annealing Strategy (Phase 1: DC-Biased Guided Descent, Phase 2: Pure QUBO Refinement) to recover the master key.

**How to Run:**
```bash
python dc_guided_annealing/dc_annealing.py
```

---

## 3. Variational Quantum Algorithms (VQA) 

This module implements the DRAW-VQA framework, a physics-informed ansatz optimization strategy to navigate the DRAW energy landscape using a 3-stage Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

- **`VQA/landscape_geometry_experiments.py`**: Computes the classical influence and pairwise synergy metrics required to construct the $U_{\text{mix}}$, $U_{\text{bias}}$, and $U_{\text{syn}}$ ansatz operators.
- **`VQA/draw_vqa.py`**: The primary execution script. It constructs the full 20-layer ansatz in Qiskit (including the Fast Walsh-Hadamard transform for $U_{\text{cost}}$) and executes the CMA-ES optimization.
- **`VQA/vqa_results.csv`**: A dataset logging the execution metrics of various VQA experimental runs.

**How to Run:**
```bash
python VQA/draw_vqa.py --pt 0xE445 --key 0x3AB6
```

---

## Additional Information

- **[`results.md`](./results.md)**: Contains detailed hardware execution fidelity metrics and physical resource requirements evaluated on IBM superconducting hardware.
- **[`SAT_formulation/SAT_Encoding_Derivation.md`](./SAT_formulation/SAT_Encoding_Derivation.md)**: Contains the full mathematical formulation and clause counts for the Key Schedule and MixColumns constraints.
