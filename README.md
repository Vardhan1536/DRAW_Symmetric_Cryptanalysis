# Quantum Cryptanalysis of Simplified AES (S-AES)


> This repository presents a unified quantum cryptanalysis pipeline for Simplified AES (S-AES) using three different strategies. The project encodes the full S-AES encryption function (key schedule, SubBytes, ShiftRows, MixColumns, AddRoundKey) as a Boolean SAT formula using Massacci–Marraro minimized prime implicants, and introduces a DRAW (reachability) weighting scheme that assigns each SAT clause a weight proportional to the number of ciphertext bits it influences. 

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Formulation](#mathematical-formulation)
  - [SAT Formulation and DRAW Weights](#sat-formulation-and-draw-weights)
- [Key Recovery Approaches](#key-recovery-approaches)
  - [Two-Phase Grover's Algorithm](#two-phase-grovers-algorithm)
  - [VQA](#vqa)
  - [DC-Guided Annealing](#dc-guided-annealing)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

---

## Overview

This repository studies the key-recovery problem on Simplified AES (S-AES) through four independent but interconnected attack strategies:

| Module | Strategy | Core Method |
|---|---|---|
| `SAT_formulation/` | Classical SAT cryptanalysis | Massacci-Marraro CNF encoding + DRAW reachability weighting |
| `VQA/` | Variational Quantum Algorithm | draw ansatz optimized with CMA-ES on Qiskit Aer |
| `dc_guided_annealing/` | Hybrid quantum-classical annealing | Differential Cryptanalysis (DC) biased QUBO + neal SA |
| `two_phase_grovers/` | Quantum Amplitude Amplification | Two-phase QAA with DRAW-split oracles (Oracle A / Oracle B) |

All four modules share a common S-AES implementation (`SAT_formulation/saes.py`) and the DRAW clause weighting system (`SAT_formulation/draw_weights.py`).

---

## Mathematical Formulation

### SAT Formulation and DRAW Weights

S-AES key recovery is cast as a Boolean Satisfiability (SAT) problem. Each cipher operation — SubBytes (S-Box), ShiftRows, AddRoundKey, and MixColumns — is translated into a set of CNF clauses over Boolean variables representing every bit in the encryption datapath. A candidate key `K` is correct if and only if all clauses are simultaneously satisfied.

The standard SAT formulation penalises every violated clause equally (weight = 1), which flattens the search landscape and hides the true cryptographic structure. We introduce DRAW (Directed Reachability Avalanche Weighting) to fix this: each clause receives a weight equal to the number of ciphertext output bits it can reach through the cipher's data-flow graph.

```
w(clause_i) = |{ ciphertext bits reachable from vars(clause_i) }|
```

#### How each cipher component is encoded

| Component | Encoding method | Notes |
|---|---|---|
| SubBytes (S-Box) | Massacci–Marraro minimized prime implicants [[1]](#references) | 23 clauses per 4-bit S-Box, lengths 3–5 |
| AddRoundKey / XOR | Exhaustive 2^k parity clauses | Exact linear encoding |
| ShiftRows | Variable re-indexing only | No new clauses needed |
| Key Schedule | XOR + S-Box clauses | Highest DRAW weight (avg 14.25, median 16.0) — [see detailed encoding →](SAT_formulation/ENCODING.md#key-schedule-encoding) |
| MixColumns | GF(2⁴) mult-by-4 encoded as XOR clauses | Triggers the weight jump: Round 1 max leaps to 16.0 vs Round 2 max of 1.0 — [see detailed encoding →](SAT_formulation/ENCODING.md#mixcolumns-encoding) |
| Boundary (PT/CT) | Unit clauses locking observed bits | Weight 1.0 — isolated, no diffusion |

---

## Key Recovery Approaches

We implement and evaluate search and optimization paradigms to explore the constraint landscape and recover the S-AES master key.

## Two-Phase Grover's Algorithm (`two_phase_grovers/`)

This module implements a Two-Phase Grover's search approach that compiles the classical cryptography into weighted SAT clauses (WCNF). It splits the physical quantum search into heavy and light constraints to optimize hardware resources.

- **`two_phase_grovers/two_phase_grovers.py`**: The unified script that dynamically calculates surviving keys and optimal iteration counts via mathematically exact classical Boolean validation. *(Note: We explicitly bypass the Qiskit Aer simulator because executing a 51 qubit uncompressed oracle would exceed memory limits and crash. Instead, this script uses mathematically exact classical Boolean validation to scale the evaluation.)*
- **`two_phase_grovers/gate_depth_analysis_new.py`**: A theoretical analysis script that explicitly constructs physical quantum circuits and uses Qiskit's compiler to extract the logical circuit depths and gate counts for forward pass only.
- **`two_phase_grovers/Reversible_Circuit/`**: Contains the quantum reversible circuit components for S-AES (like the IO-based S-Box and MixColumns) and hardware execution scripts.


### VQA
This approach runs a Variational Quantum Algorithm using Qiskit Aer to find the secret master key. It builds a diagonal cost Hamiltonian from the DRAW-weighted clause landscape and optimizes a parameterized ansatz circuit using CMA-ES.

- **`VQA/landscape_geometry_experiments.py`**: Computes the classical influence and pairwise synergy metrics required to construct the $U_{\text{mix}}$, $U_{\text{bias}}$, and $U_{\text{syn}}$ ansatz operators.
- **`VQA/draw_vqa.py`**: The primary execution script. It constructs the full 20-layer ansatz in Qiskit (including the Fast Walsh-Hadamard transform for $U_{\text{cost}}$) and executes the CMA-ES optimization.
- **`VQA/vqa_results.csv`**: A dataset logging the execution metrics of various VQA experimental runs.


### DC-Guided Annealing

This approach compiles the S-AES SAT constraints into a QUBO and runs simulated annealing using D-Wave's neal sampler. It utilizes prior key bit probability distributions derived from classical Differential Cryptanalysis (DC) to bias the QUBO's linear terms, guiding the solver toward the correct key.

- **`dc_guided_annealing/hybrid_qubo_compiler.py`**: Does the SAT-to-QUBO compilation. It transforms the DRAW logic into a Hybrid Bounded-Penalty QUBO model.
- **`dc_guided_annealing/dc_annealing.py`**: The execution script. It computes DC biases from the cipher's MixColumns operation and executes a Two-Phase Annealing Strategy (Phase 1: DC-Biased Guided Descent, Phase 2: Pure QUBO Refinement) to recover the master key.
---

## Installation

Python 3.10+ is required.

Install the necessary libraries to compile the SAT formulations and run the optimization and annealing pipelines:
```bash
pip install numpy scipy pysat dimod dwave-neal qiskit qiskit-aer cma matplotlib
```
---

## Usage

### 1. Key Recovery Approaches
The simulated annealing search script can be executed directly, while the VQA script requires plaintext and key parameters to run:

```bash

# Run the Two-Phase Grover's Algorithm
python two_phase_grovers/two_phase_grovers.py

# Run this to get the resource costs for the hardware implementation of two phase grovers
python two_phase_grovers/gate_depth_analysis_new.py

# Run the Variational Quantum Algorithm cost minimization
python VQA/draw_vqa.py --pt 0xFFFF --key 0xA73B

# Run simulated annealing guided by differential cryptanalysis
python dc_guided_annealing/dc_annealing.py
```

### 2. SAT Formulation & Landscape Analysis (Command Line Arguments)
The core SAT compiler and weight analysis utilities require you to supply observed plaintext and ciphertext pairs as alternating command line arguments:

```bash
# Compile S-AES constraints to DIMACS CNF format
python SAT_formulation/sat.py <plaintext> <ciphertext>
# Example:
python SAT_formulation/sat.py 0x6F6B 0x0738

# Calculate DRAW weights and export a WCNF file
python SAT_formulation/draw_weights.py <plaintext> <ciphertext> [-o output_file.wcnf]
# Example:
python SAT_formulation/draw_weights.py 0x6F6B 0x0738 -o saes_draw.wcnf

# Analyze fitness-distance correlation and landscape metrics
python SAT_formulation/landscape_metrics.py <plaintext> <ciphertext> [-k true_key]
# Example:
python SAT_formulation/landscape_metrics.py 0x6F6B 0x0738 -k 0xA73B
```

---

---

## References

**[1]** F. Massacci and L. Marraro (2000). *Logical Cryptanalysis as a SAT Problem: Encoding and Analysis of DES.* Journal of Automated Reasoning, 24(1–2), 165–203. [DOI:10.1023/A:1006326723002](https://doi.org/10.1023/A:1006326723002)

**[2]** T. Jones and S. Forrest (1995). *Fitness Distance Correlation as a Measure of Problem Difficulty for Genetic Algorithms.* Proceedings of ICGA.

**[3]** K. Malan and A. Engelbrecht (2013). *A Survey of Techniques for Characterising Fitness Landscapes and Some Possible Ways Forward.* Information Sciences, 241, 148–163.

**[4]** N. Courtois and J. Pieprzyk (2002). *Cryptanalysis of Block Ciphers with Overdefined Systems of Equations.* ASIACRYPT 2002. [arXiv:cs/0210044](https://arxiv.org/abs/cs/0210044)

**[5]** E. Farhi, J. Goldstone, and S. Gutmann (2014). *A Quantum Approximate Optimization Algorithm.* [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)

**[6]** L. K. Grover (1996). *A Fast Quantum Mechanical Algorithm for Database Search.* Proceedings of STOC 1996. [DOI:10.1145/237814.237866](https://doi.org/10.1145/237814.237866)

**[7]** G. Brassard, P. Høyer, M. Mosca, and A. Tapp (2002). *Quantum Amplitude Amplification and Estimation.* AMS Contemporary Mathematics, 305. [arXiv:quant-ph/0005055](https://arxiv.org/abs/quant-ph/0005055)
