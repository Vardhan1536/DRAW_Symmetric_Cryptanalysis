# Quantum-Classical Cryptanalysis of Simplified AES (S-AES)

> This repository presents a unified quantum-classical cryptanalysis pipeline targeting Simplified AES (S-AES) — a 16-bit block cipher with a 16-bit master key space of 65,536 candidates. The project encodes the full S-AES encryption function (key schedule, SubBytes, ShiftRows, MixColumns, AddRoundKey) as a Boolean SAT formula using Massacci–Marraro minimized prime implicants, and introduces a DRAW (reachability) weighting scheme that assigns each SAT clause a weight proportional to the number of ciphertext bits it influences. 
---

## Table of Contents

- [Overview](#overview)
- [Mathematical Formulation](#mathematical-formulation)
  - [SAT Formulation and DRAW Weights](#sat-formulation-and-draw-weights)
- [Key Recovery Approaches](#key-recovery-approaches)
  - [VQA — CWMC-QOC](#vqa--cwmc-qoc)
  - [DC-Guided Annealing](#dc-guided-annealing)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [References](#references)

---

## Overview

This repository studies the **key-recovery problem** on **Simplified AES (S-AES)** — a pedagogical 16-bit block cipher with a 16-bit master key — through four independent but interconnected attack strategies:

| Module | Strategy | Core Method |
|---|---|---|
| `SAT_formulation/` | Classical SAT cryptanalysis | Massacci-Marraro CNF encoding + DRAW reachability weighting |
| `VQA/` | Variational Quantum Algorithm | CWMC-QOC ansatz optimized with CMA-ES on Qiskit Aer |
| `dc_guided_annealing/` | Hybrid quantum-classical annealing | Differential Cryptanalysis (DC) biased QUBO + neal SA |
| `two_phase_grovers/` | Quantum Amplitude Amplification | Two-phase QAA with DRAW-split oracles (Oracle A / Oracle B) |

All four modules share a common S-AES implementation (`SAT_formulation/saes.py`) and the DRAW clause weighting system (`SAT_formulation/draw_weights.py`).

---


## Mathematical Formulation

### SAT Formulation and DRAW Weights

S-AES key recovery is cast as a **Boolean Satisfiability (SAT)** problem. Each cipher operation — SubBytes (S-Box), ShiftRows, AddRoundKey, and MixColumns — is translated into a set of CNF clauses over Boolean variables representing every bit in the encryption datapath. A candidate key `K` is correct if and only if **all** clauses are simultaneously satisfied.

The standard SAT formulation penalises every violated clause equally (weight = 1), which flattens the search landscape and hides the true cryptographic structure. We introduce **DRAW (Directed Reachability Avalanche Weighting)** to fix this: each clause receives a weight equal to the number of ciphertext output bits it can reach through the cipher's data-flow graph.

```
w(clause_i) = |{ ciphertext bits reachable from vars(clause_i) }|
```

#### How each cipher component is encoded

| Component | Encoding method | Notes |
|---|---|---|
| SubBytes (S-Box) | Massacci–Marraro minimized prime implicants [[1]](#references) | 23 clauses per 4-bit S-Box, lengths 3–5 |
| AddRoundKey / XOR | Exhaustive 2^k parity clauses | Exact linear encoding |
| ShiftRows | Variable re-indexing only | No new clauses needed |
| **Key Schedule** | XOR + S-Box clauses | Highest DRAW weight (avg 14.25, median **16.0**) — [see detailed encoding →](SAT_formulation/Encoding.md#1-key-schedule-cnf-encoding) |
| **MixColumns** | GF(2⁴) mult-by-4 encoded as XOR clauses | Triggers the weight jump: Round 1 max leaps to **16.0** vs Round 2 max of **1.0** — [see detailed encoding →](SAT_formulation/Encoding.md#2-mixcolumns-cnf-encoding) |
| Boundary (PT/CT) | Unit clauses locking observed bits | Weight 1.0 — isolated, no diffusion |

---

## Key Recovery Approaches

We implement and evaluate three search and optimization paradigms to explore the constraint landscape and recover the S-AES master key.

### VQA — CWMC-QOC
This approach maps the DRAW-weighted constraint landscape to a diagonal cost Hamiltonian. Rather than using generic variational ansätze, we employ a physics-informed **Quantum Optimal Control (QOC)** circuit tailored to the cipher's structure. The ansatz incorporates:
- **Weighted Mixers**: Qubit rotation rates scaled by each bit's key schedule loading.
- **Influence & Synergy Blocks**: Unitary rotations encoding single-bit influence and pairwise bit correlations.
- **Cost Driver**: Diagonal phase evolution representing the cost landscape.

The circuit parameters are optimized classically via **CMA-ES** to maximize the probability of measuring the correct secret key.

### DC-Guided Annealing
This approach combines classical Differential Cryptanalysis (DC) with simulated annealing. By analyzing the difference distribution table (DDT) of the S-AES S-box, we estimate the likelihood of specific key bits from chosen-plaintext pairs. This statistical bias is injected as linear terms into the QUBO/BQM cost function, guiding the annealing sampler toward the correct solution:
- **Strategy 1 (Two-Phase)**: Uses the DC bias to quickly find the general low-energy valley, then switches to the pure cryptographic cost landscape to isolate the exact key.
- **Strategy 2 (Iterative Refinement)**: Iteratively updates the search biases based on consistency checks between rounds.

---

## Installation

**Python 3.10+** is required.

Install the necessary libraries to compile the SAT formulations and run the optimization and annealing pipelines:
```bash
pip install numpy scipy pysat dimod dwave-neal qiskit qiskit-aer cma matplotlib
```

---

## Usage

### 1. Key Recovery Approaches (Direct Execution)
The VQA and simulated annealing search scripts can be executed directly:

```bash
# Run the Variational Quantum Algorithm cost minimization
python VQA/cwmc_qoc_vqa.py

# Run simulated annealing guided by differential cryptanalysis
python dc_guided_annealing/dc_annealing.py
```

### 2. SAT Formulation & Landscape Analysis (Command Line Arguments)
The core SAT compiler and weight analysis utilities require you to supply observed **plaintext** and **ciphertext** pairs as alternating command line arguments:

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

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{saes_quantum_cryptanalysis,
  author    = {Your Name},
  title     = {Quantum-Classical Cryptanalysis of Simplified AES (S-AES)},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/<your-username>/<your-repo>}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## References

**[1]** F. Massacci and L. Marraro (2000). *Logical Cryptanalysis as a SAT Problem: Encoding and Analysis of DES.* Journal of Automated Reasoning, 24(1–2), 165–203. [DOI:10.1023/A:1006326723002](https://doi.org/10.1023/A:1006326723002)

**[2]** T. Jones and S. Forrest (1995). *Fitness Distance Correlation as a Measure of Problem Difficulty for Genetic Algorithms.* Proceedings of ICGA.

**[3]** K. Malan and A. Engelbrecht (2013). *A Survey of Techniques for Characterising Fitness Landscapes and Some Possible Ways Forward.* Information Sciences, 241, 148–163.

**[4]** N. Courtois and J. Pieprzyk (2002). *Cryptanalysis of Block Ciphers with Overdefined Systems of Equations.* ASIACRYPT 2002. [arXiv:cs/0210044](https://arxiv.org/abs/cs/0210044)

**[5]** E. Farhi, J. Goldstone, and S. Gutmann (2014). *A Quantum Approximate Optimization Algorithm.* [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)

**[6]** L. K. Grover (1996). *A Fast Quantum Mechanical Algorithm for Database Search.* Proceedings of STOC 1996. [DOI:10.1145/237814.237866](https://doi.org/10.1145/237814.237866)

**[7]** G. Brassard, P. Høyer, M. Mosca, and A. Tapp (2002). *Quantum Amplitude Amplification and Estimation.* AMS Contemporary Mathematics, 305. [arXiv:quant-ph/0005055](https://arxiv.org/abs/quant-ph/0005055)
