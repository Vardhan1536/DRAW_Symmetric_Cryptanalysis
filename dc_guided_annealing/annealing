# Differential Cryptanalysis Guided Simulated Annealing

This directory contains the code implementation for the Differential-Informed Simulated Annealing framework described in the paper. 

## File Overview

- **`hybrid_qubo_compiler.py`**: The SAT-to-QUBO compilation . It transforms the DRAW cryptographic logic into a Hybrid Bounded Penalty QUBO model.
- **`dc_annealing.py`**: The primary execution script. It applies classical Differential Cryptanalysis (DC) to the cipher's MixColumns operation to compute log-odds ratio biases for the key bits. It then injects these biases as linear coefficients into the QUBO graph and executes a Two-Phase Annealing Strategy (Phase 1: DC-Biased Guided Descent, Phase 2: Pure QUBO Refinement) to recover the master key.

## How to Run

The main script automatically generates a random plaintext/ciphertext pair and a hidden key, compiles the QUBO, applies the DC-biasing, and executes the simulated annealing solver.

Run it directly from the command line:

```bash
python dc_annealing.py
```
