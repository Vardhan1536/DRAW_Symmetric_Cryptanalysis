# DRAW-VQA Framework

This directory contains the code implementation of the DRAW-VQA framework (Variational Quantum Algorithm) described in the paper. 

## File Overview

- **`landscape_geometry_experiments.py`**: Computes the classical influence $I(k_i)$ and pairwise synergy $s_{uv}$ metrics required to construct the $U_{\text{mix}}$, $U_{\text{bias}}$, and $U_{\text{syn}}$ ansatz operators.
- **`draw_vqa.py`**: The primary execution script. It constructs the full 20-layer ansatz in Qiskit (including the Fast Walsh-Hadamard transform for $U_{\text{cost}}$) and executes the 3-stage CMA-ES optimization to find the optimal cryptographic key.

## How to Run

You can execute the VQA solver directly from the command line, providing a custom Plaintext (`--pt`) and Target Key (`--key`) in hexadecimal format:

```bash
python draw_vqa.py --pt 0xE445 --key 0x3AB6
```
