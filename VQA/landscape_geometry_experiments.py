"""
landscape_geometry_experiments.py
=================================
Measures the geometry of the CWMC landscape to inform a CWMC-native optimizer.

Experiment 1: Bit Influence Spectrum
  I_i = E_K[ |E(K) - E(K ^ e_i)| ]

Experiment 2: Pairwise Influence Matrix (Synergy)
  I(i,j) = E_K[ |E(K) - E(K ^ e_i ^ e_j)| ]
  S(i,j) = I(i,j) - I_i - I_j

Experiment 3: Quantum Annealing Trajectory
  Tracks P(bit_i = 1) over time t during Schrodinger evolution.
"""

import numpy as np
import time
import sys
import os
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SAT_formulation')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dc_guided_annealing')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'two_phase_grovers')))
from sat import SAESMassacciCompiler as ExtendedCompiler
from draw_weights import build_draw_weights as build_avalanche_weights
from utils import generate_assignment

# ─── Constants ────────────────────────────────────────────────────────────────
K_OPT   = 0xA73B
N       = 16
N_KEYS  = 1 << N
PAIRS_1 = [(0xE445, 0x2877)]

# ─── Setup ────────────────────────────────────────────────────────────────────
def build_landscape(pt=None, ct=None):
    print("Building S-AES CWMC landscape (takes ~2 mins)...")
    t0 = time.time()
    c = ExtendedCompiler()
    if pt is not None and ct is not None:
        c.add_plaintext_ciphertext_pair(pt, ct)
        pt_val, ct_val = pt, ct
    else:
        for p, c_val in PAIRS_1:
            c.add_plaintext_ciphertext_pair(p, c_val)
        pt_val, ct_val = PAIRS_1[0]
    w4 = build_avalanche_weights(c)

    E = np.zeros(N_KEYS)
    for k in range(N_KEYS):
        asgn = generate_assignment(pt_val, ct_val, k)
        for i, clause in enumerate(c.clauses):
            satisfied = False
            for lit in clause:
                v = abs(lit)
                if v not in asgn:
                    continue
                val = asgn[v]
                if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                    satisfied = True
                    break
            if not satisfied:
                E[k] += w4[i]
    E -= E.min()

    # WCNF Mobility Weights
    kcc = np.zeros(N, dtype=np.float64)
    for i, clause in enumerate(c.clauses):
        for lit in clause:
            var = abs(lit)
            if 1 <= var <= N:
                kcc[var - 1] += w4[i]
    W_inv = 1.0 / (kcc + 1e-9)
    W = W_inv / np.mean(W_inv)
    
    print(f"Landscape built in {time.time()-t0:.1f}s.")
    return E, W

# ─── Experiment 1: Bit Influence ─────────────────────────────────────────────
def exp1_bit_influence(E):
    print("\n--- Experiment 1: Bit Influence Spectrum ---")
    I = np.zeros(N)
    for i in range(N):
        mask = 1 << i
        # Calculate |E(K) - E(K ^ mask)| for all K
        # We can do this efficiently via array slicing/indexing
        # K ranges from 0 to 65535
        keys = np.arange(N_KEYS)
        flipped_keys = keys ^ mask
        diffs = np.abs(E[keys] - E[flipped_keys])
        I[i] = np.mean(diffs)
        print(f"  Bit {i:2d} : {I[i]:.4f}")
    return I

# ─── Experiment 2: Pairwise Synergy ──────────────────────────────────────────
def exp2_pairwise_synergy(E, I):
    print("\n--- Experiment 2: Pairwise Influence Matrix (Synergy) ---")
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            mask = (1 << i) | (1 << j)
            keys = np.arange(N_KEYS)
            flipped_keys = keys ^ mask
            I_ij = np.mean(np.abs(E[keys] - E[flipped_keys]))
            S_ij = I_ij - I[i] - I[j]
            S[i, j] = S_ij
            S[j, i] = S_ij
            
    print("Synergy Matrix S(i,j):")
    # Print formatted matrix
    header = "      " + "".join([f"{j:4d}" for j in range(N)])
    print(header)
    for i in range(N):
        row_str = f"{i:4d}  " + "".join([f"{S[i,j]:4.1f}" for j in range(N)])
        print(row_str)
    
    return S

# ─── Experiment 3: Annealing Trajectory ──────────────────────────────────────
def exp3_annealing_trajectory(E, W):
    print("\n--- Experiment 3: Quantum Annealing Trajectory ---")
    T = 100.0
    steps = 1000
    dt = T / steps
    
    psi = np.ones(N_KEYS, dtype=np.complex128) / np.sqrt(N_KEYS)
    all_idx = np.arange(N_KEYS)
    
    # Store probabilities
    # We want P(bit_i = True Key Bit i) for consistency, but the prompt asked for P(bit_i = 1)
    # Let's track P(bit_i = 1) as requested.
    history_t = []
    history_P = {i: [] for i in range(N)}
    
    print("Simulating Schrodinger evolution...")
    
    # Precompute masks for marginal extraction
    masks = [1 << i for i in range(N)]
    
    for step in range(steps + 1):
        if step > 0:
            t = (step - 0.5) * dt
            s = t / T
            # Phase
            psi *= np.exp(-1.0j * dt * s * E)
            # Mixing
            for i in range(N):
                ti = dt * (1.0 - s) * W[i]
                cv = np.cos(ti); sv = 1.0j * np.sin(ti)
                idx0 = np.where((all_idx & masks[i]) == 0)[0]
                idx1 = idx0 | masks[i]
                p0, p1 = psi[idx0].copy(), psi[idx1].copy()
                psi[idx0] = cv * p0 + sv * p1
                psi[idx1] = cv * p1 + sv * p0
                
        # Record at 100 checkpoints
        if step % (steps // 100) == 0:
            current_t = step * dt
            probs = np.abs(psi) ** 2
            
            history_t.append(current_t)
            for i in range(N):
                # P(bit_i = 1)
                P_1 = np.sum(probs[np.where((all_idx & masks[i]) != 0)[0]])
                history_P[i].append(P_1)
                
            if current_t % 20 == 0:
                print(f"  t = {current_t:5.1f} | P(K*) = {probs[K_OPT]:.4f}")

    if plt is None:
        print("matplotlib not installed, skipping plot.")
        return
        
    # Plot
    plt.figure(figsize=(12, 8))
    for i in range(N):
        # We can color them based on whether the true bit is 1 or 0
        true_bit = (K_OPT >> i) & 1
        color = 'blue' if true_bit == 1 else 'red'
        linestyle = '-' if true_bit == 1 else '--'
        plt.plot(history_t, history_P[i], label=f'Bit {i} (True={true_bit})', color=color, alpha=0.6, linestyle=linestyle)
        
    plt.axhline(y=1.0, color='gray', linestyle=':')
    plt.axhline(y=0.0, color='gray', linestyle=':')
    plt.title('Experiment 3: Bit Marginals P(bit=1) over Quantum Annealing Trajectory')
    plt.xlabel('Time (t)')
    plt.ylabel('P(bit_i = 1)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('cwmc_annealing_trajectory_bits.png', dpi=300)
    print("Saved plot to 'cwmc_annealing_trajectory_bits.png'")

def main():
    E, W = build_landscape()
    I = exp1_bit_influence(E)
    S = exp2_pairwise_synergy(E, I)
    exp3_annealing_trajectory(E, W)

if __name__ == "__main__":
    main()
