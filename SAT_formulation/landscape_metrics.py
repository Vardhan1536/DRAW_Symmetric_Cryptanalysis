import os
import sys
import math
import time
import argparse
import numpy as np
from scipy.stats import pearsonr
import multiprocessing as mp
from typing import List, Dict, Tuple, Set

from sat import SAESMassacciCompiler, parse_int_or_hex, D_BOUND, D_FINAL
from draw_weights import build_draw_weights
from saes import encrypt

def solve_assignment(compiler: SAESMassacciCompiler, key_int: int) -> Dict[int, int]:
    asgn: Dict[int, int] = {}

    for i, v in enumerate(compiler.master_key):
        asgn[v] = (key_int >> (15 - i)) & 1

    for idx, cl in enumerate(compiler.clauses):
        if compiler.clause_tags[idx] == D_BOUND:
            lit = cl[0]
            asgn[abs(lit)] = 1 if lit > 0 else 0

    var_idx: Dict[int, List[int]] = {}
    for ci, cl in enumerate(compiler.clauses):
        for lit in cl:
            var_idx.setdefault(abs(lit), []).append(ci)

    def propagate() -> bool:
        changed = False
        for cl in compiler.clauses:
            unset, falsified, satisfied = [], 0, False
            for lit in cl:
                v = abs(lit)
                if v in asgn:
                    val = asgn[v]
                    if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                        satisfied = True
                        break
                    else:
                        falsified += 1
                else:
                    unset.append(lit)
            if satisfied:
                continue
            if len(unset) == 1 and falsified == len(cl) - 1:
                lit = unset[0]
                v = abs(lit)
                if v not in asgn:
                    asgn[v] = 1 if lit > 0 else 0
                    changed = True
        return changed

    while propagate():
        pass

    all_vars = set(abs(l) for cl in compiler.clauses for l in cl)
    for var in sorted(all_vars - set(asgn.keys())):
        best_val, best_pen = 0, float('inf')
        for val in (0, 1):
            asgn[var] = val
            pen = sum(
                1 for ci in var_idx.get(var, [])
                if not any((lit > 0 and asgn.get(abs(lit)) == 1) or (lit < 0 and asgn.get(abs(lit)) == 0) for lit in compiler.clauses[ci])
            )
            asgn.pop(var)
            if pen < best_pen:
                best_pen, best_val = pen, val
        asgn[var] = best_val

    return asgn

def evaluate_key(compiler: SAESMassacciCompiler, draw_w: List[float], key_int: int) -> Tuple[float, float]:
    asgn = solve_assignment(compiler, key_int)
    e_sat = 0.0
    e_draw = 0.0
    for idx, cl in enumerate(compiler.clauses):
        sat = False
        for lit in cl:
            v = abs(lit)
            val = asgn.get(v, 0)
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                sat = True
                break
        if not sat:
            e_sat += 1.0
            e_draw += draw_w[idx]
    return e_sat, e_draw

def _worker_eval_chunk(args):
    keys_chunk, pairs = args
    compiler = SAESMassacciCompiler()
    for pt, ct in pairs:
        compiler.add_plaintext_ciphertext_pair(pt, ct)
    draw_w = np.array(build_draw_weights(compiler), dtype=np.float64)

    # Pre-index variables and clauses once per worker
    var_idx: Dict[int, List[int]] = {}
    for ci, cl in enumerate(compiler.clauses):
        for lit in cl:
            var_idx.setdefault(abs(lit), []).append(ci)

    all_vars_sorted = sorted(set(abs(l) for cl in compiler.clauses for l in cl))
    bound_assignments = {}
    for idx, cl in enumerate(compiler.clauses):
        if compiler.clause_tags[idx] == D_BOUND:
            lit = cl[0]
            bound_assignments[abs(lit)] = 1 if lit > 0 else 0

    master_key_vars = list(compiler.master_key)
    clauses = compiler.clauses

    results = []
    for k in keys_chunk:
        asgn = dict(bound_assignments)
        for i, v in enumerate(master_key_vars):
            asgn[v] = (k >> (15 - i)) & 1

        def propagate():
            changed = False
            for cl in clauses:
                unset, falsified, satisfied = [], 0, False
                for lit in cl:
                    v = abs(lit)
                    if v in asgn:
                        val = asgn[v]
                        if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                            satisfied = True
                            break
                        else:
                            falsified += 1
                    else:
                        unset.append(lit)
                if satisfied:
                    continue
                if len(unset) == 1 and falsified == len(cl) - 1:
                    lit = unset[0]
                    v = abs(lit)
                    if v not in asgn:
                        asgn[v] = 1 if lit > 0 else 0
                        changed = True
            return changed

        while propagate():
            pass

        for var in all_vars_sorted:
            if var not in asgn:
                best_val, best_pen = 0, float('inf')
                for val in (0, 1):
                    asgn[var] = val
                    pen = sum(
                        1 for ci in var_idx.get(var, [])
                        if not any((lit > 0 and asgn.get(abs(lit)) == 1) or (lit < 0 and asgn.get(abs(lit)) == 0) for lit in clauses[ci])
                    )
                    asgn.pop(var)
                    if pen < best_pen:
                        best_pen, best_val = pen, val
                asgn[var] = best_val

        e_sat = 0.0
        e_draw = 0.0
        for idx, cl in enumerate(clauses):
            sat = False
            for lit in cl:
                v = abs(lit)
                val = asgn.get(v, 0)
                if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                    sat = True
                    break
            if not sat:
                e_sat += 1.0
                e_draw += draw_w[idx]

        results.append((k, e_sat, e_draw))

    return results

def compute_landscape_metrics(pairs: List[Tuple[int, int]], true_key: int = 0xA73B):
    total_keys = 65536
    keys_to_eval = list(range(total_keys))

    n_cpu = min(mp.cpu_count(), 16)
    chunk_size = max(1, len(keys_to_eval) // n_cpu)
    chunks = [keys_to_eval[i:i + chunk_size] for i in range(0, len(keys_to_eval), chunk_size)]
    args_list = [(chk, pairs) for chk in chunks]

    print("=" * 78)
    print(f"S-AES LANDSCAPE EVALUATION: ALL 65,536 KEYS ({len(pairs)} PT-CT Block(s))")
    print("=" * 78)
    print(f"[*] Evaluating full Boolean key space across {n_cpu} CPU processes...")
    t0 = time.time()
    
    with mp.Pool(processes=n_cpu) as pool:
        chunk_results = pool.map(_worker_eval_chunk, args_list)

    elapsed = time.time() - t0
    print(f"[+] Evaluation finished in {elapsed:.2f} seconds.")

    all_results = [item for sub in chunk_results for item in sub]
    all_results.sort(key=lambda x: x[0])

    eval_keys = np.array([x[0] for x in all_results])
    E_sat = np.array([x[1] for x in all_results], dtype=np.float64)
    E_draw = np.array([x[2] for x in all_results], dtype=np.float64)

    # 1. Fitness-Distance Correlation (FDC)
    hd_array = np.array([bin(k ^ true_key).count('1') for k in eval_keys])
    fdc_sat, _ = pearsonr(E_sat, hd_array)
    fdc_draw, _ = pearsonr(E_draw, hd_array)

    # 2. Autocorrelation (HD=1 neighborhood smoothness)
    masks = [1 << i for i in range(16)]
    
    def calc_autocorr(E):
        E_mean = np.mean(E)
        E_var = np.var(E)
        if E_var == 0:
            return 1.0
        cov = 0.0
        for m in masks:
            E_shifted = E[eval_keys ^ m]
            cov += np.mean((E - E_mean) * (E_shifted - E_mean))
        return (cov / 16.0) / E_var

    ac_sat = calc_autocorr(E_sat)
    ac_draw = calc_autocorr(E_draw)

    # 3. Local Minima Percentage
    def calc_local_minima_pct(E):
        loc_count = 0
        for k in range(total_keys):
            e_k = E[k]
            if all(E[k ^ m] >= e_k for m in masks):
                loc_count += 1
        return (loc_count / total_keys) * 100.0

    loc_sat_pct = calc_local_minima_pct(E_sat)
    loc_draw_pct = calc_local_minima_pct(E_draw)

    # 4. Basin Size (Steepest Descent)
    def calc_basin_size(E, target_key):
        basin = 0
        for k in range(total_keys):
            curr_k = k
            visited = set()
            while True:
                visited.add(curr_k)
                e_curr = E[curr_k]
                best_neighbor = curr_k
                best_e = e_curr
                for m in masks:
                    nbr = curr_k ^ m
                    if E[nbr] < best_e:
                        best_e = E[nbr]
                        best_neighbor = nbr
                if best_neighbor == curr_k or best_neighbor in visited:
                    if best_neighbor == target_key:
                        basin += 1
                    break
                curr_k = best_neighbor
        return basin

    basin_sat = calc_basin_size(E_sat, true_key)
    basin_draw = calc_basin_size(E_draw, true_key)

    # Display comparison table with strictly the 4 metrics
    print("\n" + "=" * 78)
    print(f"LANDSCAPE EVALUATION METRICS COMPARISON TABLE ({len(pairs)} PT-CT Block(s))")
    print("=" * 78)
    print(f"{'Landscape Metric':<38} | {'Standard SAT':<16} | {'DRAW (Ours)':<16}")
    print("-" * 78)
    print(f"{'1. Fitness Distance Correlation (FDC)':<38} | {fdc_sat:<16.4f} | {fdc_draw:<16.4f}")
    print(f"{'2. Autocorrelation (HD=1 Smoothness)':<38} | {ac_sat:<16.4f} | {ac_draw:<16.4f}")
    print(f"{'3. Local Minima Density (%)':<38} | {loc_sat_pct:<15.2f}% | {loc_draw_pct:<15.2f}%")
    print(f"{'4. Basin Size (Convergence to K*)':<38} | {basin_sat:<16d} | {basin_draw:<16d}")
    print("=" * 78)

    fdc_gain = fdc_draw - fdc_sat
    ac_gain = ac_draw - ac_sat
    loc_reduction = loc_sat_pct - loc_draw_pct
    basin_gain = basin_draw - basin_sat
    print(f"[+] Key Improvements by DRAW:")
    print(f"    * FDC Correlation Gain : {fdc_gain:+.4f} (stronger global gradient pointing to K*)")
    print(f"    * Autocorrelation Gain : {ac_gain:+.4f} (smoother gradient transitions)")
    print(f"    * Local Traps Reduction: {loc_reduction:.2f}% fewer deceptive local minima")
    print(f"    * Basin of Attraction  : {basin_gain:+d} additional keys converging to solution")
    print("=" * 78)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="S-AES Landscape Metrics Analyzer — Evaluates all 65,536 keys for Standard SAT vs DRAW.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python landscape_metrics.py 0x6F6B 0x0738
  python landscape_metrics.py 0x6F6B 0x0738 0x1234 0x9B24
        """
    )
    parser.add_argument(
        "pairs",
        nargs="*",
        help="Alternating Plaintext and Ciphertext hex/int values: PT1 CT1 PT2 CT2 ..."
    )
    parser.add_argument(
        "-k", "--key",
        type=str,
        default="0xA73B",
        help="Known true master key for metric reference (default: 0xA73B)."
    )

    args = parser.parse_args()
    true_key_val = parse_int_or_hex(args.key)

    if args.pairs:
        if len(args.pairs) % 2 != 0:
            print(f"Error: Expected even number of arguments for [PT CT] pairs. Received {len(args.pairs)} values.")
            sys.exit(1)
        parsed_pairs = []
        for i in range(0, len(args.pairs), 2):
            pt_val = parse_int_or_hex(args.pairs[i])
            ct_val = parse_int_or_hex(args.pairs[i + 1])
            parsed_pairs.append((pt_val, ct_val))
    else:
        # Default single test pair
        parsed_pairs = [(0x6F6B, 0x0738)]

    compute_landscape_metrics(
        parsed_pairs,
        true_key=true_key_val
    )
