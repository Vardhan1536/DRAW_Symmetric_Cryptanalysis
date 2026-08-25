import sys
import argparse
import numpy as np
from typing import List, Dict, Set

from sat import (
    SAESMassacciCompiler,
    parse_int_or_hex,
    D_KEY, D_R0, D_R1, D_R2, D_FINAL, D_BOUND
)
from saes import encrypt

def build_draw_weights(compiler: SAESMassacciCompiler) -> List[float]:
    adj: Dict[int, Set[int]] = {v: set() for v in range(1, compiler.var_counter)}
    for clause in compiler.clauses:
        vars_in_clause = sorted([abs(lit) for lit in clause])
        for i in range(len(vars_in_clause)):
            for j in range(i + 1, len(vars_in_clause)):
                adj[vars_in_clause[i]].add(vars_in_clause[j])

    reachable_ct: Dict[int, Set[int]] = {v: set() for v in range(1, compiler.var_counter)}

    ct_vars = set()
    for idx, clause in enumerate(compiler.clauses):
        if compiler.clause_tags[idx] == D_FINAL:
            for lit in clause:
                ct_vars.add(abs(lit))

    bound_vars = set()
    for idx, clause in enumerate(compiler.clauses):
        if compiler.clause_tags[idx] == D_BOUND:
            bound_vars.add(abs(clause[0]))

    actual_ct_vars = ct_vars.intersection(bound_vars)

    for v in actual_ct_vars:
        reachable_ct[v].add(v)

    for v in range(compiler.var_counter - 1, 0, -1):
        for neighbor in adj[v]:
            reachable_ct[v].update(reachable_ct[neighbor])

    draw_weights = []
    for clause in compiler.clauses:
        reach = set()
        for lit in clause:
            reach.update(reachable_ct[abs(lit)])
        w = float(len(reach)) if len(reach) > 0 else 1.0
        draw_weights.append(w)

    return draw_weights

def export_dimacs_wcnf(compiler: SAESMassacciCompiler, weights: List[float], filepath: str):
    top_weight = sum(weights) + 1
    with open(filepath, 'w') as f:
        f.write(f"p wcnf {compiler.var_counter - 1} {len(compiler.clauses)} {int(top_weight)}\n")
        for idx, clause in enumerate(compiler.clauses):
            w = int(round(weights[idx]))
            f.write(f"{w} " + " ".join(map(str, clause)) + " 0\n")

def run_draw_analysis(pairs, output_file=None):
    compiler = SAESMassacciCompiler()
    for pt, ct in pairs:
        compiler.add_plaintext_ciphertext_pair(pt, ct)

    weights = build_draw_weights(compiler)

    stages = {
        'Key Schedule Expansion': [],
        'Initial AddRoundKey': [],
        'Round 1 Transformations': [],
        'Round 2 SubBytes Layer': [],
        'Final AddRoundKey': [],
        'Plaintext/Ciphertext Boundaries': []
    }

    for i, dtype in enumerate(compiler.clause_tags):
        w = weights[i]
        if dtype == D_KEY:
            stages['Key Schedule Expansion'].append(w)
        elif dtype == D_R0:
            stages['Initial AddRoundKey'].append(w)
        elif dtype == D_R1:
            stages['Round 1 Transformations'].append(w)
        elif dtype == D_R2:
            stages['Round 2 SubBytes Layer'].append(w)
        elif dtype == D_FINAL:
            stages['Final AddRoundKey'].append(w)
        elif dtype == D_BOUND:
            stages['Plaintext/Ciphertext Boundaries'].append(w)

    print("=" * 73)
    print(f"DRAW WEIGHT DISTRIBUTION ANALYSIS ({len(pairs)} PT-CT Block(s))")
    print("=" * 73)
    header = f"{'Component':<35} | {'Min w':<6} | {'Max w':<6} | {'Avg w':<6} | {'Median w':<8}"
    print(header)
    print("-" * len(header))

    for name, w_list in stages.items():
        if w_list:
            min_w = min(w_list)
            max_w = max(w_list)
            avg_w = float(np.mean(w_list))
            med_w = float(np.median(w_list))
            print(f"{name:<35} | {min_w:<6.1f} | {max_w:<6.1f} | {avg_w:<6.2f} | {med_w:<8.1f}")

    print("=" * 73)
    print(f"Total Clauses Processed : {len(compiler.clauses)}")
    print(f"Total Variables Created  : {compiler.var_counter - 1}")

    if output_file:
        export_dimacs_wcnf(compiler, weights, output_file)
        print(f"[+] Weighted WCNF exported to: {output_file}")
    print("=" * 73)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DRAW Weight Analyzer - Computes reachability weights on S-AES SAT CNF clauses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python draw_weights.py 0x6F6B 0x0738
  python draw_weights.py 0x17FD 0xA55B
  python draw_weights.py 0x6F6B 0x0738 0x1234 0x9B24 -o saes_draw.wcnf
        """
    )
    parser.add_argument(
        "pairs",
        nargs="*",
        help="Alternating Plaintext and Ciphertext values: PT1 CT1 PT2 CT2 ..."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional filepath to export weighted DIMACS WCNF formula."
    )

    args = parser.parse_args()

    if not args.pairs:
        default_pairs = [(0x6F6B, 0x0738)]
        run_draw_analysis(default_pairs, output_file=args.output)
    else:
        if len(args.pairs) % 2 != 0:
            print(f"Error: Expected even number of arguments for [PT CT] pairs. Received {len(args.pairs)} values.")
            sys.exit(1)

        parsed_pairs = []
        for i in range(0, len(args.pairs), 2):
            pt_val = parse_int_or_hex(args.pairs[i])
            ct_val = parse_int_or_hex(args.pairs[i+1])
            parsed_pairs.append((pt_val, ct_val))

        run_draw_analysis(parsed_pairs, output_file=args.output)
