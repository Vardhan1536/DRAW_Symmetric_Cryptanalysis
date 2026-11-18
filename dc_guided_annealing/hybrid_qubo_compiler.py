import sys
import os
import random
import numpy as np
import dimod
from collections import Counter

def build_clause_pubo(clause, weight, var_namer):
   
    terms = {frozenset(): weight}
    for lit in clause:
        vname = var_namer(lit)
        new_terms = {}
        for vset, coef in terms.items():
            if lit > 0:
                new_terms[vset] = new_terms.get(vset, 0.0) + coef
                k2 = frozenset((*vset, vname))
                new_terms[k2] = new_terms.get(k2, 0.0) - coef
            else:
                k2 = frozenset((*vset, vname))
                new_terms[k2] = new_terms.get(k2, 0.0) + coef
        terms = new_terms
    return {k: v for k, v in terms.items() if abs(v) > 1e-12}


def add_pubo_to_bqm(bqm, pubo, weight, stats):

    max_deg = max((len(k) for k in pubo), default=0)

    if max_deg <= 2:
        
        clause_bqm = dimod.BQM('BINARY')
        for vset, coef in pubo.items():
            vlist = list(vset)
            if len(vlist) == 0:
                clause_bqm.offset += coef
            elif len(vlist) == 1:
                clause_bqm.add_linear(vlist[0], coef)
            else:
                clause_bqm.add_quadratic(vlist[0], vlist[1], coef)
        stats['no_ancilla_clauses'] += 1
    else:
        lam = weight + 0.5
        stats['max_lambda'] = max(stats['max_lambda'], lam)
        stats['high_deg_clauses'] += 1
        clause_bqm = dimod.make_quadratic(pubo, lam, vartype='BINARY')

    bqm.update(clause_bqm)



def compile_hybrid_qubo(pt_ct_pairs):
    """
    Builds one master BQM for all PT/CT pairs combined.

    Variable naming (consistent with sa_benchmark.py):
      - Key bits (vars 1..16) : 'k{var_id}'
      - Internal vars per pair : 'p{pair_idx}_v{var_id}'
      - Ancillas               : named by dimod ('v_a*v_b' style)

    Returns:
        bqm       : the master dimod BQM
        stats     : dict of statistics about the compilation
        cw_lists  : list of (compiler, weights) per pair (for verification)
    """
    import json
    sbox_gadget = None 
    bqm = dimod.BQM('BINARY')
    N_KEY = 16
    key_var_ids = None  

    stats = {
        'total_clauses'      : 0,
        'bound_clauses'      : 0,
        'xor_clauses'        : 0,
        'sbox_clauses'       : 0,
        'no_ancilla_clauses' : 0,
        'high_deg_clauses'   : 0,
        'max_lambda'         : 0.0,
        'n_pairs'            : len(pt_ct_pairs),
        'naive_lambda'       : 0.0,
    }

    cw_lists = []

    for pair_idx, (pt, ct) in enumerate(pt_ct_pairs):
        comp, weights = get_weighted_clauses(pt, ct)
        cw_lists.append((comp, weights))

        if key_var_ids is None:
            key_var_ids = set(comp.master_key)

        naive_lambda_pair = 2.0 * sum(weights)
        stats['naive_lambda'] = max(stats['naive_lambda'], naive_lambda_pair)

        def var_namer(lit, _pidx=pair_idx):
            """Map a literal to a BQM variable name."""
            var = abs(lit)
            if var in key_var_ids:
                return f'k{var}'
            return f'p{_pidx}_v{var}'

        sbox_buffer = []

        def flush_sbox_buffer():
            if not sbox_buffer: return
            if not sbox_gadget:
                for pubo, w in sbox_buffer:
                    add_pubo_to_bqm(bqm, pubo, w, stats)
                sbox_buffer.clear()
                return

            sbox_vars_strs = list(set(v for pubo, _ in sbox_buffer for vset in pubo.keys() for v in vset))

            def get_var_id(vstr):
                if vstr.startswith('k'): return int(vstr[1:])
                return int(vstr.split('_v')[1])

            sbox_vars = sorted(sbox_vars_strs, key=get_var_id)
            assert len(sbox_vars) == 8, f"Expected 8 vars in sbox, got {len(sbox_vars)}: {sbox_vars}"

           
            var_map = {}
            for i in range(4): var_map[f'x{i}'] = sbox_vars[i]
            for i in range(4): var_map[f'y{i}'] = sbox_vars[i+4]

            def map_gadget_name(name):
                parts = name.split('*')
                mapped_parts = [var_map[p] for p in parts]
                return '*'.join(mapped_parts)

            clause_bqm = dimod.BQM('BINARY')
            clause_bqm.offset += sbox_gadget.get('offset', 0.0)

            for k, v in sbox_gadget.get('linear', {}).items():
                clause_bqm.add_linear(map_gadget_name(k), v)

            for k, v in sbox_gadget.get('quadratic', {}).items():
                k1, k2 = k.split(',')
                clause_bqm.add_quadratic(map_gadget_name(k1), map_gadget_name(k2), v)

            bqm.update(clause_bqm)
            sbox_buffer.clear()

        for ci, (clause, ttype, dtype) in enumerate(comp.clauses):
            w = weights[ci]
            stats['total_clauses'] += 1

            pubo = build_clause_pubo(clause, w, var_namer)

            if ttype == T_BOUND:
                stats['bound_clauses'] += 1
                add_pubo_to_bqm(bqm, pubo, w, stats)
            elif ttype == T_XOR:
                stats['xor_clauses'] += 1
                add_pubo_to_bqm(bqm, pubo, w, stats)
            else:
                stats['sbox_clauses'] += 1
                sbox_buffer.append((pubo, w))
                if len(sbox_buffer) == 23:
                    flush_sbox_buffer()

    stats['total_bqm_vars']  = len(bqm.variables)
    stats['total_quad_terms']= len(bqm.quadratic)
    stats['n_key_vars']      = sum(1 for v in bqm.variables if str(v).startswith('k'))
    stats['n_ancilla_vars']  = sum(1 for v in bqm.variables if '*' in str(v))

    return bqm, stats, cw_lists, key_var_ids



def compute_cwmc_energy(key, cw_lists):
    
    total_cwmc = 0.0
    for comp, weights in cw_lists:
        pt_int = comp._pt_given
        asgn = comp.forward_solve(pt_int, key)
        energy = sum(
            w for i, (cl, _, _) in enumerate(comp.clauses)
            if not SAESCompiler.satisfied(cl, asgn)
            for w in [weights[i]]
        )
        total_cwmc += energy
    return total_cwmc


def compute_qubo_energy_for_key(bqm, key, cw_lists, key_var_ids):
    
    full_asgn = {}

    key_vars_sorted = sorted(key_var_ids)
    for i, kv in enumerate(key_vars_sorted):
        full_asgn[f'k{kv}'] = (key >> (15 - i)) & 1

    for pair_idx, (comp, weights) in enumerate(cw_lists):
        pt_int = comp._pt_given

        asgn = comp.forward_solve(pt_int, key)

        for var_id, val in asgn.items():
            if var_id not in key_var_ids:
                qname = f'p{pair_idx}_v{var_id}'
                if qname in bqm.variables:
                    full_asgn[qname] = val

    for av in bqm.variables:
        if av not in full_asgn:
            parts = str(av).split('*')
            val = 1
            for p in parts:
                val &= full_asgn.get(p, 0)
            full_asgn[av] = val

    energy = bqm.offset
    for v in bqm.variables:
        val_v = full_asgn.get(v, 0)
        energy += bqm.get_linear(v) * val_v
    for (u, v2), coef in bqm.quadratic.items():
        energy += coef * full_asgn.get(u, 0) * full_asgn.get(v2, 0)

    return energy



def landscape_correlation_audit(bqm, cw_lists, key_var_ids, true_key,
                                 n_sample=500, seed=42):
    """
    Samples n_sample keys from the 65536-key space and computes:
      - CWMC energy (ground truth) for each key
      - QUBO energy (from BQM evaluation) for each key
      - Pearson correlation coefficient R
      - R^2 (coefficient of determination)
      - Spearman rank correlation
      - Whether the true key k* achieves minimum QUBO energy in sample
    """
    rng = random.Random(seed)

    close_keys = [true_key ^ (1 << i) for i in range(16)]  
    sample_keys = [true_key] + close_keys
    while len(sample_keys) < n_sample:
        k = rng.randint(0, 0xFFFF)
        if k not in sample_keys:
            sample_keys.append(k)
    sample_keys = sample_keys[:n_sample]

    print(f"\n    Evaluating {n_sample} keys ({n_sample - 17} random + "
          f"true key + 16 Hamming-1 neighbors)...")

    cwmc_energies = []
    qubo_energies = []

    for idx, key in enumerate(sample_keys):
        if (idx + 1) % 50 == 0:
            print(f"      {idx + 1}/{n_sample}...", end='\r')

        ce = compute_cwmc_energy(key, cw_lists)
        qe = compute_qubo_energy_for_key(bqm, key, cw_lists, key_var_ids)
        cwmc_energies.append(ce)
        qubo_energies.append(qe)

    print(f"      {n_sample}/{n_sample}... done.")

    ce_arr = np.array(cwmc_energies)
    qe_arr = np.array(qubo_energies)

    if ce_arr.std() > 0 and qe_arr.std() > 0:
        pearson_r = np.corrcoef(ce_arr, qe_arr)[0, 1]
        r_squared = pearson_r ** 2
    else:
        pearson_r = 0.0
        r_squared = 0.0

    from scipy.stats import spearmanr
    spearman_r, spearman_p = spearmanr(ce_arr, qe_arr)


    true_key_idx = 0
    true_cwmc   = cwmc_energies[true_key_idx]
    true_qubo   = qubo_energies[true_key_idx]
    qubo_rank   = int(np.sum(qe_arr <= true_qubo))
    cwmc_rank   = int(np.sum(ce_arr <= true_cwmc))

    return {
        'n_sample'   : n_sample,
        'pearson_r'  : pearson_r,
        'r_squared'  : r_squared,
        'spearman_r' : spearman_r,
        'spearman_p' : spearman_p,
        'true_cwmc_energy' : true_cwmc,
        'true_qubo_energy' : true_qubo,
        'cwmc_rank_of_true': cwmc_rank,
        'qubo_rank_of_true': qubo_rank,
        'cwmc_energies'    : cwmc_energies,
        'qubo_energies'    : qubo_energies,
        'sample_keys'      : sample_keys,
    }


