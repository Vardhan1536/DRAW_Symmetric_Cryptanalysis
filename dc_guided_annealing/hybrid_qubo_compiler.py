import os
import sys
import random
import numpy as np
import dimod
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SAT_DIR = os.path.abspath(os.path.join(HERE, '..', 'SAT_formulation'))
sys.path.insert(0, HERE)
sys.path.insert(0, SAT_DIR)

from sat import SAESMassacciCompiler, D_BOUND, D_KEY, D_R0, D_R1, D_R2, D_FINAL
from draw_weights import build_draw_weights
from saes import encrypt

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
        lam = weight + 1.5
        stats['max_lambda'] = max(stats['max_lambda'], lam)
        stats['high_deg_clauses'] += 1
        clause_bqm = dimod.make_quadratic(pubo, lam, vartype='BINARY')

    bqm.update(clause_bqm)

def compile_hybrid_qubo(pt_ct_pairs):
    bqm = dimod.BQM('BINARY')
    key_var_ids = None

    stats = {
        'total_clauses': 0,
        'bound_clauses': 0,
        'xor_clauses': 0,
        'sbox_clauses': 0,
        'no_ancilla_clauses': 0,
        'high_deg_clauses': 0,
        'max_lambda': 0.0,
        'n_pairs': len(pt_ct_pairs),
        'naive_lambda': 0.0,
    }

    cw_lists = []

    for pair_idx, (pt, ct) in enumerate(pt_ct_pairs):
        comp = SAESMassacciCompiler()
        comp.add_plaintext_ciphertext_pair(pt, ct)
        weights = build_draw_weights(comp)
        cw_lists.append((comp, weights, pt, ct))

        if key_var_ids is None:
            key_var_ids = set(comp.master_key)

        naive_lambda_pair = 2.0 * sum(weights)
        stats['naive_lambda'] = max(stats['naive_lambda'], naive_lambda_pair)

        def var_namer(lit, _pidx=pair_idx):
            var = abs(lit)
            if var in key_var_ids:
                return f'k{var}'
            return f'p{_pidx}_v{var}'

        for ci, cl in enumerate(comp.clauses):
            w = weights[ci]
            tag = comp.clause_tags[ci]
            stats['total_clauses'] += 1

            pubo = build_clause_pubo(cl, w, var_namer)

            if tag == D_BOUND:
                stats['bound_clauses'] += 1
            elif tag in (D_R0, D_FINAL):
                stats['xor_clauses'] += 1
            else:
                stats['sbox_clauses'] += 1

            add_pubo_to_bqm(bqm, pubo, w, stats)

    stats['total_bqm_vars'] = len(bqm.variables)
    stats['total_quad_terms'] = len(bqm.quadratic)
    stats['n_key_vars'] = sum(1 for v in bqm.variables if str(v).startswith('k'))
    stats['n_ancilla_vars'] = sum(1 for v in bqm.variables if '*' in str(v))

    return bqm, stats, cw_lists, key_var_ids

def build_consistent_state(bqm, key_int, cw_lists, key_var_ids):
    key_vars_sorted = sorted(key_var_ids)
    bqm_vars_set = set(bqm.variables)
    state = {}

    for i, kv in enumerate(key_vars_sorted):
        state[f'k{kv}'] = (key_int >> (15 - i)) & 1

    for pair_idx, (comp, weights, pt, ct) in enumerate(cw_lists):
        bound_assignments = {}
        for idx, cl in enumerate(comp.clauses):
            if comp.clause_tags[idx] == D_BOUND:
                lit = cl[0]
                bound_assignments[abs(lit)] = 1 if lit > 0 else 0

        asgn = dict(bound_assignments)
        for i, v in enumerate(comp.master_key):
            asgn[v] = (key_int >> (15 - i)) & 1

        def propagate():
            changed = False
            for cl in comp.clauses:
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

        var_idx = {}
        for ci, cl in enumerate(comp.clauses):
            for lit in cl:
                var_idx.setdefault(abs(lit), []).append(ci)

        all_vars_sorted = sorted(set(abs(l) for cl in comp.clauses for l in cl))
        for var in all_vars_sorted:
            if var not in asgn:
                best_val, best_pen = 0, float('inf')
                for val in (0, 1):
                    asgn[var] = val
                    pen = sum(
                        1 for ci in var_idx.get(var, [])
                        if not any((lit > 0 and asgn.get(abs(lit)) == 1) or (lit < 0 and asgn.get(abs(lit)) == 0) for lit in comp.clauses[ci])
                    )
                    asgn.pop(var)
                    if pen < best_pen:
                        best_pen, best_val = pen, val
                asgn[var] = best_val

        for var_id, val in asgn.items():
            if var_id not in key_var_ids:
                qname = f'p{pair_idx}_v{var_id}'
                if qname in bqm_vars_set:
                    state[qname] = val

    for av in bqm.variables:
        if av not in state:
            val = 1
            for p in str(av).split('*'):
                val &= state.get(p, 0)
            state[av] = val

    return state

def compute_cwmc_energy(key, cw_lists):
    total_cwmc = 0.0
    for comp, weights, pt, ct in cw_lists:
        bound_assignments = {}
        for idx, cl in enumerate(comp.clauses):
            if comp.clause_tags[idx] == D_BOUND:
                lit = cl[0]
                bound_assignments[abs(lit)] = 1 if lit > 0 else 0

        asgn = dict(bound_assignments)
        for i, v in enumerate(comp.master_key):
            asgn[v] = (key >> (15 - i)) & 1

        def propagate():
            changed = False
            for cl in comp.clauses:
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

        var_idx = {}
        for ci, cl in enumerate(comp.clauses):
            for lit in cl:
                var_idx.setdefault(abs(lit), []).append(ci)

        all_vars_sorted = sorted(set(abs(l) for cl in comp.clauses for l in cl))
        for var in all_vars_sorted:
            if var not in asgn:
                best_val, best_pen = 0, float('inf')
                for val in (0, 1):
                    asgn[var] = val
                    pen = sum(
                        1 for ci in var_idx.get(var, [])
                        if not any((lit > 0 and asgn.get(abs(lit)) == 1) or (lit < 0 and asgn.get(abs(lit)) == 0) for lit in comp.clauses[ci])
                    )
                    asgn.pop(var)
                    if pen < best_pen:
                        best_pen, best_val = pen, val
                asgn[var] = best_val

        e_sat = 0.0
        for idx, cl in enumerate(comp.clauses):
            sat = any((lit > 0 and asgn.get(abs(lit)) == 1) or (lit < 0 and asgn.get(abs(lit)) == 0) for lit in cl)
            if not sat:
                e_sat += weights[idx]
        total_cwmc += e_sat
    return total_cwmc

def compute_qubo_energy_for_key(bqm, key_int, cw_lists, key_var_ids):
    state = build_consistent_state(bqm, key_int, cw_lists, key_var_ids)
    e = bqm.offset
    for v in bqm.variables:
        e += bqm.get_linear(v) * state.get(v, 0)
    for (u, v2), coef in bqm.quadratic.items():
        e += coef * state.get(u, 0) * state.get(v2, 0)
    return e

def audit_qubo_landscape(bqm, cw_lists, key_var_ids, true_key=0xA73B, n_samples=300):
    from scipy.stats import pearsonr, spearmanr

    keys = [true_key]
    for i in range(16):
        keys.append(true_key ^ (1 << i))
    for i in range(16):
        for j in range(i + 1, 16):
            keys.append(true_key ^ (1 << i) ^ (1 << j))

    random.seed(42)
    while len(keys) < n_samples:
        r = random.randint(0, 65535)
        if r not in keys:
            keys.append(r)

    cwmc_energies = []
    qubo_energies = []

    for k in keys:
        ce = compute_cwmc_energy(k, cw_lists)
        qe = compute_qubo_energy_for_key(bqm, k, cw_lists, key_var_ids)
        cwmc_energies.append(ce)
        qubo_energies.append(qe)

    ce_arr = np.array(cwmc_energies)
    qe_arr = np.array(qubo_energies)

    pr, _ = pearsonr(ce_arr, qe_arr)
    sr, _ = spearmanr(ce_arr, qe_arr)
    r2 = pr ** 2

    return {
        'n_keys_audited': len(keys),
        'pearson_r': float(pr),
        'r_squared': float(r2),
        'spearman_r': float(sr),
        'e_cwmc_min': float(ce_arr.min()),
        'e_cwmc_max': float(ce_arr.max()),
        'e_qubo_min': float(qe_arr.min()),
        'e_qubo_max': float(qe_arr.max()),
        'true_key_cwmc_energy': float(cwmc_energies[0]),
        'true_key_qubo_energy': float(qubo_energies[0]),
        'true_key_is_qubo_min': bool(qubo_energies[0] == qe_arr.min()),
    }
