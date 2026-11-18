import os
import sys
import time
import random
import numpy as np
import dimod
import neal
from itertools import product as iproduct
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SAT_DIR = os.path.abspath(os.path.join(HERE, '..', 'SAT_formulation'))
sys.path.insert(0, HERE)
sys.path.insert(0, SAT_DIR)

from hybrid_qubo_compiler import (
    compile_hybrid_qubo,
    compute_qubo_energy_for_key,
    build_consistent_state,
    compute_cwmc_energy
)
from sat import SAESMassacciCompiler, D_BOUND
from draw_weights import build_draw_weights
from saes import encrypt, SBOX_LUT, INV_SBOX_LUT, key_expansion, shift_rows, mult4

def decode_key(sample, key_var_ids):
    key_int = 0
    for i, kv in enumerate(sorted(key_var_ids)):
        if sample.get(f'k{kv}', 0):
            key_int |= 1 << (15 - i)
    return key_int

def verify_key(key_int, pt_ct_pairs):
    return all(encrypt(pt, key_int) == ct for pt, ct in pt_ct_pairs)

def build_saes_difference_distribution_table() -> np.ndarray:
    ddt = np.zeros((16, 16), dtype=np.int32)
    for x in range(16):
        for delta_in in range(16):
            x2 = x ^ delta_in
            delta_out = SBOX_LUT[x] ^ SBOX_LUT[x2]
            ddt[delta_in][delta_out] += 1
    return ddt

DDT = build_saes_difference_distribution_table()

def high_prob_trails(threshold: int = 4):
    trails = []
    for di in range(1, 16):
        for do in range(1, 16):
            c = int(DDT[di][do])
            if c >= threshold:
                trails.append((di, do, c))
    trails.sort(key=lambda t: -t[2])
    return trails

def _mult4_val(n: int) -> int:
    return mult4(n)

_MULT4_TABLE = np.array([_mult4_val(n) for n in range(16)], dtype=np.int64)
_INV_SBOX = np.array(INV_SBOX_LUT, dtype=np.int64)
_SBOX_NP = np.array(SBOX_LUT, dtype=np.int64)

def propagate_differential_through_round1(delta_pt: int) -> dict:
    nibble_diffs = []
    for nib in range(4):
        delta_nib_in = (delta_pt >> (12 - 4 * nib)) & 0xF
        nib_dist = {}
        if delta_nib_in == 0:
            nib_dist[0] = 1.0
        else:
            total = sum(DDT[delta_nib_in])
            for do in range(16):
                c = int(DDT[delta_nib_in][do])
                if c:
                    nib_dist[do] = c / total
        nibble_diffs.append(nib_dist)

    nibble_diffs_sr = [nibble_diffs[0], nibble_diffs[3], nibble_diffs[2], nibble_diffs[1]]

    output_dist = defaultdict(float)
    for (d0, p0), (d1, p1), (d2, p2), (d3, p3) in iproduct(
        nibble_diffs_sr[0].items(), nibble_diffs_sr[1].items(),
        nibble_diffs_sr[2].items(), nibble_diffs_sr[3].items(),
    ):
        prob = p0 * p1 * p2 * p3
        m0 = d0 ^ _mult4_val(d1)
        m1 = _mult4_val(d0) ^ d1
        m2 = d2 ^ _mult4_val(d3)
        m3 = _mult4_val(d2) ^ d3
        delta_mix = (m0 << 12) | (m1 << 8) | (m2 << 4) | m3
        output_dist[delta_mix] += prob

    return dict(output_dist)

DC_COLUMNS = [
    {'pt_nib': 0, 'ct_pos': (0, 3), 'k2_nibs': (0, 3)},
    {'pt_nib': 1, 'ct_pos': (1, 2), 'k2_nibs': (1, 2)},
]

_DELTA_IN_CANDIDATES = (1, 2, 3, 4, 8)

def score_k2_column(ct_pos_a, ct_pos_b, delta_pt, oracle, n_samples=4096):
    step = max(1, 65536 // n_samples)
    pts1 = np.arange(0, 65536, step, dtype=np.int64)
    pts2 = pts1 ^ delta_pt

    cts1 = np.array([oracle(int(p)) for p in pts1], dtype=np.int64)
    cts2 = np.array([oracle(int(p)) for p in pts2], dtype=np.int64)

    ct1_a = (cts1 >> (12 - 4 * ct_pos_a)) & 0xF
    ct1_b = (cts1 >> (12 - 4 * ct_pos_b)) & 0xF
    ct2_a = (cts2 >> (12 - 4 * ct_pos_a)) & 0xF
    ct2_b = (cts2 >> (12 - 4 * ct_pos_b)) & 0xF

    scores = np.zeros((16, 16), dtype=np.float64)
    for k2a in range(16):
        dx = _INV_SBOX[ct1_a ^ k2a] ^ _INV_SBOX[ct2_a ^ k2a]
        expected_dy = _MULT4_TABLE[dx]
        for k2b in range(16):
            dy = _INV_SBOX[ct1_b ^ k2b] ^ _INV_SBOX[ct2_b ^ k2b]
            scores[k2a, k2b] = np.sum(dy == expected_dy)
    return scores

def _peak_ratio(scores: np.ndarray) -> float:
    total = scores.sum()
    if total <= 0:
        return 0.0
    return float(scores.max() / total)

def _best_col_scores(col, oracle, n_samples=4096):
    ct_a, ct_b = col['ct_pos']
    best_scores, best_ratio, best_delta_in = None, -1.0, None

    for delta_in in _DELTA_IN_CANDIDATES:
        delta_pt = delta_in << (12 - 4 * col['pt_nib'])
        scores = score_k2_column(ct_a, ct_b, delta_pt, oracle, n_samples=n_samples)
        ratio = _peak_ratio(scores)
        if ratio > best_ratio:
            best_scores, best_ratio, best_delta_in = scores, ratio, delta_in

    return best_scores, best_ratio, best_delta_in

_DC_COLS = DC_COLUMNS

def _inv_key_schedule_k2_to_k0(k2: np.ndarray) -> np.ndarray:
    def rot_word(w):
        return ((w & 0xF) << 4) | ((w >> 4) & 0xF)

    def sub_word(w):
        hi = _SBOX_NP[(w >> 4) & 0xF]
        lo = _SBOX_NP[w & 0xF]
        return (hi << 4) | lo

    w4 = (k2 >> 8) & 0xFF
    w5 = k2 & 0xFF
    w3 = w4 ^ w5
    w2 = w4 ^ 0x30 ^ sub_word(rot_word(w3))
    w1 = w2 ^ w3
    w0 = w2 ^ 0x80 ^ sub_word(rot_word(w1))
    return (w0 << 8) | w1

def _build_master_key_dc_distribution(oracle, n_samples=4096):
    s0, _, d0 = _best_col_scores(DC_COLUMNS[0], oracle, n_samples)
    s1, _, d1 = _best_col_scores(DC_COLUMNS[1], oracle, n_samples)

    info = {
        'col0': {'k2_nibs': DC_COLUMNS[0]['k2_nibs'], 'delta_in': d0, 'total': float(s0.sum()), 'max': float(s0.max())},
        'col1': {'k2_nibs': DC_COLUMNS[1]['k2_nibs'], 'delta_in': d1, 'total': float(s1.sum()), 'max': float(s1.max())},
    }

    has_info = (s0.sum() > 1e-9) and (s1.sum() > 1e-9)
    if not has_info:
        probs = np.ones(65536) / 65536.0
        return probs, s0, s1, info

    k2 = np.arange(65536, dtype=np.int64)
    n0 = (k2 >> 12) & 0xF
    n1 = (k2 >> 8) & 0xF
    n2 = (k2 >> 4) & 0xF
    n3 = k2 & 0xF

    s0_vals = s0[n0, n3]
    s1_vals = s1[n1, n2]

    raw_scores = ((s0_vals + 1e-3) * (s1_vals + 1e-3)) ** 4
    k2_probs = raw_scores / raw_scores.sum()

    k0 = _inv_key_schedule_k2_to_k0(k2)
    probs = np.zeros(65536, dtype=np.float64)
    np.add.at(probs, k0, k2_probs)

    return probs, s0, s1, info

def run_dc_attack(oracle, n_samples=4096):
    probs, s0, s1, info = _build_master_key_dc_distribution(oracle, n_samples)
    keys = np.arange(65536, dtype=np.int64)
    preferred = np.zeros(16)
    for b in range(16):
        bit_vals = (keys >> (15 - b)) & 1
        preferred[b] = float(np.sum(probs * bit_vals))

    p = np.clip(preferred, 1e-9, 1 - 1e-9)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    W = entropy / (np.linalg.norm(entropy) + 1e-12)
    return preferred, W, info

def build_dc_biased_bqm(bqm, key_var_ids, preferred_bits, bias_scale=0.5):
    biased_bqm = bqm.copy()
    key_vars_sorted = sorted(key_var_ids)
    for i, kv in enumerate(key_vars_sorted):
        vname = f'k{kv}'
        pref = preferred_bits[i]
        delta_h = (1.0 - 2.0 * pref) * bias_scale
        if vname in biased_bqm.variables:
            biased_bqm.set_linear(vname, biased_bqm.get_linear(vname) + delta_h)
    return biased_bqm

def build_biased_states(bqm, cw_lists, key_var_ids, preferred, W,
                         num_reads=200, flip_prob_scale=0.4, seed=42):
    rng = random.Random(seed)import time
import random
import copy
import numpy as np
import dimod
import neal
from itertools import product as iproduct
from collections import defaultdict



_SBOX_NP  = np.array(SBOX_LUT,     dtype=np.int64)
_INV_SBOX = np.array(INV_SBOX_LUT, dtype=np.int64)

def _mult4_val(n: int) -> int:
    iv = [(n >> (3 - i)) & 1 for i in range(4)]
    ov = [iv[2], iv[3] ^ iv[0], iv[0] ^ iv[1], iv[1]]
    return sum(ov[i] << (3 - i) for i in range(4))

_MULT4 = np.array([_mult4_val(n) for n in range(16)], dtype=np.int64)



def _pt_from_compiler(comp):
    return comp._pt_given

def build_consistent_state(bqm, key_int, cw_lists, key_var_ids):
    key_vars_sorted = sorted(key_var_ids)
    bqm_vars_set    = set(bqm.variables)
    state = {}
    for i, kv in enumerate(key_vars_sorted):
        state[f'k{kv}'] = (key_int >> (15 - i)) & 1
    for pair_idx, (comp, _) in enumerate(cw_lists):
        pt_int = _pt_from_compiler(comp)
        asgn   = comp.forward_solve(pt_int, key_int)
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

def compute_qubo_energy_for_key(bqm, key_int, cw_lists, key_var_ids):
    state = build_consistent_state(bqm, key_int, cw_lists, key_var_ids)
    e = bqm.offset
    for v in bqm.variables:
        e += bqm.get_linear(v) * state.get(v, 0)
    for (u, v2), coef in bqm.quadratic.items():
        e += coef * state.get(u, 0) * state.get(v2, 0)
    return e

def decode_key(sample, key_var_ids):
    key_int = 0
    for i, kv in enumerate(sorted(key_var_ids)):
        if sample.get(f'k{kv}', 0):
            key_int |= 1 << (15 - i)
    return key_int

def verify_key(key_int, pt_ct_pairs):
    return all(encrypt(pt, key_int) == ct for pt, ct in pt_ct_pairs)



def build_ddt() -> np.ndarray:
    """DDT[di][do] = #{x : S(x)^S(x^di) == do}."""
    ddt = np.zeros((16, 16), dtype=np.int32)
    for x in range(16):
        for di in range(16):
            ddt[di][_SBOX_NP[x] ^ _SBOX_NP[x ^ di]] += 1
    return ddt

DDT = build_ddt()

def high_prob_trails(threshold=4):
    return [(di, do, int(DDT[di][do]))
            for di in range(1, 16) for do in range(1, 16)
            if DDT[di][do] >= threshold]



_DC_COLS = [
    {'pt_nib': 0, 'ct_pos': (0, 3)},   
    {'pt_nib': 1, 'ct_pos': (1, 2)},   
]
_DELTA_INS = (1, 2, 3, 4, 8)

def _score_k2_col(ct_a, ct_b, delta_pt, oracle):
   
    pts1 = np.arange(65536, dtype=np.int64)
    pts2 = pts1 ^ delta_pt

    cts1 = np.array([oracle(int(p)) for p in pts1], dtype=np.int64)
    cts2 = np.array([oracle(int(p)) for p in pts2], dtype=np.int64)

    ct1a = (cts1 >> (12 - 4 * ct_a)) & 0xF
    ct1b = (cts1 >> (12 - 4 * ct_b)) & 0xF
    ct2a = (cts2 >> (12 - 4 * ct_a)) & 0xF
    ct2b = (cts2 >> (12 - 4 * ct_b)) & 0xF

    scores = np.zeros((16, 16), dtype=np.float64)
    for k2a in range(16):
        dx          = _INV_SBOX[ct1a ^ k2a] ^ _INV_SBOX[ct2a ^ k2a]
        expected_dy = _MULT4[dx]
        for k2b in range(16):
            dy = _INV_SBOX[ct1b ^ k2b] ^ _INV_SBOX[ct2b ^ k2b]
            scores[k2a, k2b] = np.sum(dy == expected_dy)
    return scores

def _best_col_scores(col, oracle):
    """Try several delta_in values; keep the most discriminating score matrix."""
    ct_a, ct_b = col['ct_pos']
    best, best_ratio = None, -1.0
    best_delta = None
    for di in _DELTA_INS:
        delta_pt = di << (12 - 4 * col['pt_nib'])
        sc  = _score_k2_col(ct_a, ct_b, delta_pt, oracle)
        r   = float(sc.max() / (sc.sum() + 1e-9))
        if r > best_ratio:
            best, best_ratio, best_delta = sc, r, di
    return best, best_ratio, best_delta

def _inv_key_schedule(k2: np.ndarray) -> np.ndarray:
    """
    Closed-form S-AES key-schedule inverse K2 -> K0 (master key).
    Bijective over all 65536 keys.  No oracle calls; pure algebra.
    """
    def _rot(w): return ((w & 0xF) << 4) | ((w >> 4) & 0xF)
    def _sub(w): return (_SBOX_NP[(w >> 4) & 0xF] << 4) | _SBOX_NP[w & 0xF]

    w4 = (k2 >> 8) & 0xFF;  w5 = k2 & 0xFF
    w3 = w4 ^ w5
    w2 = w4 ^ 0x30 ^ _sub(_rot(w3))
    w1 = w2 ^ w3
    w0 = w2 ^ 0x80 ^ _sub(_rot(w1))
    return (w0 << 8) | w1

def run_dc_attack(oracle):
    
    t0 = time.time()
    sc0, r0, d0 = _best_col_scores(_DC_COLS[0], oracle)
    sc1, r1, d1 = _best_col_scores(_DC_COLS[1], oracle)

    
    k2  = np.arange(65536, dtype=np.int64)
    n0  = (k2 >> 12) & 0xF;  n3 = k2 & 0xF
    n1  = (k2 >>  8) & 0xF;  n2 = (k2 >> 4) & 0xF

    s0  = sc0[n0, n3] + 1e-3
    s1  = sc1[n1, n2] + 1e-3
    raw = (s0 * s1) ** 4          
    k2_probs = raw / raw.sum()

    
    k0 = _inv_key_schedule(k2)   
    probs = np.zeros(65536, dtype=np.float64)
    np.add.at(probs, k0, k2_probs)

    
    keys = np.arange(65536, dtype=np.int64)
    preferred = np.array([float(np.sum(probs * ((keys >> (15 - b)) & 1)))
                          for b in range(16)])

   
    p       = np.clip(preferred, 1e-9, 1 - 1e-9)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))   # in [0,1]
    W       = entropy / (np.linalg.norm(entropy) + 1e-12)

    col_info = {
        'col0': {'delta_in': d0, 'peak_ratio': r0, 'total': float(sc0.sum())},
        'col1': {'delta_in': d1, 'peak_ratio': r1, 'total': float(sc1.sum())},
        'elapsed': time.time() - t0,
    }
    return preferred, W, col_info



def build_dc_biased_bqm(bqm, key_var_ids, preferred, dc_bias_scale=0.5):

    biased = bqm.copy()
    key_vars_sorted = sorted(key_var_ids)

    for b, kv in enumerate(key_vars_sorted):
        var_name = f'k{kv}'
        if var_name not in biased.variables:
            continue
        p   = float(np.clip(preferred[b], 0.01, 0.99))
        logit = np.log(p / (1.0 - p))
        h   = -dc_bias_scale * logit          # negative -> favours x=1 when p>0.5
        biased.add_linear(var_name, h)

    return biased



def build_biased_states(bqm, cw_lists, key_var_ids,
                        preferred, W,
                        num_reads=200, flip_prob_scale=1.0, seed=42):
    """
    Builds `num_reads` BQM-consistent initial states.
    Each key bit is sampled from Bernoulli(preferred[b]),
    then randomly flipped with probability W[b]*flip_prob_scale.
    Bits where DC is certain (low W) are rarely flipped.
    Bits where DC is uncertain (high W) are flipped more.
    """
    rng = random.Random(seed)
    states = []
    for _ in range(num_reads):
        key_cand = 0
        for b in range(16):
            base = 1 if rng.random() < preferred[b] else 0
            fp   = min(max(float(W[b]) * flip_prob_scale, 0.0), 0.45)
            bit  = (1 - base) if rng.random() < fp else base
            key_cand |= bit << (15 - b)
        states.append(build_consistent_state(bqm, key_cand, cw_lists, key_var_ids))
    return states



def parse_and_report(response, key_var_ids, true_key, pt_ct_pairs,
                     elapsed, strategy_name, bqm, cw_lists):
    """
    Deduplicates by decoded key, recomputes consistent QUBO energy for each
    unique key (aux vars rebuilt deterministically via forward_solve), and
    ranks by that energy — not SA's raw energy which depends on aux-var state.
    """
    best_raw = {}
    for sample, energy, _ in response.data(['sample', 'energy', 'num_occurrences']):
        k = decode_key(sample, key_var_ids)
        if k not in best_raw or energy < best_raw[k]:
            best_raw[k] = energy

    results = []
    for k, e_sa in best_raw.items():
        e_cons = compute_qubo_energy_for_key(bqm, k, cw_lists, key_var_ids)
        hd     = bin(k ^ true_key).count('1')
        ok     = verify_key(k, pt_ct_pairs)
        results.append((e_cons, e_sa, k, hd, ok))
    results.sort(key=lambda r: (r[0], r[1]))

    found = any(k == true_key for _, _, k, _, _ in results)
    n_reads = sum(1 for _ in response.data(['sample']))

    print(f"\n    Time: {elapsed:.1f}s | Reads: {n_reads} | Unique keys: {len(results)}")
    print(f"    {'#':>2}  {'Key':>6}  {'E_cons':>8}  {'E_sa':>8}  HD")
    for rank, (ec, es, k, hd, ok) in enumerate(results[:10], 1):
        tag = " <<< TRUE KEY" if k == true_key else (" <<< DECRYPTS OK" if ok else "")
        print(f"    {rank:2d}  0x{k:04X}  {ec:8.3f}  {es:8.3f}  {hd}{tag}")

    if found:
        true_rank = next(r for r, (_, _, k, _, _) in enumerate(results, 1) if k == true_key)
        true_ec   = next(ec for ec, _, k, _, _ in results if k == true_key)
        print(f"\n    [{strategy_name}]  TRUE KEY FOUND  0x{true_key:04X}"
              f"  rank #{true_rank}/{len(results)}  E_cons={true_ec:.3f}"
              f"  {'*** RANK 1 ***' if true_rank == 1 else ''}")
    else:
        best_ec, _, best_k, best_hd, _ = results[0]
        print(f"\n    [{strategy_name}]  NOT FOUND  "
              f"best=0x{best_k:04X}  HD={best_hd}  E_cons={best_ec:.3f}")

    true_rank = (next(r for r, (_, _, k, _, _) in enumerate(results, 1)
                      if k == true_key) if found else None)
    return {
        'strategy'  : strategy_name,
        'found'     : found,
        'rank1'     : (found and true_rank == 1),
        'true_rank' : true_rank,
        'best_key'  : results[0][2],
        'best_energy': results[0][0],
        'hd'        : results[0][3],
        'elapsed'   : elapsed,
        'results'   : results,
    }




def strategy_1_dc_biased_annealing(bqm, key_var_ids, cw_lists,
                                    true_key, pt_ct_pairs, oracle,
                                    dc_bias_scale=0.5,
                                    num_reads=200,
                                    num_sweeps_dc=30_000,
                                    num_sweeps_pure=70_000,
                                    beta_range=(1.0, 50.0),
                                    beta_mid=15.0):
    print(f"\n{'─'*70}")
    print(f"  Strategy 1: Two-Phase DC-Biased Annealing")
    print(f"  Phase 1: DC-biased BQM  (sweeps={num_sweeps_dc}, beta 1->{beta_mid})")
    print(f"  Phase 2: Pure QUBO      (sweeps={num_sweeps_pure}, beta {beta_mid}->{beta_range[1]})")
    print(f"  DC bias scale = {dc_bias_scale}  (logit-scaled linear terms on key vars)")
    print(f"{'─'*70}")

    
    print(f"\n    [DC] Running coupled K2-column attack on full codebook...")
    preferred, W, col_info = run_dc_attack(oracle)
    print(f"      done in {col_info['elapsed']:.2f}s")
    print(f"      Col0 (K2 nibs 0,3): delta_in=0x{col_info['col0']['delta_in']:X}  "
          f"peak_ratio={col_info['col0']['peak_ratio']:.3f}")
    print(f"      Col1 (K2 nibs 1,2): delta_in=0x{col_info['col1']['delta_in']:X}  "
          f"peak_ratio={col_info['col1']['peak_ratio']:.3f}")
    print(f"      W         : {np.round(W, 3)}")
    print(f"      Preferred : {np.round(preferred, 3)}")

    
    biased_bqm = build_dc_biased_bqm(bqm, key_var_ids, preferred, dc_bias_scale)

    
    e_true_pure   = compute_qubo_energy_for_key(bqm,        true_key, cw_lists, key_var_ids)
    e_true_biased = compute_qubo_energy_for_key(biased_bqm, true_key, cw_lists, key_var_ids)
    print(f"\n      True key E_pure={e_true_pure:.3f}  E_biased={e_true_biased:.3f}"
          f"  (DC bias shift={e_true_biased-e_true_pure:+.3f})")

    
    print(f"\n    Building {num_reads} DC-biased initial states...", end=' ', flush=True)
    t0 = time.time()
    init_states = build_biased_states(
        bqm, cw_lists, key_var_ids, preferred, W,
        num_reads=num_reads, flip_prob_scale=0.3, seed=42
    )
    print(f"done in {time.time()-t0:.1f}s")

    sampler = neal.SimulatedAnnealingSampler()

    
    print(f"\n    [Phase 1] Annealing on DC-biased BQM...", end=' ', flush=True)
    t0 = time.time()
    resp1 = sampler.sample(
        biased_bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps_dc,
        beta_schedule_type='geometric',
        beta_range=(beta_range[0], beta_mid),
        initial_states=init_states,
    )
    print(f"done in {time.time()-t0:.1f}s")

    
    phase2_init = []
    for sample, _, _ in resp1.data(['sample', 'energy', 'num_occurrences']):
        phase2_init.append(dict(sample))

    
    print(f"    [Phase 2] Annealing on pure QUBO...", end=' ', flush=True)
    t0 = time.time()
    resp2 = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps_pure,
        beta_schedule_type='geometric',
        beta_range=(beta_mid, beta_range[1]),
        initial_states=phase2_init,
    )
    t_total = time.time() - t0

    return parse_and_report(
        resp2, key_var_ids, true_key, pt_ct_pairs,
        t_total, "DC-BiasedAnnealing", bqm, cw_lists
    )
def main():
    print("=" * 70)
    print("  Hybrid QUBO Solver -- DC-Informed Annealing  (v3)")
    print("  Two-phase bias injection + iterative K2-consistency re-weighting")
    print("=" * 70)

    random.seed(777)
    np.random.seed(777)

    true_key    = random.randint(1, 0xFFFF)
    pt_ct_pairs = [(random.randint(0, 0xFFFF), None) for _ in range(3)]
    pt_ct_pairs = [(pt, encrypt(pt, true_key)) for pt, _ in pt_ct_pairs]
    oracle      = lambda pt: encrypt(int(pt), true_key)

    print(f"\n  True Key  : 0x{true_key:04X}")
    for pt, ct in pt_ct_pairs:
        print(f"    0x{pt:04X}  ->  0x{ct:04X}")

   
    print(f"\n{'─'*70}")
    print("  Compiling Hybrid QUBO...")
    print(f"{'─'*70}")
    t0 = time.time()
    bqm, stats, cw_lists, key_var_ids = compile_hybrid_qubo(pt_ct_pairs)
    print(f"  Compiled in {time.time()-t0:.1f}s  |  Variables: {stats['total_bqm_vars']}")

    e_true = compute_qubo_energy_for_key(bqm, true_key, cw_lists, key_var_ids)
    print(f"  True key QUBO energy: {e_true:.4f}  (expected 0.0)")

    
    trails = high_prob_trails(threshold=4)
    print(f"\n  DDT: {len(trails)} trails with count>=4 "
          f"(all uniform at prob=0.25 for this S-box)")

    
    print(f"\n{'='*70}")
    print("  SOLVING")
    print(f"{'='*70}")

    summary = []
    r1 = strategy_1_dc_biased_annealing(
        bqm, key_var_ids, cw_lists, true_key, pt_ct_pairs, oracle,
        dc_bias_scale=0.5, num_reads=200,
        num_sweeps_dc=30_000, num_sweeps_pure=70_000,
        beta_range=(1.0, 50.0), beta_mid=15.0
    )
    summary.append(r1)

    
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  True key: 0x{true_key:04X}\n")
    print(f"  {'Strategy':<30}  {'Status':>12}  {'Best Key':>8}  {'HD':>4}  {'E_cons':>10}")
    print(f"  {'─'*30}  {'─'*12}  {'─'*8}  {'─'*4}  {'─'*10}")
    for r in summary:
        if r['rank1']:
            status = "RANK 1 ***"
        elif r['found']:
            status = f"rank #{r['true_rank']}"
        else:
            status = "not found"
        print(f"  {r['strategy']:<30}  {status:>12}  0x{r['best_key']:04X}  "
              f"    {r['hd']:>4}  {r['best_energy']:>10.3f}")
    print()
if __name__ == "__main__":
    main()
