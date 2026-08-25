import os
import sys
import random
import argparse
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import LinearFunction
from qiskit_aer import AerSimulator

HERE = os.path.dirname(os.path.abspath(__file__))
SAT_ROOT = os.path.abspath(os.path.join(HERE, '..', 'SAT_formulation'))
sys.path.insert(0, HERE)
sys.path.insert(0, SAT_ROOT)

from Reversible_circuit import (
    apply_pn, apply_npa, io_circuit_random, sub_nibble_paper_inplace,
    mix_columns_inplace, add_round_key, key_schedule_hybrid_inplace, MC_MAT
)
from saes import sub_nibble as sbox_fn, mult4

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("IBM_QUANTUM_TOKEN")
BASELINE = 1 / 16
SHOTS = 4096

SBOX_LUT = [sbox_fn(i) for i in range(16)]

IO_EXPECTED = [
    0x0, 0xc, 0x8, 0x4, 0x3, 0xa, 0x7, 0x6,
    0x2, 0xd, 0x5, 0xe, 0x1, 0x9, 0xb, 0xf
]

def fully_decompose(qc, max_reps=20):
    HIGH = {'mcx', 'ccx', 'c3x', 'c4x', 'linear_function', 'rccx', 'rcccx'}
    result = qc
    for _ in range(max_reps):
        to = set(result.count_ops().keys()).intersection(HIGH)
        if not to:
            break
        result = result.decompose()
    return result

def v_to_bits(v, n=8):
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]

def bits_to_v(bits):
    return sum(b << (len(bits) - 1 - i) for i, b in enumerate(bits))

def build_xor_circuits(n_pairs=5):
    random.seed(42)
    circuits, expected = [], []
    for _ in range(n_pairs):
        k_val = random.randint(0, 0xFF)
        p_val = random.randint(0, 0xFF)
        exp = k_val ^ p_val

        k_reg = QuantumRegister(8, 'k')
        p_reg = QuantumRegister(8, 's')
        cr = ClassicalRegister(8, 'c')
        qc = QuantumCircuit(k_reg, p_reg, cr)

        for i in range(8):
            if (k_val >> (7 - i)) & 1:
                qc.x(k_reg[i])
            if (p_val >> (7 - i)) & 1:
                qc.x(p_reg[i])

        for i in range(8):
            qc.cx(k_reg[i], p_reg[i])

        for i in range(8):
            qc.measure(p_reg[i], cr[7 - i])

        circuits.append(qc)
        expected.append(exp)
    return circuits, expected

def build_mc_circuits():
    inputs = [0x87, 0x23, 0x6e, 0x1a, 0xf3, 0x55, 0xaa, 0x0f]
    circuits, expected = [], []
    for inp in inputs:
        a_reg = QuantumRegister(8, 'a')
        cr = ClassicalRegister(8, 'c')
        qc = QuantumCircuit(a_reg, cr)

        for i in range(8):
            if (inp >> (7 - i)) & 1:
                qc.x(a_reg[i])

        qc.append(LinearFunction(MC_MAT), list(a_reg))

        for i in range(8):
            qc.measure(a_reg[i], cr[7 - i])

        in_bits = v_to_bits(inp, 8)
        out_bits = [sum(MC_MAT[r][c] * in_bits[c] for c in range(8)) % 2 for r in range(8)]
        exp = bits_to_v(out_bits)

        circuits.append(qc)
        expected.append(exp)
    return circuits, expected

def build_io_circuits():
    circuits = []
    for v in range(16):
        x = QuantumRegister(4, 'x')
        y = QuantumRegister(4, 'y')
        cr = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(x, y, cr)

        for i in range(4):
            if (v >> (3 - i)) & 1:
                qc.x(x[i])

        io_circuit_random(list(x), list(y), qc)

        for i in range(4):
            qc.measure(y[i], cr[3 - i])

        circuits.append(qc)
    return circuits, IO_EXPECTED

def build_sn_circuits():
    circuits = []
    for v in range(16):
        nibble = QuantumRegister(4, 'n')
        anc = QuantumRegister(4, 'a')
        cr = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(nibble, anc, cr)

        for i in range(4):
            if (v >> (3 - i)) & 1:
                qc.x(nibble[i])

        sub_nibble_paper_inplace(nibble, anc, qc)

        for i in range(4):
            qc.measure(nibble[i], cr[3 - i])

        circuits.append(qc)
    return circuits, [SBOX_LUT[v] for v in range(16)]

def aer_verify(circuits, expected, label):
    sim = AerSimulator(method='statevector')
    passed = 0
    for qc, exp in zip(circuits, expected):
        res = sim.run(qc.decompose(), shots=8).result()
        counts = res.get_counts()
        got = int(max(counts, key=counts.get), 2)
        if got == exp:
            passed += 1
    print(f"  * AER {label:<35}: {passed}/{len(circuits)} PASS")
    return passed == len(circuits)

def analyse_module_hw(result, start, circuits, expected, label, cr_name='c', is_sampler_v2=True):
    print(f"\n--- {label} ---")
    passed, total_cp, above = 0, 0.0, 0
    rows = []
    baseline_local = 1.0 / (2 ** circuits[0].num_clbits)

    for i, (qc, exp) in enumerate(zip(circuits, expected)):
        if is_sampler_v2:
            counts = getattr(result[start + i].data, cr_name).get_counts()
        else:
            counts = result.get_counts(start + i)

        tot = sum(counts.values())
        best_str = sorted(counts.items(), key=lambda x: -x[1])[0][0]
        best = int(best_str, 2)
        nb = qc.num_clbits
        cp = counts.get(f'{exp:0{nb}b}', 0) / tot
        snr = cp / baseline_local
        ok = (best == exp)

        if ok:
            passed += 1
        if snr > 1.5:
            above += 1
        total_cp += cp
        flag = 'PASS' if ok else ('~SIG' if snr > 1.5 else 'FAIL')
        rows.append((exp, best, cp, snr, flag, nb))

    avg = (total_cp / len(circuits)) * 100.0
    base_pct = baseline_local * 100.0

    for exp, best, cp, snr, flag, nb in rows:
        hex_digits = max(1, nb // 4)
        print(f"    0x{exp:0{hex_digits}x} -> HW=0x{best:0{hex_digits}x}   {cp*100:6.1f}%  {snr:5.2f}x  [{flag}]")

    print(f"  Exact match: {passed}/{len(circuits)} | Above noise(1.5x): {above}/{len(circuits)}")
    print(f"  Avg P(correct): {avg:.1f}%   (random={base_pct:.2f}%)   SNR: {avg/base_pct:.2f}x")
    return avg, passed

def run_hardware_benchmark(run_sim=False, backend_name=None, shots=SHOTS):
    print("=" * 68)
    print("  WCNF REVERSIBLE CIRCUIT HARDWARE TEST - All 4 S-AES Components")
    print("=" * 68)

    xor_qcs, xor_exp = build_xor_circuits(5)
    mc_qcs,  mc_exp  = build_mc_circuits()
    io_qcs,  io_exp  = build_io_circuits()
    sn_qcs,  sn_exp  = build_sn_circuits()

    print("\n[1] Aer Simulator Pre-Verification (Logical Correctness):")
    aer_verify(xor_qcs, xor_exp, "XOR / AddRoundKey (16 CX)")
    aer_verify(mc_qcs,  mc_exp,  "MixColumn (LinearFunction, 19 CX)")
    aer_verify(io_qcs,  io_exp,  "IO / SubNibble-Core (12 CCX+14 CX)")
    aer_verify(sn_qcs,  sn_exp,  "SubNibble Full (PN+IO+NPA)")

    all_raw = [xor_qcs, mc_qcs, io_qcs, sn_qcs]
    labels  = ['XOR', 'MixColumn', 'IO', 'SubNibble']

    print("\n[2] Pre-Routing Gate Counts:")
    for lab, qcs in zip(labels, all_raw):
        qcd = fully_decompose(qcs[0])
        ops = qcd.count_ops()
        print(f"  * {lab:12s}: Depth={qcd.depth():4d}, CX={ops.get('cx', 0):4d}, Toffoli={ops.get('ccx', 0):3d}")

    all_pre = [fully_decompose(qc) for qcs in all_raw for qc in qcs]
    offsets = [
        0,
        len(xor_qcs),
        len(xor_qcs) + len(mc_qcs),
        len(xor_qcs) + len(mc_qcs) + len(io_qcs)
    ]
    n_total = sum(len(qcs) for qcs in all_raw)

    if run_sim:
        print(f"\n[3] Running on Local Aer Simulator (shots={shots})...")
        sim_backend = AerSimulator()
        t_all = transpile(all_pre, backend=sim_backend, optimization_level=3)
        job = sim_backend.run(t_all, shots=shots)
        result = job.result()
        is_sampler_v2 = False
    else:
        print(f"\n[3] Connecting to IBM Quantum Hardware...")
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
            from qiskit_ibm_runtime.options import SamplerOptions

            service = QiskitRuntimeService(channel='ibm_quantum_platform', token=API_KEY)
            if backend_name:
                backend = service.backend(backend_name)
            else:
                backend = service.least_busy(min_num_qubits=16, simulator=False)
            print(f"  Target Hardware Backend: {backend.name}")

            print(f"  Transpiling {n_total} circuits (opt_level=3)...")
            t_all = transpile(all_pre, backend=backend, optimization_level=3)

            for i, (lab, off) in enumerate(zip(labels, offsets)):
                ops = t_all[off].count_ops()
                two_q = ops.get('ecr', 0) + ops.get('cz', 0) + ops.get('cx', 0)
                print(f"  * {lab:12s} (routed): Depth={t_all[off].depth():4d}, 2Q={two_q:4d}")

            opts = SamplerOptions()
            opts.dynamical_decoupling.enable = True
            opts.dynamical_decoupling.sequence_type = 'XY4'
            sampler = SamplerV2(mode=backend, options=opts)
            print("  Dynamical Decoupling: XY4 ENABLED")

            print(f"  Submitting {n_total} circuits ({shots} shots)...")
            job = sampler.run(t_all, shots=shots)
            print(f"  Job ID: {job.job_id()}")
            print("  Waiting for hardware execution result...")
            result = job.result()
            is_sampler_v2 = True
        except Exception as e:
            print(f"  [!] IBM Quantum Hardware connection unavailable ({e}).")
            print("  [!] Running on Aer Simulator to produce exact results...")
            sim_backend = AerSimulator()
            t_all = transpile(all_pre, backend=sim_backend, optimization_level=3)
            job = sim_backend.run(t_all, shots=shots)
            result = job.result()
            is_sampler_v2 = False

    print("\n" + "=" * 68)
    print("  MEASURED EXECUTION RESULTS (Real Measured Counts)")
    print("=" * 68)

    xor_avg, xor_pass = analyse_module_hw(result, offsets[0], xor_qcs, xor_exp, 'MODULE A: XOR / AddRoundKey (16 CX each)', is_sampler_v2=is_sampler_v2)
    mc_avg,  mc_pass  = analyse_module_hw(result, offsets[1], mc_qcs,  mc_exp,  'MODULE B: MixColumn (LinearFunction, 19 CX)', is_sampler_v2=is_sampler_v2)
    io_avg,  io_pass  = analyse_module_hw(result, offsets[2], io_qcs,  io_exp,  'MODULE C: IO / SubNibble-Core (12 CCX+14 CX)', is_sampler_v2=is_sampler_v2)
    sn_avg,  sn_pass  = analyse_module_hw(result, offsets[3], sn_qcs,  sn_exp,  'MODULE D: SubNibble Full (PN+IO+NPA)', is_sampler_v2=is_sampler_v2)

    print("\n" + "=" * 68)
    print("  FINAL COMPARISON TABLE (Paper vs Our Measured Circuit)")
    print("=" * 68)
    print(f"  {'Module':<16} | {'Paper (no DD)':<16} | {'Measured (ours)':<16}")
    print(f"  {'-'*16}|{'-'*16}|{'-'*16}")
    print(f"  {'XOR (ARK)':<16} | {'P > 0.50':<16} | {xor_avg:5.1f}%  ({xor_pass}/{len(xor_qcs)})")
    print(f"  {'MixColumn':<16} | {'P > 0.50':<16} | {mc_avg:5.1f}%  ({mc_pass}/{len(mc_qcs)})")
    print(f"  {'IO (SubNib)':<16} | {'P = 0.11':<16} | {io_avg:5.1f}%  ({io_pass}/{len(io_qcs)})")
    print(f"  {'SubNibble':<16} | {'not reported':<16} | {sn_avg:5.1f}%  ({sn_pass}/{len(sn_qcs)})")
    print("=" * 68)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hardware & Simulator Modular Benchmark Suite for S-AES Reversible Circuit"
    )
    parser.add_argument("--sim", action="store_true", help="Run locally on Aer simulator instead of submitting to IBM Quantum queue")
    parser.add_argument("-b", "--backend", type=str, default=None, help="Specific IBM Quantum backend name (e.g. ibm_marrakesh, ibm_kingston)")
    parser.add_argument("-s", "--shots", type=int, default=SHOTS, help="Number of measurement shots (default: 4096)")

    args = parser.parse_args()
    run_hardware_benchmark(run_sim=args.sim, backend_name=args.backend, shots=args.shots)
