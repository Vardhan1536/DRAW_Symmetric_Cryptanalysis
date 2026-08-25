# Reversible Circuit Hardware Validation Results

This circuit is inspired from the work of Wang et al. [8]. To validate our architecture on actual quantum computer, we executed each sub module using free tier IBM cloud services. Table 1 compares our real device execution fidelity against the hardware baseline established by Wang et al. [8].

**Table 1: Module-by-module hardware comparison.** 

| Module | Wang et al. [8] $\mathcal{P}$ | Ours $\mathcal{P}$ | Ours SNR | $\Delta\mathcal{P}$ | SNR Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XOR/ARK** | ~53% | 86.6% | 222× | +33.6 | +1.6× |
| **MixColumn** | ~62% | 71.2% | 182× | +9.2 | +1.1× |
| **IO (core)** | 11.0% | 35.1% | 5.61× | +24.1 | +4.3× |
| **Full S-box (SN)** | not tested | 22.9% | 3.66× | — | — |

As demonstrated in Table 1, our framework achieves higher hardware fidelity across all tested cryptographic modules, at the cost of a slightly larger circuit footprint. By coupling our WCNF compilation directly, the success probability of the non-linear IO block increased from 11.0% to 35.1% (+24.1 percentage points). Using our framework we have also done the hardware execution of the complete S-AES S-box (SubNibbles), yielding an average fidelity of 22.9% with a 3.66× Signal-to-Noise Ratio across all 16 input permutations.

---

### Oracle Resource Architecture

The table below shows the physical resource requirements of our proposed architecture against Wang et al. [8]. Note that these represent the resource requirements for a circuit with a single phase forward pass only (without uncomputation).
**Table 2: Oracle Architecture Resource Costs**

| Oracle Architecture | Qubits | Toffoli (CCX) | CNOT (CX) | X gates |
| :--- | :--- | :--- | :--- | :--- |
| **Wang et al. [8]** | 32 | 120 | 392 | 59 |
| **Ours (Hybrid Bidirectional)** | 37 | 384 | 528 | 35 |

---

### Reproducing the Results

To reproduce these hardware validation results, execute the main test script.

**1. Setup the API Key**  
Create a `.env` file in the `two_phase_grovers/Reversible_Circuit/` directory containing your IBM Quantum API key:
```env
IBM_QUANTUM_TOKEN=your_api_key_here
```
*(The script will automatically fetch the token from this file. Do not hardcode it.)*

**2. Run the Hardware Test**  
Execute the modular test script:
```bash
python two_phase_grovers/Reversible_Circuit/test_modular_circuit.py
```

**File Overview:**
- **[`Reversible_circuit.py`](file:///d:/Quantum/IITH_Cryptography/SAT/QA/Consolidation/Grovers/Github2/two_phase_grovers/Reversible_Circuit/Reversible_circuit.py)**: Defines the foundational quantum logic gates (IO block, MixColumns) for the S-AES reversible cryptography.
- **[`saes_grover_oracle.py`](file:///d:/Quantum/IITH_Cryptography/SAT/QA/Consolidation/Grovers/Github2/two_phase_grovers/Reversible_Circuit/saes_grover_oracle.py)**: Composes the individual modules to build the complete 37-qubit Hybrid Bidirectional Oracle (shown in Table 4).
- **[`test_modular_circuit.py`](file:///d:/Quantum/IITH_Cryptography/SAT/QA/Consolidation/Grovers/Github2/two_phase_grovers/Reversible_Circuit/test_modular_circuit.py)**: The main execution script. It submits the compiled modules to IBM quantum hardware using your API key and calculates the execution fidelity and SNR metrics (shown in Table 3).
