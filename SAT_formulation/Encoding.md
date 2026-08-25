# Key Schedule and MixColumns CNF Encodings (S-AES)

This document derives the CNF clause structure for the Key Schedule and MixColumns layers of Simplified AES (S-AES).
---

## Key Schedule Encoding

The 16-bit master key is split into two 8-bit words W0 and W1. The key schedule expands these into three 16-bit round keys:

- **K0** = (W0 || W1)
- **K1** = (W2 || W3)
- **K2** = (W4 || W5)

### Word Derivation

**Round 1:**

```
W2 = W0 ⊕ RCON(1) ⊕ SubNib(RotNib(W1))
W3 = W2 ⊕ W1
```

**Round 2:**

```
W4 = W2 ⊕ RCON(2) ⊕ SubNib(RotNib(W3))
W5 = W4 ⊕ W3
```

Where:
- **RotNib** swaps the upper and lower 4-bit nibbles of an 8-bit word — a pure index permutation, zero clause cost.
- **SubNib** applies the 4-bit S-box to each nibble. Each S-box call is compiled from the off-set prime implicants (Espresso/Quine–McCluskey), yielding **23 clauses** per nibble.
- **RCON(1)** = `0x80` = `10000000₂`, **RCON(2)** = `0x30` = `00110000₂`.

A linear XOR constraint x₁ ⊕ x₂ ⊕ … ⊕ xₖ = c encodes to **2^(k−1) clauses** of length k.  
All key schedule words are 8 bits wide, so each bit-level 3-variable XOR (bit ⊕ source ⊕ sbox_out = rhs) gives **4 clauses**.

### Clause Count

| Component | Detail | Clauses |
|-----------|--------|---------|
| SubNib (S-box) | 4 nibble calls × 23 clauses | 92 |
| W2 XOR chain | 8 bits × 4 clauses (RCON(1) injects odd parity on bit 0) | 32 |
| W3 XOR chain | 8 bits × 4 clauses (W3 = W2 ⊕ W1) | 32 |
| W4 XOR chain | 8 bits × 4 clauses (RCON(2) injects odd parity on bits 2,3) | 32 |
| W5 XOR chain | 8 bits × 4 clauses (W5 = W4 ⊕ W3) | 32 |
| **Total** | | **220** |

---

## MixColumns Encoding

MixColumns multiplies each column of the 2×2 nibble state matrix by the circulant matrix over GF(2⁴):

```
M = | 1  4 |
    | 4  1 |
```

modulo the irreducible polynomial p(x) = x⁴ + x + 1.

For a column vector [x₀, x₁]ᵀ the output is:

```
y₀ = x₀ ⊕ (4·x₁)
y₁ = (4·x₀) ⊕ x₁
```

### GF(2⁴) Multiplication by 4

Multiplication by 4 in GF(2⁴) equals multiplication by x² modulo p(x).

Let a = a₀x³ + a₁x² + a₂x + a₃  (a₀ = MSB).

**Step 1 — multiply by x²:**

```
x² · a = a₀x⁵ + a₁x⁴ + a₂x³ + a₃x²
```

**Step 2 — reduce mod p(x) = x⁴ + x + 1**, using x⁴ ≡ x + 1 and x⁵ ≡ x² + x:

```
= a₀(x² + x) + a₁(x + 1) + a₂x³ + a₃x²
= a₂x³ + (a₀ ⊕ a₃)x² + (a₀ ⊕ a₁)x + a₁
```

**Output bit equations** (o₀ = MSB):

```
o₀ = a₂
o₁ = a₀ ⊕ a₃
o₂ = a₀ ⊕ a₁
o₃ = a₁
```

This matches `add_multby4_cnf` in `sat.py`:

```python
self.add_xor_clause([out_vars[0], in_vars[2]], rhs=0)                # o0 = a2
self.add_xor_clause([out_vars[1], in_vars[3], in_vars[0]], rhs=0)   # o1 = a0 ⊕ a3
self.add_xor_clause([out_vars[2], in_vars[0], in_vars[1]], rhs=0)   # o2 = a0 ⊕ a1
self.add_xor_clause([out_vars[3], in_vars[1]], rhs=0)                # o3 = a1
```

### CNF Clause Count per MultBy4 Call

| Equation | Variables | Clauses (2^(k−1)) |
|----------|-----------|-------------------|
| o₀ ⊕ a₂ = 0 | 2 | 2 |
| o₁ ⊕ a₀ ⊕ a₃ = 0 | 3 | 4 |
| o₂ ⊕ a₀ ⊕ a₁ = 0 | 3 | 4 |
| o₃ ⊕ a₁ = 0 | 2 | 2 |
| **Total per call** | | **12** |

### Full MixColumns Layer (Round 1 only)

MixColumns is applied to 2 columns, each column processing 2 nibbles (4 bits each).

| Component | Detail | Clauses |
|-----------|--------|---------|
| MultBy4 | 2 columns × 2 nibbles × 12 clauses | 48 |
| State XOR mixing | yᵢ[k] ⊕ xᵢ[k] ⊕ (4·xⱼ)[k] = 0 per bit, 2 columns × 8 bits × 2 clauses | 32 |
| **Total** | | **80** |

---

## References

[1] F. Massacci and L. Marraro, "Logical Cryptanalysis as a SAT Problem: Encoding and Analysis of DES," *Journal of Automated Reasoning*, 24(1–2), 165–203, 2000.
