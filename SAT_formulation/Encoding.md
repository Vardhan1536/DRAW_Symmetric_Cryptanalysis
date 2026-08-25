# Key Schedule and MixColumns

This document provides a mathematical derivation of the two independently implemented CNF encodings for Simplified AES (S-AES): the **Key Schedule** and the **MixColumns** layer. All other cipher operations follow the standard propositional logic framing established by Massacci and Marraro [[1]](#references).

---

## 1. Key Schedule Encoding

The 16-bit master key is split into two 8-bit words, W0 and W1. The key schedule derives four additional words to form three 16-bit round keys:
- K0 = (W0, W1)
- K1 = (W2, W3)
- K2 = (W4, W5)

### 1.1 Word Derivations
The words are generated round-by-round using the following recurrence relation:
- W2 = W0 ⊕ RCON(1) ⊕ SubNib(RotNib(W1))
- W3 = W2 ⊕ W1
- W4 = W2 ⊕ RCON(2) ⊕ SubNib(RotNib(W3))
- W5 = W4 ⊕ W3

Where:
- **RotNib(W)** is a cyclic nibble swap (swapping the upper and lower 4-bit nibbles of the 8-bit word). In the SAT compilation, this requires no new variables or clauses; it is handled by renaming/reindexing the variables.
- **SubNib(W)** applies the S-Box substitution to each 4-bit nibble. This is represented using Massacci's minimized prime implicants of the S-box's off-set (where inputs map to incorrect outputs), producing exactly 23 clauses of lengths 3–5 per nibble.
- **RCON(1) = 0x80** and **RCON(2) = 0x30** are round constants.

### 1.2 Clause Count Analysis
The linear XOR relations are decomposed into bit-level constraints. Any k-variable parity constraint x1 ⊕ x2 ⊕ ... ⊕ xk = c (with c ∈ {0, 1}) yields 2^(k-1) CNF clauses of length k.
For instance, a 3-variable XOR sum (k=3) to a constant produces exactly 4 clauses of length 3.

- **Round 1 Word XORs (W2 and W3)**:
  - W2[0] involves a 3-variable XOR with odd parity due to RCON(1) (MSB = 1): 4 clauses.
  - W2[1..7] involve 3-variable XORs with even parity (RCON bits = 0): 7 bit positions × 4 clauses = 28 clauses.
  - W3[0..7] are 3-variable XORs (W3[i] ⊕ W2[i] ⊕ W1[i] = 0): 8 bit positions × 4 clauses = 32 clauses.
- **Round 2 Word XORs (W4 and W5)**:
  - W4[0..1] and W4[4..7] have even parity (RCON(2) bits = 0): 6 bit positions × 4 clauses = 24 clauses.
  - W4[2..3] have odd parity (RCON(2) bits = 1): 2 bit positions × 4 clauses = 8 clauses.
  - W5[0..7] are 3-variable XORs (W5[i] ⊕ W4[i] ⊕ W3[i] = 0): 8 bit positions × 4 clauses = 32 clauses.

**Total Clause Count**:
- S-Box SubNib: 4 S-Box calls total × 2 nibbles × 23 clauses = 92 clauses
- XOR Constraints: 32 (W2) + 32 (W3) + 32 (W4) + 32 (W5) = 128 clauses
- **Total**: 220 clauses.

---

## 2. MixColumns Encoding

MixColumns multiplies each column of the 2×2 state matrix (where each entry is a 4-bit nibble) by the MDS matrix:
```
| 1  4 |
| 4  1 |
```
where elements are represented in GF(2⁴) modulo the irreducible polynomial p(x) = x⁴ + x + 1.

For a column vector [x0, x1]ᵀ, the output [y0, y1]ᵀ is given by:
- y0 = x0 ⊕ (4 · x1)
- y1 = (4 · x0) ⊕ x1

### 2.1 Derivation of GF(2⁴) Multiplication by 4
In GF(2⁴), multiplication by 4 corresponds to multiplication by x².
Let an input nibble be represented by coefficients a0·x³ + a1·x² + a2·x + a3 (where a0 is the MSB).
Multiplying by x² yields:
x² · a = a0·x⁵ + a1·x⁴ + a2·x³ + a3·x²

Reducing modulo p(x) using the relations x⁴ ≡ x + 1 and x⁵ ≡ x² + x:
x² · a ≡ a0·(x² + x) + a1·(x + 1) + a2·x³ + a3·x²
x² · a ≡ a2·x³ + (a0 ⊕ a3)·x² + (a0 ⊕ a1)·x + a1

The output coefficients (o0, o1, o2, o3) are therefore:
- o0 = a2
- o1 = a0 ⊕ a3
- o2 = a0 ⊕ a1
- o3 = a1

### 2.2 Clause Generation and Verification
Each output bit is mapped to CNF clauses:
- o0 ⊕ a2 = 0 (2 variables ⇒ 2 clauses of length 2)
- o1 ⊕ a0 ⊕ a3 = 0 (3 variables ⇒ 4 clauses of length 3)
- o2 ⊕ a0 ⊕ a1 = 0 (3 variables ⇒ 4 clauses of length 3)
- o3 ⊕ a1 = 0 (2 variables ⇒ 2 clauses of length 2)

This yields exactly **12 clauses per multiplication by 4**.

### 2.3 Full MixColumns Constraint System
The MixColumns step processes 4 nibbles in total (2 per column):
- 4 calls to the multiplication-by-4 function: 4 × 12 = 48 clauses
- 16 state-mixing XOR constraints (each of the form yi[k] ⊕ xi[k] ⊕ mult4(xj)[k] = 0, requiring 2 clauses of length 2 per bit constraint): 16 × 2 = 32 clauses.

**Total**: 80 clauses. 
This transformation is applied only in Round 1, matching the S-AES algorithm where MixColumns is omitted in Round 2.

---

## References
**[1]** F. Massacci and L. Marraro (2000). *Logical Cryptanalysis as a SAT Problem: Encoding and Analysis of DES.* Journal of Automated Reasoning, 24(1–2), 165–203.
