"""
saes.py
======================================================================
Pure Python implementation of Simplified AES (S-AES).
Used as ground-truth for testing and verifying PT-CT pairs.
======================================================================
"""

SBOX_LUT = [9, 4, 10, 11, 13, 1, 8, 5, 6, 2, 0, 3, 12, 14, 15, 7]
INV_SBOX_LUT = [SBOX_LUT.index(i) for i in range(16)]

def sub_nibble(n: int) -> int:
    return SBOX_LUT[n]

def inv_sub_nibble(n: int) -> int:
    return INV_SBOX_LUT[n]

def sub_word(w: int) -> int:
    return (sub_nibble((w >> 4) & 0xF) << 4) | sub_nibble(w & 0xF)

def rot_word(w: int) -> int:
    return ((w & 0xF) << 4) | ((w >> 4) & 0xF)

def key_expansion(key: int) -> tuple:
    """
    Expands a 16-bit key into three 16-bit round keys (K0, K1, K2).
    """
    w0 = (key >> 8) & 0xFF
    w1 = key & 0xFF
    
    # RCON[1] = 0x80 for 8-bit word
    w2 = w0 ^ 0x80 ^ sub_word(rot_word(w1))
    w3 = w2 ^ w1
    
    # RCON[2] = 0x30
    w4 = w2 ^ 0x30 ^ sub_word(rot_word(w3))
    w5 = w4 ^ w3
    
    k0 = (w0 << 8) | w1
    k1 = (w2 << 8) | w3
    k2 = (w4 << 8) | w5
    return (k0, k1, k2)

def mult4(n: int) -> int:
    """Multiplies nibble n by x^2 (4) in GF(2^4) with polynomial x^4 + x + 1."""
    # From SAT equations:
    # ov[0] = iv[2]
    # ov[1] = iv[3] ^ iv[0]
    # ov[2] = iv[0] ^ iv[1]
    # ov[3] = iv[1]
    # NOTE: Our arrays in SAT have index 0 as MSB of the nibble!
    # Let's map it exactly to the SAT equations to guarantee compatibility.
    # iv[0..3] are the bits of n from MSB to LSB.
    iv0 = (n >> 3) & 1
    iv1 = (n >> 2) & 1
    iv2 = (n >> 1) & 1
    iv3 = n & 1
    
    ov0 = iv2
    ov1 = iv3 ^ iv0
    ov2 = iv0 ^ iv1
    ov3 = iv1
    
    return (ov0 << 3) | (ov1 << 2) | (ov2 << 1) | ov3

def mix_columns(state: int) -> int:
    """
    State is 16 bits (4 nibbles): N0 N1 N2 N3
    MixColumns applies to each column (N0,N1) and (N2,N3).
    Matrix is:
    [ 1  4 ]
    [ 4  1 ]
    """
    n0 = (state >> 12) & 0xF
    n1 = (state >> 8) & 0xF
    n2 = (state >> 4) & 0xF
    n3 = state & 0xF
    
    m0 = n0 ^ mult4(n1)
    m1 = mult4(n0) ^ n1
    m2 = n2 ^ mult4(n3)
    m3 = mult4(n2) ^ n3
    
    return (m0 << 12) | (m1 << 8) | (m2 << 4) | m3

def shift_rows(state: int) -> int:
    """
    State is 16 bits (4 nibbles): N0 N1 N2 N3
    ShiftRows swaps N1 and N3.
    Returns: N0 N3 N2 N1
    """
    n0 = (state >> 12) & 0xF
    n1 = (state >> 8) & 0xF
    n2 = (state >> 4) & 0xF
    n3 = state & 0xF
    
    return (n0 << 12) | (n3 << 8) | (n2 << 4) | n1

def encrypt(pt: int, key: int) -> int:
    k0, k1, k2 = key_expansion(key)
    
    # Round 0: AddRoundKey
    state = pt ^ k0
    
    # Round 1: SubBytes
    n0 = sub_nibble((state >> 12) & 0xF)
    n1 = sub_nibble((state >> 8) & 0xF)
    n2 = sub_nibble((state >> 4) & 0xF)
    n3 = sub_nibble(state & 0xF)
    state = (n0 << 12) | (n1 << 8) | (n2 << 4) | n3
    
    # Round 1: ShiftRows
    state = shift_rows(state)
    
    # Round 1: MixColumns
    state = mix_columns(state)
    
    # Round 1: AddRoundKey
    state = state ^ k1
    
    # Round 2: SubBytes
    n0 = sub_nibble((state >> 12) & 0xF)
    n1 = sub_nibble((state >> 8) & 0xF)
    n2 = sub_nibble((state >> 4) & 0xF)
    n3 = sub_nibble(state & 0xF)
    state = (n0 << 12) | (n1 << 8) | (n2 << 4) | n3
    
    # Round 2: ShiftRows
    state = shift_rows(state)
    
    # Round 2: AddRoundKey
    state = state ^ k2
    
    return state

if __name__ == "__main__":
    import argparse

    def parse_int_or_hex(val: str) -> int:
        val = val.strip()
        if val.lower().startswith("0x"):
            return int(val, 16)
        try:
            return int(val, 16) if any(c in "abcdefABCDEF" for c in val) else int(val)
        except ValueError:
            return int(val, 16)

    parser = argparse.ArgumentParser(description="Simplified AES (S-AES) Encryption")
    parser.add_argument("pt", nargs="?", default="0x6F6B", help="Plaintext in hex (e.g., 0x6F6B or 6F6B) or integer")
    parser.add_argument("key", nargs="?", default="0xA73B", help="16-bit Master Key in hex (e.g., 0xA73B) or integer")
    parser.add_argument("--expected-ct", "-e", type=str, default=None, help="Optional expected ciphertext to verify against")

    args = parser.parse_args()

    pt = parse_int_or_hex(args.pt)
    key = parse_int_or_hex(args.key)

    res = encrypt(pt, key)
    print(f"Plaintext  : 0x{pt:04X} ({pt})")
    print(f"Master Key : 0x{key:04X} ({key})")
    print(f"Ciphertext : 0x{res:04X} ({res})")

    if args.expected_ct is not None:
        expected = parse_int_or_hex(args.expected_ct)
        print(f"Expected CT: 0x{expected:04X}")
        assert res == expected, f"Mismatch! Got 0x{res:04X}, expected 0x{expected:04X}"
        print("Verification passed successfully.")
    elif pt == 0x6F6B and key == 0xA73B:
        assert res == 0x0738, "Default test vector mismatch!"
        print("Default test vector verified: 0x6F6B -> 0x0738 (Key: 0xA73B)")
