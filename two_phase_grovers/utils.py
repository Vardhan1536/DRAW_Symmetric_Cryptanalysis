from saes import SBOX_LUT, INV_SBOX_LUT

RCON = {1: 0x80, 2: 0x30}

def inv_sub_bytes(state):
    s0 = (state >> 12) & 0xF; s1 = (state >> 8) & 0xF
    s2 = (state >> 4)  & 0xF; s3 = state & 0xF
    return (INV_SBOX_LUT[s0] << 12) | (INV_SBOX_LUT[s1] << 8) | \
           (INV_SBOX_LUT[s2] << 4)  |  INV_SBOX_LUT[s3]

def shift_rows(state):
    n0 = (state >> 12) & 0xF; n1 = (state >> 8) & 0xF
    n2 = (state >> 4)  & 0xF; n3 = state & 0xF
    return (n0 << 12) | (n3 << 8) | (n2 << 4) | n1

def gf_mult(a, b):
    p = 0
    for _ in range(4):
        if b & 1: p ^= a
        hi_bit_set = a & 8
        a <<= 1
        if hi_bit_set: a ^= 0x13
        b >>= 1
    return p & 0xF

def inv_mix_columns(state):
    n0 = (state >> 12) & 0xF; n1 = (state >> 8) & 0xF
    n2 = (state >> 4)  & 0xF; n3 = state & 0xF
    m0 = gf_mult(9, n0) ^ gf_mult(2, n1)
    m1 = gf_mult(2, n0) ^ gf_mult(9, n1)
    m2 = gf_mult(9, n2) ^ gf_mult(2, n3)
    m3 = gf_mult(2, n2) ^ gf_mult(9, n3)
    return (m0 << 12) | (m1 << 8) | (m2 << 4) | m3

def assign_var(asgn, offset, val, bits):
    for i in range(bits):
        asgn[offset + i] = (val >> ((bits-1)-i)) & 1

def generate_assignment(pt, ct, key):
    asgn = {}
    assign_var(asgn, 1, key, 16)

    W_prev2 = (key >> 8) & 0xFF
    W_prev1 = key & 0xFF

    rcon_1 = RCON[1]
    sb_lower_1 = SBOX_LUT[W_prev1 & 0xF]
    sb_upper_1 = SBOX_LUT[(W_prev1 >> 4) & 0xF]
    W_new0_1 = W_prev2 ^ rcon_1 ^ (sb_lower_1 << 4) ^ sb_upper_1
    W_new1_1 = W_new0_1 ^ W_prev1

    assign_var(asgn, 17, sb_lower_1, 4); assign_var(asgn, 21, sb_upper_1, 4)
    assign_var(asgn, 25, W_new0_1, 8);   assign_var(asgn, 33, W_new1_1, 8)

    rcon_2 = RCON[2]
    sb_lower_2 = SBOX_LUT[W_new1_1 & 0xF]
    sb_upper_2 = SBOX_LUT[(W_new1_1 >> 4) & 0xF]
    W_new0_2 = W_new0_1 ^ rcon_2 ^ (sb_lower_2 << 4) ^ sb_upper_2
    W_new1_2 = W_new0_2 ^ W_new1_1

    assign_var(asgn, 41, sb_lower_2, 4); assign_var(asgn, 45, sb_upper_2, 4)
    assign_var(asgn, 49, W_new0_2, 8);   assign_var(asgn, 57, W_new1_2, 8)

    K1 = (W_new0_1 << 8) | W_new1_1
    K2 = (W_new0_2 << 8) | W_new1_2

    assign_var(asgn, 65, pt, 16); assign_var(asgn, 81, ct, 16)

    s_shift_final = ct ^ K2
    s2s = shift_rows(s_shift_final)
    assign_var(asgn, 177, s2s, 16)

    s2 = inv_sub_bytes(s2s)
    assign_var(asgn, 161, s2, 16)

    s1m = s2 ^ K1
    assign_var(asgn, 129, s1m, 16)

    s1_shift = inv_mix_columns(s1m)
    s1s = shift_rows(s1_shift)
    assign_var(asgn, 113, s1s, 16)

    s0 = inv_sub_bytes(s1s)
    assign_var(asgn, 97, s0, 16)

    def mult4(val):
        bit0 = val & 1
        bit1 = (val >> 1) & 1
        bit2 = (val >> 2) & 1
        bit3 = (val >> 3) & 1
        res = 0
        if bit0: res ^= 0x02
        if bit1: res ^= 0x04
        if bit2: res ^= 0x08
        if bit3: res ^= 0x03
        return res

    n0_0 = (s1_shift >> 12) & 0xF; n1_0 = (s1_shift >> 8) & 0xF
    assign_var(asgn, 145, mult4(n0_0), 4); assign_var(asgn, 149, mult4(n1_0), 4)

    n0_1 = (s1_shift >> 4) & 0xF; n1_1 = s1_shift & 0xF
    assign_var(asgn, 153, mult4(n0_1), 4); assign_var(asgn, 157, mult4(n1_1), 4)

    return asgn

def check_clauses(assignment, clauses):
    for clause in clauses:
        satisfied = False
        for lit in clause:
            v = abs(lit)
            if v not in assignment:
                continue
            val = assignment[v]
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                satisfied = True
                break
        if not satisfied:
            return False
    return True
