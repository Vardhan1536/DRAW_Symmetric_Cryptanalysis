import sys

# Attempt to import PySAT for in-memory solving
try:
    from pysat.solvers import Glucose4
    PYSAT_AVAILABLE = True
except ImportError:
    PYSAT_AVAILABLE = False

D_KEY = "key_schedule"
D_R0 = "round0"
D_R1 = "round1"
D_R2 = "round2"
D_FINAL = "final_round"
D_BOUND = "boundary"

class SAESMassacciCompiler:
    # S-AES S-box lookup table [9]
    SBOX = [9, 4, 10, 11, 13, 1, 8, 5, 6, 2, 0, 3, 12, 14, 15, 7]

    # Minimized Espresso Prime Implicants for the S-AES S-Box
    SBOX_MINIMIZED_CLAUSES = [
        [1, 3, -7],
        [1, 5, -7],
        [5, 8, -2],
        [2, -3, -6],
        [3, -1, -8],
        [6, -1, -5],
        [6, -2, -7],
        [7, -4, -5],
        [8, -3, -4],
        [1, 4, 8, -6],
        [2, 4, 7, -6],
        [3, 4, 6, -2],
        [4, 5, 6, -7],
        [1, 2, 4, 7, 8],
        [2, 3, 5, 6, 7],
        [1, 4, -3, -8],
        [3, 5, -6, -8],
        [4, 5, -3, -7],
        [6, 7, -3, -8],
        [2, -1, -4, -6],
        [7, -1, -2, -3],
        [-2, -3, -4, -5],
        [4, 8, -5, -6, -7],
    ]

    def __init__(self):
        self.clauses = []
        self.clause_tags = []
        self.var_counter = 1
        self.master_key = self.allocate_vars(16)
        self.round_keys = self.compile_key_schedule()

    def allocate_vars(self, count):
        """Allocates a block of contiguous, unique variable IDs."""
        vars_allocated = list(range(self.var_counter, self.var_counter + count))
        self.var_counter += count
        return vars_allocated

    def add_clause(self, clause, tag=""):
        """Appends a CNF clause and its stage tag to the clause database."""
        self.clauses.append(clause)
        self.clause_tags.append(tag)

    def add_xor_clause(self, lits, rhs=0, tag=""):
        """Translates a linear XOR constraint into CNF clauses."""
        n = len(lits)
        for i in range(1 << n):
            parity = bin(i).count('1') % 2
            if parity != rhs:
                clause = []
                for j in range(n):
                    bit = (i >> j) & 1
                    clause.append(-lits[j] if bit == 1 else lits[j])
                self.add_clause(clause, tag=tag)

    def add_sbox_cnf(self, in_vars, out_vars, tag=""):
        """Models S-box using Massacci minimized prime implicants."""
        var_map = in_vars + out_vars
        for min_clause in self.SBOX_MINIMIZED_CLAUSES:
            clause = []
            for lit in min_clause:
                idx = abs(lit) - 1
                mapped_var = var_map[idx]
                clause.append(-mapped_var if lit < 0 else mapped_var)
            self.add_clause(clause, tag=tag)

    def add_multby4_cnf(self, in_vars, out_vars, tag=""):
        """Adds linear CNF clauses for mult4 in GF(2^4)."""
        self.add_xor_clause([out_vars[0], in_vars[2]], rhs=0, tag=tag)
        self.add_xor_clause([out_vars[1], in_vars[3], in_vars[0]], rhs=0, tag=tag)
        self.add_xor_clause([out_vars[2], in_vars[0], in_vars[1]], rhs=0, tag=tag)
        self.add_xor_clause([out_vars[3], in_vars[1]], rhs=0, tag=tag)

    def compile_key_schedule(self):
        """Compiles the S-AES Key Schedule and tags clauses with D_KEY."""
        W0 = self.master_key[0:8]
        W1 = self.master_key[8:16]

        rot_w1_nib0 = W1[4:8]
        rot_w1_nib1 = W1[0:4]

        sbox_out_w2_nib0 = self.allocate_vars(4)
        sbox_out_w2_nib1 = self.allocate_vars(4)

        self.add_sbox_cnf(rot_w1_nib0, sbox_out_w2_nib0, tag=D_KEY)
        self.add_sbox_cnf(rot_w1_nib1, sbox_out_w2_nib1, tag=D_KEY)

        W2 = self.allocate_vars(8)
        self.add_xor_clause([W2[0], W0[0], sbox_out_w2_nib0[0]], rhs=1, tag=D_KEY)
        for k in range(1, 4):
            self.add_xor_clause([W2[k], W0[k], sbox_out_w2_nib0[k]], rhs=0, tag=D_KEY)
        for k in range(4, 8):
            self.add_xor_clause([W2[k], W0[k], sbox_out_w2_nib1[k-4]], rhs=0, tag=D_KEY)

        W3 = self.allocate_vars(8)
        for k in range(8):
            self.add_xor_clause([W3[k], W2[k], W1[k]], rhs=0, tag=D_KEY)

        rot_w3_nib0 = W3[4:8]
        rot_w3_nib1 = W3[0:4]

        sbox_out_w4_nib0 = self.allocate_vars(4)
        sbox_out_w4_nib1 = self.allocate_vars(4)

        self.add_sbox_cnf(rot_w3_nib0, sbox_out_w4_nib0, tag=D_KEY)
        self.add_sbox_cnf(rot_w3_nib1, sbox_out_w4_nib1, tag=D_KEY)

        W4 = self.allocate_vars(8)
        self.add_xor_clause([W4[0], W2[0], sbox_out_w4_nib0[0]], rhs=0, tag=D_KEY)
        self.add_xor_clause([W4[1], W2[1], sbox_out_w4_nib0[1]], rhs=0, tag=D_KEY)
        self.add_xor_clause([W4[2], W2[2], sbox_out_w4_nib0[2]], rhs=1, tag=D_KEY)
        self.add_xor_clause([W4[3], W2[3], sbox_out_w4_nib0[3]], rhs=1, tag=D_KEY)
        for k in range(4, 8):
            self.add_xor_clause([W4[k], W2[k], sbox_out_w4_nib1[k-4]], rhs=0, tag=D_KEY)

        W5 = self.allocate_vars(8)
        for k in range(8):
            self.add_xor_clause([W5[k], W4[k], W3[k]], rhs=0, tag=D_KEY)

        K0 = self.master_key
        K1 = W2 + W3
        K2 = W4 + W5
        return (K0, K1, K2)

    def add_plaintext_ciphertext_pair(self, plaintext_val, ciphertext_val):
        """Translates a known PT-CT pair into SAT constraints with operation tags."""
        pt_vars = self.allocate_vars(16)
        ct_vars = self.allocate_vars(16)

        for k in range(16):
            bit = (plaintext_val >> (15 - k)) & 1
            self.add_clause([pt_vars[k]] if bit == 1 else [-pt_vars[k]], tag=D_BOUND)

        for k in range(16):
            bit = (ciphertext_val >> (15 - k)) & 1
            self.add_clause([ct_vars[k]] if bit == 1 else [-ct_vars[k]], tag=D_BOUND)

        K0, K1, K2 = self.round_keys

        # Round 0: AddRoundKey
        state0 = self.allocate_vars(16)
        for k in range(16):
            self.add_xor_clause([state0[k], pt_vars[k], K0[k]], rhs=0, tag=D_R0)

        # Round 1: SubBytes
        state1_sub = self.allocate_vars(16)
        for i in range(4):
            self.add_sbox_cnf(state0[i*4 : (i+1)*4], state1_sub[i*4 : (i+1)*4], tag=D_R1)

        # Round 1: ShiftRows
        state1_shift = (
            state1_sub[0:4]   +
            state1_sub[12:16] +
            state1_sub[8:12]  +
            state1_sub[4:8]
        )

        # Round 1: MixColumns
        state1_mix = self.allocate_vars(16)
        for offset in (0, 8):
            in_nib0 = [state1_shift[offset + k] for k in range(4)]
            in_nib1 = [state1_shift[offset + k] for k in range(4, 8)]

            mb4_nib0 = self.allocate_vars(4)
            mb4_nib1 = self.allocate_vars(4)

            self.add_multby4_cnf(in_nib0, mb4_nib0, tag=D_R1)
            self.add_multby4_cnf(in_nib1, mb4_nib1, tag=D_R1)

            out_nib0 = [state1_mix[offset + k] for k in range(4)]
            out_nib1 = [state1_mix[offset + k] for k in range(4, 8)]

            for k in range(4):
                self.add_xor_clause([out_nib0[k], in_nib0[k], mb4_nib1[k]], rhs=0, tag=D_R1)
                self.add_xor_clause([out_nib1[k], mb4_nib0[k], in_nib1[k]], rhs=0, tag=D_R1)

        # Round 1: AddRoundKey
        state2 = self.allocate_vars(16)
        for k in range(16):
            self.add_xor_clause([state2[k], state1_mix[k], K1[k]], rhs=0, tag=D_R1)

        # Round 2: SubBytes
        state2_sub = self.allocate_vars(16)
        for i in range(4):
            self.add_sbox_cnf(state2[i*4 : (i+1)*4], state2_sub[i*4 : (i+1)*4], tag=D_R2)

        # Round 2: ShiftRows
        state2_shift = (
            state2_sub[0:4]   +
            state2_sub[12:16] +
            state2_sub[8:12]  +
            state2_sub[4:8]
        )

        # Round 2: AddRoundKey (Final)
        for k in range(16):
            self.add_xor_clause([ct_vars[k], state2_shift[k], K2[k]], rhs=0, tag=D_FINAL)

    def export_dimacs(self, filepath):
        """Exports the generated SAT problem as a standard DIMACS CNF file."""
        with open(filepath, 'w') as f:
            f.write("c S-AES SAT Cryptanalysis (Massacci Style)\n")
            f.write(f"p cnf {self.var_counter - 1} {len(self.clauses)}\n")
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")


def parse_int_or_hex(val: str) -> int:
    val = val.strip()
    if val.lower().startswith("0x"):
        return int(val, 16)
    try:
        return int(val, 16) if any(c in "abcdefABCDEF" for c in val) else int(val)
    except ValueError:
        return int(val, 16)

def compile_and_count(pairs, export_path=None):
    """
    Compiles given PT-CT pair(s) into SAT CNF format and displays the 
    exact variable and clause counts.
    """
    print("=" * 60)
    print(f"S-AES TO SAT FORMULATION: CONVERTING {len(pairs)} PT-CT PAIR(S)")
    print("=" * 60)

    compiler = SAESMassacciCompiler()
    for idx, (pt, ct) in enumerate(pairs, 1):
        print(f"[*] Pair #{idx} -> Plaintext: 0x{pt:04X} ({pt}), Ciphertext: 0x{ct:04X} ({ct})")
        compiler.add_plaintext_ciphertext_pair(pt, ct)

    vars_count = compiler.var_counter - 1
    clauses_count = len(compiler.clauses)

    print("-" * 60)
    print(f"[+] Total Boolean Variables Allocated : {vars_count}")
    print(f"[+] Total CNF Clauses Emitted         : {clauses_count}")
    print("-" * 60)

    if export_path:
        compiler.export_dimacs(export_path)
        print(f"[+] DIMACS CNF exported to: {export_path}")

    print("=" * 60)
    return vars_count, clauses_count

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="S-AES SAT Formulation - Converts PT-CT pairs to SAT CNF and reports variable/clause counts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python sat.py 0x6F6B 0x0738
  python sat.py 0x6F6B 0x0738 0x1234 0x9B24
  python sat.py 0x6F6B 0x0738 0x1234 0x9B24 -o saes_2blocks.cnf
        """
    )
    parser.add_argument(
        "pairs",
        nargs="*",
        help="Alternating Plaintext and Ciphertext hex/integer pairs: PT1 CT1 PT2 CT2 ..."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional filepath to export the compiled formula in standard DIMACS CNF format."
    )

    args = parser.parse_args()

    if not args.pairs:
        parser.print_help()
        sys.exit(0)

    if len(args.pairs) % 2 != 0:
        print(f"Error: Expected even number of arguments for [PT CT] pairs. Received {len(args.pairs)} values.")
        print("Example: python sat.py 0x6F6B 0x0738 0x1234 0x9B24 -o saes_2blocks.cnf")
        sys.exit(1)

    parsed_pairs = []
    for i in range(0, len(args.pairs), 2):
        pt_val = parse_int_or_hex(args.pairs[i])
        ct_val = parse_int_or_hex(args.pairs[i+1])
        parsed_pairs.append((pt_val, ct_val))

    compile_and_count(parsed_pairs, export_path=args.output)
