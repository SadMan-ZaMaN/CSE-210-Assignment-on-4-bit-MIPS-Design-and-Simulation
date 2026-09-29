"""
=============================================================================
 4-bit MIPS Assembler
 CSE 210 (Jan 2026) - Computer Architecture Sessional
 Group ID: 2   Lab Section: B1   Opcode Sequence: AJGBFIKMPNODELCH
=============================================================================

Converts the reduced MIPS assembly (as specified in the assignment) into
16-bit machine code, following the exact instruction formats given in the
assignment PDF:

    R-type : Opcode(4) SrcReg1(4) SrcReg2(4)        DstReg(4)
    S-type : Opcode(4) SrcReg1(4) DstReg(4)          Shamt(4)
    I-type : Opcode(4) SrcReg1(4) SrcReg2/DstReg(4)  Addr/Immdt(4)
    J-type : Opcode(4) TargetAddress(8)              0000(4)

USAGE
-----
    python3 mips_assembler.py program.asm

    - Prints a listing (PC, source line, binary, hex) to stdout.
    - Writes "<program>_machine_code.txt" with just the encoded words,
      one per line, in both binary and hex (handy to feed into your
      instruction-memory ROM / simulator init file).

DESIGN NOTES / ASSUMPTIONS (documented for the report)
-------------------------------------------------------
1. Registers ($zero,$t0..$t4,$sp) are 4-bit values -> register field codes:
       $zero=0000  $t0=0001  $t1=0010  $t2=0011  $t3=0100  $t4=0101  $sp=0110
   $sp is a normal 4-bit register (code 0110), addressed by the same
   4-bit register field as any other register, and updated by ordinary
   addi instructions (see note 2). Memory addresses are 8 bits wide, so
   the 4-bit $sp value is zero-extended onto the address bus when it is
   used as a base register for lw/sw.

2. push/pop are NOT given their own opcodes (all 16 opcodes 0-15 are
   already used by A-P). They are PSEUDO-INSTRUCTIONS that the assembler
   expands into two real instructions each, so that $sp is updated by an
   ordinary addi. This is required because a vanilla MIPS datapath's
   lw/sw does NOT write back to the base register (it only computes a
   temporary address = base + offset for the memory access):

       push $t0   ->  addi $sp, $sp, -1     (pre-decrement)
                      sw   $t0, 0($sp)      (store at the new top)
       pop  $t0   ->  lw   $t0, 0($sp)      (load from the current top)
                      addi $sp, $sp, 1      (post-increment)

   Because $sp is 4 bits, a push from $sp=0 decrements to $sp=0xF (4-bit
   wrap) and a pop from $sp=0xF increments to $sp=0x0. The expanded
   addi/sw/lw pairs are marked "[[expanded...]]" / "[[auto-inserted for
   push/pop]]" in the listing so you can see exactly what was inserted.

3. Immediate/offset fields (Addr/Immdt, Shamt) are 4 bits. Immediates are
   encoded as 4-bit two's complement, so the representable range is
   -8 .. +7. Addresses used by `j` (J-type) use the full 8-bit Target
   field (0-255), matching the 8-bit address bus.

4. Branch (beq/bneq) targets are PC-relative: the encoded 4-bit
   Addr/Immdt = (label_address - (PC_of_branch + 1)). This is standard
   MIPS-style "branch relative to the next instruction" addressing.
   Because the field is only 4 bits, and (per the group's hardware note)
   the circuit's PC+1 behavior makes -8 unreliable, the *usable* signed
   range is kept to -7 .. +7.
   *** If a branch's true offset falls outside that range, the assembler
   AUTOMATICALLY rewrites it into a short inverse-condition branch (fixed,
   always-in-range +1 offset) immediately followed by an unconditional
   absolute jump (j) to the real target - a classic "branch-over-jump"
   trampoline, e.g.:
       bneq $t0,$t2,farLabel        -- original, out of range
   becomes:
       beq  $t0,$t2,<skip>          -- inverse condition, offset is always +1
       j    farLabel                -- only runs when the ORIGINAL condition was true
   This never fails, because `j` addresses the full 8-bit space. The
   assembler re-checks all branches after every insertion (inserting a
   word shifts every later address) until nothing is left out of range.
   Expanded lines are marked "[[auto-expanded...]]" / "[[auto-inserted
   trampoline...]]" in the listing so you can see exactly what changed. ***
=============================================================================
"""

import re
import sys
import os

# --------------------------------------------------------------------------
# 1. OPCODE TABLE  (Group 2, Section B1 : A J G B F I K M P N O D E L C H)
# --------------------------------------------------------------------------
OPCODES = {
    'add':  0,  # A
    'srl':  1,  # J
    'or':   2,  # G
    'addi': 3,  # B
    'andi': 4,  # F
    'sll':  5,  # I
    'nor':  6,  # K
    'sw':   7,  # M
    'j':    8,  # P
    'beq':  9,  # N
    'bneq': 10, # O
    'subi': 11, # D
    'and':  12, # E
    'lw':   13, # L
    'sub':  14, # C
    'ori':  15, # H
}

# --------------------------------------------------------------------------
# 2. REGISTER TABLE
# --------------------------------------------------------------------------
REGISTERS = {
    '$zero': 0,
    '$t0': 7,
    '$t1': 1,
    '$t2': 2,
    '$t3': 3,
    '$t4': 4,
    '$sp': 6,   # normal 4-bit register, same width as the others
}

# --------------------------------------------------------------------------
# 3. INSTRUCTION FORMATS
# --------------------------------------------------------------------------
R_TYPE = {'add', 'sub', 'and', 'or', 'nor'}
S_TYPE = {'sll', 'srl'}
I_TYPE = {'addi', 'subi', 'andi', 'ori', 'lw', 'sw', 'beq', 'bneq'}
J_TYPE = {'j'}
PSEUDO_STACK = {'push', 'pop'}   # expand to addi + sw / lw + addi


def to_bin(value, bits):
    """Encode an (possibly negative) int into a bits-wide two's complement string."""
    mask = (1 << bits) - 1
    v = value & mask
    return format(v, '0{}b'.format(bits))


def check_signed_range(value, bits, context):
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    if value < lo or value > hi:
        raise ValueError(
            "'{}' needs a {}-bit signed field but value {} is outside "
            "the representable range [{}, {}]".format(context, bits, value, lo, hi)
        )


def check_unsigned_range(value, bits, context):
    hi = (1 << bits) - 1
    if value < 0 or value > hi:
        raise ValueError(
            "'{}' needs a {}-bit unsigned field but value {} is outside "
            "the representable range [0, {}]".format(context, bits, value, hi)
        )


def strip_comment(line):
    return line.split('//')[0].strip()


def split_label(line):
    """Return (label_or_None, rest_of_line)."""
    m = re.match(r'^\s*([A-Za-z_]\w*)\s*:\s*(.*)$', line)
    if m:
        return m.group(1), m.group(2).strip()
    return None, line.strip()


def tokenize(instr_text):
    """'add $t0, $t1, $t2' -> ('add', ['$t0','$t1','$t2'])"""
    parts = instr_text.replace(',', ' ').split()
    mnem = parts[0].lower()
    args = parts[1:]
    return mnem, args


def parse_mem_operand(tok):
    """'3($t2)' -> (3, '$t2')"""
    m = re.match(r'^(-?\d+)\((\$\w+)\)$', tok)
    if not m:
        raise ValueError("Could not parse memory operand '{}' (expected 'offset($reg)')".format(tok))
    return int(m.group(1)), m.group(2)


def reg_code(name, context):
    if name not in REGISTERS:
        raise ValueError("Unknown register '{}' in '{}'".format(name, context))
    return REGISTERS[name]


# --------------------------------------------------------------------------
# 4. PSEUDO-INSTRUCTION EXPANSION  (push / pop -> addi + sw / lw + addi)
# --------------------------------------------------------------------------
def expand_pseudo(entries):
    """Rewrite every push/pop entry into two real instructions so that
    $sp is updated by an ordinary addi (the vanilla MIPS datapath's
    lw/sw does NOT write back to the base register). Runs BEFORE branch
    relaxation so labels / PCs stay consistent with the expanded form.

        push $t0  ->  addi $sp, $sp, -1   ;   sw $t0, 0($sp)
        pop  $t0  ->  lw   $t0, 0($sp)    ;   addi $sp, $sp, 1
    """
    out = []
    for e in entries:
        if e['kind'] != 'instr' or e['mnem'] not in PSEUDO_STACK:
            out.append(e)
            continue

        rd = e['args'][0]
        if e['mnem'] == 'push':
            first = {
                'kind': 'instr', 'label': e['label'], 'mnem': 'addi',
                'args': ['$sp', '$sp', '-1'],
                'raw': '{}   [[expanded: addi $sp,$sp,-1]]'.format(e['raw']),
                'offset_override': None, 'synthetic': True,
            }
            second = {
                'kind': 'instr', 'label': None, 'mnem': 'sw',
                'args': [rd, '0($sp)'],
                'raw': 'sw {}, 0($sp)   [[auto-inserted for push]]'.format(rd),
                'offset_override': None, 'synthetic': True,
            }
        else:  # pop
            first = {
                'kind': 'instr', 'label': e['label'], 'mnem': 'lw',
                'args': [rd, '0($sp)'],
                'raw': '{}   [[expanded: lw {},0($sp)]]'.format(e['raw'], rd),
                'offset_override': None, 'synthetic': True,
            }
            second = {
                'kind': 'instr', 'label': None, 'mnem': 'addi',
                'args': ['$sp', '$sp', '1'],
                'raw': 'addi $sp, $sp, 1   [[auto-inserted for pop]]',
                'offset_override': None, 'synthetic': True,
            }

        out.extend([first, second])
    return out


# --------------------------------------------------------------------------
# 5. BRANCH-RELAXING ASSEMBLER
# --------------------------------------------------------------------------
# Usable signed range for the 4-bit Addr/Immdt field when used as a branch
# offset. Per the group's own hardware note: because the circuit always
# does PC+1, -8 does not behave reliably (it lands as if it were -7), so
# the *usable* range is kept symmetric: -7 .. +7, not the full -8..+7 a
# plain 4-bit two's complement field could otherwise represent.
BRANCH_LO, BRANCH_HI = -7, 7

INVERSE_BRANCH = {'beq': 'bneq', 'bneq': 'beq'}


def parse_source(source_lines):
    """Turn raw source lines into a flat list of entries:
       {'kind':'label_only', 'label':..., 'raw':...}                (e.g. "end:")
       {'kind':'instr', 'label':..., 'mnem':..., 'args':[...], 'raw':...,
        'offset_override': None, 'synthetic': False}
    """
    entries = []
    for raw in source_lines:
        line = strip_comment(raw)
        if not line:
            continue
        label, rest = split_label(line)
        if rest == '':
            entries.append({'kind': 'label_only', 'label': label, 'raw': raw.strip()})
        else:
            mnem, args = tokenize(rest)
            entries.append({
                'kind': 'instr', 'label': label, 'mnem': mnem, 'args': args,
                'raw': raw.strip(), 'offset_override': None, 'synthetic': False,
            })
    return entries


def layout(entries):
    """Assign a PC to every entry (label_only entries get the PC of
    whatever comes next, but don't consume one) and build the symbol
    table. Returns (symtab, pc_list) where pc_list[i] lines up with
    entries[i]."""
    symtab = {}
    pc_list = []
    pc = 0
    for e in entries:
        if e['label'] is not None:
            symtab[e['label']] = pc
        pc_list.append(pc)
        if e['kind'] == 'instr':
            pc += 1
    return symtab, pc_list


def find_out_of_range_branches(entries, pc_list, symtab):
    idxs = []
    for i, e in enumerate(entries):
        if e['kind'] != 'instr' or e['mnem'] not in ('beq', 'bneq'):
            continue
        if e['offset_override'] is not None:
            continue  # already-expanded inverse branch; always in range
        label = e['args'][2]
        if label not in symtab:
            raise ValueError("Undefined label '{}' in line: {}".format(label, e['raw']))
        offset = symtab[label] - (pc_list[i] + 1)
        if offset < BRANCH_LO or offset > BRANCH_HI:
            idxs.append(i)
    return idxs


def expand_branch(entries, i):
    """Replace entries[i] (an out-of-range beq/bneq) with an inverse-
    condition short branch (fixed +1 offset, always in range) followed
    by an unconditional absolute jump (j) to the real target (full 8-bit
    address, no range limit)."""
    e = entries[i]
    inverse = INVERSE_BRANCH[e['mnem']]
    rs, rt, label = e['args']
    short_branch = {
        'kind': 'instr', 'label': e['label'], 'mnem': inverse, 'args': [rs, rt],
        'raw': "{}   [[auto-expanded: was out of range for a direct branch]]".format(e['raw']),
        'offset_override': 1, 'synthetic': True,
    }
    trampoline_jump = {
        'kind': 'instr', 'label': None, 'mnem': 'j', 'args': [label],
        'raw': "j {}   [[auto-inserted trampoline for the long branch above]]".format(label),
        'offset_override': None, 'synthetic': True,
    }
    entries[i:i + 1] = [short_branch, trampoline_jump]


def assemble(source_lines):
    entries = parse_source(source_lines)

    # ---- Pseudo-instruction expansion FIRST, so any new addi/sw/lw
    # words get real PCs before branch relaxation runs. ----
    entries = expand_pseudo(entries)

    # ---- Branch relaxation: expand any out-of-range beq/bneq, then
    # re-check (an insertion shifts every later address, so a branch
    # that was fine before may now be out of range too, or vice versa) ----
    for _ in range(50):
        symtab, pc_list = layout(entries)
        idxs = find_out_of_range_branches(entries, pc_list, symtab)
        if not idxs:
            break
        for i in reversed(idxs):   # expand back-to-front so earlier indices stay valid
            expand_branch(entries, i)
    else:
        raise RuntimeError("Branch relaxation did not converge after 50 passes")

    symtab, pc_list = layout(entries)

    # ---- Final encode pass ----
    from types import SimpleNamespace
    results = []
    for e, pc in zip(entries, pc_list):
        if e['kind'] != 'instr':
            continue
        ins = SimpleNamespace(mnem=e['mnem'], args=e['args'], raw=e['raw'], pc=pc)
        binary, warning = encode(ins, symtab, e['offset_override'])
        results.append(SimpleNamespace(
            pc=pc, raw=e['raw'], binary=binary,
            hex='0x{:04X}'.format(int(binary, 2)), warning=warning,
            synthetic=e['synthetic'],
        ))

    return results, symtab, [e for e in entries if e['kind'] == 'instr']


def encode(ins, symtab, offset_override=None):
    mnem, args, pc = ins.mnem, ins.args, ins.pc
    warning = None

    # NOTE: push/pop never reach here - expand_pseudo() has already
    # rewritten them into addi/sw or lw/addi pairs.

    if mnem not in OPCODES:
        raise ValueError("Unknown mnemonic '{}' in line: {}".format(mnem, ins.raw))
    opcode = OPCODES[mnem]

    # ---- R-type: op rd, rs, rt  ->  Opcode Src1 Src2 Dst ----
    if mnem in R_TYPE:
        rd, rs, rt = args[0], args[1], args[2]
        bits = (format(opcode, '04b') +
                format(reg_code(rs, ins.raw), '04b') +
                format(reg_code(rt, ins.raw), '04b') +
                format(reg_code(rd, ins.raw), '04b'))
        return bits, warning

    # ---- S-type: op rd, rs, shamt -> Opcode Src1 Dst Shamt ----
    if mnem in S_TYPE:
        rd, rs, shamt = args[0], args[1], int(args[2])
        check_unsigned_range(shamt, 4, ins.raw)
        bits = (format(opcode, '04b') +
                format(reg_code(rs, ins.raw), '04b') +
                format(reg_code(rd, ins.raw), '04b') +
                to_bin(shamt, 4))
        return bits, warning

    # ---- I-type ----
    if mnem in I_TYPE:
        if mnem in ('lw', 'sw'):
            # lw $dst, off($base)  /  sw $src, off($base)
            reg_field, mem = args[0], args[1]
            offset, base = parse_mem_operand(mem)
            check_signed_range(offset, 4, ins.raw)
            bits = (format(opcode, '04b') +
                    format(reg_code(base, ins.raw), '04b') +
                    format(reg_code(reg_field, ins.raw), '04b') +
                    to_bin(offset, 4))
            return bits, warning

        if mnem in ('beq', 'bneq'):
            rs, rt = args[0], args[1]
            if offset_override is not None:
                # This is a synthetic short branch inserted by expand_branch():
                # it always jumps exactly over the single trampoline 'j' that
                # follows it, so the offset is a fixed, always-in-range +1 -
                # no label lookup needed.
                offset = offset_override
            else:
                label = args[2]
                if label not in symtab:
                    raise ValueError("Undefined label '{}' in line: {}".format(label, ins.raw))
                target = symtab[label]
                offset = target - (pc + 1)     # PC-relative to next instruction
                if offset < BRANCH_LO or offset > BRANCH_HI:
                    # Should never happen: assemble() runs branch relaxation
                    # first and expands anything out of range into a short
                    # branch + trampoline jump. If this still fires, something
                    # is wrong with the relaxation logic - fail loudly rather
                    # than emit silently-wrong code.
                    raise ValueError(
                        "Internal error: branch to '{}' (offset {}) should have "
                        "been relaxed into a trampoline jump but wasn't. Line: {}"
                        .format(label, offset, ins.raw))
            bits = (format(opcode, '04b') +
                    format(reg_code(rs, ins.raw), '04b') +
                    format(reg_code(rt, ins.raw), '04b') +
                    to_bin(offset, 4))
            return bits, warning

        # addi / subi / andi / ori   ->  op rd, rs, imm
        rd, rs, imm = args[0], args[1], int(args[2])
        check_signed_range(imm, 4, ins.raw)
        bits = (format(opcode, '04b') +
                format(reg_code(rs, ins.raw), '04b') +
                format(reg_code(rd, ins.raw), '04b') +
                to_bin(imm, 4))
        return bits, warning

    # ---- J-type: j target ----
    if mnem in J_TYPE:
        label = args[0]
        if label not in symtab:
            raise ValueError("Undefined label '{}' in line: {}".format(label, ins.raw))
        target = symtab[label]
        check_unsigned_range(target, 8, ins.raw)
        bits = format(opcode, '04b') + format(target, '08b') + '0000'
        return bits, warning

    raise ValueError("No encoding rule matched for '{}'".format(mnem))


# --------------------------------------------------------------------------
# 6. SIMULATOR - actually EXECUTES the assembled program (registers, $sp,
#     memory) so $sp's value after each addi / lw / sw can be checked
#     directly, rather than just trusted from the encoding.
# --------------------------------------------------------------------------
def wrap4(v):
    """Mask to 4 bits and reinterpret as signed two's complement (-8..7)."""
    v &= 0xF
    return v - 16 if v & 0x8 else v


def simulate(instr_entries, symtab, max_steps=2000):
    """
    instr_entries: the FINAL (post-expansion, post-branch-relaxation) list
                    of instr dicts, in address order (index i == PC i, since
                    every entry consumes exactly one word).
    Returns a list of per-step trace dicts, plus final (regs, sp, mem).
    """
    regs = {'$t0': 0, '$t1': 0, '$t2': 0, '$t3': 0, '$t4': 0}
    sp = 0                          # 4-bit stack pointer, 0..15
    mem = [0] * 256                 # 8-bit address space

    def get(name):
        if name == '$zero':
            return 0
        if name == '$sp':
            return sp
        return regs[name]

    def setreg(name, val):
        nonlocal sp
        if name == '$zero':
            return
        if name == '$sp':
            sp = val & 0xF          # 4-bit register -> wraps in 0..15
            return
        regs[name] = wrap4(val)

    end_pc = symtab.get('end', len(instr_entries))
    pc = 0
    trace = []
    steps = 0

    while pc < end_pc and pc < len(instr_entries) and steps < max_steps:
        e = instr_entries[pc]
        mnem, args = e['mnem'], e['args']
        sp_before = sp
        touched_addr = None
        touched_val = None
        next_pc = pc + 1
        is_stack = False

        if mnem in R_TYPE:
            rd, rs, rt = args
            a, b = get(rs), get(rt)
            if mnem == 'add':
                setreg(rd, a + b)
            elif mnem == 'sub':
                setreg(rd, a - b)
            elif mnem == 'and':
                setreg(rd, a & b)
            elif mnem == 'or':
                setreg(rd, a | b)
            elif mnem == 'nor':
                setreg(rd, ~(a | b))
        elif mnem in S_TYPE:
            rd, rs, shamt = args[0], args[1], int(args[2])
            uv = get(rs) & 0xF
            if mnem == 'sll':
                setreg(rd, (uv << shamt) & 0xF)
            else:  # srl
                setreg(rd, (uv >> shamt) & 0xF)
        elif mnem in ('lw', 'sw'):
            reg_field, mem_tok = args
            offset, base = parse_mem_operand(mem_tok)
            if base == '$sp':
                is_stack = True
            addr = (get(base) + offset) & 0xFF
            if mnem == 'lw':
                setreg(reg_field, mem[addr])
            else:
                mem[addr] = wrap4(get(reg_field))
            touched_addr = addr
        elif mnem in ('addi', 'subi', 'andi', 'ori'):
            rd, rs, imm = args[0], args[1], int(args[2])
            if rd == '$sp' or rs == '$sp':
                is_stack = True
            a = get(rs)
            if mnem == 'addi':
                setreg(rd, a + imm)
            elif mnem == 'subi':
                setreg(rd, a - imm)
            elif mnem == 'andi':
                setreg(rd, a & imm)
            else:  # ori
                setreg(rd, a | imm)
        elif mnem in ('beq', 'bneq'):
            rs, rt = args[0], args[1]
            eq = (get(rs) == get(rt))
            taken = eq if mnem == 'beq' else (not eq)
            if taken:
                if e['offset_override'] is not None:
                    next_pc = pc + 1 + e['offset_override']
                else:
                    next_pc = symtab[args[2]]
        elif mnem == 'j':
            next_pc = symtab[args[0]]
        else:
            raise ValueError("Simulator has no rule for mnemonic '{}'".format(mnem))

        trace.append({
            'pc': pc, 'raw': e['raw'], 'mnem': mnem,
            'text': mnem + ' ' + ', '.join(args),
            'regs': dict(regs), 'sp_before': sp_before, 'sp_after': sp,
            'addr': touched_addr, 'val': touched_val,
            'is_stack': is_stack,
        })
        pc = next_pc
        steps += 1

    if steps >= max_steps:
        raise RuntimeError("Simulation did not halt within {} steps - check for an infinite loop".format(max_steps))

    return trace, regs, sp, mem


def write_hex_file(path, entries, plain=False):
    """
    Write a Logisim-loadable ROM image: Logisim's "Load Image..." dialog on
    a ROM component requires the file to start with a header line
    identifying the image format. This writes the "v2.0 raw" format,
    followed by the words themselves (hex, no '0x' prefix, one per line,
    in address order - which Logisim maps straight onto the ROM's address
    inputs 0,1,2,...).

    Pass plain=True to omit the header instead, giving a bare hex-word-
    per-line file (still 4 hex digits per 16-bit word) suitable for other
    tools such as Verilog's $readmemh().
    """
    with open(path, 'w') as f:
        if not plain:
            f.write("v2.0 raw\n")
        for ins in entries:
            f.write("{:04x}\n".format(int(ins.binary, 2)))


# --------------------------------------------------------------------------
# 7. CLI
# --------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    plain = False
    if '--plain' in args:
        plain = True
        args.remove('--plain')

    if len(args) != 1:
        print("Usage: python3 mips_assembler.py <program.asm> [--plain]")
        print("  By default the ROM image .txt file starts with Logisim's")
        print("  'v2.0 raw' header, so it can be loaded directly into a ROM")
        print("  via Logisim's Menu > Load Image...")
        print("  Pass --plain to omit that header (e.g. for Verilog's")
        print("  $readmemh(), which does not expect a header line).")
        sys.exit(1)

    path = args[0]
    with open(path) as f:
        source_lines = f.readlines()

    entries, symtab, instr_entries = assemble(source_lines)

    print("Symbol table:")
    for lbl, addr in symtab.items():
        print("  {:<8} -> PC {}".format(lbl, addr))
    print()

    header = "{:<3} {:<32} {:<16} {:<8}".format("PC", "Instruction", "Binary", "Hex")
    print(header)
    print('-' * len(header))
    has_warning = False
    for ins in entries:
        tag = "  [AUTO]" if ins.synthetic else ""
        print("{:<3} {:<32} {:<16} {:<8}{}".format(ins.pc, ins.raw, ins.binary, ins.hex, tag))
        if ins.warning:
            has_warning = True
            print("     !! WARNING: {}".format(ins.warning))

    out_path = os.path.splitext(path)[0] + '_machine_code.txt'
    with open(out_path, 'w') as f:
        for ins in entries:
            f.write("{} {} // PC {}: {}\n".format(ins.binary, ins.hex, ins.pc, ins.raw))
    print("\nMachine code written to: {}".format(out_path))

    hex_path = os.path.splitext(path)[0] + '_rom_image.txt'
    write_hex_file(hex_path, entries, plain=plain)
    print("ROM image written to:    {} ({})".format(
        hex_path, "plain, no header" if plain else "Logisim ROM image"))

    if has_warning:
        print("\n*** Unexpected: a WARNING fired even after branch relaxation.")
        print("*** This should not happen - please double check the source.")

    num_auto = sum(1 for ins in entries if ins.synthetic)
    if num_auto:
        print("\nNote: {} instruction(s) were auto-inserted".format(num_auto))
        print("(marked [AUTO] above). These come from two sources:")
        print("  * push/pop expansion into addi+sw / lw+addi, and")
        print("  * branch relaxation (short inverse branch + trampoline j)")
        print("for branches whose target is outside the 4-bit offset range.")
        print("The program's behavior is unchanged; only its length and")
        print("addresses grew. Update your report's instruction count /")
        print("circuit trace accordingly.")

    # ---- Simulate execution to verify $sp (and everything else) for real ----
    trace, final_regs, final_sp, final_mem = simulate(instr_entries, symtab)

    print("\nStack-related trace ($sp changes and lw/sw through $sp):")
    print("{:<3} {:<26} {:<10} {:<10} {:<8} {:<5}".format(
        "PC", "Instr", "sp before", "sp after", "addr", "value"))
    any_stack_ops = False
    for step in trace:
        if step['is_stack'] or step['sp_before'] != step['sp_after']:
            any_stack_ops = True
            addr_s = "0x{:02X}".format(step['addr']) if step['addr'] is not None else "-"
            val_s = str(step['val']) if step['val'] is not None else "-"
            print("{:<3} {:<26} 0x{:X}        0x{:X}        {:<8} {}".format(
                step['pc'], step['text'][:26],
                step['sp_before'], step['sp_after'],
                addr_s, val_s))
    if not any_stack_ops:
        print("  (no instructions in this program touch $sp)")

    print("\nFinal state after simulated execution:")
    print("  registers: " + ", ".join("{}={}".format(r, v) for r, v in final_regs.items()))
    print("  $sp       = 0x{:X}".format(final_sp))
    touched = [(a, v) for a, v in enumerate(final_mem) if v != 0]
    if touched:
        print("  non-zero memory: " + ", ".join("mem[0x{:02X}]={}".format(a, v) for a, v in touched))


if __name__ == '__main__':
    main()