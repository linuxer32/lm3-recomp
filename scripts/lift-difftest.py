#!/usr/bin/env python3
"""Differential test for the homebrew lifter's integer data-processing coverage.

Random encodings from each A64 data-processing class are lifted with
`recomp homebrew-lift`; the resulting IR is evaluated here and compared with
a reference model built on capstone's independent decode (aliases included)
over random register and flag states.

Usage: lift-difftest.py <path/to/recomp> <work-dir> <cases-per-class> <seed>
Requires the `capstone` Python package. Exits non-zero on any mismatch.
"""
import capstone, struct, json, random, subprocess, sys, os, collections
from capstone import arm64 as A

M64 = (1 << 64) - 1
def u(v, w=64): return v & ((1 << w) - 1)
def s(v, w=64):
    v = u(v, w)
    return v - (1 << w) if v >> (w - 1) else v

# ---------------------------------------------------------------- reference
CC = {A.ARM64_CC_EQ:'eq',A.ARM64_CC_NE:'ne',A.ARM64_CC_HS:'cs',A.ARM64_CC_LO:'cc',A.ARM64_CC_MI:'mi',
      A.ARM64_CC_PL:'pl',A.ARM64_CC_VS:'vs',A.ARM64_CC_VC:'vc',A.ARM64_CC_HI:'hi',A.ARM64_CC_LS:'ls',
      A.ARM64_CC_GE:'ge',A.ARM64_CC_LT:'lt',A.ARM64_CC_GT:'gt',A.ARM64_CC_LE:'le',A.ARM64_CC_AL:'al',A.ARM64_CC_NV:'al'}
def cond(c, f):
    n, z, cf, v = f
    return {'eq':z,'ne':not z,'cs':cf,'cc':not cf,'mi':n,'pl':not n,'vs':v,'vc':not v,'hi':cf and not z,
            'ls':not cf or z,'ge':n==v,'lt':n!=v,'gt':(not z) and n==v,'le':z or n!=v,'al':True}[c]
def awc(x, y, c, w):
    us = u(x, w) + u(y, w) + c
    ss = s(x, w) + s(y, w) + c
    r = u(us, w)
    return r, (r >> (w - 1) & 1 == 1, r == 0, r != us, s(r, w) != ss)

def mem_byte(address):
    """Deterministic initial memory contents."""
    return ((address * 2654435761) >> 7) & 0xFF


class Memory:
    def __init__(self):
        self.written = {}
    def read(self, address, size):
        return int.from_bytes(bytes(self.written.get(u(address + i), mem_byte(u(address + i))) for i in range(size)), 'little')
    def write(self, address, size, value):
        for i in range(size):
            self.written[u(address + i)] = (value >> (8 * i)) & 0xFF


VIEW_BYTES = {'q': 16, 'd': 8, 's': 4, 'h': 2, 'b': 1}


class Ref:
    def __init__(self, regs, flags, vregs=None, memory=None):
        self.r = list(regs); self.f = flags
        self.v = list(vregs or [0] * 32); self.mem = memory or Memory()
    def vrd(self, name):
        return self.v[int(name[1:])] & ((1 << (8 * VIEW_BYTES.get(name[0], 16))) - 1)
    def vwr(self, name, value):
        # Scalar views (q/d/s/h/b) zero the rest of the register.
        self.v[int(name[1:])] = value & ((1 << (8 * VIEW_BYTES.get(name[0], 16))) - 1)
    def rd(self, name):
        name = {'fp': 'x29', 'lr': 'x30'}.get(name, name)
        if name in ('xzr', 'wzr'): return 0
        if name in ('sp', 'wsp'): v = self.r[31]
        else: v = self.r[int(name[1:])]
        return u(v, 32) if name[0] == 'w' else v
    def wr(self, name, v):
        name = {'fp': 'x29', 'lr': 'x30'}.get(name, name)
        if name in ('xzr', 'wzr'): return
        idx = 31 if name in ('sp', 'wsp') else int(name[1:])
        self.r[idx] = u(v, 32) if name[0] == 'w' else u(v)

def is_vector(name):
    return name[0] in 'vqdshb' and name[1:].isdigit()


VAS_ESIZE = {A.ARM64_VAS_16B: 1, A.ARM64_VAS_8B: 1, A.ARM64_VAS_1B: 1, A.ARM64_VAS_8H: 2, A.ARM64_VAS_4H: 2,
             A.ARM64_VAS_1H: 2, A.ARM64_VAS_4S: 4, A.ARM64_VAS_2S: 4, A.ARM64_VAS_1S: 4, A.ARM64_VAS_2D: 8,
             A.ARM64_VAS_1D: 8}
VAS_WIDTH = {A.ARM64_VAS_16B: 16, A.ARM64_VAS_8H: 16, A.ARM64_VAS_4S: 16, A.ARM64_VAS_2D: 16,
             A.ARM64_VAS_8B: 8, A.ARM64_VAS_4H: 8, A.ARM64_VAS_2S: 8}


def replicate(value, esize, width):
    out = 0
    for lane in range(width // esize):
        out |= (value & ((1 << (8 * esize)) - 1)) << (8 * esize * lane)
    return out


def ref_simd(ins, st):
    """Reference for SIMD&FP data movement; returns False if not covered."""
    m = ins.mnemonic; ops = ins.operands
    names = [ins.reg_name(o.reg) if o.type == A.ARM64_OP_REG else None for o in ops]
    if not names or names[0] is None:
        return False
    d = names[0]
    if m in ('ldr', 'str', 'ldur', 'stur', 'ldp', 'stp', 'ldnp', 'stnp') and is_vector(d):
        regs = [n for n in names if n and is_vector(n)]
        size = VIEW_BYTES[regs[0][0]]
        mem_op = next(o for o in ops if o.type == A.ARM64_OP_MEM)
        base_name = ins.reg_name(mem_op.mem.base)
        base = st.rd(base_name)
        post = [o for o in ops[ops.index(mem_op) + 1:] if o.type == A.ARM64_OP_IMM]
        if mem_op.mem.index:
            index = st.rd(ins.reg_name(mem_op.mem.index))
            if mem_op.ext: index = ext_val(index, mem_op.ext)
            address = u(base + (index << (mem_op.shift.value if mem_op.shift.type else 0)))
        elif post:
            address = base
        else:
            address = u(base + mem_op.mem.disp)
        for k, reg in enumerate(regs):
            if m.startswith('ld'):
                st.vwr(reg, st.mem.read(address + k * size, size))
            else:
                st.mem.write(address + k * size, size, st.vrd(reg))
        if post:
            st.wr(base_name, base + post[0].imm)
        elif ins.writeback:
            st.wr(base_name, address)
        return True
    if not d.startswith('v') and not is_vector(d) and m not in ('mov', 'umov', 'smov', 'fmov'):
        return False
    if m in ('dup', 'mov') and d.startswith('v') and names[1] and names[1].startswith('v') and ops[1].vector_index >= 0 and ops[0].vector_index < 0:
        esize = VAS_ESIZE[ops[0].vas]
        value = st.v[int(names[1][1:])] >> (8 * esize * ops[1].vector_index)
        st.v[int(d[1:])] = replicate(value, esize, VAS_WIDTH[ops[0].vas]); return True
    if m == 'dup' and ops[1].type == A.ARM64_OP_REG and not is_vector(names[1]):
        esize = VAS_ESIZE[ops[0].vas]
        st.v[int(d[1:])] = replicate(st.rd(names[1]), esize, VAS_WIDTH[ops[0].vas]); return True
    if m in ('mov', 'umov', 'smov') and not is_vector(d) and names[1] and names[1].startswith('v'):
        esize = VAS_ESIZE[ops[1].vas]; lane = ops[1].vector_index
        value = (st.v[int(names[1][1:])] >> (8 * esize * lane)) & ((1 << (8 * esize)) - 1)
        if m == 'smov': value = s(value, 8 * esize)
        st.wr(d, value); return True
    if m == 'mov' and d.startswith('v') and ops[0].vector_index >= 0:
        esize = VAS_ESIZE[ops[0].vas]; lane = ops[0].vector_index
        if names[1].startswith('v'):
            value = (st.v[int(names[1][1:])] >> (8 * esize * ops[1].vector_index)) & ((1 << (8 * esize)) - 1)
        else:
            value = st.rd(names[1]) & ((1 << (8 * esize)) - 1)
        mask = ((1 << (8 * esize)) - 1) << (8 * esize * lane)
        st.v[int(d[1:])] = (st.v[int(d[1:])] & ~mask) | (value << (8 * esize * lane)); return True
    if m in ('movi', 'mvni', 'orr', 'bic') and ops[1].type == A.ARM64_OP_IMM:
        if d.startswith('d'):
            esize, width = 8, 8
        else:
            esize, width = VAS_ESIZE[ops[0].vas], VAS_WIDTH[ops[0].vas]
        value = ops[1].imm & ((1 << 64) - 1)
        if ops[1].shift.type == A.ARM64_SFT_LSL: value <<= ops[1].shift.value
        elif ops[1].shift.type == A.ARM64_SFT_MSL: value = (value << ops[1].shift.value) | ((1 << ops[1].shift.value) - 1)
        if m == 'mvni': value = ~value
        value = replicate(value, esize, width)
        idx = int(d[1:])
        if m in ('movi', 'mvni'): st.v[idx] = value
        elif m == 'orr': st.v[idx] = (st.v[idx] | value) & ((1 << (8 * width)) - 1)
        else: st.v[idx] = (st.v[idx] & ~value) & ((1 << (8 * width)) - 1)
        return True
    if m == 'fmov':
        if ops[1].type == A.ARM64_OP_FP:
            if d.startswith('v'):
                esize = VAS_ESIZE[ops[0].vas]
                bits = struct.unpack('<I', struct.pack('<f', ops[1].fp))[0] if esize == 4 else struct.unpack('<Q', struct.pack('<d', ops[1].fp))[0]
                st.v[int(d[1:])] = replicate(bits, esize, VAS_WIDTH[ops[0].vas])
            else:
                fmt = {'s': ('<f', '<I'), 'd': ('<d', '<Q'), 'h': ('<e', '<H')}[d[0]]
                st.vwr(d, struct.unpack(fmt[1], struct.pack(fmt[0], ops[1].fp))[0])
            return True
        src = names[1]
        if src.startswith('v'):
            value = st.v[int(src[1:])] >> 64 if ops[1].vector_index == 1 else st.v[int(src[1:])]
            value &= (1 << 64) - 1
        elif is_vector(src):
            value = st.vrd(src)
        else:
            value = st.rd(src)
        if d.startswith('v'):
            st.v[int(d[1:])] = (st.v[int(d[1:])] & ((1 << 64) - 1)) | ((value & ((1 << 64) - 1)) << 64)
        elif is_vector(d):
            st.vwr(d, value)
        else:
            st.wr(d, value & ((1 << (8 * min(VIEW_BYTES.get(src[0], 8), 8))) - 1))
        return True
    if m in ('and', 'orr', 'eor', 'bic', 'orn', 'mov', 'bsl', 'bit', 'bif') and d.startswith('v'):
        width = VAS_WIDTH[ops[0].vas]; full = (1 << (8 * width)) - 1
        vd = st.v[int(d[1:])]; vn = st.v[int(names[1][1:])]
        vm = st.v[int(names[2][1:])] if len(names) > 2 else vn
        result = {'and': lambda: vn & vm, 'orr': lambda: vn | vm, 'mov': lambda: vn, 'eor': lambda: vn ^ vm,
                  'bic': lambda: vn & ~vm, 'orn': lambda: vn | ~vm, 'bsl': lambda: (vd & vn) | (~vd & vm),
                  'bit': lambda: (vd & ~vm) | (vn & vm), 'bif': lambda: (vd & vm) | (vn & ~vm)}[m]()
        st.v[int(d[1:])] = result & full
        return True
    return False


class Unpredictable(Exception):
    """The architecture allows more than one result (e.g. Rt == Rn with
    writeback); such cases are skipped."""


def ref_mem(ins, st):
    """Reference for integer loads/stores; returns False if not covered."""
    m = ins.mnemonic; ops = ins.operands
    if not (m.startswith('ld') or m.startswith('st') or m.startswith('prf')) or m.startswith('ldx') or m.startswith('stx'):
        return False
    if m.startswith('prf'):
        return True
    names = [{'fp': 'x29', 'lr': 'x30'}.get(ins.reg_name(o.reg), ins.reg_name(o.reg)) if o.type == A.ARM64_OP_REG else None for o in ops]
    regs = [n for n in names if n]
    if any(is_vector(n) for n in regs):
        return False
    size = {'b': 1, 'h': 2}.get(m.rstrip('p')[-1], None)
    signed = 'ldrs' in m or 'ldurs' in m or 'ldtrs' in m or m == 'ldpsw'
    if m == 'ldpsw' or m in ('ldrsw', 'ldursw', 'ldtrsw'): size = 4
    if size is None:
        size = 8 if regs[0][0] == 'x' else 4
    mem_op = next(o for o in ops if o.type == A.ARM64_OP_MEM)
    base_name = ins.reg_name(mem_op.mem.base)
    base = st.rd(base_name)
    post = [o for o in ops[ops.index(mem_op) + 1:] if o.type == A.ARM64_OP_IMM]
    if mem_op.mem.index:
        index = st.rd(ins.reg_name(mem_op.mem.index))
        if mem_op.ext: index = ext_val(index, mem_op.ext)
        address = u(base + (index << (mem_op.shift.value if mem_op.shift.type else 0)))
    elif post:
        address = base
    else:
        address = u(base + mem_op.mem.disp)
    data_regs = [n for n in names[:ops.index(mem_op)] if n]
    base_alias = {'fp': 'x29', 'lr': 'x30'}.get(base_name, base_name)
    if (post or ins.writeback) and any(reg[1:] == base_alias[1:] and reg[1:].isdigit() for reg in data_regs):
        raise Unpredictable()
    for k, reg in enumerate(data_regs):
        if m.startswith('ld'):
            value = st.mem.read(address + k * size, size)
            if signed: value = s(value, 8 * size)
            st.wr(reg, value)
        else:
            st.mem.write(address + k * size, size, st.rd(reg))
    if post:
        st.wr(base_name, base + post[0].imm)
    elif ins.writeback:
        st.wr(base_name, address)
    return True


def width_of(name): return 32 if name[0] == 'w' and name != 'wzr' or name in ('wzr', 'wsp') else 64
def shift_val(v, typ, n, w):
    if typ in (0, A.ARM64_SFT_INVALID) or n == 0 and typ != A.ARM64_SFT_ROR: return u(v, w)
    if typ == A.ARM64_SFT_LSL: return u(v << n, w)
    if typ == A.ARM64_SFT_LSR: return u(v, w) >> n
    if typ == A.ARM64_SFT_ASR: return u(s(v, w) >> n, w)
    if typ == A.ARM64_SFT_ROR: v = u(v, w); n %= w; return u((v >> n) | (v << (w - n)), w)
    raise ValueError('shift')
def ext_val(v, ext):
    bits = {1:8,2:16,3:32,4:64,5:8,6:16,7:32,8:64}[ext]
    return u(s(v, bits)) if ext >= 5 else u(v, bits)

def operand(st, ins, op, w):
    if op.type == A.ARM64_OP_IMM:
        v = op.imm
        if op.shift.type == A.ARM64_SFT_LSL: v <<= op.shift.value
        return u(v)
    name = ins.reg_name(op.reg)
    v = st.rd(name)
    if op.ext:
        v = ext_val(v, op.ext)
        return u(v << op.shift.value) if op.shift.type else v
    return shift_val(v, op.shift.type, op.shift.value, w)

def ref_exec(ins, st):
    if ref_simd(ins, st) or ref_mem(ins, st):
        return
    m = ins.mnemonic; ops = ins.operands
    names = [ins.reg_name(o.reg) if o.type == A.ARM64_OP_REG else None for o in ops]
    w = width_of(names[0]) if names and names[0] else 64
    val = lambda i, ww=None: operand(st, ins, ops[i], ww or w)
    imm = lambda i: ops[i].imm
    cc = CC.get(ins.cc)
    def setres(v): st.wr(names[0], v)
    if m in ('nop', 'yield', 'dmb', 'dsb', 'isb', 'clrex', 'hint'): return
    if m in ('add', 'adds', 'sub', 'subs', 'cmp', 'cmn', 'neg', 'negs', 'ngc'):
        if m in ('cmp', 'cmn'): a, b = val(0), val(1)
        elif m in ('neg', 'negs'): a, b = 0, val(1)
        else: a, b = val(1), val(2)
        sub = m in ('sub', 'subs', 'cmp', 'neg', 'negs')
        r, fl = awc(a, u(~b, w) if sub else b, 1 if sub else 0, w)
        if m in ('adds', 'subs', 'cmp', 'cmn', 'negs'): st.f = fl
        if m not in ('cmp', 'cmn'): setres(r)
        return
    if m == 'mov':
        setres(val(1)); return
    if m in ('movz', 'movn', 'movk'):
        v = ops[1].imm << (ops[1].shift.value if ops[1].shift.type else 0)
        if m == 'movz': setres(v)
        elif m == 'movn': setres(~v)
        else:
            sh = ops[1].shift.value if ops[1].shift.type else 0
            setres((st.rd(names[0]) & ~(0xFFFF << sh)) | v)
        return
    if m in ('and', 'ands', 'orr', 'eor', 'bic', 'bics', 'orn', 'eon', 'tst', 'mvn'):
        if m == 'tst': a, b = val(0), val(1)
        elif m == 'mvn': a, b = 0, val(1)
        else: a, b = val(1), val(2)
        if m in ('bic', 'bics', 'orn', 'eon', 'mvn'): b = u(~b, w)
        r = {'and':a&b,'ands':a&b,'tst':a&b,'bic':a&b,'bics':a&b,'orr':a|b,'orn':a|b,'mvn':a|b,'eor':a^b,'eon':a^b}[m]
        r = u(r, w)
        if m in ('ands', 'bics', 'tst'): st.f = (r >> (w - 1) & 1 == 1, r == 0, False, False)
        if m != 'tst': setres(r)
        return
    if m in ('lsl', 'lsr', 'asr', 'ror'):
        x = st.rd(names[1])
        n = imm(2) if ops[2].type == A.ARM64_OP_IMM else st.rd(names[2]) % w
        typ = {'lsl':A.ARM64_SFT_LSL,'lsr':A.ARM64_SFT_LSR,'asr':A.ARM64_SFT_ASR,'ror':A.ARM64_SFT_ROR}[m]
        setres(shift_val(x, typ, n, w) if n else u(x, w)); return
    if m in ('sbfx', 'ubfx', 'sbfiz', 'ubfiz', 'bfi', 'bfxil', 'bfc'):
        if m == 'bfc': x, lsb, wid = 0, imm(1), imm(2)
        else: x, lsb, wid = st.rd(names[1]), imm(2), imm(3)
        mask = (1 << wid) - 1
        if m in ('sbfx', 'ubfx', 'bfxil'): field = (u(x, w) >> lsb) & mask; pos = 0
        else: field = x & mask; pos = lsb
        if m in ('sbfx', 'sbfiz'): setres(u(s(field, wid) << pos, w)); return
        if m in ('ubfx', 'ubfiz'): setres(u(field << pos, w)); return
        d = st.rd(names[0]); setres((d & ~(mask << pos)) | (field << pos)); return
    if m in ('sxtb', 'sxth', 'sxtw', 'uxtb', 'uxth'):
        bits = {'b':8,'h':16,'w':32}[m[-1]]
        x = st.rd(names[1])
        setres(u(s(x, bits), w) if m[0] == 's' else u(x, bits)); return
    if m == 'extr':
        hi, lo, lsb = st.rd(names[1]), st.rd(names[2]), imm(3)
        setres(u(((u(hi, w) << w) | u(lo, w)) >> lsb, w)); return
    if m in ('csel', 'csinc', 'csinv', 'csneg'):
        t = cond(cc, st.f)
        a, b = st.rd(names[1]), st.rd(names[2])
        if not t: b = {'csel':b,'csinc':b+1,'csinv':~b,'csneg':-b}[m]
        setres(a if t else b); return
    if m in ('cset', 'csetm'):
        t = cond(cc, st.f); setres((1 if m == 'cset' else -1) if t else 0); return
    if m in ('cinc', 'cinv', 'cneg'):
        t = cond(cc, st.f); a = st.rd(names[1])
        setres({'cinc':a+1,'cinv':~a,'cneg':-a}[m] if t else a); return
    if m in ('ccmp', 'ccmn'):
        if cond(cc, st.f):
            a, b = val(0), val(1)
            _, st.f = awc(a, u(~b, w), 1, w) if m == 'ccmp' else awc(a, b, 0, w)
        else:
            nz = imm(2); st.f = (bool(nz & 8), bool(nz & 4), bool(nz & 2), bool(nz & 1))
        return
    if m in ('madd', 'msub', 'mul', 'mneg'):
        a, b = st.rd(names[1]), st.rd(names[2])
        acc = st.rd(names[3]) if m in ('madd', 'msub') else 0
        p = a * b
        setres(acc - p if m in ('msub', 'mneg') else acc + p); return
    if m in ('smaddl', 'smsubl', 'umaddl', 'umsubl', 'smull', 'umull', 'smnegl', 'umnegl'):
        sg = m[0] == 's'
        a, b = st.rd(names[1]), st.rd(names[2])
        a, b = (s(a, 32), s(b, 32)) if sg else (u(a, 32), u(b, 32))
        acc = st.rd(names[3]) if m.endswith('addl') or m.endswith('subl') else 0
        p = a * b
        setres(acc - p if 'sub' in m or 'neg' in m else acc + p); return
    if m in ('smulh', 'umulh'):
        a, b = st.rd(names[1]), st.rd(names[2])
        setres(((s(a) * s(b)) >> 64) if m == 'smulh' else ((a * b) >> 64)); return
    if m in ('udiv', 'sdiv'):
        a, b = st.rd(names[1]), st.rd(names[2])
        if m == 'udiv': setres(0 if b == 0 else u(a, w) // u(b, w)); return
        sa, sb = s(a, w), s(b, w)
        if sb == 0: setres(0); return
        q = abs(sa) // abs(sb)
        setres(q if (sa < 0) == (sb < 0) else -q); return
    if m in ('rbit', 'rev', 'rev16', 'rev32', 'clz', 'cls'):
        x = u(st.rd(names[1]), w)
        if m == 'rbit': setres(int(format(x, f'0{w}b')[::-1], 2)); return
        if m == 'clz': setres(w - x.bit_length()); return
        if m == 'cls':
            sx = s(x, w); y = ~sx if sx < 0 else sx
            setres(w - 1 - y.bit_length()); return
        by = x.to_bytes(w // 8, 'little')
        chunk = {'rev':w//8,'rev16':2,'rev32':4}[m]
        out = b''.join(by[i:i+chunk][::-1] for i in range(0, len(by), chunk))
        setres(int.from_bytes(out, 'little')); return
    raise KeyError(m)

# ---------------------------------------------------------------- IR eval
def ir_exec(ops, regs, flags, vregs=None, memory=None):
    env = collections.defaultdict(int)
    for i in range(32): env[f'x{i}'] = regs[i]
    for i in range(32): env[f'v{i}'] = (vregs or [0] * 32)[i]
    memory = memory or Memory()
    f = list(flags)
    g = lambda n: u(env[n])
    vg = lambda n: env[n] & ((1 << 128) - 1)
    lane = lambda b: (1 << (8 * b)) - 1
    for op in ops:
        k = op['op']
        if k in ('load_i8', 'load_i16', 'load_i32', 'load_i64'):
            size = int(k[6:]) // 8
            env[op['dst']] = memory.read(u(g(op['addr']) + op.get('offset', 0)), size)
        elif k in ('store_i8', 'store_i16', 'store_i32', 'store_i64'):
            size = int(k[7:]) // 8
            memory.write(u(g(op['addr']) + op.get('offset', 0)), size, g(op['src']))
        elif k == 'v_const': env[op['dst']] = (u(op['hi']) << 64) | u(op['lo'])
        elif k == 'v_load': env[op['dst']] = memory.read(u(g(op['addr']) + op.get('offset', 0)), op['bytes'])
        elif k == 'v_store':
            memory.write(u(g(op['addr']) + op.get('offset', 0)), op['bytes'], vg(op['src']) & lane(op['bytes']))
        elif k == 'v_mov': env[op['dst']] = vg(op['src']) & lane(min(op['bytes'], 16))
        elif k == 'v_dup': env[op['dst']] = replicate(g(op['src']), op['esize'], 16 if op['q'] else 8)
        elif k == 'v_ins_gpr':
            sh = 8 * op['esize'] * op['index']; mk = lane(op['esize'])
            env[op['dst']] = (vg(op['dst']) & ~(mk << sh)) | ((g(op['src']) & mk) << sh)
        elif k == 'v_extract':
            value = (vg(op['src']) >> (8 * op['esize'] * op['index'])) & lane(op['esize'])
            env[op['dst']] = u(s(value, 8 * op['esize']) if op['signed'] else value)
        elif k == 'v_ins_lane':
            b = 8 * op['esize']; mk = lane(op['esize'])
            value = (vg(op['src']) >> (b * op['src_index'])) & mk
            env[op['dst']] = (vg(op['dst']) & ~(mk << (b * op['dst_index']))) | (value << (b * op['dst_index']))
        elif k == 'v_bitwise':
            a, b = vg(op['lhs']), vg(op['rhs'])
            r = {'and': a & b, 'orr': a | b, 'eor': a ^ b, 'bic': a & ~b, 'orn': a | ~b}[op['kind']]
            env[op['dst']] = r & lane(16 if op['q'] else 8)
        elif k == 'v_bit_select':
            mk, st_, cl = vg(op['mask']), vg(op['set']), vg(op['clear'])
            env[op['dst']] = ((mk & st_) | (~mk & cl)) & lane(16 if op['q'] else 8)
        elif k == 'const_i64': env[op['dst']] = u(op['imm'])
        elif k == 'mov_i64': env[op['dst']] = g(op['src'])
        elif k in ('add_i64','sub_i64','and_i64','or_i64','xor_i64','mul_i64','lsl_i64','lsr_i64','asr_i64',
                   'mul_hi_u64','mul_hi_s64','udiv_i64','sdiv_i64'):
            a, b = g(op['lhs']), g(op['rhs'])
            r = {'add_i64':lambda:a+b,'sub_i64':lambda:a-b,'and_i64':lambda:a&b,'or_i64':lambda:a|b,
                 'xor_i64':lambda:a^b,'mul_i64':lambda:a*b,'lsl_i64':lambda:a<<(b&63),
                 'lsr_i64':lambda:a>>(b&63),'asr_i64':lambda:s(a)>>(b&63),
                 'mul_hi_u64':lambda:(a*b)>>64,'mul_hi_s64':lambda:(s(a)*s(b))>>64,
                 'udiv_i64':lambda:0 if b==0 else a//b,
                 'sdiv_i64':lambda:0 if b==0 else (abs(s(a))//abs(s(b)))*(1 if (s(a)<0)==(s(b)<0) else -1)}[k]()
            env[op['dst']] = u(r)
        elif k == 'cmp_i64':
            a, b = g(op['lhs']), g(op['rhs']); r, fl = awc(a, u(~b), 1, 64); f = list(fl)
        elif k == 'cmn_i64':
            a, b = g(op['lhs']), g(op['rhs']); r, fl = awc(a, b, 0, 64); f = list(fl)
        elif k == 'test_i64':
            r = g(op['lhs']) & g(op['rhs']); f = [r >> 63 == 1, r == 0, False, False]
        elif k == 'select_i64':
            env[op['dst']] = g(op['lhs']) if cond(op['cond'], f) else g(op['rhs'])
        elif k == 'clz_i64': env[op['dst']] = 64 - g(op['src']).bit_length()
        elif k == 'rbit_i64': env[op['dst']] = int(format(g(op['src']), '064b')[::-1], 2)
        elif k == 'rev_i64': env[op['dst']] = int.from_bytes(g(op['src']).to_bytes(8, 'little'), 'big')
        elif k == 'read_nzcv':
            env[op['dst']] = (f[0] << 31) | (f[1] << 30) | (f[2] << 29) | (f[3] << 28)
        elif k == 'write_nzcv':
            v = g(op['src']); f = [bool(v >> 31 & 1), bool(v >> 30 & 1), bool(v >> 29 & 1), bool(v >> 28 & 1)]
        else: raise KeyError(k)
    return [g(f'x{i}') for i in range(32)], tuple(f), [vg(f'v{i}') for i in range(32)], memory.written

# ---------------------------------------------------------------- driver
CLASSES = {
    'addsub_imm':   (0x1F << 24, 0b10001 << 24),
    'addsub_shift': ((0x1F << 24) | (1 << 21), 0b01011 << 24),
    'addsub_ext':   ((0x1F << 24) | (1 << 21), (0b01011 << 24) | (1 << 21)),
    'logic_imm':    (0x3F << 23, 0b100100 << 23),
    'logic_reg':    (0x1F << 24, 0b01010 << 24),
    'movwide':      (0x3F << 23, 0b100101 << 23),
    'bitfield':     (0x3F << 23, 0b100110 << 23),
    'extr':         (0x3F << 23, 0b100111 << 23),
    'csel':         (0xFF << 21, 0b11010100 << 21),
    'ccmp':         (0xFF << 21, 0b11010010 << 21),
    'dp3':          (0x1F << 24, 0b11011 << 24),
    'dp2':          ((1 << 30) | (0xFF << 21), 0b11010110 << 21),
    'dp1':          ((1 << 30) | (0xFF << 21), (1 << 30) | (0b11010110 << 21)),
    'ldst_imm':     (0x3F << 24, 0x39 << 24),
    'ldst_unscaled':((0x3F << 24) | (1 << 21), 0x38 << 24),
    'ldst_regoff':  ((0x3F << 24) | (1 << 21) | (3 << 10), (0x38 << 24) | (1 << 21) | (2 << 10)),
    'ldst_pair':    ((0x1F << 25), 0b10100 << 25),
    'simd_ldst_imm':(0x3F << 24, 0x3D << 24),
    'simd_ldst_unsc':((0x3F << 24) | (1 << 21), 0x3C << 24),
    'simd_ldst_reg':((0x3F << 24) | (1 << 21) | (3 << 10), (0x3C << 24) | (1 << 21) | (2 << 10)),
    'simd_pair':    ((0x1F << 25), 0b10110 << 25),
    'simd_copy':    (0x9FE08400, 0x0E000400),
    'simd_modimm':  (0x9FF80C00, 0x0F000400),
    'fmov_general': (0x7F26FC00, 0x1E260000),
    'fmov_reg':     (0xFF3FFC00, 0x1E204000),
    'fmov_imm':     (0xFF201FE0, 0x1E201000),
    'simd_logic':   (0x9F20FC00, 0x0E201C00),
}

def main(recomp, workdir, per_class, seed):
    rng = random.Random(seed)
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM); md.detail = True
    cases = []
    for cls, (mask, val) in CLASSES.items():
        got = tries = 0
        while got < per_class and tries < per_class * 50:
            tries += 1
            w = (rng.getrandbits(32) & ~mask) | val
            ins = list(md.disasm(struct.pack('<I', w), 0))
            if not ins: continue
            cases.append((cls, w)); got += 1
    n = len(cases)
    pairs = (4 * n + 0xFFF) & ~0xFFF
    text = bytearray(pairs + 8 * n)
    for i, (_, w) in enumerate(cases):
        tgt = pairs + 8 * i
        struct.pack_into('<I', text, 4 * i, 0x94000000 | (((tgt - 4 * i) >> 2) & 0x3FFFFFF))
        struct.pack_into('<II', text, tgt, w, 0xD65F03C0)
    os.makedirs(workdir, exist_ok=True)
    seg = os.path.join(workdir, 'text.bin'); open(seg, 'wb').write(text)
    mod = {"schema_version": "1", "module_type": "homebrew", "modules": [{
        "name": "diff", "format": "nro", "input_path": "diff.nro", "input_sha256": "a" * 64,
        "input_size": len(text), "build_id": "diff", "segments": [{"name": "text", "file_offset": 0,
        "file_size": len(text), "memory_offset": 0, "memory_size": len(text), "permissions": "r-x",
        "output_path": "text.bin"}], "bss": {"size": 0, "memory_offset": 0}}]}
    json.dump(mod, open(os.path.join(workdir, 'module.json'), 'w'))
    subprocess.run([recomp, 'homebrew-lift', '--module-json', os.path.join(workdir, 'module.json'),
                    '--out-dir', os.path.join(workdir, 'lifted'), '--entry-address', f'{pairs:x}'],
                   check=True, capture_output=True)
    lifted = json.load(open(os.path.join(workdir, 'lifted', 'module.json')))
    by_addr = {}
    for fn in lifted['functions']:
        addr = pairs if fn['name'] == 'entry' else int(fn['name'].rsplit('_', 1)[1], 16)
        by_addr[addr] = fn
    stats = collections.defaultdict(collections.Counter)
    failures = collections.defaultdict(list)
    for i, (cls, w) in enumerate(cases):
        ins = next(md.disasm(struct.pack('<I', w), 0))
        fn = by_addr.get(pairs + 8 * i)
        block = fn['blocks'][0] if fn else None
        if not block or block['terminator']['op'] == 'trap':
            stats[cls]['trap:' + ins.mnemonic] += 1; continue
        for trial in range(4):
            regs = [rng.getrandbits(64) for _ in range(32)]
            if trial == 1: regs = [rng.choice([0, 1, M64, 1 << 63, (1 << 63) - 1, 0x80000000, 0xFFFFFFFF, 0x7FFFFFFF]) for _ in range(32)]
            flags = tuple(rng.random() < 0.5 for _ in range(4))
            vregs = [rng.getrandbits(128) for _ in range(32)]
            ref = Ref(regs, flags, vregs, Memory())
            try: ref_exec(ins, ref)
            except Unpredictable:
                stats[cls]['skip_unpredictable'] += 1; break
            except KeyError as e:
                stats[cls]['no_ref:' + str(e)] += 1; break
            got_regs, got_flags, got_v, got_mem = ir_exec(block['ops'], regs, flags, vregs, Memory())
            ref_flags = tuple(bool(x) for x in ref.f)
            if got_regs != ref.r or tuple(got_flags) != ref_flags or got_v != ref.v or got_mem != ref.mem.written:
                stats[cls]['FAIL'] += 1
                diff = [f'x{j}: want {ref.r[j]:#x} got {got_regs[j]:#x}' for j in range(32) if got_regs[j] != ref.r[j]]
                if tuple(got_flags) != ref_flags: diff.append(f'nzcv want {ref_flags} got {tuple(got_flags)}')
                diff += [f'v{j}: want {ref.v[j]:#x} got {got_v[j]:#x}' for j in range(32) if got_v[j] != ref.v[j]]
                if got_mem != ref.mem.written: diff.append(f'mem want {sorted(ref.mem.written.items())[:4]} got {sorted(got_mem.items())[:4]}')
                failures[cls].append(f'{w:#010x} {ins.mnemonic} {ins.op_str}: ' + '; '.join(diff[:3]))
                break
        else:
            stats[cls]['ok'] += 1
    for cls in CLASSES:
        print(f'{cls:13} ' + ' '.join(f'{k}={v}' for k, v in sorted(stats[cls].items())))
    for cls, items in failures.items():
        print(f'--- {cls} failures ({len(items)}):')
        for line in items[:6]: print('   ', line)
    return sum(len(v) for v in failures.values())

if __name__ == '__main__':
    sys.exit(1 if main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])) else 0)
