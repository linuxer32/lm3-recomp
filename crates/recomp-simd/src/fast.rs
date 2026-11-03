//! Compile-time specialized entry points for the floating-point and Advanced
//! SIMD instructions recompiled code runs most.
//!
//! [`exec`](super::exec) decodes its instruction word on every call, which
//! makes floating-point code many times slower than integer code. For the
//! classes below, [`specialize`] decodes the word once, when the code is
//! generated, into a call of one of this module's functions with the
//! decoded fields as constants. Each function computes exactly what `exec`
//! does for those words, with the same helpers (the tests compare them on
//! random words and values); anything `specialize` does not claim still
//! goes through `exec`.

use super::*;

/// One argument of a specialized call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arg {
    /// The value of vector register `n` (u128).
    Vector(u8),
    /// The general register the instruction reads (u64).
    General,
    /// The NZCV flags, N in bit 3 (u8).
    Flags,
    Int(u32),
    Bool(bool),
    /// A whole result known when the code is generated (u128).
    Constant(u128),
}

/// What a specialized call returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// The destination's new value, as [`Outcome::value`] (u128).
    Value,
    /// The new NZCV flags (u8).
    Flags,
}

/// A call of `function` in this module that computes an instruction's
/// result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub function: &'static str,
    pub args: Vec<Arg>,
    pub output: Output,
}

fn call(function: &'static str, args: Vec<Arg>) -> Option<Call> {
    Some(Call {
        function,
        args,
        output: Output::Value,
    })
}

/// The specialized call that computes `word`, if there is one.
pub fn specialize(word: u32) -> Option<Call> {
    if word & 0x5E00_0000 == 0x1E00_0000 {
        return specialize_scalar(word);
    }
    // The cryptographic forms share these groups; none is specialized.
    if word & 0xFFFF_CC00 == 0x4E28_4800
        || word & 0xFFE0_8C00 == 0x5E00_0000
        || word & 0xFFFF_CC00 == 0x5E28_0800
    {
        return None;
    }
    match word & 0xDF00_0000 {
        0x0E00_0000 | 0x4E00_0000 => specialize_three_reg(word),
        0x5E00_0000 => specialize_scalar_simd(word).or_else(|| specialize_dup_scalar(word)),
        0x0F00_0000 | 0x4F00_0000 => {
            specialize_element(word, false).or_else(|| specialize_shift_immediate(word, false))
        }
        0x5F00_0000 => {
            specialize_element(word, true).or_else(|| specialize_shift_immediate(word, true))
        }
        _ => None,
    }
}

/// Scalar Advanced SIMD three-register operations, and the scalar two-register
/// forms (conversions, estimates, compares with zero) that `exec` evaluates.
///
/// The three-register FABD `Sd/Dd` has its own host path in
/// [`scalar_fp2`]; the vector forms are covered by [`specialize_three_reg`].
fn specialize_scalar_simd(word: u32) -> Option<Call> {
    if !bit(word, 21) {
        return None;
    }
    if bit(word, 10) {
        if !bit(word, 29) || !bit(word, 30) {
            return None;
        }
        if bits(word, 15, 11) != 0b11010 {
            return None;
        }
        let size = bits(word, 23, 22);
        if size < 2 {
            return None;
        }
        let (_, rn, rm) = regs(word);
        return call(
            "scalar_fp2",
            vec![
                Arg::Int(9),
                Arg::Bool(size == 3),
                Arg::Vector(rn as u8),
                Arg::Vector(rm as u8),
            ],
        );
    }
    if !bit(word, 11) || bits(word, 20, 17) != 0 {
        return None;
    }
    specialize_two_misc(word, true)
}

/// 32 or 64 for single and double precision; half precision is left to
/// `exec`.
fn single_or_double(ftype: u32) -> Option<bool> {
    match ftype {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

fn size_of(double: bool) -> u32 {
    if double {
        64
    } else {
        32
    }
}

fn specialize_scalar(word: u32) -> Option<Call> {
    if bit(word, 29) {
        return None;
    }
    let (_, rn, rm) = regs(word);
    let (rn, rm) = (rn as u8, rm as u8);
    let ftype = bits(word, 23, 22);
    if bit(word, 24) {
        if bit(word, 31) {
            return None;
        }
        let double = single_or_double(ftype)?;
        let ra = bits(word, 14, 10) as u8;
        let negate = (bits(word, 21, 21) << 1) | bits(word, 15, 15);
        return call(
            "scalar_fma",
            vec![
                Arg::Int(negate),
                Arg::Bool(double),
                Arg::Vector(ra),
                Arg::Vector(rn),
                Arg::Vector(rm),
            ],
        );
    }
    if !bit(word, 21) {
        return None;
    }
    match bits(word, 11, 10) {
        0b01 => {
            // FCCMP, FCCMPE
            if bit(word, 31) {
                return None;
            }
            let double = single_or_double(ftype)?;
            Some(Call {
                function: "scalar_fccmp",
                args: vec![
                    Arg::Bool(double),
                    Arg::Int(bits(word, 15, 12)),
                    Arg::Flags,
                    Arg::Vector(rn),
                    Arg::Vector(rm),
                    Arg::Int(bits(word, 3, 0)),
                ],
                output: Output::Flags,
            })
        }
        0b10 => {
            if bit(word, 31) || bits(word, 15, 12) > 8 {
                return None;
            }
            let double = single_or_double(ftype)?;
            call(
                "scalar_fp2",
                vec![
                    Arg::Int(bits(word, 15, 12)),
                    Arg::Bool(double),
                    Arg::Vector(rn),
                    Arg::Vector(rm),
                ],
            )
        }
        0b11 => {
            if bit(word, 31) {
                return None;
            }
            let double = single_or_double(ftype)?;
            call(
                "scalar_select",
                vec![
                    Arg::Bool(double),
                    Arg::Int(bits(word, 15, 12)),
                    Arg::Flags,
                    Arg::Vector(rn),
                    Arg::Vector(rm),
                ],
            )
        }
        0b00 => {
            if bits(word, 15, 10) == 0 {
                return specialize_int_conversion(word);
            }
            if bit(word, 31) {
                return None;
            }
            if bits(word, 14, 10) == 0b10000 {
                let opcode = bits(word, 20, 15);
                single_or_double(ftype)?;
                let valid = if opcode & 0b11_1100 == 0b00_0100 {
                    // FCVT between single and double.
                    matches!((ftype, opcode & 3), (0, 1) | (1, 0))
                } else {
                    matches!(opcode, 0..=3 | 8..=12 | 14 | 15)
                };
                if !valid {
                    return None;
                }
                return call(
                    "scalar_fp1",
                    vec![Arg::Int(opcode), Arg::Int(ftype), Arg::Vector(rn)],
                );
            }
            if bits(word, 13, 10) == 0b1000 {
                if bits(word, 15, 14) != 0 || bits(word, 2, 0) != 0 {
                    return None;
                }
                let double = single_or_double(ftype)?;
                let m = if bit(word, 3) {
                    Arg::Constant(0)
                } else {
                    Arg::Vector(rm)
                };
                return Some(Call {
                    function: "scalar_compare",
                    args: vec![Arg::Bool(double), Arg::Vector(rn), m],
                    output: Output::Flags,
                });
            }
            if bits(word, 12, 10) == 0b100 && rn == 0 {
                let double = single_or_double(ftype)?;
                let value = vfp_expand_imm(bits(word, 20, 13), size_of(double));
                return call("constant", vec![Arg::Constant(u128::from(value))]);
            }
            None
        }
        _ => None,
    }
}

fn specialize_int_conversion(word: u32) -> Option<Call> {
    let (_, rn, _) = regs(word);
    let sf = bit(word, 31);
    let ftype = bits(word, 23, 22);
    let (rmode, opcode) = (bits(word, 20, 19), bits(word, 18, 16));
    match (ftype, sf, rmode, opcode) {
        // FMOV from a vector register to a general one: the low 32 or 64
        // bits.
        (0, false, 0, 0b110) => call("low_bits", vec![Arg::Int(32), Arg::Vector(rn as u8)]),
        (1, true, 0, 0b110) => call("low_bits", vec![Arg::Int(64), Arg::Vector(rn as u8)]),
        // FMOV from a general register.
        (0, false, 0, 0b111) => call("from_general", vec![Arg::Int(32), Arg::General]),
        (1, true, 0, 0b111) => call("from_general", vec![Arg::Int(64), Arg::General]),
        (0 | 1, _, 0, 0b010 | 0b011) => call(
            "int_to_float",
            vec![
                Arg::Bool(ftype == 1),
                Arg::Bool(opcode == 0b011),
                Arg::Bool(sf),
                Arg::General,
            ],
        ),
        (0 | 1, _, _, 0b000 | 0b001) | (0 | 1, _, 0, 0b100 | 0b101) => call(
            "float_to_int",
            vec![
                Arg::Bool(ftype == 1),
                Arg::Int((rmode << 3) | opcode),
                Arg::Bool(sf),
                Arg::Vector(rn as u8),
            ],
        ),
        _ => None,
    }
}

fn specialize_three_reg(word: u32) -> Option<Call> {
    let (_, rn, rm) = regs(word);
    let (rn, rm) = (rn as u8, rm as u8);
    let q = bit(word, 30);
    if bit(word, 21) {
        if bit(word, 10) {
            return specialize_three_same(word);
        }
        if !bit(word, 11) {
            // Three different: widening multiplies and adds are not
            // specialized yet.
            return None;
        }
        return match bits(word, 20, 17) {
            0b0000 => specialize_two_misc(word, false),
            0b1000 => specialize_across_lanes(word),
            _ => None,
        };
    }
    if word & 0xBF20_8C00 == 0x0E00_0800 {
        // ZIP, UZP, TRN (opcodes 0 and 4 are unallocated, 64-bit lanes
        // need a 128-bit vector).
        let (size, op) = (bits(word, 23, 22), bits(word, 14, 12));
        if op & 3 == 0 || (size == 3 && !q) {
            return None;
        }
        return call(
            "permute",
            vec![
                Arg::Int(op),
                Arg::Int(8 << size),
                Arg::Bool(q),
                Arg::Vector(rn),
                Arg::Vector(rm),
            ],
        );
    }
    if word & 0xBFE0_8400 == 0x2E00_0000 {
        let position = bits(word, 14, 11);
        if position >= if q { 16 } else { 8 } {
            return None;
        }
        return call(
            "extract",
            vec![
                Arg::Bool(q),
                Arg::Int(position),
                Arg::Vector(rn),
                Arg::Vector(rm),
            ],
        );
    }
    None
}

/// Three-same vector operations: integer add/subtract/compare/... and the
/// floating-point forms.
fn specialize_three_same(word: u32) -> Option<Call> {
    let (rd, rn, rm) = regs(word);
    let (rd, rn, rm) = (rd as u8, rn as u8, rm as u8);
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 15, 11),
    );
    if opcode == 0b00011 {
        // AND, BIC, ORR, ORN, EOR, BSL, BIT, BIF.
        let kind = (u32::from(u) << 2) | size;
        return call(
            "vector_bitwise",
            vec![
                Arg::Int(kind),
                Arg::Bool(q),
                if u && size != 0 {
                    Arg::Vector(rd)
                } else {
                    Arg::Constant(0)
                },
                Arg::Vector(rn),
                Arg::Vector(rm),
            ],
        );
    }
    let esize = 8 << size;
    if opcode >= 0b11000 {
        // Three same, floating point.
        let kind = (u32::from(u) << 6) | ((size >> 1) << 5) | opcode;
        let accumulates = match vector_kind(kind) {
            Some(VectorKind::Fma(_)) => true,
            Some(_) => false,
            None => return None,
        };
        let double = size & 1 == 1;
        if double && !q {
            return None;
        }
        return call(
            "vector3",
            vec![
                Arg::Int(kind),
                Arg::Bool(double),
                Arg::Bool(q),
                if accumulates {
                    Arg::Vector(rd)
                } else {
                    Arg::Constant(0)
                },
                Arg::Vector(rn),
                Arg::Vector(rm),
            ],
        );
    }
    // The evaluator's validity rules for the integer forms.
    let valid = match opcode {
        _ if size == 3 && !q => false,
        0b10011 if u => size == 0,
        0b10111 => !u,
        0b10110 => size == 1 || size == 2,
        0b00000 | 0b00010 | 0b00100 | 0b01100..=0b01111 | 0b10010..=0b10101 => size != 3,
        _ => true,
    };
    if !valid {
        return None;
    }
    let lanes = width(q) / esize;
    // SABA and MLA/MLS accumulate into the destination.
    let d = if matches!(opcode, 0b01111 | 0b10010) {
        Arg::Vector(rd)
    } else {
        Arg::Constant(0)
    };
    call(
        "int_three",
        vec![
            Arg::Bool(u),
            Arg::Int(opcode),
            Arg::Int(esize),
            Arg::Int(lanes),
            d,
            Arg::Vector(rn),
            Arg::Vector(rm),
        ],
    )
}

/// Two-register miscellaneous operations: REV, ADDLP, ABS/NEG, compares with
/// zero, narrowing, and the like.
fn specialize_two_misc(word: u32, scalar: bool) -> Option<Call> {
    let (rd, rn, _) = regs(word);
    let (rd, rn) = (rd as u8, rn as u8);
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 16, 12),
    );
    // The evaluator sends these opcodes to its floating-point forms.
    let is_fp = match opcode {
        0b01100..=0b01111 => size >= 2,
        0b11100 => size < 2,
        0b10110..=0b11111 => true,
        _ => false,
    };
    if is_fp {
        return specialize_fp_two_misc(word, scalar);
    }
    let esize = 8 << size;
    let lanes = if scalar { 1 } else { width(q) / esize };
    match (u, opcode) {
        (_, 0b00000) | (false, 0b00001) => {
            // REV64, REV32, REV16
            let container = match (u, opcode) {
                (false, 0) => 64,
                (true, 0) => 32,
                _ => 16,
            };
            if scalar || esize >= container {
                return None;
            }
            call(
                "vector_rev",
                vec![
                    Arg::Int(container),
                    Arg::Int(esize),
                    Arg::Bool(q),
                    Arg::Vector(rn),
                ],
            )
        }
        (_, 0b00010) | (_, 0b00110) => {
            // SADDLP, UADDLP, SADALP, UADALP
            if scalar || size == 3 {
                return None;
            }
            call(
                "vector_pairwise_sum",
                vec![
                    Arg::Bool(u),
                    Arg::Bool(opcode == 0b00110),
                    Arg::Int(esize),
                    Arg::Int(width(q) / (esize * 2)),
                    if opcode == 0b00110 {
                        Arg::Vector(rd)
                    } else {
                        Arg::Constant(0)
                    },
                    Arg::Vector(rn),
                ],
            )
        }
        (_, 0b00011) => {
            // SUQADD, USQADD
            if !scalar && size == 3 && !q {
                return None;
            }
            call(
                "vector_int2",
                vec![
                    Arg::Int((u32::from(u) << 7) | (size << 5) | opcode),
                    Arg::Int(esize),
                    Arg::Int(lanes),
                    Arg::Vector(rd),
                    Arg::Vector(rn),
                ],
            )
        }
        (_, 0b00100) => {
            // CLS, CLZ
            if scalar || size == 3 {
                return None;
            }
            call(
                "vector_int2",
                vec![
                    Arg::Int((u32::from(u) << 7) | (size << 5) | opcode),
                    Arg::Int(esize),
                    Arg::Int(lanes),
                    Arg::Constant(0),
                    Arg::Vector(rn),
                ],
            )
        }
        (_, 0b00101) => {
            // CNT, NOT, RBIT (byte lanes regardless of `size`)
            if scalar || size > 1 || (!u && size != 0) {
                return None;
            }
            call(
                "vector_int2",
                vec![
                    Arg::Int((u32::from(u) << 7) | (size << 5) | opcode),
                    Arg::Int(8),
                    Arg::Int(width(q) / 8),
                    Arg::Constant(0),
                    Arg::Vector(rn),
                ],
            )
        }
        (_, 0b00111) | (_, 0b01000..=0b01011) => {
            // SQABS, SQNEG, compares with zero, ABS, NEG
            if (scalar && opcode != 0b00111 && size != 3) || (!scalar && size == 3 && !q) {
                return None;
            }
            if u && opcode == 0b01010 {
                return None;
            }
            call(
                "vector_int2",
                vec![
                    Arg::Int((u32::from(u) << 7) | (size << 5) | opcode),
                    Arg::Int(esize),
                    Arg::Int(lanes),
                    Arg::Constant(0),
                    Arg::Vector(rn),
                ],
            )
        }
        (_, 0b10010) | (_, 0b10100) => {
            // XTN, SQXTUN, SQXTN, UQXTN
            if size == 3 || (scalar && !u && opcode == 0b10010) {
                return None;
            }
            let mode = if scalar { 2 } else { u32::from(q) };
            call(
                "vector_narrow",
                vec![
                    Arg::Int((u32::from(u) << 1) | u32::from(opcode == 0b10100)),
                    Arg::Int(esize),
                    Arg::Int(mode),
                    Arg::Int(if scalar { 1 } else { 64 / esize }),
                    Arg::Vector(rn),
                    if mode == 1 {
                        Arg::Vector(rd)
                    } else {
                        Arg::Constant(0)
                    },
                ],
            )
        }
        (true, 0b10011) => {
            // SHLL
            if scalar || size == 3 {
                return None;
            }
            call(
                "vector_shift_long",
                vec![Arg::Bool(q), Arg::Int(esize), Arg::Vector(rn)],
            )
        }
        _ => None,
    }
}

/// The floating-point two-register forms: conversions, estimates, compares
/// with zero, absolute value and negation.
fn specialize_fp_two_misc(word: u32, scalar: bool) -> Option<Call> {
    #[derive(Clone, Copy)]
    enum Kind {
        One(u32),
        Cmp(u32),
        ToInt(u32, bool),
        FromInt(bool),
    }
    let (_, rn, _) = regs(word);
    let rn = rn as u8;
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 16, 12),
    );
    let (a, sz) = (size >> 1, size & 1);
    let esize = if sz == 1 { 64 } else { 32 };
    if (a == 0 && opcode == 0b10110) || (!u && a == 0 && opcode == 0b10111) {
        // FCVTN, FCVTXN, FCVTL; they narrow or widen, so they have their own
        // shapes and are not specialized yet.
        return None;
    }
    let kind = match (u, a, opcode) {
        (false, 1, 0b01100) => Kind::Cmp(0),
        (false, 1, 0b01101) => Kind::Cmp(1),
        (false, 1, 0b01110) => Kind::Cmp(2),
        (true, 1, 0b01100) => Kind::Cmp(3),
        (true, 1, 0b01101) => Kind::Cmp(4),
        (false, 1, 0b01111) => Kind::One(0),
        (true, 1, 0b01111) => Kind::One(1),
        (true, 1, 0b11111) => Kind::One(2),
        (false, 0, 0b11000) => Kind::One(3),
        (false, 0, 0b11001) => Kind::One(5),
        (false, 1, 0b11000) => Kind::One(4),
        (false, 1, 0b11001) => Kind::One(6),
        (true, 0, 0b11000) => Kind::One(7),
        // FRINTX and FRINTI use the current rounding mode.
        (true, 0, 0b11001) | (true, 1, 0b11001) => Kind::One(3),
        (_, 0, 0b11010) => Kind::ToInt(0, u),
        (_, 0, 0b11011) => Kind::ToInt(1, u),
        (_, 0, 0b11100) => Kind::ToInt(2, u),
        (_, 1, 0b11010) => Kind::ToInt(3, u),
        (_, 1, 0b11011) => Kind::ToInt(4, u),
        (_, 0, 0b11101) => Kind::FromInt(u),
        (false, 1, 0b11101) => Kind::One(8),
        (true, 1, 0b11101) => Kind::One(9),
        _ => return None,
    };
    if scalar && matches!(kind, Kind::One(0..=7)) {
        return None;
    }
    if !scalar && esize == 64 && !q {
        return None;
    }
    let lanes = if scalar { 1 } else { width(q) / esize };
    match kind {
        Kind::One(op) => call(
            "vector_fp1",
            vec![
                Arg::Int(op),
                Arg::Int(esize),
                Arg::Int(lanes),
                Arg::Vector(rn),
            ],
        ),
        Kind::Cmp(cmp) => call(
            "vector_fp_cmp0",
            vec![
                Arg::Int(cmp),
                Arg::Int(esize),
                Arg::Int(lanes),
                Arg::Vector(rn),
            ],
        ),
        Kind::ToInt(rounding, unsigned) => call(
            "vector_to_int",
            vec![
                Arg::Int(rounding),
                Arg::Bool(unsigned),
                Arg::Int(esize),
                Arg::Int(lanes),
                Arg::Vector(rn),
            ],
        ),
        Kind::FromInt(unsigned) => call(
            "vector_from_int",
            vec![
                Arg::Bool(unsigned),
                Arg::Int(esize),
                Arg::Int(lanes),
                Arg::Vector(rn),
            ],
        ),
    }
}

/// Across-lanes reductions: ADDV, SADDLV, SMAXV, SMINV and their unsigned
/// forms.
fn specialize_across_lanes(word: u32) -> Option<Call> {
    let (_, rn, _) = regs(word);
    let rn = rn as u8;
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 16, 12),
    );
    if opcode == 0b01100 || opcode == 0b01111 {
        // FMAXNMV, FMINNMV, FMAXV, FMINV
        return None;
    }
    if size == 3 || (size == 2 && !q) {
        return None;
    }
    let valid = matches!(
        (u, opcode),
        (_, 0b00011 | 0b01010 | 0b11010) | (false, 0b11011)
    );
    if !valid {
        return None;
    }
    let esize = 8 << size;
    call(
        "vector_across",
        vec![
            Arg::Int((u32::from(u) << 5) | opcode),
            Arg::Int(esize),
            Arg::Int(width(q) / esize),
            Arg::Vector(rn),
        ],
    )
}

/// The fixed-point conversions and shifts in the shift-by-immediate group.
fn specialize_shift_immediate(word: u32, scalar: bool) -> Option<Call> {
    if !bit(word, 10) || bit(word, 23) {
        return None;
    }
    let immh = bits(word, 22, 19);
    if immh == 0 {
        return None;
    }
    let (_, rn, _) = regs(word);
    let (rd, rn) = (bits(word, 4, 0) as u8, rn as u8);
    let q = bit(word, 30);
    let u = bit(word, 29);
    let opcode = bits(word, 15, 11);
    let esize = 8 << (31 - immh.leading_zeros());
    let lanes = if scalar { 1 } else { width(q) / esize };
    let imm = bits(word, 22, 16);
    if matches!(opcode, 0b11100 | 0b11111) {
        // SCVTF, UCVTF, FCVTZS, FCVTZU (fixed point)
        if esize < 32 || (!scalar && esize == 64 && !q) {
            return None;
        }
        return call(
            "vector_fixed_convert",
            vec![
                Arg::Bool(opcode == 0b11100),
                Arg::Bool(u),
                Arg::Int(esize),
                Arg::Int(lanes),
                Arg::Int(2 * esize - imm),
                Arg::Vector(rn),
            ],
        );
    }
    // SSHR, USHR, SRSHR, URSHR, SRI, SHL, SLI, SQSHL, UQSHL, SQSHLU.
    let valid = matches!(
        (u, opcode),
        (_, 0b00000 | 0b00010 | 0b00100 | 0b00110 | 0b01010 | 0b01110) | (true, 0b01000 | 0b01100)
    );
    if !valid
        || (scalar && esize != 64 && !matches!(opcode, 0b01100 | 0b01110))
        || (!scalar && esize == 64 && !q)
    {
        return None;
    }
    let accumulate =
        matches!(opcode, 0b00010 | 0b00110) || (u && matches!(opcode, 0b01000 | 0b01010));
    call(
        "vector_shift",
        vec![
            Arg::Bool(u),
            Arg::Int(opcode),
            Arg::Int(esize),
            Arg::Int(lanes),
            Arg::Int(imm),
            if accumulate {
                Arg::Vector(rd)
            } else {
                Arg::Constant(0)
            },
            Arg::Vector(rn),
        ],
    )
}

fn specialize_dup_scalar(word: u32) -> Option<Call> {
    if word & 0xFFE0_FC00 != 0x5E00_0400 {
        return None;
    }
    let (_, rn, _) = regs(word);
    let imm5 = bits(word, 20, 16);
    let size = imm5.trailing_zeros();
    if size > 3 {
        return None;
    }
    call(
        "element",
        vec![
            Arg::Int(8 << size),
            Arg::Int(imm5 >> (size + 1)),
            Arg::Vector(rn as u8),
        ],
    )
}

fn specialize_element(word: u32, scalar: bool) -> Option<Call> {
    if bit(word, 10) {
        return None;
    }
    let (rd, rn, _) = regs(word);
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 15, 12),
    );
    let op = match (u, opcode) {
        (false, 0b0001) => 0,
        (false, 0b0101) => 1,
        (false, 0b1001) => 2,
        (true, 0b1001) => 3,
        _ => return None,
    };
    if size < 2 {
        return None;
    }
    let double = size & 1 == 1;
    let (l, h) = (bits(word, 21, 21), bits(word, 11, 11));
    let index = if double {
        if l == 1 {
            return None;
        }
        h
    } else {
        (h << 1) | l
    };
    if !scalar && double && !q {
        return None;
    }
    let rm = bits(word, 20, 16) as u8;
    call(
        "vector_element",
        vec![
            Arg::Int(op),
            Arg::Bool(double),
            Arg::Bool(q),
            Arg::Bool(scalar),
            Arg::Int(index),
            if op < 2 {
                Arg::Vector(rd as u8)
            } else {
                Arg::Constant(0)
            },
            Arg::Vector(rn as u8),
            Arg::Vector(rm),
        ],
    )
}

// ---------------------------------------------------------------------------
// The specialized functions

const SCALAR_FP2: [Fp2; 10] = [
    Fp2::Mul,
    Fp2::Div,
    Fp2::Add,
    Fp2::Sub,
    Fp2::Max,
    Fp2::Min,
    Fp2::MaxNum,
    Fp2::MinNum,
    Fp2::Nmul,
    Fp2::Abd,
];

// Host fast paths. For a result that is not NaN, the host computes exactly
// what ARM does (round to nearest, no flush to zero), which is also what the
// exact evaluator does: it only adds ARM's NaN rules (which operand's NaN,
// quieted, or the default NaN) and per-lane dispatch. So the fast paths
// compute every lane directly and leave an operation to the evaluator only
// when a lane came out NaN. Matrix and vector math in job threads runs
// through here; at one evaluator dispatch per lane it was most of LM3's
// title-screen setup time.

/// The f32 lanes of `v`.
#[inline(always)]
fn f32_lanes(v: u128) -> [f32; 4] {
    [0, 1, 2, 3].map(|i| f32::from_bits((v >> (32 * i)) as u32))
}

/// Whether any single-precision lane of `bits` is NaN.
///
/// A lane is NaN when its exponent is all ones and its mantissa is
/// non-zero, that is when `bits & 0x7fff_ffff > 0x7f80_0000`. Clearing the
/// sign and adding `0x007f_ffff` sets bit 31 of exactly the NaN lanes (the
/// add cannot carry into the next lane: a cleared lane is at most
/// `0x7fff_ffff`, so the sum stays below `2^32`). One test covers all four
/// lanes, so the portable path no longer branches per lane.
#[inline(always)]
fn any_nan_f32_bits(bits: u128) -> bool {
    const ABS: u128 = 0x7FFF_FFFF_7FFF_FFFF_7FFF_FFFF_7FFF_FFFF;
    const NAN_CARRY: u128 = 0x007F_FFFF_007F_FFFF_007F_FFFF_007F_FFFF;
    const SIGN: u128 = 0x8000_0000_8000_0000_8000_0000_8000_0000;
    (bits & ABS).wrapping_add(NAN_CARRY) & SIGN != 0
}

/// The first `lanes` values packed, or `None` when one is NaN. The lanes are
/// packed first and checked together with [`any_nan_f32_bits`].
#[inline(always)]
fn pack_f32(lanes: usize, values: [f32; 4]) -> Option<u128> {
    let mut result = 0u128;
    for (i, value) in values.iter().enumerate().take(lanes) {
        result |= u128::from(value.to_bits()) << (32 * i);
    }
    (!any_nan_f32_bits(result)).then_some(result)
}

/// `a * b + c` per lane, fused (on the host's FMA unit when it has one:
/// without the target feature, `mul_add` is a call into libm).
#[inline]
fn fma_f32(a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU has FMA.
        return unsafe { fma_f32_host(a, b, c) };
    }
    [
        a[0].mul_add(b[0], c[0]),
        a[1].mul_add(b[1], c[1]),
        a[2].mul_add(b[2], c[2]),
        a[3].mul_add(b[3], c[3]),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn fma_f32_host(a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
    [
        a[0].mul_add(b[0], c[0]),
        a[1].mul_add(b[1], c[1]),
        a[2].mul_add(b[2], c[2]),
        a[3].mul_add(b[3], c[3]),
    ]
}

/// Scalar `a * b + c`, fused.
#[inline]
fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU has FMA.
        return unsafe { fma_f64_host(a, b, c) };
    }
    a.mul_add(b, c)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn fma_f64_host(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// A scalar result's bits, or `None` when it is NaN.
#[inline(always)]
fn scalar_bits(double: bool, value: f64, single: f32) -> Option<u128> {
    if double {
        (!value.is_nan()).then(|| u128::from(value.to_bits()))
    } else {
        (!single.is_nan()).then(|| u128::from(single.to_bits()))
    }
}

/// FMUL, FDIV, FADD, FSUB, FNMUL (scalar) on the host.
fn scalar_fp2_host(op: u32, double: bool, n: u128, m: u128) -> Option<u128> {
    let (a, b) = (f64::from_bits(n as u64), f64::from_bits(m as u64));
    let (x, y) = (f32::from_bits(n as u32), f32::from_bits(m as u32));
    match op {
        0 => scalar_bits(double, a * b, x * y),
        1 => scalar_bits(double, a / b, x / y),
        2 => scalar_bits(double, a + b, x + y),
        3 => scalar_bits(double, a - b, x - y),
        8 => scalar_bits(double, -(a * b), -(x * y)),
        9 => scalar_bits(double, (a - b).abs(), (x - y).abs()),
        _ => None,
    }
}

/// FMUL, FDIV, FADD, FSUB, FMAX, FMIN, FMAXNM, FMINNM, FNMUL, FABD (scalar;
/// `op` is bits 15:12).
pub fn scalar_fp2(op: u32, double: bool, n: u128, m: u128) -> u128 {
    if let Some(result) = scalar_fp2_host(op, double, n, m) {
        return result;
    }
    let size = size_of(double);
    let op = SCALAR_FP2[op as usize];
    u128::from(sized!(size, fp2(op, lane(n, size, 0), lane(m, size, 0))))
}

/// FMADD, FMSUB, FNMADD, FNMSUB: `negate` bit 0 negates n, bit 1 negates
/// a (bits 21 and 15 of the instruction).
pub fn scalar_fma(negate: u32, double: bool, a: u128, n: u128, m: u128) -> u128 {
    let (negate_a, negate_n) = match negate {
        0 => (false, false),
        1 => (false, true),
        2 => (true, true),
        _ => (true, false),
    };
    let host = if double {
        let sign = |negative: bool, raw: u128| {
            let value = f64::from_bits(raw as u64);
            if negative {
                -value
            } else {
                value
            }
        };
        let value = fma_f64(
            sign(negate_n, n),
            f64::from_bits(m as u64),
            sign(negate_a, a),
        );
        scalar_bits(true, value, 0.0)
    } else {
        let sign = |negative: bool, raw: u128| {
            let value = f32::from_bits(raw as u32);
            if negative {
                -value
            } else {
                value
            }
        };
        let value = fma_f32(
            [sign(negate_n, n), 0.0, 0.0, 0.0],
            [f32::from_bits(m as u32), 0.0, 0.0, 0.0],
            [sign(negate_a, a), 0.0, 0.0, 0.0],
        )[0];
        scalar_bits(false, 0.0, value)
    };
    if let Some(result) = host {
        return result;
    }
    let size = size_of(double);
    let flip = 1u64 << (size - 1);
    let (a, n, m) = (lane(a, size, 0), lane(n, size, 0), lane(m, size, 0));
    let (a, n) = match negate {
        0 => (a, n),
        1 => (a, n ^ flip),
        2 => (a ^ flip, n ^ flip),
        _ => (a ^ flip, n),
    };
    u128::from(sized!(size, fp_mul_add(a, n, m)))
}

/// FCMP, FCMPE: the flags.
pub fn scalar_compare(double: bool, n: u128, m: u128) -> u8 {
    let size = size_of(double);
    let (n, m) = (lane(n, size, 0), lane(m, size, 0));
    compare_flags(sized!(size, fp_compare(n, m)))
}

/// FCSEL.
pub fn scalar_select(double: bool, cond: u32, nzcv: u8, n: u128, m: u128) -> u128 {
    let chosen = if condition_holds(cond, nzcv) { n } else { m };
    u128::from(lane(chosen, size_of(double), 0))
}

/// FMOV, FABS, FNEG, FSQRT, FRINT*, and FCVT between single and double
/// (`opcode` is bits 20:15, `ftype` bits 23:22).
pub fn scalar_fp1(opcode: u32, ftype: u32, n: u128) -> u128 {
    let from = if ftype == 1 { 64 } else { 32 };
    if opcode & 0b11_1100 == 0b00_0100 {
        let to = if opcode & 3 == 1 { 64 } else { 32 };
        return u128::from(fp_convert(lane(n, from, 0), from, to));
    }
    let op = match opcode {
        1 => Some(Fp1::Abs),
        2 => Some(Fp1::Neg),
        3 => Some(Fp1::Sqrt),
        8 => Some(Fp1::Round(Rounding::TiesEven)),
        9 => Some(Fp1::Round(Rounding::Up)),
        10 => Some(Fp1::Round(Rounding::Down)),
        11 => Some(Fp1::Round(Rounding::Zero)),
        12 => Some(Fp1::Round(Rounding::TiesAway)),
        14 | 15 => Some(Fp1::Round(Rounding::TiesEven)),
        _ => None,
    };
    let n = lane(n, from, 0);
    u128::from(match op {
        Some(op) => sized!(from, fp1(op, n)),
        None => n,
    })
}

/// A value fixed when the code was generated (FMOV immediate).
pub fn constant(value: u128) -> u128 {
    value
}

/// The low `width` bits of a vector register (FMOV to a general register).
pub fn low_bits(width: u32, n: u128) -> u128 {
    u128::from(lane(n, width, 0))
}

/// The low `width` bits of a general register (FMOV to a vector register).
pub fn from_general(width: u32, x: u64) -> u128 {
    u128::from(x & mask(width))
}

/// SCVTF, UCVTF from a 32- or 64-bit general register.
pub fn int_to_float(double: bool, unsigned: bool, wide: bool, x: u64) -> u128 {
    let int_bits = if wide { 64 } else { 32 };
    u128::from(sized!(size_of(double), int_to_fp(x, unsigned, int_bits, 0)))
}

/// FCVT[NPMZA][SU] to a general register (`mode` is rmode << 3 | opcode).
pub fn float_to_int(double: bool, mode: u32, wide: bool, n: u128) -> u128 {
    let size = size_of(double);
    let (rmode, opcode) = (mode >> 3, mode & 7);
    let rounding = match (rmode, opcode) {
        (0, 0b100) | (0, 0b101) => Rounding::TiesAway,
        (0, _) => Rounding::TiesEven,
        (1, _) => Rounding::Up,
        (2, _) => Rounding::Down,
        _ => Rounding::Zero,
    };
    let int_bits = if wide { 64 } else { 32 };
    let unsigned = opcode & 1 == 1;
    u128::from(sized!(
        size,
        fp_to_int(lane(n, size, 0), rounding, 0, unsigned, int_bits)
    ))
}

#[derive(Clone, Copy)]
enum VectorKind {
    Two(Fp2),
    Pair(Fp2),
    Fma(bool),
    Cmp(Cmp),
}

/// The three-same floating-point operation of `kind` (U << 6 | size<1> << 5
/// | opcode), as `exec` decodes it.
fn vector_kind(kind: u32) -> Option<VectorKind> {
    let (u, high, opcode) = (kind >> 6 == 1, (kind >> 5) & 1, kind & 0x1F);
    Some(match (u, high, opcode) {
        (false, 0, 0b11000) => VectorKind::Two(Fp2::MaxNum),
        (false, 1, 0b11000) => VectorKind::Two(Fp2::MinNum),
        (false, 0, 0b11001) => VectorKind::Fma(false),
        (false, 1, 0b11001) => VectorKind::Fma(true),
        (false, 0, 0b11010) => VectorKind::Two(Fp2::Add),
        (false, 1, 0b11010) => VectorKind::Two(Fp2::Sub),
        (false, 0, 0b11011) => VectorKind::Two(Fp2::MulX),
        (false, 0, 0b11100) => VectorKind::Cmp(Cmp::Eq),
        (false, 0, 0b11110) => VectorKind::Two(Fp2::Max),
        (false, 1, 0b11110) => VectorKind::Two(Fp2::Min),
        (false, 0, 0b11111) => VectorKind::Two(Fp2::RecipStep),
        (false, 1, 0b11111) => VectorKind::Two(Fp2::RsqrtStep),
        (true, 0, 0b11000) => VectorKind::Pair(Fp2::MaxNum),
        (true, 1, 0b11000) => VectorKind::Pair(Fp2::MinNum),
        (true, 0, 0b11010) => VectorKind::Pair(Fp2::Add),
        (true, 1, 0b11010) => VectorKind::Two(Fp2::Abd),
        (true, 0, 0b11011) => VectorKind::Two(Fp2::Mul),
        (true, 0, 0b11100) => VectorKind::Cmp(Cmp::Ge),
        (true, 1, 0b11100) => VectorKind::Cmp(Cmp::Gt),
        (true, 0, 0b11101) => VectorKind::Cmp(Cmp::AbsGe),
        (true, 1, 0b11101) => VectorKind::Cmp(Cmp::AbsGt),
        (true, 0, 0b11110) => VectorKind::Pair(Fp2::Max),
        (true, 1, 0b11110) => VectorKind::Pair(Fp2::Min),
        (true, 0, 0b11111) => VectorKind::Two(Fp2::Div),
        _ => return None,
    })
}

/// A three-same floating-point vector operation (FADD, FMUL, FMLA,
/// FADDP, FCMGE, ...) over the lanes of 64 or 128 bits.
pub fn vector3(kind: u32, double: bool, q: bool, d: u128, n: u128, m: u128) -> u128 {
    // A vector register at a time: four single-precision lanes (two for the
    // 64-bit form) or two double-precision lanes. NaN results fall back to
    // the scalar path below, which leaves Arm's NaN rules to the evaluator.
    #[cfg(target_arch = "x86_64")]
    if x86::avx2_fma() {
        // SAFETY: the CPU has AVX2 and FMA.
        let fast = if double {
            if q {
                unsafe { x86::vector3_f64(kind, d, n, m) }
            } else {
                None
            }
        } else {
            unsafe { x86::vector3_f32(kind, q, d, n, m) }
        };
        if let Some(result) = fast {
            return result;
        }
    }
    if !double {
        if let Some(result) = vector3_f32(kind, q, d, n, m) {
            return result;
        }
    }
    let Some(kind) = vector_kind(kind) else {
        return 0;
    };
    let esize = size_of(double);
    let lanes = width(q) / esize;
    let mut result = 0u128;
    for i in 0..lanes {
        let (a, b) = match kind {
            VectorKind::Pair(_) => pair(n, m, esize, lanes, i),
            _ => (lane(n, esize, i), lane(m, esize, i)),
        };
        let value = match kind {
            VectorKind::Two(op) | VectorKind::Pair(op) => sized!(esize, fp2(op, a, b)),
            VectorKind::Fma(subtract) => {
                let a = if subtract { a ^ (1 << (esize - 1)) } else { a };
                sized!(esize, fp_mul_add(lane(d, esize, i), a, b))
            }
            VectorKind::Cmp(cmp) => ones(sized!(esize, fp_cmp(cmp, a, b)), esize),
        };
        result = put(result, esize, i, value);
    }
    result
}

/// The common single-precision three-same operations on the host.
fn vector3_f32(kind: u32, q: bool, d: u128, n: u128, m: u128) -> Option<u128> {
    let lanes = if q { 4 } else { 2 };
    let (a, b) = (f32_lanes(n), f32_lanes(m));
    let values = match vector_kind(kind)? {
        VectorKind::Two(Fp2::Add) => [0, 1, 2, 3].map(|i| a[i] + b[i]),
        VectorKind::Two(Fp2::Sub) => [0, 1, 2, 3].map(|i| a[i] - b[i]),
        // FMULX differs from FMUL only for 0 * infinity, which is NaN here.
        VectorKind::Two(Fp2::Mul | Fp2::MulX) => [0, 1, 2, 3].map(|i| a[i] * b[i]),
        VectorKind::Two(Fp2::Div) => [0, 1, 2, 3].map(|i| a[i] / b[i]),
        VectorKind::Two(Fp2::Abd) => [0, 1, 2, 3].map(|i| (a[i] - b[i]).abs()),
        // FMAX/FMIN and their maxNum/minNum forms. A NaN operand takes the
        // exact path: the host maximum cannot reproduce the signalling-NaN
        // and default-NaN rules. Equal values differ at most in the sign of
        // zero, which the bitwise combination resolves like the evaluator.
        VectorKind::Two(op @ (Fp2::Max | Fp2::MaxNum | Fp2::Min | Fp2::MinNum)) => {
            if pack_f32(lanes, a).is_none() || pack_f32(lanes, b).is_none() {
                return None;
            }
            let max = matches!(op, Fp2::Max | Fp2::MaxNum);
            [0, 1, 2, 3].map(|i| {
                let (x, y) = (a[i], b[i]);
                let (greater, less) = (x > y, x < y);
                if max && greater || !max && less {
                    x
                } else if max && less || !max && greater {
                    y
                } else if max {
                    f32::from_bits(x.to_bits() & y.to_bits())
                } else {
                    f32::from_bits(x.to_bits() | y.to_bits())
                }
            })
        }
        VectorKind::Fma(subtract) => {
            let a = if subtract { a.map(|x| -x) } else { a };
            fma_f32(a, b, f32_lanes(d))
        }
        VectorKind::Pair(Fp2::Add) => {
            // Pairs of the concatenation n:m.
            let element = |j: usize| if j < lanes { a[j] } else { b[j - lanes] };
            [0, 1, 2, 3].map(|i| {
                if i < lanes {
                    element(2 * i) + element(2 * i + 1)
                } else {
                    0.0
                }
            })
        }
        _ => return None,
    };
    pack_f32(lanes, values)
}

/// FMLA, FMLS, FMUL, FMULX by element for single precision, as the host
/// computes them, or `None` when a written lane is NaN (the evaluator's Arm
/// NaN rules apply then). All lanes are packed and checked together.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn vector_element_f32(
    op: u32,
    q: bool,
    scalar: bool,
    index: u32,
    d: u128,
    n: u128,
    m: u128,
) -> Option<u128> {
    let lanes = if scalar {
        1
    } else if q {
        4
    } else {
        2
    };
    let element = [f32::from_bits((m >> (32 * index)) as u32); 4];
    let a = f32_lanes(n);
    let values = match op {
        0 => fma_f32(a, element, f32_lanes(d)),
        1 => fma_f32(a.map(|x| -x), element, f32_lanes(d)),
        // FMULX differs from FMUL only for 0 * infinity, NaN here.
        _ => [0, 1, 2, 3].map(|i| a[i] * element[i]),
    };
    pack_f32(lanes, values)
}

/// FMLA, FMLS, FMUL, FMULX by element (`op` 0..=3), vector or scalar.
#[allow(clippy::too_many_arguments)]
pub fn vector_element(
    op: u32,
    double: bool,
    q: bool,
    scalar: bool,
    index: u32,
    d: u128,
    n: u128,
    m: u128,
) -> u128 {
    #[cfg(target_arch = "x86_64")]
    if x86::avx2_fma() {
        // SAFETY: the CPU has AVX2 and FMA.
        let fast = if double {
            unsafe { x86::vector_element_f64(op, scalar, index, d, n, m) }
        } else {
            unsafe { x86::vector_element_f32(op, q, scalar, index, d, n, m) }
        };
        if let Some(result) = fast {
            return result;
        }
    }
    if !double {
        if let Some(result) = vector_element_f32(op, q, scalar, index, d, n, m) {
            return result;
        }
    }
    let esize = size_of(double);
    let element = lane(m, esize, index);
    let lanes = if scalar { 1 } else { width(q) / esize };
    let mut result = 0u128;
    for i in 0..lanes {
        let a = lane(n, esize, i);
        let value = match op {
            0 => sized!(esize, fp_mul_add(lane(d, esize, i), a, element)),
            1 => sized!(
                esize,
                fp_mul_add(lane(d, esize, i), a ^ (1 << (esize - 1)), element)
            ),
            2 => sized!(esize, fp2(Fp2::Mul, a, element)),
            _ => sized!(esize, fp2(Fp2::MulX, a, element)),
        };
        result = put(result, esize, i, value);
    }
    result
}

/// EXT: bytes `position..` of n:m.
pub fn extract(q: bool, position: u32, n: u128, m: u128) -> u128 {
    let bytes = if q { 16 } else { 8 };
    let position = position as usize;
    let n = n.to_le_bytes();
    let m = m.to_le_bytes();
    let mut concat = [0u8; 32];
    concat[..bytes].copy_from_slice(&n[..bytes]);
    concat[bytes..2 * bytes].copy_from_slice(&m[..bytes]);
    let mut out = [0u8; 16];
    out[..bytes].copy_from_slice(&concat[position..position + bytes]);
    u128::from_le_bytes(out)
}

/// DUP (element), scalar: lane `index` of `esize` bits.
pub fn element(esize: u32, index: u32, n: u128) -> u128 {
    u128::from(lane(n, esize, index))
}

/// ZIP1, UZP1, TRN1, ZIP2, UZP2, TRN2 (`op` is bits 14:12) on `esize`-bit
/// lanes of 64 or 128 bits.
pub fn permute(op: u32, esize: u32, q: bool, n: u128, m: u128) -> u128 {
    let lanes = width(q) / esize;
    let second = u32::from(op >= 5);
    let mut result = 0u128;
    for i in 0..lanes {
        let (pair, odd) = (i / 2, i % 2 == 1);
        let value = match op & 3 {
            1 => {
                let j = 2 * i + second;
                if j < lanes {
                    lane(n, esize, j)
                } else {
                    lane(m, esize, j - lanes)
                }
            }
            2 => lane(if odd { m } else { n }, esize, 2 * pair + second),
            _ => lane(if odd { m } else { n }, esize, pair + second * lanes / 2),
        };
        result = put(result, esize, i, value);
    }
    result
}

/// Three-same integer operations (ADD, SUB, MUL, compares, pairwise forms,
/// shifts, saturating arithmetic, ...): `opcode` is bits 15:11 and `lanes`
/// the number of `esize`-bit lanes.
pub fn int_three(u: bool, opcode: u32, esize: u32, lanes: u32, d: u128, n: u128, m: u128) -> u128 {
    // One 128-bit host operation for the common three-same forms; forms the
    // host cannot express (8- and 64-bit MUL, for instance) stay scalar.
    #[cfg(target_arch = "x86_64")]
    if x86::avx2() {
        // SAFETY: the CPU has AVX2.
        if let Some(result) = unsafe { x86::int_three(u, opcode, esize, lanes, d, n, m) } {
            return result;
        }
    }
    int_three_scalar(u, opcode, esize, lanes, d, n, m)
}

fn int_three_scalar(
    u: bool,
    opcode: u32,
    esize: u32,
    lanes: u32,
    d: u128,
    n: u128,
    m: u128,
) -> u128 {
    let mut result = 0u128;
    if matches!(opcode, 0b10100 | 0b10101 | 0b10111) {
        // SMAXP, UMAXP, SMINP, UMINP, ADDP: pairs of the n:m concatenation.
        for i in 0..lanes {
            let (a, b) = pair(n, m, esize, lanes, i);
            let (x, y) = (int(a, esize, u), int(b, esize, u));
            let value = match opcode {
                0b10100 => wrap(x.max(y), esize),
                0b10101 => wrap(x.min(y), esize),
                _ => wrap(x + y, esize),
            };
            result = put(result, esize, i, value);
        }
        return result;
    }
    for i in 0..lanes {
        let value = int_three_same(
            u,
            opcode,
            esize,
            lane(n, esize, i),
            lane(m, esize, i),
            lane(d, esize, i),
        )
        .unwrap_or(0);
        result = put(result, esize, i, value);
    }
    result
}

// ---------------------------------------------------------------------------
// AVX2 host paths
//
// The portable paths above compute each lane separately, extracting and
// inserting 8-, 16-, 32- or 64-bit fields through 128-bit shifts and i128
// arithmetic. These do a whole register per instruction. They compute what
// the portable paths compute; anything the host cannot express (8- and
// 64-bit MUL, for instance) falls back. Floating-point results that come out
// NaN also fall back, so Arm's NaN and signed-zero rules stay in the
// evaluator.
//
// AArch64 (and every other non-x86_64 host) does not compile this module:
// the portable per-lane paths above are the exact fallback there.

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    /// Whether the host has AVX2 (all the integer paths need).
    #[inline]
    pub(super) fn avx2() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
    }

    /// Whether the host has the AVX2 and FMA the single-precision paths use.
    #[inline]
    pub(super) fn avx2_fma() -> bool {
        avx2() && std::arch::is_x86_feature_detected!("fma")
    }

    /// A vector register's 128 bits.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load(v: u128) -> __m128i {
        _mm_set_epi64x((v >> 64) as i64, v as i64)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store(v: __m128i) -> u128 {
        let mut bytes = [0u8; 16];
        _mm_storeu_si128(bytes.as_mut_ptr() as *mut __m128i, v);
        u128::from_le_bytes(bytes)
    }

    /// The lane mask of the first `lanes` single-precision lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lane_mask_f32(lanes: usize) -> __m128i {
        _mm_set_epi32(
            if lanes > 3 { -1 } else { 0 },
            if lanes > 2 { -1 } else { 0 },
            if lanes > 1 { -1 } else { 0 },
            -1,
        )
    }

    /// The first `lanes` single-precision lanes, or `None` when one is NaN.
    /// One compare and one `vptest` cover every lane together.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn pack_f32(lanes: usize, value: __m128) -> Option<u128> {
        let nan = _mm_cmpunord_ps(value, value);
        if _mm_testz_si128(_mm_castps_si128(nan), lane_mask_f32(lanes)) == 0 {
            return None;
        }
        let mut bits = store(_mm_castps_si128(value));
        if lanes < 4 {
            bits &= (1u128 << (32 * lanes)) - 1;
        }
        Some(bits)
    }

    /// A floating-point compare as all-ones lanes (a NaN compares false).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn compare_f32(cmp: Cmp, lanes: usize, a: __m128, b: __m128) -> Option<u128> {
        let sign = _mm_castsi128_ps(_mm_set1_epi32(0x8000_0000u32 as i32));
        let result = match cmp {
            Cmp::Eq => _mm_cmpeq_ps(a, b),
            Cmp::Ge => _mm_cmpge_ps(a, b),
            Cmp::Gt => _mm_cmpgt_ps(a, b),
            Cmp::AbsGe => _mm_cmpge_ps(_mm_andnot_ps(sign, a), _mm_andnot_ps(sign, b)),
            Cmp::AbsGt => _mm_cmpgt_ps(_mm_andnot_ps(sign, a), _mm_andnot_ps(sign, b)),
            _ => return None,
        };
        let mut bits = store(_mm_castps_si128(result));
        if lanes < 4 {
            bits &= (1u128 << (32 * lanes)) - 1;
        }
        Some(bits)
    }

    /// Whether any lane of either operand is a NaN.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn any_nan_f32(lanes: usize, a: __m128, b: __m128) -> bool {
        let nan = _mm_cmpunord_ps(a, b);
        _mm_testz_si128(_mm_castps_si128(nan), lane_mask_f32(lanes)) == 0
    }

    /// FMAXNM: the numeric operand wins over a NaN, and equal values (only
    /// -0/+0 can differ) combine as bits so +0 wins. A NaN second operand
    /// yields a NaN here, which [`pack_f32`] leaves to the scalar path.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn max_num_f32(a: __m128, b: __m128) -> __m128 {
        let equal = _mm_cmpeq_ps(a, b);
        _mm_blendv_ps(_mm_max_ps(a, b), _mm_and_ps(a, b), equal)
    }

    /// FMINNM: the numeric operand wins over a NaN, and equal values combine
    /// as bits so -0 wins.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn min_num_f32(a: __m128, b: __m128) -> __m128 {
        let equal = _mm_cmpeq_ps(a, b);
        _mm_blendv_ps(_mm_min_ps(a, b), _mm_or_ps(a, b), equal)
    }

    /// FMAXP/FMINP/FMAXNMP/FMINNMP for single precision: the max or min of
    /// the pairs of the concatenation `n:m`. NaN handling is the caller's:
    /// the max/min forms must fall back before this, while the maxNum/minNum
    /// forms select the numeric operand and leave an all-NaN result to
    /// [`pack_f32`].
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn pair_max_min_f32(lanes: usize, max: bool, a: __m128, b: __m128) -> __m128 {
        if lanes == 4 {
            // [n0 n2 m0 m2] against [n1 n3 m1 m3] gives the four pairs.
            let x = _mm_shuffle_ps::<0b10_00_10_00>(a, b);
            let y = _mm_shuffle_ps::<0b11_01_11_01>(a, b);
            if max {
                max_num_f32(x, y)
            } else {
                min_num_f32(x, y)
            }
        } else {
            // A 64-bit vector reduces n.2s and m.2s separately.
            let x = _mm_shuffle_ps::<0b00_00_00_00>(a, b);
            let y = _mm_shuffle_ps::<0b01_01_01_01>(a, b);
            let pairs = if max {
                max_num_f32(x, y)
            } else {
                min_num_f32(x, y)
            };
            _mm_shuffle_ps::<0b10_00_10_00>(pairs, pairs)
        }
    }

    /// FADD, FSUB, FMUL, FMULX, FDIV, FMAX, FMIN, FMAXNM, FMINNM, FMLA,
    /// FMLS, FADDP, FMAXP, FMINP, FRECPS, FRSQRTS and the floating-point
    /// compares for single precision.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vector3_f32(
        kind: u32,
        q: bool,
        d: u128,
        n: u128,
        m: u128,
    ) -> Option<u128> {
        let lanes = if q { 4 } else { 2 };
        let a = _mm_castsi128_ps(load(n));
        let b = _mm_castsi128_ps(load(m));
        let c = _mm_castsi128_ps(load(d));
        match vector_kind(kind)? {
            VectorKind::Two(Fp2::Add) => pack_f32(lanes, _mm_add_ps(a, b)),
            VectorKind::Two(Fp2::Sub) => pack_f32(lanes, _mm_sub_ps(a, b)),
            VectorKind::Two(Fp2::Div) => pack_f32(lanes, _mm_div_ps(a, b)),
            VectorKind::Two(Fp2::Abd) => {
                // |n - m|; a NaN difference falls back below.
                let sign = _mm_castsi128_ps(_mm_set1_epi32(0x8000_0000u32 as i32));
                pack_f32(lanes, _mm_andnot_ps(sign, _mm_sub_ps(a, b)))
            }
            // FMULX differs from FMUL only for 0 * infinity: NaN here, so
            // the scalar path (which knows the special result) takes it.
            VectorKind::Two(Fp2::Mul | Fp2::MulX) => pack_f32(lanes, _mm_mul_ps(a, b)),
            // FMAX/FMIN must return a NaN when either operand is one, which
            // the host maximum would not do for a NaN first operand; the
            // maxNum/minNum forms distinguish quiet from signalling NaNs,
            // which the host maximum does not. NaN operands take the exact
            // path in every case.
            VectorKind::Two(Fp2::Max | Fp2::Min | Fp2::MaxNum | Fp2::MinNum)
                if any_nan_f32(lanes, a, b) =>
            {
                None
            }
            VectorKind::Two(Fp2::Max | Fp2::MaxNum) => pack_f32(lanes, max_num_f32(a, b)),
            VectorKind::Two(Fp2::Min | Fp2::MinNum) => pack_f32(lanes, min_num_f32(a, b)),
            // The reciprocal steps are a fused 2 - n*m and 1.5 - n*(m/2);
            // infinity times zero produces a NaN the scalar path owns.
            VectorKind::Two(Fp2::RecipStep) => {
                pack_f32(lanes, _mm_fnmadd_ps(a, b, _mm_set1_ps(2.0)))
            }
            VectorKind::Two(Fp2::RsqrtStep) => pack_f32(
                lanes,
                _mm_fnmadd_ps(a, _mm_mul_ps(b, _mm_set1_ps(0.5)), _mm_set1_ps(1.5)),
            ),
            VectorKind::Fma(subtract) => {
                let value = if subtract {
                    // d - n * m, with a single rounding like FMLS.
                    _mm_fnmadd_ps(a, b, c)
                } else {
                    _mm_fmadd_ps(a, b, c)
                };
                pack_f32(lanes, value)
            }
            VectorKind::Pair(Fp2::Add) => {
                let pairs = _mm_hadd_ps(a, b);
                // For the 64-bit form the second result lane is the third
                // lane of the 128-bit pairwise sum.
                let value = if lanes == 4 {
                    pairs
                } else {
                    _mm_shuffle_ps::<0b10_10_10_00>(pairs, pairs)
                };
                pack_f32(lanes, value)
            }
            // FMAXP/FMINP propagate a NaN, unlike the maxNum/minNum forms;
            // NaN operands take the exact path in every case.
            VectorKind::Pair(Fp2::Max | Fp2::Min | Fp2::MaxNum | Fp2::MinNum)
                if any_nan_f32(lanes, a, b) =>
            {
                None
            }
            VectorKind::Pair(Fp2::Max | Fp2::MaxNum) => {
                pack_f32(lanes, pair_max_min_f32(lanes, true, a, b))
            }
            VectorKind::Pair(Fp2::Min | Fp2::MinNum) => {
                pack_f32(lanes, pair_max_min_f32(lanes, false, a, b))
            }
            VectorKind::Cmp(cmp) => compare_f32(cmp, lanes, a, b),
            _ => None,
        }
    }

    /// FMLA, FMLS, FMUL and FMULX by element for single precision.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vector_element_f32(
        op: u32,
        q: bool,
        scalar: bool,
        index: u32,
        d: u128,
        n: u128,
        m: u128,
    ) -> Option<u128> {
        let lanes = if scalar {
            1
        } else if q {
            4
        } else {
            2
        };
        let n = _mm_castsi128_ps(load(n));
        let d = _mm_castsi128_ps(load(d));
        let b = _mm_castsi128_ps(load(m));
        let element = _mm_permutevar_ps(b, _mm_set1_epi32((index & 3) as i32));
        let value = match op {
            0 => _mm_fmadd_ps(n, element, d),
            1 => _mm_fnmadd_ps(n, element, d),
            _ => _mm_mul_ps(n, element),
        };
        pack_f32(lanes, value)
    }

    /// The first `lanes` double-precision lanes, or `None` when one is NaN.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn pack_f64(lanes: usize, value: __m128d) -> Option<u128> {
        let nan = _mm_movemask_pd(_mm_cmpunord_pd(value, value)) as u32;
        if nan & ((1u32 << lanes) - 1) != 0 {
            return None;
        }
        let mut bits = store(_mm_castpd_si128(value));
        if lanes < 2 {
            bits &= u128::from(u64::MAX);
        }
        Some(bits)
    }

    /// A double-precision compare as all-ones lanes (a NaN compares false).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn compare_f64(cmp: Cmp, lanes: usize, a: __m128d, b: __m128d) -> Option<u128> {
        let sign = _mm_castsi128_pd(_mm_set1_epi64x(i64::MIN));
        let result = match cmp {
            Cmp::Eq => _mm_cmpeq_pd(a, b),
            Cmp::Ge => _mm_cmpge_pd(a, b),
            Cmp::Gt => _mm_cmpgt_pd(a, b),
            Cmp::AbsGe => _mm_cmpge_pd(_mm_andnot_pd(sign, a), _mm_andnot_pd(sign, b)),
            Cmp::AbsGt => _mm_cmpgt_pd(_mm_andnot_pd(sign, a), _mm_andnot_pd(sign, b)),
            _ => return None,
        };
        let mut bits = store(_mm_castpd_si128(result));
        if lanes < 2 {
            bits &= u128::from(u64::MAX);
        }
        Some(bits)
    }

    /// Whether any lane of either operand is a NaN.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn any_nan_f64(a: __m128d, b: __m128d) -> bool {
        _mm_movemask_pd(_mm_cmpunord_pd(a, b)) != 0
    }

    /// FMAXNM for double precision; see [`max_num_f32`].
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn max_num_f64(a: __m128d, b: __m128d) -> __m128d {
        let equal = _mm_cmpeq_pd(a, b);
        _mm_blendv_pd(_mm_max_pd(a, b), _mm_and_pd(a, b), equal)
    }

    /// FMINNM for double precision; see [`min_num_f32`].
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn min_num_f64(a: __m128d, b: __m128d) -> __m128d {
        let equal = _mm_cmpeq_pd(a, b);
        _mm_blendv_pd(_mm_min_pd(a, b), _mm_or_pd(a, b), equal)
    }

    /// FMAXP/FMINP/FMAXNMP/FMINNMP for double precision: the max or min of
    /// `n0:m0` and `n1:m1`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn pair_max_min_f64(max: bool, a: __m128d, b: __m128d) -> __m128d {
        // [n0 m0] against [n1 m1].
        let x = _mm_unpacklo_pd(a, b);
        let y = _mm_unpackhi_pd(a, b);
        if max {
            max_num_f64(x, y)
        } else {
            min_num_f64(x, y)
        }
    }

    /// The double-precision forms of [`vector3_f32`]: FADD, FSUB, FMUL,
    /// FMULX, FDIV, FMAX, FMIN, FMAXNM, FMINNM, FMLA, FMLS, FABD, FADDP,
    /// FMAXP, FMINP, FRECPS, FRSQRTS and the compares.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vector3_f64(kind: u32, d: u128, n: u128, m: u128) -> Option<u128> {
        let a = _mm_castsi128_pd(load(n));
        let b = _mm_castsi128_pd(load(m));
        let c = _mm_castsi128_pd(load(d));
        match vector_kind(kind)? {
            VectorKind::Two(Fp2::Add) => pack_f64(2, _mm_add_pd(a, b)),
            VectorKind::Two(Fp2::Sub) => pack_f64(2, _mm_sub_pd(a, b)),
            VectorKind::Two(Fp2::Div) => pack_f64(2, _mm_div_pd(a, b)),
            VectorKind::Two(Fp2::Mul | Fp2::MulX) => pack_f64(2, _mm_mul_pd(a, b)),
            VectorKind::Two(Fp2::Abd) => {
                let sign = _mm_castsi128_pd(_mm_set1_epi64x(i64::MIN));
                pack_f64(2, _mm_andnot_pd(sign, _mm_sub_pd(a, b)))
            }
            VectorKind::Two(Fp2::Max | Fp2::Min | Fp2::MaxNum | Fp2::MinNum)
                if any_nan_f64(a, b) =>
            {
                None
            }
            VectorKind::Two(Fp2::Max | Fp2::MaxNum) => pack_f64(2, max_num_f64(a, b)),
            VectorKind::Two(Fp2::Min | Fp2::MinNum) => pack_f64(2, min_num_f64(a, b)),
            VectorKind::Two(Fp2::RecipStep) => pack_f64(2, _mm_fnmadd_pd(a, b, _mm_set1_pd(2.0))),
            VectorKind::Two(Fp2::RsqrtStep) => pack_f64(
                2,
                _mm_fnmadd_pd(a, _mm_mul_pd(b, _mm_set1_pd(0.5)), _mm_set1_pd(1.5)),
            ),
            VectorKind::Fma(subtract) => {
                let value = if subtract {
                    _mm_fnmadd_pd(a, b, c)
                } else {
                    _mm_fmadd_pd(a, b, c)
                };
                pack_f64(2, value)
            }
            VectorKind::Pair(Fp2::Add) => pack_f64(2, _mm_hadd_pd(a, b)),
            VectorKind::Pair(Fp2::Max | Fp2::Min | Fp2::MaxNum | Fp2::MinNum)
                if any_nan_f64(a, b) =>
            {
                None
            }
            VectorKind::Pair(Fp2::Max | Fp2::MaxNum) => pack_f64(2, pair_max_min_f64(true, a, b)),
            VectorKind::Pair(Fp2::Min | Fp2::MinNum) => pack_f64(2, pair_max_min_f64(false, a, b)),
            VectorKind::Cmp(cmp) => compare_f64(cmp, 2, a, b),
            _ => None,
        }
    }

    /// The double-precision forms of [`vector_element_f32`]: FMLA, FMLS,
    /// FMUL and FMULX by element, scalar or two-lane.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn vector_element_f64(
        op: u32,
        scalar: bool,
        index: u32,
        d: u128,
        n: u128,
        m: u128,
    ) -> Option<u128> {
        let lanes = if scalar { 1 } else { 2 };
        let n = _mm_castsi128_pd(load(n));
        let d = _mm_castsi128_pd(load(d));
        let b = _mm_castsi128_pd(load(m));
        let element = if index & 1 == 0 {
            _mm_shuffle_pd::<0b00>(b, b)
        } else {
            _mm_shuffle_pd::<0b11>(b, b)
        };
        let value = match op {
            0 => _mm_fmadd_pd(n, element, d),
            1 => _mm_fnmadd_pd(n, element, d),
            _ => _mm_mul_pd(n, element),
        };
        pack_f64(lanes, value)
    }

    /// The sign bit of each lane, to turn signed compares into unsigned ones.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sign_bit(esize: u32) -> __m128i {
        match esize {
            8 => _mm_set1_epi8(0x80u8 as i8),
            16 => _mm_set1_epi16(0x8000u16 as i16),
            32 => _mm_set1_epi32(0x8000_0000u32 as i32),
            _ => _mm_set1_epi64x(i64::MIN),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn one(esize: u32) -> __m128i {
        match esize {
            8 => _mm_set1_epi8(1),
            16 => _mm_set1_epi16(1),
            32 => _mm_set1_epi32(1),
            _ => _mm_set1_epi64x(1),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_epi(esize: u32, a: __m128i, b: __m128i) -> Option<__m128i> {
        Some(match esize {
            8 => _mm_add_epi8(a, b),
            16 => _mm_add_epi16(a, b),
            32 => _mm_add_epi32(a, b),
            64 => _mm_add_epi64(a, b),
            _ => return None,
        })
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_epi(esize: u32, a: __m128i, b: __m128i) -> Option<__m128i> {
        Some(match esize {
            8 => _mm_sub_epi8(a, b),
            16 => _mm_sub_epi16(a, b),
            32 => _mm_sub_epi32(a, b),
            64 => _mm_sub_epi64(a, b),
            _ => return None,
        })
    }

    /// The low bits of a per-lane product; AVX2 has no 8- or 64-bit form.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_epi(esize: u32, a: __m128i, b: __m128i) -> Option<__m128i> {
        Some(match esize {
            16 => _mm_mullo_epi16(a, b),
            32 => _mm_mullo_epi32(a, b),
            _ => return None,
        })
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpeq_epi(esize: u32, a: __m128i, b: __m128i) -> Option<__m128i> {
        Some(match esize {
            8 => _mm_cmpeq_epi8(a, b),
            16 => _mm_cmpeq_epi16(a, b),
            32 => _mm_cmpeq_epi32(a, b),
            64 => _mm_cmpeq_epi64(a, b),
            _ => return None,
        })
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpgt_epi(esize: u32, unsigned: bool, a: __m128i, b: __m128i) -> Option<__m128i> {
        let (a, b) = if unsigned {
            let flip = sign_bit(esize);
            (_mm_xor_si128(a, flip), _mm_xor_si128(b, flip))
        } else {
            (a, b)
        };
        Some(match esize {
            8 => _mm_cmpgt_epi8(a, b),
            16 => _mm_cmpgt_epi16(a, b),
            32 => _mm_cmpgt_epi32(a, b),
            64 => _mm_cmpgt_epi64(a, b),
            _ => return None,
        })
    }

    /// CMTST: all ones where `a & b` is nonzero.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmptst_epi(esize: u32, a: __m128i, b: __m128i) -> Option<__m128i> {
        let equal = cmpeq_epi(esize, _mm_and_si128(a, b), _mm_setzero_si128())?;
        Some(_mm_xor_si128(equal, _mm_set1_epi32(-1)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn max_epi(esize: u32, unsigned: bool, a: __m128i, b: __m128i) -> Option<__m128i> {
        Some(match (esize, unsigned) {
            (8, false) => _mm_max_epi8(a, b),
            (8, true) => _mm_max_epu8(a, b),
            (16, false) => _mm_max_epi16(a, b),
            (16, true) => _mm_max_epu16(a, b),
            (32, false) => _mm_max_epi32(a, b),
            (32, true) => _mm_max_epu32(a, b),
            (64, false) => {
                let greater = _mm_cmpgt_epi64(a, b);
                _mm_blendv_epi8(b, a, greater)
            }
            (64, true) => {
                let flip = sign_bit(64);
                let greater = _mm_cmpgt_epi64(_mm_xor_si128(a, flip), _mm_xor_si128(b, flip));
                _mm_blendv_epi8(b, a, greater)
            }
            _ => return None,
        })
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn min_epi(esize: u32, unsigned: bool, a: __m128i, b: __m128i) -> Option<__m128i> {
        Some(match (esize, unsigned) {
            (8, false) => _mm_min_epi8(a, b),
            (8, true) => _mm_min_epu8(a, b),
            (16, false) => _mm_min_epi16(a, b),
            (16, true) => _mm_min_epu16(a, b),
            (32, false) => _mm_min_epi32(a, b),
            (32, true) => _mm_min_epu32(a, b),
            (64, false) => {
                let greater = _mm_cmpgt_epi64(a, b);
                _mm_blendv_epi8(a, b, greater)
            }
            (64, true) => {
                let flip = sign_bit(64);
                let greater = _mm_cmpgt_epi64(_mm_xor_si128(a, flip), _mm_xor_si128(b, flip));
                _mm_blendv_epi8(a, b, greater)
            }
            _ => return None,
        })
    }

    /// SABD: `max(a, b) - min(a, b)` in the lanes' own (unsigned) width.
    /// The difference of two signed lanes also fits there; the wrapping
    /// subtraction produces the same bits as the scalar `abs`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sabd_epi(esize: u32, unsigned: bool, a: __m128i, b: __m128i) -> Option<__m128i> {
        let high = max_epi(esize, unsigned, a, b)?;
        let low = min_epi(esize, unsigned, a, b)?;
        sub_epi(esize, high, low)
    }

    /// Shift each lane right by one, arithmetic or logical, without the
    /// overflow a widening add would have.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shr1_epi(esize: u32, unsigned: bool, v: __m128i) -> Option<__m128i> {
        Some(match (esize, unsigned) {
            (8, _) => {
                // There is no byte shift: a word shift with the bit from the
                // next byte, masked, is one.
                let shifted = _mm_and_si128(_mm_srli_epi16(v, 1), _mm_set1_epi8(0x7F));
                if unsigned {
                    shifted
                } else {
                    _mm_or_si128(shifted, _mm_and_si128(v, _mm_set1_epi8(0x80u8 as i8)))
                }
            }
            (16, true) => _mm_srli_epi16(v, 1),
            (16, false) => _mm_srai_epi16(v, 1),
            (32, true) => _mm_srli_epi32(v, 1),
            (32, false) => _mm_srai_epi32(v, 1),
            (64, true) => _mm_srli_epi64(v, 1),
            (64, false) => _mm_or_si128(
                _mm_srli_epi64(v, 1),
                _mm_and_si128(v, _mm_set1_epi64x(i64::MIN)),
            ),
            _ => return None,
        })
    }

    /// SHADD: `(a + b) >> 1`, floored, without overflowing the lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shadd_epi(esize: u32, unsigned: bool, a: __m128i, b: __m128i) -> Option<__m128i> {
        let half = add_epi(
            esize,
            shr1_epi(esize, unsigned, a)?,
            shr1_epi(esize, unsigned, b)?,
        )?;
        let carry = _mm_and_si128(_mm_and_si128(a, b), one(esize));
        add_epi(esize, half, carry)
    }

    /// SHSUB: `(a - b) >> 1`, floored.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn shsub_epi(esize: u32, unsigned: bool, a: __m128i, b: __m128i) -> Option<__m128i> {
        let half = sub_epi(
            esize,
            shr1_epi(esize, unsigned, a)?,
            shr1_epi(esize, unsigned, b)?,
        )?;
        // A zero low bit in `a` and a one in `b` rounds the difference down.
        let down = _mm_and_si128(_mm_andnot_si128(a, b), one(esize));
        sub_epi(esize, half, down)
    }

    /// ADDP: adjacent pairs of the concatenation n:m.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn addp_epi(esize: u32, q: bool, a: __m128i, b: __m128i) -> Option<__m128i> {
        Some(match esize {
            8 => {
                // There is no byte horizontal add; adjacent byte pairs sum
                // into 16-bit lanes (no saturation: 510 fits) and a shuffle
                // takes each sum's low byte.
                let pair_sums = |v: __m128i| {
                    let sums = _mm_maddubs_epi16(v, _mm_set1_epi8(1));
                    _mm_shuffle_epi8(
                        sums,
                        _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1),
                    )
                };
                let a = pair_sums(a);
                let b = pair_sums(b);
                if q {
                    _mm_unpacklo_epi64(a, b)
                } else {
                    // Four pairs from each source in the 64-bit form.
                    _mm_unpacklo_epi32(a, b)
                }
            }
            16 => {
                let pairs = _mm_hadd_epi16(a, b);
                if q {
                    pairs
                } else {
                    _mm_unpacklo_epi32(pairs, _mm_srli_si128::<8>(pairs))
                }
            }
            32 => {
                let pairs = _mm_hadd_epi32(a, b);
                if q {
                    pairs
                } else {
                    // [a0+a1, a2+a3, b0+b1, b2+b3] -> [a0+a1, b0+b1].
                    _mm_shuffle_epi32::<0x08>(pairs)
                }
            }
            64 => {
                let first = _mm_add_epi64(a, _mm_srli_si128::<8>(a));
                let second = _mm_add_epi64(b, _mm_srli_si128::<8>(b));
                _mm_unpacklo_epi64(first, second)
            }
            _ => return None,
        })
    }

    /// The integer three-same operations, one 128-bit register at a time.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn int_three(
        u: bool,
        opcode: u32,
        esize: u32,
        lanes: u32,
        d: u128,
        n: u128,
        m: u128,
    ) -> Option<u128> {
        let bits = lanes * esize;
        if bits != 64 && bits != 128 {
            return None;
        }
        let a = load(n);
        let b = load(m);
        let c = load(d);
        let result = match opcode {
            // ADD and SUB.
            0b10000 => {
                if u {
                    sub_epi(esize, a, b)?
                } else {
                    add_epi(esize, a, b)?
                }
            }
            0b00000 => shadd_epi(esize, u, a, b)?,
            0b00100 => shsub_epi(esize, u, a, b)?,
            0b00110 => cmpgt_epi(esize, u, a, b)?,
            0b00111 => {
                let greater = cmpgt_epi(esize, u, a, b)?;
                let equal = cmpeq_epi(esize, a, b)?;
                _mm_or_si128(greater, equal)
            }
            0b01100 => max_epi(esize, u, a, b)?,
            0b01101 => min_epi(esize, u, a, b)?,
            0b01110 => sabd_epi(esize, u, a, b)?,
            0b01111 => {
                let difference = sabd_epi(esize, u, a, b)?;
                add_epi(esize, difference, c)?
            }
            // CMTST and CMEQ.
            0b10001 => {
                if u {
                    cmpeq_epi(esize, a, b)?
                } else {
                    cmptst_epi(esize, a, b)?
                }
            }
            // MLA and MLS.
            0b10010 => {
                let product = mul_epi(esize, a, b)?;
                if u {
                    sub_epi(esize, c, product)?
                } else {
                    add_epi(esize, c, product)?
                }
            }
            0b10011 if !u => mul_epi(esize, a, b)?,
            0b10111 if !u => addp_epi(esize, bits == 128, a, b)?,
            _ => return None,
        };
        let mut value = store(result);
        if bits < 128 {
            value &= (1u128 << bits) - 1;
        }
        Some(value)
    }
}

/// AND, BIC, ORR, ORN, EOR, BSL, BIT, BIF (`kind` is U << 2 | size).
pub fn vector_bitwise(kind: u32, q: bool, d: u128, n: u128, m: u128) -> u128 {
    let (u, size) = (kind & 4 != 0, kind & 3);
    let result = match (u, size) {
        (false, 0) => n & m,
        (false, 1) => n & !m,
        (false, 2) => n | m,
        (false, _) => n | !m,
        (true, 0) => n ^ m,
        (true, 1) => (d & n) | (!d & m),
        (true, 2) => (d & !m) | (n & m),
        _ => (d & m) | (n & !m),
    };
    if q {
        result
    } else {
        low(result)
    }
}

/// REV64, REV32, REV16: reverse the elements within each `container` bits.
pub fn vector_rev(container: u32, esize: u32, q: bool, n: u128) -> u128 {
    let per = container / esize;
    let mut result = 0u128;
    for i in 0..width(q) / esize {
        let base = i / per * per;
        result = put(
            result,
            esize,
            i,
            lane(n, esize, base + per - 1 - (i - base)),
        );
    }
    result
}

/// SADDLP, UADDLP, SADALP, UADALP: pairwise sums into lanes twice as wide.
pub fn vector_pairwise_sum(
    u: bool,
    accumulate: bool,
    esize: u32,
    lanes: u32,
    d: u128,
    n: u128,
) -> u128 {
    let wide = esize * 2;
    let mut result = 0u128;
    for i in 0..lanes {
        let mut sum =
            int(lane(n, esize, 2 * i), esize, u) + int(lane(n, esize, 2 * i + 1), esize, u);
        if accumulate {
            sum += i128::from(lane(d, wide, i));
        }
        result = put(result, wide, i, wrap(sum, wide));
    }
    result
}

/// Two-register integer forms (`kind` is U << 7 | size << 5 | opcode): CLS,
/// CLZ, CNT, NOT, RBIT, SUQADD, USQADD, SQABS, SQNEG, and compares with zero.
pub fn vector_int2(kind: u32, esize: u32, lanes: u32, d: u128, n: u128) -> u128 {
    let (u, size, opcode) = (kind & 0x80 != 0, (kind >> 5) & 3, kind & 0x1F);
    let mut result = 0u128;
    match (u, opcode) {
        (_, 0b00100) => {
            for i in 0..lanes {
                let x = lane(n, esize, i);
                let count = if u {
                    (x << (64 - esize)).leading_zeros().min(esize)
                } else {
                    let t = (x ^ (x >> 1)) & mask(esize - 1);
                    esize - 1 - (64 - t.leading_zeros())
                };
                result = put(result, esize, i, u64::from(count));
            }
        }
        (_, 0b00101) => {
            for i in 0..lanes {
                let x = lane(n, 8, i) as u8;
                let value = match (u, size) {
                    (false, _) => x.count_ones() as u8,
                    (true, 0) => !x,
                    _ => x.reverse_bits(),
                };
                result = put(result, 8, i, u64::from(value));
            }
        }
        (_, 0b00011) => {
            for i in 0..lanes {
                let (a, b) = (lane(d, esize, i), lane(n, esize, i));
                let value = if u {
                    sat(int(a, esize, true) + int(b, esize, false), esize, true)
                } else {
                    sat(int(a, esize, false) + int(b, esize, true), esize, false)
                };
                result = put(result, esize, i, value);
            }
        }
        (_, 0b00111) | (_, 0b01000..=0b01011) => {
            for i in 0..lanes {
                let x = int(lane(n, esize, i), esize, false);
                let value = match (u, opcode) {
                    (false, 0b00111) => sat(x.abs(), esize, false),
                    (true, 0b00111) => sat(-x, esize, false),
                    (false, 0b01000) => ones(x > 0, esize),
                    (true, 0b01000) => ones(x >= 0, esize),
                    (false, 0b01001) => ones(x == 0, esize),
                    (true, 0b01001) => ones(x <= 0, esize),
                    (false, 0b01010) => ones(x < 0, esize),
                    (false, 0b01011) => wrap(x.abs(), esize),
                    _ => wrap(-x, esize),
                };
                result = put(result, esize, i, value);
            }
        }
        _ => {}
    }
    result
}

/// XTN, SQXTUN, SQXTN, UQXTN. `mode` 0 writes the low half and zeroes the
/// upper, 1 keeps the destination's low half and writes the upper, 2 returns
/// the scalar result (`lanes` is then 1).
pub fn vector_narrow(kind: u32, esize: u32, mode: u32, lanes: u32, n: u128, d: u128) -> u128 {
    let wide = esize * 2;
    let mut narrowed = 0u128;
    for i in 0..lanes {
        let x = lane(n, wide, i);
        let value = match kind {
            0 => wrap(i128::from(x), esize),
            1 => sat(int(x, wide, false), esize, false),
            2 => sat(int(x, wide, false), esize, true),
            _ => sat(int(x, wide, true), esize, true),
        };
        narrowed = put(narrowed, esize, i, value);
    }
    match mode {
        1 => low(d) | (narrowed << 64),
        2 => narrowed,
        _ => low(narrowed),
    }
}

/// SHLL: shift each element of the low (or high) half into a lane twice as
/// wide.
pub fn vector_shift_long(q: bool, esize: u32, n: u128) -> u128 {
    let source = half(n, q);
    let mut result = 0u128;
    for i in 0..64 / esize {
        result = put(result, esize * 2, i, lane(source, esize, i) << esize);
    }
    result
}

/// One-source floating-point operations (`op`: 0 FABS, 1 FNEG, 2 FSQRT,
/// 3 FRINTN, 4 FRINTP, 5 FRINTM, 6 FRINTZ, 7 FRINTA, 8 FRECPE, 9 FRSQRTE).
pub fn vector_fp1(op: u32, esize: u32, lanes: u32, n: u128) -> u128 {
    if op < 2 {
        // FABS and FNEG only touch the sign.
        let bits = lanes * esize;
        let sign_bit = 1u128 << (esize - 1);
        let mut mask = 0u128;
        for i in 0..lanes {
            mask |= sign_bit << (i * esize);
        }
        let field = if bits == 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        return (if op == 0 { n & !mask } else { n ^ mask }) & field;
    }
    let op = match op {
        2 => Fp1::Sqrt,
        3 => Fp1::Round(Rounding::TiesEven),
        4 => Fp1::Round(Rounding::Up),
        5 => Fp1::Round(Rounding::Down),
        6 => Fp1::Round(Rounding::Zero),
        7 => Fp1::Round(Rounding::TiesAway),
        8 => Fp1::RecipEstimate,
        _ => Fp1::RsqrtEstimate,
    };
    let mut result = 0u128;
    for i in 0..lanes {
        let value = sized!(esize, fp1(op, lane(n, esize, i)));
        result = put(result, esize, i, value);
    }
    result
}

/// FCMGT, FCMEQ, FCMLT, FCMGE, FCMLE against zero (`cmp`: 0 GT, 1 EQ,
/// 2 LT, 3 GE, 4 LE).
pub fn vector_fp_cmp0(cmp: u32, esize: u32, lanes: u32, n: u128) -> u128 {
    let cmp = match cmp {
        0 => Cmp::Gt,
        1 => Cmp::Eq,
        2 => Cmp::Lt,
        3 => Cmp::Ge,
        _ => Cmp::Le,
    };
    let mut result = 0u128;
    for i in 0..lanes {
        let value = ones(sized!(esize, fp_cmp(cmp, lane(n, esize, i), 0)), esize);
        result = put(result, esize, i, value);
    }
    result
}

/// FCVT[NPMZA][SU] to integer (`rounding`: 0 ties even, 1 down, 2 ties away,
/// 3 up, 4 zero).
pub fn vector_to_int(rounding: u32, unsigned: bool, esize: u32, lanes: u32, n: u128) -> u128 {
    let rounding = match rounding {
        0 => Rounding::TiesEven,
        1 => Rounding::Down,
        2 => Rounding::TiesAway,
        3 => Rounding::Up,
        _ => Rounding::Zero,
    };
    let mut result = 0u128;
    for i in 0..lanes {
        let value = sized!(
            esize,
            fp_to_int(lane(n, esize, i), rounding, 0, unsigned, esize)
        );
        result = put(result, esize, i, value);
    }
    result
}

/// SCVTF, UCVTF from integer.
pub fn vector_from_int(unsigned: bool, esize: u32, lanes: u32, n: u128) -> u128 {
    let mut result = 0u128;
    for i in 0..lanes {
        let value = sized!(esize, int_to_fp(lane(n, esize, i), unsigned, esize, 0));
        result = put(result, esize, i, value);
    }
    result
}

/// Across-lanes reductions (`kind` is U << 5 | opcode): ADDV, SADDLV,
/// UADDLV, SMAXV, UMAXV, SMINV, UMINV.
pub fn vector_across(kind: u32, esize: u32, lanes: u32, n: u128) -> u128 {
    let (u, opcode) = (kind & 0x20 != 0, kind & 0x1F);
    let values = (0..lanes).map(|i| int(lane(n, esize, i), esize, u));
    let result = match (u, opcode) {
        (_, 0b00011) => wrap(values.sum(), esize * 2),
        (_, 0b01010) => wrap(values.max().unwrap_or(0), esize),
        (_, 0b11010) => wrap(values.min().unwrap_or(0), esize),
        _ => wrap(values.sum(), esize),
    };
    u128::from(result)
}

/// SCVTF, UCVTF, FCVTZS, FCVTZU with a fixed-point immediate: scale by
/// 2^-`fbits` (to float) or 2^`fbits` (to integer).
#[allow(clippy::too_many_arguments)]
pub fn vector_fixed_convert(
    to_float: bool,
    unsigned: bool,
    esize: u32,
    lanes: u32,
    fbits: u32,
    n: u128,
) -> u128 {
    let mut result = 0u128;
    for i in 0..lanes {
        let x = lane(n, esize, i);
        let value = if to_float {
            sized!(esize, int_to_fp(x, unsigned, esize, fbits))
        } else {
            sized!(esize, fp_to_int(x, Rounding::Zero, fbits, unsigned, esize))
        };
        result = put(result, esize, i, value);
    }
    result
}

/// SSHR, USHR, SRSHR, URSHR, SRI, SHL, SLI, SQSHL, UQSHL, SQSHLU: shift
/// each element by the immediate's left or right amount.
#[allow(clippy::too_many_arguments)]
pub fn vector_shift(
    u: bool,
    opcode: u32,
    esize: u32,
    lanes: u32,
    imm: u32,
    d: u128,
    n: u128,
) -> u128 {
    let right = 2 * esize - imm;
    let left = imm - esize;
    let mut result = 0u128;
    for i in 0..lanes {
        let raw = lane(n, esize, i);
        let x = int(raw, esize, u);
        let acc = lane(d, esize, i);
        let round = 1i128 << (right - 1);
        let value = match (u, opcode) {
            (_, 0b00000) => wrap(x >> right, esize),
            (_, 0b00010) => wrap(i128::from(acc) + (x >> right), esize),
            (_, 0b00100) => wrap((x + round) >> right, esize),
            (_, 0b00110) => wrap(i128::from(acc) + ((x + round) >> right), esize),
            (true, 0b01000) => {
                // SRI
                let field = mask(esize).checked_shr(right).unwrap_or(0);
                (acc & !field) | (raw.checked_shr(right).unwrap_or(0) & field)
            }
            (false, 0b01010) => wrap(i128::from(raw) << left, esize),
            (true, 0b01010) => {
                // SLI
                let field = (mask(esize) << left) & mask(esize);
                (acc & !field) | ((raw << left) & field)
            }
            (true, 0b01100) => sat(int(raw, esize, false) << left, esize, true),
            _ => sat(x << left, esize, u),
        };
        result = put(result, esize, i, value);
    }
    result
}

/// FCCMP, FCCMPE: compare, or the immediate flags when the condition fails.
pub fn scalar_fccmp(double: bool, cond: u32, nzcv: u8, n: u128, m: u128, immediate: u32) -> u8 {
    if condition_holds(cond, nzcv) {
        scalar_compare(double, n, m)
    } else {
        immediate as u8
    }
}

/// Run a specialized call on register values, as generated code does.
/// For tests and tools; generated code calls the functions directly.
pub fn run(call: &Call, vector: &dyn Fn(u8) -> u128, general: u64, nzcv: u8) -> u128 {
    let int = |index: usize| match call.args[index] {
        Arg::Int(value) => value,
        _ => panic!("argument {index} of {} is not an integer", call.function),
    };
    let flag = |index: usize| match call.args[index] {
        Arg::Bool(value) => value,
        _ => panic!("argument {index} of {} is not a flag", call.function),
    };
    let value = |index: usize| match call.args[index] {
        Arg::Vector(reg) => vector(reg),
        Arg::Constant(value) => value,
        Arg::General => u128::from(general),
        _ => panic!("argument {index} of {} is not a value", call.function),
    };
    match call.function {
        "scalar_fp2" => scalar_fp2(int(0), flag(1), value(2), value(3)),
        "scalar_fma" => scalar_fma(int(0), flag(1), value(2), value(3), value(4)),
        "scalar_compare" => u128::from(scalar_compare(flag(0), value(1), value(2))),
        "scalar_select" => scalar_select(flag(0), int(1), nzcv, value(3), value(4)),
        "scalar_fp1" => scalar_fp1(int(0), int(1), value(2)),
        "constant" => constant(value(0)),
        "low_bits" => low_bits(int(0), value(1)),
        "from_general" => from_general(int(0), general),
        "int_to_float" => int_to_float(flag(0), flag(1), flag(2), general),
        "float_to_int" => float_to_int(flag(0), int(1), flag(2), value(3)),
        "vector3" => vector3(int(0), flag(1), flag(2), value(3), value(4), value(5)),
        "vector_element" => vector_element(
            int(0),
            flag(1),
            flag(2),
            flag(3),
            int(4),
            value(5),
            value(6),
            value(7),
        ),
        "extract" => extract(flag(0), int(1), value(2), value(3)),
        "element" => element(int(0), int(1), value(2)),
        "permute" => permute(int(0), int(1), flag(2), value(3), value(4)),
        "int_three" => int_three(
            flag(0),
            int(1),
            int(2),
            int(3),
            value(4),
            value(5),
            value(6),
        ),
        "vector_bitwise" => vector_bitwise(int(0), flag(1), value(2), value(3), value(4)),
        "vector_rev" => vector_rev(int(0), int(1), flag(2), value(3)),
        "vector_pairwise_sum" => {
            vector_pairwise_sum(flag(0), flag(1), int(2), int(3), value(4), value(5))
        }
        "vector_int2" => vector_int2(int(0), int(1), int(2), value(3), value(4)),
        "vector_narrow" => vector_narrow(int(0), int(1), int(2), int(3), value(4), value(5)),
        "vector_shift_long" => vector_shift_long(flag(0), int(1), value(2)),
        "vector_fp1" => vector_fp1(int(0), int(1), int(2), value(3)),
        "vector_fp_cmp0" => vector_fp_cmp0(int(0), int(1), int(2), value(3)),
        "vector_to_int" => vector_to_int(int(0), flag(1), int(2), int(3), value(4)),
        "vector_from_int" => vector_from_int(flag(0), int(1), int(2), value(3)),
        "vector_across" => vector_across(int(0), int(1), int(2), value(3)),
        "vector_fixed_convert" => {
            vector_fixed_convert(flag(0), flag(1), int(2), int(3), int(4), value(5))
        }
        "vector_shift" => vector_shift(flag(0), int(1), int(2), int(3), int(4), value(5), value(6)),
        "scalar_fccmp" => u128::from(scalar_fccmp(
            flag(0),
            int(1),
            nzcv,
            value(3),
            value(4),
            int(5),
        )),
        other => panic!("no specialized function {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// Lanes that exercise the special cases: zeros, infinities, NaNs (quiet
    /// and signalling), denormals, extremes, and ordinary numbers.
    fn special_lane(rng: &mut Rng, size: u32) -> u64 {
        let pick = rng.next() % 16;
        let values32: [u32; 12] = [
            0,
            0x8000_0000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0001,
            0x7F80_0001,
            0x0000_0001,
            0x3F80_0000,
            0xBF80_0000,
            0x7F7F_FFFF,
            0x4F00_0000,
            0xCF00_0001,
        ];
        let values64: [u64; 12] = [
            0,
            0x8000_0000_0000_0000,
            0x7FF0_0000_0000_0000,
            0xFFF0_0000_0000_0000,
            0x7FF8_0000_0000_0001,
            0x7FF0_0000_0000_0001,
            1,
            0x3FF0_0000_0000_0000,
            0xBFF0_0000_0000_0000,
            0x7FEF_FFFF_FFFF_FFFF,
            0x43E0_0000_0000_0000,
            0xC3E0_0000_0000_0001,
        ];
        if (pick as usize) < values32.len() {
            if size == 32 {
                u64::from(values32[pick as usize])
            } else {
                values64[pick as usize]
            }
        } else {
            rng.next() & mask(size)
        }
    }

    fn register(rng: &mut Rng) -> u128 {
        let size = if rng.next() & 1 == 0 { 32 } else { 64 };
        let mut value = 0u128;
        for i in 0..128 / size {
            value = put(value, size, i, special_lane(rng, size));
        }
        value
    }

    /// Integer lanes that include the wrapping and signedness edges.
    fn int_register(rng: &mut Rng, esize: u32) -> u128 {
        let lanes = 128 / esize;
        let edge = |rng: &mut Rng| -> u64 {
            let full = mask(esize);
            match rng.next() % 8 {
                0 => 0,
                1 => full,
                2 => 1,
                3 => full - 1,
                4 => 1u64 << (esize - 1),
                5 => (1u64 << (esize - 1)) - 1,
                6 => (1u64 << (esize - 1)) | 1,
                _ => rng.next() & full,
            }
        };
        let mut value = 0u128;
        for i in 0..lanes {
            value = put(value, esize, i, edge(rng));
        }
        value
    }

    /// A vector register with the floating-point edge lanes at `esize`.
    fn fp_register(rng: &mut Rng, esize: u32) -> u128 {
        let mut value = 0u128;
        for i in 0..128 / esize {
            value = put(value, esize, i, special_lane(rng, esize));
        }
        value
    }

    /// A vector register with finite lanes only: operations on these cannot
    /// produce a NaN, so the host fast path must take them.
    fn finite_register(rng: &mut Rng, esize: u32) -> u128 {
        let values32: [u32; 8] = [
            0,
            0x8000_0000,
            0x3F80_0000,
            0xBF80_0000,
            0x0000_0001,
            0x0080_0000,
            0x7F7F_FFFF,
            0x4F00_0000,
        ];
        let values64: [u64; 8] = [
            0,
            0x8000_0000_0000_0000,
            0x3FF0_0000_0000_0000,
            0xBFF0_0000_0000_0000,
            1,
            0x0010_0000_0000_0000,
            0x7FEF_FFFF_FFFF_FFFF,
            0x43E0_0000_0000_0000,
        ];
        let mut value = 0u128;
        for i in 0..128 / esize {
            let pick = (rng.next() % values32.len() as u64) as usize;
            let lane = if esize == 32 {
                u64::from(values32[pick])
            } else {
                values64[pick]
            };
            value = put(value, esize, i, lane);
        }
        value
    }

    /// Three-same encoding with v1 and v2 as sources and v3 as destination.
    fn three_same_word(u: bool, size: u32, opcode: u32, q: bool) -> u32 {
        let group = if q { 0x4E00_0000 } else { 0x0E00_0000 };
        group
            | (u32::from(u) << 29)
            | (size << 22)
            | (1 << 21)
            | (1 << 10)
            | (opcode << 11)
            | 3
            | (1 << 5)
            | (2 << 16)
    }

    /// By-element encoding for FMLA/FMLS/FMUL/FMULX (`op` 0..=3).
    fn element_word(op: u32, double: bool, q: bool, scalar: bool, index: u32) -> u32 {
        let (u, opcode) = match op {
            0 => (false, 0b0001),
            1 => (false, 0b0101),
            2 => (false, 0b1001),
            _ => (true, 0b1001),
        };
        let group = if scalar {
            0x5F00_0000
        } else if q {
            0x4F00_0000
        } else {
            0x0F00_0000
        };
        let size = if double { 3 } else { 2 };
        let (l, h) = if double {
            (0, index << 11)
        } else {
            ((index & 1) << 21, (index >> 1) << 11)
        };
        group
            | (u32::from(u) << 29)
            | (size << 22)
            | l
            | (2 << 16)
            | (opcode << 12)
            | h
            | (1 << 5)
            | 3
    }

    #[test]
    fn specialized_calls_compute_what_exec_does() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let groups: [u32; 16] = [
            0x1E, 0x9E, 0x1F, 0x9F, 0x0E, 0x4E, 0x2E, 0x6E, 0x0F, 0x4F, 0x2F, 0x6F, 0x5E, 0x5F,
            0x7E, 0xFE,
        ];
        let mut covered: std::collections::BTreeMap<&str, u32> = Default::default();
        // Narrow encodings random words rarely hit: (fixed bits, free bits).
        let templates: [(u32, u32); 14] = [
            // FCMP, FCMPE (with zero)
            (
                0x1E20_2000,
                (1 << 22) | (0x1F << 16) | (0x1F << 5) | (3 << 3),
            ),
            // FMOV between general and vector registers
            (0x1E26_0000, (1 << 16) | 0x3FF),
            (0x9E66_0000, (1 << 16) | 0x3FF),
            // SCVTF, UCVTF
            (0x1E22_0000, (1 << 31) | (1 << 22) | (1 << 16) | 0x3FF),
            // FCVT* to integer
            (
                0x1E20_0000,
                (1 << 31) | (1 << 22) | (3 << 19) | (7 << 16) | 0x3FF,
            ),
            // FMOV immediate
            (0x1E20_1000, (1 << 22) | (0xFF << 13) | 0x1F),
            // FP one source
            (0x1E20_4000, (1 << 22) | (0x3F << 15) | 0x3FF),
            // Three same, integer and floating point (ADDP, compares, ...)
            (0x0E20_0400, 0x60DF_FBFF),
            // Two-register miscellaneous: REV, ABS/NEG, compares with zero,
            // narrowing, and the floating-point conversions and estimates.
            (0x0E20_0800, 0x60C1_F3FF),
            // The same, scalar.
            (0x5E20_0800, 0x20C1_F3FF),
            // Across lanes: ADDV, SMAXV, SMINV, SADDLV.
            (0x0E30_0800, 0x60C1_F3FF),
            // Shift by immediate: fixed-point conversions and shifts.
            (0x4F00_0400, 0x207F_FBFF),
            (0x5F00_0400, 0x207F_FBFF),
            // FCCMP, FCCMPE
            (0x1E20_0400, 0x00DF_F3EF),
        ];
        for iteration in 0..3_000_000u32 {
            let word = if iteration % 3 == 0 {
                let (fixed, free) = templates[(rng.next() % templates.len() as u64) as usize];
                fixed | (rng.next() as u32 & free)
            } else {
                let group = groups[(rng.next() % groups.len() as u64) as usize];
                (group << 24) | (rng.next() as u32 & 0x00FF_FFFF)
            };
            let Some(call) = specialize(word) else {
                continue;
            };
            let operands = operands(word).unwrap_or_else(|| {
                panic!(
                    "{word:#010x} is specialized as {} but exec does not run it",
                    call.function
                )
            });
            let file: Vec<u128> = (0..32).map(|_| register(&mut rng)).collect();
            let general = if rng.next() & 3 == 0 {
                special_lane(&mut rng, 64)
            } else {
                rng.next()
            };
            let nzcv = (rng.next() & 0xF) as u8;
            let vector: Vec<(u8, u128)> = operands
                .vector_reads
                .iter()
                .map(|reg| (*reg, file[*reg as usize]))
                .collect();
            let expected = exec(word, &vector, general, nzcv)
                .unwrap_or_else(|| panic!("exec rejects {word:#010x}"));
            // Specialized calls may name registers exec does not read (and
            // must not depend on them): read them from the same file.
            let got = run(&call, &|reg| file[reg as usize], general, nzcv);
            match call.output {
                Output::Value => assert_eq!(
                    got, expected.value,
                    "{word:#010x} ({}) with {vector:x?}, x {general:#x}, nzcv {nzcv:#x}",
                    call.function
                ),
                Output::Flags => assert_eq!(
                    got as u8, expected.nzcv,
                    "{word:#010x} ({}) with {vector:x?}",
                    call.function
                ),
            }
            *covered.entry(call.function).or_default() += 1;
        }
        for function in [
            "scalar_fp2",
            "scalar_fma",
            "scalar_compare",
            "scalar_select",
            "scalar_fp1",
            "constant",
            "low_bits",
            "from_general",
            "int_to_float",
            "float_to_int",
            "vector3",
            "vector_element",
            "extract",
            "element",
            "permute",
            "int_three",
            "vector_bitwise",
            "vector_rev",
            "vector_pairwise_sum",
            "vector_int2",
            "vector_narrow",
            "vector_shift_long",
            "vector_fp1",
            "vector_fp_cmp0",
            "vector_to_int",
            "vector_from_int",
            "vector_across",
            "vector_fixed_convert",
            "vector_shift",
            "scalar_fccmp",
        ] {
            let count = covered.get(function).copied().unwrap_or(0);
            assert!(
                count >= 200,
                "{function} only covered {count} times: {covered:?}"
            );
        }
    }

    /// Check a word if `specialize` claims it: `exec` must accept it and the
    /// specialized call must match on four register files.
    fn check_specialized(
        word: u32,
        rng: &mut Rng,
        covered: &mut std::collections::BTreeMap<&'static str, u32>,
    ) {
        let Some(call) = specialize(word) else {
            return;
        };
        let operands = operands(word).unwrap_or_else(|| {
            panic!(
                "{word:#010x} is specialized as {} but exec does not run it",
                call.function
            )
        });
        for _ in 0..8 {
            let file: Vec<u128> = (0..32).map(|_| register(rng)).collect();
            let general = rng.next();
            let nzcv = (rng.next() & 0xF) as u8;
            let vector: Vec<(u8, u128)> = operands
                .vector_reads
                .iter()
                .map(|reg| (*reg, file[*reg as usize]))
                .collect();
            let expected = exec(word, &vector, general, nzcv)
                .unwrap_or_else(|| panic!("exec rejects {word:#010x}"));
            let got = run(&call, &|reg| file[reg as usize], general, nzcv);
            match call.output {
                Output::Value => assert_eq!(
                    got, expected.value,
                    "{word:#010x} ({}) with {vector:x?}",
                    call.function
                ),
                Output::Flags => assert_eq!(
                    got as u8, expected.nzcv,
                    "{word:#010x} ({}) with {vector:x?}",
                    call.function
                ),
            }
            *covered.entry(call.function).or_default() += 1;
        }
    }

    #[test]
    fn every_field_combination_matches_exec() {
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        let mut covered: std::collections::BTreeMap<&str, u32> = Default::default();
        for group in [0x0Eu32, 0x4E, 0x2E, 0x6E, 0x5E, 0x7E] {
            let top = group << 24;
            // Three same.
            for size in 0..4 {
                for opcode in 0..32 {
                    let word =
                        top | (size << 22) | (1 << 21) | (1 << 10) | (opcode << 11) | 0x0000_0041;
                    check_specialized(word, &mut rng, &mut covered);
                }
            }
            // Two-register miscellaneous and across lanes.
            for size in 0..4 {
                for bits in [0u32, 8] {
                    for opcode in 0..32 {
                        let word = top
                            | (size << 22)
                            | (1 << 21)
                            | (1 << 11)
                            | (bits << 17)
                            | (opcode << 12)
                            | 0x0000_0041;
                        check_specialized(word, &mut rng, &mut covered);
                    }
                }
            }
        }
        // Shift by immediate.
        for group in [0x0Fu32, 0x4F, 0x2F, 0x6F, 0x5F, 0x7F] {
            for imm in 1..128u32 {
                for opcode in 0..32u32 {
                    let word =
                        (group << 24) | (imm << 16) | (1 << 10) | (opcode << 11) | 0x0000_0041;
                    check_specialized(word, &mut rng, &mut covered);
                }
            }
        }
        // Scalar FCCMP.
        for ftype in 0..4u32 {
            for cond in 0..16u32 {
                for imm in 0..16u32 {
                    let word = 0x1E20_0000 | (ftype << 22) | (cond << 12) | (1 << 10) | imm;
                    check_specialized(word, &mut rng, &mut covered);
                }
            }
        }
        for function in [
            "int_three",
            "vector_bitwise",
            "vector_rev",
            "vector_pairwise_sum",
            "vector_int2",
            "vector_narrow",
            "vector_shift_long",
            "vector_fp1",
            "vector_fp_cmp0",
            "vector_to_int",
            "vector_from_int",
            "vector_across",
            "vector_fixed_convert",
            "vector_shift",
            "scalar_fccmp",
        ] {
            let count = covered.get(function).copied().unwrap_or(0);
            assert!(
                count >= 40,
                "{function} only covered {count} times: {covered:?}"
            );
        }
    }

    #[test]
    fn the_common_instructions_are_specialized() {
        assert!(specialize(0x7EA1_D402).is_some(), "scalar FABD");
        // fmul s0, s1, s0; fmadd; fcmp s0, #0; fmla v.4s; fmla by element;
        // fadd v.4s; ext; scvtf s0, w8; fcvtzs w8, s7; mov s0, v1.s[1].
        for word in [
            0x1E20_0820u32,
            0x1F00_0C20,
            0x1E20_2008,
            0x4E20_CC20,
            0x4F80_1020,
            0x4E20_D420,
            0x6E00_4020,
            0x1E22_0100,
            0x1E38_00E8,
            0x5E0C_0420,
            // zip2 v0.4s, v0.4s, v5.4s (LM3's job code)
            0x4E85_7800,
            // The hottest words RECOMP_SIMD_STATS found in the evaluator:
            // ucvtf s0, w0; frecpe v2.4s; addp v0.2s; cmlt v0.2s, #0;
            // scvtf v3.4s; frsqrte v2.2s; fcvtzu v0.4s, #17; sqxtn v0.4h;
            // fabs v0.2s; fccmp s0, s1, #8, le.
            0x7E21_D800,
            0x4EA1_D822,
            0x4EA1_DA95,
            0x4EA1_DA53,
            0x0EA0_BC00,
            0x0EA1_BC00,
            0x0EA0_0A94,
            0x0EA0_0AB5,
            0x0EA0_0A31,
            0x0EA0_08A5,
            0x0EA0_0AF7,
            0x0EA0_094A,
            0x0EA0_0908,
            0x4E21_D863,
            0x2EA1_DA12,
            0x4F31_FC00,
            0x0E61_4800,
            0x0EA0_F800,
            0x2EA1_DA14,
            0x1E3C_A7A8,
            0x1E29_B560,
            // Other hot two-register and shift forms:
            // rev64 v16.2s, v16.2s; cmlt v2.2s, v2.2s, #0;
            // sshr v4.8h, v2.8h, #15; shl v2.4s, v1.4s, #16;
            // srshr v2.4s, v2.4s, #16.
            0x0EA0_0A10,
            0x0EA0_0A94,
            0x4F11_0444,
            0x4F30_5422,
            0x4F30_2442,
        ] {
            assert!(
                specialize(word).is_some(),
                "{word:#010x} is not specialized"
            );
        }
    }

    /// Every integer three-same form the host handles, at every lane size
    /// and both vector widths, against `exec` with wrapping and signedness
    /// edges. The AVX2 helper itself is checked when the host has it.
    #[test]
    fn integer_three_same_matches_exec_on_edge_values() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF0);
        let cases: [(u32, bool); 24] = [
            (0b10000, false), // ADD
            (0b10000, true),  // SUB
            (0b00000, false), // SHADD
            (0b00000, true),
            (0b00100, false), // SHSUB
            (0b00100, true),
            (0b00110, false), // CMGT
            (0b00110, true),
            (0b00111, false), // CMGE
            (0b00111, true),
            (0b01100, false), // SMAX
            (0b01100, true),
            (0b01101, false), // SMIN
            (0b01101, true),
            (0b01110, false), // SABD
            (0b01110, true),
            (0b01111, false), // SABA
            (0b01111, true),
            (0b10001, false), // CMTST
            (0b10001, true),  // CMEQ
            (0b10010, false), // MLA
            (0b10010, true),  // MLS
            (0b10011, false), // MUL
            (0b10111, false), // ADDP
        ];
        #[cfg(target_arch = "x86_64")]
        let mut fast_hits = 0u32;
        for size in 0..4u32 {
            for q in [false, true] {
                if size == 3 && !q {
                    continue;
                }
                let esize = 8 << size;
                let lanes = width(q) / esize;
                for (opcode, u) in cases {
                    let word = three_same_word(u, size, opcode, q);
                    for _ in 0..32 {
                        let n = int_register(&mut rng, esize);
                        let m = int_register(&mut rng, esize);
                        let d = int_register(&mut rng, esize);
                        let Some(expected) = exec(word, &[(1, n), (2, m), (3, d)], 0, 0) else {
                            continue;
                        };
                        let accumulates = matches!(opcode, 0b01111 | 0b10010);
                        let d = if accumulates { d } else { 0 };
                        let got = int_three(u, opcode, esize, lanes, d, n, m);
                        assert_eq!(
                            got, expected.value,
                            "{word:#010x} (u {u}, opcode {opcode:#07b}, esize {esize}, \
                             lanes {lanes}) with n {n:#034x}, m {m:#034x}, d {d:#034x}"
                        );
                        // The portable fallback must agree whether or not
                        // this host has AVX2.
                        assert_eq!(
                            int_three_scalar(u, opcode, esize, lanes, d, n, m),
                            expected.value,
                            "{word:#010x} scalar (esize {esize}) with n {n:#034x}, \
                             m {m:#034x}, d {d:#034x}"
                        );
                        #[cfg(target_arch = "x86_64")]
                        if x86::avx2() {
                            // 8- and 64-bit MUL have no AVX2 form; the
                            // helper must decline them instead of guessing.
                            let unsupported = (opcode == 0b10011 && matches!(esize, 8 | 64))
                                || (opcode == 0b10010 && esize == 8);
                            let fast = unsafe { x86::int_three(u, opcode, esize, lanes, d, n, m) };
                            if unsupported {
                                assert_eq!(
                                    fast, None,
                                    "{word:#010x} should stay scalar at esize {esize}"
                                );
                            } else {
                                assert_eq!(
                                    fast,
                                    Some(expected.value),
                                    "{word:#010x} AVX2 (esize {esize}) with n {n:#034x}, \
                                     m {m:#034x}, d {d:#034x}"
                                );
                                fast_hits += 1;
                            }
                        }
                    }
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        assert!(fast_hits > 0, "the AVX2 integer path was never taken");
    }

    /// FADD/FSUB/FMUL/FMULX/FMLA/FMLS/FADDP and the floating-point
    /// compares at both widths, with NaNs, signed zeros, infinities and
    /// denormals; NaN results must fall back and still match `exec`.
    #[test]
    fn floating_three_same_matches_exec_on_edge_values() {
        let mut rng = Rng(0x0FED_CBA9_8765_4321);
        #[cfg(target_arch = "x86_64")]
        let mut fast_hits = 0u32;
        for size in 0..4u32 {
            for q in [false, true] {
                if size & 1 == 1 && !q {
                    continue;
                }
                let double = size & 1 == 1;
                let esize = if double { 64 } else { 32 };
                for opcode in [
                    0b11000u32, 0b11001, 0b11010, 0b11011, 0b11100, 0b11101, 0b11110,
                ] {
                    for u in [false, true] {
                        let word = three_same_word(u, size, opcode, q);
                        let kind = (u32::from(u) << 6) | ((size >> 1) << 5) | opcode;
                        for iteration in 0..48 {
                            // Alternate edge lanes with finite ones, so the
                            // non-NaN fast path is exercised as well.
                            let finite = iteration % 4 == 3;
                            let make = |rng: &mut Rng| {
                                if finite {
                                    finite_register(rng, esize)
                                } else {
                                    fp_register(rng, esize)
                                }
                            };
                            let n = make(&mut rng);
                            let m = make(&mut rng);
                            let d = make(&mut rng);
                            let Some(expected) = exec(word, &[(1, n), (2, m), (3, d)], 0, 0) else {
                                continue;
                            };
                            let got = vector3(kind, double, q, d, n, m);
                            assert_eq!(
                                got, expected.value,
                                "{word:#010x} (kind {kind:#04x}, double {double}, q {q}) \
                                 with n {n:#034x}, m {m:#034x}, d {d:#034x}"
                            );
                            // The portable fallback must agree whether or
                            // not this host has AVX2/FMA.
                            if !double {
                                if let Some(fallback) = vector3_f32(kind, q, d, n, m) {
                                    assert_eq!(
                                        fallback, expected.value,
                                        "{word:#010x} scalar (kind {kind:#04x}, q {q}) with \
                                         n {n:#034x}, m {m:#034x}, d {d:#034x}"
                                    );
                                }
                            }
                            #[cfg(target_arch = "x86_64")]
                            if x86::avx2_fma() {
                                let fast = if double {
                                    unsafe { x86::vector3_f64(kind, d, n, m) }
                                } else {
                                    unsafe { x86::vector3_f32(kind, q, d, n, m) }
                                };
                                if finite {
                                    // Finite operands cannot produce a NaN
                                    // for these operations, so the AVX2 path
                                    // must have taken them.
                                    assert_eq!(
                                        fast,
                                        Some(expected.value),
                                        "{word:#010x} AVX2 (kind {kind:#04x}, q {q}) \
                                         with finite n {n:#034x}, m {m:#034x}, d {d:#034x}"
                                    );
                                    fast_hits += 1;
                                } else if let Some(fast) = fast {
                                    fast_hits += 1;
                                    assert_eq!(
                                        fast, expected.value,
                                        "{word:#010x} AVX2 (kind {kind:#04x}, q {q}) with \
                                         n {n:#034x}, m {m:#034x}, d {d:#034x}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        assert!(
            fast_hits > 0,
            "the AVX2 floating-point path was never taken"
        );
    }

    /// FMAX, FMIN, FMAXNM, FMINNM, FMAXP, FMINP, FMAXNMP and FMINNMP
    /// against `exec` with NaNs, infinities and both signs of zero in every
    /// lane. These are the operations whose NaN rules the host maximum
    /// cannot reproduce by itself.
    #[test]
    fn floating_max_min_special_cases_match_exec() {
        let values32: [u32; 12] = [
            0,
            0x8000_0000,
            0x3F80_0000,
            0xBF80_0000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0001,
            0x7F80_0001,
            0x0000_0001,
            0x7F7F_FFFF,
            0x4F00_0000,
            0xCF00_0001,
        ];
        let values64: [u64; 12] = [
            0,
            0x8000_0000_0000_0000,
            0x3FF0_0000_0000_0000,
            0xBFF0_0000_0000_0000,
            0x7FF0_0000_0000_0000,
            0xFFF0_0000_0000_0000,
            0x7FF8_0000_0000_0001,
            0x7FF0_0000_0000_0001,
            1,
            0x7FEF_FFFF_FFFF_FFFF,
            0x43E0_0000_0000_0000,
            0xC3E0_0000_0000_0001,
        ];
        let mut rng = Rng(0x51AA_5EED_0BAD_F00D);
        #[cfg(target_arch = "x86_64")]
        let mut fast_hits = 0u32;
        for size in 0..4u32 {
            let double = size & 1 == 1;
            let esize = if double { 64 } else { 32 };
            for q in [false, true] {
                if size & 1 == 1 && !q {
                    continue;
                }
                for u in [false, true] {
                    for opcode in [0b11000u32, 0b11110] {
                        let word = three_same_word(u, size, opcode, q);
                        let kind = (u32::from(u) << 6) | ((size >> 1) << 5) | opcode;
                        for iteration in 0..128 {
                            let finite = iteration % 4 == 3;
                            let make = |rng: &mut Rng| {
                                let mut value = 0u128;
                                for i in 0..128 / esize {
                                    let lane = if double {
                                        if finite {
                                            values64[(rng.next() % 6) as usize]
                                        } else {
                                            values64[(rng.next() as usize) % values64.len()]
                                        }
                                    } else if finite {
                                        u64::from(values32[(rng.next() % 6) as usize])
                                    } else {
                                        u64::from(values32[(rng.next() as usize) % values32.len()])
                                    };
                                    value = put(value, esize, i, lane);
                                }
                                value
                            };
                            let n = make(&mut rng);
                            let m = make(&mut rng);
                            let d = make(&mut rng);
                            let Some(expected) = exec(word, &[(1, n), (2, m), (3, d)], 0, 0) else {
                                continue;
                            };
                            let got = vector3(kind, double, q, d, n, m);
                            assert_eq!(
                                got, expected.value,
                                "{word:#010x} (kind {kind:#04x}, double {double}, q {q}) \
                                 with n {n:#034x}, m {m:#034x}, d {d:#034x}"
                            );
                            #[cfg(target_arch = "x86_64")]
                            if x86::avx2_fma() {
                                let fast = if double {
                                    unsafe { x86::vector3_f64(kind, d, n, m) }
                                } else {
                                    unsafe { x86::vector3_f32(kind, q, d, n, m) }
                                };
                                if finite {
                                    assert_eq!(
                                        fast,
                                        Some(expected.value),
                                        "{word:#010x} AVX2 (kind {kind:#04x}, q {q}) with \
                                         finite n {n:#034x}, m {m:#034x}, d {d:#034x}"
                                    );
                                    fast_hits += 1;
                                } else if let Some(fast) = fast {
                                    fast_hits += 1;
                                    assert_eq!(
                                        fast, expected.value,
                                        "{word:#010x} AVX2 (kind {kind:#04x}, q {q}) with \
                                         n {n:#034x}, m {m:#034x}, d {d:#034x}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        assert!(fast_hits > 0, "the AVX2 max/min path was never taken");
    }

    /// The portable NaN detector must agree with `is_nan` for every lane
    /// position and every interesting encoding: quiet and signalling NaNs,
    /// infinities (not NaN), the largest finite value, zeros and denormals.
    #[test]
    fn the_word_wide_nan_check_agrees_with_is_nan() {
        let encodings = [
            0u32,
            0x8000_0000,
            0x0000_0001,
            0x007F_FFFF,
            0x0080_0000,
            0x3F80_0000,
            0x7F7F_FFFF,
            0x7F80_0000,
            0xFF80_0000,
            0x7F80_0001,
            0xFF80_0001,
            0x7FC0_0000,
            0xFFFF_FFFF,
            0x7FFF_FFFF,
        ];
        for (lane, bits) in encodings.iter().enumerate() {
            let mut value = 0u128;
            for i in 0..4 {
                value |= u128::from(if i == lane { *bits } else { 0x3F80_0000 }) << (32 * i);
            }
            for lanes in 1..=4 {
                let expected =
                    (0..lanes).any(|i| f32::from_bits((value >> (32 * i)) as u32).is_nan());
                assert_eq!(
                    any_nan_f32_bits(value & lane_bits(lanes)),
                    expected,
                    "lane {lane} bits {bits:#010x} among {lanes} lanes"
                );
                assert_eq!(pack_f32(lanes, f32_lanes(value)).is_none(), expected);
            }
        }
    }

    fn lane_bits(lanes: usize) -> u128 {
        if lanes == 4 {
            u128::MAX
        } else {
            (1u128 << (32 * lanes)) - 1
        }
    }

    /// The portable single-precision fast paths must take finite 4-lane
    /// inputs (so the per-lane evaluator is not reached) and fall back when
    /// a lane is NaN.
    #[test]
    fn the_portable_fast_paths_take_finite_four_lane_inputs() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF0);
        for kind in [
            0b011010u32, // FADD
            0b011011,    // FMUL
            0b011001,    // FMLA
            0b011110,    // FMAX
        ] {
            let (u, high, opcode) = (kind >> 6 == 1, (kind >> 5) & 1, kind & 0x1F);
            assert!(!u && high == 0);
            let word = three_same_word(u, 0, opcode, true);
            for iteration in 0..64 {
                let finite = iteration % 2 == 1;
                let make = |rng: &mut Rng| {
                    if finite {
                        finite_register(rng, 32)
                    } else {
                        fp_register(rng, 32)
                    }
                };
                let n = make(&mut rng);
                let m = make(&mut rng);
                let d = make(&mut rng);
                let expected = exec(word, &[(1, n), (2, m), (3, d)], 0, 0)
                    .expect("the operation is implemented")
                    .value;
                let fast = vector3_f32(kind, true, d, n, m);
                if finite {
                    assert_eq!(
                        fast,
                        Some(expected),
                        "portable fast path skipped {word:#010x} with finite \\
                         n {n:#034x}, m {m:#034x}, d {d:#034x}"
                    );
                } else if let Some(fast) = fast {
                    assert_eq!(fast, expected, "{word:#010x}");
                }
            }
        }
        for op in 0..4u32 {
            for index in 0..4u32 {
                let word = element_word(op, false, true, false, index);
                for iteration in 0..32 {
                    let finite = iteration % 2 == 1;
                    let make = |rng: &mut Rng| {
                        if finite {
                            finite_register(rng, 32)
                        } else {
                            fp_register(rng, 32)
                        }
                    };
                    let n = make(&mut rng);
                    let m = make(&mut rng);
                    let d = make(&mut rng);
                    let expected = exec(word, &[(3, d), (1, n), (2, m)], 0, 0)
                        .expect("the operation is implemented")
                        .value;
                    let fast = vector_element_f32(op, true, false, index, d, n, m);
                    if finite {
                        assert_eq!(
                            fast,
                            Some(expected),
                            "portable fast path skipped {word:#010x} (op {op}, \
                             index {index}) with finite lanes"
                        );
                    } else if let Some(fast) = fast {
                        assert_eq!(fast, expected, "{word:#010x}");
                    }
                }
            }
        }
    }

    /// FMLA/FMLS/FMUL/FMULX by element for every index, vector width and
    /// scalar form, against `exec` with edge lane values.
    #[test]
    fn floating_element_matches_exec_on_edge_values() {
        let mut rng = Rng(0xC001_D00D_1234_5678);
        #[cfg(target_arch = "x86_64")]
        let mut fast_hits = 0u32;
        for size in 0..2u32 {
            let double = size & 1 == 1;
            let esize = if double { 64 } else { 32 };
            for scalar in [false, true] {
                for q in [false, true] {
                    if scalar && !q {
                        continue;
                    }
                    if double && !scalar && !q {
                        continue;
                    }
                    let indexes = if double { 2 } else { 4 };
                    for op in 0..4u32 {
                        for index in 0..indexes {
                            let word = element_word(op, double, q, scalar, index);
                            assert!(
                                specialize(word).is_some(),
                                "{word:#010x} is not specialized"
                            );
                            for iteration in 0..24 {
                                let finite = iteration % 3 == 2;
                                let make = |rng: &mut Rng| {
                                    if finite {
                                        finite_register(rng, esize)
                                    } else {
                                        fp_register(rng, esize)
                                    }
                                };
                                let n = make(&mut rng);
                                let m = make(&mut rng);
                                let d = make(&mut rng);
                                let Some(expected) = exec(word, &[(3, d), (1, n), (2, m)], 0, 0)
                                else {
                                    continue;
                                };
                                let got = vector_element(op, double, q, scalar, index, d, n, m);
                                assert_eq!(
                                    got, expected.value,
                                    "{word:#010x} (op {op}, double {double}, q {q}, \
                                     scalar {scalar}, index {index}) with n {n:#034x}, \
                                     m {m:#034x}, d {d:#034x}"
                                );
                                #[cfg(target_arch = "x86_64")]
                                if x86::avx2_fma() {
                                    let fast = if double {
                                        unsafe {
                                            x86::vector_element_f64(op, scalar, index, d, n, m)
                                        }
                                    } else {
                                        unsafe {
                                            x86::vector_element_f32(op, q, scalar, index, d, n, m)
                                        }
                                    };
                                    if finite {
                                        assert_eq!(
                                            fast,
                                            Some(expected.value),
                                            "{word:#010x} AVX2 (op {op}, q {q}, scalar \
                                             {scalar}, index {index}) with finite n {n:#034x}, \
                                             m {m:#034x}, d {d:#034x}"
                                        );
                                        fast_hits += 1;
                                    } else if let Some(fast) = fast {
                                        fast_hits += 1;
                                        assert_eq!(
                                            fast, expected.value,
                                            "{word:#010x} AVX2 (op {op}, q {q}, scalar \
                                             {scalar}, index {index}) with n {n:#034x}, \
                                             m {m:#034x}, d {d:#034x}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        assert!(fast_hits > 0, "the AVX2 element path was never taken");
    }
}
