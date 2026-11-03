//! Floating-point and Advanced SIMD data processing for the A64 lifter and
//! the code it generates.
//!
//! [`operands`] tells the lifter which registers and flags an instruction
//! reads and writes; [`exec`] computes its result from those values at run
//! time. Both run the same evaluator ([`operands`] on a register file that
//! records what is touched), so they cannot disagree.
//!
//! Floating point follows the Arm pseudocode with FPCR.{DN,FZ,AHP} = 0 and
//! round-to-nearest-even as the current rounding mode. Exception flags and
//! FPSR.QC are not modelled. Loads and stores, and the copy and
//! modified-immediate forms the lifter handles natively, are not covered.

use std::cmp::Ordering;

/// The registers and flags an instruction touches.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Operands {
    /// Vector registers read, in first-use order without repeats.
    pub vector_reads: Vec<u8>,
    /// General-purpose register read; 31 is the zero register.
    pub general_read: Option<u8>,
    /// Vector register written.
    pub vector_write: Option<u8>,
    /// General-purpose register written; 31 discards the result.
    pub general_write: Option<u8>,
    pub reads_flags: bool,
    pub writes_flags: bool,
}

/// What [`exec`] produces: the destination's new value (a general
/// register's in the low 64 bits) and the NZCV flags (N in bit 3).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Outcome {
    pub value: u128,
    pub nzcv: u8,
}

/// The registers and flags `word` touches, or `None` if the evaluator does
/// not implement it.
pub fn operands(word: u32) -> Option<Operands> {
    let mut recorder = Recorder::default();
    eval(word, &mut recorder)?;
    let operands = recorder.0;
    let destinations = usize::from(operands.vector_write.is_some())
        + usize::from(operands.general_write.is_some())
        + usize::from(operands.writes_flags);
    (destinations == 1).then_some(operands)
}

/// Evaluate `word`. `vector` holds `(register, value)` for each of
/// [`Operands::vector_reads`], `general` the general register read (zero
/// for the zero register) and `nzcv` the flags.
pub fn exec(word: u32, vector: &[(u8, u128)], general: u64, nzcv: u8) -> Option<Outcome> {
    let mut values = Values {
        vector,
        general,
        nzcv,
        out: Outcome { value: 0, nzcv },
        complete: true,
    };
    eval(word, &mut values)?;
    values.complete.then_some(values.out)
}

trait Regs {
    fn v(&mut self, reg: u32) -> u128;
    fn x(&mut self, reg: u32) -> u64;
    fn flags(&mut self) -> u8;
    fn set_v(&mut self, reg: u32, value: u128);
    fn set_x(&mut self, reg: u32, value: u64);
    fn set_flags(&mut self, nzcv: u8);
}

#[derive(Default)]
struct Recorder(Operands);

impl Regs for Recorder {
    fn v(&mut self, reg: u32) -> u128 {
        let reg = reg as u8;
        if !self.0.vector_reads.contains(&reg) {
            self.0.vector_reads.push(reg);
        }
        0
    }
    fn x(&mut self, reg: u32) -> u64 {
        self.0.general_read = Some(reg as u8);
        0
    }
    fn flags(&mut self) -> u8 {
        self.0.reads_flags = true;
        0
    }
    fn set_v(&mut self, reg: u32, _: u128) {
        self.0.vector_write = Some(reg as u8);
    }
    fn set_x(&mut self, reg: u32, _: u64) {
        self.0.general_write = Some(reg as u8);
    }
    fn set_flags(&mut self, _: u8) {
        self.0.writes_flags = true;
    }
}

struct Values<'a> {
    vector: &'a [(u8, u128)],
    general: u64,
    nzcv: u8,
    out: Outcome,
    /// False once a register was read that the caller did not pass.
    complete: bool,
}

impl Regs for Values<'_> {
    fn v(&mut self, reg: u32) -> u128 {
        match self.vector.iter().find(|(r, _)| u32::from(*r) == reg) {
            Some((_, value)) => *value,
            None => {
                self.complete = false;
                0
            }
        }
    }
    fn x(&mut self, _: u32) -> u64 {
        self.general
    }
    fn flags(&mut self) -> u8 {
        self.nzcv
    }
    fn set_v(&mut self, _: u32, value: u128) {
        self.out.value = value;
    }
    fn set_x(&mut self, _: u32, value: u64) {
        self.out.value = u128::from(value);
    }
    fn set_flags(&mut self, nzcv: u8) {
        self.out.nzcv = nzcv;
    }
}

fn eval(word: u32, s: &mut impl Regs) -> Option<()> {
    // Bit 30 clear, bits 28:25 = 1111: scalar floating point.
    if word & 0x5E00_0000 == 0x1E00_0000 {
        return scalar_fp(word, s);
    }
    if word & 0xFFFF_CC00 == 0x4E28_4800 {
        return aes(word, s);
    }
    if word & 0xFFE0_8C00 == 0x5E00_0000 || word & 0xFFFF_CC00 == 0x5E28_0800 {
        return sha(word, s);
    }
    match word & 0xDF00_0000 {
        0x0E00_0000 | 0x4E00_0000 => simd_three_reg(word, s, false),
        0x5E00_0000 => simd_three_reg(word, s, true),
        0x0F00_0000 | 0x4F00_0000 => simd_element_or_shift(word, s, false),
        0x5F00_0000 => simd_element_or_shift(word, s, true),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Bit and lane helpers

fn bits(word: u32, hi: u32, lo: u32) -> u32 {
    (word >> lo) & ((1 << (hi - lo + 1)) - 1)
}

fn bit(word: u32, at: u32) -> bool {
    (word >> at) & 1 == 1
}

fn regs(word: u32) -> (u32, u32, u32) {
    (bits(word, 4, 0), bits(word, 9, 5), bits(word, 20, 16))
}

fn mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1 << width) - 1
    }
}

fn sext(value: u64, width: u32) -> i64 {
    let shift = 64 - width;
    ((value << shift) as i64) >> shift
}

/// Element `index` of `esize` bits.
fn lane(v: u128, esize: u32, index: u32) -> u64 {
    (v >> (index * esize)) as u64 & mask(esize)
}

fn put(v: u128, esize: u32, index: u32, value: u64) -> u128 {
    let shift = index * esize;
    let field = u128::from(mask(esize)) << shift;
    (v & !field) | ((u128::from(value) << shift) & field)
}

/// An element as a signed or unsigned integer.
fn int(value: u64, esize: u32, unsigned: bool) -> i128 {
    if unsigned {
        i128::from(value & mask(esize))
    } else {
        i128::from(sext(value, esize))
    }
}

fn wrap(value: i128, esize: u32) -> u64 {
    value as u64 & mask(esize)
}

/// Saturate to the signed or unsigned `esize`-bit range.
fn sat(value: i128, esize: u32, unsigned: bool) -> u64 {
    let (lo, hi) = if unsigned {
        (0, i128::from(mask(esize)))
    } else {
        (-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1)
    };
    wrap(value.clamp(lo, hi), esize)
}

fn ones(condition: bool, esize: u32) -> u64 {
    if condition {
        mask(esize)
    } else {
        0
    }
}

fn width(q: bool) -> u32 {
    if q {
        128
    } else {
        64
    }
}

fn low(v: u128) -> u128 {
    v & u128::from(u64::MAX)
}

/// The 64-bit half a "2" (upper) instruction form reads.
fn half(v: u128, upper: bool) -> u128 {
    if upper {
        v >> 64
    } else {
        low(v)
    }
}

/// Element `index` of the pairwise concatenation `m:n` (`n` low).
fn pair(n: u128, m: u128, esize: u32, lanes: u32, index: u32) -> (u64, u64) {
    let element = |j: u32| {
        if j < lanes {
            lane(n, esize, j)
        } else {
            lane(m, esize, j - lanes)
        }
    };
    (element(2 * index), element(2 * index + 1))
}

/// Write 64 bits of narrowed results to the lower half (zeroing the upper),
/// or to the upper half keeping the lower ("2" forms).
fn write_narrow(s: &mut impl Regs, rd: u32, upper: bool, narrowed: u128) {
    if upper {
        let d = s.v(rd);
        s.set_v(rd, low(d) | (narrowed << 64));
    } else {
        s.set_v(rd, low(narrowed));
    }
}

fn condition_holds(cond: u32, nzcv: u8) -> bool {
    let (n, z, c, v) = (nzcv & 8 != 0, nzcv & 4 != 0, nzcv & 2 != 0, nzcv & 1 != 0);
    let result = match cond >> 1 {
        0 => z,
        1 => c,
        2 => n,
        3 => v,
        4 => c && !z,
        5 => n == v,
        6 => n == v && !z,
        _ => true,
    };
    if cond & 1 == 1 && cond != 0b1111 {
        !result
    } else {
        result
    }
}

// ---------------------------------------------------------------------------
// Floating-point arithmetic on raw bit patterns

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rounding {
    TiesEven,
    TiesAway,
    Up,
    Down,
    Zero,
}

trait Float:
    Copy
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    const BITS: u32;
    const FRACTION: u32;
    fn from_raw(raw: u64) -> Self;
    fn raw(self) -> u64;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_i64(value: i64) -> Self;
    fn from_u64(value: u64) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn sqrt(self) -> Self;
    fn is_nan(self) -> bool;
    fn round_with(self, rounding: Rounding) -> Self;
}

macro_rules! float {
    ($ty:ty, $fraction:expr, $raw:ty) => {
        impl Float for $ty {
            const BITS: u32 = <$raw>::BITS;
            const FRACTION: u32 = $fraction;
            fn from_raw(raw: u64) -> Self {
                <$ty>::from_bits(raw as $raw)
            }
            fn raw(self) -> u64 {
                u64::from(self.to_bits())
            }
            fn from_f64(value: f64) -> Self {
                value as $ty
            }
            fn to_f64(self) -> f64 {
                f64::from(self)
            }
            fn from_i64(value: i64) -> Self {
                value as $ty
            }
            fn from_u64(value: u64) -> Self {
                value as $ty
            }
            fn mul_add(self, a: Self, b: Self) -> Self {
                <$ty>::mul_add(self, a, b)
            }
            fn sqrt(self) -> Self {
                <$ty>::sqrt(self)
            }
            fn is_nan(self) -> bool {
                <$ty>::is_nan(self)
            }
            fn round_with(self, rounding: Rounding) -> Self {
                match rounding {
                    Rounding::TiesEven => self.round_ties_even(),
                    Rounding::TiesAway => self.round(),
                    Rounding::Up => self.ceil(),
                    Rounding::Down => self.floor(),
                    Rounding::Zero => self.trunc(),
                }
            }
        }
    };
}

float!(f32, 23, u32);
float!(f64, 52, u64);

/// 2^n, exactly, for the small non-negative `n` used by fixed-point
/// conversions.
#[inline]
fn pow2(n: u32) -> f64 {
    debug_assert!(n < 0x3FF);
    f64::from_bits((1023 + u64::from(n)) << 52)
}

/// 2^-n, exactly, for `n` up to 64 (the largest fixed-point shift).
#[inline]
fn pow2_negative(n: u32) -> f64 {
    debug_assert!(n <= 64);
    f64::from_bits((1023 - u64::from(n)) << 52)
}

// The host fast paths below compute the same expression the exact bodies do
// and only skip their NaN selection and default-NaN formatting. For a result
// that is not NaN, the host's IEEE operation is the exact body's operation, so
// the two agree bit for bit even when the host's MXCSR flushes denormals
// (guest FPCR.FZ is mirrored there): both sides run the same instruction.

/// True when a binary64 value is exactly halfway between two binary32 values.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn binary32_midpoint(value: f64) -> bool {
    value.to_bits() & 0x1FFF_FFFF == 0x1000_0000
}

/// A correctly rounded single-precision `a * b + c`, or `None` when this host
/// cannot compute one with a single rounding. A NaN result is left to the
/// exact evaluator's NaN rules.
#[inline]
fn fused_mul_add_f32(a: f32, b: f32, c: f32) -> Option<f32> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU has the FMA feature.
        return unsafe { fused_mul_add_f32_host(a, b, c) };
    }
    // aarch64 has a fused multiply-add in its baseline floating-point unit
    // (`mul_add` compiles to FMADD), so no feature check is needed.
    #[cfg(target_arch = "aarch64")]
    {
        let value = a.mul_add(b, c);
        (!value.is_nan()).then_some(value)
    }
    // Without a hardware FMA, the product of two binary32 values is exact in
    // binary64, so the sum gets one rounding there. Rounding that again to
    // binary32 can differ from a single rounding only when the binary64 value
    // is exactly a binary32 midpoint, which is rejected.
    #[cfg(not(target_arch = "aarch64"))]
    {
        let sum = f64::from(a) * f64::from(b) + f64::from(c);
        if sum.is_nan() || binary32_midpoint(sum) {
            return None;
        }
        let value = sum as f32;
        (!value.is_nan()).then_some(value)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn fused_mul_add_f32_host(a: f32, b: f32, c: f32) -> Option<f32> {
    let value = a.mul_add(b, c);
    (!value.is_nan()).then_some(value)
}

/// A correctly rounded double-precision `a * b + c`, or `None` when the host
/// has no hardware FMA. A NaN result is left to the exact evaluator.
#[inline]
fn fused_mul_add_f64(a: f64, b: f64, c: f64) -> Option<f64> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: the CPU has the FMA feature.
            return unsafe { fused_mul_add_f64_host(a, b, c) };
        }
        None
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let value = a.mul_add(b, c);
        (!value.is_nan()).then_some(value)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn fused_mul_add_f64_host(a: f64, b: f64, c: f64) -> Option<f64> {
    let value = a.mul_add(b, c);
    (!value.is_nan()).then_some(value)
}

/// Call a function generic over [`Float`] for a 32- or 64-bit element.
macro_rules! sized {
    ($size:expr, $func:ident($($arg:expr),*)) => {
        if $size == 32 {
            $func::<f32>($($arg),*)
        } else {
            $func::<f64>($($arg),*)
        }
    };
}

fn sign<F: Float>() -> u64 {
    1 << (F::BITS - 1)
}

fn quiet<F: Float>() -> u64 {
    1 << (F::FRACTION - 1)
}

fn exponent<F: Float>() -> u64 {
    mask(F::BITS - 1) & !mask(F::FRACTION)
}

fn infinity<F: Float>(negative: bool) -> u64 {
    exponent::<F>() | if negative { sign::<F>() } else { 0 }
}

fn default_nan<F: Float>() -> u64 {
    exponent::<F>() | quiet::<F>()
}

fn is_nan<F: Float>(raw: u64) -> bool {
    // An exponent of all ones is larger than the infinity encoding only when
    // the fraction is nonzero. Callers pass a lane already masked to F's
    // width, so the sign bit is the only bit above the exponent to clear.
    raw & !sign::<F>() > exponent::<F>()
}

fn is_snan<F: Float>(raw: u64) -> bool {
    is_nan::<F>(raw) && raw & quiet::<F>() == 0
}

fn is_qnan<F: Float>(raw: u64) -> bool {
    is_nan::<F>(raw) && raw & quiet::<F>() != 0
}

fn is_inf<F: Float>(raw: u64) -> bool {
    raw & mask(F::BITS - 1) == exponent::<F>()
}

fn is_zero<F: Float>(raw: u64) -> bool {
    raw & mask(F::BITS - 1) == 0
}

/// FPProcessNaNs: the first signalling NaN, else the first quiet NaN,
/// quietened.
fn process_nans<F: Float>(operands: &[u64]) -> Option<u64> {
    operands
        .iter()
        .find(|raw| is_snan::<F>(**raw))
        .or_else(|| operands.iter().find(|raw| is_nan::<F>(**raw)))
        .map(|raw| raw | quiet::<F>())
}

/// A computed result: an invalid operation yields the default NaN.
fn finish<F: Float>(value: F) -> u64 {
    if value.is_nan() {
        default_nan::<F>()
    } else {
        value.raw()
    }
}

fn infinity_times_zero<F: Float>(a: u64, b: u64) -> bool {
    (is_inf::<F>(a) && is_zero::<F>(b)) || (is_zero::<F>(a) && is_inf::<F>(b))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fp2 {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    MaxNum,
    MinNum,
    MulX,
    Abd,
    Nmul,
    RecipStep,
    RsqrtStep,
}

#[inline]
fn fp2<F: Float>(op: Fp2, a: u64, b: u64) -> u64 {
    if let Some(value) = fp2_fast::<F>(op, a, b) {
        return value;
    }
    fp2_exact::<F>(op, a, b)
}

/// FMUL, FDIV, FADD, FSUB, FMAX, FMIN, FMAXNM, FMINNM, FNMUL, FABD and the
/// reciprocal steps on the host. Every operation here is correctly rounded
/// IEEE arithmetic, so a non-NaN result is bit-identical to the exact body's;
/// the only cases the host cannot answer are NaN operands (whose payload and
/// signalling/quieting rules are ARM-specific), NaN results, FMULX's
/// architected infinity-times-zero result, and a fused step the host cannot
/// perform with one rounding. Those return `None` for the exact body.
#[inline]
fn fp2_fast<F: Float>(op: Fp2, a: u64, b: u64) -> Option<u64> {
    if is_nan::<F>(a) || is_nan::<F>(b) {
        return None;
    }
    if F::BITS == 32 {
        let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
        let value = match op {
            Fp2::Add => x + y,
            Fp2::Sub => x - y,
            Fp2::Mul => x * y,
            Fp2::Div => x / y,
            Fp2::Nmul => -(x * y),
            Fp2::Abd => (x - y).abs(),
            // Equal values differ at most in the sign of zero.
            Fp2::Max | Fp2::MaxNum => {
                return Some(if x > y {
                    a
                } else if x < y {
                    b
                } else {
                    a & b
                })
            }
            Fp2::Min | Fp2::MinNum => {
                return Some(if x < y {
                    a
                } else if x > y {
                    b
                } else {
                    a | b
                })
            }
            Fp2::RecipStep => fused_mul_add_f32(-x, y, 2.0)?,
            Fp2::RsqrtStep => fused_mul_add_f32(-x, y * 0.5, 1.5)?,
            // FMULX's 0 * infinity is 2.0 signed by the operands, not NaN.
            Fp2::MulX => return None,
        };
        if value.is_nan() {
            return None;
        }
        Some(u64::from(value.to_bits()))
    } else {
        let (x, y) = (f64::from_bits(a), f64::from_bits(b));
        let value = match op {
            Fp2::Add => x + y,
            Fp2::Sub => x - y,
            Fp2::Mul => x * y,
            Fp2::Div => x / y,
            Fp2::Nmul => -(x * y),
            Fp2::Abd => (x - y).abs(),
            Fp2::Max | Fp2::MaxNum => {
                return Some(if x > y {
                    a
                } else if x < y {
                    b
                } else {
                    a & b
                })
            }
            Fp2::Min | Fp2::MinNum => {
                return Some(if x < y {
                    a
                } else if x > y {
                    b
                } else {
                    a | b
                })
            }
            Fp2::RecipStep => fused_mul_add_f64(-x, y, 2.0)?,
            Fp2::RsqrtStep => fused_mul_add_f64(-x, y * 0.5, 1.5)?,
            Fp2::MulX => return None,
        };
        if value.is_nan() {
            return None;
        }
        Some(value.to_bits())
    }
}

/// The exact two-operand body: ARM's NaN selection, default NaN, signed-zero
/// max/min rules and FMULX's architected special cases.
#[cold]
#[inline(never)]
fn fp2_exact<F: Float>(op: Fp2, a: u64, b: u64) -> u64 {
    // FRECPS and FRSQRTS negate the first operand before anything else.
    let a = if matches!(op, Fp2::RecipStep | Fp2::RsqrtStep) {
        a ^ sign::<F>()
    } else {
        a
    };
    // FMAXNM/FMINNM: a quiet NaN loses to a number.
    let (a, b) = if matches!(op, Fp2::MaxNum | Fp2::MinNum) {
        let loser = infinity::<F>(op == Fp2::MaxNum);
        match (is_qnan::<F>(a), is_qnan::<F>(b)) {
            (true, false) => (loser, b),
            (false, true) => (a, loser),
            _ => (a, b),
        }
    } else {
        (a, b)
    };
    if let Some(nan) = process_nans::<F>(&[a, b]) {
        return match op {
            Fp2::Abd => nan & !sign::<F>(),
            Fp2::Nmul => nan ^ sign::<F>(),
            _ => nan,
        };
    }
    let special = infinity_times_zero::<F>(a, b);
    let (x, y) = (F::from_raw(a), F::from_raw(b));
    match op {
        Fp2::Add => finish(x + y),
        Fp2::Sub => finish(x - y),
        Fp2::Mul => finish(x * y),
        Fp2::Nmul => finish(x * y) ^ sign::<F>(),
        Fp2::Div => finish(x / y),
        // Equal values differ at most in the sign of zero.
        Fp2::Max | Fp2::MaxNum => match x.partial_cmp(&y) {
            Some(Ordering::Greater) => a,
            Some(Ordering::Less) => b,
            _ => a & b,
        },
        Fp2::Min | Fp2::MinNum => match x.partial_cmp(&y) {
            Some(Ordering::Less) => a,
            Some(Ordering::Greater) => b,
            _ => a | b,
        },
        Fp2::MulX if special => ((a ^ b) & sign::<F>()) | F::from_f64(2.0).raw(),
        Fp2::MulX => finish(x * y),
        Fp2::Abd => finish(x - y) & !sign::<F>(),
        Fp2::RecipStep if special => F::from_f64(2.0).raw(),
        Fp2::RecipStep => finish(x.mul_add(y, F::from_f64(2.0))),
        Fp2::RsqrtStep if special => F::from_f64(1.5).raw(),
        Fp2::RsqrtStep => finish(x.mul_add(y * F::from_f64(0.5), F::from_f64(1.5))),
    }
}

/// FPMulAdd: `addend + a * b` with a single rounding.
#[inline]
fn fp_mul_add<F: Float>(addend: u64, a: u64, b: u64) -> u64 {
    if let Some(value) = fp_mul_add_fast::<F>(addend, a, b) {
        return value;
    }
    fp_mul_add_exact::<F>(addend, a, b)
}

/// The host fused multiply-add: correct whenever it returns `Some`, because
/// `fused_mul_add_*` rounds once (or refuses). NaN operands and results, and
/// the infinity-times-zero special case, go to the exact body.
#[inline]
fn fp_mul_add_fast<F: Float>(addend: u64, a: u64, b: u64) -> Option<u64> {
    if is_nan::<F>(addend) || is_nan::<F>(a) || is_nan::<F>(b) {
        return None;
    }
    if F::BITS == 32 {
        let value = fused_mul_add_f32(
            f32::from_bits(a as u32),
            f32::from_bits(b as u32),
            f32::from_bits(addend as u32),
        )?;
        Some(u64::from(value.to_bits()))
    } else {
        let value =
            fused_mul_add_f64(f64::from_bits(a), f64::from_bits(b), f64::from_bits(addend))?;
        Some(value.to_bits())
    }
}

#[cold]
#[inline(never)]
fn fp_mul_add_exact<F: Float>(addend: u64, a: u64, b: u64) -> u64 {
    let special = infinity_times_zero::<F>(a, b);
    if is_qnan::<F>(addend) && special {
        return default_nan::<F>();
    }
    if let Some(nan) = process_nans::<F>(&[addend, a, b]) {
        return nan;
    }
    finish(F::from_raw(a).mul_add(F::from_raw(b), F::from_raw(addend)))
}

fn fp_compare<F: Float>(a: u64, b: u64) -> Option<Ordering> {
    F::from_raw(a).partial_cmp(&F::from_raw(b))
}

/// NZCV after FCMP: unordered 0011, equal 0110, less 1000, greater 0010.
fn compare_flags(order: Option<Ordering>) -> u8 {
    match order {
        None => 0b0011,
        Some(Ordering::Equal) => 0b0110,
        Some(Ordering::Less) => 0b1000,
        Some(Ordering::Greater) => 0b0010,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cmp {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
    AbsGe,
    AbsGt,
}

#[inline]
fn fp_cmp<F: Float>(cmp: Cmp, a: u64, b: u64) -> bool {
    let (a, b) = if matches!(cmp, Cmp::AbsGe | Cmp::AbsGt) {
        (a & !sign::<F>(), b & !sign::<F>())
    } else {
        (a, b)
    };
    // Rust's comparisons are false for NaN, which is what FCMxx does.
    let (x, y) = (F::from_raw(a), F::from_raw(b));
    match cmp {
        Cmp::Eq => x == y,
        Cmp::Ge | Cmp::AbsGe => x >= y,
        Cmp::Gt | Cmp::AbsGt => x > y,
        Cmp::Le => x <= y,
        Cmp::Lt => x < y,
    }
}

/// The evaluator's pre-fast-path compare, kept as the differential reference.
#[cfg(test)]
fn fp_cmp_reference<F: Float>(cmp: Cmp, a: u64, b: u64) -> bool {
    let (a, b) = if matches!(cmp, Cmp::AbsGe | Cmp::AbsGt) {
        (a & !sign::<F>(), b & !sign::<F>())
    } else {
        (a, b)
    };
    let Some(order) = fp_compare::<F>(a, b) else {
        return false;
    };
    match cmp {
        Cmp::Eq => order == Ordering::Equal,
        Cmp::Ge | Cmp::AbsGe => order != Ordering::Less,
        Cmp::Gt | Cmp::AbsGt => order == Ordering::Greater,
        Cmp::Le => order != Ordering::Greater,
        Cmp::Lt => order == Ordering::Less,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fp1 {
    Abs,
    Neg,
    Sqrt,
    Round(Rounding),
    RecipEstimate,
    RsqrtEstimate,
}

#[inline]
fn fp1<F: Float>(op: Fp1, a: u64) -> u64 {
    match op {
        // FABS and FNEG only touch the sign, NaN or not.
        Fp1::Abs => a & !sign::<F>(),
        Fp1::Neg => a ^ sign::<F>(),
        Fp1::Sqrt => {
            // IEEE square root, correctly rounded on the host. A NaN result
            // means a NaN or negative operand; the cold formatter applies
            // FPProcessNaNs and the invalid operation's default NaN.
            let value = F::from_raw(a).sqrt();
            if value.is_nan() {
                sqrt_nan::<F>(a)
            } else {
                value.raw()
            }
        }
        Fp1::Round(rounding) => {
            // Rounding propagates NaN; every ARM rounding mode returns the
            // input's quietened payload, which the operand still holds.
            let value = F::from_raw(a).round_with(rounding);
            if value.is_nan() {
                a | quiet::<F>()
            } else {
                value.raw()
            }
        }
        Fp1::RecipEstimate | Fp1::RsqrtEstimate => {
            // FPProcessNaNs for a single source, then the architected table.
            if is_nan::<F>(a) {
                a | quiet::<F>()
            } else {
                fp1_estimate::<F>(op, a)
            }
        }
    }
}

/// The two NaN outcomes of FSQRT: quieten a NaN operand, or return the invalid
/// operation's default NaN for a negative one.
#[cold]
#[inline(never)]
fn sqrt_nan<F: Float>(a: u64) -> u64 {
    if is_nan::<F>(a) {
        a | quiet::<F>()
    } else {
        default_nan::<F>()
    }
}

/// The architected estimate table lookups stay out of line so the square-root
/// and rounding bodies above inline into their callers. They are still sized
/// from the same tables; no host reciprocal or rsqrt instruction is used.
#[inline(never)]
fn fp1_estimate<F: Float>(op: Fp1, a: u64) -> u64 {
    match op {
        Fp1::RecipEstimate => recip_estimate::<F>(a),
        Fp1::RsqrtEstimate => rsqrt_estimate::<F>(a),
        _ => unreachable!(),
    }
}

/// The evaluator's pre-fast-path one-source body, kept as the differential
/// reference for [`fp1`].
#[cfg(test)]
fn fp1_reference<F: Float>(op: Fp1, a: u64) -> u64 {
    // FABS and FNEG only touch the sign, NaN or not.
    match op {
        Fp1::Abs => return a & !sign::<F>(),
        Fp1::Neg => return a ^ sign::<F>(),
        _ => {}
    }
    if let Some(nan) = process_nans::<F>(&[a]) {
        return nan;
    }
    let x = F::from_raw(a);
    match op {
        Fp1::Sqrt => finish(x.sqrt()),
        Fp1::Round(rounding) => x.round_with(rounding).raw(),
        Fp1::RecipEstimate => recip_estimate::<F>(a),
        Fp1::RsqrtEstimate => rsqrt_estimate::<F>(a),
        Fp1::Abs | Fp1::Neg => unreachable!(),
    }
}

/// The fraction left-aligned to 52 bits and the biased exponent.
fn unpack<F: Float>(raw: u64) -> (u64, i64) {
    let fraction = (raw & mask(F::FRACTION)) << (52 - F::FRACTION);
    let exp = ((raw >> F::FRACTION) & mask(F::BITS - 1 - F::FRACTION)) as i64;
    (fraction, exp)
}

fn pack<F: Float>(sign_bit: u64, exp: i64, fraction: u64) -> u64 {
    let exp_bits = F::BITS - 1 - F::FRACTION;
    sign_bit | ((exp as u64 & mask(exp_bits)) << F::FRACTION) | (fraction >> (52 - F::FRACTION))
}

/// FRECPE's 8-bit result for each `scaled` fraction byte (0x100..0x1FF),
/// computed once at compile time instead of dividing on every estimate.
const fn recip_estimate_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut index = 0;
    while index < 256 {
        let a = (0x100 + index as u64) * 2 + 1;
        let b = (1u64 << 19) / a;
        table[index] = b.div_ceil(2) as u8;
        index += 1;
    }
    table
}
const RECIP_ESTIMATE: [u8; 256] = recip_estimate_table();

/// FRSQRTE's 8-bit result for each `scaled` fraction (0x80..0x1FF). The
/// architected form finds the largest `b` whose reciprocal square still
/// fits 28 bits; doing that walk once at compile time removes it from the
/// hot path.
const fn rsqrt_estimate_table() -> [u8; 512] {
    let mut table = [0u8; 512];
    let mut index = 128usize;
    while index < 512 {
        let scaled = index as u64;
        let a = if scaled < 256 {
            scaled * 2 + 1
        } else {
            (((scaled >> 1) << 1) + 1) * 2
        };
        let mut b = 512u64;
        while a * (b + 1) * (b + 1) < (1u64 << 28) {
            b += 1;
        }
        table[index] = b.div_ceil(2) as u8;
        index += 1;
    }
    table
}
const RSQRT_ESTIMATE: [u8; 512] = rsqrt_estimate_table();

/// FRECPE (FPRecipEstimate, 8-bit table-free form).
fn recip_estimate<F: Float>(raw: u64) -> u64 {
    let sign_bit = raw & sign::<F>();
    if is_inf::<F>(raw) {
        return sign_bit;
    }
    if is_zero::<F>(raw) {
        return infinity::<F>(sign_bit != 0);
    }
    let magnitude = F::from_raw(raw & !sign_bit).to_f64();
    let tiny = if F::BITS == 32 {
        f64::from_bits((1023 - 128) << 52)
    } else {
        f64::from_bits(1 << 50)
    };
    if magnitude < tiny {
        return infinity::<F>(sign_bit != 0);
    }
    let (mut fraction, mut exp) = unpack::<F>(raw);
    if exp == 0 {
        if (fraction >> 51) & 1 == 0 {
            exp = -1;
            fraction = (fraction & mask(50)) << 2;
        } else {
            fraction = (fraction & mask(51)) << 1;
        }
    }
    let scaled = 0x100 | ((fraction >> 44) & 0xFF);
    let mut result_exp = if F::BITS == 32 { 253 - exp } else { 2045 - exp };
    let estimate = u64::from(RECIP_ESTIMATE[(scaled & 0xFF) as usize]);
    let mut fraction = (estimate & 0xFF) << 44;
    if result_exp == 0 {
        fraction = (1 << 51) | (fraction >> 1);
    } else if result_exp == -1 {
        fraction = (1 << 50) | (fraction >> 2);
        result_exp = 0;
    }
    pack::<F>(sign_bit, result_exp, fraction)
}

/// FRSQRTE (FPRSqrtEstimate).
fn rsqrt_estimate<F: Float>(raw: u64) -> u64 {
    let sign_bit = raw & sign::<F>();
    if is_zero::<F>(raw) {
        return infinity::<F>(sign_bit != 0);
    }
    if sign_bit != 0 {
        return default_nan::<F>();
    }
    if is_inf::<F>(raw) {
        return 0;
    }
    let (mut fraction, mut exp) = unpack::<F>(raw);
    if exp == 0 {
        while (fraction >> 51) & 1 == 0 {
            fraction = (fraction & mask(51)) << 1;
            exp -= 1;
        }
        fraction = (fraction & mask(51)) << 1;
    }
    let scaled = if exp & 1 == 0 {
        0x100 | ((fraction >> 44) & 0xFF)
    } else {
        0x80 | ((fraction >> 45) & 0x7F)
    };
    let result_exp = if F::BITS == 32 {
        (380 - exp) / 2
    } else {
        (3068 - exp) / 2
    };
    let estimate = u64::from(RSQRT_ESTIMATE[scaled as usize]);
    pack::<F>(0, result_exp, (estimate & 0xFF) << 44)
}

/// FCVT* to an integer of `int_bits`, with `fbits` fraction bits.
#[inline]
fn fp_to_int<F: Float>(
    raw: u64,
    rounding: Rounding,
    fbits: u32,
    unsigned: bool,
    int_bits: u32,
) -> u64 {
    let cast = |value: f64| match (unsigned, int_bits) {
        (false, 32) => u64::from(value as i32 as u32),
        (false, _) => value as i64 as u64,
        (true, 32) => u64::from(value as u32),
        (true, _) => value as u64,
    };
    // NaN converts to zero, and infinity or an out-of-range value saturates,
    // through the same saturating casts as the scaled form.
    let value = if fbits == 0 {
        F::from_raw(raw).to_f64()
    } else {
        F::from_raw(raw).to_f64() * pow2(fbits)
    };
    cast(value.round_with(rounding))
}

/// The evaluator's pre-fast-path conversion, kept as the differential
/// reference for [`fp_to_int`].
#[cfg(test)]
fn fp_to_int_reference<F: Float>(
    raw: u64,
    rounding: Rounding,
    fbits: u32,
    unsigned: bool,
    int_bits: u32,
) -> u64 {
    if is_nan::<F>(raw) {
        return 0;
    }
    let value = (F::from_raw(raw).to_f64() * 2f64.powi(fbits as i32)).round_with(rounding);
    // `as` saturates, as the conversions do.
    match (unsigned, int_bits) {
        (false, 32) => u64::from(value as i32 as u32),
        (false, _) => value as i64 as u64,
        (true, 32) => u64::from(value as u32),
        (true, _) => value as u64,
    }
}

/// SCVTF/UCVTF from an integer of `int_bits` with `fbits` fraction bits.
#[inline]
fn int_to_fp<F: Float>(value: u64, unsigned: bool, int_bits: u32, fbits: u32) -> u64 {
    let converted = if unsigned {
        F::from_u64(value & mask(int_bits))
    } else {
        F::from_i64(sext(value, int_bits))
    };
    if fbits == 0 {
        return converted.raw();
    }
    (converted * F::from_f64(pow2_negative(fbits))).raw()
}

/// The evaluator's pre-fast-path conversion, kept as the differential
/// reference for [`int_to_fp`].
#[cfg(test)]
fn int_to_fp_reference<F: Float>(value: u64, unsigned: bool, int_bits: u32, fbits: u32) -> u64 {
    let converted = if unsigned {
        F::from_u64(value & mask(int_bits))
    } else {
        F::from_i64(sext(value, int_bits))
    };
    if fbits == 0 {
        return converted.raw();
    }
    (converted * F::from_f64(0.5f64.powi(fbits as i32))).raw()
}

fn fraction_bits(size: u32) -> u32 {
    match size {
        16 => 10,
        32 => 23,
        _ => 52,
    }
}

/// FCVT between half (16), single (32) and double (64) precision.
#[inline]
fn fp_convert(raw: u64, from: u32, to: u32) -> u64 {
    match (from, to) {
        (32, 64) if !is_nan::<f32>(raw) => {
            return f64::from(f32::from_bits(raw as u32)).to_bits();
        }
        (64, 32) if !is_nan::<f64>(raw) => {
            return u64::from((f64::from_bits(raw) as f32).to_bits());
        }
        (16, 32) if !is_nan_half(raw as u16) => {
            return u64::from((half_to_f64(raw as u16) as f32).to_bits());
        }
        (16, 64) if !is_nan_half(raw as u16) => {
            return half_to_f64(raw as u16).to_bits();
        }
        (32, 16) if !is_nan::<f32>(raw) => {
            return u64::from(f64_to_half(f64::from(f32::from_bits(raw as u32))));
        }
        (64, 16) if !is_nan::<f64>(raw) => {
            return u64::from(f64_to_half(f64::from_bits(raw)));
        }
        _ => {}
    }
    fp_convert_exact(raw, from, to)
}

fn is_nan_half(raw: u16) -> bool {
    raw & 0x7C00 == 0x7C00 && raw & 0x03FF != 0
}

/// The exact conversion, including ARM's NaN payload mapping.
#[cold]
#[inline(never)]
fn fp_convert_exact(raw: u64, from: u32, to: u32) -> u64 {
    let (from_fraction, to_fraction) = (fraction_bits(from), fraction_bits(to));
    let from_exp = mask(from - 1) & !mask(from_fraction);
    let sign_bit = (raw >> (from - 1)) & 1;
    if raw & from_exp == from_exp && raw & mask(from_fraction) != 0 {
        // NaNs keep their sign and top payload bits and become quiet.
        let payload = raw & mask(from_fraction);
        let payload = if to_fraction >= from_fraction {
            payload << (to_fraction - from_fraction)
        } else {
            payload >> (from_fraction - to_fraction)
        };
        let to_exp = mask(to - 1) & !mask(to_fraction);
        return (sign_bit << (to - 1)) | to_exp | (1 << (to_fraction - 1)) | payload;
    }
    let value = match from {
        16 => half_to_f64(raw as u16),
        32 => f64::from(f32::from_bits(raw as u32)),
        _ => f64::from_bits(raw),
    };
    match to {
        16 => u64::from(f64_to_half(value)),
        32 => u64::from((value as f32).to_bits()),
        _ => value.to_bits(),
    }
}

/// A non-NaN IEEE half as a double (exact).
fn half_to_f64(half: u16) -> f64 {
    let sign = if half & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((half >> 10) & 0x1F);
    let fraction = f64::from(half & 0x3FF);
    sign * match exp {
        0 => fraction * 2f64.powi(-24),
        31 => f64::INFINITY,
        _ => (1024.0 + fraction) * 2f64.powi(exp - 25),
    }
}

/// A non-NaN double rounded (to nearest even) to an IEEE half.
fn f64_to_half(value: f64) -> u16 {
    let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
    let magnitude = value.abs();
    if magnitude >= 65520.0 {
        return sign | 0x7C00;
    }
    if magnitude < 2f64.powi(-14) {
        // Subnormal: units of 2^-24 (1024 rounds up to the smallest normal).
        return sign | (magnitude * 2f64.powi(24)).round_ties_even() as u16;
    }
    let mut exp = ((magnitude.to_bits() >> 52) as i32) - 1023;
    let mut mantissa = (magnitude * 2f64.powi(10 - exp)).round_ties_even() as u16;
    if mantissa == 2048 {
        mantissa = 1024;
        exp += 1;
    }
    if exp > 15 {
        return sign | 0x7C00;
    }
    sign | (((exp + 15) as u16) << 10) | (mantissa - 1024)
}

/// FCVTXN: double to single, rounding to odd.
fn round_to_odd(raw: u64) -> u64 {
    let value = f64::from_bits(raw);
    if value.is_nan() {
        return fp_convert(raw, 64, 32);
    }
    let mut single = value as f32;
    if f64::from(single).abs() > value.abs() {
        single = f32::from_bits(single.to_bits() - 1);
    }
    if f64::from(single) != value {
        single = f32::from_bits(single.to_bits() | 1);
    }
    u64::from(single.to_bits())
}

/// VFPExpandImm: the 8-bit floating-point immediate of FMOV.
fn vfp_expand_imm(imm8: u32, size: u32) -> u64 {
    let exp_bits = if size == 32 { 8 } else { 11 };
    let fraction = size - exp_bits - 1;
    let sign = u64::from((imm8 >> 7) & 1);
    let b6 = u64::from((imm8 >> 6) & 1);
    let replicated = if b6 == 1 { mask(exp_bits - 3) } else { 0 };
    let exp = ((1 - b6) << (exp_bits - 1)) | (replicated << 2) | u64::from((imm8 >> 4) & 0x3);
    (sign << (size - 1)) | (exp << fraction) | (u64::from(imm8 & 0xF) << (fraction - 4))
}

// ---------------------------------------------------------------------------
// Scalar floating point

fn fp_size(ftype: u32) -> Option<u32> {
    match ftype {
        0 => Some(32),
        1 => Some(64),
        _ => None,
    }
}

fn scalar_fp(word: u32, s: &mut impl Regs) -> Option<()> {
    if bit(word, 29) {
        return None;
    }
    let (rd, rn, rm) = regs(word);
    let ftype = bits(word, 23, 22);
    if bit(word, 24) {
        // FMADD, FMSUB, FNMADD, FNMSUB
        if bit(word, 31) {
            return None;
        }
        let size = fp_size(ftype)?;
        let ra = bits(word, 14, 10);
        let n = lane(s.v(rn), size, 0);
        let m = lane(s.v(rm), size, 0);
        let a = lane(s.v(ra), size, 0);
        let flip = 1u64 << (size - 1);
        let (a, n) = match (bit(word, 21), bit(word, 15)) {
            (false, false) => (a, n),
            (false, true) => (a, n ^ flip),
            (true, false) => (a ^ flip, n ^ flip),
            (true, true) => (a ^ flip, n),
        };
        s.set_v(rd, u128::from(sized!(size, fp_mul_add(a, n, m))));
        return Some(());
    }
    if !bit(word, 21) {
        return fixed_conversion(word, s);
    }
    match bits(word, 11, 10) {
        0b01 => {
            // FCCMP, FCCMPE
            if bit(word, 31) {
                return None;
            }
            let size = fp_size(ftype)?;
            let n = lane(s.v(rn), size, 0);
            let m = lane(s.v(rm), size, 0);
            let flags = s.flags();
            let result = if condition_holds(bits(word, 15, 12), flags) {
                compare_flags(sized!(size, fp_compare(n, m)))
            } else {
                bits(word, 3, 0) as u8
            };
            s.set_flags(result);
        }
        0b10 => {
            if bit(word, 31) {
                return None;
            }
            let size = fp_size(ftype)?;
            let op = match bits(word, 15, 12) {
                0 => Fp2::Mul,
                1 => Fp2::Div,
                2 => Fp2::Add,
                3 => Fp2::Sub,
                4 => Fp2::Max,
                5 => Fp2::Min,
                6 => Fp2::MaxNum,
                7 => Fp2::MinNum,
                8 => Fp2::Nmul,
                _ => return None,
            };
            let n = lane(s.v(rn), size, 0);
            let m = lane(s.v(rm), size, 0);
            s.set_v(rd, u128::from(sized!(size, fp2(op, n, m))));
        }
        0b11 => {
            // FCSEL
            if bit(word, 31) {
                return None;
            }
            let size = fp_size(ftype)?;
            let n = s.v(rn);
            let m = s.v(rm);
            let flags = s.flags();
            let chosen = if condition_holds(bits(word, 15, 12), flags) {
                n
            } else {
                m
            };
            s.set_v(rd, u128::from(lane(chosen, size, 0)));
        }
        _ => {
            if bits(word, 15, 10) == 0 {
                return int_conversion(word, s);
            }
            if bit(word, 31) {
                return None;
            }
            if bits(word, 14, 10) == 0b10000 {
                return fp_one_source(word, s);
            }
            if bits(word, 13, 10) == 0b1000 {
                // FCMP, FCMPE (bit 3: against zero)
                if bits(word, 15, 14) != 0 || bits(word, 2, 0) != 0 {
                    return None;
                }
                let size = fp_size(ftype)?;
                let n = lane(s.v(rn), size, 0);
                let m = if bit(word, 3) {
                    0
                } else {
                    lane(s.v(rm), size, 0)
                };
                s.set_flags(compare_flags(sized!(size, fp_compare(n, m))));
                return Some(());
            }
            if bits(word, 12, 10) == 0b100 && rn == 0 {
                // FMOV (scalar, immediate)
                let size = fp_size(ftype)?;
                s.set_v(rd, u128::from(vfp_expand_imm(bits(word, 20, 13), size)));
                return Some(());
            }
            return None;
        }
    }
    Some(())
}

fn fp_one_source(word: u32, s: &mut impl Regs) -> Option<()> {
    let (rd, rn, _) = regs(word);
    let opcode = bits(word, 20, 15);
    let from = match bits(word, 23, 22) {
        0 => 32,
        1 => 64,
        3 => 16,
        _ => return None,
    };
    if opcode & 0b11_1100 == 0b00_0100 {
        // FCVT
        let to = match opcode & 3 {
            0 => 32,
            1 => 64,
            3 => 16,
            _ => return None,
        };
        if to == from {
            return None;
        }
        let n = lane(s.v(rn), from, 0);
        s.set_v(rd, u128::from(fp_convert(n, from, to)));
        return Some(());
    }
    if from == 16 {
        return None;
    }
    let op = match opcode {
        0 => None,
        1 => Some(Fp1::Abs),
        2 => Some(Fp1::Neg),
        3 => Some(Fp1::Sqrt),
        8 => Some(Fp1::Round(Rounding::TiesEven)),
        9 => Some(Fp1::Round(Rounding::Up)),
        10 => Some(Fp1::Round(Rounding::Down)),
        11 => Some(Fp1::Round(Rounding::Zero)),
        12 => Some(Fp1::Round(Rounding::TiesAway)),
        // FRINTX, FRINTI: the current rounding mode.
        14 | 15 => Some(Fp1::Round(Rounding::TiesEven)),
        _ => return None,
    };
    let n = lane(s.v(rn), from, 0);
    let result = match op {
        Some(op) => sized!(from, fp1(op, n)),
        None => n,
    };
    s.set_v(rd, u128::from(result));
    Some(())
}

/// Conversions between floating point and integers, and FMOV to and from
/// general registers.
fn int_conversion(word: u32, s: &mut impl Regs) -> Option<()> {
    let (rd, rn, _) = regs(word);
    let sf = bit(word, 31);
    let ftype = bits(word, 23, 22);
    let (rmode, opcode) = (bits(word, 20, 19), bits(word, 18, 16));
    let int_bits = if sf { 64 } else { 32 };
    match (rmode, opcode) {
        (_, 0b110) | (_, 0b111) => match (ftype, sf, rmode, opcode) {
            (0, false, 0, 0b110) => {
                let n = s.v(rn);
                s.set_x(rd, lane(n, 32, 0));
            }
            (0, false, 0, 0b111) => {
                let x = s.x(rn);
                s.set_v(rd, u128::from(x & mask(32)));
            }
            (1, true, 0, 0b110) => {
                let n = s.v(rn);
                s.set_x(rd, lane(n, 64, 0));
            }
            (1, true, 0, 0b111) => {
                let x = s.x(rn);
                s.set_v(rd, u128::from(x));
            }
            (2, true, 1, 0b110) => {
                let n = s.v(rn);
                s.set_x(rd, lane(n, 64, 1));
            }
            (2, true, 1, 0b111) => {
                let x = s.x(rn);
                let d = s.v(rd);
                s.set_v(rd, put(d, 64, 1, x));
            }
            _ => return None,
        },
        (0, 0b010) | (0, 0b011) => {
            let size = fp_size(ftype)?;
            let x = s.x(rn);
            let unsigned = opcode == 0b011;
            s.set_v(
                rd,
                u128::from(sized!(size, int_to_fp(x, unsigned, int_bits, 0))),
            );
        }
        (_, 0b000) | (_, 0b001) | (0, 0b100) | (0, 0b101) => {
            let size = fp_size(ftype)?;
            let rounding = match (rmode, opcode) {
                (0, 0b100) | (0, 0b101) => Rounding::TiesAway,
                (0, _) => Rounding::TiesEven,
                (1, _) => Rounding::Up,
                (2, _) => Rounding::Down,
                _ => Rounding::Zero,
            };
            let n = lane(s.v(rn), size, 0);
            let unsigned = opcode & 1 == 1;
            s.set_x(
                rd,
                sized!(size, fp_to_int(n, rounding, 0, unsigned, int_bits)),
            );
        }
        _ => return None,
    }
    Some(())
}

/// SCVTF/UCVTF/FCVTZS/FCVTZU (scalar, fixed-point).
fn fixed_conversion(word: u32, s: &mut impl Regs) -> Option<()> {
    let (rd, rn, _) = regs(word);
    let sf = bit(word, 31);
    let size = fp_size(bits(word, 23, 22))?;
    let scale = bits(word, 15, 10);
    if !sf && scale < 32 {
        return None;
    }
    let fbits = 64 - scale;
    let int_bits = if sf { 64 } else { 32 };
    let opcode = bits(word, 18, 16);
    match (bits(word, 20, 19), opcode) {
        (0, 0b010) | (0, 0b011) => {
            let x = s.x(rn);
            let unsigned = opcode == 0b011;
            s.set_v(
                rd,
                u128::from(sized!(size, int_to_fp(x, unsigned, int_bits, fbits))),
            );
        }
        (3, 0b000) | (3, 0b001) => {
            let n = lane(s.v(rn), size, 0);
            let unsigned = opcode == 0b001;
            s.set_x(
                rd,
                sized!(
                    size,
                    fp_to_int(n, Rounding::Zero, fbits, unsigned, int_bits)
                ),
            );
        }
        _ => return None,
    }
    Some(())
}

// ---------------------------------------------------------------------------
// Advanced SIMD: three registers, two-register misc, across lanes, permutes

fn simd_three_reg(word: u32, s: &mut impl Regs, scalar: bool) -> Option<()> {
    if bit(word, 21) {
        if bit(word, 10) {
            return three_same(word, s, scalar);
        }
        if !bit(word, 11) {
            return three_different(word, s, scalar);
        }
        return match bits(word, 20, 17) {
            0b0000 => two_misc(word, s, scalar),
            0b1000 if scalar => scalar_pairwise(word, s),
            0b1000 => across_lanes(word, s),
            _ => None,
        };
    }
    if scalar {
        // DUP (element), scalar: MOV Vd, Vn.T[i]
        if word & 0xFFE0_FC00 != 0x5E00_0400 {
            return None;
        }
        let (rd, rn, _) = regs(word);
        let imm5 = bits(word, 20, 16);
        let size = imm5.trailing_zeros();
        if size > 3 {
            return None;
        }
        let n = s.v(rn);
        s.set_v(rd, u128::from(lane(n, 8 << size, imm5 >> (size + 1))));
        return Some(());
    }
    if word & 0xBFE0_8400 == 0x2E00_0000 {
        return extract(word, s);
    }
    if word & 0xBF20_8C00 == 0x0E00_0800 {
        return permute(word, s);
    }
    if word & 0xBFE0_9C00 == 0x0E00_0000 {
        return table_lookup(word, s);
    }
    None
}

fn three_same(word: u32, s: &mut impl Regs, scalar: bool) -> Option<()> {
    let (rd, rn, rm) = regs(word);
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 15, 11),
    );
    if opcode >= 0b11000 {
        return fp_three_same(word, s, scalar);
    }
    if opcode == 0b00011 {
        // AND, BIC, ORR, ORN, EOR, BSL, BIT, BIF
        if scalar {
            return None;
        }
        let n = s.v(rn);
        let m = s.v(rm);
        let result = match (u, size) {
            (false, 0) => n & m,
            (false, 1) => n & !m,
            (false, 2) => n | m,
            (false, _) => n | !m,
            (true, 0) => n ^ m,
            (true, _) => {
                let d = s.v(rd);
                match size {
                    1 => (d & n) | (!d & m),
                    2 => (d & !m) | (n & m),
                    _ => (d & m) | (n & !m),
                }
            }
        };
        s.set_v(rd, if q { result } else { low(result) });
        return Some(());
    }
    let esize = 8 << size;
    let valid = if scalar {
        match opcode {
            0b00001 | 0b00101 | 0b01001 | 0b01011 => true,
            0b00110 | 0b00111 | 0b01000 | 0b01010 | 0b10000 | 0b10001 => size == 3,
            0b10110 => size == 1 || size == 2,
            _ => false,
        }
    } else {
        match opcode {
            _ if size == 3 && !q => false,
            0b10011 if u => size == 0,
            0b10111 => !u,
            0b10110 => size == 1 || size == 2,
            0b00000 | 0b00010 | 0b00100 | 0b01100..=0b01111 | 0b10010..=0b10101 => size != 3,
            _ => true,
        }
    };
    if !valid {
        return None;
    }
    let n = s.v(rn);
    let m = s.v(rm);
    let d = if matches!(opcode, 0b01111 | 0b10010) {
        s.v(rd)
    } else {
        0
    };
    let lanes = if scalar { 1 } else { width(q) / esize };
    let mut result = 0u128;
    for i in 0..lanes {
        let value = if matches!(opcode, 0b10100 | 0b10101 | 0b10111) {
            let (a, b) = pair(n, m, esize, lanes, i);
            let (x, y) = (int(a, esize, u), int(b, esize, u));
            match opcode {
                0b10100 => wrap(x.max(y), esize),
                0b10101 => wrap(x.min(y), esize),
                _ => wrap(x + y, esize),
            }
        } else {
            int_three_same(
                u,
                opcode,
                esize,
                lane(n, esize, i),
                lane(m, esize, i),
                lane(d, esize, i),
            )?
        };
        result = put(result, esize, i, value);
    }
    s.set_v(rd, result);
    Some(())
}

fn int_three_same(u: bool, opcode: u32, esize: u32, a: u64, b: u64, d: u64) -> Option<u64> {
    let (x, y) = (int(a, esize, u), int(b, esize, u));
    let (sx, sy) = (int(a, esize, false), int(b, esize, false));
    Some(match opcode {
        0b00000 => wrap((x + y) >> 1, esize),
        0b00001 => sat(x + y, esize, u),
        0b00010 => wrap((x + y + 1) >> 1, esize),
        0b00100 => wrap((x - y) >> 1, esize),
        0b00101 => sat(x - y, esize, u),
        0b00110 => ones(x > y, esize),
        0b00111 => ones(x >= y, esize),
        0b01000..=0b01011 => {
            let shift = i32::from(b as u8 as i8);
            shift_by(x, shift, opcode & 2 != 0, opcode & 1 != 0, esize, u)
        }
        0b01100 => wrap(x.max(y), esize),
        0b01101 => wrap(x.min(y), esize),
        0b01110 => wrap((x - y).abs(), esize),
        0b01111 => wrap(i128::from(d) + (x - y).abs(), esize),
        0b10000 if u => wrap(x - y, esize),
        0b10000 => wrap(x + y, esize),
        0b10001 if u => ones(a & mask(esize) == b & mask(esize), esize),
        0b10001 => ones(a & b & mask(esize) != 0, esize),
        0b10010 if u => wrap(i128::from(d) - x * y, esize),
        0b10010 => wrap(i128::from(d) + x * y, esize),
        0b10011 if u => (clmul(a & 0xFF, b & 0xFF) & 0xFF) as u64,
        0b10011 => wrap(x * y, esize),
        // SQDMULH, SQRDMULH
        0b10110 => {
            let round = if u { 1i128 << (esize - 1) } else { 0 };
            sat((2 * sx * sy + round) >> esize, esize, false)
        }
        _ => return None,
    })
}

/// SSHL/USHL and their rounding and saturating forms: shift left by a
/// signed amount (negative shifts right).
fn shift_by(
    x: i128,
    shift: i32,
    rounding: bool,
    saturating: bool,
    esize: u32,
    unsigned: bool,
) -> u64 {
    let value = if shift >= 0 {
        if x == 0 {
            0
        } else if shift > 63 {
            // Any nonzero value overflows; the low bits are zero.
            x.signum() << 100
        } else {
            x << shift
        }
    } else {
        let amount = (-shift).min(100) as u32;
        let round = if rounding { 1i128 << (amount - 1) } else { 0 };
        (x + round) >> amount
    };
    if saturating {
        sat(value, esize, unsigned)
    } else {
        wrap(value, esize)
    }
}

/// Carry-less (polynomial) multiplication.
fn clmul(a: u64, b: u64) -> u128 {
    (0..64)
        .filter(|i| (b >> i) & 1 == 1)
        .fold(0u128, |acc, i| acc ^ (u128::from(a) << i))
}

fn fp_three_same(word: u32, s: &mut impl Regs, scalar: bool) -> Option<()> {
    #[derive(Clone, Copy)]
    enum Kind {
        Two(Fp2),
        Pair(Fp2),
        Fma(bool),
        Cmp(Cmp),
    }
    let (rd, rn, rm) = regs(word);
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 15, 11),
    );
    let esize = if size & 1 == 1 { 64 } else { 32 };
    if !scalar && esize == 64 && !q {
        return None;
    }
    let kind = match (u, size >> 1, opcode) {
        (false, 0, 0b11000) => Kind::Two(Fp2::MaxNum),
        (false, 1, 0b11000) => Kind::Two(Fp2::MinNum),
        (false, 0, 0b11001) => Kind::Fma(false),
        (false, 1, 0b11001) => Kind::Fma(true),
        (false, 0, 0b11010) => Kind::Two(Fp2::Add),
        (false, 1, 0b11010) => Kind::Two(Fp2::Sub),
        (false, 0, 0b11011) => Kind::Two(Fp2::MulX),
        (false, 0, 0b11100) => Kind::Cmp(Cmp::Eq),
        (false, 0, 0b11110) => Kind::Two(Fp2::Max),
        (false, 1, 0b11110) => Kind::Two(Fp2::Min),
        (false, 0, 0b11111) => Kind::Two(Fp2::RecipStep),
        (false, 1, 0b11111) => Kind::Two(Fp2::RsqrtStep),
        (true, 0, 0b11000) => Kind::Pair(Fp2::MaxNum),
        (true, 1, 0b11000) => Kind::Pair(Fp2::MinNum),
        (true, 0, 0b11010) => Kind::Pair(Fp2::Add),
        (true, 1, 0b11010) => Kind::Two(Fp2::Abd),
        (true, 0, 0b11011) => Kind::Two(Fp2::Mul),
        (true, 0, 0b11100) => Kind::Cmp(Cmp::Ge),
        (true, 1, 0b11100) => Kind::Cmp(Cmp::Gt),
        (true, 0, 0b11101) => Kind::Cmp(Cmp::AbsGe),
        (true, 1, 0b11101) => Kind::Cmp(Cmp::AbsGt),
        (true, 0, 0b11110) => Kind::Pair(Fp2::Max),
        (true, 1, 0b11110) => Kind::Pair(Fp2::Min),
        (true, 0, 0b11111) => Kind::Two(Fp2::Div),
        _ => return None,
    };
    if scalar
        && !matches!(
            kind,
            Kind::Two(Fp2::MulX | Fp2::RecipStep | Fp2::RsqrtStep | Fp2::Abd) | Kind::Cmp(_)
        )
    {
        return None;
    }
    let n = s.v(rn);
    let m = s.v(rm);
    let d = if matches!(kind, Kind::Fma(_)) {
        s.v(rd)
    } else {
        0
    };
    let lanes = if scalar { 1 } else { width(q) / esize };
    let mut result = 0u128;
    for i in 0..lanes {
        let (a, b) = match kind {
            Kind::Pair(_) => pair(n, m, esize, lanes, i),
            _ => (lane(n, esize, i), lane(m, esize, i)),
        };
        let value = match kind {
            Kind::Two(op) | Kind::Pair(op) => sized!(esize, fp2(op, a, b)),
            Kind::Fma(subtract) => {
                let a = if subtract { a ^ (1 << (esize - 1)) } else { a };
                sized!(esize, fp_mul_add(lane(d, esize, i), a, b))
            }
            Kind::Cmp(cmp) => ones(sized!(esize, fp_cmp(cmp, a, b)), esize),
        };
        result = put(result, esize, i, value);
    }
    s.set_v(rd, result);
    Some(())
}

fn three_different(word: u32, s: &mut impl Regs, scalar: bool) -> Option<()> {
    let (rd, rn, rm) = regs(word);
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 15, 12),
    );
    let esize = 8 << size;
    let wide = esize * 2;
    if !u && opcode == 0b1110 {
        // PMULL: 8 x 8 -> 16 bits per lane, or one 64 x 64 -> 128.
        if scalar {
            return None;
        }
        let n = s.v(rn);
        let m = s.v(rm);
        let result = match size {
            0 => {
                let (n, m) = (half(n, q), half(m, q));
                (0..8).fold(0u128, |acc, i| {
                    put(acc, 16, i, clmul(lane(n, 8, i), lane(m, 8, i)) as u64)
                })
            }
            3 => clmul(lane(n, 64, u32::from(q)), lane(m, 64, u32::from(q))),
            _ => return None,
        };
        s.set_v(rd, result);
        return Some(());
    }
    let saturating = matches!(opcode, 0b1001 | 0b1011 | 0b1101);
    let valid = if scalar {
        !u && saturating && (size == 1 || size == 2)
    } else {
        size != 3
            && opcode != 0b1111
            && opcode != 0b1110
            && (!saturating || (!u && (size == 1 || size == 2)))
    };
    if !valid {
        return None;
    }
    let n = s.v(rn);
    let m = s.v(rm);
    if matches!(opcode, 0b0100 | 0b0110) {
        // ADDHN, RADDHN, SUBHN, RSUBHN
        let round = if u { 1i128 << (esize - 1) } else { 0 };
        let mut narrowed = 0u128;
        for i in 0..64 / esize {
            let (a, b) = (
                int(lane(n, wide, i), wide, true),
                int(lane(m, wide, i), wide, true),
            );
            let sum = if opcode == 0b0100 { a + b } else { a - b } + round;
            narrowed = put(narrowed, esize, i, wrap(sum, wide) >> esize);
        }
        write_narrow(s, rd, q, narrowed);
        return Some(());
    }
    let accumulate = matches!(opcode, 0b0101 | 0b1000 | 0b1001 | 0b1010 | 0b1011);
    let d = if accumulate { s.v(rd) } else { 0 };
    let lanes = if scalar { 1 } else { 64 / esize };
    let (nh, mh) = if scalar {
        (n, m)
    } else {
        (half(n, q), half(m, q))
    };
    let mut result = 0u128;
    for i in 0..lanes {
        let x = if matches!(opcode, 0b0001 | 0b0011) {
            int(lane(n, wide, i), wide, u)
        } else {
            int(lane(nh, esize, i), esize, u)
        };
        let y = int(lane(mh, esize, i), esize, u);
        let acc = int(lane(d, wide, i), wide, u);
        let doubled = || sext(sat(2 * x * y, wide, false), wide) as i128;
        let value = match opcode {
            0b0000 | 0b0001 => wrap(x + y, wide),
            0b0010 | 0b0011 => wrap(x - y, wide),
            0b0101 => wrap(acc + (x - y).abs(), wide),
            0b0111 => wrap((x - y).abs(), wide),
            0b1000 => wrap(acc + x * y, wide),
            0b1010 => wrap(acc - x * y, wide),
            0b1100 => wrap(x * y, wide),
            0b1001 => sat(acc + doubled(), wide, false),
            0b1011 => sat(acc - doubled(), wide, false),
            _ => sat(2 * x * y, wide, false),
        };
        result = put(result, wide, i, value);
    }
    s.set_v(rd, result);
    Some(())
}

fn two_misc(word: u32, s: &mut impl Regs, scalar: bool) -> Option<()> {
    let (rd, rn, _) = regs(word);
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 16, 12),
    );
    let is_fp = match opcode {
        0b01100..=0b01111 => size >= 2,
        0b11100 => size < 2,
        0b10110..=0b11111 => true,
        _ => false,
    };
    if is_fp {
        return fp_two_misc(word, s, scalar);
    }
    let esize = 8 << size;
    let lanes = if scalar { 1 } else { width(q) / esize };
    let n = s.v(rn);
    let mut result = 0u128;
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
            let per = container / esize;
            for i in 0..lanes {
                let base = i / per * per;
                result = put(
                    result,
                    esize,
                    i,
                    lane(n, esize, base + per - 1 - (i - base)),
                );
            }
        }
        (_, 0b00010) | (_, 0b00110) => {
            // SADDLP, UADDLP, SADALP, UADALP
            if scalar || size == 3 {
                return None;
            }
            let wide = esize * 2;
            let d = if opcode == 0b00110 { s.v(rd) } else { 0 };
            for i in 0..width(q) / wide {
                let sum = int(lane(n, esize, 2 * i), esize, u)
                    + int(lane(n, esize, 2 * i + 1), esize, u)
                    + i128::from(lane(d, wide, i));
                result = put(result, wide, i, wrap(sum, wide));
            }
        }
        (_, 0b00011) => {
            // SUQADD, USQADD
            if !scalar && size == 3 && !q {
                return None;
            }
            let d = s.v(rd);
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
        (_, 0b00100) => {
            // CLS, CLZ
            if scalar || size == 3 {
                return None;
            }
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
            // CNT, NOT, RBIT
            if scalar || size > 1 || (!u && size != 0) {
                return None;
            }
            for i in 0..width(q) / 8 {
                let x = lane(n, 8, i) as u8;
                let value = match (u, size) {
                    (false, _) => x.count_ones() as u8,
                    (true, 0) => !x,
                    _ => x.reverse_bits(),
                };
                result = put(result, 8, i, u64::from(value));
            }
        }
        (_, 0b00111) | (_, 0b01000..=0b01011) => {
            // SQABS, SQNEG, compares with zero, ABS, NEG
            if (scalar && opcode != 0b00111 && size != 3) || (!scalar && size == 3 && !q) {
                return None;
            }
            if u && opcode == 0b01010 {
                return None;
            }
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
        (_, 0b10010) | (_, 0b10100) => {
            // XTN, SQXTUN, SQXTN, UQXTN
            if size == 3 || (scalar && !u && opcode == 0b10010) {
                return None;
            }
            let wide = esize * 2;
            let narrow_lanes = if scalar { 1 } else { 64 / esize };
            for i in 0..narrow_lanes {
                let x = lane(n, wide, i);
                let value = match (u, opcode) {
                    (false, 0b10010) => wrap(i128::from(x), esize),
                    (true, 0b10010) => sat(int(x, wide, false), esize, true),
                    (false, _) => sat(int(x, wide, false), esize, false),
                    (true, _) => sat(int(x, wide, true), esize, true),
                };
                result = put(result, esize, i, value);
            }
            if scalar {
                s.set_v(rd, result);
            } else {
                write_narrow(s, rd, q, result);
            }
            return Some(());
        }
        (true, 0b10011) => {
            // SHLL
            if scalar || size == 3 {
                return None;
            }
            let source = half(n, q);
            for i in 0..64 / esize {
                result = put(result, esize * 2, i, lane(source, esize, i) << esize);
            }
        }
        _ => return None,
    }
    s.set_v(rd, result);
    Some(())
}

fn fp_two_misc(word: u32, s: &mut impl Regs, scalar: bool) -> Option<()> {
    #[derive(Clone, Copy)]
    enum Kind {
        One(Fp1),
        Cmp(Cmp),
        ToInt(Rounding, bool),
        FromInt(bool),
    }
    let (rd, rn, _) = regs(word);
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 16, 12),
    );
    let (a, sz) = (size >> 1, size & 1);
    let esize = if sz == 1 { 64 } else { 32 };
    match (u, a, opcode) {
        (_, 0, 0b10110) => {
            // FCVTN, FCVTXN (double to single by rounding to odd)
            if u && sz == 0 {
                return None;
            }
            let n = s.v(rn);
            if scalar {
                if !u {
                    return None;
                }
                s.set_v(rd, u128::from(round_to_odd(lane(n, 64, 0))));
                return Some(());
            }
            let (from, to) = if sz == 1 { (64, 32) } else { (32, 16) };
            let mut narrowed = 0u128;
            for i in 0..64 / to {
                let x = lane(n, from, i);
                let value = if u {
                    round_to_odd(x)
                } else {
                    fp_convert(x, from, to)
                };
                narrowed = put(narrowed, to, i, value);
            }
            write_narrow(s, rd, q, narrowed);
            return Some(());
        }
        (false, 0, 0b10111) => {
            // FCVTL
            if scalar {
                return None;
            }
            let (from, to) = if sz == 1 { (32, 64) } else { (16, 32) };
            let source = half(s.v(rn), q);
            let mut result = 0u128;
            for i in 0..128 / to {
                result = put(result, to, i, fp_convert(lane(source, from, i), from, to));
            }
            s.set_v(rd, result);
            return Some(());
        }
        _ => {}
    }
    let kind = match (u, a, opcode) {
        (false, 1, 0b01100) => Kind::Cmp(Cmp::Gt),
        (false, 1, 0b01101) => Kind::Cmp(Cmp::Eq),
        (false, 1, 0b01110) => Kind::Cmp(Cmp::Lt),
        (true, 1, 0b01100) => Kind::Cmp(Cmp::Ge),
        (true, 1, 0b01101) => Kind::Cmp(Cmp::Le),
        (false, 1, 0b01111) => Kind::One(Fp1::Abs),
        (true, 1, 0b01111) => Kind::One(Fp1::Neg),
        (true, 1, 0b11111) => Kind::One(Fp1::Sqrt),
        (false, 0, 0b11000) => Kind::One(Fp1::Round(Rounding::TiesEven)),
        (false, 0, 0b11001) => Kind::One(Fp1::Round(Rounding::Down)),
        (false, 1, 0b11000) => Kind::One(Fp1::Round(Rounding::Up)),
        (false, 1, 0b11001) => Kind::One(Fp1::Round(Rounding::Zero)),
        (true, 0, 0b11000) => Kind::One(Fp1::Round(Rounding::TiesAway)),
        // FRINTX, FRINTI: the current rounding mode.
        (true, 0, 0b11001) | (true, 1, 0b11001) => Kind::One(Fp1::Round(Rounding::TiesEven)),
        (_, 0, 0b11010) => Kind::ToInt(Rounding::TiesEven, u),
        (_, 0, 0b11011) => Kind::ToInt(Rounding::Down, u),
        (_, 0, 0b11100) => Kind::ToInt(Rounding::TiesAway, u),
        (_, 1, 0b11010) => Kind::ToInt(Rounding::Up, u),
        (_, 1, 0b11011) => Kind::ToInt(Rounding::Zero, u),
        (_, 0, 0b11101) => Kind::FromInt(u),
        (false, 1, 0b11101) => Kind::One(Fp1::RecipEstimate),
        (true, 1, 0b11101) => Kind::One(Fp1::RsqrtEstimate),
        _ => return None,
    };
    if scalar
        && matches!(
            kind,
            Kind::One(Fp1::Abs | Fp1::Neg | Fp1::Sqrt | Fp1::Round(_))
        )
    {
        return None;
    }
    if !scalar && esize == 64 && !q {
        return None;
    }
    let lanes = if scalar { 1 } else { width(q) / esize };
    let n = s.v(rn);
    let mut result = 0u128;
    for i in 0..lanes {
        let x = lane(n, esize, i);
        let value = match kind {
            Kind::One(op) => sized!(esize, fp1(op, x)),
            Kind::Cmp(cmp) => ones(sized!(esize, fp_cmp(cmp, x, 0)), esize),
            Kind::ToInt(rounding, unsigned) => {
                sized!(esize, fp_to_int(x, rounding, 0, unsigned, esize))
            }
            Kind::FromInt(unsigned) => sized!(esize, int_to_fp(x, unsigned, esize, 0)),
        };
        result = put(result, esize, i, value);
    }
    s.set_v(rd, result);
    Some(())
}

fn across_lanes(word: u32, s: &mut impl Regs) -> Option<()> {
    let (rd, rn, _) = regs(word);
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 16, 12),
    );
    if opcode == 0b01100 || opcode == 0b01111 {
        // FMAXNMV, FMINNMV, FMAXV, FMINV (4S only)
        if !u || !q || size & 1 == 1 {
            return None;
        }
        let op = match (opcode, size >> 1) {
            (0b01100, 0) => Fp2::MaxNum,
            (0b01100, _) => Fp2::MinNum,
            (_, 0) => Fp2::Max,
            _ => Fp2::Min,
        };
        let n = s.v(rn);
        let f = |a, b| fp2::<f32>(op, a, b);
        let result = f(
            f(lane(n, 32, 0), lane(n, 32, 1)),
            f(lane(n, 32, 2), lane(n, 32, 3)),
        );
        s.set_v(rd, u128::from(result));
        return Some(());
    }
    let esize = 8 << size;
    if size == 3 || (size == 2 && !q) {
        return None;
    }
    let lanes = width(q) / esize;
    let n = s.v(rn);
    let values = (0..lanes).map(|i| int(lane(n, esize, i), esize, u));
    let result = match (u, opcode) {
        (_, 0b00011) => wrap(values.sum(), esize * 2),
        (_, 0b01010) => wrap(values.max()?, esize),
        (_, 0b11010) => wrap(values.min()?, esize),
        (false, 0b11011) => wrap(values.sum(), esize),
        _ => return None,
    };
    s.set_v(rd, u128::from(result));
    Some(())
}

fn scalar_pairwise(word: u32, s: &mut impl Regs) -> Option<()> {
    let (rd, rn, _) = regs(word);
    let (u, size, opcode) = (bit(word, 29), bits(word, 23, 22), bits(word, 16, 12));
    let result = match (u, opcode) {
        (false, 0b11011) if size == 3 => {
            let n = s.v(rn);
            lane(n, 64, 0).wrapping_add(lane(n, 64, 1))
        }
        (true, 0b01100 | 0b01101 | 0b01111) => {
            let esize = if size & 1 == 1 { 64 } else { 32 };
            let op = match (opcode, size >> 1) {
                (0b01100, 0) => Fp2::MaxNum,
                (0b01100, _) => Fp2::MinNum,
                (0b01101, 0) => Fp2::Add,
                (0b01111, 0) => Fp2::Max,
                (0b01111, _) => Fp2::Min,
                _ => return None,
            };
            let n = s.v(rn);
            sized!(esize, fp2(op, lane(n, esize, 0), lane(n, esize, 1)))
        }
        _ => return None,
    };
    s.set_v(rd, u128::from(result));
    Some(())
}

/// EXT: bytes of `m:n` starting at byte `imm4`.
fn extract(word: u32, s: &mut impl Regs) -> Option<()> {
    let (rd, rn, rm) = regs(word);
    let q = bit(word, 30);
    let position = bits(word, 14, 11) as usize;
    let bytes = if q { 16 } else { 8 };
    if position >= bytes {
        return None;
    }
    let n = s.v(rn).to_le_bytes();
    let m = s.v(rm).to_le_bytes();
    let mut concat = [0u8; 32];
    concat[..bytes].copy_from_slice(&n[..bytes]);
    concat[bytes..2 * bytes].copy_from_slice(&m[..bytes]);
    let mut out = [0u8; 16];
    out[..bytes].copy_from_slice(&concat[position..position + bytes]);
    s.set_v(rd, u128::from_le_bytes(out));
    Some(())
}

/// UZP1/2, TRN1/2, ZIP1/2.
fn permute(word: u32, s: &mut impl Regs) -> Option<()> {
    let (rd, rn, rm) = regs(word);
    let (q, size, opcode) = (bit(word, 30), bits(word, 23, 22), bits(word, 14, 12));
    let esize = 8 << size;
    if size == 3 && !q {
        return None;
    }
    let lanes = width(q) / esize;
    let part = opcode >> 2;
    let n = s.v(rn);
    let m = s.v(rm);
    let mut result = 0u128;
    for i in 0..lanes {
        let value = match opcode & 3 {
            1 => {
                let j = 2 * i + part;
                if j < lanes {
                    lane(n, esize, j)
                } else {
                    lane(m, esize, j - lanes)
                }
            }
            2 => {
                let j = (i & !1) + part;
                if i & 1 == 0 {
                    lane(n, esize, j)
                } else {
                    lane(m, esize, j)
                }
            }
            3 => {
                let j = part * lanes / 2 + i / 2;
                if i & 1 == 0 {
                    lane(n, esize, j)
                } else {
                    lane(m, esize, j)
                }
            }
            _ => return None,
        };
        result = put(result, esize, i, value);
    }
    s.set_v(rd, result);
    Some(())
}

/// TBL/TBX over one to four consecutive table registers.
fn table_lookup(word: u32, s: &mut impl Regs) -> Option<()> {
    let (rd, rn, rm) = regs(word);
    let q = bit(word, 30);
    let count = bits(word, 14, 13) as usize + 1;
    let mut table = [0u8; 64];
    for i in 0..count {
        let v = s.v((rn + i as u32) % 32);
        table[16 * i..16 * i + 16].copy_from_slice(&v.to_le_bytes());
    }
    let indices = s.v(rm).to_le_bytes();
    let fallback = if bit(word, 12) { s.v(rd) } else { 0 }.to_le_bytes();
    let mut out = [0u8; 16];
    for i in 0..if q { 16 } else { 8 } {
        let index = usize::from(indices[i]);
        out[i] = if index < 16 * count {
            table[index]
        } else {
            fallback[i]
        };
    }
    s.set_v(rd, u128::from_le_bytes(out));
    Some(())
}

// ---------------------------------------------------------------------------
// Advanced SIMD: by element, shift by immediate

fn simd_element_or_shift(word: u32, s: &mut impl Regs, scalar: bool) -> Option<()> {
    if !bit(word, 10) {
        return by_element(word, s, scalar);
    }
    // Shift by immediate needs immh != 0; immh == 0 is a modified immediate,
    // which the lifter handles itself.
    if bit(word, 23) || bits(word, 22, 19) == 0 {
        return None;
    }
    shift_immediate(word, s, scalar)
}

fn by_element(word: u32, s: &mut impl Regs, scalar: bool) -> Option<()> {
    let (rd, rn, _) = regs(word);
    let (q, u, size, opcode) = (
        bit(word, 30),
        bit(word, 29),
        bits(word, 23, 22),
        bits(word, 15, 12),
    );
    let (l, m_bit, h) = (bits(word, 21, 21), bits(word, 20, 20), bits(word, 11, 11));
    if matches!(
        (u, opcode),
        (false, 0b0001 | 0b0101 | 0b1001) | (true, 0b1001)
    ) {
        // FMLA, FMLS, FMUL, FMULX (by element)
        if size < 2 {
            return None;
        }
        let esize = if size & 1 == 1 { 64 } else { 32 };
        let index = if esize == 64 {
            if l == 1 {
                return None;
            }
            h
        } else {
            (h << 1) | l
        };
        if !scalar && esize == 64 && !q {
            return None;
        }
        let rm = bits(word, 20, 16);
        let n = s.v(rn);
        let element = lane(s.v(rm), esize, index);
        let accumulate = matches!(opcode, 0b0001 | 0b0101);
        let d = if accumulate { s.v(rd) } else { 0 };
        let lanes = if scalar { 1 } else { width(q) / esize };
        let mut result = 0u128;
        for i in 0..lanes {
            let a = lane(n, esize, i);
            let value = match (u, opcode) {
                (false, 0b0001) => sized!(esize, fp_mul_add(lane(d, esize, i), a, element)),
                (false, 0b0101) => sized!(
                    esize,
                    fp_mul_add(lane(d, esize, i), a ^ (1 << (esize - 1)), element)
                ),
                (false, _) => sized!(esize, fp2(Fp2::Mul, a, element)),
                _ => sized!(esize, fp2(Fp2::MulX, a, element)),
            };
            result = put(result, esize, i, value);
        }
        s.set_v(rd, result);
        return Some(());
    }
    let (esize, index, rm) = match size {
        1 => (16, (h << 2) | (l << 1) | m_bit, bits(word, 19, 16)),
        2 => (32, (h << 1) | l, bits(word, 20, 16)),
        _ => return None,
    };
    let long = matches!(opcode, 0b0010 | 0b0011 | 0b0110 | 0b0111 | 0b1010 | 0b1011);
    let valid = match (u, opcode) {
        (true, 0b0000 | 0b0100) | (false, 0b1000) | (_, 0b0010 | 0b0110 | 0b1010) => !scalar,
        (false, 0b0011 | 0b0111 | 0b1011 | 0b1100 | 0b1101) => true,
        _ => false,
    };
    if !valid {
        return None;
    }
    let n = s.v(rn);
    let element = lane(s.v(rm), esize, index);
    let accumulate = matches!(opcode, 0b0000 | 0b0010 | 0b0011 | 0b0100 | 0b0110 | 0b0111);
    let d = if accumulate { s.v(rd) } else { 0 };
    let mut result = 0u128;
    if long {
        let wide = esize * 2;
        let source = if scalar { n } else { half(n, q) };
        let lanes = if scalar { 1 } else { 64 / esize };
        let unsigned = u;
        let y = int(element, esize, unsigned);
        for i in 0..lanes {
            let x = int(lane(source, esize, i), esize, unsigned);
            let acc = int(lane(d, wide, i), wide, unsigned);
            let doubled = || sext(sat(2 * x * y, wide, false), wide) as i128;
            let value = match opcode {
                0b0010 => wrap(acc + x * y, wide),
                0b0110 => wrap(acc - x * y, wide),
                0b1010 => wrap(x * y, wide),
                0b0011 => sat(acc + doubled(), wide, false),
                0b0111 => sat(acc - doubled(), wide, false),
                _ => sat(2 * x * y, wide, false),
            };
            result = put(result, wide, i, value);
        }
    } else {
        let lanes = if scalar { 1 } else { width(q) / esize };
        let y = int(element, esize, false);
        for i in 0..lanes {
            let x = int(lane(n, esize, i), esize, false);
            let value = match opcode {
                0b0000 => wrap(i128::from(lane(d, esize, i)) + x * y, esize),
                0b0100 => wrap(i128::from(lane(d, esize, i)) - x * y, esize),
                0b1000 => wrap(x * y, esize),
                0b1100 => sat((2 * x * y) >> esize, esize, false),
                _ => sat((2 * x * y + (1 << (esize - 1))) >> esize, esize, false),
            };
            result = put(result, esize, i, value);
        }
    }
    s.set_v(rd, result);
    Some(())
}

fn shift_immediate(word: u32, s: &mut impl Regs, scalar: bool) -> Option<()> {
    let (rd, rn, _) = regs(word);
    let (q, u, opcode) = (bit(word, 30), bit(word, 29), bits(word, 15, 11));
    let immh = bits(word, 22, 19);
    let imm = bits(word, 22, 16);
    let esize = 8 << (31 - immh.leading_zeros());
    let right = 2 * esize - imm;
    let left = imm - esize;
    match opcode {
        0b10000..=0b10011 => {
            // SHRN, RSHRN, SQSHRUN, SQRSHRUN, SQSHRN, UQSHRN, SQRSHRN, UQRSHRN
            if immh & 0b1000 != 0 || (scalar && !u && opcode <= 0b10001) {
                return None;
            }
            let wide = esize * 2;
            let truncating = !u && opcode <= 0b10001;
            let (source_unsigned, dest_unsigned) = match opcode {
                0b10000 | 0b10001 => (false, u),
                _ => (u, u),
            };
            let round = if opcode & 1 == 1 {
                1i128 << (right - 1)
            } else {
                0
            };
            let n = s.v(rn);
            let lanes = if scalar { 1 } else { 64 / esize };
            let mut narrowed = 0u128;
            for i in 0..lanes {
                let x = int(lane(n, wide, i), wide, source_unsigned || truncating);
                let shifted = (x + round) >> right;
                let value = if truncating {
                    wrap(shifted, esize)
                } else {
                    sat(shifted, esize, dest_unsigned)
                };
                narrowed = put(narrowed, esize, i, value);
            }
            if scalar {
                s.set_v(rd, narrowed);
            } else {
                write_narrow(s, rd, q, narrowed);
            }
        }
        0b10100 => {
            // SSHLL, USHLL
            if scalar || immh & 0b1000 != 0 {
                return None;
            }
            let source = half(s.v(rn), q);
            let mut result = 0u128;
            for i in 0..64 / esize {
                let x = int(lane(source, esize, i), esize, u);
                result = put(result, esize * 2, i, wrap(x << left, esize * 2));
            }
            s.set_v(rd, result);
        }
        0b11100 | 0b11111 => {
            // SCVTF, UCVTF, FCVTZS, FCVTZU (fixed point)
            if esize < 32 || (!scalar && esize == 64 && !q) {
                return None;
            }
            let n = s.v(rn);
            let lanes = if scalar { 1 } else { width(q) / esize };
            let mut result = 0u128;
            for i in 0..lanes {
                let x = lane(n, esize, i);
                let value = if opcode == 0b11100 {
                    sized!(esize, int_to_fp(x, u, esize, right))
                } else {
                    sized!(esize, fp_to_int(x, Rounding::Zero, right, u, esize))
                };
                result = put(result, esize, i, value);
            }
            s.set_v(rd, result);
        }
        _ => {
            let valid = matches!(
                (u, opcode),
                (_, 0b00000 | 0b00010 | 0b00100 | 0b00110 | 0b01010 | 0b01110)
                    | (true, 0b01000 | 0b01100)
            );
            if !valid
                || (scalar && esize != 64 && !matches!(opcode, 0b01100 | 0b01110))
                || (!scalar && esize == 64 && !q)
            {
                return None;
            }
            let n = s.v(rn);
            let accumulate =
                matches!(opcode, 0b00010 | 0b00110) || (u && matches!(opcode, 0b01000 | 0b01010));
            let d = if accumulate { s.v(rd) } else { 0 };
            let lanes = if scalar { 1 } else { width(q) / esize };
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
            s.set_v(rd, result);
        }
    }
    Some(())
}

// ---------------------------------------------------------------------------
// AES

const fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut product = 0;
    while b != 0 {
        if b & 1 != 0 {
            product ^= a;
        }
        let carry = a & 0x80 != 0;
        a <<= 1;
        if carry {
            a ^= 0x1B;
        }
        b >>= 1;
    }
    product
}

/// x^254, the multiplicative inverse in GF(2^8) (0 maps to 0).
const fn gf_inverse(x: u8) -> u8 {
    let mut result = 1u8;
    let mut base = x;
    let mut exp = 254u32;
    while exp != 0 {
        if exp & 1 != 0 {
            result = gf_mul(result, base);
        }
        base = gf_mul(base, base);
        exp >>= 1;
    }
    result
}

const SBOX: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let x = gf_inverse(i as u8);
        table[i] =
            x ^ x.rotate_left(1) ^ x.rotate_left(2) ^ x.rotate_left(3) ^ x.rotate_left(4) ^ 0x63;
        i += 1;
    }
    table
};

const INV_SBOX: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        table[SBOX[i] as usize] = i as u8;
        i += 1;
    }
    table
};

/// AESE, AESD, AESMC, AESIMC. The state is column-major, byte 0 lowest.
fn aes(word: u32, s: &mut impl Regs) -> Option<()> {
    let (rd, rn, _) = regs(word);
    let opcode = bits(word, 13, 12);
    let n = s.v(rn);
    let state = if opcode < 2 { s.v(rd) ^ n } else { n }.to_le_bytes();
    let mut out = [0u8; 16];
    match opcode {
        0 | 1 => {
            let inverse = opcode == 1;
            for column in 0..4 {
                for row in 0..4 {
                    let source = (column + if inverse { 4 - row } else { row }) % 4;
                    let byte = state[row + 4 * source];
                    out[row + 4 * column] = if inverse {
                        INV_SBOX[usize::from(byte)]
                    } else {
                        SBOX[usize::from(byte)]
                    };
                }
            }
        }
        _ => {
            let coefficients: [u8; 4] = if opcode == 2 {
                [2, 3, 1, 1]
            } else {
                [14, 11, 13, 9]
            };
            for column in 0..4 {
                for row in 0..4 {
                    out[row + 4 * column] = (0..4).fold(0, |acc, k| {
                        acc ^ gf_mul(coefficients[(k + 4 - row) % 4], state[k + 4 * column])
                    });
                }
            }
        }
    }
    s.set_v(rd, u128::from_le_bytes(out));
    Some(())
}

// ---------------------------------------------------------------------------
// SHA-1 and SHA-256

fn words(v: u128) -> [u32; 4] {
    [
        v as u32,
        (v >> 32) as u32,
        (v >> 64) as u32,
        (v >> 96) as u32,
    ]
}

fn from_words(w: [u32; 4]) -> u128 {
    w.iter()
        .rev()
        .fold(0u128, |acc, word| (acc << 32) | u128::from(*word))
}

fn sha_choose(x: u32, y: u32, z: u32) -> u32 {
    ((y ^ z) & x) ^ z
}

fn sha_majority(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | ((x | y) & z)
}

/// SHA1C/SHA1P/SHA1M: four rounds on `abcd` with `e` and the schedule `wk`.
fn sha1_rounds(abcd: u128, e: u32, wk: u128, f: fn(u32, u32, u32) -> u32) -> u128 {
    let mut x = words(abcd);
    let mut y = e;
    for w in words(wk) {
        let t = f(x[1], x[2], x[3]);
        y = y
            .wrapping_add(x[0].rotate_left(5))
            .wrapping_add(t)
            .wrapping_add(w);
        x[1] = x[1].rotate_left(30);
        // <Y, X> = ROL(Y:X, 32)
        let top = x[3];
        x = [y, x[0], x[1], x[2]];
        y = top;
    }
    from_words(x)
}

/// SHA256H (`part1`) and SHA256H2: four rounds; returns X or Y.
fn sha256_rounds(x: u128, y: u128, wk: u128, part1: bool) -> u128 {
    let (mut x, mut y) = (words(x), words(y));
    for w in words(wk) {
        let chs = sha_choose(y[0], y[1], y[2]);
        let maj = sha_majority(x[0], x[1], x[2]);
        let sigma1 = y[0].rotate_right(6) ^ y[0].rotate_right(11) ^ y[0].rotate_right(25);
        let sigma0 = x[0].rotate_right(2) ^ x[0].rotate_right(13) ^ x[0].rotate_right(22);
        let t = y[3].wrapping_add(sigma1).wrapping_add(chs).wrapping_add(w);
        x[3] = t.wrapping_add(x[3]);
        y[3] = t.wrapping_add(sigma0).wrapping_add(maj);
        // <Y, X> = ROL(Y:X, 32)
        let (x_top, y_top) = (x[3], y[3]);
        x = [y_top, x[0], x[1], x[2]];
        y = [x_top, y[0], y[1], y[2]];
    }
    from_words(if part1 { x } else { y })
}

fn sha(word: u32, s: &mut impl Regs) -> Option<()> {
    let (rd, rn, rm) = regs(word);
    let result = if bit(word, 21) {
        // SHA1H, SHA1SU1, SHA256SU0
        let opcode = bits(word, 16, 12);
        if opcode > 2 {
            return None;
        }
        let n = s.v(rn);
        if opcode == 0 {
            u128::from((n as u32).rotate_left(30))
        } else {
            let d = s.v(rd);
            if opcode == 1 {
                let t = words(d ^ (n >> 32));
                from_words([
                    t[0].rotate_left(1),
                    t[1].rotate_left(1),
                    t[2].rotate_left(1),
                    t[3].rotate_left(1) ^ t[0].rotate_left(2),
                ])
            } else {
                let t = words((n << 96) | (d >> 32));
                let d = words(d);
                from_words(std::array::from_fn(|e| {
                    let x = t[e];
                    (x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)).wrapping_add(d[e])
                }))
            }
        }
    } else {
        let opcode = bits(word, 14, 12);
        if opcode == 7 {
            return None;
        }
        let d = s.v(rd);
        let n = s.v(rn);
        let m = s.v(rm);
        match opcode {
            0 => sha1_rounds(d, n as u32, m, sha_choose),
            1 => sha1_rounds(d, n as u32, m, |x, y, z| x ^ y ^ z),
            2 => sha1_rounds(d, n as u32, m, sha_majority),
            3 => ((n << 64) | (d >> 64)) ^ d ^ m,
            4 => sha256_rounds(d, n, m, true),
            5 => sha256_rounds(n, d, m, false),
            _ => {
                let sigma1 = |x: u32| x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10);
                let t0 = words((m << 96) | (n >> 32));
                let d = words(d);
                let t1 = words(m >> 64);
                let mut result = [0u32; 4];
                for e in 0..2 {
                    result[e] = sigma1(t1[e]).wrapping_add(d[e]).wrapping_add(t0[e]);
                }
                for e in 2..4 {
                    result[e] = sigma1(result[e - 2]).wrapping_add(d[e]).wrapping_add(t0[e]);
                }
                from_words(result)
            }
        }
    };
    s.set_v(rd, result);
    Some(())
}

// After the helper macros, which it uses.
pub mod fast;

#[cfg(test)]
mod estimate_tables {
    use super::*;

    #[test]
    fn recip_table_matches_the_architected_division() {
        for index in 0..256u64 {
            let a = (0x100 + index) * 2 + 1;
            let expected = ((1u64 << 19) / a).div_ceil(2) as u8;
            assert_eq!(
                RECIP_ESTIMATE[index as usize], expected,
                "scaled {index:#x}"
            );
        }
    }

    #[test]
    fn rsqrt_table_matches_the_architected_search() {
        for index in 128..512u64 {
            let a = if index < 256 {
                index * 2 + 1
            } else {
                (((index >> 1) << 1) + 1) * 2
            };
            let mut b = 512u64;
            while a * (b + 1) * (b + 1) < (1u64 << 28) {
                b += 1;
            }
            assert_eq!(
                RSQRT_ESTIMATE[index as usize],
                b.div_ceil(2) as u8,
                "scaled {index:#x}"
            );
        }
    }
}

/// The native fast paths must agree bit for bit with the exact bodies they
/// replaced, including NaN payloads, signed zeros, infinities and denormals.
#[cfg(test)]
mod fast_paths {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// Ordinary values, zeros, infinities, quiet and signalling NaNs,
    /// denormals, rounding boundaries, and random bit patterns.
    fn values32() -> Vec<u64> {
        let mut values: Vec<u64> = [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x007F_FFFF,
            0x0080_0000,
            0x8080_0000,
            0x3F00_0000,
            0xBF00_0000,
            0x3F7F_FFFF,
            0x3F80_0000,
            0xBF80_0000,
            0x3FC0_0000,
            0x4000_0000,
            0xC000_0000,
            0x4040_0000,
            0xC040_0000,
            0x4020_0000,
            0xC020_0000,
            0x477F_FFFF,
            0x4A7F_FFFF,
            0x4B00_0000,
            0x4B80_0000,
            0x4F00_0000,
            0x4F80_0000,
            0x5F00_0000,
            0x5F80_0000,
            0x7F7F_FFFF,
            0xFF7F_FFFF,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0000,
            0xFFC0_0000,
            0x7FC0_1234,
            0xFFC0_0001,
            0x7F80_0001,
            0xFF80_0001,
            0x3E99_999A,
            0xBE99_999A,
            0x3400_0000,
            0xB400_0000,
        ]
        .to_vec();
        let mut rng = Rng(0x243F_6A88_85A3_08D3);
        for _ in 0..192 {
            values.push(u64::from(rng.next() as u32));
        }
        values
    }

    fn values64() -> Vec<u64> {
        let mut values: Vec<u64> = [
            0x0000_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x0000_0000_0000_0001,
            0x000F_FFFF_FFFF_FFFF,
            0x0010_0000_0000_0000,
            0x8010_0000_0000_0000,
            0x3FE0_0000_0000_0000,
            0xBFE0_0000_0000_0000,
            0x3FEF_FFFF_FFFF_FFFF,
            0x3FF0_0000_0000_0000,
            0xBFF0_0000_0000_0000,
            0x3FF8_0000_0000_0000,
            0x4000_0000_0000_0000,
            0xC000_0000_0000_0000,
            0x4004_0000_0000_0000,
            0xC004_0000_0000_0000,
            0x4003_FFFF_FFFF_FFFF,
            0x41DF_FFFF_FFFF_FFFF,
            0x41E0_0000_0000_0000,
            0x41F0_0000_0000_0000,
            0x4330_0000_0000_0000,
            0x4340_0000_0000_0000,
            0x7FEF_FFFF_FFFF_FFFF,
            0xFFEF_FFFF_FFFF_FFFF,
            0x7FF0_0000_0000_0000,
            0xFFF0_0000_0000_0000,
            0x7FF8_0000_0000_0000,
            0xFFF8_0000_0000_0000,
            0x7FF8_0000_0000_1234,
            0xFFF8_0000_0000_0001,
            0x7FF0_0000_0000_0001,
            0xFFF0_0000_0000_0001,
            0x3FD9_9999_9999_999A,
            0xBFD9_9999_9999_999A,
        ]
        .to_vec();
        let mut rng = Rng(0x1319_8A2E_0370_7344);
        for _ in 0..192 {
            values.push(rng.next());
        }
        values
    }

    const ROUNDINGS: [Rounding; 5] = [
        Rounding::TiesEven,
        Rounding::TiesAway,
        Rounding::Up,
        Rounding::Down,
        Rounding::Zero,
    ];

    fn one_source_ops() -> Vec<Fp1> {
        let mut ops = vec![Fp1::Abs, Fp1::Neg, Fp1::Sqrt];
        ops.extend(ROUNDINGS.map(Fp1::Round));
        ops.push(Fp1::RecipEstimate);
        ops.push(Fp1::RsqrtEstimate);
        ops
    }

    const TWO_SOURCE_OPS: [Fp2; 13] = [
        Fp2::Add,
        Fp2::Sub,
        Fp2::Mul,
        Fp2::Div,
        Fp2::Max,
        Fp2::Min,
        Fp2::MaxNum,
        Fp2::MinNum,
        Fp2::MulX,
        Fp2::Abd,
        Fp2::Nmul,
        Fp2::RecipStep,
        Fp2::RsqrtStep,
    ];

    const COMPARES: [Cmp; 7] = [
        Cmp::Eq,
        Cmp::Ge,
        Cmp::Gt,
        Cmp::Le,
        Cmp::Lt,
        Cmp::AbsGe,
        Cmp::AbsGt,
    ];

    #[test]
    fn one_source_fast_path_matches_the_exact_body() {
        let ops = one_source_ops();
        for raw in values32() {
            for &op in &ops {
                assert_eq!(
                    fp1::<f32>(op, raw),
                    fp1_reference::<f32>(op, raw),
                    "f32 {op:?} {raw:#010x}"
                );
            }
        }
        for raw in values64() {
            for &op in &ops {
                assert_eq!(
                    fp1::<f64>(op, raw),
                    fp1_reference::<f64>(op, raw),
                    "f64 {op:?} {raw:#018x}"
                );
            }
        }
    }

    #[test]
    fn two_source_fast_path_matches_the_exact_body() {
        let values = values32();
        for &a in &values {
            for &b in &values {
                for &op in &TWO_SOURCE_OPS {
                    assert_eq!(
                        fp2::<f32>(op, a, b),
                        fp2_exact::<f32>(op, a, b),
                        "f32 {op:?} {a:#010x} {b:#010x}"
                    );
                }
            }
        }
        let values = values64();
        for &a in &values {
            for &b in &values {
                for &op in &TWO_SOURCE_OPS {
                    assert_eq!(
                        fp2::<f64>(op, a, b),
                        fp2_exact::<f64>(op, a, b),
                        "f64 {op:?} {a:#018x} {b:#018x}"
                    );
                }
            }
        }
    }

    #[test]
    fn fused_multiply_add_fast_path_matches_the_exact_body() {
        let mut rng = Rng(0xA409_3822_299F_31D0);
        let special32 = &values32()[..40];
        for &a in special32 {
            for &b in special32 {
                for &c in &[0, 0x8000_0000, 0x3F80_0000, 0xBF80_0000, 0x7F80_0000] {
                    assert_eq!(
                        fp_mul_add::<f32>(c, a, b),
                        fp_mul_add_exact::<f32>(c, a, b),
                        "f32 fma {a:#010x} {b:#010x} {c:#010x}"
                    );
                }
            }
        }
        let special64 = &values64()[..34];
        for &a in special64 {
            for &b in special64 {
                for &c in &[0, 0x8000_0000_0000_0000, 0x3FF0_0000_0000_0000] {
                    assert_eq!(
                        fp_mul_add::<f64>(c, a, b),
                        fp_mul_add_exact::<f64>(c, a, b),
                        "f64 fma {a:#018x} {b:#018x} {c:#018x}"
                    );
                }
            }
        }
        for _ in 0..20_000 {
            let (a, b, c) = (rng.next(), rng.next(), rng.next());
            assert_eq!(
                fp_mul_add::<f64>(c, a, b),
                fp_mul_add_exact::<f64>(c, a, b),
                "f64 random fma {a:#018x} {b:#018x} {c:#018x}"
            );
            let (a, b, c) = (
                u64::from(rng.next() as u32),
                u64::from(rng.next() as u32),
                u64::from(rng.next() as u32),
            );
            assert_eq!(
                fp_mul_add::<f32>(c, a, b),
                fp_mul_add_exact::<f32>(c, a, b),
                "f32 random fma {a:#010x} {b:#010x} {c:#010x}"
            );
        }
    }

    #[test]
    fn float_to_integer_fast_path_matches_the_exact_body() {
        let fb = [0u32, 1, 7, 8, 16, 17, 31, 32, 52, 63, 64];
        for raw in values32() {
            for &rounding in &ROUNDINGS {
                for &fbits in &fb {
                    for &unsigned in &[false, true] {
                        for &int_bits in &[32u32, 64] {
                            assert_eq!(
                                fp_to_int::<f32>(raw, rounding, fbits, unsigned, int_bits),
                                fp_to_int_reference::<f32>(
                                    raw, rounding, fbits, unsigned, int_bits
                                ),
                                "f32 {raw:#010x} {rounding:?} {fbits} {unsigned} {int_bits}"
                            );
                        }
                    }
                }
            }
        }
        for raw in values64() {
            for &rounding in &ROUNDINGS {
                for &fbits in &fb {
                    for &unsigned in &[false, true] {
                        for &int_bits in &[32u32, 64] {
                            assert_eq!(
                                fp_to_int::<f64>(raw, rounding, fbits, unsigned, int_bits),
                                fp_to_int_reference::<f64>(
                                    raw, rounding, fbits, unsigned, int_bits
                                ),
                                "f64 {raw:#018x} {rounding:?} {fbits} {unsigned} {int_bits}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn integer_to_float_fast_path_matches_the_exact_body() {
        let mut values: Vec<u64> = vec![
            0,
            1,
            2,
            3,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FFFF,
            0x1_0000_0000,
            0x0020_0000_0000_0001,
            0x7FFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000,
            0xFFFF_FFFF_FFFF_FFFF,
        ];
        let mut rng = Rng(0x082E_FA98_EC4E_6C89);
        for _ in 0..128 {
            values.push(rng.next());
        }
        for &value in &values {
            for &unsigned in &[false, true] {
                for &int_bits in &[32u32, 64] {
                    for &fbits in &[0u32, 1, 7, 16, 31, 32, 52, 63, 64] {
                        assert_eq!(
                            int_to_fp::<f32>(value, unsigned, int_bits, fbits),
                            int_to_fp_reference::<f32>(value, unsigned, int_bits, fbits),
                            "f32 {value:#x} {unsigned} {int_bits} {fbits}"
                        );
                        assert_eq!(
                            int_to_fp::<f64>(value, unsigned, int_bits, fbits),
                            int_to_fp_reference::<f64>(value, unsigned, int_bits, fbits),
                            "f64 {value:#x} {unsigned} {int_bits} {fbits}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn precision_conversion_fast_path_matches_the_exact_body() {
        for raw in values32() {
            for &(from, to) in &[(32u32, 64u32), (32, 16)] {
                assert_eq!(
                    fp_convert(raw, from, to),
                    fp_convert_exact(raw, from, to),
                    "f32 {raw:#010x} -> {to}"
                );
            }
        }
        for raw in values64() {
            for &(from, to) in &[(64u32, 32u32), (64, 16)] {
                assert_eq!(
                    fp_convert(raw, from, to),
                    fp_convert_exact(raw, from, to),
                    "f64 {raw:#018x} -> {to}"
                );
            }
        }
        let mut rng = Rng(0x4528_21E6_38D0_1377);
        let mut halves: Vec<u64> = vec![
            0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0x7C00, 0xFC00, 0x7E00,
            0xFE00, 0x7C01, 0xFC01, 0x7F80, 0xFFFF,
        ];
        for _ in 0..128 {
            halves.push(rng.next() & 0xFFFF);
        }
        for raw in halves {
            for &(from, to) in &[(16u32, 32u32), (16, 64)] {
                assert_eq!(
                    fp_convert(raw, from, to),
                    fp_convert_exact(raw, from, to),
                    "f16 {raw:#06x} -> {to}"
                );
            }
        }
    }

    #[test]
    fn compare_fast_path_matches_the_exact_body() {
        let values = values32();
        for &a in &values {
            for &b in &values {
                for &cmp in &COMPARES {
                    assert_eq!(
                        fp_cmp::<f32>(cmp, a, b),
                        fp_cmp_reference::<f32>(cmp, a, b),
                        "f32 {cmp:?} {a:#010x} {b:#010x}"
                    );
                }
            }
        }
        let values = values64();
        for &a in &values {
            for &b in &values {
                for &cmp in &COMPARES {
                    assert_eq!(
                        fp_cmp::<f64>(cmp, a, b),
                        fp_cmp_reference::<f64>(cmp, a, b),
                        "f64 {cmp:?} {a:#018x} {b:#018x}"
                    );
                }
            }
        }
    }

    /// The runtime mirrors guest FPCR.FZ into the host's MXCSR FTZ and DAZ
    /// bits, so the fast paths must stay exact with denormals flushed too.
    /// FMA is deliberately left out: the exact body's software `mul_add`
    /// need not follow MXCSR, while ARM's FZ contract is the host operation's.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fast_paths_hold_with_flush_to_zero() {
        fn get_mxcsr() -> u32 {
            let mut value = 0u32;
            // SAFETY: stmxcsr only reads the SSE control register into a
            // valid local.
            unsafe {
                std::arch::asm!("stmxcsr [{}]", in(reg) &mut value, options(nostack));
            }
            value
        }

        fn set_mxcsr(value: u32) {
            // SAFETY: ldmxcsr only writes the SSE control register, and the
            // caller restores the previous value.
            unsafe {
                std::arch::asm!("ldmxcsr [{}]", in(reg) &value, options(nostack));
            }
        }

        struct Restore(u32);
        impl Drop for Restore {
            fn drop(&mut self) {
                set_mxcsr(self.0);
            }
        }

        let saved = get_mxcsr();
        let _restore = Restore(saved);
        // FTZ (bit 15) and DAZ (bit 6).
        set_mxcsr(saved | (1 << 15) | (1 << 6));

        let values = values32();
        for raw in &values {
            for &op in &[Fp1::Sqrt, Fp1::Round(Rounding::TiesEven)] {
                assert_eq!(
                    fp1::<f32>(op, *raw),
                    fp1_reference::<f32>(op, *raw),
                    "f32 {op:?} {raw:#010x} under FZ"
                );
            }
            for &fbits in &[0u32, 17] {
                for &rounding in &ROUNDINGS {
                    assert_eq!(
                        fp_to_int::<f32>(*raw, rounding, fbits, false, 32),
                        fp_to_int_reference::<f32>(*raw, rounding, fbits, false, 32),
                        "f32 {raw:#010x} {rounding:?} {fbits} under FZ"
                    );
                }
            }
        }
        for &a in &values {
            for &b in &values {
                for &op in &[
                    Fp2::Add,
                    Fp2::Sub,
                    Fp2::Mul,
                    Fp2::Div,
                    Fp2::Max,
                    Fp2::Min,
                    Fp2::Abd,
                    Fp2::Nmul,
                ] {
                    assert_eq!(
                        fp2::<f32>(op, a, b),
                        fp2_exact::<f32>(op, a, b),
                        "f32 {op:?} {a:#010x} {b:#010x} under FZ"
                    );
                }
            }
        }
    }
}
