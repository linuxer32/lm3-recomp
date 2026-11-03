//! The stable boundary between generated guest code and the runtime.
//!
//! The chunk crates of a recompiled program depend only on this crate. The
//! runtime defines the `__recomp_rt_*` symbols declared here, and the linker
//! resolves them in the final binary, so a runtime change relinks the program
//! instead of recompiling every chunk. Keep this crate small and its API
//! stable: any change to it rebuilds all generated code.

#![cfg_attr(all(windows, recomp_fast_tls), feature(thread_local))]

use std::cell::Cell;
use std::fmt;
use std::mem::size_of;

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// An error that stops guest execution. It wraps the runtime's own error
/// type without exposing it, which keeps generated code independent of the
/// runtime crate.
pub struct Fault(Box<dyn std::error::Error + Send + Sync + 'static>);

impl Fault {
    pub fn new<E: std::error::Error + Send + Sync + 'static>(error: E) -> Self {
        Self(Box::new(error))
    }

    pub fn downcast_ref<E: std::error::Error + 'static>(&self) -> Option<&E> {
        self.0.downcast_ref()
    }
}

impl fmt::Debug for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

pub type GuestResult = Result<(), Fault>;

/// The guest register file generated code operates on: x0-x30 and sp
/// (index 31), v0-v31, and NZCV. A function generated with the guest-context
/// ABI takes a pointer to it and reads and writes the fields directly, the
/// way the Arm recompilers do, instead of copying a whole register snapshot
/// at entry. The storage is the calling thread's (see `guest_context`), so
/// separate guest threads still have separate register files.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestContext {
    /// x0-x30 and sp (index 31).
    pub x: [i64; 32],
    /// v0-v31, each in a u128.
    pub v: [u128; 32],
    /// NZCV (N 8, Z 4, C 2, V 1).
    pub nzcv: u8,
}

impl GuestContext {
    /// An all-zero context (x0-x30, sp, v0-v31 and the flags).
    pub const ZERO: Self = Self {
        x: [0; 32],
        v: [0; 32],
        nzcv: 0,
    };

    /// A general register (index 31 is sp).
    #[inline]
    pub fn gpr(&self, index: usize) -> i64 {
        self.x[index]
    }

    /// Set a general register.
    #[inline]
    pub fn set_gpr(&mut self, index: usize, value: i64) {
        self.x[index] = value;
    }

    /// A vector register.
    #[inline]
    pub fn vreg(&self, index: usize) -> u128 {
        self.v[index]
    }

    /// Set a vector register.
    #[inline]
    pub fn set_vreg(&mut self, index: usize, value: u128) {
        self.v[index] = value;
    }

    /// The NZCV flags packed as `n << 3 | z << 2 | c << 1 | v`.
    #[inline]
    pub fn nzcv(&self) -> u8 {
        self.nzcv
    }

    /// Replace the NZCV flags.
    #[inline]
    pub fn set_nzcv(&mut self, nzcv: u8) {
        self.nzcv = nzcv;
    }
}

/// Raised when execution reaches an instruction the lifter could not
/// translate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedInstruction {
    pub pc: u64,
    pub word: u32,
}

impl fmt::Display for UnsupportedInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unsupported guest instruction {:#010x} at {:#x}",
            self.word, self.pc
        )
    }
}

impl std::error::Error for UnsupportedInstruction {}

#[cold]
pub fn unsupported(pc: u64, word: u32) -> Fault {
    Fault::new(UnsupportedInstruction { pc, word })
}

#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static BLOCK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static BACKEDGES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };

#[cfg(not(all(windows, recomp_fast_tls)))]
thread_local! {
    /// Start address of the guest block this thread is running, so a fault
    /// can say where it happened.
    static BLOCK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Loop iterations since the scheduler was last consulted.
    static BACKEDGES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Loop iterations between scheduler checks.
const BACKEDGE_INTERVAL: u32 = 4096;

/// Generated code calls this on every branch back to an earlier block, so a
/// loop that makes no calls (polling memory, decompressing) still lets
/// other guest threads run, as they would on other cores.
#[inline(always)]
pub fn backedge() {
    #[cfg(all(windows, recomp_fast_tls))]
    {
        let next = BACKEDGES.get().wrapping_add(1);
        BACKEDGES.set(next);
        if next.is_multiple_of(BACKEDGE_INTERVAL) {
            loop_preempt();
        }
    }
    #[cfg(not(all(windows, recomp_fast_tls)))]
    BACKEDGES.with(|count| {
        let next = count.get().wrapping_add(1);
        count.set(next);
        if next.is_multiple_of(BACKEDGE_INTERVAL) {
            loop_preempt();
        }
    });
}

/// Generated code calls this as each block starts. Titles with
/// `[runtime] block_tracking = false` call it once at function entry
/// instead, so the value names the function rather than the block.
#[inline(always)]
pub fn enter_block(address: u64) {
    #[cfg(all(windows, recomp_fast_tls))]
    BLOCK.set(address);
    #[cfg(not(all(windows, recomp_fast_tls)))]
    BLOCK.with(|block| block.set(address));
}

/// The block the calling thread entered last.
pub fn current_block() -> u64 {
    #[cfg(all(windows, recomp_fast_tls))]
    return BLOCK.get();
    #[cfg(not(all(windows, recomp_fast_tls)))]
    BLOCK.with(|block| block.get())
}

/// Floating-point and Advanced SIMD instructions. `recomp-simd` evaluates
/// them; the runtime links it in and defines the `__recomp_simd_*` symbols
/// declared here, so changing the evaluator relinks a program instead of
/// recompiling its generated code (this crate does not depend on it).
pub mod simd {
    /// What `exec` produces: the destination's new value (a general
    /// register's in the low 64 bits) and the NZCV flags (N in bit 3).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct Outcome {
        pub value: u128,
        pub nzcv: u8,
    }

    extern "Rust" {
        fn __recomp_simd_exec(
            word: u32,
            vector: &[(u8, u128)],
            general: u64,
            nzcv: u8,
        ) -> Option<Outcome>;
    }

    /// Evaluate `word` on the registers the lifter passes (see
    /// `recomp_simd::exec`).
    pub fn exec(
        word: u32,
        vector: &[(u8, u128)],
        general: u64,
        nzcv: u8,
    ) -> Result<Outcome, crate::Fault> {
        // SAFETY: the runtime defines this symbol with this signature.
        unsafe { __recomp_simd_exec(word, vector, general, nzcv) }
            .ok_or_else(|| crate::unsupported(0, word))
    }

    /// Instructions decoded when the code was generated (see
    /// `recomp_simd::fast`): generated code calls these directly.
    pub mod fast {
        macro_rules! linked {
            ($(fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty = $symbol:ident;)*) => {
                extern "Rust" {
                    $(fn $symbol($($arg: $ty),*) -> $ret;)*
                }
                $(
                    #[inline]
                    #[allow(clippy::too_many_arguments)]
                    pub fn $name($($arg: $ty),*) -> $ret {
                        // SAFETY: the runtime defines every symbol with
                        // exactly this signature.
                        unsafe { $symbol($($arg),*) }
                    }
                )*
            };
        }

        /// The evaluator's versions of the operations inlined below, for
        /// what the inline paths leave to it (NaN results, other forms).
        pub mod exact {
            linked! {
                fn scalar_fp2(op: u32, double: bool, n: u128, m: u128) -> u128 = __recomp_simd_scalar_fp2;
                fn scalar_fma(negate: u32, double: bool, a: u128, n: u128, m: u128) -> u128 = __recomp_simd_scalar_fma;
                fn vector3(kind: u32, double: bool, q: bool, d: u128, n: u128, m: u128) -> u128 = __recomp_simd_vector3;
                fn vector_element(op: u32, double: bool, q: bool, scalar: bool, index: u32, d: u128, n: u128, m: u128) -> u128 = __recomp_simd_vector_element;
                fn scalar_fp1(opcode: u32, ftype: u32, n: u128) -> u128 = __recomp_simd_scalar_fp1;
            }
        }

        // The other specializations, decoded when the code is generated:
        // generated code calls the evaluator's exact implementations directly
        // instead of decoding and dispatching the instruction on every run.
        linked! {
            fn int_three(u: bool, opcode: u32, esize: u32, lanes: u32, d: u128, n: u128, m: u128) -> u128 = __recomp_simd_int_three;
            fn vector_bitwise(kind: u32, q: bool, d: u128, n: u128, m: u128) -> u128 = __recomp_simd_vector_bitwise;
            fn vector_rev(container: u32, esize: u32, q: bool, n: u128) -> u128 = __recomp_simd_vector_rev;
            fn vector_pairwise_sum(u: bool, accumulate: bool, esize: u32, lanes: u32, d: u128, n: u128) -> u128 = __recomp_simd_vector_pairwise_sum;
            fn vector_int2(kind: u32, esize: u32, lanes: u32, d: u128, n: u128) -> u128 = __recomp_simd_vector_int2;
            fn vector_narrow(kind: u32, esize: u32, mode: u32, lanes: u32, n: u128, d: u128) -> u128 = __recomp_simd_vector_narrow;
            fn vector_shift_long(q: bool, esize: u32, n: u128) -> u128 = __recomp_simd_vector_shift_long;
            fn vector_fp1(op: u32, esize: u32, lanes: u32, n: u128) -> u128 = __recomp_simd_vector_fp1;
            fn vector_fp_cmp0(cmp: u32, esize: u32, lanes: u32, n: u128) -> u128 = __recomp_simd_vector_fp_cmp0;
            fn vector_to_int(rounding: u32, unsigned: bool, esize: u32, lanes: u32, n: u128) -> u128 = __recomp_simd_vector_to_int;
            fn vector_from_int(unsigned: bool, esize: u32, lanes: u32, n: u128) -> u128 = __recomp_simd_vector_from_int;
            fn vector_across(kind: u32, esize: u32, lanes: u32, n: u128) -> u128 = __recomp_simd_vector_across;
            fn vector_fixed_convert(to_float: bool, unsigned: bool, esize: u32, lanes: u32, fbits: u32, n: u128) -> u128 = __recomp_simd_vector_fixed_convert;
            fn vector_shift(u: bool, opcode: u32, esize: u32, lanes: u32, imm: u32, d: u128, n: u128) -> u128 = __recomp_simd_vector_shift;
            fn scalar_fccmp(double: bool, cond: u32, nzcv: u8, n: u128, m: u128, immediate: u32) -> u8 = __recomp_simd_scalar_fccmp;
        }

        // Inline paths for the hottest operations (LM3's job threads spent
        // a third of their time calling the evaluator for them). Generated
        // code passes the decoded fields as constants, so each call folds
        // to a few host instructions. Like the evaluator's own fast paths,
        // they compute every lane on the host, which for a result that is
        // not NaN is exactly what Arm does (round to nearest; flush to zero
        // follows FPCR.FZ through the host's MXCSR), and leave NaN results
        // to the evaluator's Arm NaN rules. `mul_add` is one fused
        // instruction when the program is built with FMA
        // (`target-cpu=native`), a libm call otherwise.

        /// The single-precision lanes of a vector register.
        #[inline(always)]
        fn lanes(v: u128) -> [f32; 4] {
            [0, 1, 2, 3].map(|i| f32::from_bits((v >> (32 * i)) as u32))
        }

        /// Whether a `count`-lane single-precision value has a NaN lane.
        ///
        /// A lane is NaN when its exponent is all ones and its mantissa is
        /// non-zero, that is when `bits & 0x7fff_ffff > 0x7f80_0000`. With
        /// the sign cleared, adding `0x007f_ffff` sets bit 31 of exactly the
        /// NaN lanes (and cannot carry into the next lane: a cleared lane is
        /// at most `0x7fff_ffff`, so the sum is below `2^32`). One branch
        /// covers all four lanes, which keeps the check out of the per-lane
        /// path the portable code used to take.
        #[inline(always)]
        fn any_nan_bits(bits: u128) -> bool {
            const ABS: u128 = 0x7FFF_FFFF_7FFF_FFFF_7FFF_FFFF_7FFF_FFFF;
            const NAN_CARRY: u128 = 0x007F_FFFF_007F_FFFF_007F_FFFF_007F_FFFF;
            const SIGN: u128 = 0x8000_0000_8000_0000_8000_0000_8000_0000;
            (bits & ABS).wrapping_add(NAN_CARRY) & SIGN != 0
        }

        /// The first `count` lanes packed (the rest zero), or `None` when
        /// one is NaN. All lanes are checked together, after packing.
        #[inline(always)]
        fn pack(count: usize, values: [f32; 4]) -> Option<u128> {
            let mut result = 0u128;
            for (i, value) in values.iter().enumerate().take(count) {
                result |= u128::from(value.to_bits()) << (32 * i);
            }
            (!any_nan_bits(result)).then_some(result)
        }

        #[inline(always)]
        fn fma4(a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
            [0, 1, 2, 3].map(|i| a[i].mul_add(b[i], c[i]))
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

        /// FMAX/FMAXNM of numbers (not NaN): equal values differ at most in
        /// the sign of zero, and +0 wins.
        #[inline(always)]
        fn max_f32(x: f32, y: f32) -> f32 {
            if x > y {
                x
            } else if y > x {
                y
            } else {
                f32::from_bits(x.to_bits() & y.to_bits())
            }
        }

        /// FMIN/FMINNM of numbers: -0 wins.
        #[inline(always)]
        fn min_f32(x: f32, y: f32) -> f32 {
            if x < y {
                x
            } else if y < x {
                y
            } else {
                f32::from_bits(x.to_bits() | y.to_bits())
            }
        }

        #[inline(always)]
        fn max_f64(x: f64, y: f64) -> f64 {
            if x > y {
                x
            } else if y > x {
                y
            } else {
                f64::from_bits(x.to_bits() & y.to_bits())
            }
        }

        #[inline(always)]
        fn min_f64(x: f64, y: f64) -> f64 {
            if x < y {
                x
            } else if y < x {
                y
            } else {
                f64::from_bits(x.to_bits() | y.to_bits())
            }
        }

        /// Whether any of the first `count` lanes is NaN (the evaluator's
        /// NaN rules apply then). One word-wide check: see [`any_nan_bits`].
        #[inline(always)]
        fn any_nan(count: usize, values: [f32; 4]) -> bool {
            let mut bits = 0u128;
            for (i, value) in values.iter().enumerate().take(count) {
                bits |= u128::from(value.to_bits()) << (32 * i);
            }
            any_nan_bits(bits)
        }

        /// FMUL, FDIV, FADD, FSUB, FMAX, FMIN, FMAXNM, FMINNM, FNMUL
        /// (scalar; `op` is bits 15:12).
        #[inline(always)]
        pub fn scalar_fp2(op: u32, double: bool, n: u128, m: u128) -> u128 {
            let (a, b) = (f64::from_bits(n as u64), f64::from_bits(m as u64));
            let (x, y) = (f32::from_bits(n as u32), f32::from_bits(m as u32));
            let numbers = if double {
                !a.is_nan() && !b.is_nan()
            } else {
                !x.is_nan() && !y.is_nan()
            };
            let host = match op {
                0 => scalar_bits(double, a * b, x * y),
                1 => scalar_bits(double, a / b, x / y),
                2 => scalar_bits(double, a + b, x + y),
                3 => scalar_bits(double, a - b, x - y),
                4 | 6 if numbers => scalar_bits(double, max_f64(a, b), max_f32(x, y)),
                5 | 7 if numbers => scalar_bits(double, min_f64(a, b), min_f32(x, y)),
                8 => scalar_bits(double, -(a * b), -(x * y)),
                _ => None,
            };
            host.unwrap_or_else(|| exact::scalar_fp2(op, double, n, m))
        }

        /// FMADD, FMSUB, FNMADD, FNMSUB: `negate` bit 0 negates n, bit 1
        /// negates a (bits 21 and 15 of the instruction).
        #[inline(always)]
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
                let value = sign(negate_n, n).mul_add(f64::from_bits(m as u64), sign(negate_a, a));
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
                let value = sign(negate_n, n).mul_add(f32::from_bits(m as u32), sign(negate_a, a));
                scalar_bits(false, 0.0, value)
            };
            host.unwrap_or_else(|| exact::scalar_fma(negate, double, a, n, m))
        }

        /// The single-precision vector operations and byte moves on the
        /// host's 128-bit vectors (SSE in AVX encodings, with FMA; what
        /// `target-cpu=native` gives on any recent x86 CPU): a register's
        /// lanes go through each instruction together and stay in a vector
        /// register between instructions. Lane by lane, LM3's job code moved
        /// every lane between general and vector registers and back, about
        /// 30 instructions for one FMLA by element. The results are the
        /// portable versions' bit for bit; NaN results still go to the
        /// evaluator.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        mod host {
            use core::arch::x86_64::*;

            // SAFETY (every intrinsic here): SSE through SSE4.1, AVX and FMA
            // are enabled at compile time (the module's cfg).

            #[inline(always)]
            fn ps(v: u128) -> __m128 {
                // SAFETY: both are 16 plain bytes.
                unsafe { core::mem::transmute::<u128, __m128>(v) }
            }

            #[inline(always)]
            fn bits(v: __m128) -> u128 {
                // SAFETY: as in `ps`.
                unsafe { core::mem::transmute::<__m128, u128>(v) }
            }

            /// The first `count` lanes, the rest zero.
            #[inline(always)]
            fn low(count: usize, v: __m128) -> __m128 {
                unsafe {
                    match count {
                        1 => _mm_move_ss(_mm_setzero_ps(), v),
                        2 => _mm_movelh_ps(v, _mm_setzero_ps()),
                        _ => v,
                    }
                }
            }

            /// The first `count` lanes, or `None` when one is NaN (the
            /// caller then takes the evaluator's Arm NaN rules). The compare
            /// and test see all `count` lanes at once. Four lanes use
            /// `vptest` on the compare's all-ones mask (no general-register
            /// transfer); the narrower forms keep the `movemask` test, which
            /// lowers better than a masked `ptest` for one or two lanes.
            #[inline(always)]
            fn checked(count: usize, v: __m128) -> Option<__m128> {
                // SAFETY: SSE4.1 is implied by the AVX the module needs.
                let nan = unsafe { _mm_cmpunord_ps(v, v) };
                let any = if count == 4 {
                    unsafe { _mm_testz_si128(_mm_castps_si128(nan), _mm_castps_si128(nan)) == 0 }
                } else {
                    // SAFETY: SSE2, part of the x86-64 baseline.
                    unsafe { _mm_movemask_ps(nan) as u32 & ((1u32 << count) - 1) != 0 }
                };
                if any {
                    None
                } else {
                    Some(low(count, v))
                }
            }

            /// Store a whole vector register, one 16-byte host store. The
            /// generated code's destination is a `u128` in the guest context;
            /// returning the bits instead made the compiler split the value
            /// into two general registers and two stores at every use.
            #[inline(always)]
            unsafe fn store(out: *mut u128, v: __m128) {
                // SAFETY: the caller passes a valid, aligned-enough `u128`.
                unsafe { _mm_storeu_si128(out.cast::<__m128i>(), _mm_castps_si128(v)) }
            }

            // The evaluator's results, with vectors in and out: with an
            // integer result (or integer operands) to merge with, the fast
            // path's vectors went to pairs of general registers and back
            // around every operation.

            #[cold]
            #[inline(never)]
            fn vector3_exact(kind: u32, q: bool, d: __m128, n: __m128, m: __m128) -> __m128 {
                ps(super::exact::vector3(
                    kind,
                    false,
                    q,
                    bits(d),
                    bits(n),
                    bits(m),
                ))
            }

            #[cold]
            #[inline(never)]
            #[allow(clippy::too_many_arguments)]
            fn vector_element_exact(
                op: u32,
                q: bool,
                scalar: bool,
                index: u32,
                d: __m128,
                n: __m128,
                m: __m128,
            ) -> __m128 {
                ps(super::exact::vector_element(
                    op,
                    false,
                    q,
                    scalar,
                    index,
                    bits(d),
                    bits(n),
                    bits(m),
                ))
            }

            /// Whether a lane of either is NaN among the first `count`.
            #[inline(always)]
            fn any_nan(count: usize, a: __m128, b: __m128) -> bool {
                let nan = unsafe { _mm_cmpunord_ps(a, b) };
                if count == 4 {
                    unsafe { _mm_testz_si128(_mm_castps_si128(nan), _mm_castps_si128(nan)) == 0 }
                } else {
                    unsafe { _mm_movemask_ps(nan) as u32 & ((1u32 << count) - 1) != 0 }
                }
            }

            #[inline(always)]
            fn abs(v: __m128) -> __m128 {
                unsafe { _mm_and_ps(v, _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF))) }
            }

            /// FMAX(NM) of numbers: x86 returns the second of equal operands,
            /// Arm +0 of +0 and -0.
            #[inline(always)]
            fn max(x: __m128, y: __m128) -> __m128 {
                unsafe { _mm_blendv_ps(_mm_max_ps(x, y), _mm_and_ps(x, y), _mm_cmpeq_ps(x, y)) }
            }

            /// FMIN(NM) of numbers: -0 of +0 and -0.
            #[inline(always)]
            fn min(x: __m128, y: __m128) -> __m128 {
                unsafe { _mm_blendv_ps(_mm_min_ps(x, y), _mm_or_ps(x, y), _mm_cmpeq_ps(x, y)) }
            }

            /// The first and second elements of the pairs of the
            /// concatenation n:m (`count` lanes each).
            #[inline(always)]
            fn pairs(count: usize, a: __m128, b: __m128) -> (__m128, __m128) {
                unsafe {
                    if count == 4 {
                        (
                            _mm_shuffle_ps::<0b10_00_10_00>(a, b),
                            _mm_shuffle_ps::<0b11_01_11_01>(a, b),
                        )
                    } else {
                        let joined = _mm_unpacklo_ps(a, b);
                        (joined, _mm_movehl_ps(joined, joined))
                    }
                }
            }

            /// The host result of a single-precision `vector3`, or `None`
            /// when a lane is NaN (or the operation is not one of the host's
            /// forms). `None` means the evaluator owns the Arm NaN rules.
            #[inline(always)]
            fn vector3_hot(kind: u32, q: bool, a: __m128, b: __m128, d: __m128) -> Option<__m128> {
                let count = if q { 4 } else { 2 };
                let (u, high, opcode) = (kind >> 6 == 1, (kind >> 5) & 1, kind & 0x1F);
                let (first, second) = pairs(count, a, b);
                unsafe {
                    match (u, high, opcode) {
                        // A NaN compares false.
                        (false, 0, 0b11100) => Some(low(count, _mm_cmpeq_ps(a, b))),
                        (true, 0, 0b11100) => Some(low(count, _mm_cmpge_ps(a, b))),
                        (true, 1, 0b11100) => Some(low(count, _mm_cmpgt_ps(a, b))),
                        (true, 0, 0b11101) => Some(low(count, _mm_cmpge_ps(abs(a), abs(b)))),
                        (true, 1, 0b11101) => Some(low(count, _mm_cmpgt_ps(abs(a), abs(b)))),
                        // Max and min pick an operand: NaNs go to the evaluator.
                        (_, _, 0b11000 | 0b11110) if any_nan(count, a, b) => None,
                        (false, 0, 0b11000 | 0b11110) => Some(low(count, max(a, b))),
                        (false, 1, 0b11000 | 0b11110) => Some(low(count, min(a, b))),
                        (true, 0, 0b11000 | 0b11110) => Some(low(count, max(first, second))),
                        (true, 1, 0b11000 | 0b11110) => Some(low(count, min(first, second))),
                        // FRECPS, FRSQRTS: -(a * b) + c is (-a) * b + c, fused.
                        (false, 0, 0b11111) => {
                            checked(count, _mm_fnmadd_ps(a, b, _mm_set1_ps(2.0)))
                        }
                        (false, 1, 0b11111) => checked(
                            count,
                            _mm_fnmadd_ps(a, _mm_mul_ps(b, _mm_set1_ps(0.5)), _mm_set1_ps(1.5)),
                        ),
                        (false, 0, 0b11010) => checked(count, _mm_add_ps(a, b)),
                        (false, 1, 0b11010) => checked(count, _mm_sub_ps(a, b)),
                        (_, 0, 0b11011) => checked(count, _mm_mul_ps(a, b)),
                        (true, 0, 0b11111) => checked(count, _mm_div_ps(a, b)),
                        (true, 1, 0b11010) => checked(count, abs(_mm_sub_ps(a, b))),
                        (false, 0, 0b11001) => checked(count, _mm_fmadd_ps(a, b, d)),
                        (false, 1, 0b11001) => checked(count, _mm_fnmadd_ps(a, b, d)),
                        (true, 0, 0b11010) => checked(count, _mm_add_ps(first, second)),
                        _ => None,
                    }
                }
            }

            /// `vector3` for single precision.
            #[inline(always)]
            pub fn vector3(kind: u32, q: bool, d: u128, n: u128, m: u128) -> u128 {
                let (d, a, b) = (ps(d), ps(n), ps(m));
                match vector3_hot(kind, q, a, b, d) {
                    Some(value) => bits(value),
                    None => bits(vector3_exact(kind, q, d, a, b)),
                }
            }

            /// `vector3`, writing the whole register to `out` in one store.
            ///
            /// # Safety
            /// `out` must be a valid, writable `u128`.
            #[inline(always)]
            pub unsafe fn vector3_into(
                out: *mut u128,
                kind: u32,
                q: bool,
                d: u128,
                n: u128,
                m: u128,
            ) {
                let (d, a, b) = (ps(d), ps(n), ps(m));
                let value = match vector3_hot(kind, q, a, b, d) {
                    Some(value) => value,
                    None => vector3_exact(kind, q, d, a, b),
                };
                // SAFETY: the caller passes a valid destination.
                unsafe { store(out, value) }
            }

            /// The host result of a single-precision by-element operation,
            /// or `None` when a lane is NaN.
            #[inline(always)]
            fn element_hot(
                op: u32,
                q: bool,
                scalar: bool,
                index: u32,
                d: __m128,
                n: __m128,
                m: __m128,
            ) -> Option<__m128> {
                let count = if scalar {
                    1
                } else if q {
                    4
                } else {
                    2
                };
                unsafe {
                    let element = _mm_permutevar_ps(m, _mm_set1_epi32((index & 3) as i32));
                    let value = match op {
                        0 => _mm_fmadd_ps(n, element, d),
                        1 => _mm_fnmadd_ps(n, element, d),
                        // FMULX differs from FMUL only for 0 * infinity, NaN here.
                        _ => _mm_mul_ps(n, element),
                    };
                    checked(count, value)
                }
            }

            /// `vector_element` for single precision.
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            pub fn vector_element(
                op: u32,
                q: bool,
                scalar: bool,
                index: u32,
                d: u128,
                n: u128,
                m: u128,
            ) -> u128 {
                let (d, n, m) = (ps(d), ps(n), ps(m));
                match element_hot(op, q, scalar, index, d, n, m) {
                    Some(value) => bits(value),
                    None => bits(vector_element_exact(op, q, scalar, index, d, n, m)),
                }
            }

            /// `vector_element`, writing the whole register to `out` in one
            /// store.
            ///
            /// # Safety
            /// `out` must be a valid, writable `u128`.
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn vector_element_into(
                out: *mut u128,
                op: u32,
                q: bool,
                scalar: bool,
                index: u32,
                d: u128,
                n: u128,
                m: u128,
            ) {
                let (d, n, m) = (ps(d), ps(n), ps(m));
                let value = match element_hot(op, q, scalar, index, d, n, m) {
                    Some(value) => value,
                    None => vector_element_exact(op, q, scalar, index, d, n, m),
                };
                // SAFETY: the caller passes a valid destination.
                unsafe { store(out, value) }
            }

            /// pshufb masks moving byte `source[i]` of n:m (n's below 16, m's
            /// from 16, none for -1) to byte `i`: one for n, one for m.
            const fn masks(source: [i8; 16]) -> [[u8; 16]; 2] {
                let mut masks = [[0x80u8; 16]; 2];
                let mut i = 0;
                while i < 16 {
                    let j = source[i];
                    if j >= 16 {
                        masks[1][i] = (j - 16) as u8;
                    } else if j >= 0 {
                        masks[0][i] = j as u8;
                    }
                    i += 1;
                }
                masks
            }

            /// EXT: bytes `position..` of n:m (of the joined low halves for a
            /// 64-bit vector).
            const fn ext_source(q: bool, position: usize) -> [i8; 16] {
                let mut source = [-1i8; 16];
                let mut i = 0;
                while i < 16 {
                    let j = i + position;
                    if q {
                        source[i] = j as i8;
                    } else if i < 8 {
                        source[i] = if j < 8 { j } else { 16 + j - 8 } as i8;
                    }
                    i += 1;
                }
                source
            }

            /// DUP (element), scalar: lane `index` of `bytes` bytes.
            const fn element_source(bytes: usize, index: usize) -> [i8; 16] {
                let mut source = [-1i8; 16];
                let mut i = 0;
                while i < bytes {
                    source[i] = (index * bytes + i) as i8;
                    i += 1;
                }
                source
            }

            /// ZIP, UZP, TRN (`op` as in `fast::permute`) on `bytes`-byte lanes.
            const fn permute_source(op: usize, bytes: usize, q: bool) -> [i8; 16] {
                let mut source = [-1i8; 16];
                let count = if q { 16 } else { 8 } / bytes;
                let second = if op >= 5 { 1 } else { 0 };
                let mut byte = 0;
                while byte < count * bytes {
                    let (i, offset) = (byte / bytes, byte % bytes);
                    let (pair, odd) = (i / 2, i % 2 == 1);
                    let (from_m, lane) = match op & 3 {
                        // UZP: the even (odd) elements of n:m.
                        1 => {
                            let j = 2 * i + second;
                            if j < count {
                                (false, j)
                            } else {
                                (true, j - count)
                            }
                        }
                        // TRN: element 2k (2k + 1) of n, then of m.
                        2 => (odd, 2 * pair + second),
                        // ZIP: the low (high) halves interleaved.
                        _ => (odd, pair + second * count / 2),
                    };
                    source[byte] = ((if from_m { 16 } else { 0 }) + lane * bytes + offset) as i8;
                    byte += 1;
                }
                source
            }

            /// Masks by [q][position].
            const EXT: [[[[u8; 16]; 2]; 16]; 2] = {
                let mut table = [[[[0; 16]; 2]; 16]; 2];
                let mut position = 0;
                while position < 16 {
                    table[0][position] = masks(ext_source(false, position));
                    table[1][position] = masks(ext_source(true, position));
                    position += 1;
                }
                table
            };

            /// Masks by [log2 of the lane bytes][index].
            const ELEMENT: [[[[u8; 16]; 2]; 16]; 4] = {
                let mut table = [[[[0; 16]; 2]; 16]; 4];
                let mut size = 0;
                while size < 4 {
                    let bytes = 1 << size;
                    let mut index = 0;
                    while index < 16 / bytes {
                        table[size][index] = masks(element_source(bytes, index));
                        index += 1;
                    }
                    size += 1;
                }
                table
            };

            /// Masks by [q][op][log2 of the lane bytes].
            const PERMUTE: [[[[[u8; 16]; 2]; 4]; 8]; 2] = {
                let mut table = [[[[[0; 16]; 2]; 4]; 8]; 2];
                let mut op = 1;
                while op < 8 {
                    let mut size = 0;
                    while size < 4 {
                        if op & 3 != 0 {
                            table[0][op][size] = masks(permute_source(op, 1 << size, false));
                            table[1][op][size] = masks(permute_source(op, 1 << size, true));
                        }
                        size += 1;
                    }
                    op += 1;
                }
                table
            };

            /// n:m through a pair of masks. The tables are `const`, so each
            /// crate of generated code has them, and the constant fields it
            /// passes fold their loads to constant masks (a mask that takes
            /// nothing to a zero). A `static`'s contents are not visible
            /// across crates, which left a table load at every use.
            #[inline(always)]
            fn shuffle(n: u128, m: u128, masks: &[[u8; 16]; 2]) -> u128 {
                let take = |v: u128, mask: &[u8; 16]| unsafe {
                    _mm_shuffle_epi8(
                        core::mem::transmute::<u128, __m128i>(v),
                        _mm_loadu_si128(mask.as_ptr().cast()),
                    )
                };
                let picked = unsafe { _mm_or_si128(take(n, &masks[0]), take(m, &masks[1])) };
                // SAFETY: 16 plain bytes.
                unsafe { core::mem::transmute::<__m128i, u128>(picked) }
            }

            /// The table row of `esize`-bit lanes.
            #[inline(always)]
            fn size_row(esize: u32) -> usize {
                (esize / 8).trailing_zeros() as usize
            }

            #[inline(always)]
            pub fn extract(q: bool, position: u32, n: u128, m: u128) -> u128 {
                shuffle(n, m, &EXT[usize::from(q)][position as usize])
            }

            #[inline(always)]
            pub fn element(esize: u32, index: u32, n: u128) -> u128 {
                shuffle(n, 0, &ELEMENT[size_row(esize)][index as usize])
            }

            #[inline(always)]
            pub fn permute(op: u32, esize: u32, q: bool, n: u128, m: u128) -> u128 {
                shuffle(n, m, &PERMUTE[usize::from(q)][op as usize][size_row(esize)])
            }
        }

        /// All-ones lanes where `holds` (a NaN compares false), as FCMEQ,
        /// FCMGE, FCMGT, FACGE and FACGT write them.
        #[inline(always)]
        fn compare4(count: usize, a: [f32; 4], b: [f32; 4], holds: fn(f32, f32) -> bool) -> u128 {
            let mut result = 0u128;
            for i in 0..count {
                if holds(a[i], b[i]) {
                    result |= u128::from(u32::MAX) << (32 * i);
                }
            }
            result
        }

        /// A three-same floating-point vector operation (`kind` is U << 6
        /// | size<1> << 5 | opcode): FADD, FSUB, FMUL, FMULX, FDIV, FABD,
        /// FMLA, FMLS, FADDP, FMAX(NM)(P), FMIN(NM)(P), FRECPS, FRSQRTS
        /// and the compares inline for single precision.
        #[inline(always)]
        pub fn vector3(kind: u32, double: bool, q: bool, d: u128, n: u128, m: u128) -> u128 {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            if !double {
                return host::vector3(kind, q, d, n, m);
            }
            if !double {
                let count = if q { 4 } else { 2 };
                let (a, b) = (lanes(n), lanes(m));
                let (u, high, opcode) = (kind >> 6 == 1, (kind >> 5) & 1, kind & 0x1F);
                let compared = match (u, high, opcode) {
                    (false, 0, 0b11100) => Some(compare4(count, a, b, |x, y| x == y)),
                    (true, 0, 0b11100) => Some(compare4(count, a, b, |x, y| x >= y)),
                    (true, 1, 0b11100) => Some(compare4(count, a, b, |x, y| x > y)),
                    (true, 0, 0b11101) => Some(compare4(count, a, b, |x, y| x.abs() >= y.abs())),
                    (true, 1, 0b11101) => Some(compare4(count, a, b, |x, y| x.abs() > y.abs())),
                    _ => None,
                };
                if let Some(result) = compared {
                    return result;
                }
                // Max and min pick an operand: NaNs go to the evaluator.
                let numbers = !any_nan(count, a) && !any_nan(count, b);
                // Pairs of the concatenation n:m.
                let element = |j: usize| if j < count { a[j] } else { b[j - count] };
                let pairs = |op: fn(f32, f32) -> f32| {
                    [0, 1, 2, 3].map(|i| {
                        if i < count {
                            op(element(2 * i), element(2 * i + 1))
                        } else {
                            0.0
                        }
                    })
                };
                let values = match (u, high, opcode) {
                    (false, 0, 0b11000 | 0b11110) if numbers => {
                        Some([0, 1, 2, 3].map(|i| max_f32(a[i], b[i])))
                    }
                    (false, 1, 0b11000 | 0b11110) if numbers => {
                        Some([0, 1, 2, 3].map(|i| min_f32(a[i], b[i])))
                    }
                    (true, 0, 0b11000 | 0b11110) if numbers => Some(pairs(max_f32)),
                    (true, 1, 0b11000 | 0b11110) if numbers => Some(pairs(min_f32)),
                    // FRECPS, FRSQRTS: infinity times zero is NaN here, and
                    // the evaluator's special case.
                    (false, 0, 0b11111) => Some([0, 1, 2, 3].map(|i| (-a[i]).mul_add(b[i], 2.0))),
                    (false, 1, 0b11111) => {
                        Some([0, 1, 2, 3].map(|i| (-a[i]).mul_add(b[i] * 0.5, 1.5)))
                    }
                    _ => None,
                };
                if let Some(result) = values.and_then(|values| pack(count, values)) {
                    return result;
                }
                let values = match (u, high, opcode) {
                    (false, 0, 0b11010) => Some([0, 1, 2, 3].map(|i| a[i] + b[i])),
                    (false, 1, 0b11010) => Some([0, 1, 2, 3].map(|i| a[i] - b[i])),
                    // FMULX differs from FMUL only for 0 * infinity, NaN here.
                    (false, 0, 0b11011) | (true, 0, 0b11011) => {
                        Some([0, 1, 2, 3].map(|i| a[i] * b[i]))
                    }
                    (true, 0, 0b11111) => Some([0, 1, 2, 3].map(|i| a[i] / b[i])),
                    (true, 1, 0b11010) => Some([0, 1, 2, 3].map(|i| (a[i] - b[i]).abs())),
                    (false, 0, 0b11001) => Some(fma4(a, b, lanes(d))),
                    (false, 1, 0b11001) => Some(fma4(a.map(|x| -x), b, lanes(d))),
                    (true, 0, 0b11010) => {
                        // Pairs of the concatenation n:m.
                        let element = |j: usize| if j < count { a[j] } else { b[j - count] };
                        Some([0, 1, 2, 3].map(|i| {
                            if i < count {
                                element(2 * i) + element(2 * i + 1)
                            } else {
                                0.0
                            }
                        }))
                    }
                    _ => None,
                };
                if let Some(result) = values.and_then(|values| pack(count, values)) {
                    return result;
                }
            }
            exact::vector3(kind, double, q, d, n, m)
        }

        /// [`vector3`], writing the result register through `out` instead of
        /// returning its bits. Generated code stores every vector operation
        /// into a guest vector register, so this lets the host keep the
        /// result in one vector register and store it with one 16-byte
        /// access: the `u128` return had to be split into two general
        /// registers and two stores at each use.
        ///
        /// # Safety
        /// `out` must be a valid, writable `u128`.
        #[inline(always)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn vector3_into(
            out: *mut u128,
            kind: u32,
            double: bool,
            q: bool,
            d: u128,
            n: u128,
            m: u128,
        ) {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            if !double {
                // SAFETY: forwarded from the caller.
                return unsafe { host::vector3_into(out, kind, q, d, n, m) };
            }
            // SAFETY: forwarded from the caller.
            unsafe { out.write(vector3(kind, double, q, d, n, m)) }
        }

        /// FMLA, FMLS, FMUL, FMULX by element (`op` 0..=3), vector or
        /// scalar.
        #[inline(always)]
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
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            if !double {
                return host::vector_element(op, q, scalar, index, d, n, m);
            }
            if !double {
                let count = if scalar {
                    1
                } else if q {
                    4
                } else {
                    2
                };
                let element = [f32::from_bits((m >> (32 * (index & 3))) as u32); 4];
                let a = lanes(n);
                let values = match op {
                    0 => fma4(a, element, lanes(d)),
                    1 => fma4(a.map(|x| -x), element, lanes(d)),
                    // FMULX differs from FMUL only for 0 * infinity, NaN here.
                    _ => [0, 1, 2, 3].map(|i| a[i] * element[i]),
                };
                if let Some(result) = pack(count, values) {
                    return result;
                }
            }
            exact::vector_element(op, double, q, scalar, index, d, n, m)
        }

        /// [`vector_element`], writing the result register through `out`.
        ///
        /// # Safety
        /// `out` must be a valid, writable `u128`.
        #[inline(always)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn vector_element_into(
            out: *mut u128,

            op: u32,
            double: bool,
            q: bool,
            scalar: bool,
            index: u32,
            d: u128,
            n: u128,
            m: u128,
        ) {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            if !double {
                // SAFETY: forwarded from the caller.
                return unsafe { host::vector_element_into(out, op, q, scalar, index, d, n, m) };
            }
            // SAFETY: forwarded from the caller.
            unsafe { out.write(vector_element(op, double, q, scalar, index, d, n, m)) }
        }

        // The rest compute exactly what the evaluator does with the same
        // host operations (comparisons, conversions, bit moves), so they
        // need no fallback; LM3's gameplay called the evaluator 18 million
        // times a second for them before they were inline.

        /// The low `width` bits.
        #[inline(always)]
        fn bits(width: u32, value: u128) -> u128 {
            if width >= 128 {
                value
            } else {
                value & ((1u128 << width) - 1)
            }
        }

        /// FCMP, FCMPE: NZCV (unordered 0011, equal 0110, less 1000,
        /// greater 0010).
        #[inline(always)]
        pub fn scalar_compare(double: bool, n: u128, m: u128) -> u8 {
            let order = if double {
                f64::from_bits(n as u64).partial_cmp(&f64::from_bits(m as u64))
            } else {
                f32::from_bits(n as u32).partial_cmp(&f32::from_bits(m as u32))
            };
            match order {
                None => 0b0011,
                Some(core::cmp::Ordering::Equal) => 0b0110,
                Some(core::cmp::Ordering::Less) => 0b1000,
                Some(core::cmp::Ordering::Greater) => 0b0010,
            }
        }

        /// Whether condition `cond` holds for `nzcv` (N in bit 3).
        #[inline(always)]
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

        /// FCSEL.
        #[inline(always)]
        pub fn scalar_select(double: bool, cond: u32, nzcv: u8, n: u128, m: u128) -> u128 {
            let chosen = if condition_holds(cond, nzcv) { n } else { m };
            bits(if double { 64 } else { 32 }, chosen)
        }

        /// FMOV, FABS, FNEG and FSQRT inline (`opcode` is bits 20:15,
        /// `ftype` bits 23:22); FRINT* and FCVT go to the evaluator.
        #[inline(always)]
        pub fn scalar_fp1(opcode: u32, ftype: u32, n: u128) -> u128 {
            let double = ftype == 1;
            let (raw, sign) = if double {
                (n as u64, 1u64 << 63)
            } else {
                (u64::from(n as u32), 1u64 << 31)
            };
            match opcode {
                0 => return u128::from(raw),
                // FABS and FNEG only touch the sign, NaN or not.
                1 => return u128::from(raw & !sign),
                2 => return u128::from(raw ^ sign),
                3 => {
                    let root = if double {
                        let root = f64::from_bits(raw).sqrt();
                        (!root.is_nan()).then(|| u128::from(root.to_bits()))
                    } else {
                        let root = f32::from_bits(raw as u32).sqrt();
                        (!root.is_nan()).then(|| u128::from(root.to_bits()))
                    };
                    if let Some(root) = root {
                        return root;
                    }
                }
                _ => {}
            }
            exact::scalar_fp1(opcode, ftype, n)
        }

        /// A value fixed when the code was generated (FMOV immediate).
        #[inline(always)]
        pub fn constant(value: u128) -> u128 {
            value
        }

        /// The low `width` bits of a vector register (FMOV to a general
        /// register).
        #[inline(always)]
        pub fn low_bits(width: u32, n: u128) -> u128 {
            bits(width, n)
        }

        /// The low `width` bits of a general register (FMOV to a vector
        /// register).
        #[inline(always)]
        pub fn from_general(width: u32, x: u64) -> u128 {
            bits(width, u128::from(x))
        }

        /// SCVTF, UCVTF from a 32- or 64-bit general register (the host's
        /// conversion rounds to nearest, as the evaluator's does).
        #[inline(always)]
        pub fn int_to_float(double: bool, unsigned: bool, wide: bool, x: u64) -> u128 {
            match (double, unsigned, wide) {
                (false, false, false) => u128::from((x as i32 as f32).to_bits()),
                (false, false, true) => u128::from((x as i64 as f32).to_bits()),
                (false, true, false) => u128::from((x as u32 as f32).to_bits()),
                (false, true, true) => u128::from((x as f32).to_bits()),
                (true, false, false) => u128::from((f64::from(x as i32)).to_bits()),
                (true, false, true) => u128::from((x as i64 as f64).to_bits()),
                (true, true, false) => u128::from((f64::from(x as u32)).to_bits()),
                (true, true, true) => u128::from((x as f64).to_bits()),
            }
        }

        /// FCVT[NPMZA][SU] to a general register (`mode` is rmode << 3 |
        /// opcode): rounded as a double, then saturated (`as`), NaN 0.
        #[inline(always)]
        pub fn float_to_int(double: bool, mode: u32, wide: bool, n: u128) -> u128 {
            let value = if double {
                f64::from_bits(n as u64)
            } else {
                f64::from(f32::from_bits(n as u32))
            };
            let (rmode, opcode) = (mode >> 3, mode & 7);
            let value = match (rmode, opcode) {
                (0, 0b100) | (0, 0b101) => value.round(),
                (0, _) => value.round_ties_even(),
                (1, _) => value.ceil(),
                (2, _) => value.floor(),
                _ => value.trunc(),
            };
            let unsigned = opcode & 1 == 1;
            u128::from(match (unsigned, wide) {
                (false, false) => u64::from(value as i32 as u32),
                (false, true) => value as i64 as u64,
                (true, false) => u64::from(value as u32),
                (true, true) => value as u64,
            })
        }

        /// EXT: bytes `position..` of n:m (`position` is less than the
        /// vector's bytes).
        #[inline(always)]
        pub fn extract(q: bool, position: u32, n: u128, m: u128) -> u128 {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                host::extract(q, position, n, m)
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let shift = 8 * position;
                if q {
                    if shift == 0 {
                        n
                    } else {
                        (n >> shift) | (m << (128 - shift))
                    }
                } else {
                    let joined = bits(64, n) | (bits(64, m) << 64);
                    bits(64, joined >> shift)
                }
            }
        }

        /// DUP (element), scalar: lane `index` of `esize` bits.
        #[inline(always)]
        pub fn element(esize: u32, index: u32, n: u128) -> u128 {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                host::element(esize, index, n)
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                bits(esize, n >> (index * esize))
            }
        }

        /// ZIP1, UZP1, TRN1, ZIP2, UZP2, TRN2 on `esize`-bit lanes (`op`
        /// is bits 14:12: 1 UZP1, 2 TRN1, 3 ZIP1, 5 UZP2, 6 TRN2, 7 ZIP2).
        #[inline(always)]
        pub fn permute(op: u32, esize: u32, q: bool, n: u128, m: u128) -> u128 {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
            {
                host::permute(op, esize, q, n, m)
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx",
                target_feature = "fma"
            )))]
            {
                let count = if q { 128 } else { 64 } / esize;
                let lane = |v: u128, i: u32| bits(esize, v >> (i * esize));
                let second = u32::from(op >= 5);
                let mut result = 0u128;
                for i in 0..count {
                    let (pair, odd) = (i / 2, i % 2 == 1);
                    let value = match op & 3 {
                        // UZP: the even (odd) elements of n:m.
                        1 => {
                            let j = 2 * i + second;
                            if j < count {
                                lane(n, j)
                            } else {
                                lane(m, j - count)
                            }
                        }
                        // TRN: element 2k (2k + 1) of n, then of m.
                        2 => lane(if odd { m } else { n }, 2 * pair + second),
                        // ZIP: the low (high) halves interleaved.
                        _ => lane(if odd { m } else { n }, pair + second * count / 2),
                    };
                    result |= value << (i * esize);
                }
                result
            }
        }
    }
}

/// Registers written by one SVC: `x0` always carries the result code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SvcOutcome {
    pub x0: i64,
    pub x1: Option<i64>,
    pub x2: Option<i64>,
}

/// System registers reachable from EL0 code.
pub mod sysreg {
    pub const TPIDR_EL0: u32 = 0;
    pub const TPIDRRO_EL0: u32 = 1;
    pub const CNTPCT_EL0: u32 = 2;
    pub const CNTFRQ_EL0: u32 = 3;
    pub const CTR_EL0: u32 = 4;
    pub const FPCR: u32 = 5;
    pub const FPSR: u32 = 6;
}

mod rt {
    use super::{Fault, GuestResult, SvcOutcome};

    extern "Rust" {
        pub fn __recomp_rt_mem_load_u8(address: u64) -> Result<u64, Fault>;
        pub fn __recomp_rt_mem_load_u16(address: u64) -> Result<u64, Fault>;
        pub fn __recomp_rt_mem_load_u32(address: u64) -> Result<u64, Fault>;
        pub fn __recomp_rt_mem_load_u64(address: u64) -> Result<u64, Fault>;
        pub fn __recomp_rt_mem_store_u8(address: u64, value: u64) -> GuestResult;
        pub fn __recomp_rt_mem_store_u16(address: u64, value: u64) -> GuestResult;
        pub fn __recomp_rt_mem_store_u32(address: u64, value: u64) -> GuestResult;
        pub fn __recomp_rt_mem_store_u64(address: u64, value: u64) -> GuestResult;
        pub fn __recomp_rt_mem_load_exclusive(
            address: u64,
            size: u64,
            pair: bool,
        ) -> Result<(u64, u64), Fault>;
        pub fn __recomp_rt_mem_store_exclusive(
            address: u64,
            size: u64,
            pair: bool,
            value: u64,
            second_value: u64,
        ) -> Result<u64, Fault>;
        pub fn __recomp_rt_clear_exclusive();
        pub fn __recomp_rt_memory_fence();
        pub fn __recomp_rt_dispatch(address: u64) -> GuestResult;
        pub fn __recomp_rt_count_calls(count: u32);
        pub fn __recomp_rt_svc(name: &str, args: &[i64]) -> Result<SvcOutcome, Fault>;
        pub fn __recomp_rt_svc_id(id: u64, args: &[i64]) -> Result<SvcOutcome, Fault>;
        pub fn __recomp_rt_syscall_stub(behavior: &str, name: &str, args: &[i64]) -> GuestResult;
        pub fn __recomp_rt_sysreg_read(reg: u32) -> u64;
        pub fn __recomp_rt_sysreg_write(reg: u32, value: u64);
        pub fn __recomp_rt_loop_preempt();
    }
}

macro_rules! forward {
    ($(fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty = $target:ident;)*) => {
        $(
            #[inline]
            pub fn $name($($arg: $ty),*) -> $ret {
                // SAFETY: the runtime defines every `__recomp_rt_*` symbol
                // with exactly these signatures.
                unsafe { rt::$target($($arg),*) }
            }
        )*
    };
}

#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static CONTEXT: std::cell::UnsafeCell<GuestContext> =
    const { std::cell::UnsafeCell::new(GuestContext::ZERO) };
#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static CALLS: Cell<u32> = const { Cell::new(0) };

#[cfg(not(all(windows, recomp_fast_tls)))]
thread_local! {
    /// The calling thread's guest register file. Generated code spills into
    /// it around calls, so the AAPCS64 convention holds across separately
    /// lifted functions; the runtime reads and writes it for SVCs and thread
    /// starts. It lives here, not in the runtime, so the accesses are plain
    /// thread-local ones.
    static CONTEXT: Cell<GuestContext> = const { Cell::new(GuestContext::ZERO) };
    /// Table resolutions `dispatch` made since the thread last reported
    /// them (its direct-call fast path no longer counts every call).
    static CALLS: Cell<u32> = const { Cell::new(0) };
}

/// The calling thread's guest register file. Generated functions built with
/// the guest-context ABI take this pointer as their first argument and read
/// and write the fields directly; the rest of the runtime reaches the same
/// storage through `cpu_regs`/`cpu_store`/`cpu_get` below.
#[inline(always)]
pub fn guest_context() -> *mut GuestContext {
    #[cfg(all(windows, recomp_fast_tls))]
    return CONTEXT.get();
    #[cfg(not(all(windows, recomp_fast_tls)))]
    CONTEXT.with(Cell::as_ptr)
}

/// A snapshot of the guest registers: lifted code built with the snapshot
/// ABI loads its registers from one snapshot after each call.
#[inline(always)]
pub fn cpu_regs() -> [i64; 32] {
    // SAFETY: the cell is this thread's, and nothing holds a reference
    // into it (a `Cell` never hands one out).
    unsafe { (*guest_context()).x }
}

/// Write several guest registers at once (spilled before a call). Only the
/// thread-local's address is fetched here, so each call site stores its
/// registers straight into it (a shared closure that copied a slice of
/// pairs was 3% of LM3's busiest threads).
#[inline(always)]
pub fn cpu_store(values: &[(usize, i64)]) {
    let context = guest_context();
    for &(index, value) in values {
        // SAFETY: as in `cpu_regs`.
        unsafe { (*context).x[index] = value };
    }
}

/// Replace every guest register (a thread start).
pub fn cpu_replace(values: [i64; 32]) {
    let context = guest_context();
    // SAFETY: as in `cpu_regs`.
    unsafe { (*context).x = values };
}

#[inline]
pub fn cpu_get(index: usize) -> i64 {
    // SAFETY: as in `cpu_regs`.
    unsafe { (*guest_context()).x[index] }
}

#[inline]
pub fn cpu_set(index: usize, value: i64) {
    cpu_store(&[(index, value)]);
}

/// The guest NZCV flags as of the last call or return.
#[inline(always)]
pub fn cpu_nzcv() -> u8 {
    // SAFETY: as in `cpu_regs`.
    unsafe { (*guest_context()).nzcv }
}

/// Store the guest NZCV flags (before a call or return).
#[inline(always)]
pub fn cpu_set_nzcv(nzcv: u8) {
    let context = guest_context();
    // SAFETY: as in `cpu_regs`.
    unsafe { (*context).nzcv = nzcv };
}

#[inline(always)]
pub fn cpu_vregs() -> [u128; 32] {
    // SAFETY: as in `cpu_regs`.
    unsafe { (*guest_context()).v }
}

#[inline(always)]
pub fn cpu_vstore(values: &[(usize, u128)]) {
    let context = guest_context();
    for &(index, value) in values {
        // SAFETY: as in `cpu_regs`.
        unsafe { (*context).v[index] = value };
    }
}

/// A recompiled guest function.
pub type GuestFn = fn() -> GuestResult;

/// The program's functions by guest address, hashed: every guest call looks
/// its target up (a binary search over a large title's ~90,000 functions
/// took 17 probes).
pub struct Targets {
    shift: u32,
    slots: Box<[(u64, Option<GuestFn>)]>,
}

impl Targets {
    pub fn new(targets: &[(u64, GuestFn)]) -> Self {
        // At most half full, so a probe sequence stays short.
        let bits = (targets.len().max(1) * 2)
            .next_power_of_two()
            .trailing_zeros()
            .max(4);
        let mut table = Self {
            shift: 64 - bits,
            slots: vec![(0, None); 1 << bits].into_boxed_slice(),
        };
        for &(address, function) in targets {
            let mut index = table.slot(address);
            while table.slots[index].1.is_some() && table.slots[index].0 != address {
                index = (index + 1) & (table.slots.len() - 1);
            }
            table.slots[index] = (address, Some(function));
        }
        table
    }

    #[inline]
    fn slot(&self, address: u64) -> usize {
        ((address >> 2).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    #[inline]
    pub fn get(&self, address: u64) -> Option<GuestFn> {
        let mut index = self.slot(address);
        loop {
            let (key, function) = self.slots[index];
            if key == address || function.is_none() {
                return function;
            }
            index = (index + 1) & (self.slots.len() - 1);
        }
    }
}

/// The table calls go straight through, when the runtime allows it (see
/// `publish_targets`).
static DIRECT_TARGETS: AtomicPtr<Targets> = AtomicPtr::new(std::ptr::null_mut());

/// Calls between reports to the runtime's progress count.
const CALL_BATCH: u32 = 1024;

/// Let `dispatch` call functions in `targets` without entering the runtime
/// (only when the runtime has nothing to do at a call: parallel threads, no
/// call hooks), or stop it with `None`.
pub fn publish_targets(targets: Option<&'static Targets>) {
    let pointer = targets.map_or(std::ptr::null_mut(), |targets| {
        std::ptr::from_ref(targets).cast_mut()
    });
    DIRECT_TARGETS.store(pointer, Ordering::Release);
}

/// Whether generated code may call the functions it names straight (see
/// `call`): only while the table is published and no script hooks a
/// function (a hook is only in the table).
static DIRECT_CALLS: AtomicBool = AtomicBool::new(false);

/// Let `call` go straight to its target (`true`), or make it go through
/// `dispatch` again.
pub fn allow_direct_calls(allowed: bool) {
    DIRECT_CALLS.store(allowed, Ordering::Release);
}

/// Count a call made without entering the runtime, for its progress count.
#[inline(always)]
fn count_call() {
    #[cfg(all(windows, recomp_fast_tls))]
    let report = {
        let next = CALLS.get() + 1;
        CALLS.set(if next == CALL_BATCH { 0 } else { next });
        next == CALL_BATCH
    };
    #[cfg(not(all(windows, recomp_fast_tls)))]
    let report = CALLS.with(|calls| {
        let next = calls.get() + 1;
        calls.set(if next == CALL_BATCH { 0 } else { next });
        next == CALL_BATCH
    });
    if report {
        // SAFETY: the runtime defines this symbol with this signature.
        unsafe { rt::__recomp_rt_count_calls(CALL_BATCH) };
    }
}

/// Whether generated code's direct calls may skip the runtime (see `call`).
///
/// Snapshot calls (`dispatch`) and script hooks read the caller's registers
/// from the thread's register file, so generated code spills every register
/// before them. While this returns true it may spill only the registers the
/// callee needs, which is much less.
#[inline(always)]
pub fn direct_calls_enabled() -> bool {
    DIRECT_CALLS.load(Ordering::Relaxed)
}

/// Call a guest function that is known by name, without the `dispatch`
/// fallback. Generated code only uses this after `direct_calls_enabled`
/// returned true and has spilled just the callee's inputs; a fallback into
/// `dispatch` there would show a hook stale registers. The call is not
/// counted here: generated loops report through `backedge`, and `dispatch`
/// reports when it resolves a target, so the runtime's hang monitor still
/// hears from a busy thread without a TLS increment per call.
#[inline(always)]
pub fn call_direct(function: impl FnOnce() -> GuestResult) -> GuestResult {
    function()
}

/// Call the guest function at `address`, which generated code knows by name
/// (`function`, a direct call to it): straight there while direct calls are
/// allowed, else through `dispatch`. Looking every call up in the table
/// (several MB for a large title, so most lookups missed the cache) was a
/// twentieth of LM3's busiest threads.
///
/// The straight path does not report progress: a generated loop already
/// reports through `backedge`, and `dispatch` reports on every table
/// resolution, so the runtime's hang monitor still hears from a busy
/// thread. This path is only taken in parallel mode (the runtime allows
/// direct calls only when there is no run lock to hand over), so serialized
/// preemption is unchanged.
#[inline(always)]
pub fn call(address: u64, function: impl FnOnce() -> GuestResult) -> GuestResult {
    if direct_calls_enabled() {
        return call_direct(function);
    }
    dispatch(address)
}

/// Call the guest function at `address`.
#[inline(never)]
pub fn dispatch(address: u64) -> GuestResult {
    let targets = DIRECT_TARGETS.load(Ordering::Acquire);
    if !targets.is_null() {
        // SAFETY: published tables are `'static`.
        if let Some(function) = unsafe { &*targets }.get(address) {
            count_call();
            return function();
        }
    }
    // SAFETY: the runtime defines this symbol with this signature.
    unsafe { rt::__recomp_rt_dispatch(address) }
}

/// LDAR's ordering: no later access moves before the load. A compiler barrier
/// on x86-64 (total store order), a `dmb ishld` on AArch64.
#[inline(always)]
pub fn acquire_fence() {
    core::sync::atomic::fence(Ordering::Acquire);
}

/// STLR's ordering: no earlier access moves after the store. A compiler
/// barrier on x86-64, a `dmb ish` on AArch64.
#[inline(always)]
pub fn release_fence() {
    core::sync::atomic::fence(Ordering::Release);
}

/// Host address of guest address 0 when the runtime keeps guest memory in
/// one host reservation (see `recomp-runtime`'s `fastmem`), else 0. Loads
/// and stores below then access it directly instead of calling the runtime.
static MEMORY_BASE: AtomicUsize = AtomicUsize::new(0);

/// The guest address space a reservation covers when the whole space was
/// reserved.
const MEMORY_SPACE: u64 = 1 << 39;

/// End (guest address, exclusive) of the range `MEMORY_BASE` covers. A
/// host cannot always reserve all 512 GiB (a 39-bit user address space, as
/// on many Android devices, cannot hold it), so the runtime may reserve
/// only the window the title's memory layout uses. Addresses at or above
/// this end stay on the checked path; below it, `base + address` is always
/// inside the reservation.
static MEMORY_LIMIT: AtomicU64 = AtomicU64::new(0);

/// Published by the runtime once guest memory is in place; 0 turns direct
/// access off. `end` is the covered range's end in guest addresses.
pub fn set_memory_range(base: usize, end: u64) {
    // The limit is written before the base: a non-zero base publishes both.
    MEMORY_LIMIT.store(end.min(MEMORY_SPACE), Ordering::Release);
    MEMORY_BASE.store(base, Ordering::Release);
}

/// The guest address space a reservation covers.
pub fn set_memory_base(base: usize) {
    set_memory_range(base, if base == 0 { 0 } else { MEMORY_SPACE });
}

/// The published base (0: every access goes through the runtime).
pub fn memory_base() -> usize {
    MEMORY_BASE.load(Ordering::Acquire)
}

/// The end (guest address) the published base covers; 0 with no base. Host
/// code that turns guest addresses into pointers checks against this, as
/// generated code does.
pub fn memory_limit() -> u64 {
    MEMORY_LIMIT.load(Ordering::Acquire)
}

/// The host address of `size` bytes of guest memory at `address`, when
/// accesses go there directly.
#[inline(always)]
fn direct(address: u64, size: u64) -> Option<usize> {
    let base = MEMORY_BASE.load(Ordering::Relaxed);
    let limit = MEMORY_LIMIT.load(Ordering::Relaxed);
    (base != 0 && size <= limit && address <= limit - size).then(|| base + address as usize)
}

/// Guest memory as one generated function accesses it: the reservation's
/// base, read once as the function starts (it is published before guest
/// code runs and never moves), and the end of the addresses accessed
/// there directly (0 when every access goes through the runtime). An
/// access is then one compare and the host load or store; reading the
/// shared base at every access cost LM3's job code ten instructions each.
#[derive(Debug, Clone, Copy)]
pub struct Memory {
    base: usize,
    limit: u64,
}

/// The view of guest memory generated functions take as they start.
#[inline(always)]
pub fn memory() -> Memory {
    let base = MEMORY_BASE.load(Ordering::Relaxed);
    let limit = MEMORY_LIMIT.load(Ordering::Relaxed);
    Memory {
        base,
        // Room for the widest access (16 bytes) below the covered end.
        limit: if base == 0 || limit < 16 {
            0
        } else {
            limit - 16
        },
    }
}

macro_rules! memory_view {
    ($($load:ident, $store:ident, $ty:ty, $rt_load:ident, $rt_store:ident;)*) => {
        impl Memory {
            $(
                #[inline(always)]
                pub fn $load(self, address: u64) -> Result<u64, Fault> {
                    if address < self.limit {
                        // SAFETY: the reservation covers the address; an
                        // unmapped page faults into the runtime's handler.
                        return Ok(unsafe {
                            ((self.base + address as usize) as *const $ty).read_unaligned()
                        } as u64);
                    }
                    slow::$load(address)
                }

                #[inline(always)]
                pub fn $store(self, address: u64, value: u64) -> GuestResult {
                    if address < self.limit {
                        // SAFETY: as for the load.
                        unsafe {
                            ((self.base + address as usize) as *mut $ty).write_unaligned(value as $ty)
                        };
                        return Ok(());
                    }
                    slow::$store(address, value)
                }
            )*
        }

        /// Accesses outside the reservation, kept out of the generated code.
        mod slow {
            use super::{rt, Fault, GuestResult};
            $(
                #[cold]
                #[inline(never)]
                pub fn $load(address: u64) -> Result<u64, Fault> {
                    // SAFETY: the runtime defines this symbol with this signature.
                    unsafe { rt::$rt_load(address) }
                }

                #[cold]
                #[inline(never)]
                pub fn $store(address: u64, value: u64) -> GuestResult {
                    // SAFETY: the runtime defines this symbol with this signature.
                    unsafe { rt::$rt_store(address, value) }
                }
            )*
        }
    };
}

impl Memory {
    /// A vector register's 16 bytes in one host access, as a host vector:
    /// the value can then stay in a vector register into the SIMD
    /// operations using it (as an integer, the load and its slow path
    /// merged in a pair of general registers).
    #[inline(always)]
    pub fn load_u128(self, address: u64) -> Result<u128, Fault> {
        let value = if address < self.limit {
            // SAFETY: as for the narrower loads; `limit` leaves room for 16
            // bytes.
            Ok(unsafe { ((self.base + address as usize) as *const Wide).read_unaligned() })
        } else {
            load_u128_slow(address)
        };
        value.map(from_wide)
    }

    #[inline(always)]
    pub fn store_u128(self, address: u64, value: u128) -> GuestResult {
        let value = to_wide(value);
        if address < self.limit {
            // SAFETY: as for the load.
            unsafe { ((self.base + address as usize) as *mut Wide).write_unaligned(value) };
            return Ok(());
        }
        store_u128_slow(address, value)
    }
}

/// 16 bytes as the host keeps them in a vector register.
#[cfg(target_arch = "x86_64")]
type Wide = core::arch::x86_64::__m128;
#[cfg(not(target_arch = "x86_64"))]
type Wide = u128;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn from_wide(value: Wide) -> u128 {
    // SAFETY: both are 16 plain bytes.
    unsafe { core::mem::transmute::<Wide, u128>(value) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn from_wide(value: Wide) -> u128 {
    value
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn to_wide(value: u128) -> Wide {
    // SAFETY: as in `from_wide`.
    unsafe { core::mem::transmute::<u128, Wide>(value) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn to_wide(value: u128) -> Wide {
    value
}

/// Outside the reservation, in two halves.
#[cold]
#[inline(never)]
fn load_u128_slow(address: u64) -> Result<Wide, Fault> {
    let low = slow::load_u64(address)?;
    let high = slow::load_u64(address.wrapping_add(8))?;
    Ok(to_wide(u128::from(low) | (u128::from(high) << 64)))
}

#[cold]
#[inline(never)]
fn store_u128_slow(address: u64, value: Wide) -> GuestResult {
    let value = from_wide(value);
    slow::store_u64(address, value as u64)?;
    slow::store_u64(address.wrapping_add(8), (value >> 64) as u64)
}

memory_view! {
    load_u8, store_u8, u8, __recomp_rt_mem_load_u8, __recomp_rt_mem_store_u8;
    load_u16, store_u16, u16, __recomp_rt_mem_load_u16, __recomp_rt_mem_store_u16;
    load_u32, store_u32, u32, __recomp_rt_mem_load_u32, __recomp_rt_mem_store_u32;
    load_u64, store_u64, u64, __recomp_rt_mem_load_u64, __recomp_rt_mem_store_u64;
}

macro_rules! memory_access {
    ($($load:ident, $store:ident, $ty:ty, $rt_load:ident, $rt_store:ident;)*) => {
        $(
            #[inline]
            pub fn $load(address: u64) -> Result<u64, Fault> {
                if let Some(host) = direct(address, size_of::<$ty>() as u64) {
                    // SAFETY: the reservation covers the address; an
                    // unmapped page faults into the runtime's handler.
                    return Ok(unsafe { (host as *const $ty).read_unaligned() } as u64);
                }
                // SAFETY: the runtime defines this symbol with this signature.
                unsafe { rt::$rt_load(address) }
            }

            #[inline]
            pub fn $store(address: u64, value: u64) -> GuestResult {
                if let Some(host) = direct(address, size_of::<$ty>() as u64) {
                    // SAFETY: as for the load.
                    unsafe { (host as *mut $ty).write_unaligned(value as $ty) };
                    return Ok(());
                }
                // SAFETY: the runtime defines this symbol with this signature.
                unsafe { rt::$rt_store(address, value) }
            }
        )*
    };
}

memory_access! {
    mem_load_u8, mem_store_u8, u8, __recomp_rt_mem_load_u8, __recomp_rt_mem_store_u8;
    mem_load_u16, mem_store_u16, u16, __recomp_rt_mem_load_u16, __recomp_rt_mem_store_u16;
    mem_load_u32, mem_store_u32, u32, __recomp_rt_mem_load_u32, __recomp_rt_mem_store_u32;
    mem_load_u64, mem_store_u64, u64, __recomp_rt_mem_load_u64, __recomp_rt_mem_store_u64;
}

forward! {
    // The value(s) an exclusive load read; the reservation is kept by the
    // runtime, per thread.
    fn mem_load_exclusive(address: u64, size: u64, pair: bool) -> Result<(u64, u64), Fault> = __recomp_rt_mem_load_exclusive;
    // The exclusive store's success flag: 0 stored, 1 the reservation was lost.
    fn mem_store_exclusive(address: u64, size: u64, pair: bool, value: u64, second_value: u64) -> Result<u64, Fault> = __recomp_rt_mem_store_exclusive;
    fn clear_exclusive() -> () = __recomp_rt_clear_exclusive;
    fn memory_fence() -> () = __recomp_rt_memory_fence;
    fn svc_dispatch(name: &str, args: &[i64]) -> Result<SvcOutcome, Fault> = __recomp_rt_svc;
    // The numeric form generated code uses: the pipeline resolves the name
    // to its kernel id at recompilation time, so the hot path carries no
    // string and the runtime skips the name lookup.
    fn svc_dispatch_id(id: u64, args: &[i64]) -> Result<SvcOutcome, Fault> = __recomp_rt_svc_id;
    fn syscall_stub(behavior: &str, name: &str, args: &[i64]) -> GuestResult = __recomp_rt_syscall_stub;
    fn sysreg_read(reg: u32) -> u64 = __recomp_rt_sysreg_read;
    fn sysreg_write(reg: u32, value: u64) -> () = __recomp_rt_sysreg_write;
    fn loop_preempt() -> () = __recomp_rt_loop_preempt;
}
