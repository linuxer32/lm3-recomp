//! Differential tests for the inline SIMD fast paths against the evaluator's
//! `exec`.
//!
//! `recomp-abi` does not depend on `recomp-simd`: the runtime defines the
//! `__recomp_simd_*` symbols the inline paths fall back to at link time.
//! These tests play the runtime's part, and count the fallbacks: a finite
//! single-precision result must never reach the evaluator, which is what
//! proves the host path was actually taken.

use std::cell::Cell;

use recomp_abi::simd::fast;
use recomp_simd::exec;

// Per thread, so the tests (which run in parallel) do not observe each
// other's fallbacks.
thread_local! {
    static VECTOR3_EXACT: Cell<usize> = const { Cell::new(0) };
    static VECTOR_ELEMENT_EXACT: Cell<usize> = const { Cell::new(0) };
}

fn fallbacks(counter: &'static std::thread::LocalKey<Cell<usize>>) -> usize {
    counter.with(Cell::get)
}

#[no_mangle]
pub fn __recomp_simd_vector3(kind: u32, double: bool, q: bool, d: u128, n: u128, m: u128) -> u128 {
    VECTOR3_EXACT.with(|count| count.set(count.get() + 1));
    recomp_simd::fast::vector3(kind, double, q, d, n, m)
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub fn __recomp_simd_vector_element(
    op: u32,
    double: bool,
    q: bool,
    scalar: bool,
    index: u32,
    d: u128,
    n: u128,
    m: u128,
) -> u128 {
    VECTOR_ELEMENT_EXACT.with(|count| count.set(count.get() + 1));
    recomp_simd::fast::vector_element(op, double, q, scalar, index, d, n, m)
}

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

/// Zeros, infinities, quiet and signalling NaNs, denormals, extremes and
/// ordinary numbers.
const EDGE32: [u32; 12] = [
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

/// Values whose operations (except divide) cannot produce a NaN.
const FINITE32: [u32; 8] = [
    0,
    0x8000_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x0000_0001,
    0x0080_0000,
    0x7F7F_FFFF,
    0x4F00_0000,
];

fn put(value: u128, esize: u32, index: u32, lane: u64) -> u128 {
    let shift = esize * index;
    let lane_mask = if esize == 128 {
        u128::MAX
    } else {
        (1u128 << esize) - 1
    };
    (value & !(lane_mask << shift)) | (u128::from(lane) << shift)
}

fn register(rng: &mut Rng, table: &[u32]) -> u128 {
    let mut value = 0u128;
    for i in 0..4 {
        value = put(
            value,
            32,
            i,
            u64::from(table[(rng.next() as usize) % table.len()]),
        );
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
fn element_word(op: u32, q: bool, scalar: bool, index: u32) -> u32 {
    let (u, opcode) = match op {
        0 => (false, 0b0001u32),
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
    let size = 2;
    let (l, h) = ((index & 1) << 21, (index >> 1) << 11);
    group as u32
        | (u32::from(u) << 29)
        | (size << 22)
        | l
        | (2 << 16)
        | (opcode << 12)
        | h
        | (1 << 5)
        | 3
}

/// Whether any of the first `lanes` single-precision lanes of `bits` is NaN.
fn any_nan(lanes: usize, bits: u128) -> bool {
    const ABS: u128 = 0x7FFF_FFFF_7FFF_FFFF_7FFF_FFFF_7FFF_FFFF;
    const CARRY: u128 = 0x007F_FFFF_007F_FFFF_007F_FFFF_007F_FFFF;
    const SIGN: u128 = 0x8000_0000_8000_0000_8000_0000_8000_0000;
    let mut lanes_mask = 0u128;
    for i in 0..lanes {
        lanes_mask |= u128::from(u32::MAX) << (32 * i);
    }
    ((bits & lanes_mask & ABS).wrapping_add(CARRY)) & SIGN != 0
}

/// The in-place calls the generated direct-register code uses must write the
/// same register as the value-shaped calls, including the zeroing of the
/// lanes a 64-bit vector does not write.
#[test]
fn the_in_place_paths_match_the_value_paths() {
    let mut rng = Rng(0xA5A5_5A5A_0123_4567);
    let mut out: u128;
    for q in [false, true] {
        let lanes_mask = if q { u128::MAX } else { u128::from(u64::MAX) };
        for opcode in [
            0b11000u32, 0b11001, 0b11010, 0b11011, 0b11100, 0b11101, 0b11110, 0b11111,
        ] {
            for u in [false, true] {
                let word = three_same_word(u, 0, opcode, q);
                let kind = (u32::from(u) << 6) | opcode;
                let n = register(&mut rng, &EDGE32);
                let m = register(&mut rng, &EDGE32);
                let d = register(&mut rng, &EDGE32);
                let Some(expected) = exec(word, &[(1, n), (2, m), (3, d)], 0, 0) else {
                    continue;
                };
                let expected = expected.value;
                let value = fast::vector3(kind, false, q, d, n, m);
                out = u128::MAX;
                unsafe { fast::vector3_into(&mut out, kind, false, q, d, n, m) };
                assert_eq!(value, out, "{word:#010x}");
                assert_eq!(value, expected & lanes_mask, "{word:#010x}");
                assert_eq!(out, expected & lanes_mask, "{word:#010x}");
            }
        }
        for scalar in [false, true] {
            if scalar && !q {
                continue;
            }
            for op in 0..4u32 {
                for index in 0..4u32 {
                    let word = element_word(op, q, scalar, index);
                    let n = register(&mut rng, &EDGE32);
                    let m = register(&mut rng, &EDGE32);
                    let d = register(&mut rng, &EDGE32);
                    let Some(expected) = exec(word, &[(3, d), (1, n), (2, m)], 0, 0) else {
                        continue;
                    };
                    let expected = expected.value;
                    let value = fast::vector_element(op, false, q, scalar, index, d, n, m);
                    out = u128::MAX;
                    unsafe {
                        fast::vector_element_into(&mut out, op, false, q, scalar, index, d, n, m)
                    };
                    assert_eq!(value, out, "{word:#010x} (op {op}, index {index})");
                    assert_eq!(
                        value,
                        expected & lanes_mask,
                        "{word:#010x} (op {op}, index {index})"
                    );
                }
            }
        }
    }
}

#[test]
fn vector3_fast_path_matches_exec_on_edge_values() {
    let mut rng = Rng(0x0FED_CBA9_8765_4321);
    for size in 0..4u32 {
        for q in [false, true] {
            if size & 1 == 1 && !q {
                continue;
            }
            let double = size & 1 == 1;
            let esize = if double { 64 } else { 32 };
            if double {
                // The single-precision inline path is the subject here; the
                // double forms still go through `exec`-equivalent code.
                continue;
            }
            for opcode in [
                0b11000u32, 0b11001, 0b11010, 0b11011, 0b11100, 0b11101, 0b11110,
            ] {
                for u in [false, true] {
                    let word = three_same_word(u, size, opcode, q);
                    let kind = (u32::from(u) << 6) | ((size >> 1) << 5) | opcode;
                    for iteration in 0..64 {
                        let finite = iteration % 4 == 3;
                        let make = |rng: &mut Rng| {
                            if finite {
                                register(rng, &FINITE32)
                            } else {
                                register(rng, &EDGE32)
                            }
                        };
                        let n = make(&mut rng);
                        let m = make(&mut rng);
                        let d = make(&mut rng);
                        let Some(expected) = exec(word, &[(1, n), (2, m), (3, d)], 0, 0) else {
                            continue;
                        };
                        let before = fallbacks(&VECTOR3_EXACT);
                        let got = fast::vector3(kind, false, q, d, n, m);
                        assert_eq!(
                            got, expected.value,
                            "{word:#010x} (kind {kind:#04x}, q {q}) with \
                             n {n:#034x}, m {m:#034x}, d {d:#034x}"
                        );
                        // FADD, FSUB, FMUL, FMULX, FMLA and FMLS cannot
                        // produce a NaN from finite operands, so the host
                        // path must have handled them.
                        let arithmetic = matches!(opcode, 0b11001..=0b11011) && !u;
                        if finite && arithmetic {
                            assert_eq!(
                                fallbacks(&VECTOR3_EXACT),
                                before,
                                "the host path was skipped for finite {word:#010x}"
                            );
                        }
                        // The compare forms write all-ones lanes, which are
                        // not NaNs; only the arithmetic results can make the
                        // host path fall back.
                        let arithmetic_result = !matches!(opcode, 0b11100 | 0b11101);
                        if !finite
                            && arithmetic_result
                            && any_nan(if q { 4 } else { 2 }, expected.value)
                        {
                            // Edge operands sometimes still produce finite
                            // results; only require the fallback when they
                            // actually produce a NaN.
                            assert!(
                                fallbacks(&VECTOR3_EXACT) > before,
                                "a NaN result skipped the evaluator for {word:#010x}"
                            );
                        }
                    }
                }
            }
            let _ = esize;
        }
    }
}

#[test]
fn vector_element_fast_path_matches_exec_on_edge_values() {
    let mut rng = Rng(0xC001_D00D_1234_5678);
    for scalar in [false, true] {
        for q in [false, true] {
            if scalar && !q {
                continue;
            }
            if !scalar && !q {
                continue;
            }
            for op in 0..4u32 {
                for index in 0..4u32 {
                    let word = element_word(op, q, scalar, index);
                    for iteration in 0..32 {
                        let finite = iteration % 3 == 2;
                        let make = |rng: &mut Rng| {
                            if finite {
                                register(rng, &FINITE32)
                            } else {
                                register(rng, &EDGE32)
                            }
                        };
                        let n = make(&mut rng);
                        let m = make(&mut rng);
                        let d = make(&mut rng);
                        let Some(expected) = exec(word, &[(3, d), (1, n), (2, m)], 0, 0) else {
                            continue;
                        };
                        let before = fallbacks(&VECTOR_ELEMENT_EXACT);
                        let got = fast::vector_element(op, false, q, scalar, index, d, n, m);
                        assert_eq!(
                            got, expected.value,
                            "{word:#010x} (op {op}, q {q}, scalar {scalar}, index \
                             {index}) with n {n:#034x}, m {m:#034x}, d {d:#034x}"
                        );
                        // FMLA/FMLS/FMUL/FMULX with finite operands cannot
                        // produce a NaN: the host path must have taken them.
                        if finite {
                            assert_eq!(
                                fallbacks(&VECTOR_ELEMENT_EXACT),
                                before,
                                "the host path was skipped for finite {word:#010x}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Every single-precision NaN encoding in every lane must fall back, and
/// every non-NaN value (including infinities and denormals) must stay on the
/// host path.
#[test]
fn nan_detection_is_exact_in_every_lane() {
    let mut rng = Rng(0x51AA_5EED_0BAD_F00D);
    for lane in 0..4u32 {
        for bits in [
            0x7F80_0001u32, // signalling NaN
            0x7FC0_0000,    // quiet NaN
            0xFFFF_FFFF,    // negative quiet NaN with all mantissa bits
            0x7FFF_FFFF,    // largest NaN payload
            0xFF80_0001,    // negative signalling NaN
        ] {
            let mut value = register(&mut rng, &FINITE32);
            value = put(value, 32, lane, u64::from(bits));
            let before = fallbacks(&VECTOR3_EXACT);
            // FMUL by one keeps the lane's NaN, which must reach the
            // evaluator with Arm's quieting and payload selection.
            let one = 0x3F80_0000_3F80_0000_3F80_0000_3F80_0000u128;
            let got = fast::vector3(0b011011, false, true, 0, value, one);
            assert_eq!(
                got,
                exec(0x6E22DC23, &[(1, value), (2, one)], 0, 0)
                    .expect("fmul.4s is implemented")
                    .value
            );
            assert!(
                fallbacks(&VECTOR3_EXACT) > before,
                "lane {lane} NaN {bits:#010x} did not take the exact path"
            );
        }
    }
    // Infinity, the largest finite value and a denormal are not NaNs and
    // must stay on the host path.
    for finite in [0x7F80_0000u32, 0x7F7F_FFFF, 0x0000_0001, 0x8000_0000] {
        let value = u128::from(finite)
            | (u128::from(finite) << 32)
            | (u128::from(finite) << 64)
            | (u128::from(finite) << 96);
        let one = 0x3F80_0000_3F80_0000_3F80_0000_3F80_0000u128;
        let before = fallbacks(&VECTOR3_EXACT);
        assert_eq!(
            fast::vector3(0b011011, false, true, 0, value, one),
            exec(0x6E22DC23, &[(1, value), (2, one)], 0, 0)
                .expect("fmul.4s is implemented")
                .value
        );
        assert_eq!(
            fallbacks(&VECTOR3_EXACT),
            before,
            "{finite:#010x} is not a NaN"
        );
    }
}
