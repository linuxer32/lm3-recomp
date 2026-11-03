//! Known-answer tests for the evaluator. Encodings come from llvm-mc.

use recomp_simd::{exec, operands, Operands};

fn ops(word: u32) -> Operands {
    operands(word).unwrap_or_else(|| panic!("{word:#010x} unsupported"))
}

/// Run `word` with the given vector registers; every register it reads
/// must be supplied.
fn run(word: u32, vector: &[(u8, u128)]) -> u128 {
    full(word, vector, 0, 0).value
}

fn full(word: u32, vector: &[(u8, u128)], general: u64, nzcv: u8) -> recomp_simd::Outcome {
    let operands = ops(word);
    for reg in &operands.vector_reads {
        assert!(
            vector.iter().any(|(r, _)| r == reg),
            "{word:#010x} reads v{reg}"
        );
    }
    exec(word, vector, general, nzcv).expect("evaluates")
}

fn s(value: f32) -> u128 {
    u128::from(value.to_bits())
}

fn d(value: f64) -> u128 {
    u128::from(value.to_bits())
}

fn s4(values: [f32; 4]) -> u128 {
    values
        .iter()
        .rev()
        .fold(0, |acc, v| (acc << 32) | u128::from(v.to_bits()))
}

fn w4(values: [u32; 4]) -> u128 {
    values
        .iter()
        .rev()
        .fold(0, |acc, v| (acc << 32) | u128::from(*v))
}

const GARBAGE: u128 = 0xDEAD_BEEF_DEAD_BEEF_DEAD_BEEF_0000_0000;

#[test]
fn operands_name_registers_and_flags() {
    // ucvtf s0, w8
    let o = ops(0x1e23_0100);
    assert_eq!(o.general_read, Some(8));
    assert_eq!(o.vector_write, Some(0));
    assert!(o.vector_reads.is_empty());
    // fcmp s0, s1
    let o = ops(0x1e21_2000);
    assert!(o.writes_flags && !o.reads_flags);
    assert_eq!(o.vector_reads, vec![0, 1]);
    // fccmp d0, d1, #4, ne
    let o = ops(0x1e61_1404);
    assert!(o.writes_flags && o.reads_flags);
    // fcvtzs w0, s1
    assert_eq!(ops(0x1e38_0020).general_write, Some(0));
    // fmla v0.4s, v1.4s, v2.s[1] reads its destination.
    assert_eq!(ops(0x4fa2_1020).vector_reads, vec![1, 2, 0]);
    // Loads and integer instructions are not ours.
    assert_eq!(operands(0x3dc0_0000), None); // ldr q0, [x0]
    assert_eq!(operands(0x8b02_0020), None); // add x0, x1, x2
}

#[test]
fn integer_conversions() {
    // ucvtf s0, w8: only the low 32 bits count.
    let out = full(0x1e23_0100, &[], 0x1_0000_0003, 0);
    assert_eq!(out.value, s(3.0));
    assert_eq!(
        full(0x1e23_0100, &[], 0xFFFF_FFFF, 0).value,
        s(4_294_967_296.0)
    );
    // scvtf d1, x2
    assert_eq!(full(0x9e62_0041, &[], u64::MAX, 0).value, d(-1.0));
    // fcvtzs w0, s1: truncates, zero-extends, saturates.
    assert_eq!(full(0x1e38_0020, &[(1, s(-3.7))], 0, 0).value, 0xFFFF_FFFD);
    assert_eq!(full(0x1e38_0020, &[(1, s(1e20))], 0, 0).value, 0x7FFF_FFFF);
    assert_eq!(full(0x1e38_0020, &[(1, s(f32::NAN))], 0, 0).value, 0);
    // fcvtzu x0, d1 of a negative value is 0.
    assert_eq!(full(0x9e79_0020, &[(1, d(-1.0))], 0, 0).value, 0);
    // fcvtms x0, d1 rounds down; fcvtas rounds ties away; fcvtns to even.
    assert_eq!(
        full(0x9e70_0020, &[(1, d(-2.5))], 0, 0).value,
        (-3i64) as u64 as u128
    );
    assert_eq!(full(0x1e24_0020, &[(1, s(2.5))], 0, 0).value, 3);
    assert_eq!(full(0x1e20_0020, &[(1, s(2.5))], 0, 0).value, 2);
    // fcvtzs w0, s1, #8: fixed point with 8 fraction bits.
    assert_eq!(full(0x1e18_e020, &[(1, s(1.5))], 0, 0).value, 384);
    // scvtf v0.4s, v1.4s, #16
    let fixed = w4([0x0001_0000, 0x0000_8000, 0xFFFF_0000, 0]);
    assert_eq!(run(0x4f30_e420, &[(1, fixed)]), s4([1.0, 0.5, -1.0, 0.0]));
}

#[test]
fn scalar_arithmetic_zeroes_the_rest_of_the_register() {
    // fadd s0, s1, s2
    assert_eq!(
        run(0x1e22_2820, &[(1, s(1.5) | GARBAGE), (2, s(2.25))]),
        s(3.75)
    );
    // fmadd d0, d1, d2, d3 = d3 + d1 * d2
    assert_eq!(
        run(0x1f42_0c20, &[(1, d(2.0)), (2, d(3.0)), (3, d(1.0))]),
        d(7.0)
    );
    // fnmsub s0, s1, s2, s3 = -s3 + s1 * s2
    assert_eq!(
        run(0x1f22_8c20, &[(1, s(2.0)), (2, s(3.0)), (3, s(1.0))]),
        s(5.0)
    );
    // frintm d0, d1
    assert_eq!(run(0x1e65_4020, &[(1, d(-1.5))]), d(-2.0));
    // frecps s0, s1, s2 = 2 - s1 * s2
    assert_eq!(run(0x5e22_fc20, &[(1, s(1.5)), (2, s(0.5))]), s(1.25));
}

#[test]
fn nans_and_signed_zeros_follow_the_arm_rules() {
    // fsub s0, s1, s2: inf - inf is the positive default NaN.
    let inf = s(f32::INFINITY);
    assert_eq!(run(0x1e22_3820, &[(1, inf), (2, inf)]), 0x7FC0_0000);
    // A signalling NaN operand comes back quiet, ahead of a quiet one.
    let snan = 0xFF80_0001;
    assert_eq!(
        run(0x1e22_2820, &[(1, 0x7FC0_0002), (2, snan)]),
        0xFFC0_0001
    );
    // fmax/fmin s0, s1, s2 on +0 and -0.
    let (pz, nz) = (s(0.0), s(-0.0));
    assert_eq!(run(0x1e22_4820, &[(1, nz), (2, pz)]), pz);
    assert_eq!(run(0x1e22_5820, &[(1, pz), (2, nz)]), nz);
    // fmaxnm prefers a number over a quiet NaN; fmax propagates it.
    let qnan = 0x7FC0_1234;
    assert_eq!(run(0x1e22_6820, &[(1, qnan), (2, s(1.0))]), s(1.0));
    assert_eq!(run(0x1e22_4820, &[(1, qnan), (2, s(1.0))]), qnan);
}

#[test]
fn compares_and_conditional_selects() {
    // fcmp s0, s1
    let cmp = |a: f32, b: f32| full(0x1e21_2000, &[(0, s(a)), (1, s(b))], 0, 0).nzcv;
    assert_eq!(cmp(1.0, 1.0), 0b0110);
    assert_eq!(cmp(1.0, 2.0), 0b1000);
    assert_eq!(cmp(2.0, 1.0), 0b0010);
    assert_eq!(cmp(f32::NAN, 1.0), 0b0011);
    // fcmpe d1, #0.0
    assert_eq!(full(0x1e60_2038, &[(1, d(-0.0))], 0, 0).nzcv, 0b0110);
    // fcsel s0, s1, s2, eq
    let sel = |nzcv| full(0x1e22_0c20, &[(1, s(1.0)), (2, s(2.0))], 0, nzcv).value;
    assert_eq!(sel(0b0100), s(1.0));
    assert_eq!(sel(0b0000), s(2.0));
    // fccmp d0, d1, #4, ne: condition false keeps the immediate flags.
    let regs = [(0, d(1.0)), (1, d(2.0))];
    assert_eq!(full(0x1e61_1404, &regs, 0, 0b0100).nzcv, 0b0100);
    assert_eq!(full(0x1e61_1404, &regs, 0, 0b0000).nzcv, 0b1000);
}

#[test]
fn precision_conversions() {
    // fcvt d0, s1; fcvt h0, s1; fcvt s0, h1; fcvt s0, d1
    assert_eq!(run(0x1e22_c020, &[(1, s(1.5))]), d(1.5));
    assert_eq!(run(0x1e23_c020, &[(1, s(1.0))]), 0x3C00);
    assert_eq!(run(0x1e23_c020, &[(1, s(65520.0))]), 0x7C00);
    assert_eq!(run(0x1e23_c020, &[(1, s(5.960_464_5e-8))]), 0x0001);
    assert_eq!(run(0x1ee2_4020, &[(1, 0xC000)]), s(-2.0));
    assert_eq!(run(0x1e62_4020, &[(1, d(0.1))]), s(0.1));
    // A signalling double NaN narrows to a quiet single NaN, keeping the
    // top payload bits.
    assert_eq!(run(0x1e62_4020, &[(1, 0x7FF4_0000_0000_0000)]), 0x7FE0_0000);
    // fcvtl v0.2d, v1.2s; fcvtn v0.4h, v1.4s
    assert_eq!(
        run(0x0e61_7820, &[(1, s4([1.0, -2.0, 9.0, 9.0]))]),
        d(1.0) | (d(-2.0) << 64)
    );
    assert_eq!(
        run(0x0e21_6820, &[(1, s4([1.0, 2.0, -1.0, 0.5]))]),
        0x3800_BC00_4000_3C00
    );
}

#[test]
fn estimates_match_the_architected_tables() {
    // frecpe s0, s1 and frsqrte s0, s1 of 1.0 give 0.998046875.
    assert_eq!(run(0x5ea1_d820, &[(1, s(1.0))]), 0x3F7F_8000);
    assert_eq!(run(0x7ea1_d820, &[(1, s(1.0))]), 0x3F7F_8000);
    assert_eq!(run(0x5ea1_d820, &[(1, s(0.0))]), s(f32::INFINITY));
    assert_eq!(run(0x7ea1_d820, &[(1, s(-1.0))]), 0x7FC0_0000);
}

#[test]
fn vector_floating_point() {
    let a = s4([1.0, 2.0, 3.0, 4.0]);
    let b = s4([10.0, 20.0, 30.0, 40.0]);
    // fadd v0.4s, v1.4s, v2.4s
    assert_eq!(
        run(0x4e22_d420, &[(1, a), (2, b)]),
        s4([11.0, 22.0, 33.0, 44.0])
    );
    // fmla v0.4s, v1.4s, v2.s[1] accumulates into v0.
    let acc = s4([1.0, 1.0, 1.0, 1.0]);
    assert_eq!(
        run(0x4fa2_1020, &[(0, acc), (1, a), (2, b)]),
        s4([21.0, 41.0, 61.0, 81.0])
    );
    // faddp v0.4s, v1.4s, v2.4s
    assert_eq!(
        run(0x6e22_d420, &[(1, a), (2, b)]),
        s4([3.0, 7.0, 30.0, 70.0])
    );
    // fmul v0.2d, v1.2d, v2.d[1]
    let pair = |x: f64, y: f64| d(x) | (d(y) << 64);
    assert_eq!(
        run(0x4fc2_9820, &[(1, pair(2.0, 3.0)), (2, pair(0.0, 5.0))]),
        pair(10.0, 15.0)
    );
}

#[test]
fn vector_integer() {
    let a = w4([1, 2, 0xFFFF_FFFF, 4]);
    let b = w4([10, 20, 1, 40]);
    // add v0.4s, v1.4s, v2.4s wraps.
    assert_eq!(run(0x4ea2_8420, &[(1, a), (2, b)]), w4([11, 22, 0, 44]));
    // sqadd v0.8h saturates.
    let h = |v: [u16; 8]| {
        v.iter()
            .rev()
            .fold(0u128, |acc, x| (acc << 16) | u128::from(*x))
    };
    assert_eq!(
        run(
            0x4e62_0c20,
            &[
                (1, h([0x7FFF, 0x8000, 1, 0, 0, 0, 0, 0])),
                (2, h([1, 0xFFFF, 1, 0, 0, 0, 0, 0]))
            ]
        ),
        h([0x7FFF, 0x8000, 2, 0, 0, 0, 0, 0])
    );
    // ushr v0.4s, v1.4s, #3
    assert_eq!(
        run(0x6f3d_0420, &[(1, w4([16, 0xFFFF_FFFF, 7, 8]))]),
        w4([2, 0x1FFF_FFFF, 0, 1])
    );
    // sshll v0.2d, v1.2s, #0 (sxtl)
    assert_eq!(
        run(0x0f20_a420, &[(1, w4([0xFFFF_FFFE, 5, 0, 0]))]),
        u128::from(u64::MAX - 1) | (5 << 64)
    );
    // xtn2 v0.8h, v1.4s keeps the lower half of v0.
    assert_eq!(
        run(
            0x4e61_2820,
            &[
                (0, 0x1111_2222_3333_4444),
                (1, w4([0x1_0001, 2, 3, 0xABCD_0004]))
            ]
        ),
        0x1111_2222_3333_4444 | (0x0004_0003_0002_0001u128 << 64)
    );
    // uzp1, zip2, trn1
    assert_eq!(
        run(0x4e82_1820, &[(1, w4([0, 1, 2, 3])), (2, w4([4, 5, 6, 7]))]),
        w4([0, 2, 4, 6])
    );
    assert_eq!(
        run(0x4e82_7820, &[(1, w4([0, 1, 2, 3])), (2, w4([4, 5, 6, 7]))]),
        w4([2, 6, 3, 7])
    );
    assert_eq!(
        run(
            0x0e02_2820,
            &[(1, 0x0706_0504_0302_0100), (2, 0x1716_1514_1312_1110)]
        ),
        0x1606_1404_1202_1000
    );
    // ext v0.16b, v1.16b, v2.16b, #3
    let bytes = |start: u8| u128::from_le_bytes(std::array::from_fn(|i| start + i as u8));
    assert_eq!(run(0x6e02_1820, &[(1, bytes(0)), (2, bytes(16))]), bytes(3));
    // tbl v0.16b, {v1.16b, v2.16b}, v3.16b: out-of-range indices give 0.
    let indices = u128::from_le_bytes([31, 0, 16, 32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255, 15]);
    let expected = u128::from_le_bytes([31, 0, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 15]);
    assert_eq!(
        run(0x4e03_2020, &[(1, bytes(0)), (2, bytes(16)), (3, indices)]),
        expected
    );
    // cnt v0.8b; addv s0, v1.4s; umaxv b0, v1.16b
    assert_eq!(run(0x0e20_5820, &[(1, 0xFF00_0F03)]), 0x0800_0402);
    assert_eq!(run(0x4eb1_b820, &[(1, w4([1, 2, 3, 0xFFFF_FFFF]))]), 5);
    assert_eq!(run(0x6e30_a820, &[(1, bytes(100))]), 115);
    // cmeq v0.4s, v1.4s, #0
    assert_eq!(
        run(0x4ea0_9820, &[(1, w4([0, 1, 0, 2]))]),
        w4([u32::MAX, 0, u32::MAX, 0])
    );
    // mul v0.4s, v1.4s, v2.s[2]
    assert_eq!(
        run(0x4f82_8820, &[(1, w4([1, 2, 3, 4])), (2, w4([0, 0, 7, 0]))]),
        w4([7, 14, 21, 28])
    );
    // smull v0.2d, v1.2s, v2.2s
    assert_eq!(
        run(
            0x0ea2_c020,
            &[
                (1, w4([0xFFFF_FFFF, 3, 0, 0])),
                (2, w4([5, 0x8000_0000, 0, 0]))
            ]
        ),
        u128::from((-5i64) as u64) | (u128::from((-(3i64 << 31)) as u64) << 64)
    );
    // umlal2 v0.2d, v1.4s, v2.4s uses the upper halves.
    assert_eq!(
        run(
            0x6ea2_8020,
            &[
                (0, 1 | (2 << 64)),
                (1, w4([0, 0, 3, 0xFFFF_FFFF])),
                (2, w4([0, 0, 4, 2]))
            ]
        ),
        13 | (u128::from(0x1_FFFF_FFFEu64 + 2) << 64)
    );
    // mov s1, v0.s[1]
    assert_eq!(run(0x5e0c_0401, &[(0, w4([1, 2, 3, 4]))]), 2);
    // rev64 v0.16b
    assert_eq!(
        run(0x4e20_0820, &[(1, bytes(0))]),
        u128::from_le_bytes([7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8])
    );
    // sshl v0.2d, v1.2d, v2.2d: negative shifts go right, arithmetically.
    assert_eq!(
        run(
            0x4ee2_4420,
            &[
                (1, u128::from(u64::MAX - 7) | (1 << 64)),
                (2, u128::from(0xFEu64) | (3 << 64))
            ]
        ),
        u128::from(u64::MAX - 1) | (8 << 64)
    );
    // srshr v0.4s, v1.4s, #4 rounds.
    assert_eq!(
        run(0x4f3c_2420, &[(1, w4([8, 7, 0xFFFF_FFF8, 24]))]),
        w4([1, 0, 0, 2])
    );
    // sqrshrn s0, d1, #8 saturates.
    assert_eq!(
        run(0x5f38_9c20, &[(1, u128::from(0x007F_FFFF_FFFF_u64))]),
        0x7FFF_FFFF
    );
}

#[test]
fn aes_128_encrypts_the_fips_197_block() {
    // aese v0.16b, v1.16b / aesmc v0.16b, v0.16b
    let (aese, aesmc) = (0x4e28_4820, 0x4e28_6800);
    let key: [u8; 16] = std::array::from_fn(|i| i as u8);
    let plaintext: [u8; 16] = std::array::from_fn(|i| (i as u8) * 0x11);
    // Key expansion with SubWord from AESE: with four equal columns
    // ShiftRows is a no-op and AESE with a zero key is SubBytes.
    let sub_word = |w: u32| run(aese, &[(0, w4([w; 4])), (1, 0)]) as u32;
    let mut schedule: Vec<u32> = (0..4)
        .map(|i| u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap()))
        .collect();
    let mut rcon = 1u32;
    for i in 4..44 {
        let mut t = schedule[i - 1];
        if i % 4 == 0 {
            t = sub_word(t.rotate_right(8)) ^ rcon;
            rcon = if rcon & 0x80 != 0 {
                (rcon << 1) ^ 0x11B
            } else {
                rcon << 1
            };
        }
        schedule.push(schedule[i - 4] ^ t);
    }
    let round_key = |r: usize| w4(schedule[4 * r..4 * r + 4].try_into().unwrap());
    let mut state = u128::from_le_bytes(plaintext);
    for r in 0..9 {
        state = run(aese, &[(0, state), (1, round_key(r))]);
        state = run(aesmc, &[(0, state)]);
    }
    state = run(aese, &[(0, state), (1, round_key(9))]) ^ round_key(10);
    let expected = [
        0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5,
        0x5a,
    ];
    assert_eq!(state.to_le_bytes(), expected);
}

/// The single padded block of "abc", as big-endian words.
fn abc_block() -> [u128; 4] {
    let mut w = [0u32; 16];
    w[0] = 0x6162_6380;
    w[15] = 24;
    std::array::from_fn(|g| w4(w[4 * g..4 * g + 4].try_into().unwrap()))
}

fn add4(a: u128, b: u128) -> u128 {
    let (a, b) = (a.to_le_bytes(), b.to_le_bytes());
    let word = |x: &[u8; 16], i: usize| u32::from_le_bytes(x[4 * i..4 * i + 4].try_into().unwrap());
    w4(std::array::from_fn(|i| {
        word(&a, i).wrapping_add(word(&b, i))
    }))
}

#[test]
fn sha256_hashes_abc() {
    let (h, h2, su0, su1) = (0x5e02_4020, 0x5e02_5020, 0x5e28_2820, 0x5e02_6020);
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let initial = [
        w4([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a]),
        w4([0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]),
    ];
    let (mut abcd, mut efgh) = (initial[0], initial[1]);
    let mut msg = abc_block();
    for i in 0..16 {
        let wk = add4(msg[i % 4], w4(K[4 * i..4 * i + 4].try_into().unwrap()));
        let saved = abcd;
        abcd = run(h, &[(0, abcd), (1, efgh), (2, wk)]);
        efgh = run(h2, &[(0, efgh), (1, saved), (2, wk)]);
        if i < 12 {
            let t = run(su0, &[(0, msg[i % 4]), (1, msg[(i + 1) % 4])]);
            msg[i % 4] = run(su1, &[(0, t), (1, msg[(i + 2) % 4]), (2, msg[(i + 3) % 4])]);
        }
    }
    assert_eq!(
        add4(abcd, initial[0]),
        w4([0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223])
    );
    assert_eq!(
        add4(efgh, initial[1]),
        w4([0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad])
    );
}

#[test]
fn sha1_hashes_abc() {
    let (c, p, m, sha1h, su0, su1) = (
        0x5e02_0020,
        0x5e02_1020,
        0x5e02_2020,
        0x5e28_0820,
        0x5e02_3020,
        0x5e28_1820,
    );
    let initial = w4([0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476]);
    let initial_e = 0xc3d2e1f0u32;
    let k = [0x5a827999u32, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6];
    let mut abcd = initial;
    let mut e = initial_e;
    let mut msg = abc_block();
    for i in 0..20 {
        let wk = add4(msg[i % 4], w4([k[i / 5]; 4]));
        let next_e = run(sha1h, &[(1, abcd)]) as u32;
        let rounds = [c, p, m, p][i / 5];
        abcd = run(rounds, &[(0, abcd), (1, u128::from(e)), (2, wk)]);
        e = next_e;
        if i < 16 {
            let t = run(
                su0,
                &[
                    (0, msg[i % 4]),
                    (1, msg[(i + 1) % 4]),
                    (2, msg[(i + 2) % 4]),
                ],
            );
            msg[i % 4] = run(su1, &[(0, t), (1, msg[(i + 3) % 4])]);
        }
    }
    assert_eq!(
        add4(abcd, initial),
        w4([0xa9993e36, 0x4706816a, 0xba3e2571, 0x7850c26c])
    );
    assert_eq!(e.wrapping_add(initial_e), 0x9cd0d89d);
}

// ---------------------------------------------------------------------------
// The native fast paths. These tests compare the public evaluator against a
// straightforward model written here, against the scalar form of the same
// instruction, and against the specialized calls generated code uses.
//
// The evaluator models FPCR.{DN,FZ,AHP} = 0 and RMode = RN, so FPCR.FZ is
// exercised by toggling the host FTZ/DAZ bits (which the runtime mirrors from
// it) in lib.rs's `fast_paths_hold_with_flush_to_zero`; the FRINT and FCVT
// variants below cover the rounding modes that appear in encodings.

/// Element `index` of `esize` bits.
fn lane(v: u128, esize: u32, index: u32) -> u64 {
    let mask = if esize == 64 {
        u64::MAX
    } else {
        (1 << esize) - 1
    };
    ((v >> (index * esize)) as u64) & mask
}

fn pack(values: &[u64], esize: u32) -> u128 {
    values.iter().enumerate().fold(0, |acc, (i, value)| {
        acc | (u128::from(*value) << (i as u32 * esize))
    })
}

fn f32_bits(value: f32) -> u64 {
    u64::from(value.to_bits())
}

fn f64_bits(value: f64) -> u64 {
    value.to_bits()
}

const QUIET32: u32 = 0x0040_0000;
const QUIET64: u64 = 0x0008_0000_0000_0000;
const DEFAULT_NAN32: u64 = 0x7FC0_0000;
const DEFAULT_NAN64: u64 = 0x7FF8_0000_0000_0000;

fn is_nan32(raw: u32) -> bool {
    raw & 0x7F80_0000 == 0x7F80_0000 && raw & 0x007F_FFFF != 0
}

fn is_nan64(raw: u64) -> bool {
    raw & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000 && raw & 0x000F_FFFF_FFFF_FFFF != 0
}

/// Ordinary values, signed zeros, infinities, quiet and signalling NaNs,
/// denormals, and rounding boundaries.
fn f32_corpus() -> Vec<u32> {
    vec![
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
        0x4020_0000,
        0xC020_0000,
        0x4000_0000,
        0xC000_0000,
        0x4040_0000,
        0xC040_0000,
        0x477F_FFFF,
        0x4B00_0001,
        0x4F00_0000,
        0xCF00_0000,
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
}

fn f64_corpus() -> Vec<u64> {
    vec![
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
        0x4004_0000_0000_0000,
        0xC004_0000_0000_0000,
        0x4000_0000_0000_0000,
        0xC000_0000_0000_0000,
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
}

/// A specialized call must reproduce `exec` for every register file.
fn check_specialized(word: u32, vector: &[(u8, u128)], general: u64) {
    assert!(
        recomp_simd::fast::specialize(word).is_some(),
        "{word:#010x} is not specialized"
    );
    check_optional_specialized(word, vector, general);
}

/// The same check for forms that may still take `exec` directly (FCVTN,
/// FCVTL, FCVTXN): when a specialized call exists it must agree.
fn check_optional_specialized(word: u32, vector: &[(u8, u128)], general: u64) {
    let expected = full(word, vector, general, 0);
    let Some(call) = recomp_simd::fast::specialize(word) else {
        return;
    };
    let file = |reg: u8| {
        vector
            .iter()
            .find(|(r, _)| *r == reg)
            .map(|(_, v)| *v)
            .unwrap_or(0)
    };
    let got = recomp_simd::fast::run(&call, &file, general, 0);
    assert_eq!(
        got, expected.value,
        "{word:#010x} ({}) specialized call",
        call.function
    );
}

fn scalar_fp1_model(op: &str, raw: u32) -> u32 {
    match op {
        "abs" => return raw & 0x7FFF_FFFF,
        "neg" => return raw ^ 0x8000_0000,
        _ => {}
    }
    if is_nan32(raw) {
        return raw | QUIET32;
    }
    let value = f32::from_bits(raw);
    match op {
        "sqrt" => {
            let result = value.sqrt();
            if result.is_nan() {
                DEFAULT_NAN32 as u32
            } else {
                result.to_bits()
            }
        }
        "n" => value.round_ties_even().to_bits(),
        "p" => value.ceil().to_bits(),
        "m" => value.floor().to_bits(),
        "z" => value.trunc().to_bits(),
        "a" => value.round().to_bits(),
        _ => unreachable!(),
    }
}

fn f64_fp1_model(op: &str, raw: u64) -> u64 {
    match op {
        "abs" => return raw & 0x7FFF_FFFF_FFFF_FFFF,
        "neg" => return raw ^ 0x8000_0000_0000_0000,
        _ => {}
    }
    if is_nan64(raw) {
        return raw | QUIET64;
    }
    let value = f64::from_bits(raw);
    match op {
        "sqrt" => {
            let result = value.sqrt();
            if result.is_nan() {
                DEFAULT_NAN64
            } else {
                result.to_bits()
            }
        }
        "n" => value.round_ties_even().to_bits(),
        "p" => value.ceil().to_bits(),
        "m" => value.floor().to_bits(),
        "z" => value.trunc().to_bits(),
        "a" => value.round().to_bits(),
        _ => unreachable!(),
    }
}

#[test]
fn one_source_fast_paths_match_the_model() {
    // (instruction, model operation, lane width)
    let scalar32 = [
        (0x1E20_C020u32, "abs"),
        (0x1E21_4020, "neg"),
        (0x1E21_C020, "sqrt"),
        (0x1E24_4020, "n"),
        (0x1E24_C020, "p"),
        (0x1E25_4020, "m"),
        (0x1E25_C020, "z"),
        (0x1E26_4020, "a"),
    ];
    let scalar64 = [
        (0x1E60_C020u32, "abs"),
        (0x1E61_4020, "neg"),
        (0x1E61_C020, "sqrt"),
        (0x1E64_4020, "n"),
        (0x1E64_C020, "p"),
        (0x1E65_4020, "m"),
        (0x1E65_C020, "z"),
        (0x1E66_4020, "a"),
    ];
    let vector32 = [
        (0x0EA0_F820u32, "abs", 2),
        (0x2EA0_F820, "neg", 2),
        (0x2EA1_F820, "sqrt", 2),
        (0x4EA0_F820, "abs", 4),
        (0x6EA0_F820, "neg", 4),
        (0x6EA1_F820, "sqrt", 4),
        (0x4E21_8820, "n", 4),
        (0x4EA1_8820, "p", 4),
        (0x4E21_9820, "m", 4),
        (0x4EA1_9820, "z", 4),
        (0x6E21_8820, "a", 4),
        (0x6E21_9820, "n", 4),
        (0x6EA1_9820, "n", 4),
    ];
    let vector64 = [
        (0x4EE0_F820u32, "abs", 2),
        (0x6EE0_F820, "neg", 2),
        (0x6EE1_F820, "sqrt", 2),
        (0x4E61_8820, "n", 2),
        (0x4EE1_8820, "p", 2),
        (0x4E61_9820, "m", 2),
        (0x4EE1_9820, "z", 2),
        (0x6E61_8820, "a", 2),
        (0x6E61_9820, "n", 2),
        (0x6EE1_9820, "n", 2),
    ];
    let values = f32_corpus();
    for &raw in &values {
        for &(word, op) in &scalar32 {
            let got = run(word, &[(1, u128::from(raw))]);
            let expected = u128::from(u64::from(scalar_fp1_model(op, raw)));
            assert_eq!(got, expected, "{word:#010x} {op} {raw:#010x}");
            check_specialized(word, &[(1, u128::from(raw))], 0);
        }
        for &(word, op, lanes) in &vector32 {
            let input = pack(
                &values
                    .iter()
                    .skip((raw as usize) % values.len())
                    .chain(values.iter())
                    .take(lanes)
                    .map(|v| u64::from(*v))
                    .collect::<Vec<_>>(),
                32,
            );
            let got = run(word, &[(1, input)]);
            let expected = pack(
                &(0..lanes)
                    .map(|i| u64::from(scalar_fp1_model(op, lane(input, 32, i as u32) as u32)))
                    .collect::<Vec<_>>(),
                32,
            );
            assert_eq!(got, expected, "{word:#010x} {op}");
            check_specialized(word, &[(1, input)], 0);
        }
    }
    let values = f64_corpus();
    for &raw in &values {
        for &(word, op) in &scalar64 {
            let got = run(word, &[(1, u128::from(raw))]);
            let expected = u128::from(f64_fp1_model(op, raw));
            assert_eq!(got, expected, "{word:#010x} {op} {raw:#018x}");
            check_specialized(word, &[(1, u128::from(raw))], 0);
        }
        for &(word, op, lanes) in &vector64 {
            let input = pack(
                &values
                    .iter()
                    .skip((raw as usize) % values.len())
                    .chain(values.iter())
                    .take(lanes)
                    .copied()
                    .collect::<Vec<_>>(),
                64,
            );
            let got = run(word, &[(1, input)]);
            let expected = pack(
                &(0..lanes)
                    .map(|i| f64_fp1_model(op, lane(input, 64, i as u32)))
                    .collect::<Vec<_>>(),
                64,
            );
            assert_eq!(got, expected, "{word:#010x} {op}");
            check_specialized(word, &[(1, input)], 0);
        }
    }
}

/// A widening conversion reads `lanes` elements starting at `first` in the
/// low half and writes lanes of twice the size.
fn widen_lanes_match_scalar(
    vector_word: u32,
    scalar_word: u32,
    from_esize: u32,
    first: u32,
    lanes: u32,
    n: u128,
) {
    let got = run(vector_word, &[(1, n)]);
    for i in 0..lanes {
        let source = lane(n, from_esize, first + i);
        let expected = run(scalar_word, &[(1, u128::from(source))]);
        assert_eq!(
            lane(got, from_esize * 2, i),
            lane(expected, from_esize * 2, 0),
            "{vector_word:#010x} lane {i} against {scalar_word:#010x}"
        );
    }
    check_optional_specialized(vector_word, &[(1, n)], 0);
}

/// A narrowing conversion reads `lanes` elements from the low half and writes
/// half-size elements at `offset`.
fn narrow_lanes_match_scalar(
    vector_word: u32,
    scalar_word: u32,
    from_esize: u32,
    lanes: u32,
    offset: u32,
    n: u128,
) {
    let got = run(vector_word, &[(1, n)]);
    for i in 0..lanes {
        let source = lane(n, from_esize, i);
        let expected = run(scalar_word, &[(1, u128::from(source))]);
        assert_eq!(
            lane(got, from_esize / 2, offset + i),
            lane(expected, from_esize / 2, 0),
            "{vector_word:#010x} lane {i} against {scalar_word:#010x}"
        );
    }
    check_optional_specialized(vector_word, &[(1, n)], 0);
}

/// A vector SCVTF/UCVTF reads integer lanes; its scalar form reads a general
/// register.
fn from_int_lanes_match_scalar(vector_word: u32, scalar_word: u32, esize: u32, n: u128) {
    let lanes = 128 / esize;
    let got = run(vector_word, &[(1, n)]);
    for i in 0..lanes {
        let expected = full(scalar_word, &[], lane(n, esize, i), 0).value;
        assert_eq!(
            lane(got, esize, i),
            lane(expected, esize, 0),
            "{vector_word:#010x} lane {i} against {scalar_word:#010x}"
        );
    }
    check_optional_specialized(vector_word, &[(1, n)], 0);
}

/// A vector FCVT[NSAU]... reads floating-point lanes; its scalar form writes a
/// general register.
fn to_int_lanes_match_scalar(vector_word: u32, scalar_word: u32, esize: u32, n: u128) {
    let lanes = 128 / esize;
    let got = run(vector_word, &[(1, n)]);
    for i in 0..lanes {
        let expected = full(scalar_word, &[(1, u128::from(lane(n, esize, i)))], 0, 0).value;
        assert_eq!(
            lane(got, esize, i),
            lane(expected, esize, 0),
            "{vector_word:#010x} lane {i} against {scalar_word:#010x}"
        );
    }
    check_optional_specialized(vector_word, &[(1, n)], 0);
}

fn pair_lanes_match_scalar(vector_word: u32, scalar_word: u32, esize: u32, n: u128, m: u128) {
    let lanes = 128 / esize;
    let got = run(vector_word, &[(1, n), (2, m)]);
    for i in 0..lanes {
        let expected = run(
            scalar_word,
            &[
                (1, u128::from(lane(n, esize, i))),
                (2, u128::from(lane(m, esize, i))),
            ],
        );
        assert_eq!(
            lane(got, esize, i),
            lane(expected, esize, 0),
            "{vector_word:#010x} lane {i} against {scalar_word:#010x}"
        );
    }
    check_specialized(vector_word, &[(1, n), (2, m)], 0);
}

#[test]
fn two_source_fast_paths_match_the_scalar_evaluator() {
    let values = f32_corpus();
    let scalar32 = [
        (0x1E22_2820u32, 0x4E22_D420, "add"),
        (0x1E22_3820, 0x4EA2_D420, "sub"),
        (0x1E22_0820, 0x6E22_DC20, "mul"),
        (0x1E22_1820, 0x6E22_FC20, "div"),
        (0x1E22_4820, 0x4E22_F420, "max"),
        (0x1E22_5820, 0x4EA2_F420, "min"),
        (0x1E22_6820, 0x4E22_C420, "maxnm"),
        (0x1E22_7820, 0x4EA2_C420, "minnm"),
        (0x7EA2_D420, 0x6EA2_D420, "abd"),
        (0x5E22_FC20, 0x4E22_FC20, "recps"),
        (0x5EA2_FC20, 0x4EA2_FC20, "rsqrts"),
    ];
    for &a in &values {
        for &b in &values {
            let n = u128::from(a)
                | (u128::from(b) << 32)
                | (u128::from(a) << 64)
                | (u128::from(b) << 96);
            let m = u128::from(b)
                | (u128::from(a) << 32)
                | (u128::from(b) << 64)
                | (u128::from(a) << 96);
            for &(scalar, vector, op) in &scalar32 {
                let _ = op;
                pair_lanes_match_scalar(vector, scalar, 32, n, m);
            }
        }
    }
    // The same operations on two double-precision lanes.
    let values = f64_corpus();
    let scalar64 = [
        (0x1E62_2820u32, 0x4E62_D420, "add"),
        (0x1E62_1820, 0x6E62_FC20, "div"),
        (0x1E62_4820, 0x4E62_F420, "max"),
        (0x1E62_7820, 0x4EE2_C420, "minnm"),
    ];
    for &a in &values {
        for &b in &values {
            let n = u128::from(a) | (u128::from(b) << 64);
            let m = u128::from(b) | (u128::from(a) << 64);
            for &(scalar, vector, op) in &scalar64 {
                let _ = op;
                pair_lanes_match_scalar(vector, scalar, 64, n, m);
            }
        }
    }
}

#[test]
fn two_source_fast_paths_keep_the_special_cases() {
    let inf = f32_bits(f32::INFINITY);
    // FMAX and FMIN choose the positive zero for equal values; FMAXNM keeps a
    // number over a quiet NaN, while FMAX propagates the NaN.
    assert_eq!(run(0x1E22_4820, &[(1, s(-0.0)), (2, s(0.0))]), s(0.0));
    assert_eq!(run(0x1E22_5820, &[(1, s(0.0)), (2, s(-0.0))]), s(-0.0));
    assert_eq!(run(0x1E22_6820, &[(1, 0x7FC0_1234), (2, s(1.0))]), s(1.0));
    assert_eq!(
        run(0x1E22_4820, &[(1, 0x7FC0_1234), (2, s(1.0))]),
        0x7FC0_1234
    );
    // A signalling NaN wins over a quiet one and comes back quiet.
    assert_eq!(
        run(0x1E22_2820, &[(1, 0x7FC0_0002), (2, 0xFF80_0001)]),
        0xFFC0_0001
    );
    // FRECPS and FRSQRTS have an architected infinity-times-zero result.
    assert_eq!(
        run(0x5E22_FC20, &[(1, u128::from(inf)), (2, s(0.0))]),
        s(2.0)
    );
    assert_eq!(
        run(0x5EA2_FC20, &[(1, u128::from(inf)), (2, s(0.0))]),
        s(1.5)
    );
    // Ordinary values are correctly rounded on the host.
    assert_eq!(run(0x1E22_2820, &[(1, s(1.0)), (2, s(2.0))]), s(3.0));
    assert_eq!(run(0x1E22_1820, &[(1, s(1.0)), (2, s(3.0))]), s(1.0 / 3.0));
    // FADD of two denormals is not flushed with FPCR.FZ = 0 (the mode the
    // evaluator models).
    assert_eq!(
        run(0x1E22_2820, &[(1, u128::from(1u64)), (2, u128::from(1u64))]),
        u128::from(2u64)
    );
    // But an overflowing sum still becomes infinity.
    assert_eq!(
        run(0x1E22_2820, &[(1, s(f32::MAX)), (2, s(f32::MAX))]),
        s(f32::INFINITY)
    );
}

#[test]
fn one_source_fast_paths_keep_the_special_cases() {
    // FSQRT of a denormal is normal and must not be flushed: the smallest
    // subnormal's square root is about 3.7e-23.
    let root = run(0x1E21_C020, &[(1, u128::from(1u64))]);
    assert_eq!(root, u128::from(f32_bits(f32::from_bits(1).sqrt())));
    assert_ne!(root, 0);
    // A negative input is the invalid operation's default NaN.
    assert_eq!(run(0x1E21_C020, &[(1, s(-1.0))]), u128::from(DEFAULT_NAN32));
    // A signalling NaN comes back quiet with its payload.
    assert_eq!(
        run(0x1E21_C020, &[(1, u128::from(0x7F80_0001u64))]),
        u128::from(0x7FC0_0001u64)
    );
    // Every FRINT variant preserves the sign of a zero result.
    assert_eq!(run(0x1E24_4020, &[(1, s(2.5))]), s(2.0));
    assert_eq!(run(0x1E26_4020, &[(1, s(2.5))]), s(3.0));
    assert_eq!(run(0x1E25_4020, &[(1, s(-1.5))]), s(-2.0));
    assert_eq!(run(0x1E25_C020, &[(1, s(-1.5))]), s(-1.0));
    assert_eq!(run(0x1E24_C020, &[(1, s(-0.5))]), s(-0.0));
    assert_eq!(run(0x1E25_4020, &[(1, s(0.5))]), s(0.0));
    assert_eq!(run(0x1E24_4020, &[(1, s(-0.5))]), s(-0.0));
    assert_eq!(run(0x1E24_4020, &[(1, s(f32::INFINITY))]), s(f32::INFINITY));
    assert_eq!(
        run(0x1E24_4020, &[(1, u128::from(0x7FC0_1234u64))]),
        u128::from(0x7FC0_1234u64)
    );
}

#[test]
fn conversion_fast_paths_match_the_scalar_evaluator() {
    let values = f32_corpus();
    // FCVT d0, s1 / FCVT s0, d1 round trip every ordinary value exactly.
    for &raw in &values {
        let widened = run(0x1E22_C020, &[(1, u128::from(raw))]);
        let narrowed = run(0x1E62_4020, &[(1, widened)]);
        let expected = if is_nan32(raw) {
            u128::from(raw | QUIET32)
        } else {
            u128::from(raw)
        };
        assert_eq!(narrowed, expected, "fcvt round trip {raw:#010x}");
    }
    // FCVTL and FCVTN agree with the scalar FCVT forms lane by lane.
    let pair = pack(&[u64::from(values[8]), u64::from(values[9])], 32);
    let quad = pack(
        &values[10..14]
            .iter()
            .map(|v| u64::from(*v))
            .collect::<Vec<_>>(),
        32,
    );
    widen_lanes_match_scalar(0x0E61_7820, 0x1E22_C020, 32, 0, 2, pair);
    widen_lanes_match_scalar(0x4E61_7820, 0x1E22_C020, 32, 2, 2, quad);
    narrow_lanes_match_scalar(0x0E21_6820, 0x1E23_C020, 32, 4, 0, quad);
    // FCVTN2 keeps the destination's low half and writes the upper half.
    let got = run(0x4E21_6820, &[(0, 0), (1, quad)]);
    for i in 0..4 {
        let expected = run(0x1E23_C020, &[(1, u128::from(lane(quad, 32, i)))]);
        assert_eq!(lane(got, 16, 4 + i), lane(expected, 16, 0));
    }
    // FCVTXN uses round-to-odd and matches the scalar form.
    let doubles = pack(
        &[
            f64::from_bits(0x3FF0_0000_0000_0001),
            f64::from_bits(0x4004_0000_0000_0001),
        ]
        .map(f64_bits),
        64,
    );
    narrow_lanes_match_scalar(0x2E61_6820, 0x7E61_6820, 64, 2, 0, doubles);
    // The fixed-point forms match their scalar counterparts too.
    let fixed_ints = pack(&[0, 1, 0x0001_0000, 0xFFFF_0000], 32);
    let fixed_wide =
        u128::from(0x7FFF_FFFF_FFFF_FFFFu64) | (u128::from(0xFFFF_FFFF_FFFF_0000u64) << 64);
    to_int_lanes_match_scalar(0x6F2F_FC20, 0x1E19_BC20, 32, quad);
    from_int_lanes_match_scalar(0x4F30_E420, 0x1E02_C020, 32, fixed_ints);
    to_int_lanes_match_scalar(0x4F78_FC20, 0x9E58_E020, 64, doubles);
    from_int_lanes_match_scalar(0x4F78_E420, 0x9E42_E020, 64, fixed_wide);

    // Rounding-mode variants of FCVT to integer.
    for &(scalar, vector) in &[
        (0x1E38_0020u32, 0x4EA1_B820u32), // fcvtzs
        (0x1E39_0020, 0x6EA1_B820),       // fcvtzu
        (0x1E24_0020, 0x4E21_C820),       // fcvtas
        (0x1E30_0020, 0x4E21_B820),       // fcvtms
        (0x1E28_0020, 0x4EA1_A820),       // fcvtps
        (0x1E20_0020, 0x4E21_A820),       // fcvtns
    ] {
        to_int_lanes_match_scalar(vector, scalar, 32, quad);
    }
    for &(scalar, vector) in &[(0x9E78_0020u32, 0x4EE1_B820u32), (0x9E79_0020, 0x6EE1_B820)] {
        to_int_lanes_match_scalar(vector, scalar, 64, doubles);
    }

    // SCVTF and UCVTF from general registers, scalar and vector.
    let integers = [
        0u32,
        1,
        2,
        3,
        0x7FFF_FFFF,
        0x8000_0000,
        0xFFFF_FFFF,
        0x0000_8000,
    ];
    let vector_input = pack(
        &integers[..4]
            .iter()
            .map(|v| u64::from(*v))
            .collect::<Vec<_>>(),
        32,
    );
    for &(scalar, vector) in &[(0x1E22_0020u32, 0x4E21_D820u32), (0x1E23_0020, 0x6E21_D820)] {
        from_int_lanes_match_scalar(vector, scalar, 32, vector_input);
    }
    // The same conversions with double precision and 64-bit integers.
    let wide = [
        0u64,
        1,
        2,
        3,
        0x7FFF_FFFF,
        0x8000_0000,
        0xFFFF_FFFF,
        0x1_0000_0000,
        0x7FFF_FFFF_FFFF_FFFF,
        0x8000_0000_0000_0000,
        0xFFFF_FFFF_FFFF_FFFF,
    ];
    let pair = u128::from(wide[8]) | (u128::from(wide[9]) << 64);
    for &(scalar, vector) in &[(0x9E62_0020u32, 0x4E61_D820u32), (0x9E63_0020, 0x6E61_D820)] {
        from_int_lanes_match_scalar(vector, scalar, 64, pair);
    }
}

#[test]
fn conversion_fast_paths_keep_the_special_cases() {
    // NaN converts to zero in every rounding mode.
    for word in [
        0x1E38_0020u32, // fcvtzs w0, s1
        0x1E39_0020,    // fcvtzu w0, s1
        0x1E24_0020,    // fcvtas w0, s1
        0x1E30_0020,    // fcvtms w0, s1
        0x1E28_0020,    // fcvtps w0, s1
        0x1E20_0020,    // fcvtns w0, s1
    ] {
        assert_eq!(full(word, &[(1, s(f32::NAN))], 0, 0).value, 0);
    }
    // Infinity and out-of-range values saturate, and negative values clamp to
    // zero for the unsigned forms.
    assert_eq!(
        full(0x1E38_0020, &[(1, s(f32::INFINITY))], 0, 0).value,
        0x7FFF_FFFF
    );
    assert_eq!(
        full(0x1E38_0020, &[(1, s(f32::NEG_INFINITY))], 0, 0).value,
        0x8000_0000
    );
    assert_eq!(full(0x1E39_0020, &[(1, s(-1.0))], 0, 0).value, 0);
    assert_eq!(
        full(0x1E39_0020, &[(1, s(f32::INFINITY))], 0, 0).value,
        0xFFFF_FFFF
    );
    // Ties go to even for FCVTN[SAU], away for FCVTAS, down for FCVTMS, up
    // for FCVTPS.
    assert_eq!(full(0x1E20_0020, &[(1, s(2.5))], 0, 0).value, 2);
    assert_eq!(full(0x1E20_0020, &[(1, s(3.5))], 0, 0).value, 4);
    assert_eq!(full(0x1E24_0020, &[(1, s(2.5))], 0, 0).value, 3);
    assert_eq!(
        full(0x1E30_0020, &[(1, s(-1.5))], 0, 0).value,
        u128::from((-2i32) as u32 as u64)
    );
    assert_eq!(
        full(0x1E28_0020, &[(1, s(-1.5))], 0, 0).value,
        u128::from((-1i32) as u32 as u64)
    );
    // A denormal source is not flushed with FPCR.FZ = 0; the smallest one
    // rounds up to 1 under FCVTPS and to 0 under FCVTNS.
    assert_eq!(full(0x1E28_0020, &[(1, u128::from(1u64))], 0, 0).value, 1);
    assert_eq!(full(0x1E20_0020, &[(1, u128::from(1u64))], 0, 0).value, 0);
    // FCVT widening of a binary32 denormal is exact and does not flush.
    let widened = run(0x1E22_C020, &[(1, u128::from(1u64))]);
    assert_eq!(widened, u128::from(f64_bits(f64::from(f32::from_bits(1)))));
    assert_ne!(widened, 0);
    // FCVTN of a value that lands in the binary16 denormal range keeps it.
    // 1.5 units of 2^-24 rounds to even: 2.
    let tiny = f32_bits(2f32.powi(-24) * 1.5);
    assert_eq!(run(0x1E23_C020, &[(1, u128::from(tiny))]), u128::from(2u64));
    // Half NaN payloads are quietened and kept, widened to binary32.
    assert_eq!(
        run(0x1EE2_4020, &[(1, u128::from(0x7C01u64))]),
        u128::from(0x7FC0_2000u64)
    );
}
