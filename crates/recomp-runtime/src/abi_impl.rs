//! Definitions of the `__recomp_rt_*` and `__recomp_simd_*` symbols declared
//! by `recomp-abi`.
//! Generated code reaches the runtime only through these, so their
//! signatures must match the declarations there exactly.

use recomp_abi::{Fault, GuestResult, SvcOutcome};

#[no_mangle]
pub fn __recomp_rt_mem_load_u8(address: u64) -> Result<u64, Fault> {
    Ok(crate::mem_load_u8(address)?)
}

#[no_mangle]
pub fn __recomp_rt_mem_load_u16(address: u64) -> Result<u64, Fault> {
    Ok(crate::mem_load_u16(address)?)
}

#[no_mangle]
pub fn __recomp_rt_mem_load_u32(address: u64) -> Result<u64, Fault> {
    Ok(crate::mem_load_u32(address)?)
}

#[no_mangle]
pub fn __recomp_rt_mem_load_u64(address: u64) -> Result<u64, Fault> {
    Ok(crate::mem_load_u64(address)?)
}

#[no_mangle]
pub fn __recomp_rt_mem_store_u8(address: u64, value: u64) -> GuestResult {
    crate::watch_store(address, 1, value);
    Ok(crate::mem_store_u8(address, value)?)
}

#[no_mangle]
pub fn __recomp_rt_mem_store_u16(address: u64, value: u64) -> GuestResult {
    crate::watch_store(address, 2, value);
    Ok(crate::mem_store_u16(address, value)?)
}

#[no_mangle]
pub fn __recomp_rt_mem_store_u32(address: u64, value: u64) -> GuestResult {
    crate::watch_store(address, 4, value);
    Ok(crate::mem_store_u32(address, value)?)
}

#[no_mangle]
pub fn __recomp_rt_mem_store_u64(address: u64, value: u64) -> GuestResult {
    crate::watch_store(address, 8, value);
    Ok(crate::mem_store_u64(address, value)?)
}

/// LDXR, LDAXR and the pair forms: the value(s) read, and the reservation a
/// store-exclusive compares against.
#[no_mangle]
pub fn __recomp_rt_mem_load_exclusive(
    address: u64,
    size: u64,
    pair: bool,
) -> Result<(u64, u64), Fault> {
    Ok(crate::mem_load_exclusive(address, size, pair)?)
}

/// STXR, STLXR and the pair forms. The result is the success flag the
/// instruction returns: 0 when the store happened, 1 when the reservation was
/// lost to another thread.
#[no_mangle]
pub fn __recomp_rt_mem_store_exclusive(
    address: u64,
    size: u64,
    pair: bool,
    value: u64,
    second_value: u64,
) -> Result<u64, Fault> {
    crate::watch_store(address, size, value);
    Ok(crate::mem_store_exclusive(
        address,
        size,
        pair,
        value,
        second_value,
    )?)
}

/// CLREX.
#[no_mangle]
pub fn __recomp_rt_clear_exclusive() {
    crate::clear_exclusive();
}

/// DMB: guest threads run on host threads, so a barrier is a real fence.
#[no_mangle]
pub fn __recomp_rt_memory_fence() {
    crate::memory_fence();
}

#[no_mangle]
pub fn __recomp_rt_dispatch(address: u64) -> GuestResult {
    crate::dispatch(address)
}

#[no_mangle]
pub fn __recomp_rt_count_calls(count: u32) {
    crate::threads::count_calls(count)
}

#[no_mangle]
pub fn __recomp_rt_svc(name: &str, args: &[i64]) -> Result<SvcOutcome, Fault> {
    Ok(crate::svc_dispatch(name, args)?)
}

#[no_mangle]
pub fn __recomp_rt_syscall_stub(behavior: &str, name: &str, args: &[i64]) -> GuestResult {
    let result = match behavior {
        "log" => crate::syscall_log(name, args),
        "noop" => crate::syscall_noop(name, args),
        _ => crate::syscall_panic(name, args),
    };
    Ok(result?)
}

#[no_mangle]
pub fn __recomp_rt_sysreg_read(reg: u32) -> u64 {
    crate::sysreg_read(reg)
}

#[no_mangle]
pub fn __recomp_rt_sysreg_write(reg: u32, value: u64) {
    crate::sysreg_write(reg, value)
}

#[no_mangle]
#[cold]
pub fn __recomp_rt_loop_preempt() {
    crate::threads::loop_point();
}

#[no_mangle]
pub fn __recomp_simd_exec(
    word: u32,
    vector: &[(u8, u128)],
    general: u64,
    nzcv: u8,
) -> Option<recomp_abi::simd::Outcome> {
    if simd_stats::enabled() {
        simd_stats::count("exec", word);
    }
    recomp_simd::exec(word, vector, general, nzcv).map(|outcome| recomp_abi::simd::Outcome {
        value: outcome.value,
        nzcv: outcome.nzcv,
    })
}

/// The `__recomp_simd_*` symbols for `recomp_abi::simd::fast`: the
/// evaluator is linked here, so changing it only relinks a program.
macro_rules! simd_fast {
    ($(fn $symbol:ident($($arg:ident: $ty:ty),*) -> $ret:ty = $function:ident;)*) => {
        $(
            #[no_mangle]
            #[allow(clippy::too_many_arguments)]
            pub fn $symbol($($arg: $ty),*) -> $ret {
                if simd_stats::enabled() {
                    // The call's small arguments (operation, sizes), a
                    // byte each.
                    let mut key = 0u32;
                    let mut shift = 0;
                    $(
                        if let Some(small) = simd_stats::Small::small(&$arg) {
                            if shift < 32 {
                                key |= (small & 0xFF) << shift;
                            }
                            shift += 8;
                        }
                    )*
                    let _ = shift;
                    simd_stats::count(stringify!($function), key);
                }
                recomp_simd::fast::$function($($arg),*)
            }
        )*
    };
}

simd_fast! {
    fn __recomp_simd_scalar_fp2(op: u32, double: bool, n: u128, m: u128) -> u128 = scalar_fp2;
    fn __recomp_simd_scalar_fma(negate: u32, double: bool, a: u128, n: u128, m: u128) -> u128 = scalar_fma;
    fn __recomp_simd_scalar_compare(double: bool, n: u128, m: u128) -> u8 = scalar_compare;
    fn __recomp_simd_scalar_select(double: bool, cond: u32, nzcv: u8, n: u128, m: u128) -> u128 = scalar_select;
    fn __recomp_simd_scalar_fp1(opcode: u32, ftype: u32, n: u128) -> u128 = scalar_fp1;
    fn __recomp_simd_constant(value: u128) -> u128 = constant;
    fn __recomp_simd_low_bits(width: u32, n: u128) -> u128 = low_bits;
    fn __recomp_simd_from_general(width: u32, x: u64) -> u128 = from_general;
    fn __recomp_simd_int_to_float(double: bool, unsigned: bool, wide: bool, x: u64) -> u128 = int_to_float;
    fn __recomp_simd_float_to_int(double: bool, mode: u32, wide: bool, n: u128) -> u128 = float_to_int;
    fn __recomp_simd_vector3(kind: u32, double: bool, q: bool, d: u128, n: u128, m: u128) -> u128 = vector3;
    fn __recomp_simd_vector_element(op: u32, double: bool, q: bool, scalar: bool, index: u32, d: u128, n: u128, m: u128) -> u128 = vector_element;
    fn __recomp_simd_extract(q: bool, position: u32, n: u128, m: u128) -> u128 = extract;
    fn __recomp_simd_element(esize: u32, index: u32, n: u128) -> u128 = element;
    fn __recomp_simd_int_three(u: bool, opcode: u32, esize: u32, lanes: u32, d: u128, n: u128, m: u128) -> u128 = int_three;
    fn __recomp_simd_vector_bitwise(kind: u32, q: bool, d: u128, n: u128, m: u128) -> u128 = vector_bitwise;
    fn __recomp_simd_vector_rev(container: u32, esize: u32, q: bool, n: u128) -> u128 = vector_rev;
    fn __recomp_simd_vector_pairwise_sum(u: bool, accumulate: bool, esize: u32, lanes: u32, d: u128, n: u128) -> u128 = vector_pairwise_sum;
    fn __recomp_simd_vector_int2(kind: u32, esize: u32, lanes: u32, d: u128, n: u128) -> u128 = vector_int2;
    fn __recomp_simd_vector_narrow(kind: u32, esize: u32, mode: u32, lanes: u32, n: u128, d: u128) -> u128 = vector_narrow;
    fn __recomp_simd_vector_shift_long(q: bool, esize: u32, n: u128) -> u128 = vector_shift_long;
    fn __recomp_simd_vector_fp1(op: u32, esize: u32, lanes: u32, n: u128) -> u128 = vector_fp1;
    fn __recomp_simd_vector_fp_cmp0(cmp: u32, esize: u32, lanes: u32, n: u128) -> u128 = vector_fp_cmp0;
    fn __recomp_simd_vector_to_int(rounding: u32, unsigned: bool, esize: u32, lanes: u32, n: u128) -> u128 = vector_to_int;
    fn __recomp_simd_vector_from_int(unsigned: bool, esize: u32, lanes: u32, n: u128) -> u128 = vector_from_int;
    fn __recomp_simd_vector_across(kind: u32, esize: u32, lanes: u32, n: u128) -> u128 = vector_across;
    fn __recomp_simd_vector_fixed_convert(to_float: bool, unsigned: bool, esize: u32, lanes: u32, fbits: u32, n: u128) -> u128 = vector_fixed_convert;
    fn __recomp_simd_vector_shift(u: bool, opcode: u32, esize: u32, lanes: u32, imm: u32, d: u128, n: u128) -> u128 = vector_shift;
    fn __recomp_simd_scalar_fccmp(double: bool, cond: u32, nzcv: u8, n: u128, m: u128, immediate: u32) -> u8 = scalar_fccmp;
}

/// `RECOMP_SIMD_STATS=1`: how often generated code reaches the evaluator,
/// per operation and its small arguments (per instruction word for
/// `exec`), printed every 10 s: what the inline paths in
/// `recomp_abi::simd::fast` should cover next.
mod simd_stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    const SLOTS: usize = 1 << 13;
    const NAMES: [&str; 32] = [
        "exec",
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
        "?", // anything else unknown
    ];
    static KEYS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static COUNTS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];

    /// An argument that names the operation (a field or flag), not data.
    pub trait Small {
        fn small(&self) -> Option<u32>;
    }
    impl Small for u32 {
        fn small(&self) -> Option<u32> {
            Some(*self)
        }
    }
    impl Small for bool {
        fn small(&self) -> Option<u32> {
            Some(u32::from(*self))
        }
    }
    impl Small for u8 {
        fn small(&self) -> Option<u32> {
            None
        }
    }
    impl Small for u64 {
        fn small(&self) -> Option<u32> {
            None
        }
    }
    impl Small for u128 {
        fn small(&self) -> Option<u32> {
            None
        }
    }

    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let on = std::env::var_os("RECOMP_SIMD_STATS").is_some();
            if on {
                let _ = std::thread::Builder::new()
                    .name("recomp-simd-stats".to_string())
                    .spawn(report);
            }
            on
        })
    }

    pub fn count(name: &str, key: u32) {
        let operation = NAMES
            .iter()
            .position(|known| *known == name)
            .unwrap_or(NAMES.len() - 1) as u64;
        let id = ((operation << 32) | u64::from(key)) + 1;
        let mut slot = (id.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 51) as usize;
        for _ in 0..64 {
            match KEYS[slot].compare_exchange(0, id, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => {}
                Err(found) if found == id => {}
                Err(_) => {
                    slot = (slot + 1) % SLOTS;
                    continue;
                }
            }
            COUNTS[slot].fetch_add(1, Ordering::Relaxed);
            return;
        }
    }

    fn report() {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            let mut rows: Vec<(u64, u64)> = (0..SLOTS)
                .filter_map(|slot| {
                    let count = COUNTS[slot].swap(0, Ordering::Relaxed);
                    (count > 0).then(|| (count, KEYS[slot].load(Ordering::Relaxed) - 1))
                })
                .collect();
            rows.sort_unstable_by(|a, b| b.cmp(a));
            let total: u64 = rows.iter().map(|row| row.0).sum();
            println!("[recomp-simd] {total} evaluator calls in 10 s; most:");
            for (count, id) in rows.iter().take(40) {
                println!(
                    "[recomp-simd] {count:>10} {} {:#x}",
                    NAMES[(id >> 32) as usize],
                    id & 0xFFFF_FFFF
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// Values that exercise every case: zeros, infinities, NaNs, denormals,
    /// the largest and smallest normals, and random bits.
    fn values(seed: &mut u64) -> u128 {
        let mut next = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        };
        const SPECIAL: [u32; 10] = [
            0,
            0x8000_0000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0000,
            0x7F80_0001,
            0x0000_0001,
            0x007F_FFFF,
            0x7F7F_FFFF,
            0x3F80_0000,
        ];
        const SPECIAL64: [u64; 6] = [
            0,
            0x7FF0_0000_0000_0000,
            0x7FF8_0000_0000_0000,
            0x0000_0000_0000_0001,
            0x3FF0_0000_0000_0000,
            0x7FEF_FFFF_FFFF_FFFF,
        ];
        let mut value = 0u128;
        for lane in 0..4 {
            let pick = next();
            let bits = match pick % 4 {
                0 => SPECIAL[(pick >> 8) as usize % SPECIAL.len()],
                // Ordinary values near 1, where results are not NaN.
                1 => 0x3F00_0000 | (pick >> 12) as u32 & 0x00FF_FFFF,
                _ => (pick >> 16) as u32,
            };
            value |= u128::from(bits) << (32 * lane);
        }
        if next() % 3 == 0 {
            value = (value & !u128::from(u64::MAX))
                | u128::from(SPECIAL64[next() as usize % SPECIAL64.len()]);
        }
        value
    }

    #[test]
    fn inline_simd_paths_match_the_evaluator() {
        use recomp_abi::simd::fast as inline;
        use recomp_simd::fast as exact;
        let mut seed = 0x2545_F491_4F6C_DD1D;
        let kinds: Vec<u32> = (0..128).collect();
        for _ in 0..20_000 {
            let (d, n, m) = (values(&mut seed), values(&mut seed), values(&mut seed));
            for width in [32, 64] {
                assert_eq!(inline::low_bits(width, n), exact::low_bits(width, n));
                assert_eq!(
                    inline::from_general(width, n as u64),
                    exact::from_general(width, n as u64)
                );
            }
            for (esize, lanes) in [(8, 16), (16, 8), (32, 4), (64, 2)] {
                for index in 0..lanes {
                    assert_eq!(
                        inline::element(esize, index, n),
                        exact::element(esize, index, n),
                        "element {esize} {index}"
                    );
                }
            }
            for double in [false, true] {
                for op in [0, 1, 2, 3, 4, 5, 6, 7, 8] {
                    assert_eq!(
                        inline::scalar_fp2(op, double, n, m),
                        exact::scalar_fp2(op, double, n, m),
                        "scalar_fp2 {op} {double} {n:#x} {m:#x}"
                    );
                }
                for negate in 0..4 {
                    assert_eq!(
                        inline::scalar_fma(negate, double, d, n, m),
                        exact::scalar_fma(negate, double, d, n, m),
                        "scalar_fma {negate} {double} {d:#x} {n:#x} {m:#x}"
                    );
                }
                assert_eq!(
                    inline::scalar_compare(double, n, m),
                    exact::scalar_compare(double, n, m),
                    "scalar_compare {double} {n:#x} {m:#x}"
                );
                let nzcv = (d & 0xF) as u8;
                for cond in 0..16 {
                    assert_eq!(
                        inline::scalar_select(double, cond, nzcv, n, m),
                        exact::scalar_select(double, cond, nzcv, n, m),
                        "scalar_select {double} {cond} {nzcv}"
                    );
                }
                let ftype = u32::from(double);
                for opcode in 0..16 {
                    assert_eq!(
                        inline::scalar_fp1(opcode, ftype, n),
                        exact::scalar_fp1(opcode, ftype, n),
                        "scalar_fp1 {opcode} {ftype} {n:#x}"
                    );
                }
                for unsigned in [false, true] {
                    for wide in [false, true] {
                        let x = n as u64;
                        assert_eq!(
                            inline::int_to_float(double, unsigned, wide, x),
                            exact::int_to_float(double, unsigned, wide, x),
                            "int_to_float {double} {unsigned} {wide} {x:#x}"
                        );
                    }
                }
                for rmode in 0..4 {
                    for opcode in [0, 1, 4, 5] {
                        for wide in [false, true] {
                            let mode = (rmode << 3) | opcode;
                            assert_eq!(
                                inline::float_to_int(double, mode, wide, n),
                                exact::float_to_int(double, mode, wide, n),
                                "float_to_int {double} {mode:#x} {wide} {n:#x}"
                            );
                        }
                    }
                }
                for q in [false, true] {
                    // 64-bit lanes need a 128-bit vector (q).
                    if double && !q {
                        continue;
                    }
                    for kind in &kinds {
                        assert_eq!(
                            inline::vector3(*kind, double, q, d, n, m),
                            exact::vector3(*kind, double, q, d, n, m),
                            "vector3 {kind:#x} {double} {q} {d:#x} {n:#x} {m:#x}"
                        );
                    }
                    for position in 0..if q { 16 } else { 8 } {
                        assert_eq!(
                            inline::extract(q, position, n, m),
                            exact::extract(q, position, n, m),
                            "extract {q} {position}"
                        );
                    }
                    for esize in [8, 16, 32, 64] {
                        if esize == 64 && !q {
                            continue;
                        }
                        for op in [1, 2, 3, 5, 6, 7] {
                            assert_eq!(
                                inline::permute(op, esize, q, n, m),
                                exact::permute(op, esize, q, n, m),
                                "permute {op} {esize} {q}"
                            );
                        }
                    }
                    for op in 0..4 {
                        for scalar in [false, true] {
                            for index in 0..if double { 2 } else { 4 } {
                                assert_eq!(
                                    inline::vector_element(op, double, q, scalar, index, d, n, m),
                                    exact::vector_element(op, double, q, scalar, index, d, n, m),
                                    "vector_element {op} {double} {q} {scalar} {index}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The new specializations are linked through `recomp_abi::simd::fast`'s
    /// symbols, so a mismatch between the declarations and the runtime would
    /// otherwise only fail when a generated chunk calls one.
    #[test]
    fn the_new_specialized_symbols_are_registered() {
        use recomp_abi::simd::fast as inline;
        use recomp_simd::fast as exact;
        let d = 0x0123_4567_89AB_CDEFu128;
        let n = 0x3F80_0000_4000_0000_7FC0_0001_8000_0000u128;
        let m = 0x4000_0000_3F80_0000_7FF0_0000_0000_0001u128;
        for u in [false, true] {
            for opcode in [0b01110, 0b10011, 0b10100, 0b10111] {
                assert_eq!(
                    inline::int_three(u, opcode, 32, 4, d, n, m),
                    exact::int_three(u, opcode, 32, 4, d, n, m)
                );
            }
            for kind in 0..8 {
                assert_eq!(
                    inline::vector_bitwise(kind, true, d, n, m),
                    exact::vector_bitwise(kind, true, d, n, m)
                );
            }
            assert_eq!(
                inline::vector_pairwise_sum(u, true, 16, 4, d, n),
                exact::vector_pairwise_sum(u, true, 16, 4, d, n)
            );
            assert_eq!(
                inline::vector_from_int(u, 32, 4, n),
                exact::vector_from_int(u, 32, 4, n)
            );
            assert_eq!(
                inline::vector_to_int(2, u, 64, 2, n),
                exact::vector_to_int(2, u, 64, 2, n)
            );
            assert_eq!(
                inline::vector_fixed_convert(u, u, 32, 4, 5, n),
                exact::vector_fixed_convert(u, u, 32, 4, 5, n)
            );
            assert_eq!(
                inline::vector_shift(u, 0b00100, 16, 8, 30, d, n),
                exact::vector_shift(u, 0b00100, 16, 8, 30, d, n)
            );
        }
        for container in [64, 32, 16] {
            assert_eq!(
                inline::vector_rev(container, 8, true, n),
                exact::vector_rev(container, 8, true, n)
            );
        }
        for kind in 0..4 {
            assert_eq!(
                inline::vector_int2(kind, 16, 8, d, n),
                exact::vector_int2(kind, 16, 8, d, n)
            );
        }
        for kind in 0..4 {
            for mode in 0..3 {
                assert_eq!(
                    inline::vector_narrow(kind, 16, mode, 4, n, d),
                    exact::vector_narrow(kind, 16, mode, 4, n, d)
                );
            }
        }
        for q in [false, true] {
            assert_eq!(
                inline::vector_shift_long(q, 16, n),
                exact::vector_shift_long(q, 16, n)
            );
        }
        for op in 0..10 {
            assert_eq!(
                inline::vector_fp1(op, 32, 4, n),
                exact::vector_fp1(op, 32, 4, n)
            );
        }
        for cmp in 0..5 {
            assert_eq!(
                inline::vector_fp_cmp0(cmp, 32, 4, n),
                exact::vector_fp_cmp0(cmp, 32, 4, n)
            );
        }
        for kind in [0b00011, 0b01010, 0b11010, 0b11011, 0b11011 | 0x20] {
            assert_eq!(
                inline::vector_across(kind, 16, 8, n),
                exact::vector_across(kind, 16, 8, n)
            );
        }
        for double in [false, true] {
            for cond in 0..16 {
                assert_eq!(
                    inline::scalar_fccmp(double, cond, 6, n, m, 8),
                    exact::scalar_fccmp(double, cond, 6, n, m, 8)
                );
            }
        }
    }
}
