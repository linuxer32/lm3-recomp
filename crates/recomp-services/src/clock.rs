//! The console's clocks, from one start point shared by every thread: the
//! system counter (CNTPCT_EL0, svcGetSystemTick) at Horizon's 19.2 MHz, and
//! nanoseconds for what services timestamp (buffer queue history, GPU
//! reports). The SDK converts between the two (`nn::os::ConvertToTick`), so
//! they must count from the same moment.
//!
//! A title reads the system counter very often (`NvOsGetTimeUS` polls it),
//! so the clock does not go through `clock_gettime`/vDSO on x86-64: it reads
//! the invariant TSC once calibrated and scales it with one multiply. Both
//! still count from the same start.

use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// The system counter's frequency (CNTFRQ_EL0).
pub const TICKS_PER_SECOND: u64 = 19_200_000;

struct Clock {
    start: Instant,
    /// The TSC at `start`, or zero when the host has no usable TSC.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    start_tsc: u64,
    /// Nanoseconds per TSC cycle, scaled by `2^32`.
    ns_scale: u64,
    /// Microseconds per TSC cycle, scaled by `2^32`.
    us_scale: u64,
}

/// The calibrated clock. `CLOCK_PTR` publishes it for the hot path below.
static CLOCK: OnceLock<Clock> = OnceLock::new();
/// The `CLOCK` value once initialized, or null. Reading this pointer costs one
/// atomic load; the `OnceLock` state check stays on the one-time slow path, so
/// a title polling `NvOsGetTimeUS` does not pay it on every call.
static CLOCK_PTR: AtomicPtr<Clock> = AtomicPtr::new(std::ptr::null_mut());

#[inline(always)]
fn clock() -> &'static Clock {
    let pointer = CLOCK_PTR.load(Ordering::Acquire);
    if !pointer.is_null() {
        // SAFETY: `clock_slow` publishes the pointer only after `CLOCK` has
        // finished initializing. `Clock` is a static and is never moved or
        // dropped, so the reference stays valid for the program's lifetime.
        return unsafe { &*pointer };
    }
    clock_slow()
}

#[cold]
fn clock_slow() -> &'static Clock {
    let clock = CLOCK.get_or_init(calibrate);
    CLOCK_PTR.store(clock as *const Clock as *mut Clock, Ordering::Release);
    clock
}

#[cold]
fn calibrate() -> Clock {
    let start = Instant::now();
    #[cfg(target_arch = "x86_64")]
    {
        // An invariant TSC ticks at a constant rate and is synced
        // across cores, so it can stand in for the vDSO clock. Measure
        // it once against the monotonic clock; the error is far below
        // the millisecond timing a title cares about.
        let start_tsc = unsafe { core::arch::x86_64::_rdtsc() };
        std::thread::sleep(std::time::Duration::from_millis(20));
        let end_tsc = unsafe { core::arch::x86_64::_rdtsc() };
        let elapsed = start.elapsed().as_nanos();
        if invariant_tsc() && end_tsc > start_tsc && elapsed > 0 {
            let cycles = u128::from(end_tsc - start_tsc);
            let hz = cycles * 1_000_000_000 / elapsed;
            let ns_scale = (1u128 << 32) * 1_000_000_000 / hz;
            let us_scale = (1u128 << 32) * 1_000_000 / hz;
            return Clock {
                start,
                start_tsc,
                ns_scale: u64::try_from(ns_scale).unwrap_or(0),
                us_scale: u64::try_from(us_scale).unwrap_or(0),
            };
        }
    }
    Clock {
        start,
        start_tsc: 0,
        ns_scale: 0,
        us_scale: 0,
    }
}

/// Whether the CPU advertises an invariant, synchronized TSC (CPUID
/// 80000007H:EDX[8]).
#[cfg(target_arch = "x86_64")]
fn invariant_tsc() -> bool {
    let leaf = core::arch::x86_64::__cpuid(0x8000_0007);
    leaf.edx & (1 << 8) != 0
}

/// Nanoseconds since the clock started.
#[inline]
pub fn now_ns() -> u64 {
    let clock = clock();
    if clock.ns_scale != 0 {
        #[cfg(target_arch = "x86_64")]
        {
            let now = unsafe { core::arch::x86_64::_rdtsc() };
            let cycles = u128::from(now.wrapping_sub(clock.start_tsc));
            return ((cycles * u128::from(clock.ns_scale)) >> 32) as u64;
        }
    }
    clock.start.elapsed().as_nanos() as u64
}

/// Microseconds since the clock started, for titles that poll the clock
/// very often (`NvOsGetTimeUS`); scales the TSC directly instead of dividing
/// nanoseconds.
#[inline(always)]
pub fn now_us() -> u64 {
    let clock = clock();
    #[cfg(target_arch = "x86_64")]
    {
        if clock.us_scale != 0 {
            let now = unsafe { core::arch::x86_64::_rdtsc() };
            let cycles = u128::from(now.wrapping_sub(clock.start_tsc));
            return ((cycles * u128::from(clock.us_scale)) >> 32) as u64;
        }
    }
    clock.start.elapsed().as_micros() as u64
}

/// System counter ticks since the clock started.
#[inline]
pub fn ticks() -> u64 {
    ns_to_ticks(now_ns())
}

/// Nanoseconds to 19.2 MHz ticks: `ns * 192 / 10_000`, as one 64-bit
/// multiply (a year of nanoseconds still fits).
#[inline]
pub fn ns_to_ticks(ns: u64) -> u64 {
    ns / 10_000 * 192 + (ns % 10_000) * 192 / 10_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_run_at_19_2_mhz_from_the_shared_start() {
        assert_eq!(ns_to_ticks(1_000_000_000), TICKS_PER_SECOND);
        assert_eq!(ns_to_ticks(16_666_667), 320_000);
        let (ns, ticks) = (now_ns(), ticks());
        // Read just after, the counter is within a millisecond of the
        // nanosecond clock, scaled.
        assert!(ticks >= ns_to_ticks(ns) && ticks - ns_to_ticks(ns) < 19_200);
    }

    #[test]
    fn microseconds_track_nanoseconds() {
        let (ns, us) = (now_ns(), now_us());
        let us_ns = us * 1000;
        assert!(
            us_ns.abs_diff(ns) < 1_000_000,
            "ns {ns} vs us {us} (scaled {us_ns})"
        );
    }

    #[test]
    fn the_calibrated_clock_is_cached() {
        let first = clock() as *const Clock;
        for _ in 0..1_000 {
            assert_eq!(clock() as *const Clock, first);
        }
        assert_eq!(CLOCK_PTR.load(Ordering::Acquire), first as *mut Clock);
    }

    #[test]
    fn nanosecond_clock_is_monotonic() {
        let mut last = now_ns();
        for _ in 0..100_000 {
            let next = now_ns();
            assert!(next >= last, "clock went backwards: {last} then {next}");
            last = next;
        }
    }

    #[test]
    fn microsecond_clock_is_monotonic() {
        let mut last = now_us();
        for _ in 0..100_000 {
            let next = now_us();
            assert!(next >= last, "clock went backwards: {last} then {next}");
            last = next;
        }
    }

    #[test]
    fn ns_to_ticks_matches_the_exact_ratio() {
        for ns in [
            0,
            1,
            9999,
            10_000,
            1_000_000,
            1_666_666_667,
            86_400_000_000_000,
        ] {
            assert_eq!(
                ns_to_ticks(ns),
                (u128::from(ns) * u128::from(TICKS_PER_SECOND) / 1_000_000_000) as u64
            );
        }
    }
}
