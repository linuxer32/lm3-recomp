//! `RECOMP_SVC_STATS`: per-SVC call counts for finding the hot dispatcher
//! entries. Off unless the environment variable is set; the only cost when
//! off is one cached `OnceLock` read per SVC.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// The interval between reports.
const REPORT_INTERVAL: Duration = Duration::from_secs(10);
/// How many ids each report lists.
const REPORT_TOP: usize = 12;

/// Every kernel SVC id is below this (the numeric table has 0x7e as the
/// largest entry), so a flat array needs no lock.
const MAX_ID: usize = 128;

static COUNTS: [AtomicU64; MAX_ID] = [const { AtomicU64::new(0) }; MAX_ID];

/// Count one dispatch. `id` outside the kernel table is ignored.
#[inline]
pub(crate) fn record(id: u64) {
    if !enabled() {
        return;
    }
    if let Some(counter) = COUNTS.get(id as usize) {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// Whether `RECOMP_SVC_STATS` enables collection; the first enabled call
/// starts the reporter. `0`, `false` and `off` keep it disabled, so a
/// script can export the variable unconditionally.
#[inline]
fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = match std::env::var("RECOMP_SVC_STATS") {
            Ok(value) => !matches!(value.trim(), "" | "0" | "false" | "off"),
            Err(_) => false,
        };
        if enabled {
            spawn_reporter();
        }
        enabled
    })
}

fn snapshot() -> [u64; MAX_ID] {
    std::array::from_fn(|index| COUNTS[index].load(Ordering::Relaxed))
}

/// Print the top ids by calls in the window, with the rate and share, every
/// `REPORT_INTERVAL`. An idle interval still prints a zero line, so a run
/// cannot look like stats are missing.
fn spawn_reporter() {
    let _ = std::thread::Builder::new()
        .name("recomp-svc-stats".to_string())
        .spawn(|| {
            eprintln!(
                "[svc-stats] enabled: reporting the top {REPORT_TOP} kernel SVC ids every {}s",
                REPORT_INTERVAL.as_secs()
            );
            let mut previous = snapshot();
            loop {
                let started = Instant::now();
                std::thread::sleep(REPORT_INTERVAL);
                let current = snapshot();
                eprintln!("{}", report(&previous, &current, started.elapsed()));
                previous = current;
            }
        });
}

/// The report body, split out so it is testable without the environment.
fn report(previous: &[u64; MAX_ID], current: &[u64; MAX_ID], elapsed: Duration) -> String {
    let mut entries: Vec<(u64, u64)> = current
        .iter()
        .zip(previous)
        .enumerate()
        .filter_map(|(id, (now, before))| {
            let calls = now.saturating_sub(*before);
            (calls > 0).then_some((id as u64, calls))
        })
        .collect();
    entries.sort_by(|(left_id, left), (right_id, right)| {
        right.cmp(left).then_with(|| left_id.cmp(right_id))
    });
    let total: u64 = entries.iter().map(|(_, calls)| calls).sum();
    let seconds = elapsed.as_secs_f64().max(f64::EPSILON);

    let mut out = String::new();
    let _ = writeln!(
        out,
        "[svc-stats] {:.1}s: {total} SVCs ({:.0}/s)",
        elapsed.as_secs_f64(),
        total as f64 / seconds
    );
    for (id, calls) in entries.into_iter().take(REPORT_TOP) {
        let name = recomp_services::kernel_svc_name(id).unwrap_or("svc_unknown");
        let _ = writeln!(
            out,
            "[svc-stats]   0x{id:02x} {name:<32} {calls:>12} {:>6.2}% {:>10.0}/s",
            calls as f64 * 100.0 / total as f64,
            calls as f64 / seconds
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_sorts_by_calls_and_names_each_id() {
        let previous = [0; MAX_ID];
        let mut current = [0; MAX_ID];
        current[0x0B] = 900;
        current[0x11] = 100;
        current[0x1E] = 0;
        let text = report(&previous, &current, Duration::from_secs(10));
        assert!(text.contains("1000 SVCs"), "{text}");
        let sleep = text.find("svcSleepThread").expect("sleep listed");
        let signal = text.find("svcSignalEvent").expect("signal listed");
        assert!(sleep < signal, "higher count first:\n{text}");
        assert!(!text.contains("svcGetSystemTick"), "{text}");
        assert!(text.contains("90.00%"), "{text}");
    }

    #[test]
    fn report_uses_the_window_delta_not_the_totals() {
        let mut previous = [0; MAX_ID];
        let mut current = [0; MAX_ID];
        previous[0x0B] = 5000;
        current[0x0B] = 5120;
        let text = report(&previous, &current, Duration::from_secs(10));
        assert!(text.contains("120 SVCs"), "{text}");
        assert!(text.contains("12/s"), "{text}");
    }
}
