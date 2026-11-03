//! `RECOMP_PROFILE=<file>`: a sampling profiler for finding where a
//! recompiled program spends its time. Every millisecond of process CPU
//! time, SIGPROF interrupts the running thread and records its host program
//! counter and thread id. `RECOMP_PROFILE_AT=<seconds>` starts sampling that
//! long after launch (to skip loading). A thread rewrites the file every
//! 2 s: a `base <hex>` line (where the executable is loaded), a
//! `map <start> <end> <path>` line per executable mapping of a library, then
//! one `<pc offset hex> <tid>` line per sample, offsets relative to that
//! base (`x <pc hex> <tid>` for code outside the executable: libraries and
//! drivers).
//! `scripts/profile-report.py <executable> <file>` names the functions.
//!
//! Only Linux and Android implement the SIGPROF sampler and
//! `/proc/self/maps` reader. Elsewhere `RECOMP_PROFILE` is accepted and
//! reported as unsupported, so startup is never blocked by profiling.
//!
//! On Android the code lives in `librecomp_game.so`, so its samples are the
//! `x` lines of that library's mapping (the executable is `app_process`);
//! `scripts/profile-report.py` names them from the library's symbols.

#[cfg(any(target_os = "linux", target_os = "android"))]
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

#[cfg(target_os = "linux")]
use libc::ITIMER_PROF;
/// bionic's `ITIMER_PROF`, which the libc crate does not define for Android.
#[cfg(target_os = "android")]
const ITIMER_PROF: libc::c_int = 2;

/// Samples kept: about 35 minutes of one busy thread.
#[cfg(any(target_os = "linux", target_os = "android"))]
const CAPACITY: usize = 1 << 21;

#[cfg(any(target_os = "linux", target_os = "android"))]
static PCS: [AtomicU64; CAPACITY] = [const { AtomicU64::new(0) }; CAPACITY];
#[cfg(any(target_os = "linux", target_os = "android"))]
static TIDS: [AtomicU32; CAPACITY] = [const { AtomicU32::new(0) }; CAPACITY];
#[cfg(any(target_os = "linux", target_os = "android"))]
static COUNT: AtomicUsize = AtomicUsize::new(0);

/// The interrupted thread's program counter.
#[cfg(any(target_os = "linux", target_os = "android"))]
unsafe fn interrupted_pc(context: *mut libc::c_void) -> u64 {
    let context = unsafe { &*context.cast::<libc::ucontext_t>() };
    #[cfg(target_arch = "x86_64")]
    {
        context.uc_mcontext.gregs[libc::REG_RIP as usize] as u64
    }
    #[cfg(target_arch = "aarch64")]
    {
        context.uc_mcontext.pc as u64
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = context;
        0
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
extern "C" fn on_sample(_: libc::c_int, _: *mut libc::siginfo_t, context: *mut libc::c_void) {
    let index = COUNT.fetch_add(1, Ordering::Relaxed);
    if index >= CAPACITY {
        return;
    }
    // SAFETY: the kernel passes the interrupted thread's context.
    let pc = unsafe { interrupted_pc(context) };
    PCS[index].store(pc, Ordering::Relaxed);
    // SAFETY: gettid has no preconditions and is async-signal-safe.
    TIDS[index].store(unsafe { libc::gettid() } as u32, Ordering::Relaxed);
}

/// Start sampling when `RECOMP_PROFILE` asks for it.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn start() {
    let Some(path) = std::env::var_os("RECOMP_PROFILE") else {
        return;
    };
    let delay = std::env::var("RECOMP_PROFILE_AT")
        .ok()
        .and_then(|seconds| seconds.parse::<f64>().ok())
        .unwrap_or(0.0);
    let _ = std::thread::Builder::new()
        .name("recomp-profile".to_string())
        .spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs_f64(delay));
            // SAFETY: the handler only touches atomics and calls gettid.
            unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = on_sample as *const () as usize;
                action.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
                libc::sigemptyset(&mut action.sa_mask);
                libc::sigaction(libc::SIGPROF, &action, std::ptr::null_mut());
                let period = libc::timeval {
                    tv_sec: 0,
                    tv_usec: 1000,
                };
                let timer = libc::itimerval {
                    it_interval: period,
                    it_value: period,
                };
                libc::setitimer(ITIMER_PROF, &timer, std::ptr::null_mut());
            }
            println!("[recomp-runtime] profiling into {}", path.to_string_lossy());
            let (base, end) = executable_range().unwrap_or((0, u64::MAX));
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let count = COUNT.load(Ordering::Relaxed).min(CAPACITY);
                let mut text = format!("base {base:#x}\n");
                for (start, end, library) in libraries() {
                    text.push_str(&format!("map {start:x} {end:x} {library}\n"));
                }
                for index in 0..count {
                    let pc = PCS[index].load(Ordering::Relaxed);
                    let tid = TIDS[index].load(Ordering::Relaxed);
                    if (base..end).contains(&pc) {
                        text.push_str(&format!("{:x} {tid}\n", pc - base));
                    } else {
                        text.push_str(&format!("x {pc:x} {tid}\n"));
                    }
                }
                let _ = std::fs::write(&path, text);
            }
        });
}

/// Profiling is implemented on Linux only; a request is reported once at
/// startup rather than blocking the title.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) fn start() {
    if std::env::var_os("RECOMP_PROFILE").is_some() {
        eprintln!("[recomp-runtime] profiling is not implemented on this platform; RECOMP_PROFILE is ignored");
    }
}

/// The executable mappings of libraries (`/proc/self/maps`).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn libraries() -> Vec<(u64, u64, String)> {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    maps.lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let span = fields.next()?;
            if !fields.next()?.contains('x') {
                return None;
            }
            let path = fields.nth(3)?;
            let (start, end) = span.split_once('-')?;
            Some((
                u64::from_str_radix(start, 16).ok()?,
                u64::from_str_radix(end, 16).ok()?,
                path.to_string(),
            ))
        })
        .collect()
}

/// Where the executable's own mappings start and end (`/proc/self/maps`).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn executable_range() -> Option<(u64, u64)> {
    let exe = std::env::current_exe().ok()?;
    let exe = exe.to_string_lossy();
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    let mut range: Option<(u64, u64)> = None;
    for line in maps.lines().filter(|line| line.ends_with(exe.as_ref())) {
        let (span, _) = line.split_once(' ')?;
        let (start, end) = span.split_once('-')?;
        let (start, end) = (
            u64::from_str_radix(start, 16).ok()?,
            u64::from_str_radix(end, 16).ok()?,
        );
        range = Some(range.map_or((start, end), |(low, high)| (low.min(start), high.max(end))));
    }
    range
}
