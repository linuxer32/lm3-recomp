//! Guest memory as one host reservation ("fastmem").
//!
//! The guest's 39-bit address space is reserved once (inaccessible, no
//! memory committed), and every guest region is mapped at reservation +
//! guest address. Generated code then loads and stores with one host access
//! at that address (`recomp_abi::set_memory_range`) instead of calling the
//! runtime. An access to a guest address with nothing mapped hits an
//! inaccessible page; the fault handler reports it with the guest address
//! and block, as the checked path does, and stops the process.
//!
//! Horizon pages are 4 KiB, but host pages are not: Android 15 kernels use
//! 16 KiB pages (and some hosts use 64 KiB). The host page size is detected
//! at runtime and all commits, protections and host addresses are aligned
//! to it. A host page can hold several guest pages, so mapping one guest
//! page makes its neighbours in that host page accessible too. Guest
//! offsets stay 1:1, so loads and stores are still correct; the extra
//! bytes read as zero, and a later region claims and zeroes its own guest
//! pages. The checked path keeps the exact 4 KiB semantics.
//!
//! A 512 GiB reservation does not fit in a 39-bit user address space
//! (common on Android). The runtime then reserves only the window the
//! title's memory layout reaches (`reserve_for`): generated code keeps the
//! direct path below the window's end and the checked path above it, so a
//! guest title with a few GiB of address space runs at full speed on a
//! phone instead of checking every load and store. Only when even that
//! window cannot be reserved does every access stay checked (`Pages::Heap`).
//! It never rebases guest memory, so an address is either mapped at its
//! guest address or rejected.
//!
//! `RECOMP_FASTMEM=0` keeps every access on the checked path, as does
//! `RECOMP_WATCH_STORE` (which has to see every store).
//! `RECOMP_FASTMEM_GIB=<n>` forces a window of `n` GiB, which exercises the
//! bounded path on a host that could reserve the whole space.

use std::collections::BTreeMap;
use std::ops::Range;
#[cfg(unix)]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
#[cfg(unix)]
use std::sync::OnceLock;

/// The guest address space the reservation covers.
pub(crate) const SPACE: u64 = 1 << 39;
/// Horizon's page size, used by the guest memory layout and the checked path.
pub(crate) const PAGE: u64 = 0x1000;
/// A host page larger than this is treated as unsupported (the checked
/// path takes over) rather than trusted in the arithmetic below.
#[cfg(unix)]
const MAX_HOST_PAGE: u64 = 1 << 20;

/// Host address of guest address 0, for the fault handler.
static BASE: AtomicUsize = AtomicUsize::new(0);
/// End (guest address, exclusive) of the range the reservation covers.
#[cfg(unix)]
static COVERED: AtomicU64 = AtomicU64::new(0);
/// The largest layout end `reserve_for` was told about; a host that cannot
/// reserve the whole space reserves this much instead.
#[cfg(unix)]
static SOLICITED: AtomicU64 = AtomicU64::new(0);
/// Where the layout ends (rounded up to a GiB): the start of the room
/// `reserve_for` keeps after it for the thread-stack region.
#[cfg(unix)]
static LAYOUT_END: AtomicU64 = AtomicU64::new(0);
/// The thread-stack region's size (StackRegionSize), as the kernel reports
/// it by default.
#[cfg(unix)]
const STACK_REGION_SIZE: u64 = recomp_services::DEFAULT_STACK_REGION.1;
/// A region inside the space could not be mapped at its address: generated
/// code must then use the checked path.
static UNMAPPABLE: AtomicBool = AtomicBool::new(false);
/// How many regions use each mapped guest page (guest page number). Regions
/// need not be page-aligned: a title's `.bss` starts where its `.data`
/// ends, so the two share a page.
static PAGE_USERS: Mutex<BTreeMap<u64, u32>> = Mutex::new(BTreeMap::new());

/// The host's page size, detected once. Guest pages stay 4 KiB; this is the
/// granularity at which the kernel maps and protects memory.
/// `RECOMP_HOST_PAGE_SIZE` overrides detection for tests (it must still be a
/// multiple of the real page size to keep `MAP_FIXED` valid).
#[cfg(unix)]
pub(crate) fn host_page() -> u64 {
    static HOST_PAGE: OnceLock<u64> = OnceLock::new();
    *HOST_PAGE.get_or_init(|| {
        // SAFETY: sysconf(3) takes no pointers and has no preconditions.
        let actual = page_size_from_sysconf(unsafe { libc::sysconf(libc::_SC_PAGESIZE) });

        // An explicit size is mainly a test hook (a 4 KiB host can exercise
        // the arithmetic and mapping path a 16 KiB Android host uses). It
        // must be a multiple of the real page size, or MAP_FIXED would
        // reject every address and fastmem would silently fall back.
        let Some(value) = std::env::var("RECOMP_HOST_PAGE_SIZE")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
        else {
            return actual;
        };
        match checked_page_size(value) {
            Some(page) if page % actual == 0 => page,
            _ => {
                eprintln!(
                    "[recomp-runtime] ignoring invalid RECOMP_HOST_PAGE_SIZE={value}; using the host page size {actual}"
                );
                actual
            }
        }
    })
}

/// Windows maps 4 KiB pages, like Horizon.
#[cfg(not(unix))]
#[allow(dead_code)]
pub(crate) fn host_page() -> u64 {
    PAGE
}

/// Validate a page size. Anything that cannot hold a 4 KiB guest page
/// exactly (or is not a power of two) is unsupported.
#[cfg(unix)]
fn checked_page_size(value: u64) -> Option<u64> {
    (value >= PAGE && value.is_power_of_two() && value <= MAX_HOST_PAGE).then_some(value)
}

/// Validate a `sysconf(_SC_PAGESIZE)` result. Junk falls back to 4 KiB,
/// which only makes the fast path more conservative.
#[cfg(unix)]
fn page_size_from_sysconf(value: libc::c_long) -> u64 {
    u64::try_from(value)
        .ok()
        .and_then(checked_page_size)
        .unwrap_or(PAGE)
}

/// The reservation's host address, created on first use; `None` when fastmem
/// is off or the full reservation failed.
#[cfg(unix)]
pub(crate) fn reservation() -> Option<u64> {
    static RESERVATION: OnceLock<Option<u64>> = OnceLock::new();
    *RESERVATION.get_or_init(|| {
        if std::env::var("RECOMP_FASTMEM").is_ok_and(|value| value == "0") {
            return None;
        }
        let (base, covered) = match reserve_guest_space() {
            Ok(reserved) => reserved,
            Err(error) => {
                eprintln!(
                    "[recomp-runtime] could not reserve the 39-bit guest address space ({error}); memory accesses stay checked"
                );
                return None;
            }
        };
        // The handler must be in place before generated code can use BASE.
        if !install_fault_handler() {
            // SAFETY: the reservation is ours and nothing has used it.
            unsafe { libc::munmap(base as *mut libc::c_void, covered as usize) };
            eprintln!(
                "[recomp-runtime] could not install the guest memory fault handler; memory accesses stay checked"
            );
            return None;
        }
        COVERED.store(covered, Ordering::Release);
        BASE.store(base as usize, Ordering::Release);
        if covered < SPACE {
            println!(
                "[recomp-runtime] guest memory: {:.1} GiB window reserved at {base:#x} (the host address space cannot hold all {:.0} GiB)",
                covered as f64 / (1u64 << 30) as f64,
                SPACE as f64 / (1u64 << 30) as f64
            );
        }
        Some(base)
    })
}

/// The thread-stack region to report (StackRegionAddress and Size) when
/// only a window could be reserved: right after the memory layout, inside
/// the window. The default region (2 GiB at 128 GiB) lies far above such a
/// window, and the SDK aliases every thread's stack there, so every stack
/// access of every thread but the main one took the checked path: on a
/// phone, LM3 spent about 40% of its loading time there. `None` when the
/// whole space is reserved (or nothing is), where the default region is
/// direct anyway.
#[cfg(unix)]
pub(crate) fn window_stack_region() -> Option<(u64, u64)> {
    reservation()?;
    let covered = covered();
    let start = LAYOUT_END.load(Ordering::Acquire);
    (covered < SPACE && start != 0 && start + STACK_REGION_SIZE <= covered)
        .then_some((start, STACK_REGION_SIZE))
}

/// The end (guest address) of the range the reservation covers, once it is
/// made; `SPACE` before, and when the whole space was reserved.
#[cfg(unix)]
pub(crate) fn covered() -> u64 {
    match COVERED.load(Ordering::Acquire) {
        0 => SPACE,
        covered => covered,
    }
}

/// Reserve the whole guest address space without committing memory. On a
/// 39-bit user address space that fails; the reservation then covers only
/// the window the guest's memory layout asked for (`reserve_for`), which
/// keeps generated code on the direct path.
#[cfg(unix)]
fn reserve_guest_space() -> Result<(u64, u64), std::io::Error> {
    if let Some(window) = forced_window() {
        let size = usize::try_from(window).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "guest address space does not fit in usize",
            )
        })?;
        return Ok((try_reserve(size)?, window));
    }
    let full = usize::try_from(SPACE).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "guest address space does not fit in usize",
        )
    })?;
    let error = match try_reserve(full) {
        Ok(base) => return Ok((base, SPACE)),
        Err(error) => error,
    };
    let wanted = SOLICITED.load(Ordering::Acquire);
    if wanted == 0 || wanted >= SPACE {
        return Err(error);
    }
    let size = usize::try_from(wanted).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "guest address space does not fit in usize",
        )
    })?;
    let base = try_reserve(size)?;
    Ok((base, wanted))
}

/// `RECOMP_FASTMEM_GIB`: reserve a window of this size instead of the whole
/// space. A test hook: a host that can reserve 512 GiB never takes the
/// bounded path, so this makes it reproducible there. Android does not need
/// it; its layout end picks the window.
#[cfg(unix)]
fn forced_window() -> Option<u64> {
    std::env::var("RECOMP_FASTMEM_GIB")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(|gib| gib.saturating_mul(1 << 30))
        .filter(|bytes| *bytes >= 1 << 20 && *bytes <= SPACE)
}

/// Reserve `size` inaccessible bytes the kernel places; nothing else refers
/// to that range. MAP_NORESERVE commits no physical memory.
#[cfg(unix)]
fn try_reserve(size: usize) -> Result<u64, std::io::Error> {
    // SAFETY: a fresh mapping; MAP_NORESERVE commits no physical memory.
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(base as u64)
    }
}

/// Tell the reservation how far the guest's memory layout reaches, so a
/// host that cannot reserve all 512 GiB reserves just that window, and the
/// thread-stack region after it (`window_stack_region`). Call before
/// regions are mapped; later, larger requests cannot grow the window (they
/// warn and fall back to the checked path).
#[cfg(unix)]
pub(crate) fn reserve_for(end: u64) {
    if end == 0 || end > SPACE {
        return;
    }
    // Round up to a GiB: the window is address space only, so a little
    // slack costs nothing and keeps the end host-page aligned.
    let layout_end = end.div_ceil(1 << 30).saturating_mul(1 << 30).min(SPACE);
    LAYOUT_END.fetch_max(layout_end, Ordering::AcqRel);
    let wanted = layout_end.saturating_add(STACK_REGION_SIZE).min(SPACE);
    let previous = SOLICITED.fetch_max(wanted, Ordering::AcqRel);
    if previous >= wanted {
        return;
    }
    if reservation().is_none() {
        return;
    }
    let covered = covered();
    if covered < wanted {
        eprintln!(
            "[recomp-runtime] guest memory layout reaches {wanted:#x}, beyond the reserved window {covered:#x}; accesses above it stay checked"
        );
    }
}

/// The guest pages `[start, end)` touches, in Horizon's 4 KiB pages.
#[cfg(unix)]
fn pages_between(start: u64, end: u64, page: u64) -> Range<u64> {
    debug_assert!(page >= PAGE && page.is_power_of_two());
    start / page..end.div_ceil(page)
}

/// The guest pages `len` bytes at `guest` touch.
#[cfg(unix)]
fn pages_of(guest: u64, len: usize) -> Range<u64> {
    pages_between(guest, guest + len as u64, PAGE)
}

/// The host pages `len` bytes at `guest` touch. The kernel maps at host page
/// granularity, which can be coarser than the guest's.
#[cfg(unix)]
fn host_pages_of(guest: u64, len: usize) -> Range<u64> {
    pages_between(guest, guest + len as u64, host_page())
}

/// The guest pages a host page holds, named by guest page number.
#[cfg(unix)]
fn guest_pages_in_host(host_page: u64, host: u64) -> Range<u64> {
    let per_host = host / PAGE;
    let start = host_page * per_host;
    start..start + per_host
}

/// The host address and length of host pages `pages`. The reservation base
/// is host-page aligned, so both are aligned whenever the page number range
/// is (which the arithmetic guarantees).
#[cfg(unix)]
fn host_range(base: u64, pages: Range<u64>, page: u64) -> (u64, usize) {
    let start = base + pages.start * page;
    let len = (pages.end - pages.start) * page;
    (start, len as usize)
}

/// Calls `apply` with each run of consecutive pages in `pages`.
fn runs(pages: &[u64], mut apply: impl FnMut(Range<u64>)) {
    let mut iter = pages.iter().copied().peekable();
    while let Some(start) = iter.next() {
        let mut end = start + 1;
        while iter.next_if_eq(&end).is_some() {
            end += 1;
        }
        apply(start..end);
    }
}

/// Give host pages `pages` the protection `protection`, with fresh zeroed
/// memory when they become accessible. The address and length are checked
/// against the host page size because `MAP_FIXED` rejects a misaligned
/// address; on a 16 KiB-page kernel a guest page is not necessarily aligned.
#[cfg(unix)]
fn protect_with_page(base: u64, pages: Range<u64>, protection: libc::c_int, page: u64) -> bool {
    let (start, len) = host_range(base, pages, page);
    if len == 0 {
        return true;
    }
    if page < PAGE || !page.is_power_of_two() || start % page != 0 || len % page as usize != 0 {
        return false;
    }
    // SAFETY: the pages lie inside the reservation, which this module owns,
    // and no region uses them. MAP_FIXED is safe on an owned range and the
    // host address/length are host-page aligned above.
    let mapped = unsafe {
        libc::mmap(
            start as *mut libc::c_void,
            len,
            protection,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    mapped != libc::MAP_FAILED
}

/// Give host pages `pages` the protection `protection`.
#[cfg(unix)]
fn protect(base: u64, pages: Range<u64>, protection: libc::c_int) -> bool {
    protect_with_page(base, pages, protection, host_page())
}

/// Map `len` zeroed bytes of guest memory at `guest` inside the
/// reservation. A guest page the region covers only in part and another
/// region already uses (the two share it) is kept, with the new region's
/// bytes zeroed. `None` (and the checked path for generated code) when the
/// range is not in the space or cannot be mapped.
#[cfg(unix)]
pub(crate) fn map_pages(guest: u64, len: usize) -> Option<*mut u8> {
    let base = reservation()?;
    let end = match guest.checked_add(len as u64) {
        Some(end) => end,
        None => {
            UNMAPPABLE.store(true, Ordering::Release);
            return None;
        }
    };
    if end > SPACE || len == 0 {
        // A region outside the space cannot be represented at its guest
        // address, so generated code must not use a direct base.
        UNMAPPABLE.store(true, Ordering::Release);
        return None;
    }
    let covered = covered();
    if end > covered {
        // Beyond the reserved window there is no page to MAP_FIXED into
        // (and it could land on the host's own memory), so the region lives
        // in a buffer of its own. Generated code reaches addresses at or
        // above the window's end through the checked path anyway, so a
        // region wholly above it leaves direct access on for the rest: LM3
        // on a phone maps such regions while it loads, and turning direct
        // access off for all of guest memory made every load and store a
        // runtime call (its loading screen took minutes). Only a region
        // that starts inside the window, whose lower part direct accesses
        // would reach, needs the checked path everywhere.
        if guest < covered {
            UNMAPPABLE.store(true, Ordering::Release);
        }
        return None;
    }

    let host = host_page();
    let mut users = PAGE_USERS.lock().unwrap_or_else(|e| e.into_inner());
    let host_pages = host_pages_of(guest, len);
    // A host page is fresh when no guest page in it is in use. Mapping a
    // fresh host page also maps every guest page it contains (zeros).
    let fresh: Vec<u64> = host_pages
        .clone()
        .filter(|page| {
            users
                .range(guest_pages_in_host(*page, host))
                .next()
                .is_none()
        })
        .collect();

    let mut mapped = true;
    runs(&fresh, |pages| {
        mapped &= protect_with_page(base, pages, libc::PROT_READ | libc::PROT_WRITE, host);
    });
    if !mapped {
        // Do not leave a half-mapped region behind: discard the host pages
        // this call mapped, then use the checked path.
        runs(&fresh, |pages| {
            protect_with_page(base, pages, libc::PROT_NONE, host);
        });
        UNMAPPABLE.store(true, Ordering::Release);
        return None;
    }

    // Host pages that were already mapped may contain another region's
    // bytes. Zero only the guest pages this region newly claims (the whole
    // page, matching a fresh mmap) or, for a page shared with another
    // region, only this region's bytes.
    for page in pages_of(guest, len) {
        let host_page = page / (host / PAGE);
        if fresh.binary_search(&host_page).is_ok() {
            continue;
        }
        let (start, stop) = if users.contains_key(&page) {
            (guest.max(page * PAGE), end.min((page + 1) * PAGE))
        } else {
            (page * PAGE, (page + 1) * PAGE)
        };
        // SAFETY: the host page is mapped and these bytes belong to the new
        // region (or to no region yet).
        unsafe { std::ptr::write_bytes((base + start) as *mut u8, 0, (stop - start) as usize) };
    }
    for page in pages_of(guest, len) {
        *users.entry(page).or_insert(0) += 1;
    }
    Some((base + guest) as *mut u8)
}

/// Unmap a region `map_pages` mapped: its guest pages no other region uses
/// become inaccessible again, releasing their memory. A host page is
/// released only when no guest page in it is in use.
#[cfg(unix)]
pub(crate) fn release_pages(data: *mut u8, len: usize) {
    let base = BASE.load(Ordering::Acquire) as u64;
    if base == 0 {
        return;
    }
    let guest = data as u64 - base;
    let host = host_page();
    let mut users = PAGE_USERS.lock().unwrap_or_else(|e| e.into_inner());
    let mut touched_hosts = Vec::new();
    for page in pages_of(guest, len) {
        let host_page = page / (host / PAGE);
        if touched_hosts.last() != Some(&host_page) {
            touched_hosts.push(host_page);
        }
        if let Some(count) = users.get_mut(&page) {
            *count -= 1;
            if *count == 0 {
                users.remove(&page);
            }
        }
    }
    // A host page shared with another region stays mapped. Zero the bytes
    // this region used so a later direct access cannot see its old data
    // (the checked path still reports the guest address as unmapped).
    for page in pages_of(guest, len) {
        let (start, stop) = if users.contains_key(&page) {
            (
                guest.max(page * PAGE),
                (guest + len as u64).min((page + 1) * PAGE),
            )
        } else {
            (page * PAGE, (page + 1) * PAGE)
        };
        // SAFETY: the region's host pages are still mapped here; the range
        // is inside this region or a guest page no region uses now.
        unsafe { std::ptr::write_bytes((base + start) as *mut u8, 0, (stop - start) as usize) };
    }
    let unused: Vec<u64> = touched_hosts
        .into_iter()
        .filter(|page| {
            users
                .range(guest_pages_in_host(*page, host))
                .next()
                .is_none()
        })
        .collect();
    runs(&unused, |pages| {
        protect(base, pages, libc::PROT_NONE);
    });
}

/// Let generated code access guest memory directly, unless something needs
/// the checked path.
pub(crate) fn publish_base() {
    let direct = reservation().filter(|_| {
        !UNMAPPABLE.load(Ordering::Acquire) && std::env::var_os("RECOMP_WATCH_STORE").is_none()
    });
    match direct {
        Some(base) => recomp_abi::set_memory_range(base as usize, covered()),
        None => recomp_abi::set_memory_range(0, 0),
    }
}

#[cfg(unix)]
static mut PREVIOUS: std::mem::MaybeUninit<libc::sigaction> = std::mem::MaybeUninit::uninit();

#[cfg(unix)]
fn install_fault_handler() -> bool {
    // SAFETY: called once, before generated code can fault in the
    // reservation; the previous action is kept for faults that are not
    // guest accesses (Rust's stack overflow detection among them).
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = on_fault as *const () as usize;
        action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigemptyset(&mut action.sa_mask);
        libc::sigaction(
            libc::SIGSEGV,
            &action,
            (&raw mut PREVIOUS).cast::<libc::sigaction>(),
        ) == 0
    }
}

/// A fixed buffer to format a message in without allocating.
#[cfg(unix)]
struct Message {
    bytes: [u8; 256],
    len: usize,
}

#[cfg(unix)]
impl std::fmt::Write for Message {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let room = self.bytes.len() - self.len;
        let take = text.len().min(room);
        self.bytes[self.len..self.len + take].copy_from_slice(&text.as_bytes()[..take]);
        self.len += take;
        Ok(())
    }
}

#[cfg(unix)]
extern "C" fn on_fault(
    signal: libc::c_int,
    info: *mut libc::siginfo_t,
    context: *mut libc::c_void,
) {
    // SAFETY: the kernel passes a valid siginfo for SA_SIGINFO handlers.
    let address = unsafe { (*info).si_addr() } as usize;
    let base = BASE.load(Ordering::Acquire);
    if base != 0 && (base..base + covered() as usize).contains(&address) {
        use std::fmt::Write;
        let mut message = Message {
            bytes: [0; 256],
            len: 0,
        };
        let _ = writeln!(
            message,
            "[recomp-runtime] memory access error: guest address {:#x} is not mapped (in the guest block at {:#x}, thread {:#x})",
            address - base,
            recomp_abi::current_block(),
            crate::threads::current_thread(),
        );
        // SAFETY: write(2) and _exit(2) are async-signal-safe.
        unsafe {
            libc::write(2, message.bytes.as_ptr().cast(), message.len);
            crate::let_the_log_drain();
            libc::_exit(1);
        }
    }
    // Not a guest access: hand it to whoever handled SIGSEGV before.
    // SAFETY: PREVIOUS was filled in by install_fault_handler.
    unsafe {
        let previous = (&raw const PREVIOUS).cast::<libc::sigaction>().read();
        if previous.sa_sigaction > libc::SIG_IGN && previous.sa_flags & libc::SA_SIGINFO != 0 {
            let handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                std::mem::transmute(previous.sa_sigaction);
            handler(signal, info, context);
        } else if previous.sa_sigaction > libc::SIG_IGN {
            let handler: extern "C" fn(libc::c_int) = std::mem::transmute(previous.sa_sigaction);
            handler(signal);
        } else {
            // Default action: restore it and let the fault happen again.
            libc::sigaction(libc::SIGSEGV, &previous, std::ptr::null_mut());
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn host_page_detection_accepts_real_page_sizes_and_rejects_junk() {
        assert_eq!(page_size_from_sysconf(4096), 4096);
        assert_eq!(page_size_from_sysconf(16384), 16384);
        assert_eq!(page_size_from_sysconf(65536), 65536);
        // Below a guest page, not a power of two, negative, zero or absurd:
        // stay on the 4 KiB fallback rather than trust the arithmetic.
        assert_eq!(page_size_from_sysconf(1024), PAGE);
        assert_eq!(page_size_from_sysconf(3000), PAGE);
        assert_eq!(page_size_from_sysconf(-1), PAGE);
        assert_eq!(page_size_from_sysconf(0), PAGE);
        assert_eq!(
            page_size_from_sysconf((MAX_HOST_PAGE * 2) as libc::c_long),
            PAGE
        );
    }

    #[test]
    fn guest_page_arithmetic_stays_at_four_kib() {
        assert_eq!(pages_of(0x1000, 1), 1..2);
        assert_eq!(pages_of(0x1fff, 2), 1..3);
        assert_eq!(pages_of(0x2000, 0x1000), 2..3);
        // A 64 KiB region starting mid-page touches 17 guest pages.
        assert_eq!(pages_of(0x1234, 0x10000), 1..18);
    }

    #[test]
    fn host_page_arithmetic_covers_every_guest_page_it_holds() {
        // One 16 KiB host page holds four 4 KiB guest pages, including a
        // region that starts halfway through the first one.
        assert_eq!(pages_between(0x1234, 0x5234, 0x4000), 0..2);
        assert_eq!(pages_between(0x1000, 0x5000, 0x4000), 0..2);
        assert_eq!(pages_between(0x4000, 0x4001, 0x4000), 1..2);
        // A 64 KiB host page covers sixteen guest pages.
        assert_eq!(pages_between(0x1000, 0x10001, 0x10000), 0..2);
    }

    #[test]
    fn host_ranges_are_page_aligned_for_map_fixed() {
        const BASE: u64 = 0x7f00_0000_0000;
        for page in [0x1000u64, 0x4000, 0x10000] {
            for guest in [0x1000u64, 0x1234, 0x1_0000] {
                let pages = pages_between(guest, guest + 0x2345, page);
                let (start, len) = host_range(BASE, pages, page);
                assert_eq!(start % page, 0, "address aligned for {page:#x}");
                assert_eq!(len % page as usize, 0, "length aligned for {page:#x}");
                assert!(start >= BASE);
                assert!(start + len as u64 <= BASE + SPACE);
            }
        }
        // A misaligned base is refused before MAP_FIXED can fail (or worse,
        // replace an unrelated neighbour).
        let (start, _) = host_range(BASE + 0x100, 0..2, 0x4000);
        assert_ne!(start % 0x4000, 0);
    }

    #[test]
    fn detected_host_page_holds_whole_guest_pages_and_divides_the_space() {
        let page = host_page();
        assert!(page >= PAGE, "host page {page:#x} >= guest page");
        assert!(page.is_power_of_two());
        assert_eq!(SPACE % page, 0, "the 39-bit space ends on a host page");
    }
}

/// Windows: one reserved allocation and a vectored exception handler.
#[cfg(windows)]
mod windows {
    use std::ffi::c_void;
    use std::ops::Range;
    use std::sync::atomic::Ordering;
    use std::sync::OnceLock;

    use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE,
        MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE,
    };
    use windows_sys::Win32::System::SystemServices::MEM_WRITE_WATCH;

    use super::{runs, BASE, PAGE, PAGE_USERS, SPACE, UNMAPPABLE};

    /// The reservation's host address, created on first use; `None` when
    /// fastmem is off or the reservation failed.
    pub(crate) fn reservation() -> Option<u64> {
        static RESERVATION: OnceLock<Option<u64>> = OnceLock::new();
        *RESERVATION.get_or_init(|| {
            if std::env::var("RECOMP_FASTMEM").is_ok_and(|value| value == "0") {
                return None;
            }
            // SAFETY: a fresh reservation the system places; nothing else
            // refers to that range. MEM_RESERVE commits no physical memory.
            // MEM_WRITE_WATCH makes GetWriteWatch work for committed pages,
            // which the GPU cache uses to notice guest rewrites.
            let base = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    SPACE as usize,
                    MEM_RESERVE | MEM_WRITE_WATCH,
                    PAGE_NOACCESS,
                )
            };
            if base.is_null() {
                eprintln!("[recomp-runtime] could not reserve the guest address space; memory accesses stay checked");
                return None;
            }
            BASE.store(base as usize, Ordering::Release);
            if !install_fault_handler() {
                // SAFETY: the reservation is ours and nothing has used it.
                unsafe { VirtualFree(base, 0, MEM_RELEASE) };
                eprintln!(
                    "[recomp-runtime] could not install the guest memory fault handler; memory accesses stay checked"
                );
                return None;
            }
            Some(base as u64)
        })
    }

    /// The guest pages `len` bytes at `guest` touch.
    fn pages_of(guest: u64, len: usize) -> Range<u64> {
        guest / PAGE..(guest + len as u64).div_ceil(PAGE)
    }

    /// Commit `pages` with fresh zeroed memory, or make committed pages
    /// writable again.
    fn commit(base: u64, pages: Range<u64>) -> bool {
        let address = (base + pages.start * PAGE) as *const c_void;
        let size = ((pages.end - pages.start) * PAGE) as usize;
        // SAFETY: the pages lie inside the reservation, which this module
        // owns, and no region uses them. The system zeroes newly committed
        // pages; VirtualProtect covers a page that was only re-protected.
        unsafe {
            if !VirtualAlloc(address, size, MEM_COMMIT, PAGE_READWRITE).is_null() {
                return true;
            }
            let mut previous = 0;
            VirtualProtect(address, size, PAGE_READWRITE, &mut previous) != 0
        }
    }

    fn decommit(base: u64, pages: Range<u64>) -> bool {
        let address = (base + pages.start * PAGE) as *mut c_void;
        let size = ((pages.end - pages.start) * PAGE) as usize;
        // SAFETY: as in commit. MEM_DECOMMIT releases the pages' memory but
        // leaves the reservation in place, so a later map can reuse them.
        unsafe {
            if VirtualFree(address, size, MEM_DECOMMIT) != 0 {
                return true;
            }
            let mut previous = 0;
            VirtualProtect(address, size, PAGE_NOACCESS, &mut previous) != 0
        }
    }

    /// Map `len` zeroed bytes of guest memory at `guest` inside the
    /// reservation. A page the region covers only in part and another region
    /// already uses (the two share it) is kept, with the new region's bytes
    /// zeroed. `None` (and the checked path for generated code) when the
    /// range is not in the space or cannot be mapped.
    pub(crate) fn map_pages(guest: u64, len: usize) -> Option<*mut u8> {
        let base = reservation()?;
        let end = guest.checked_add(len as u64)?;
        if end > SPACE || len == 0 {
            return None;
        }
        let mut users = PAGE_USERS.lock().unwrap_or_else(|e| e.into_inner());
        let whole = |page: u64| page * PAGE >= guest && (page + 1) * PAGE <= end;
        let (fresh, shared): (Vec<u64>, Vec<u64>) =
            pages_of(guest, len).partition(|page| whole(*page) || !users.contains_key(page));
        let mut mapped = true;
        runs(&fresh, |pages| {
            mapped &= commit(base, pages);
        });
        if !mapped {
            UNMAPPABLE.store(true, Ordering::Release);
            return None;
        }
        for page in shared {
            let start = guest.max(page * PAGE);
            let stop = end.min((page + 1) * PAGE);
            // SAFETY: the page is mapped (another region uses it) and these
            // bytes belong to the new region.
            unsafe { std::ptr::write_bytes((base + start) as *mut u8, 0, (stop - start) as usize) };
        }
        for page in pages_of(guest, len) {
            *users.entry(page).or_insert(0) += 1;
        }
        Some((base + guest) as *mut u8)
    }

    /// Unmap a region `map_pages` mapped: its pages no other region uses
    /// become inaccessible again, releasing their memory.
    pub(crate) fn release_pages(data: *mut u8, len: usize) {
        let base = BASE.load(Ordering::Acquire) as u64;
        if base == 0 {
            return;
        }
        let guest = data as u64 - base;
        let mut users = PAGE_USERS.lock().unwrap_or_else(|e| e.into_inner());
        let mut unused = Vec::new();
        for page in pages_of(guest, len) {
            if let Some(count) = users.get_mut(&page) {
                *count -= 1;
                if *count == 0 {
                    users.remove(&page);
                    unused.push(page);
                }
            }
        }
        runs(&unused, |pages| {
            decommit(base, pages);
        });
    }

    fn install_fault_handler() -> bool {
        // SAFETY: called once, before generated code can fault in the
        // reservation. First in the chain so the guard page's own handler
        // does not turn a guest access into a crash report first.
        let handle = unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) };
        if handle.is_null() {
            return false;
        }
        true
    }

    /// A first-chance access violation inside the reservation: give the
    /// faulting page zeroed read/write memory and resume the instruction.
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        if info.is_null() {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: the system passes a valid exception record and pointers
        // that live for the duration of this call.
        let record = unsafe { (*info).ExceptionRecord };
        if record.is_null() {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let record = unsafe { &*record };
        if record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record.NumberParameters < 2 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // ExceptionInformation[0] is the access kind (0 read, 1 write,
        // 8 execute); only read and write faults can be resumed by making
        // the page readable and writable.
        if record.ExceptionInformation[0] > 1 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let base = BASE.load(Ordering::Acquire);
        let address = record.ExceptionInformation[1];
        if base == 0 || address < base || address - base >= SPACE as usize {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let page = (address & !(PAGE as usize - 1)) as *const c_void;
        // SAFETY: the page is inside the reservation, which this module
        // owns. These calls allocate and change no other memory.
        unsafe {
            if !VirtualAlloc(page, PAGE as usize, MEM_COMMIT, PAGE_READWRITE).is_null() {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            let mut previous = 0;
            if VirtualProtect(page, PAGE as usize, PAGE_READWRITE, &mut previous) != 0 {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }
}

#[cfg(windows)]
pub(crate) use windows::{map_pages, release_pages, reservation};

/// Every other target: no reservation, so every memory access stays on the
/// checked runtime path.
#[cfg(not(any(unix, windows)))]
pub(crate) fn reservation() -> Option<u64> {
    None
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn map_pages(_guest: u64, _len: usize) -> Option<*mut u8> {
    None
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn release_pages(_data: *mut u8, _len: usize) {}

/// Windows reserves the whole guest address space, so the covered end is
/// always `SPACE`; there is nothing to size.
#[cfg(not(unix))]
pub(crate) fn covered() -> u64 {
    SPACE
}

#[cfg(not(unix))]
pub(crate) fn reserve_for(_end: u64) {}

/// Windows reserves the whole space: the default stack region is direct.
#[cfg(not(unix))]
pub(crate) fn window_stack_region() -> Option<(u64, u64)> {
    None
}
