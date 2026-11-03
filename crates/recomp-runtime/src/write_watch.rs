//! Which pages of guest memory were written since they were last checked,
//! without reading them: userfaultfd's asynchronous write protection
//! (`UFFD_FEATURE_WP_ASYNC`, Linux 6.7). A watched page is write-protected;
//! the kernel lets the first write through and marks the page, with no
//! signal or handler involved, so writes the kernel makes (a file read into
//! guest memory) work as before. `PAGEMAP_SCAN` reports a range's marked
//! pages and protects them again in one call (0.2 us for 64 KiB).
//!
//! The GPU worker uses this to skip hashing vertex ranges that were not
//! written since it copied them: tens of MB a frame in LM3's rooms.
//! `RECOMP_WRITE_WATCH=0` turns it off (the worker then hashes everything).
//!
//! Linux uses userfaultfd; Windows uses `GetWriteWatch` on a reservation made
//! with `MEM_WRITE_WATCH`. Other hosts return `None` and the worker hashes the
//! ranges it copied; nothing blocks startup.

#[cfg(target_os = "linux")]
mod linux {
    use std::collections::BTreeMap;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::sync::{Mutex, OnceLock};

    use crate::fastmem::{self, PAGE};

    // <linux/userfaultfd.h> and <linux/fs.h>.
    const UFFD_USER_MODE_ONLY: libc::c_long = 1;
    const UFFD_API: u64 = 0xAA;
    const UFFD_FEATURE_WP_ASYNC: u64 = 1 << 15;
    const UFFDIO_REGISTER_MODE_WP: u64 = 1 << 1;
    const UFFDIO_WRITEPROTECT_MODE_WP: u64 = 1 << 0;
    const PM_SCAN_WP_MATCHING: u64 = 1 << 0;
    const PM_SCAN_CHECK_WPASYNC: u64 = 1 << 1;
    const PAGE_IS_WRITTEN: u64 = 1 << 1;

    const fn iowr(kind: u8, number: u8, size: usize) -> libc::c_ulong {
        (3 << 30)
            | ((size as libc::c_ulong) << 16)
            | ((kind as libc::c_ulong) << 8)
            | number as libc::c_ulong
    }
    const UFFDIO_API: libc::c_ulong = iowr(0xAA, 0x3F, std::mem::size_of::<UffdioApi>());
    const UFFDIO_REGISTER: libc::c_ulong = iowr(0xAA, 0x00, std::mem::size_of::<UffdioRegister>());
    const UFFDIO_WRITEPROTECT: libc::c_ulong =
        iowr(0xAA, 0x06, std::mem::size_of::<UffdioWriteprotect>());
    const PAGEMAP_SCAN: libc::c_ulong = iowr(b'f', 16, std::mem::size_of::<ScanArgs>());

    #[repr(C)]
    struct UffdioApi {
        api: u64,
        features: u64,
        ioctls: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct UffdioRange {
        start: u64,
        len: u64,
    }

    #[repr(C)]
    struct UffdioRegister {
        range: UffdioRange,
        mode: u64,
        ioctls: u64,
    }

    #[repr(C)]
    struct UffdioWriteprotect {
        range: UffdioRange,
        mode: u64,
    }

    #[repr(C)]
    struct ScanArgs {
        size: u64,
        flags: u64,
        start: u64,
        end: u64,
        walk_end: u64,
        vec: u64,
        vec_len: u64,
        max_pages: u64,
        category_inverted: u64,
        category_mask: u64,
        category_anyof_mask: u64,
        return_mask: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct PageRegion {
        start: u64,
        end: u64,
        categories: u64,
    }

    struct Watch {
        uffd: OwnedFd,
        pagemap: std::fs::File,
        /// Host address of guest address 0 (the fastmem reservation).
        base: u64,
        /// Guest pages scans found written, with the check that found them: a
        /// scan protects its pages again, so another range sharing a page learns
        /// of the write here.
        written: BTreeMap<u64, u64>,
        /// Checks made so far.
        checks: u64,
    }

    fn watch() -> Option<&'static Mutex<Watch>> {
        static WATCH: OnceLock<Option<Mutex<Watch>>> = OnceLock::new();
        WATCH
            .get_or_init(|| match open() {
                Ok(watch) => Some(Mutex::new(watch)),
                Err(reason) => {
                    println!("[recomp-runtime] guest writes are not watched ({reason}); the renderer hashes vertex data instead");
                    None
                }
            })
            .as_ref()
    }

    fn open() -> Result<Watch, String> {
        if std::env::var_os("RECOMP_WRITE_WATCH").is_some_and(|value| value == "0") {
            return Err("RECOMP_WRITE_WATCH=0".to_string());
        }
        // userfaultfd and PAGEMAP_SCAN work in host pages, while the watch and
        // its callers speak in 4 KiB guest pages. A 16 KiB-page host cannot
        // protect half a page, so use the hashing fallback there.
        if fastmem::host_page() != PAGE {
            return Err(format!(
                "host page size is {} bytes, not 4 KiB",
                fastmem::host_page()
            ));
        }
        open_at(fastmem::reservation().ok_or("no fastmem reservation")?)
    }

    /// A watch for memory whose guest address 0 is at host address `base`.
    fn open_at(base: u64) -> Result<Watch, String> {
        // User-mode-only descriptors need no privilege
        // (vm.unprivileged_userfaultfd = 0); asynchronous write protection
        // resolves kernel-mode writes itself anyway.
        // SAFETY: a plain system call; the descriptor is owned from here on.
        let uffd = unsafe {
            let fd = libc::syscall(
                libc::SYS_userfaultfd,
                libc::O_CLOEXEC as libc::c_long
                    | libc::O_NONBLOCK as libc::c_long
                    | UFFD_USER_MODE_ONLY,
            );
            if fd < 0 {
                return Err(format!("userfaultfd: {}", std::io::Error::last_os_error()));
            }
            OwnedFd::from_raw_fd(fd as libc::c_int)
        };
        let mut api = UffdioApi {
            api: UFFD_API,
            features: UFFD_FEATURE_WP_ASYNC,
            ioctls: 0,
        };
        // SAFETY: the argument is the structure UFFDIO_API takes.
        if unsafe { libc::ioctl(uffd.as_raw_fd(), UFFDIO_API as libc::Ioctl, &mut api) } != 0 {
            return Err(format!(
                "asynchronous write protection: {}",
                std::io::Error::last_os_error()
            ));
        }
        let pagemap = std::fs::File::open("/proc/self/pagemap")
            .map_err(|error| format!("/proc/self/pagemap: {error}"))?;
        Ok(Watch {
            uffd,
            pagemap,
            base,
            written: BTreeMap::new(),
            checks: 0,
        })
    }

    impl Watch {
        /// Write-protect guest pages `pages`, registering them first (a page
        /// mapped again since it was last registered is not).
        fn protect(&self, pages: std::ops::Range<u64>) -> bool {
            let range = UffdioRange {
                start: self.base + pages.start * PAGE,
                len: (pages.end - pages.start) * PAGE,
            };
            let mut register = UffdioRegister {
                range,
                mode: UFFDIO_REGISTER_MODE_WP,
                ioctls: 0,
            };
            let mut protect = UffdioWriteprotect {
                range,
                mode: UFFDIO_WRITEPROTECT_MODE_WP,
            };
            // SAFETY: the arguments are the structures these requests take; the
            // range lies in the reservation, which stays mapped.
            unsafe {
                libc::ioctl(
                    self.uffd.as_raw_fd(),
                    UFFDIO_REGISTER as libc::Ioctl,
                    &mut register,
                ) == 0
                    && libc::ioctl(
                        self.uffd.as_raw_fd(),
                        UFFDIO_WRITEPROTECT as libc::Ioctl,
                        &mut protect,
                    ) == 0
            }
        }

        /// The written pages among `pages`, protected again; false when a page
        /// is not watched (mapped again since it was registered).
        fn scan(&self, pages: std::ops::Range<u64>, found: &mut Vec<std::ops::Range<u64>>) -> bool {
            let mut regions = [PageRegion::default(); 32];
            let end = self.base + pages.end * PAGE;
            let mut start = self.base + pages.start * PAGE;
            while start < end {
                let mut args = ScanArgs {
                    size: std::mem::size_of::<ScanArgs>() as u64,
                    flags: PM_SCAN_WP_MATCHING | PM_SCAN_CHECK_WPASYNC,
                    start,
                    end,
                    walk_end: 0,
                    vec: regions.as_mut_ptr() as u64,
                    vec_len: regions.len() as u64,
                    max_pages: 0,
                    category_inverted: 0,
                    category_mask: PAGE_IS_WRITTEN,
                    category_anyof_mask: 0,
                    return_mask: PAGE_IS_WRITTEN,
                };
                // SAFETY: the argument is the structure PAGEMAP_SCAN takes and
                // `vec` has room for `vec_len` regions.
                let count = unsafe {
                    libc::ioctl(
                        self.pagemap.as_raw_fd(),
                        PAGEMAP_SCAN as libc::Ioctl,
                        &mut args,
                    )
                };
                if count < 0 || args.walk_end <= start {
                    return false;
                }
                for region in &regions[..count as usize] {
                    found.push((region.start - self.base) / PAGE..(region.end - self.base) / PAGE);
                }
                start = args.walk_end;
            }
            true
        }
    }

    /// Watch writes to the `len` bytes of guest memory at `address`, and report
    /// the parts written since check `since` to `written` (offset into the range
    /// and length: whole pages, clipped to the range). `since` `None` starts
    /// watching and reports nothing; the caller reads the range after this call,
    /// so it sees any write made before. Returns this check's number, the next
    /// call's `since`; `None` when writes there cannot be watched.
    pub(crate) fn written_since(
        address: u64,
        len: usize,
        since: Option<u64>,
        written: &mut dyn FnMut(usize, usize),
    ) -> Option<u64> {
        let end = address.checked_add(len as u64)?;
        let host = crate::memory::host_address(address, len)?;
        let mut watch = watch()?.lock().unwrap_or_else(|error| error.into_inner());
        // Only memory in the reservation: a region mapped elsewhere would leave
        // its pages here unwritten whatever the guest does.
        if len == 0 || host != watch.base + address {
            return None;
        }
        let pages = address / PAGE..end.div_ceil(PAGE);
        watch.checks += 1;
        let check = watch.checks;
        let Some(since) = since else {
            return watch.protect(pages).then_some(check);
        };
        let mut found = Vec::new();
        if !watch.scan(pages.clone(), &mut found) {
            // Mapped again since: watch the new pages, and take all as written.
            written(0, len);
            return watch.protect(pages).then_some(check);
        }
        for range in found {
            for page in range {
                watch.written.insert(page, check);
            }
        }
        let mut report = |run: std::ops::Range<u64>| {
            let start = (run.start * PAGE).max(address);
            let stop = (run.end * PAGE).min(end);
            written((start - address) as usize, (stop - start) as usize);
        };
        let mut run: Option<std::ops::Range<u64>> = None;
        for (&page, _) in watch
            .written
            .range(pages)
            .filter(|(_, found)| **found > since)
        {
            match run {
                Some(ref mut current) if current.end == page => current.end += 1,
                _ => {
                    if let Some(done) = run.replace(page..page + 1) {
                        report(done);
                    }
                }
            }
        }
        if let Some(done) = run {
            report(done);
        }
        Some(check)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn request_numbers_match_the_kernel_headers() {
            assert_eq!(UFFDIO_API, 0xC018_AA3F);
            assert_eq!(UFFDIO_REGISTER, 0xC020_AA00);
            assert_eq!(UFFDIO_WRITEPROTECT, 0xC018_AA06);
            assert_eq!(PAGEMAP_SCAN, 0xC060_6610);
        }

        #[test]
        fn scans_report_written_pages_once() {
            const PAGES: u64 = 16;
            // SAFETY: a fresh private mapping this test owns.
            let memory = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    (PAGES * PAGE) as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(memory, libc::MAP_FAILED);
            let memory = memory.cast::<u8>();
            let first = memory as u64 / PAGE;
            let watch = match open_at(0) {
                Ok(watch) => watch,
                Err(reason) => {
                    eprintln!("skipped: {reason}");
                    return;
                }
            };
            // SAFETY: every offset is inside the mapping.
            let write = |page: u64| unsafe { memory.add((page * PAGE + 5) as usize).write(7) };
            (0..PAGES).for_each(write);
            assert!(watch.protect(first..first + PAGES));
            let scan = || {
                let mut found = Vec::new();
                assert!(watch.scan(first..first + PAGES, &mut found));
                found
                    .iter()
                    .map(|range| range.start - first..range.end - first)
                    .collect::<Vec<_>>()
            };
            assert_eq!(scan(), []);
            write(3);
            write(9);
            write(10);
            // A write the kernel makes (here a pipe read) counts too.
            let mut pipe = [0; 2];
            // SAFETY: plain system calls on descriptors this test owns.
            unsafe {
                assert_eq!(libc::pipe(pipe.as_mut_ptr()), 0);
                assert_eq!(libc::write(pipe[1], b"hello".as_ptr().cast(), 5), 5);
                assert_eq!(
                    libc::read(pipe[0], memory.add((12 * PAGE) as usize).cast(), 5),
                    5
                );
                libc::close(pipe[0]);
                libc::close(pipe[1]);
            }
            assert_eq!(scan(), [3..4, 9..11, 12..13]);
            assert_eq!(scan(), []);
            // SAFETY: the mapping is not used after this.
            unsafe { libc::munmap(memory.cast(), (PAGES * PAGE) as usize) };
        }
    }
}

/// Windows: `GetWriteWatch` on the guest reservation. The reservation is
/// created with `MEM_WRITE_WATCH`, and each check resets only the range it
/// asked about, so one cached vertex range or texture does not clear another
/// range's dirty pages.
#[cfg(windows)]
mod windows {
    use std::ffi::c_void;

    use windows_sys::Win32::System::Memory::GetWriteWatch;
    use windows_sys::Win32::System::SystemServices::WRITE_WATCH_FLAG_RESET;

    const PAGE: u64 = 4096;

    pub(crate) fn written_since(
        address: u64,
        len: usize,
        _since: Option<u64>,
        written: &mut dyn FnMut(usize, usize),
    ) -> Option<u64> {
        if len == 0 {
            return Some(0);
        }
        let start = address & !(PAGE - 1);
        let end = (address + len as u64).div_ceil(PAGE) * PAGE;
        if end <= start {
            return None;
        }
        // A range's writes are normally a handful of pages. If a single
        // check dirties more than this, fall back to hashing that range
        // instead of growing an unbounded buffer.
        let mut addresses = vec![std::ptr::null_mut::<c_void>(); 65_536];
        let mut count = addresses.len();
        let mut granularity = 0u32;
        let result = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                start as *const c_void,
                (end - start) as usize,
                addresses.as_mut_ptr(),
                &mut count,
                &mut granularity,
            )
        };
        if result != 0 {
            return None;
        }
        for page in &addresses[..count] {
            let page = *page as u64;
            let begin = page.max(address);
            let stop = (page + PAGE).min(address + len as u64);
            if begin < stop {
                written((begin - address) as usize, (stop - begin) as usize);
            }
        }
        Some(0)
    }
}

/// Other platforms have no asynchronous write-protection equivalent, so
/// writes are not watched: the GPU worker hashes the ranges it copied.
#[cfg(not(any(target_os = "linux", windows)))]
pub(crate) fn written_since(
    _address: u64,
    _len: usize,
    _since: Option<u64>,
    _written: &mut dyn FnMut(usize, usize),
) -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
pub(crate) use linux::written_since;
#[cfg(windows)]
pub(crate) use windows::written_since;

#[cfg(all(test, not(any(target_os = "linux", windows))))]
mod tests {
    use super::*;

    #[test]
    fn unwatched_ranges_report_nothing() {
        let mut written = false;
        assert_eq!(
            written_since(0x1000, 0x1000, None, &mut |_, _| written = true),
            None
        );
        assert!(!written);
    }
}

#[cfg(all(test, windows))]
mod windows_tests {
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
    };
    use windows_sys::Win32::System::SystemServices::MEM_WRITE_WATCH;

    use super::written_since;

    #[test]
    fn a_write_is_reported_once() {
        let size = 0x1000usize;
        let memory = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                size,
                MEM_RESERVE | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        assert!(!memory.is_null());
        let commit = unsafe { VirtualAlloc(memory, size, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!commit.is_null());
        // First watch establishes the baseline.
        let mut written = Vec::new();
        assert_eq!(
            written_since(memory as u64, size, None, &mut |offset, len| written
                .push((offset, len))),
            Some(0)
        );
        assert!(written.is_empty());
        unsafe { *(memory as *mut u8) = 1 };
        assert_eq!(
            written_since(memory as u64, size, Some(0), &mut |offset, len| written
                .push((offset, len))),
            Some(0)
        );
        assert_eq!(written, [(0, 0x1000)]);
        unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
    }
}
