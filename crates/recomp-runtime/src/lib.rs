#![cfg_attr(all(windows, recomp_fast_tls), feature(thread_local))]

use std::fmt;

mod abi_impl;
mod audio;
mod boot;
mod fastmem;
mod homebrew;
mod host_functions;
mod input;
mod input_replay;
mod memory;
mod profile;
mod script;
mod svc_stats;
pub mod threads;
#[cfg(any(target_os = "linux", windows))]
mod write_watch;
/// Other targets (Android's NDK, for example) use the hashing fallback.
#[cfg(not(any(target_os = "linux", windows)))]
mod write_watch {
    pub(crate) fn written_since(
        _address: u64,
        _len: usize,
        _since: Option<u64>,
        _written: &mut dyn FnMut(usize, usize),
    ) -> Option<u64> {
        None
    }
}

pub const ABI_VERSION: &str = "0.1.0";

pub use audio::{AudioBackend, AudioBuffer, AudioError, StubAudioBackend};
pub use boot::{
    BootAssets, BootContext, BootError, BootPlan, BootStep, BootTrace, ServiceCallSpec,
};
pub use homebrew::{
    entrypoint_shim, DeterministicClock, InputEvent, InputQueue, LoaderConfig, LoaderConfigBuilder,
    LoaderConfigEntry, LoaderConfigKey, NroEntrypoint, RuntimeManifest, ServiceStub, NRO_ENTRY_X1,
};
pub use input::{InputBackend, InputFrame, StubInputBackend};
pub use input_replay::{
    InputMarker, InputMetadata, InputPlayback, InputScript, InputScriptEvent, InputScriptMarker,
    TimingMode,
};
pub use memory::{
    init_memory, recomp_mem_load_u16, recomp_mem_load_u32, recomp_mem_load_u64, recomp_mem_load_u8,
    recomp_mem_store_u16, recomp_mem_store_u32, recomp_mem_store_u64, recomp_mem_store_u8,
    MemoryInitSegment, MemoryLayout, MemoryLayoutError, MemoryPermissions, MemoryRegionSpec,
    MemoryStatus, MemoryZeroSegment,
};
use memory::{
    mem_load_u32 as raw_mem_load_u32, mem_load_u64 as raw_mem_load_u64,
    mem_load_u8 as raw_mem_load_u8, mem_store_u32 as raw_mem_store_u32,
    mem_store_u64 as raw_mem_store_u64, mem_store_u8 as raw_mem_store_u8,
};
pub use recomp_abi::{Fault, GuestResult, SvcOutcome};
pub use recomp_gfx::{
    CommandStream, FrameDescriptor, GraphicsBackend, GraphicsError, GraphicsPresenter, StubBackend,
    StubPresenter,
};
pub use recomp_services::assets::{asset_roots, resolve_asset, resolve_relative_asset};
pub use recomp_services::{
    kernel_svc_id, kernel_svc_known, kernel_svc_name, register_stubbed_services, stub_handler,
    KernelSvcDispatcher, KernelSvcResult, ServiceAccessControl, ServiceCall, ServiceError,
    ServiceLogger, ServiceRegistry, ServiceStubSpec, StubBehavior, KERNEL_SVC_NOT_ALLOWED,
    KERNEL_SVC_TABLE, KERNEL_SVC_UNSUPPORTED,
};
pub use recomp_timing::Scheduler;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceMode {
    Handheld,
    Docked,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub performance_mode: PerformanceMode,
}

impl RuntimeConfig {
    pub fn new(performance_mode: PerformanceMode) -> Self {
        Self { performance_mode }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            performance_mode: PerformanceMode::Handheld,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("stubbed syscall: {name}")]
    StubbedSyscall { name: String },
    #[error("missing loader config key: {key:?}")]
    MissingLoaderConfigKey { key: LoaderConfigKey },
    #[error("runtime manifest serialization failed: {message}")]
    ManifestSerialize { message: String },
    #[error("io error: {message}")]
    Io { message: String },
    #[error("memory layout error: {0}")]
    MemoryLayout(#[from] MemoryLayoutError),
    #[error("memory access error {status:?} at {address:#x} size {size}")]
    MemoryAccess {
        status: MemoryStatus,
        address: u64,
        size: usize,
    },
    #[error("indirect branch to unmapped guest address {address:#x}")]
    InvalidBranch { address: u64 },
    #[error("unimplemented kernel call {name}")]
    UnimplementedSvc { name: String },
    /// svcExitThread unwinds the guest thread's frames with this error.
    #[error("guest thread exited")]
    ThreadExit,
    #[error("guest called svcBreak(reason={reason:#x}, info={info:#x}, size={size:#x}){detail}")]
    GuestBreak {
        reason: u64,
        info: u64,
        size: u64,
        /// Decoded payload, e.g. the Result an SDK abort carries.
        detail: String,
    },
}

pub type RuntimeResult<T> = Result<T, RuntimeError>;

impl From<RuntimeError> for Fault {
    fn from(error: RuntimeError) -> Self {
        // Runtime errors that reach guest code stop it, so this is the point
        // to record where the guest was. A thread exit is not an error.
        if !matches!(error, RuntimeError::ThreadExit) {
            eprintln!(
                "[recomp-runtime] {error} (in the guest block at {:#x})",
                recomp_abi::current_block()
            );
            report_guest_stack();
        }
        Fault::new(error)
    }
}

/// A Horizon Result as `0xVALUE (2MMM-DDDD)`: module plus 2000, then the
/// description.
pub fn describe_result(code: u32) -> String {
    let module = code & 0x1FF;
    let description = (code >> 9) & 0x1FFF;
    format!("{code:#x} ({:04}-{description:04})", module + 2000)
}

/// The guest call stack, innermost first. Lifted functions are named
/// `fn_<guest address>`, so a symbolized host backtrace is the guest stack.
fn guest_stack() -> String {
    let trace = std::backtrace::Backtrace::force_capture().to_string();
    let frames: Vec<&str> = trace
        .lines()
        .filter_map(|line| {
            let name = line.trim().split_once(": ")?.1;
            let function = name.rsplit("::").next()?;
            (function.starts_with("fn_") || function == "entry").then_some(function)
        })
        .collect();
    frames.join(" <- ")
}

/// `RECOMP_WATCH_STORE=<address>[:<length>]` (hex): report every guest
/// store that touches the range, with its block, thread and stack (a data
/// watchpoint, for finding who corrupts or frees an object).
static WATCH_STORE: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();

#[inline]
pub fn watch_store(address: u64, size: u64, value: u64) {
    let &(start, end) = WATCH_STORE.get_or_init(|| {
        let hex = |text: &str| u64::from_str_radix(text.trim_start_matches("0x"), 16).ok();
        std::env::var("RECOMP_WATCH_STORE")
            .ok()
            .and_then(|spec| {
                let (start, length) = spec.split_once(':').unwrap_or((spec.as_str(), "8"));
                let start = hex(start)?;
                Some((start, start + hex(length)?))
            })
            .unwrap_or((0, 0))
    });
    if address < end && address + size > start {
        eprintln!(
            "[watch] store {value:#x} ({size} bytes) at {address:#x} in the guest block {:#x} t{:x}: {}",
            recomp_abi::current_block(),
            threads::current_thread(),
            guest_stack()
        );
    }
}

/// Print the guest call stack (innermost first), the registers and the
/// object `x0` points to.
pub fn report_guest_stack() {
    let frames = guest_stack();
    if !frames.is_empty() {
        eprintln!("[recomp-runtime] guest stack: {frames}");
    }
    // Lifted code keeps registers in locals and stores them at calls, so
    // these are current for a failed call and a little stale otherwise.
    let regs = cpu_regs();
    let named: Vec<String> = regs
        .iter()
        .enumerate()
        .map(|(index, value)| match index {
            31 => format!("sp={value:#x}"),
            _ => format!("x{index}={value:#x}"),
        })
        .collect();
    eprintln!(
        "[recomp-runtime] registers at the last call: {}",
        named.join(" ")
    );
    // A failed virtual call: show the object x0 points to and its vtable.
    let words = |address: u64, count: u64| -> Option<Vec<u64>> {
        (0..count)
            .map(|index| mem_load_u64(address + index * 8).ok())
            .collect()
    };
    if let Some(object) = words(regs[0] as u64, 4) {
        let vtable = words(object[0], 6)
            .map(|slots| format!("{slots:#x?}"))
            .unwrap_or_else(|| "unreadable".to_string());
        eprintln!("[recomp-runtime] [x0] = {object:#x?}; vtable [x0][0..6] = {vtable}",);
    }
}

impl From<MemoryLayoutError> for Fault {
    fn from(error: MemoryLayoutError) -> Self {
        Fault::new(error)
    }
}

pub fn abi_version() -> &'static str {
    ABI_VERSION
}

/// Reset the guest register file and install the initial stack pointer.
pub fn cpu_init(sp: u64) {
    let mut regs = [0; 32];
    regs[31] = sp as i64;
    recomp_abi::cpu_replace(regs);
}

/// Handle the kernel passes to a process's main thread in X1.
pub const MAIN_THREAD_HANDLE: u64 = 0xF001;

/// Put the guest CPU in the state Horizon starts a process's main thread
/// in: X0 = 0 (no exception context), X1 = the main thread handle, SP at the
/// top of the main thread stack. TPIDRRO_EL0 is set when the IPC buffer
/// (the thread-local region) is attached.
pub fn boot_main_thread(stack_top: u64) {
    cpu_init(stack_top);
    cpu_set(1, MAIN_THREAD_HANDLE as i64);
    threads::adopt_main_thread(MAIN_THREAD_HANDLE as u32, tls_get());
}

pub fn cpu_get(index: usize) -> i64 {
    recomp_abi::cpu_get(index)
}

pub fn cpu_set(index: usize, value: i64) {
    recomp_abi::cpu_set(index, value);
}

pub fn cpu_vregs() -> [u128; 32] {
    recomp_abi::cpu_vregs()
}

pub fn cpu_vstore(values: &[(usize, u128)]) {
    recomp_abi::cpu_vstore(values);
}

/// Snapshot the guest register file (it lives in `recomp_abi`, where
/// generated code reaches it without a call).
pub fn cpu_regs() -> [i64; 32] {
    recomp_abi::cpu_regs()
}

/// Write several guest registers at once.
pub fn cpu_store(values: &[(usize, i64)]) {
    recomp_abi::cpu_store(values);
}

#[cfg(not(all(windows, recomp_fast_tls)))]
thread_local! {
    /// TPIDRRO_EL0: the kernel-owned thread-local region of the current guest
    /// thread (IPC message buffer, then nn::os per-thread state).
    static THREAD_POINTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// TPIDR_EL0: the guest's own thread pointer, written with MSR.
    static USER_THREAD_POINTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static FPCR: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static FPSR: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static THREAD_POINTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static USER_THREAD_POINTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static FPCR: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static FPSR: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };

pub fn tls_get() -> u64 {
    #[cfg(all(windows, recomp_fast_tls))]
    return THREAD_POINTER.get();
    #[cfg(not(all(windows, recomp_fast_tls)))]
    THREAD_POINTER.with(|value| value.get())
}

pub fn tls_set(value: u64) {
    #[cfg(all(windows, recomp_fast_tls))]
    THREAD_POINTER.set(value);
    #[cfg(not(all(windows, recomp_fast_tls)))]
    THREAD_POINTER.with(|cell| cell.set(value));
}

/// MRS of a system register reachable from EL0 (`recomp_abi::sysreg` ids).
pub fn sysreg_read(reg: u32) -> u64 {
    use recomp_abi::sysreg;
    match reg {
        sysreg::TPIDR_EL0 => {
            #[cfg(all(windows, recomp_fast_tls))]
            {
                USER_THREAD_POINTER.get()
            }
            #[cfg(not(all(windows, recomp_fast_tls)))]
            {
                USER_THREAD_POINTER.with(|value| value.get())
            }
        }
        sysreg::TPIDRRO_EL0 => tls_get(),
        sysreg::CNTPCT_EL0 => cntpct_el0(),
        sysreg::CNTFRQ_EL0 => cntfrq_el0(),
        sysreg::CTR_EL0 => ctr_el0(),
        sysreg::FPCR => fpcr_get(),
        sysreg::FPSR => fpsr_get(),
        _ => 0,
    }
}

/// MSR of a system register writable from EL0; other writes are ignored.
pub fn sysreg_write(reg: u32, value: u64) {
    use recomp_abi::sysreg;
    match reg {
        sysreg::TPIDR_EL0 => {
            #[cfg(all(windows, recomp_fast_tls))]
            USER_THREAD_POINTER.set(value);
            #[cfg(not(all(windows, recomp_fast_tls)))]
            USER_THREAD_POINTER.with(|cell| cell.set(value));
        }
        sysreg::FPCR => fpcr_set(value),
        sysreg::FPSR => fpsr_set(value),
        _ => {}
    }
}

/// CNTPCT_EL0: the system counter, at the 19.2 MHz CNTFRQ_EL0 reports and
/// from the start every thread shares (`recomp_services::clock`). It used to
/// count nanoseconds from each thread's first read: guest time ran 52 times
/// fast, and threads disagreed about it.
pub fn cntpct_el0() -> u64 {
    recomp_services::clock::ticks()
}

/// CNTFRQ_EL0: system counter frequency.
pub fn cntfrq_el0() -> u64 {
    recomp_services::clock::TICKS_PER_SECOND
}

/// CTR_EL0: cache type register. Values only matter for cache-line sizing.
pub fn ctr_el0() -> u64 {
    0x8444_C004
}

/// Give the log a moment before the process exits on a fatal error. An
/// Android app's stdout and stderr reach logcat through a pipe a thread of
/// the game library reads (Android sends them nowhere otherwise): exiting
/// at once lost the line that said why. Async-signal-safe (nanosleep).
pub(crate) fn let_the_log_drain() {
    #[cfg(target_os = "android")]
    {
        let pause = libc::timespec {
            tv_sec: 0,
            tv_nsec: 200_000_000,
        };
        // SAFETY: nanosleep only reads `pause`.
        unsafe { libc::nanosleep(&pause, std::ptr::null_mut()) };
    }
}

pub fn fpcr_get() -> u64 {
    #[cfg(all(windows, recomp_fast_tls))]
    return FPCR.get();
    #[cfg(not(all(windows, recomp_fast_tls)))]
    FPCR.with(|value| value.get())
}

pub fn fpcr_set(value: u64) {
    #[cfg(all(windows, recomp_fast_tls))]
    FPCR.set(value);
    #[cfg(not(all(windows, recomp_fast_tls)))]
    FPCR.with(|cell| cell.set(value));
    host_fp_mode(value);
    // Rounding mode, flush-to-zero and default NaN change floating-point
    // results; say once which settings the title uses.
    static SEEN: std::sync::Mutex<std::collections::BTreeSet<(u32, u64)>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    if SEEN
        .lock()
        .is_ok_and(|mut seen| seen.insert((threads::current_thread(), value)))
    {
        eprintln!(
            "[recomp-runtime] guest FPCR = {value:#x} (RMode {}, FZ {}, DN {}) t{:x}",
            (value >> 22) & 3,
            (value >> 24) & 1,
            (value >> 25) & 1,
            threads::current_thread()
        );
    }
}

/// Make the host thread's floating point follow FPCR.FZ: guest threads run
/// on their own host threads, so the SSE control register (MXCSR) can hold
/// the guest's flush-to-zero (FTZ, bit 15) and denormals-are-zero (DAZ,
/// bit 6), which together are what Arm's FZ does. Every host float
/// operation, the SIMD evaluator's included, then flushes as the hardware
/// does. LM3 sets FZ; without it, values a simulation should settle to zero
/// linger as denormals (and denormal arithmetic is slow on x86). An AArch64
/// host has the same bit in its own FPCR.
fn host_fp_mode(fpcr: u64) {
    #[cfg(target_arch = "aarch64")]
    {
        const FZ: u64 = 1 << 24;
        let mut host: u64;
        // SAFETY: FPCR only holds this thread's floating-point modes; the
        // value keeps every field but FZ.
        unsafe {
            std::arch::asm!("mrs {}, fpcr", out(reg) host, options(nostack));
            host = (host & !FZ) | (fpcr & FZ);
            std::arch::asm!("msr fpcr, {}", in(reg) host, options(nostack));
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        const FTZ_DAZ: u32 = (1 << 15) | (1 << 6);
        let mut csr: u32 = 0;
        // SAFETY: stmxcsr and ldmxcsr only move the SSE control register to
        // and from this local.
        unsafe {
            std::arch::asm!("stmxcsr [{}]", in(reg) &mut csr, options(nostack));
        }
        csr = if fpcr & (1 << 24) != 0 {
            csr | FTZ_DAZ
        } else {
            csr & !FTZ_DAZ
        };
        // SAFETY: as above; the value keeps every other MXCSR field.
        unsafe {
            std::arch::asm!("ldmxcsr [{}]", in(reg) &csr, options(nostack));
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = fpcr;
}

pub fn fpsr_get() -> u64 {
    #[cfg(all(windows, recomp_fast_tls))]
    return FPSR.get();
    #[cfg(not(all(windows, recomp_fast_tls)))]
    FPSR.with(|value| value.get())
}

pub fn fpsr_set(value: u64) {
    #[cfg(all(windows, recomp_fast_tls))]
    FPSR.set(value);
    #[cfg(not(all(windows, recomp_fast_tls)))]
    FPSR.with(|cell| cell.set(value));
}

pub fn init(config: &RuntimeConfig) {
    // The system counter starts here.
    recomp_services::clock::now_ns();
    // Before anything reads a setting the script may change.
    script::load_at_start();
    profile::start();
    println!(
        "[recomp-runtime] init performance_mode={:?}",
        config.performance_mode
    );
}

pub fn init_default_memory(layout: MemoryLayout) -> RuntimeResult<()> {
    init_memory(layout)?;
    Ok(())
}

pub fn apply_memory_image(
    init_segments: &[MemoryInitSegment],
    zero_segments: &[MemoryZeroSegment],
) -> RuntimeResult<()> {
    memory::apply_memory_image(init_segments, zero_segments).map_err(|status| {
        RuntimeError::MemoryAccess {
            status,
            address: 0,
            size: 0,
        }
    })
}

pub fn mem_load_u8(address: u64) -> RuntimeResult<u64> {
    memory::mem_load_u8(address).map_err(|status| RuntimeError::MemoryAccess {
        status,
        address,
        size: 1,
    })
}

pub fn mem_load_u16(address: u64) -> RuntimeResult<u64> {
    memory::mem_load_u16(address).map_err(|status| RuntimeError::MemoryAccess {
        status,
        address,
        size: 2,
    })
}

pub fn mem_load_u32(address: u64) -> RuntimeResult<u64> {
    memory::mem_load_u32(address).map_err(|status| RuntimeError::MemoryAccess {
        status,
        address,
        size: 4,
    })
}

pub fn mem_load_u64(address: u64) -> RuntimeResult<u64> {
    memory::mem_load_u64(address).map_err(|status| RuntimeError::MemoryAccess {
        status,
        address,
        size: 8,
    })
}

pub fn mem_store_u8(address: u64, value: u64) -> RuntimeResult<()> {
    memory::mem_store_u8(address, value).map_err(|status| RuntimeError::MemoryAccess {
        status,
        address,
        size: 1,
    })
}

pub fn mem_store_u16(address: u64, value: u64) -> RuntimeResult<()> {
    memory::mem_store_u16(address, value).map_err(|status| RuntimeError::MemoryAccess {
        status,
        address,
        size: 2,
    })
}

pub fn mem_store_u32(address: u64, value: u64) -> RuntimeResult<()> {
    memory::mem_store_u32(address, value).map_err(|status| RuntimeError::MemoryAccess {
        status,
        address,
        size: 4,
    })
}

pub fn mem_store_u64(address: u64, value: u64) -> RuntimeResult<()> {
    memory::mem_store_u64(address, value).map_err(|status| RuntimeError::MemoryAccess {
        status,
        address,
        size: 8,
    })
}

/// LDXR and the pair forms: read the value(s) and reserve the address for the
/// store-exclusive. The reservation belongs to the calling thread.
pub fn mem_load_exclusive(address: u64, size: u64, pair: bool) -> RuntimeResult<(u64, u64)> {
    memory::mem_load_exclusive(address, size as usize, pair).map_err(|status| {
        RuntimeError::MemoryAccess {
            status,
            address,
            size: size as usize,
        }
    })
}

/// STXR and the pair forms: write only if the address still holds what the
/// load observed. Returns the instruction's success flag: 0 stored, 1 lost.
pub fn mem_store_exclusive(
    address: u64,
    size: u64,
    pair: bool,
    value: u64,
    second_value: u64,
) -> RuntimeResult<u64> {
    memory::mem_store_exclusive(address, size as usize, pair, value, second_value).map_err(
        |status| RuntimeError::MemoryAccess {
            status,
            address,
            size: size as usize,
        },
    )
}

/// CLREX: drop the calling thread's reservation.
pub fn clear_exclusive() {
    memory::clear_exclusive();
}

/// DMB and friends: guest threads run on host threads, so guest ordering has
/// to be real.
pub fn memory_fence() {
    memory::memory_barrier();
}

pub fn syscall_log(name: &str, args: &[i64]) -> RuntimeResult<()> {
    println!("[recomp-runtime] syscall {name} args={}", ArgsDisplay(args));
    Ok(())
}

pub fn syscall_noop(_name: &str, _args: &[i64]) -> RuntimeResult<()> {
    Ok(())
}

pub fn syscall_panic(name: &str, _args: &[i64]) -> RuntimeResult<()> {
    Err(RuntimeError::StubbedSyscall {
        name: name.to_string(),
    })
}

static KERNEL_DISPATCHER: std::sync::OnceLock<KernelSvcDispatcher> = std::sync::OnceLock::new();
static PROCESS_EXIT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(-1);

/// Install the kernel SVC dispatcher (allow-list, heap window). Call before entry.
pub fn init_kernel_services(allowed: Vec<u64>, heap_base: u64, max_heap_size: u64) {
    KERNEL_DISPATCHER.get_or_init(|| {
        let mut dispatcher = KernelSvcDispatcher::new(allowed, heap_base, max_heap_size);
        if let Some((base, size)) = fastmem::window_stack_region() {
            println!(
                "[recomp-runtime] thread stacks go to {base:#x}, inside the guest memory window"
            );
            dispatcher = dispatcher.with_stack_region(base, size);
        }
        let memory: std::sync::Arc<dyn recomp_services::MemoryAccess> =
            std::sync::Arc::new(GlobalMemoryAccess);
        dispatcher.attach_memory(std::sync::Arc::clone(&memory));
        // The GPU worker decodes command lists on its own thread, so it needs
        // the process-wide memory view rather than a caller's borrow.
        recomp_services::gpu::attach_memory(std::sync::Arc::clone(&memory));
        recomp_services::hid::attach_memory(memory);
        dispatcher
    });
}

/// Guest-memory view used by the IPC/fs: hub and memory-reporting SVCs.
#[derive(Debug)]
struct GlobalMemoryAccess;

impl recomp_services::MemoryAccess for GlobalMemoryAccess {
    fn load_u8(&self, addr: u64) -> Option<u8> {
        raw_mem_load_u8(addr).ok().map(|value| value as u8)
    }
    fn load_u32(&self, addr: u64) -> Option<u32> {
        raw_mem_load_u32(addr).ok().map(|value| value as u32)
    }
    fn load_u64(&self, addr: u64) -> Option<u64> {
        raw_mem_load_u64(addr).ok()
    }
    fn store_u32(&self, addr: u64, value: u32) -> bool {
        raw_mem_store_u32(addr, value as u64).is_ok()
    }
    fn store_u64(&self, addr: u64, value: u64) -> bool {
        raw_mem_store_u64(addr, value).is_ok()
    }
    fn read_bytes(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        if let Ok(bytes) = memory::read_block(addr, len) {
            return Some(bytes);
        }
        // Spans regions: byte by byte.
        let mut out = Vec::with_capacity(len);
        for offset in 0..len {
            out.push(self.load_u8(addr + offset as u64)?);
        }
        Some(out)
    }
    fn written_since(
        &self,
        addr: u64,
        len: usize,
        since: Option<u64>,
        written: &mut dyn FnMut(usize, usize),
    ) -> Option<u64> {
        write_watch::written_since(addr, len, since, written)
    }
    fn read_with(&self, addr: u64, len: usize, read: &mut dyn FnMut(&[u8])) -> bool {
        if memory::with_block(addr, len, read) {
            return true;
        }
        let Some(bytes) = self.read_bytes(addr, len) else {
            return false;
        };
        read(&bytes);
        true
    }
    fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
        if memory::write_block(addr, bytes).is_ok() {
            return true;
        }
        for (offset, byte) in bytes.iter().enumerate() {
            if raw_mem_store_u8(addr + offset as u64, *byte as u64).is_err() {
                return false;
            }
        }
        true
    }
    fn query_block(&self, addr: u64) -> Option<recomp_services::ipc::MemoryBlock> {
        memory::query_block(addr)
    }
    fn update_u32(&self, addr: u64, update: &mut dyn FnMut(u32) -> u32) -> Option<u32> {
        memory::update_u32_exclusive(addr, update).ok()
    }
}

/// Attach the IPC hub. `romfs_image` is a host path to a decompressed RomFS
/// image served by fsp-srv; the in-memory window (`romfs_base`/`romfs_size`)
/// is used when no image is configured. `RECOMP_ROMFS` overrides the compiled
/// path when set (Android cannot use the build host's path).
pub fn init_ipc_services(
    romfs_base: u64,
    romfs_size: u64,
    romfs_image: Option<&std::path::Path>,
    ipc_buffer: Option<u64>,
) -> RuntimeResult<()> {
    let romfs_image = std::env::var_os("RECOMP_ROMFS")
        .filter(|path| !path.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| romfs_image.map(std::path::PathBuf::from));
    let hub = recomp_services::IpcHub::new(
        std::sync::Arc::new(GlobalMemoryAccess),
        romfs_base,
        romfs_size,
        romfs_image,
    );
    let dispatcher = KERNEL_DISPATCHER
        .get()
        .ok_or_else(|| RuntimeError::StubbedSyscall {
            name: "kernel services not initialized".to_string(),
        })?;
    dispatcher.attach_ipc(hub);
    if let Some(address) = ipc_buffer {
        // The SDK reads TPIDRRO_EL0 directly as the IPC buffer pointer.
        tls_set(address);
    }
    Ok(())
}

/// Dispatch one kernel SVC by NPDM name; returns the resulting output
/// registers. `x0` holds the kernel result code (0 on success); failures are
/// returned to the guest as on hardware. Only SVCs the kernel layer does not
/// implement stop execution.
pub fn svc_dispatch(name: &str, args: &[i64]) -> RuntimeResult<SvcOutcome> {
    let traced = svc_tracing();
    if traced {
        trace_svc_args(name, args);
    }
    // Unknown names keep the diagnostic path, including the name in the
    // error. Generated code is checked at recompilation time and uses the
    // numeric entry point below instead.
    let outcome = match kernel_svc_id(name) {
        Some(id) => dispatch_known_id(id, args),
        None => Err(RuntimeError::StubbedSyscall {
            name: name.to_string(),
        }),
    };
    if traced {
        trace_svc_outcome(name, &outcome);
    }
    outcome
}

/// Dispatch one kernel SVC by its numeric id. Generated code resolves the
/// name at recompilation time and calls this, so the string and the
/// name-to-id `HashMap` probe are no longer in the per-SVC hot path. The
/// string entry point above remains for diagnostics and unknown names.
#[inline]
pub fn svc_dispatch_id(id: u64, args: &[i64]) -> RuntimeResult<SvcOutcome> {
    if svc_tracing() {
        return svc_dispatch_id_traced(id, args);
    }
    dispatch_known_id(id, args)
}

/// The generated-code entry point for `recomp_abi::svc_dispatch_id`.
#[no_mangle]
pub fn __recomp_rt_svc_id(id: u64, args: &[i64]) -> Result<SvcOutcome, Fault> {
    // The error conversion records diagnostics and allocates, so keep it off
    // the successful path; an inlined `?` on the untraced branch compiles to
    // the same move.
    if svc_tracing() {
        return svc_dispatch_id_traced(id, args).map_err(Fault::from);
    }
    dispatch_known_id(id, args).map_err(Fault::from)
}

/// The tracing path is cold: it resolves the name and writes two lines per
/// SVC, so it stays out of `__recomp_rt_svc_id`'s hot body.
#[cold]
fn svc_dispatch_id_traced(id: u64, args: &[i64]) -> RuntimeResult<SvcOutcome> {
    let name = kernel_svc_name(id).unwrap_or("svc_unknown");
    trace_svc_args(name, args);
    let outcome = dispatch_known_id(id, args);
    trace_svc_outcome(name, &outcome);
    outcome
}

/// Cached `RECOMP_TRACE_CALLS || RECOMP_TRACE_SVC`: 0 unknown, 1 off, 2 on.
static SVC_TRACE_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
#[cfg(test)]
const SVC_TRACE_UNKNOWN: u8 = 0;
const SVC_TRACE_OFF: u8 = 1;
const SVC_TRACE_ON: u8 = 2;

/// Whether SVC tracing is on, without a `OnceLock` check per SVC. The first
/// call reads the environment; every later call is one relaxed atomic load.
#[inline(always)]
fn svc_tracing() -> bool {
    match SVC_TRACE_STATE.load(std::sync::atomic::Ordering::Relaxed) {
        SVC_TRACE_ON => true,
        SVC_TRACE_OFF => false,
        _ => svc_tracing_init(),
    }
}

#[cold]
fn svc_tracing_init() -> bool {
    let enabled = std::env::var_os("RECOMP_TRACE_CALLS").is_some()
        || std::env::var_os("RECOMP_TRACE_SVC").is_some();
    SVC_TRACE_STATE.store(
        if enabled { SVC_TRACE_ON } else { SVC_TRACE_OFF },
        std::sync::atomic::Ordering::Relaxed,
    );
    enabled
}

#[cold]
fn trace_svc_args(name: &str, args: &[i64]) {
    eprintln!(
        "[trace] svc {name}({:#x}, {:#x}, {:#x}, {:#x}) on thread {:#x}",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0),
        args.get(3).copied().unwrap_or(0),
        threads::current_thread()
    );
}

#[cold]
fn trace_svc_outcome(name: &str, outcome: &RuntimeResult<SvcOutcome>) {
    match outcome {
        Ok(out) => eprintln!("[trace] svc {name} -> {:#x} {:?}", out.x0, out.x1),
        Err(error) => eprintln!("[trace] svc {name} -> {error}"),
    }
}

#[inline]
fn svc_outcome(x0: i64) -> SvcOutcome {
    SvcOutcome {
        x0,
        x1: None,
        x2: None,
    }
}

#[inline]
fn svc_outcome_with_x1(x1: u64) -> SvcOutcome {
    SvcOutcome {
        x0: 0,
        x1: Some(x1 as i64),
        x2: None,
    }
}

/// Common entry for the numeric and named dispatchers. Hot, simple SVCs
/// take a direct path; everything else goes through the generic dispatcher.
#[inline]
fn dispatch_known_id(id: u64, args: &[i64]) -> RuntimeResult<SvcOutcome> {
    svc_stats::record(id);
    match svc_fast_path(id, args) {
        Some(outcome) => outcome,
        None => svc_dispatch_id_inner(id, args),
    }
}

/// Direct handlers for SVCs that do not need the dispatcher's state lock,
/// IPC hub or argument buffer. Each arm returns `None` back to the generic
/// path when the dispatcher is not initialized, when the id is not on the
/// allow-list, or when the call needs the full dispatcher for another
/// reason. Results must stay identical to `thread_svc` and
/// `KernelSvcDispatcher::dispatch`.
#[inline]
fn svc_fast_path(id: u64, args: &[i64]) -> Option<RuntimeResult<SvcOutcome>> {
    let arg = |index: usize| args.get(index).copied().unwrap_or(0) as u64;
    // The generic path serves threads/events before its allow-list check,
    // and dispatcher-served ids after it. Check only presence here; the
    // allow-list-aware arms below check it themselves.
    let dispatcher = KERNEL_DISPATCHER.get()?;
    match id {
        // svcSleepThread: the threads module owns the run-lock handoff and
        // the 0/negative yield semantics. No kernel state, no argument
        // buffer, and no dispatcher lock.
        0x0B => {
            threads::sleep(args.first().copied().unwrap_or(0));
            Some(Ok(svc_outcome(0)))
        }
        // svcGetSystemTick: read the system counter directly instead of
        // taking the kernel state lock.
        0x1E if dispatcher.allows(id) => {
            Some(Ok(svc_outcome(recomp_services::clock::ticks() as i64)))
        }
        // GetThreadPriority(handle) -> priority; unknown handles keep the
        // threads module's default, as in `thread_svc`.
        0x0C => Some(Ok(svc_outcome_with_x1(u64::from(
            threads::priority(threads::resolve(arg(1) as u32)).unwrap_or(44),
        )))),
        // SetThreadPriority(handle, priority).
        0x0D => {
            threads::set_priority(threads::resolve(arg(0) as u32), arg(1) as u32);
            Some(Ok(svc_outcome(0)))
        }
        // GetThreadCoreMask: the dispatcher reports priority 0 and the
        // all-cores mask.
        0x0E if dispatcher.allows(id) => Some(Ok(SvcOutcome {
            x0: 0,
            x1: Some(0),
            x2: Some(0xF),
        })),
        0x0F if dispatcher.allows(id) => Some(Ok(svc_outcome(0))),
        // GetCurrentProcessorNumber.
        0x10 if dispatcher.allows(id) => Some(Ok(svc_outcome(0))),
        // SignalEvent / ClearEvent / ResetSignal: kernel state, but no
        // scheduler handoff.
        0x11 => Some(Ok(svc_outcome(
            recomp_services::kernel::kernel().signal_event(arg(0) as u32) as i64,
        ))),
        0x12 => Some(Ok(svc_outcome(
            recomp_services::kernel::kernel().clear_event(arg(0) as u32) as i64,
        ))),
        0x17 => Some(Ok(svc_outcome(
            recomp_services::kernel::kernel().reset_signal(arg(0) as u32) as i64,
        ))),
        // ArbitrateUnlock and the light synchronisation signals only wake
        // waiters; they never hand the scheduler away.
        0x1B => Some(Ok(svc_outcome(
            recomp_services::kernel::kernel().arbitrate_unlock(&GlobalMemoryAccess, arg(0)) as i64,
        ))),
        0x1D => {
            recomp_services::kernel::kernel().signal_process_wide_key(
                &GlobalMemoryAccess,
                arg(0),
                arg(1) as i32,
            );
            Some(Ok(svc_outcome(0)))
        }
        0x35 => Some(Ok(svc_outcome(
            recomp_services::kernel::kernel().signal_to_address(
                &GlobalMemoryAccess,
                arg(0),
                arg(1) as u32,
                arg(2) as i32,
                arg(3) as i32,
            ) as i64,
        ))),
        // CloseHandle: only kernel-owned handles take the direct path; an
        // IPC-owned or unknown handle keeps the generic path (hub close).
        0x16 => {
            let handle = arg(0) as u32;
            if recomp_services::kernel::kernel().owns(handle) {
                recomp_services::kernel::kernel().close(handle);
                Some(Ok(svc_outcome(0)))
            } else {
                None
            }
        }
        // GetProcessId.
        0x24 if dispatcher.allows(id) => Some(Ok(svc_outcome_with_x1(0x1_234_567_890))),
        // GetThreadId(handle) -> id.
        0x25 => {
            match recomp_services::kernel::kernel().thread_id(threads::resolve(arg(1) as u32)) {
                Some(thread_id) => Some(Ok(svc_outcome_with_x1(thread_id))),
                None => Some(Ok(svc_outcome(
                    recomp_services::kernel::RESULT_INVALID_HANDLE as i64,
                ))),
            }
        }
        _ => None,
    }
}

fn svc_dispatch_id_inner(id: u64, args: &[i64]) -> RuntimeResult<SvcOutcome> {
    // Generated code only emits ids it resolved from the table, but the ABI
    // entry point is public. This is one table load, not a map probe.
    if !kernel_svc_known(id) {
        return Err(RuntimeError::StubbedSyscall {
            name: format!("svc 0x{id:02x}"),
        });
    }
    let dispatcher = KERNEL_DISPATCHER
        .get()
        .ok_or_else(|| RuntimeError::StubbedSyscall {
            name: "kernel services not initialized".to_string(),
        })?;
    // An SVC takes at most eight arguments. `i64` and `u64` have the same
    // layout, and every bit pattern is a valid `u64`, so the dispatcher can
    // read the caller's slice directly instead of copying it to a stack array
    // on every call.
    //
    // SAFETY: the source is a shared `i64` slice; the cast preserves its
    // length and alignment, and no mutable reference aliases it.
    let args: &[u64] = unsafe { std::slice::from_raw_parts(args.as_ptr().cast(), args.len()) };
    let args = &args[..args.len().min(8)];
    let arg = |index: usize| args.get(index).copied().unwrap_or(0);
    match id {
        // svcBreak(reason, info, size): the guest aborted.
        0x26 => {
            // SDK aborts pass the failing Result as a 4-byte payload.
            let detail = match (arg(2), mem_load_u32(arg(1))) {
                (4, Ok(code)) => format!(" result {}", describe_result(code as u32)),
                _ => String::new(),
            };
            return Err(RuntimeError::GuestBreak {
                reason: arg(0),
                info: arg(1),
                size: arg(2),
                detail,
            });
        }
        // svcOutputDebugString(str, len): the SDK's own diagnostics.
        0x27 => {
            let len = arg(1).min(0x1000) as usize;
            let bytes = (0..len)
                .map(|offset| mem_load_u8(arg(0) + offset as u64).unwrap_or(b'?' as u64) as u8)
                .collect::<Vec<_>>();
            println!("[guest] {}", String::from_utf8_lossy(&bytes).trim_end());
            return Ok(SvcOutcome {
                x0: 0,
                x1: None,
                x2: None,
            });
        }
        _ => {}
    }
    if let Some(outcome) = thread_svc(id, args)? {
        return Ok(outcome);
    }
    let result;
    if id == 0x21 {
        // svcSendSyncRequest takes only the session handle; the message is
        // in the calling thread's thread-local region. Most calls pass just
        // the handle, so avoid copying the tail in that case.
        let len = args.len();
        if len <= 2 {
            let request = [arg(0), tls_get()];
            result = dispatcher.dispatch(id, &request);
        } else {
            let mut request = [0u64; 8];
            request[..len].copy_from_slice(args);
            request[1] = tls_get();
            result = dispatcher.dispatch(id, &request[..len]);
        }
    } else {
        result = dispatcher.dispatch(id, args);
    }
    if result.terminates {
        PROCESS_EXIT.store(result.code as i64, std::sync::atomic::Ordering::SeqCst);
        return Ok(SvcOutcome {
            x0: result.code as i64,
            x1: None,
            x2: None,
        });
    }
    if result.code == KERNEL_SVC_UNSUPPORTED || result.code == KERNEL_SVC_NOT_ALLOWED {
        // Cold path: only the diagnostic needs the name, so the numeric
        // hot path never pays for the reverse lookup.
        let name = kernel_svc_name(id).unwrap_or("svc_unknown");
        return Err(RuntimeError::UnimplementedSvc {
            name: format!("{name} (0x{id:02x})"),
        });
    }
    if result.code != 0 {
        return Ok(SvcOutcome {
            x0: result.code as i64,
            x1: None,
            x2: None,
        });
    }
    let (writes_x1, writes_x2) = recomp_services::kernel_svc_writes(id);
    Ok(SvcOutcome {
        x0: result.x0 as i64,
        x1: result.x1.filter(|_| writes_x1).map(|value| value as i64),
        x2: result.x2.filter(|_| writes_x2).map(|value| value as i64),
    })
}

/// Threads, events and synchronization SVCs, served by the threads module
/// and the services kernel. None for every other SVC.
fn thread_svc(id: u64, args: &[u64]) -> RuntimeResult<Option<SvcOutcome>> {
    use recomp_services::kernel::kernel;
    let arg = |index: usize| args.get(index).copied().unwrap_or(0);
    let handle = |index: usize| threads::resolve(arg(index) as u32);
    let memory = GlobalMemoryAccess;
    let done = |x0: u64| {
        Ok(Some(SvcOutcome {
            x0: x0 as i64,
            x1: None,
            x2: None,
        }))
    };
    let with_x1 = |x0: u64, x1: u64| {
        Ok(Some(SvcOutcome {
            x0: x0 as i64,
            x1: Some(x1 as i64),
            x2: None,
        }))
    };
    let rw = memory::MemoryPermissions::new(true, true, false);
    match id {
        // MapMemory(dst, src, size): Horizon makes the source inaccessible
        // while the alias exists, so moving the bytes there and back on
        // UnmapMemory is equivalent.
        0x04 => {
            let (dst, src, size) = (arg(0), arg(1), arg(2));
            let Ok(bytes) = memory::read_block(src, size as usize) else {
                return done(recomp_services::kernel::RESULT_INVALID_MEMORY);
            };
            let spec = memory::MemoryRegionSpec::new(format!("stack-alias-{dst:x}"), dst, size, rw);
            match memory::map_region(spec, &bytes) {
                Ok(()) => done(0),
                Err(_) => done(recomp_services::kernel::RESULT_INVALID_MEMORY),
            }
        }
        0x05 => {
            let (dst, src) = (arg(0), arg(1));
            match memory::unmap_region(dst) {
                Some(bytes) if memory::write_block(src, &bytes).is_ok() => done(0),
                _ => done(recomp_services::kernel::RESULT_INVALID_MEMORY),
            }
        }
        // MapSharedMemory(handle, address, size, permission)
        0x13 => {
            let (handle, address, size) = (arg(0) as u32, arg(1), arg(2));
            let Some(bytes) = kernel().map_shared_memory(handle, address) else {
                return done(recomp_services::kernel::RESULT_INVALID_HANDLE);
            };
            if bytes.len() as u64 != size {
                kernel().unmap_shared_memory(handle, address);
                return done(recomp_services::kernel::RESULT_INVALID_SIZE);
            }
            let spec =
                memory::MemoryRegionSpec::new(format!("shared-{handle:x}"), address, size, rw);
            match memory::map_region(spec, &bytes) {
                Ok(()) => done(0),
                Err(_) => {
                    kernel().unmap_shared_memory(handle, address);
                    done(recomp_services::kernel::RESULT_INVALID_MEMORY)
                }
            }
        }
        0x14 => {
            let (handle, address) = (arg(0) as u32, arg(1));
            kernel().unmap_shared_memory(handle, address);
            memory::unmap_region(address);
            done(0)
        }
        // GetInfo(UserExceptionContextAddress, current process)
        0x29 if arg(1) == 20 => with_x1(0, threads::user_exception_context()),
        // CreateThread(entry, arg, stack_top, priority, core) -> handle
        0x08 => {
            let created =
                threads::create_thread(arg(1), arg(2), arg(3), arg(4) as u32, arg(5) as i32);
            with_x1(0, u64::from(created))
        }
        0x09 => {
            threads::start_thread(handle(0))?;
            done(0)
        }
        0x0A => Err(RuntimeError::ThreadExit),
        0x0B => {
            threads::sleep(arg(0) as i64);
            done(0)
        }
        // GetThreadPriority(handle) -> priority
        0x0C => with_x1(0, u64::from(threads::priority(handle(1)).unwrap_or(44))),
        0x0D => {
            threads::set_priority(handle(0), arg(1) as u32);
            done(0)
        }
        0x11 => done(kernel().signal_event(arg(0) as u32)),
        0x12 => done(kernel().clear_event(arg(0) as u32)),
        0x16 if kernel().owns(arg(0) as u32) => {
            kernel().close(arg(0) as u32);
            done(0)
        }
        0x17 => done(kernel().reset_signal(arg(0) as u32)),
        // WaitSynchronization(handles, count, timeout) -> index
        0x18 => {
            let count = arg(2).min(0x40);
            let handles: Vec<u32> = (0..count)
                .map(|index| {
                    mem_load_u32(arg(1) + index * 4).map(|value| threads::resolve(value as u32))
                })
                .collect::<RuntimeResult<_>>()?;
            match kernel().wait_synchronization(&handles, arg(3) as i64, threads::current_thread())
            {
                Ok(index) => with_x1(0, index as u64),
                Err(code) => done(code),
            }
        }
        0x19 => done(kernel().cancel_synchronization(handle(0))),
        0x1A => done(kernel().arbitrate_lock(&memory, arg(0) as u32, arg(1), arg(2) as u32)),
        0x1B => done(kernel().arbitrate_unlock(&memory, arg(0))),
        0x1C => done(kernel().wait_process_wide_key(
            &memory,
            arg(0),
            arg(1),
            arg(2) as u32,
            arg(3) as i64,
        )),
        0x1D => {
            kernel().signal_process_wide_key(&memory, arg(0), arg(1) as i32);
            done(0)
        }
        // GetThreadId(handle) -> id
        0x25 => match kernel().thread_id(handle(1)) {
            Some(thread_id) => with_x1(0, thread_id),
            None => done(recomp_services::kernel::RESULT_INVALID_HANDLE),
        },
        0x34 => done(kernel().wait_for_address(
            &memory,
            arg(0),
            arg(1) as u32,
            arg(2) as i32,
            arg(3) as i64,
        )),
        0x35 => done(kernel().signal_to_address(
            &memory,
            arg(0),
            arg(1) as u32,
            arg(2) as i32,
            arg(3) as i32,
        )),
        // CreateEvent -> (writable, readable)
        0x45 => {
            let (write, read) = kernel().create_event();
            Ok(Some(SvcOutcome {
                x0: 0,
                x1: Some(i64::from(write)),
                x2: Some(i64::from(read)),
            }))
        }
        _ => Ok(None),
    }
}

pub fn process_exit_code() -> Option<i64> {
    let value = PROCESS_EXIT.load(std::sync::atomic::Ordering::SeqCst);
    if value < 0 {
        None
    } else {
        Some(value)
    }
}

pub type GuestFn = fn() -> GuestResult;
pub type DispatchFn = fn(u64) -> GuestResult;

static GUEST_DISPATCHER: std::sync::atomic::AtomicPtr<()> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

pub fn register_dispatcher(f: DispatchFn) {
    GUEST_DISPATCHER.store(f as *mut (), std::sync::atomic::Ordering::Release);
}

/// The call table in use (a script's reload may replace it).
static TARGETS: std::sync::atomic::AtomicPtr<recomp_abi::Targets> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

fn targets() -> Option<&'static recomp_abi::Targets> {
    // SAFETY: installed tables are leaked, so they live for the program.
    unsafe { TARGETS.load(std::sync::atomic::Ordering::Acquire).as_ref() }
}

/// Register the program's functions so a call finds its target in one or
/// two probes; addresses not listed go to the registered dispatcher. When
/// the runtime has nothing to do at a call (parallel threads, no call
/// hooks), generated code then calls through the table without entering
/// the runtime at all. Functions a script hooks are listed through their
/// hooks (`script`).
pub fn register_targets(targets: &[(u64, GuestFn)]) {
    script::set_functions(targets);
    install_targets(script::table());
    script::start();
}

/// Make `table` the one calls go through. Threads may still be calling
/// through the previous one, so it is never freed.
pub(crate) fn install_targets(table: recomp_abi::Targets) {
    let table: &'static recomp_abi::Targets = Box::leak(Box::new(table));
    TARGETS.store(
        std::ptr::from_ref(table).cast_mut(),
        std::sync::atomic::Ordering::Release,
    );
    if threads::parallel() && !debug_hooks() {
        recomp_abi::publish_targets(Some(table));
        // A hook's trampoline is only in the table: calls to the function
        // generated code names must go through it while one is set.
        recomp_abi::allow_direct_calls(!script::hooks_set());
    }
}

/// Log indirect calls (target, x0-x3, returned x0) when `RECOMP_TRACE_CALLS`
/// is set: virtual calls and callbacks all go through `dispatch`.
static TRACE_CALLS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether any per-call debugging is on (`RECOMP_WATCH_MEMORY`,
/// `RECOMP_WATCH_STRINGS`, `RECOMP_WATCH_CALLS`, `RECOMP_TRACE_CALLS`).
fn debug_hooks() -> bool {
    static HOOKS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HOOKS.get_or_init(|| {
        [
            "RECOMP_WATCH_MEMORY",
            "RECOMP_WATCH_STRINGS",
            "RECOMP_WATCH_CALLS",
            "RECOMP_TRACE_CALLS",
        ]
        .iter()
        .any(|name| std::env::var_os(name).is_some())
    })
}

/// `RECOMP_WATCH_STRINGS=<address>:x<N>,...`: when the function at
/// `address` is called, print the C string `xN` points to (file paths,
/// names).
static WATCH_STRINGS: std::sync::OnceLock<Vec<(u64, usize)>> = std::sync::OnceLock::new();

fn watch_strings() -> &'static [(u64, usize)] {
    WATCH_STRINGS.get_or_init(|| {
        std::env::var("RECOMP_WATCH_STRINGS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|entry| {
                let (address, reg) = entry.split_once(":x")?;
                let address = u64::from_str_radix(address.trim_start_matches("0x"), 16).ok()?;
                Some((address, reg.parse().ok().filter(|reg| *reg < 31)?))
            })
            .collect()
    })
}

/// `RECOMP_WATCH_MEMORY=<address>:x<N>+<offset>:<length>,...`: when the
/// function at `address` is called, load the pointer at `xN + offset` and
/// hex-dump `length` bytes centred on it (a script's program counter, a
/// buffer cursor). `&x<N>+<offset>` dumps `length` bytes at `xN + offset`
/// itself (an object's fields).
static WATCH_MEMORY: std::sync::OnceLock<Vec<MemoryWatch>> = std::sync::OnceLock::new();

/// One `RECOMP_WATCH_MEMORY` entry.
struct MemoryWatch {
    function: u64,
    reg: usize,
    offset: u64,
    length: u64,
    /// `&xN`: dump at `xN + offset` instead of the pointer stored there.
    direct: bool,
}

fn watch_memory() -> &'static [MemoryWatch] {
    WATCH_MEMORY.get_or_init(|| {
        let hex = |text: &str| u64::from_str_radix(text.trim_start_matches("0x"), 16).ok();
        std::env::var("RECOMP_WATCH_MEMORY")
            .unwrap_or_default()
            .split(',')
            .filter_map(|entry| {
                let mut parts = entry.split(':');
                let address = hex(parts.next()?)?;
                let register = parts.next()?;
                let direct = register.starts_with('&');
                let (reg, offset) = register
                    .trim_start_matches('&')
                    .trim_start_matches('x')
                    .split_once('+')?;
                let length = hex(parts.next()?)?;
                Some(MemoryWatch {
                    function: address,
                    reg: reg.parse().ok().filter(|reg| *reg < 31)?,
                    offset: hex(offset)?,
                    length,
                    direct,
                })
            })
            .collect()
    })
}

/// `RECOMP_WATCH_CALLS=<address>,...`: log each call to these functions
/// with `x0`-`x3`, the return address and the thread, then what it returned
/// in `x0` (direct calls
/// included, unlike
/// `RECOMP_TRACE_CALLS`'s full trace).
static WATCH_CALLS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();

fn watch_calls() -> &'static [u64] {
    WATCH_CALLS.get_or_init(|| {
        std::env::var("RECOMP_WATCH_CALLS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|entry| u64::from_str_radix(entry.trim_start_matches("0x"), 16).ok())
            .collect()
    })
}

fn dump_watched_memory(address: u64) {
    for &MemoryWatch {
        function,
        reg,
        offset,
        length,
        direct,
    } in watch_memory()
    {
        if function != address {
            continue;
        }
        let field = (cpu_get(reg) as u64).wrapping_add(offset);
        let (pointer, start) = if direct {
            (field, field)
        } else {
            let Ok(pointer) = mem_load_u64(field) else {
                continue;
            };
            (pointer, pointer.wrapping_sub(length / 2))
        };
        let bytes: Vec<String> = (0..length)
            .map(|index| {
                mem_load_u8(start + index).map_or("??".to_string(), |byte| format!("{byte:02x}"))
            })
            .collect();
        eprintln!(
            "[watch] {address:#x}: [x{reg}+{offset:#x}] = {pointer:#x}; {start:#x}: {} t{:x}",
            bytes.join(" "),
            threads::current_thread()
        );
    }
}

fn guest_c_string(address: u64) -> String {
    let bytes: Vec<u8> = (0..256)
        .map_while(|offset| mem_load_u8(address + offset).ok().map(|byte| byte as u8))
        .take_while(|byte| *byte != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[inline]
pub fn dispatch(address: u64) -> GuestResult {
    let ptr = GUEST_DISPATCHER.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        return Err(RuntimeError::InvalidBranch { address }.into());
    }
    let f: DispatchFn = unsafe { std::mem::transmute(ptr) };
    threads::preempt_point();
    // Every guest call comes through here: with no watch or trace asked
    // for, go straight to the function.
    if !debug_hooks() {
        if let Some(function) = targets().and_then(|targets| targets.get(address)) {
            return function();
        }
        return f(address);
    }
    dump_watched_memory(address);
    for &(watched, reg) in watch_strings() {
        if watched == address {
            let text = guest_c_string(cpu_get(reg) as u64);
            eprintln!(
                "[watch] {address:#x}(x{reg} = \"{text}\") t{:x}",
                threads::current_thread()
            );
        }
    }
    if watch_calls().contains(&address) {
        let lr = cpu_get(30);
        eprintln!(
            "[watch] call {address:#x}({:#x}, {:#x}, {:#x}, {:#x}) lr {lr:#x} t{:x}",
            cpu_get(0),
            cpu_get(1),
            cpu_get(2),
            cpu_get(3),
            threads::current_thread()
        );
        let result = f(address);
        eprintln!(
            "[watch] ret  {address:#x} -> {:#x} lr {lr:#x} t{:x}",
            cpu_get(0),
            threads::current_thread()
        );
        return result;
    }
    if !*TRACE_CALLS.get_or_init(|| std::env::var_os("RECOMP_TRACE_CALLS").is_some()) {
        return f(address);
    }
    let regs = cpu_regs();
    let thread = threads::current_thread();
    eprintln!(
        "[trace] call {address:#x}({:#x}, {:#x}, {:#x}, {:#x}) t{thread:x}",
        regs[0], regs[1], regs[2], regs[3]
    );
    let result = f(address);
    eprintln!("[trace] ret  {address:#x} -> {:#x} t{thread:x}", cpu_get(0));
    result
}

pub struct Runtime {
    pub scheduler: Scheduler,
    pub services: ServiceRegistry,
    pub access: ServiceAccessControl,
    pub logger: ServiceLogger,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            services: ServiceRegistry::new(),
            access: ServiceAccessControl::default(),
            logger: ServiceLogger::default(),
        }
    }

    pub fn dispatch_service(&self, call: &ServiceCall) -> Result<(), ServiceError> {
        self.access.check(call)?;
        self.logger.log_call(call);
        self.services.call(call)
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

struct ArgsDisplay<'a>(&'a [i64]);

impl fmt::Display for ArgsDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (idx, value) in self.0.iter().enumerate() {
            if idx > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{value}")?;
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_find_every_listed_function() {
        fn first() -> GuestResult {
            Ok(())
        }
        fn second() -> GuestResult {
            Err(RuntimeError::InvalidBranch { address: 2 }.into())
        }
        let listed: Vec<(u64, GuestFn)> = (0..5000u64)
            .map(|index| {
                let function: GuestFn = if index % 2 == 0 { first } else { second };
                (0x800_4000 + index * 0x10, function)
            })
            .collect();
        let targets = recomp_abi::Targets::new(&listed);
        for &(address, function) in &listed {
            let found = targets.get(address).expect("listed");
            assert_eq!(found as usize, function as usize, "{address:#x}");
        }
        assert!(targets.get(0x800_4008).is_none());
        assert!(targets.get(0).is_none());
    }

    #[test]
    fn noop_syscall_is_ok() {
        assert!(syscall_noop("svc_test", &[]).is_ok());
    }

    #[test]
    fn panic_syscall_returns_error() {
        let err = syscall_panic("svc_test", &[]).unwrap_err();
        assert!(matches!(
            err,
            RuntimeError::StubbedSyscall { name } if name == "svc_test"
        ));
    }

    #[test]
    fn dispatch_service_respects_access_control() {
        let mut runtime = Runtime::new();
        runtime.services.register("svc_ok", |_| Ok(()));
        runtime.access = ServiceAccessControl::from_allowed(vec!["svc_ok".to_string()]);

        let ok_call = ServiceCall {
            client: "demo".to_string(),
            service: "svc_ok".to_string(),
            args: vec![],
        };
        assert!(runtime.dispatch_service(&ok_call).is_ok());

        let bad_call = ServiceCall {
            client: "demo".to_string(),
            service: "svc_bad".to_string(),
            args: vec![],
        };
        let err = runtime.dispatch_service(&bad_call).unwrap_err();
        assert!(matches!(err, ServiceError::AccessDenied(_)));
    }

    #[test]
    fn runtime_config_defaults_to_handheld() {
        let config = RuntimeConfig::default();
        assert!(matches!(config.performance_mode, PerformanceMode::Handheld));
    }

    /// Serializes the tests that toggle the process-wide trace variables.
    static SVC_TRACE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Restores `RECOMP_TRACE_*` and the cached tracing state after a test.
    struct SvcTraceEnv {
        trace_calls: Option<std::ffi::OsString>,
        trace_svc: Option<std::ffi::OsString>,
        state: u8,
    }

    impl SvcTraceEnv {
        fn capture() -> Self {
            Self {
                trace_calls: std::env::var_os("RECOMP_TRACE_CALLS"),
                trace_svc: std::env::var_os("RECOMP_TRACE_SVC"),
                state: SVC_TRACE_STATE.load(std::sync::atomic::Ordering::Relaxed),
            }
        }
    }

    impl Drop for SvcTraceEnv {
        fn drop(&mut self) {
            for (name, value) in [
                ("RECOMP_TRACE_CALLS", &self.trace_calls),
                ("RECOMP_TRACE_SVC", &self.trace_svc),
            ] {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
            SVC_TRACE_STATE.store(self.state, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[test]
    fn svc_tracing_caches_on_and_off() {
        let _lock = SVC_TRACE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _saved = SvcTraceEnv::capture();

        // An off state must stay off even if the environment changes later.
        std::env::set_var("RECOMP_TRACE_SVC", "1");
        std::env::remove_var("RECOMP_TRACE_CALLS");
        SVC_TRACE_STATE.store(SVC_TRACE_OFF, std::sync::atomic::Ordering::Relaxed);
        assert!(!svc_tracing());

        // An on state must stay on after the environment is cleared.
        std::env::remove_var("RECOMP_TRACE_SVC");
        SVC_TRACE_STATE.store(SVC_TRACE_ON, std::sync::atomic::Ordering::Relaxed);
        assert!(svc_tracing());
    }

    #[test]
    fn svc_tracing_initializes_once_from_the_environment() {
        let _lock = SVC_TRACE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _saved = SvcTraceEnv::capture();

        std::env::remove_var("RECOMP_TRACE_CALLS");
        std::env::remove_var("RECOMP_TRACE_SVC");
        SVC_TRACE_STATE.store(SVC_TRACE_UNKNOWN, std::sync::atomic::Ordering::Relaxed);
        assert!(!svc_tracing());
        assert_eq!(
            SVC_TRACE_STATE.load(std::sync::atomic::Ordering::Relaxed),
            SVC_TRACE_OFF
        );

        std::env::set_var("RECOMP_TRACE_SVC", "1");
        SVC_TRACE_STATE.store(SVC_TRACE_UNKNOWN, std::sync::atomic::Ordering::Relaxed);
        assert!(svc_tracing());
        assert_eq!(
            SVC_TRACE_STATE.load(std::sync::atomic::Ordering::Relaxed),
            SVC_TRACE_ON
        );

        // `RECOMP_TRACE_CALLS` enables the same trace, as before.
        std::env::remove_var("RECOMP_TRACE_SVC");
        std::env::set_var("RECOMP_TRACE_CALLS", "1");
        SVC_TRACE_STATE.store(SVC_TRACE_UNKNOWN, std::sync::atomic::Ordering::Relaxed);
        assert!(svc_tracing());
    }
}
