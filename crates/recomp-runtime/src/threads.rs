//! Guest threads. Each runs on its own host thread.
//!
//! By default one thread runs at a time: a guest thread holds the run lock
//! while it executes guest code, gives it up while it blocks in the kernel,
//! and, once its time slice is used and another thread is waiting, at a call
//! boundary. Switching only there keeps exclusive load/store pairs atomic.
//!
//! With `RECOMP_PARALLEL=1` no thread takes the lock: guest threads really
//! run at the same time on host cores, as they do on hardware. Exclusive pairs
//! then contend for real (the runtime keeps a per-thread reservation and does a
//! compare-and-swap), data barriers become host fences, and the services guard
//! their own state. The serialized mode stays the default while the parallel
//! path is being validated. A parallel-mode yield pauses for a short bounded
//! spin (`RECOMP_YIELD_SPIN`) before handing over to the host scheduler, so a
//! poll loop does not make a syscall on every iteration; 0 restores a plain
//! yield.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use recomp_services::kernel::{kernel, CURRENT_THREAD};

use crate::{cpu_init, cpu_set, dispatch, tls_set, RuntimeError, RuntimeResult};

/// Bytes of thread-local region per thread (Horizon's TLS slot size).
const TLS_SLOT: u64 = 0x200;
/// Time a thread may run while others wait before it yields. Short, so
/// threads advance together as they do on separate cores.
const TIME_SLICE: Duration = Duration::from_micros(100);

/// Pause iterations before a parallel-mode yield (`RECOMP_YIELD_SPIN`).
///
/// A guest `svcSleepThread(0)` yield has no wait object the runtime can
/// check: the yielding code polls its own word after the yield returns. The
/// spin is therefore a bounded pause budget, kept short so it cannot delay
/// the handoff by more than a few microseconds. 0 yields immediately.
const YIELD_SPIN_DEFAULT: u32 = 64;
/// Upper bound for `RECOMP_YIELD_SPIN`, so a typo cannot spin for seconds.
const YIELD_SPIN_MAX: u32 = 1_000_000;

/// The spin budget for a parallel-mode yield (`RECOMP_YIELD_SPIN`).
fn yield_spin() -> u32 {
    static SPIN: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *SPIN.get_or_init(|| parse_yield_spin(std::env::var("RECOMP_YIELD_SPIN").ok().as_deref()))
}

fn parse_yield_spin(value: Option<&str>) -> u32 {
    value
        .and_then(|value| value.trim().parse::<u32>().ok())
        .map_or(YIELD_SPIN_DEFAULT, |iterations| {
            iterations.min(YIELD_SPIN_MAX)
        })
}

/// Pause briefly before giving the host scheduler its turn. `spin_loop` is
/// a `pause` on x86-64, which also keeps the core from wasting power while
/// it waits. The default budget is roughly a microsecond.
#[inline]
fn spin_before_yield() {
    for _ in 0..yield_spin() {
        core::hint::spin_loop();
    }
}

/// Whether guest threads run at the same time, without the run lock.
pub(crate) fn parallel() -> bool {
    static PARALLEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PARALLEL.get_or_init(|| std::env::var_os("RECOMP_PARALLEL").is_some())
}

/// The time slice, or `None` when threads switch only where they block
/// (`RECOMP_TIME_SLICE_US=<microseconds>`, 0 for never; for telling a
/// guest race from a scheduling assumption).
fn time_slice() -> Option<Duration> {
    static SLICE: std::sync::OnceLock<Option<Duration>> = std::sync::OnceLock::new();
    *SLICE.get_or_init(|| {
        match std::env::var("RECOMP_TIME_SLICE_US")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(0) => None,
            Some(us) => Some(Duration::from_micros(us)),
            None => Some(TIME_SLICE),
        }
    })
}
/// Host stack for each guest thread on desktop hosts: lifted guest calls
/// nest on it.
const HOST_STACK_DESKTOP_MB: usize = 256;
/// Android devices often have a 39-bit user address space, where reserving
/// 256 MiB of stack per guest thread competes with the guest memory
/// reservation and the host's own mappings. 16 MiB still gives lifted calls
/// several times the guest's stack depth.
const HOST_STACK_ANDROID_MB: usize = 16;
/// Bounds for `RECOMP_THREAD_STACK_MB`; a typo cannot reserve gigabytes per
/// thread or starve the guest call stack.
const HOST_STACK_MIN_MB: usize = 1;
const HOST_STACK_MAX_MB: usize = 1024;

/// The host stack size for each guest thread, in MiB. Desktop keeps the
/// previous 256 MiB default; Android starts at 16 MiB.
fn host_stack_size() -> usize {
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        let default = if cfg!(target_os = "android") {
            HOST_STACK_ANDROID_MB
        } else {
            HOST_STACK_DESKTOP_MB
        };
        let value = std::env::var("RECOMP_THREAD_STACK_MB").ok();
        parse_host_stack_mb(value.as_deref(), default) << 20
    })
}

fn parse_host_stack_mb(value: Option<&str>, default_mb: usize) -> usize {
    match value.and_then(|value| value.trim().parse::<usize>().ok()) {
        Some(0) | None => default_mb,
        Some(mb) => mb.clamp(HOST_STACK_MIN_MB, HOST_STACK_MAX_MB),
    }
}

/// Priority the main thread runs at (the usual NPDM value for games).
const MAIN_THREAD_PRIORITY: u32 = 44;
/// A waiting thread is let in once it has waited this long, whatever its
/// priority: on hardware lower priorities still run, on other cores.
const STARVATION_LIMIT: Duration = Duration::from_millis(10);

/// The starvation limit (`RECOMP_STARVATION_MS` overrides it).
fn starvation_limit() -> Duration {
    static LIMIT: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("RECOMP_STARVATION_MS")
            .ok()
            .and_then(|ms| ms.parse().ok())
            .map_or(STARVATION_LIMIT, Duration::from_millis)
    })
}

/// Where a thread waiting for the run lock stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turn {
    /// Preempted by a higher-priority thread: first among its priority, as
    /// a preempted thread is on hardware.
    Preempted = 0,
    /// Woken from a kernel wait, or newly started: a running thread of
    /// lower priority gives way to it.
    Woken = 1,
    /// Yielded or used up its time slice.
    Queued = 2,
}

/// Who the lock goes to when it is released.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Handoff {
    /// The highest-priority waiter: the releasing thread blocks or is
    /// preempted.
    Priority,
    /// The longest-waiting thread: a yield or an expired time slice.
    Oldest,
}

#[derive(Debug)]
struct Waiter {
    ticket: u64,
    priority: u32,
    /// The core the thread was created for.
    core: usize,
    turn: Turn,
    since: Instant,
}

/// Cores a thread can be created for.
const CORES: usize = 4;

struct RunState {
    next_ticket: u64,
    waiters: Vec<Waiter>,
    /// The ticket that may take the lock next.
    granted: Option<u64>,
    held: bool,
}

/// The lock a guest thread holds while it runs. It is handed on by
/// priority, like Horizon's scheduler on one core, except that yields and
/// expired time slices pass it to the longest-waiting thread so lower
/// priorities are not starved (on hardware they run on other cores).
struct RunLock {
    state: Mutex<RunState>,
    turn: Condvar,
    waiting: AtomicUsize,
    /// The best priority among woken waiters (`u32::MAX` if none): a
    /// running thread of lower priority gives way at its next call.
    urgent: AtomicU32,
    /// The same per core: a woken thread of equal priority on another core
    /// also gets to run at once, as it would on hardware.
    urgent_by_core: [AtomicU32; CORES],
}

static RUN_LOCK: RunLock = RunLock {
    state: Mutex::new(RunState {
        next_ticket: 0,
        waiters: Vec::new(),
        granted: None,
        held: false,
    }),
    turn: Condvar::new(),
    waiting: AtomicUsize::new(0),
    urgent: AtomicU32::new(u32::MAX),
    urgent_by_core: [const { AtomicU32::new(u32::MAX) }; CORES],
};

/// Calls made by guest code, to tell a busy process from a stuck one. Each
/// thread counts in a thread-local and adds its count here every
/// `PROGRESS_BATCH` calls: with threads on several cores, one counter every
/// call touches would bounce between them millions of times a second.
static PROGRESS: AtomicU64 = AtomicU64::new(0);
const PROGRESS_BATCH: u32 = 1024;

/// Count one guest call (or loop check) toward `PROGRESS`.
#[inline]
/// Calls generated code made straight to their targets (see
/// `recomp_abi::dispatch`), reported in batches.
pub(crate) fn count_calls(count: u32) {
    PROGRESS.fetch_add(u64::from(count), Ordering::Relaxed);
}

fn count_progress() {
    #[cfg(all(windows, recomp_fast_tls))]
    let flush = {
        let next = LOCAL_PROGRESS.get() + 1;
        if next == PROGRESS_BATCH {
            LOCAL_PROGRESS.set(0);
            true
        } else {
            LOCAL_PROGRESS.set(next);
            false
        }
    };
    #[cfg(not(all(windows, recomp_fast_tls)))]
    let flush = LOCAL_PROGRESS.with(|count| {
        let next = count.get() + 1;
        if next == PROGRESS_BATCH {
            count.set(0);
            true
        } else {
            count.set(next);
            false
        }
    });
    if flush {
        PROGRESS.fetch_add(u64::from(PROGRESS_BATCH), Ordering::Relaxed);
    }
}

#[cfg(not(all(windows, recomp_fast_tls)))]
thread_local! {
    static LOCAL_PROGRESS: Cell<u32> = const { Cell::new(0) };
    static CURRENT: Cell<u32> = const { Cell::new(0) };
    static PRIORITY: Cell<u32> = const { Cell::new(MAIN_THREAD_PRIORITY) };
    static CORE: Cell<usize> = const { Cell::new(0) };
    static SLICE_START: Cell<Option<Instant>> = const { Cell::new(None) };
    static CALLS: Cell<u32> = const { Cell::new(0) };
}

#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static LOCAL_PROGRESS: Cell<u32> = const { Cell::new(0) };
#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static CURRENT: Cell<u32> = const { Cell::new(0) };
#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static PRIORITY: Cell<u32> = const { Cell::new(MAIN_THREAD_PRIORITY) };
#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static CORE: Cell<usize> = const { Cell::new(0) };
#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static SLICE_START: Cell<Option<Instant>> = const { Cell::new(None) };
#[cfg(all(windows, recomp_fast_tls))]
#[thread_local]
static CALLS: Cell<u32> = const { Cell::new(0) };

/// The waiter the lock goes to. A thread preempted by the one blocking now
/// resumes first, as it would at once on its own core; otherwise a thread
/// that has waited too long goes before higher priorities.
fn pick(waiters: &[Waiter], handoff: Handoff, now: Instant) -> Option<u64> {
    let oldest = waiters.iter().min_by_key(|waiter| waiter.ticket)?;
    if handoff == Handoff::Oldest {
        return Some(oldest.ticket);
    }
    let best = waiters
        .iter()
        .min_by_key(|waiter| (waiter.priority, waiter.turn as u8, waiter.ticket))?;
    if best.turn != Turn::Preempted && now.duration_since(oldest.since) >= starvation_limit() {
        return Some(oldest.ticket);
    }
    Some(best.ticket)
}

fn lock_state() -> std::sync::MutexGuard<'static, RunState> {
    RUN_LOCK
        .state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn update_urgent(state: &RunState) {
    let mut by_core = [u32::MAX; CORES];
    for waiter in state
        .waiters
        .iter()
        .filter(|waiter| waiter.turn == Turn::Woken)
    {
        let best = &mut by_core[waiter.core];
        *best = (*best).min(waiter.priority);
    }
    for (core, urgent) in by_core.iter().enumerate() {
        RUN_LOCK.urgent_by_core[core].store(*urgent, Ordering::Relaxed);
    }
    let urgent = by_core.into_iter().min().unwrap_or(u32::MAX);
    RUN_LOCK.urgent.store(urgent, Ordering::Relaxed);
}

/// Whether a thread running at `priority` on `core` gives way to the woken
/// threads (`urgent_by_core`: the best woken priority per core): to a better
/// priority on any core, as a single core would, and to an equal priority on
/// another core, which on hardware would run at the same time. LM3's render
/// thread (core 1) wakes a job thread (core 2) of its priority that must read
/// the swapchain index before the render thread resets it.
fn gives_way(priority: u32, core: usize, urgent_by_core: &[u32; CORES]) -> bool {
    urgent_by_core
        .iter()
        .enumerate()
        .any(|(other, urgent)| *urgent < priority || (other != core && *urgent == priority))
}

/// The core a thread runs on: its ideal core, with the process default
/// (`-2`) and out-of-range values on core 0.
fn core_of(core: i32) -> usize {
    usize::try_from(core)
        .ok()
        .filter(|core| *core < CORES)
        .unwrap_or(0)
}

fn acquire_as(turn: Turn) {
    let (priority, core) = {
        let threads = threads();
        let thread = current_thread();
        (
            threads
                .priorities
                .get(&thread)
                .copied()
                .unwrap_or(MAIN_THREAD_PRIORITY),
            threads.cores.get(&thread).copied().unwrap_or(0),
        )
    };
    #[cfg(all(windows, recomp_fast_tls))]
    {
        PRIORITY.set(priority);
        CORE.set(core);
    }
    #[cfg(not(all(windows, recomp_fast_tls)))]
    {
        PRIORITY.with(|cell| cell.set(priority));
        CORE.with(|cell| cell.set(core));
    }
    RUN_LOCK.waiting.fetch_add(1, Ordering::SeqCst);
    let mut state = lock_state();
    let ticket = state.next_ticket;
    state.next_ticket += 1;
    state.waiters.push(Waiter {
        ticket,
        priority,
        core,
        turn,
        since: Instant::now(),
    });
    if !state.held && state.granted.is_none() {
        state.granted = pick(&state.waiters, Handoff::Priority, Instant::now());
        RUN_LOCK.turn.notify_all();
    }
    update_urgent(&state);
    while state.granted != Some(ticket) {
        state = RUN_LOCK
            .turn
            .wait(state)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
    }
    state.granted = None;
    state.held = true;
    state.waiters.retain(|waiter| waiter.ticket != ticket);
    update_urgent(&state);
    drop(state);
    RUN_LOCK.waiting.fetch_sub(1, Ordering::SeqCst);
    #[cfg(all(windows, recomp_fast_tls))]
    SLICE_START.set(Some(Instant::now()));
    #[cfg(not(all(windows, recomp_fast_tls)))]
    SLICE_START.with(|start| start.set(Some(Instant::now())));
}

fn release_to(handoff: Handoff) {
    let mut state = lock_state();
    state.held = false;
    state.granted = pick(&state.waiters, handoff, Instant::now());
    RUN_LOCK.turn.notify_all();
}

/// Take the run lock after being woken (a kernel wait ended, a sleep
/// elapsed, or the thread just started). In parallel mode there is no lock
/// to take.
pub fn acquire() {
    if parallel() {
        return;
    }
    acquire_as(Turn::Woken);
}

/// Give the run lock up to block: the highest-priority waiter runs next.
pub fn release() {
    if parallel() {
        return;
    }
    release_to(Handoff::Priority);
}

/// Let every thread that is waiting run first. In parallel mode there is no
/// run queue to hand over, so a short bounded spin (`RECOMP_YIELD_SPIN`)
/// precedes the host yield: it avoids a `sched_yield` syscall when the
/// other thread sets the condition almost at once, and still yields after
/// at most a few microseconds. The spin checks no waking condition because
/// the runtime cannot see the guest word the yielding code polls.
pub fn yield_now() {
    if parallel() {
        spin_before_yield();
        std::thread::yield_now();
        return;
    }
    release_to(Handoff::Oldest);
    acquire_as(Turn::Queued);
}

/// Called on every guest call: give way to a woken thread of higher
/// priority, and yield once the time slice is used up and another thread
/// wants to run. In parallel mode threads need neither: the host scheduler
/// puts them on cores, so this only records progress.
#[inline]
pub fn preempt_point() {
    count_progress();
    if parallel() {
        return;
    }
    if give_way_to_urgent() {
        return;
    }
    #[cfg(all(windows, recomp_fast_tls))]
    let calls = {
        let next = CALLS.get().wrapping_add(1);
        CALLS.set(next);
        next
    };
    #[cfg(not(all(windows, recomp_fast_tls)))]
    let calls = CALLS.with(|calls| {
        let next = calls.get().wrapping_add(1);
        calls.set(next);
        next
    });
    if calls.is_multiple_of(256) {
        yield_if_slice_used();
    }
}

/// Called every few thousand iterations of a loop in guest code (which may
/// make no calls at all): the same checks as at a call, the slice every
/// time.
pub fn loop_point() {
    count_progress();
    if parallel() {
        return;
    }
    if !give_way_to_urgent() {
        yield_if_slice_used();
    }
}

/// Preempt the running thread for a woken one of higher priority, or of
/// equal priority on another core (see `gives_way`).
#[inline]
fn give_way_to_urgent() -> bool {
    let urgent = RUN_LOCK.urgent.load(Ordering::Relaxed);
    #[cfg(all(windows, recomp_fast_tls))]
    let priority = PRIORITY.get();
    #[cfg(not(all(windows, recomp_fast_tls)))]
    let priority = PRIORITY.with(Cell::get);
    if urgent == u32::MAX || urgent > priority {
        return false;
    }
    let by_core = std::array::from_fn(|core| RUN_LOCK.urgent_by_core[core].load(Ordering::Relaxed));
    #[cfg(all(windows, recomp_fast_tls))]
    let core = CORE.get();
    #[cfg(not(all(windows, recomp_fast_tls)))]
    let core = CORE.with(Cell::get);
    if !gives_way(priority, core, &by_core) {
        return false;
    }
    release_to(Handoff::Priority);
    acquire_as(Turn::Preempted);
    true
}

fn yield_if_slice_used() {
    if RUN_LOCK.waiting.load(Ordering::Relaxed) == 0 {
        return;
    }
    let Some(slice) = time_slice() else {
        return;
    };
    #[cfg(all(windows, recomp_fast_tls))]
    let expired = SLICE_START.get().is_some_and(|at| at.elapsed() >= slice);
    #[cfg(not(all(windows, recomp_fast_tls)))]
    let expired = SLICE_START.with(|start| start.get().is_some_and(|at| at.elapsed() >= slice));
    if expired {
        yield_now();
    }
}

/// The calling guest thread's handle (pseudo-handles resolve to it).
pub fn current_thread() -> u32 {
    #[cfg(all(windows, recomp_fast_tls))]
    return CURRENT.get();
    #[cfg(not(all(windows, recomp_fast_tls)))]
    CURRENT.with(Cell::get)
}

pub fn resolve(handle: u32) -> u32 {
    if handle == CURRENT_THREAD {
        current_thread()
    } else {
        handle
    }
}

struct PendingThread {
    entry: u64,
    arg: u64,
    stack_top: u64,
    /// The creating thread's FPCR: Horizon clones the creator's FPU control
    /// state into a new thread (`KThread::Initialize`).
    fpcr: u64,
}

struct Threads {
    pending: BTreeMap<u32, PendingThread>,
    priorities: BTreeMap<u32, u32>,
    /// Each thread's core (see `core_of`).
    cores: BTreeMap<u32, usize>,
    /// Thread-local region: base, slot count and slots in use (slot 0 is
    /// the main thread's).
    tls_base: u64,
    tls_slots: usize,
    tls_used: Vec<bool>,
}

static THREADS: Mutex<Threads> = Mutex::new(Threads {
    pending: BTreeMap::new(),
    priorities: BTreeMap::new(),
    cores: BTreeMap::new(),
    tls_base: 0,
    tls_slots: 0,
    tls_used: Vec::new(),
});

fn threads() -> std::sync::MutexGuard<'static, Threads> {
    THREADS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Make the calling host thread the main guest thread: it owns `handle`
/// and the thread-local slot at `tls`, which sits at the start of the
/// thread-local region.
pub fn adopt_main_thread(handle: u32, tls: u64) {
    kernel().set_blocking_hooks(release, acquire);
    kernel().adopt_thread(handle);
    let region = crate::memory::query_block(tls)
        .filter(|block| block.state == recomp_services::ipc::memory_state::THREAD_LOCAL);
    {
        let mut threads = threads();
        threads.tls_base = tls;
        threads.tls_slots = region.map_or(1, |block| {
            ((block.base + block.size - tls) / TLS_SLOT).max(1) as usize
        });
        threads.tls_used = vec![false; threads.tls_slots];
        // Slot 0 is the main thread's; slot 1 holds the process's user
        // exception context (svcGetInfo 20).
        for used in threads.tls_used.iter_mut().take(2) {
            *used = true;
        }
    }
    threads().priorities.insert(handle, MAIN_THREAD_PRIORITY);
    #[cfg(all(windows, recomp_fast_tls))]
    CURRENT.set(handle);
    #[cfg(not(all(windows, recomp_fast_tls)))]
    CURRENT.with(|current| current.set(handle));
    acquire();
    spawn_progress_monitor();
}

/// svcGetInfo(UserExceptionContextAddress): a thread-local block the
/// kernel reserves for the process (the SDK keeps its dying message there).
pub fn user_exception_context() -> u64 {
    let threads = threads();
    if threads.tls_slots > 1 {
        threads.tls_base + TLS_SLOT
    } else {
        0
    }
}

/// svcCreateThread: a thread object that runs once started.
pub fn create_thread(entry: u64, arg: u64, stack_top: u64, priority: u32, core: i32) -> u32 {
    let handle = kernel().create_thread();
    eprintln!(
        "[recomp-runtime] thread {handle:#x} created by {:#x}: entry {entry:#x}, arg {arg:#x}, priority {priority}, core {core}",
        current_thread()
    );
    let mut threads = threads();
    threads.pending.insert(
        handle,
        PendingThread {
            entry,
            arg,
            stack_top,
            fpcr: crate::fpcr_get(),
        },
    );
    threads.priorities.insert(handle, priority);
    threads.cores.insert(handle, core_of(core));
    handle
}

pub fn priority(handle: u32) -> Option<u32> {
    threads().priorities.get(&resolve(handle)).copied()
}

pub fn set_priority(handle: u32, priority: u32) {
    threads().priorities.insert(resolve(handle), priority);
}

fn allocate_tls() -> RuntimeResult<u64> {
    let mut threads = threads();
    let slot = threads
        .tls_used
        .iter()
        .position(|used| !used)
        .ok_or_else(|| RuntimeError::StubbedSyscall {
            name: format!(
                "svcStartThread: all {} thread-local slots are in use (enlarge the tls region)",
                threads.tls_slots
            ),
        })?;
    threads.tls_used[slot] = true;
    let address = threads.tls_base + slot as u64 * TLS_SLOT;
    drop(threads);
    for offset in (0..TLS_SLOT).step_by(8) {
        crate::mem_store_u64(address + offset, 0)?;
    }
    Ok(address)
}

fn free_tls(address: u64) {
    let mut threads = threads();
    let slot = ((address - threads.tls_base) / TLS_SLOT) as usize;
    if let Some(used) = threads.tls_used.get_mut(slot) {
        *used = false;
    }
}

/// svcStartThread: run the guest thread on a new host thread. Like
/// Horizon, it starts with X0 = argument, X1 = its handle and SP at its
/// stack top.
pub fn start_thread(handle: u32) -> RuntimeResult<()> {
    let thread = threads()
        .pending
        .remove(&handle)
        .ok_or_else(|| RuntimeError::StubbedSyscall {
            name: format!("svcStartThread: unknown thread handle {handle:#x}"),
        })?;
    let tls = allocate_tls()?;
    std::thread::Builder::new()
        .name(format!("guest-{handle:x}"))
        .stack_size(host_stack_size())
        .spawn(move || {
            #[cfg(all(windows, recomp_fast_tls))]
            CURRENT.set(handle);
            #[cfg(not(all(windows, recomp_fast_tls)))]
            CURRENT.with(|current| current.set(handle));
            acquire();
            tls_set(tls);
            crate::fpcr_set(thread.fpcr);
            cpu_init(thread.stack_top);
            cpu_set(0, thread.arg as i64);
            cpu_set(1, i64::from(handle));
            if let Err(fault) = dispatch(thread.entry) {
                let exited = matches!(
                    fault.downcast_ref::<RuntimeError>(),
                    Some(RuntimeError::ThreadExit)
                );
                if !exited {
                    // The fault was reported where it happened; a crashed
                    // thread takes the process down as it would on hardware.
                    eprintln!(
                        "[recomp-runtime] guest thread {handle:#x} (entry {:#x}) stopped: {fault}",
                        thread.entry
                    );
                    crate::let_the_log_drain();
                    std::process::exit(1);
                }
            }
            free_tls(tls);
            kernel().exit_thread(handle);
            release();
        })
        .map_err(|error| RuntimeError::Io {
            message: format!("spawning guest thread: {error}"),
        })?;
    Ok(())
}

/// svcSleepThread: 0 and negative values yield; others sleep without
/// holding the run lock.
pub fn sleep(ns: i64) {
    if ns <= 0 {
        yield_now();
        return;
    }
    release();
    std::thread::sleep(Duration::from_nanos(ns as u64));
    acquire();
}

/// Report when guest code stops making calls for a while, with how many
/// threads are blocked in the kernel, so a hang is visible.
fn spawn_progress_monitor() {
    // `RECOMP_DUMP_WAITERS_AT=<seconds>`: list the threads blocked in the
    // kernel at that time, with the words they wait on (finding deadlocks).
    if let Some(seconds) = std::env::var("RECOMP_DUMP_WAITERS_AT")
        .ok()
        .and_then(|seconds| seconds.parse().ok())
    {
        let _ = std::thread::Builder::new()
            .name("recomp-waiters".to_string())
            .spawn(move || {
                std::thread::sleep(Duration::from_secs(seconds));
                for (kind, key, tag) in kernel().waiter_list() {
                    let value = crate::mem_load_u32(key)
                        .map_or("unreadable".to_string(), |value| format!("{value:#x}"));
                    eprintln!(
                        "[recomp-runtime] waiter: {kind} at {key:#x} (now {value}) tag {tag:#x}"
                    );
                }
                for (syncpoint, threshold) in recomp_services::gpu::waiters() {
                    eprintln!(
                        "[recomp-runtime] waiter: GPU syncpoint {syncpoint} waiting for {threshold} (now {})",
                        recomp_services::gpu::syncpoint_value(syncpoint)
                    );
                }
            });
    }
    let _ = std::thread::Builder::new()
        .name("recomp-progress".to_string())
        .spawn(|| {
            let mut last = PROGRESS.load(Ordering::Relaxed);
            let mut reported = false;
            loop {
                std::thread::sleep(Duration::from_secs(5));
                let now = PROGRESS.load(Ordering::Relaxed);
                if now == last && !reported {
                    eprintln!(
                        "[recomp-runtime] no guest calls for 5s; {} waiter(s) blocked in the kernel, {} blocked on a GPU syncpoint, {} thread(s) waiting to run{}",
                        kernel().waiting(),
                        recomp_services::gpu::waiting(),
                        RUN_LOCK.waiting.load(Ordering::Relaxed),
                        if parallel() { " (parallel mode: no run lock)" } else { "" }
                    );
                    reported = true;
                } else if now != last {
                    reported = false;
                }
                last = now;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::{
        core_of, gives_way, parse_host_stack_mb, parse_yield_spin, pick, Handoff, Turn, Waiter,
        CORES, HOST_STACK_ANDROID_MB, HOST_STACK_DESKTOP_MB, HOST_STACK_MAX_MB, HOST_STACK_MIN_MB,
        YIELD_SPIN_DEFAULT, YIELD_SPIN_MAX,
    };
    use std::time::{Duration, Instant};

    fn waiter(ticket: u64, priority: u32, turn: Turn, since: Instant) -> Waiter {
        Waiter {
            ticket,
            priority,
            core: 0,
            turn,
            since,
        }
    }

    #[test]
    fn woken_threads_preempt_by_priority_and_by_core() {
        let mut urgent = [u32::MAX; CORES];
        assert!(!gives_way(44, 1, &urgent), "nothing woken");
        urgent[1] = 44;
        assert!(!gives_way(44, 1, &urgent), "equal priority, same core");
        urgent[1] = 40;
        assert!(gives_way(44, 1, &urgent), "better priority");
        urgent = [u32::MAX; CORES];
        urgent[2] = 44;
        assert!(gives_way(44, 1, &urgent), "equal priority, its own core");
        urgent[2] = 59;
        assert!(!gives_way(44, 1, &urgent), "worse priority waits its turn");
        assert_eq!((core_of(-2), core_of(2), core_of(7)), (0, 2, 0));
    }

    #[test]
    fn the_lock_goes_by_priority_then_by_age() {
        let now = Instant::now();
        let waiters = [
            waiter(1, 44, Turn::Queued, now),
            waiter(2, 44, Turn::Preempted, now),
            waiter(3, 30, Turn::Woken, now),
            waiter(4, 59, Turn::Queued, now),
        ];
        // Blocking hands the lock to the best priority, yielding to the
        // longest-waiting thread.
        assert_eq!(pick(&waiters, Handoff::Priority, now), Some(3));
        assert_eq!(pick(&waiters, Handoff::Oldest, now), Some(1));
        // A preempted thread resumes before others of its priority.
        assert_eq!(pick(&waiters[..2], Handoff::Priority, now), Some(2));
        // A thread that has waited too long goes first, but not before a
        // preempted thread resumes.
        let late = now + Duration::from_millis(20);
        let starved = [
            waiter(1, 59, Turn::Queued, now),
            waiter(2, 30, Turn::Woken, late),
        ];
        assert_eq!(pick(&starved, Handoff::Priority, late), Some(1));
        let preempted = [
            waiter(1, 44, Turn::Queued, now),
            waiter(2, 44, Turn::Preempted, late),
        ];
        assert_eq!(pick(&preempted, Handoff::Priority, late), Some(2));
        assert_eq!(pick(&[], Handoff::Priority, now), None);
    }

    #[test]
    fn thread_stack_size_is_bounded_and_configurable() {
        // Unset and invalid values keep the per-platform default.
        assert_eq!(
            parse_host_stack_mb(None, HOST_STACK_DESKTOP_MB),
            HOST_STACK_DESKTOP_MB
        );
        assert_eq!(
            parse_host_stack_mb(Some(""), HOST_STACK_ANDROID_MB),
            HOST_STACK_ANDROID_MB
        );
        assert_eq!(
            parse_host_stack_mb(Some("0"), HOST_STACK_ANDROID_MB),
            HOST_STACK_ANDROID_MB
        );
        assert_eq!(
            parse_host_stack_mb(Some("large"), HOST_STACK_DESKTOP_MB),
            HOST_STACK_DESKTOP_MB
        );
        // A configured value is clamped to a usable range, including huge
        // (unparsable as usize) and tiny values.
        assert_eq!(parse_host_stack_mb(Some("8"), 16), 8);
        assert_eq!(parse_host_stack_mb(Some(" 32 "), 16), 32);
        assert_eq!(parse_host_stack_mb(Some("1"), 16), HOST_STACK_MIN_MB);
        assert_eq!(
            parse_host_stack_mb(Some(&(HOST_STACK_MAX_MB + 1).to_string()), 16),
            HOST_STACK_MAX_MB
        );
        assert_eq!(
            parse_host_stack_mb(Some("999999999999999999999999"), 16),
            16
        );
        // The defaults themselves are inside the range and keep desktop at
        // 256 MiB and Android at 16 MiB.
        assert!((HOST_STACK_MIN_MB..=HOST_STACK_MAX_MB).contains(&HOST_STACK_DESKTOP_MB));
        assert!((HOST_STACK_MIN_MB..=HOST_STACK_MAX_MB).contains(&HOST_STACK_ANDROID_MB));
    }

    #[test]
    fn yield_spin_budget_parses_zero_and_clamps() {
        // Unset or invalid entries use the default; 0 is the documented
        // "yield immediately" value, not an error.
        assert_eq!(parse_yield_spin(None), YIELD_SPIN_DEFAULT);
        assert_eq!(parse_yield_spin(Some("")), YIELD_SPIN_DEFAULT);
        assert_eq!(parse_yield_spin(Some("0")), 0);
        assert_eq!(parse_yield_spin(Some("1")), 1);
        assert_eq!(parse_yield_spin(Some("128")), 128);
        assert_eq!(parse_yield_spin(Some(" 64 ")), 64);
        assert_eq!(parse_yield_spin(Some("-1")), YIELD_SPIN_DEFAULT);
        assert_eq!(parse_yield_spin(Some("pause")), YIELD_SPIN_DEFAULT);
        // Values beyond u32 and beyond the spin cap do not overflow.
        assert_eq!(parse_yield_spin(Some("4294967296")), YIELD_SPIN_DEFAULT);
        assert_eq!(
            parse_yield_spin(Some(&(u64::from(YIELD_SPIN_MAX) + 1).to_string())),
            YIELD_SPIN_MAX
        );
        assert_eq!(parse_yield_spin(Some("4294967295")), YIELD_SPIN_MAX);
    }
}
