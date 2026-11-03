//! Kernel synchronization with real host threads.

use recomp_services::kernel::{
    Kernel, MUTEX_WAITERS, RESULT_CANCELLED, RESULT_INVALID_STATE, RESULT_TIMED_OUT,
};
use recomp_services::MemoryAccess;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Memory(Mutex<Vec<u8>>);

impl MemoryAccess for Memory {
    fn load_u8(&self, addr: u64) -> Option<u8> {
        self.0.lock().unwrap().get(addr as usize).copied()
    }
    fn load_u32(&self, addr: u64) -> Option<u32> {
        let bytes = self.read_bytes(addr, 4)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    }
    fn load_u64(&self, addr: u64) -> Option<u64> {
        let bytes = self.read_bytes(addr, 8)?;
        Some(u64::from_le_bytes(bytes.try_into().ok()?))
    }
    fn store_u32(&self, addr: u64, value: u32) -> bool {
        self.write_bytes(addr, &value.to_le_bytes())
    }
    fn store_u64(&self, addr: u64, value: u64) -> bool {
        self.write_bytes(addr, &value.to_le_bytes())
    }
    fn read_bytes(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        let memory = self.0.lock().unwrap();
        memory
            .get(addr as usize..addr as usize + len)
            .map(<[u8]>::to_vec)
    }
    fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
        let mut memory = self.0.lock().unwrap();
        match memory.get_mut(addr as usize..addr as usize + bytes.len()) {
            Some(slice) => {
                slice.copy_from_slice(bytes);
                true
            }
            None => false,
        }
    }
}

fn setup() -> (Arc<Kernel>, Arc<Memory>) {
    (
        Arc::new(Kernel::new()),
        Arc::new(Memory(Mutex::new(vec![0; 0x1000]))),
    )
}

/// Wait (bounded) until `kernel` has `count` blocked waiters.
fn until_waiting(kernel: &Kernel, count: usize) {
    let start = Instant::now();
    while kernel.waiting() != count {
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "waiters never reached {count}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn events_and_thread_exit_signal_waiters() {
    let (kernel, _) = setup();
    let (write, read) = kernel.create_event();
    assert_eq!(
        kernel.wait_synchronization(&[read], 0, 0),
        Err(RESULT_TIMED_OUT)
    );
    assert_eq!(kernel.reset_signal(read), RESULT_INVALID_STATE);
    assert_eq!(kernel.signal_event(write), 0);
    assert_eq!(kernel.wait_synchronization(&[read], -1, 0), Ok(0));
    // Events stay signalled until cleared.
    assert_eq!(kernel.wait_synchronization(&[read], 0, 0), Ok(0));
    assert_eq!(kernel.reset_signal(read), 0);
    assert_eq!(
        kernel.wait_synchronization(&[read], 1_000_000, 0),
        Err(RESULT_TIMED_OUT)
    );

    let thread = kernel.create_thread();
    let worker = {
        let kernel = kernel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            kernel.exit_thread(thread);
        })
    };
    assert_eq!(kernel.wait_synchronization(&[read, thread], -1, 0), Ok(1));
    worker.join().unwrap();
}

#[test]
fn arbitrated_mutex_is_handed_to_the_waiter() {
    let (kernel, mem) = setup();
    let (owner, waiter_tag) = (0x10, 0x20);
    // The owner holds the lock and someone has marked it contended.
    mem.store_u32(0x100, owner | MUTEX_WAITERS);
    let waiter = {
        let (kernel, mem) = (kernel.clone(), mem.clone());
        std::thread::spawn(move || kernel.arbitrate_lock(&*mem, owner, 0x100, waiter_tag))
    };
    until_waiting(&kernel, 1);
    assert_eq!(kernel.arbitrate_unlock(&*mem, 0x100), 0);
    assert_eq!(waiter.join().unwrap(), 0);
    assert_eq!(mem.load_u32(0x100), Some(waiter_tag));
    // A stale word (no waiters flag) returns at once.
    assert_eq!(kernel.arbitrate_lock(&*mem, owner, 0x100, waiter_tag), 0);
}

#[test]
fn condition_variable_releases_and_retakes_the_mutex() {
    let (kernel, mem) = setup();
    let (main_tag, waiter_tag) = (0x10, 0x20);
    let (mutex, key) = (0x100, 0x200);
    mem.store_u32(mutex, waiter_tag);
    let waiter = {
        let (kernel, mem) = (kernel.clone(), mem.clone());
        std::thread::spawn(move || kernel.wait_process_wide_key(&*mem, mutex, key, waiter_tag, -1))
    };
    until_waiting(&kernel, 1);
    assert_eq!(mem.load_u32(mutex), Some(0), "waiting releases the mutex");
    assert_eq!(mem.load_u32(key), Some(1));
    // Signal while holding the mutex: the waiter must queue for it.
    mem.store_u32(mutex, main_tag);
    kernel.signal_process_wide_key(&*mem, key, 1);
    assert_eq!(mem.load_u32(key), Some(0));
    until_waiting(&kernel, 1);
    assert_eq!(mem.load_u32(mutex), Some(main_tag | MUTEX_WAITERS));
    kernel.arbitrate_unlock(&*mem, mutex);
    assert_eq!(waiter.join().unwrap(), 0);
    assert_eq!(mem.load_u32(mutex), Some(waiter_tag));

    // A timed wait gives up without the mutex: the SDK takes it back in
    // user space (taking it here too made the thread wait on itself).
    mem.store_u32(mutex, main_tag);
    assert_eq!(
        kernel.wait_process_wide_key(&*mem, mutex, key, main_tag, 1_000_000),
        RESULT_TIMED_OUT
    );
    assert_eq!(mem.load_u32(mutex), Some(0));
    assert_eq!(mem.load_u32(key), Some(0));
}

#[test]
fn address_waits_follow_their_conditions() {
    let (kernel, mem) = setup();
    mem.store_u32(0x300, 5);
    // WaitIfEqual with a different value does not wait.
    assert_eq!(
        kernel.wait_for_address(&*mem, 0x300, 2, 4, -1),
        RESULT_INVALID_STATE
    );
    assert_eq!(
        kernel.wait_for_address(&*mem, 0x300, 2, 5, 1_000_000),
        RESULT_TIMED_OUT
    );
    let waiter = {
        let (kernel, mem) = (kernel.clone(), mem.clone());
        // DecrementAndWaitIfLessThan(6): 5 < 6, so decrement to 4 and wait.
        std::thread::spawn(move || kernel.wait_for_address(&*mem, 0x300, 1, 6, -1))
    };
    until_waiting(&kernel, 1);
    assert_eq!(mem.load_u32(0x300), Some(4));
    // SignalAndIncrementIfEqual(4) bumps the value and wakes the waiter.
    assert_eq!(kernel.signal_to_address(&*mem, 0x300, 1, 4, 1), 0);
    assert_eq!(waiter.join().unwrap(), 0);
    assert_eq!(mem.load_u32(0x300), Some(5));
    assert_eq!(
        kernel.signal_to_address(&*mem, 0x300, 1, 4, 1),
        RESULT_INVALID_STATE
    );
}

#[test]
fn signals_that_modify_by_waiting_count_consume_the_wake() {
    let (kernel, mem) = setup();
    let wait = |kernel: &Arc<Kernel>, mem: &Arc<Memory>| {
        let (kernel, mem) = (kernel.clone(), mem.clone());
        // WaitIfEqual(1): an nn::os light event with waiters.
        std::thread::spawn(move || kernel.wait_for_address(&*mem, 0x300, 2, 1, -1))
    };
    // One waiter, count 1: woken, and the value drops to 0 (not signalled).
    mem.store_u32(0x300, 1);
    let waiter = wait(&kernel, &mem);
    until_waiting(&kernel, 1);
    assert_eq!(kernel.signal_to_address(&*mem, 0x300, 2, 1, 1), 0);
    assert_eq!(waiter.join().unwrap(), 0);
    assert_eq!(mem.load_u32(0x300), Some(0));
    // Two waiters, count 1: one woken, the other still waits, the value
    // stays.
    mem.store_u32(0x300, 1);
    let first = wait(&kernel, &mem);
    let second = wait(&kernel, &mem);
    until_waiting(&kernel, 2);
    assert_eq!(kernel.signal_to_address(&*mem, 0x300, 2, 1, 1), 0);
    assert_eq!(mem.load_u32(0x300), Some(1));
    until_waiting(&kernel, 1);
    assert_eq!(kernel.signal_to_address(&*mem, 0x300, 2, 1, 1), 0);
    assert_eq!((first.join().unwrap(), second.join().unwrap()), (0, 0));
    assert_eq!(mem.load_u32(0x300), Some(0));
    // No waiters: the value rises (the event is left signalled).
    mem.store_u32(0x300, 1);
    assert_eq!(kernel.signal_to_address(&*mem, 0x300, 2, 1, 1), 0);
    assert_eq!(mem.load_u32(0x300), Some(2));
}

#[test]
fn cancel_synchronization_interrupts_one_wait() {
    let (kernel, _) = setup();
    let thread = kernel.create_thread();
    let (_, read) = kernel.create_event();
    assert_eq!(kernel.cancel_synchronization(thread), 0);
    assert_eq!(
        kernel.wait_synchronization(&[read], -1, thread),
        Err(RESULT_CANCELLED)
    );
    assert_eq!(
        kernel.wait_synchronization(&[read], 0, thread),
        Err(RESULT_TIMED_OUT)
    );
}

/// A stand-in for the runtime's run lock: which host thread may touch
/// guest memory, and whether any thread touched the mutex without it.
mod run_lock {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex};
    use std::thread::ThreadId;

    static HOLDER: Mutex<Option<ThreadId>> = Mutex::new(None);
    static TURN: Condvar = Condvar::new();
    pub static VIOLATIONS: AtomicUsize = AtomicUsize::new(0);

    pub fn acquire() {
        let mut holder = HOLDER.lock().unwrap();
        while holder.is_some() {
            holder = TURN.wait(holder).unwrap();
        }
        *holder = Some(std::thread::current().id());
    }

    pub fn release() {
        *HOLDER.lock().unwrap() = None;
        TURN.notify_all();
    }

    pub fn check() {
        if *HOLDER.lock().unwrap() != Some(std::thread::current().id()) {
            VIOLATIONS.fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// Guest memory whose accesses to the watched word must hold the run lock.
#[derive(Debug)]
struct Checked {
    inner: Arc<Memory>,
    watched: u64,
}

impl MemoryAccess for Checked {
    fn load_u8(&self, addr: u64) -> Option<u8> {
        self.inner.load_u8(addr)
    }
    fn load_u32(&self, addr: u64) -> Option<u32> {
        if addr == self.watched {
            run_lock::check();
        }
        self.inner.load_u32(addr)
    }
    fn load_u64(&self, addr: u64) -> Option<u64> {
        self.inner.load_u64(addr)
    }
    fn store_u32(&self, addr: u64, value: u32) -> bool {
        if addr == self.watched {
            run_lock::check();
        }
        self.inner.store_u32(addr, value)
    }
    fn store_u64(&self, addr: u64, value: u64) -> bool {
        self.inner.store_u64(addr, value)
    }
    fn read_bytes(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        self.inner.read_bytes(addr, len)
    }
    fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
        self.inner.write_bytes(addr, bytes)
    }
}

#[test]
fn a_woken_waiter_retakes_the_mutex_only_once_it_runs() {
    use std::sync::atomic::Ordering;
    let (kernel, mem) = setup();
    kernel.set_blocking_hooks(run_lock::release, run_lock::acquire);
    let (main_tag, waiter_tag) = (0x10, 0x20);
    let (mutex, key) = (0x100, 0x200);
    let checked = Arc::new(Checked {
        inner: mem.clone(),
        watched: mutex,
    });
    mem.store_u32(mutex, waiter_tag);
    let waiter = {
        let (kernel, checked) = (kernel.clone(), checked.clone());
        std::thread::spawn(move || {
            run_lock::acquire();
            let result = kernel.wait_process_wide_key(&*checked, mutex, key, waiter_tag, -1);
            run_lock::release();
            result
        })
    };
    until_waiting(&kernel, 1);
    // Run, signal and keep running: the waiter wakes but must not touch
    // the mutex until it gets to run, by which time this thread owns it.
    run_lock::acquire();
    kernel.signal_process_wide_key(&*checked, key, 1);
    std::thread::sleep(Duration::from_millis(50));
    mem.store_u32(mutex, main_tag);
    run_lock::release();
    until_waiting(&kernel, 1);
    assert_eq!(mem.load_u32(mutex), Some(main_tag | MUTEX_WAITERS));
    run_lock::acquire();
    kernel.arbitrate_unlock(&*checked, mutex);
    run_lock::release();
    assert_eq!(waiter.join().unwrap(), 0);
    assert_eq!(mem.load_u32(mutex), Some(waiter_tag));
    assert_eq!(run_lock::VIOLATIONS.load(Ordering::SeqCst), 0);
}
