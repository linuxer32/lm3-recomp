//! Kernel synchronization: events, thread objects, mutex and
//! condition-variable arbitration, and address waits.
//!
//! Every waiter blocks on one condition variable and rechecks its own
//! condition whenever anything changes. Around the first block of an SVC the
//! kernel calls the hooks the runtime installs, so a guest thread gives up
//! the scheduler while it sleeps and takes it back before returning.

use std::collections::BTreeMap;
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use crate::ipc::MemoryAccess;

/// svc::ResultInvalidHandle (module 1, description 114).
pub const RESULT_INVALID_HANDLE: u64 = 0xE401;
/// svc::ResultTimedOut (module 1, description 117).
pub const RESULT_TIMED_OUT: u64 = 0xEA01;
/// svc::ResultCancelled (module 1, description 118).
pub const RESULT_CANCELLED: u64 = 0xEC01;
/// svc::ResultInvalidState (module 1, description 125).
pub const RESULT_INVALID_STATE: u64 = 0xFA01;
/// svc::ResultInvalidSize (module 1, description 101).
pub const RESULT_INVALID_SIZE: u64 = 0xCA01;
/// svc::ResultInvalidCurrentMemory (module 1, description 106).
pub const RESULT_INVALID_MEMORY: u64 = 0xD401;
/// svc::ResultInvalidEnumValue (module 1, description 120).
pub const RESULT_INVALID_ENUM: u64 = 0xF001;
/// Mutex word flag: other threads wait for the owner to unlock.
pub const MUTEX_WAITERS: u32 = 0x4000_0000;
/// Pseudo-handle for the calling thread.
pub const CURRENT_THREAD: u32 = 0xFFFF_8000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Object {
    /// One side of an event; both sides share the signal `id`.
    Event {
        id: u64,
        writable: bool,
    },
    Thread {
        id: u64,
    },
    SharedMemory {
        id: u64,
    },
}

/// Shared memory: its bytes before the first mapping and where the guest
/// has mapped it since.
#[derive(Debug, Default)]
struct SharedMemory {
    contents: Vec<u8>,
    mappings: Vec<u64>,
}

#[derive(Debug, Default, Clone, Copy)]
struct ThreadState {
    exited: bool,
    /// Set by CancelSynchronization; consumed by the next wait.
    cancelled: bool,
}

#[derive(Debug)]
struct Waiter {
    id: u64,
    /// Mutex address, condition-variable key or arbitrated address.
    key: u64,
    tag: u32,
    woken: bool,
}

#[derive(Debug, Default)]
struct State {
    objects: BTreeMap<u32, Object>,
    next_handle: u32,
    next_id: u64,
    events: BTreeMap<u64, bool>,
    threads: BTreeMap<u64, ThreadState>,
    shared: BTreeMap<u64, SharedMemory>,
    mutex_waiters: Vec<Waiter>,
    cv_waiters: Vec<Waiter>,
    address_waiters: Vec<Waiter>,
}

impl State {
    fn new_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn new_handle(&mut self, object: Object) -> u32 {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.objects.insert(handle, object);
        handle
    }

    fn signalled(&self, handle: u32) -> bool {
        match self.objects.get(&handle) {
            Some(Object::Event { id, .. }) => self.events.get(id).copied().unwrap_or(false),
            Some(Object::Thread { id }) => self.threads.get(id).is_some_and(|thread| thread.exited),
            Some(Object::SharedMemory { .. }) | None => false,
        }
    }

    fn thread(&mut self, handle: u32) -> Option<&mut ThreadState> {
        match self.objects.get(&handle) {
            Some(Object::Thread { id }) => self.threads.get_mut(id),
            _ => None,
        }
    }
}

/// Remove waiter `id` from `list` if it was woken; true if it was.
fn take_woken(list: &mut Vec<Waiter>, id: u64) -> bool {
    match list
        .iter()
        .position(|waiter| waiter.id == id && waiter.woken)
    {
        Some(index) => {
            list.remove(index);
            true
        }
        None => false,
    }
}

/// Wake up to `count` (all when `count <= 0`) unwoken waiters on `key`.
fn wake(list: &mut [Waiter], key: u64, count: i32) -> usize {
    let limit = if count <= 0 {
        usize::MAX
    } else {
        count as usize
    };
    let mut woken = 0;
    for waiter in list
        .iter_mut()
        .filter(|waiter| waiter.key == key && !waiter.woken)
    {
        if woken == limit {
            break;
        }
        waiter.woken = true;
        woken += 1;
    }
    woken
}

fn pending(list: &[Waiter], key: u64) -> usize {
    list.iter()
        .filter(|waiter| waiter.key == key && !waiter.woken)
        .count()
}

/// None waits forever; zero polls.
fn deadline(timeout_ns: i64) -> Option<Instant> {
    (timeout_ns >= 0).then(|| Instant::now() + Duration::from_nanos(timeout_ns as u64))
}

struct Hooks {
    release: fn(),
    acquire: fn(),
}

pub struct Kernel {
    state: Mutex<State>,
    changed: Condvar,
    hooks: OnceLock<Hooks>,
}

impl Default for Kernel {
    fn default() -> Self {
        Self::new()
    }
}

/// The process-wide kernel.
pub fn kernel() -> &'static Kernel {
    static KERNEL: OnceLock<Kernel> = OnceLock::new();
    KERNEL.get_or_init(Kernel::new)
}

impl Kernel {
    pub fn new() -> Self {
        Kernel {
            state: Mutex::new(State {
                next_handle: 0x0003_0000,
                ..State::default()
            }),
            changed: Condvar::new(),
            hooks: OnceLock::new(),
        }
    }

    /// `release` runs before a thread first blocks in an SVC and `acquire`
    /// before it returns from that SVC.
    pub fn set_blocking_hooks(&self, release: fn(), acquire: fn()) {
        let _ = self.hooks.set(Hooks { release, acquire });
    }

    /// Run a host operation that may block for long (a service lock another
    /// guest thread holds while it waits) with the scheduler released, as a
    /// blocking SVC does, so the thread holding it can run.
    pub fn blocking<T>(&self, operation: impl FnOnce() -> T) -> T {
        if let Some(hooks) = self.hooks.get() {
            (hooks.release)();
        }
        let value = operation();
        if let Some(hooks) = self.hooks.get() {
            (hooks.acquire)();
        }
        value
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Wait once for a change or the deadline; true when the deadline
    /// passed. The first wait releases the scheduler; `finish` takes it back.
    fn sleep<'a>(
        &'a self,
        state: MutexGuard<'a, State>,
        deadline: Option<Instant>,
        released: &mut bool,
    ) -> (MutexGuard<'a, State>, bool) {
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return (state, true);
        }
        if !*released {
            if let Some(hooks) = self.hooks.get() {
                (hooks.release)();
            }
            *released = true;
        }
        match deadline {
            None => (
                self.changed
                    .wait(state)
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
                false,
            ),
            Some(deadline) => {
                let timeout = deadline.saturating_duration_since(Instant::now());
                let (state, _) = self
                    .changed
                    .wait_timeout(state, timeout)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                (state, false)
            }
        }
    }

    fn finish(&self, state: MutexGuard<'_, State>, released: bool) {
        drop(state);
        if released {
            if let Some(hooks) = self.hooks.get() {
                (hooks.acquire)();
            }
        }
    }

    /// Wake every waiter so it rechecks its own condition. The state lock is
    /// held while notifying, so a waiter cannot slip between checking its
    /// condition and sleeping.
    pub fn notify_waiters(&self) {
        let _state = self.lock();
        self.changed.notify_all();
    }

    /// Block until `ready` reports true, or `timeout_ns` passes (negative
    /// waits forever, zero only polls). The scheduler is released around the
    /// first block and taken back before returning, as a blocking SVC does.
    /// Returns whether `ready` became true.
    ///
    /// `ready` must not take the kernel state lock; the GPU worker's syncpoint
    /// waits use it while the state lock is held.
    pub fn wait_for(&self, timeout_ns: i64, ready: impl Fn() -> bool) -> bool {
        let deadline = deadline(timeout_ns);
        let mut state = self.lock();
        let mut released = false;
        loop {
            if ready() {
                self.finish(state, released);
                return true;
            }
            let (next, timed_out) = self.sleep(state, deadline, &mut released);
            state = next;
            if timed_out {
                self.finish(state, released);
                return false;
            }
        }
    }

    /// Blocked waiters for diagnostics: (kind, key, tag) for each mutex,
    /// condition-variable and address waiter not yet woken. Mutex and
    /// condition-variable tags are the waiting thread's handle.
    pub fn waiter_list(&self) -> Vec<(&'static str, u64, u32)> {
        let state = self.lock();
        [
            ("mutex", &state.mutex_waiters),
            ("condvar", &state.cv_waiters),
            ("address", &state.address_waiters),
        ]
        .iter()
        .flat_map(|(kind, list)| {
            list.iter()
                .filter(|waiter| !waiter.woken)
                .map(|waiter| (*kind, waiter.key, waiter.tag))
        })
        .collect()
    }

    /// Waiters still blocked, for diagnostics.
    pub fn waiting(&self) -> usize {
        let state = self.lock();
        [
            &state.mutex_waiters,
            &state.cv_waiters,
            &state.address_waiters,
        ]
        .iter()
        .map(|list| list.iter().filter(|waiter| !waiter.woken).count())
        .sum()
    }

    pub fn owns(&self, handle: u32) -> bool {
        self.lock().objects.contains_key(&handle)
    }

    pub fn close(&self, handle: u32) -> bool {
        self.lock().objects.remove(&handle).is_some()
    }

    /// CreateEvent: (writable handle, readable handle).
    pub fn create_event(&self) -> (u32, u32) {
        let mut state = self.lock();
        let id = state.new_id();
        state.events.insert(id, false);
        let write = state.new_handle(Object::Event { id, writable: true });
        let read = state.new_handle(Object::Event {
            id,
            writable: false,
        });
        (write, read)
    }

    fn set_event(&self, handle: u32, signalled: bool, need_writable: bool) -> u64 {
        let mut state = self.lock();
        let Some(Object::Event { id, writable }) = state.objects.get(&handle).copied() else {
            return RESULT_INVALID_HANDLE;
        };
        if need_writable && !writable {
            return RESULT_INVALID_HANDLE;
        }
        state.events.insert(id, signalled);
        if signalled {
            self.changed.notify_all();
        }
        0
    }

    /// SignalEvent (writable side).
    pub fn signal_event(&self, handle: u32) -> u64 {
        self.set_event(handle, true, true)
    }

    /// ClearEvent (either side).
    pub fn clear_event(&self, handle: u32) -> u64 {
        self.set_event(handle, false, false)
    }

    /// ResetSignal: clears a signalled event, InvalidState if it was not.
    pub fn reset_signal(&self, handle: u32) -> u64 {
        let mut state = self.lock();
        let Some(Object::Event { id, .. }) = state.objects.get(&handle).copied() else {
            return RESULT_INVALID_HANDLE;
        };
        match state.events.insert(id, false) {
            Some(true) => 0,
            _ => RESULT_INVALID_STATE,
        }
    }

    /// Shared memory a service hands out, initially holding `contents`.
    pub fn create_shared_memory(&self, contents: Vec<u8>) -> u32 {
        let mut state = self.lock();
        let id = state.new_id();
        state.shared.insert(
            id,
            SharedMemory {
                contents,
                mappings: Vec::new(),
            },
        );
        state.new_handle(Object::SharedMemory { id })
    }

    fn shared_id(state: &State, handle: u32) -> Option<u64> {
        match state.objects.get(&handle) {
            Some(Object::SharedMemory { id }) => Some(*id),
            _ => None,
        }
    }

    /// Record a guest mapping of shared memory at `address`; returns the
    /// bytes to fill it with.
    pub fn map_shared_memory(&self, handle: u32, address: u64) -> Option<Vec<u8>> {
        let mut state = self.lock();
        let id = Self::shared_id(&state, handle)?;
        let shared = state.shared.get_mut(&id)?;
        shared.mappings.push(address);
        Some(shared.contents.clone())
    }

    pub fn unmap_shared_memory(&self, handle: u32, address: u64) {
        let mut state = self.lock();
        if let Some(id) = Self::shared_id(&state, handle) {
            if let Some(shared) = state.shared.get_mut(&id) {
                shared.mappings.retain(|mapped| *mapped != address);
            }
        }
    }

    /// Update shared memory: its stored bytes and every guest mapping.
    pub fn write_shared_memory(
        &self,
        mem: &dyn MemoryAccess,
        handle: u32,
        offset: usize,
        bytes: &[u8],
    ) {
        let mut state = self.lock();
        let Some(id) = Self::shared_id(&state, handle) else {
            return;
        };
        let Some(shared) = state.shared.get_mut(&id) else {
            return;
        };
        if let Some(target) = shared.contents.get_mut(offset..offset + bytes.len()) {
            target.copy_from_slice(bytes);
        }
        for mapped in &shared.mappings {
            mem.write_bytes(mapped + offset as u64, bytes);
        }
    }

    /// A new thread object (signalled once the thread exits).
    pub fn create_thread(&self) -> u32 {
        let mut state = self.lock();
        let id = state.new_id();
        state.threads.insert(id, ThreadState::default());
        state.new_handle(Object::Thread { id })
    }

    /// Register a thread under a handle chosen by the caller (the main
    /// thread's handle is fixed at boot).
    pub fn adopt_thread(&self, handle: u32) {
        let mut state = self.lock();
        let id = state.new_id();
        state.threads.insert(id, ThreadState::default());
        state.objects.insert(handle, Object::Thread { id });
    }

    pub fn thread_id(&self, handle: u32) -> Option<u64> {
        match self.lock().objects.get(&handle) {
            Some(Object::Thread { id }) => Some(*id),
            _ => None,
        }
    }

    pub fn exit_thread(&self, handle: u32) {
        let mut state = self.lock();
        if let Some(thread) = state.thread(handle) {
            thread.exited = true;
            self.changed.notify_all();
        }
    }

    /// CancelSynchronization: the thread's current or next wait returns
    /// Cancelled.
    pub fn cancel_synchronization(&self, handle: u32) -> u64 {
        let mut state = self.lock();
        match state.thread(handle) {
            Some(thread) => {
                thread.cancelled = true;
                self.changed.notify_all();
                0
            }
            None => RESULT_INVALID_HANDLE,
        }
    }

    /// WaitSynchronization: the index of a signalled handle. `current` is
    /// the calling thread's handle.
    pub fn wait_synchronization(
        &self,
        handles: &[u32],
        timeout_ns: i64,
        current: u32,
    ) -> Result<usize, u64> {
        let deadline = deadline(timeout_ns);
        let mut state = self.lock();
        if handles
            .iter()
            .any(|handle| !state.objects.contains_key(handle))
        {
            return Err(RESULT_INVALID_HANDLE);
        }
        let mut released = false;
        let result = loop {
            if let Some(thread) = state.thread(current) {
                if std::mem::take(&mut thread.cancelled) {
                    break Err(RESULT_CANCELLED);
                }
            }
            if let Some(index) = handles.iter().position(|handle| state.signalled(*handle)) {
                break Ok(index);
            }
            let (next, timed_out) = self.sleep(state, deadline, &mut released);
            state = next;
            if timed_out {
                break Err(RESULT_TIMED_OUT);
            }
        };
        self.finish(state, released);
        result
    }

    /// Hand the mutex at `addr` to its first waiter, or free it.
    fn unlock_mutex(&self, state: &mut State, mem: &dyn MemoryAccess, addr: u64) {
        let next = state
            .mutex_waiters
            .iter_mut()
            .find(|waiter| waiter.key == addr && !waiter.woken);
        let value = match next {
            Some(waiter) => {
                waiter.woken = true;
                let tag = waiter.tag;
                let more = pending(&state.mutex_waiters, addr) > 0;
                tag | if more { MUTEX_WAITERS } else { 0 }
            }
            None => 0,
        };
        mem.update_u32(addr, &mut |_| value);
        self.changed.notify_all();
    }

    /// Take the mutex at `addr` for `tag`, waiting for the owner if needed.
    /// The caller must be running (hold the scheduler), since the mutex word
    /// is changed by guest code too.
    fn lock_mutex<'a>(
        &'a self,
        mut state: MutexGuard<'a, State>,
        mem: &dyn MemoryAccess,
        addr: u64,
        tag: u32,
        released: &mut bool,
    ) -> MutexGuard<'a, State> {
        // Take it if free, else mark it as having waiters, in one step: the
        // owner may release it in user space at any moment.
        let value = mem
            .update_u32(addr, &mut |value| {
                if value & !MUTEX_WAITERS == 0 {
                    tag | (value & MUTEX_WAITERS)
                } else {
                    value | MUTEX_WAITERS
                }
            })
            .unwrap_or(0);
        if value & !MUTEX_WAITERS == 0 {
            return state;
        }
        let id = state.new_id();
        state.mutex_waiters.push(Waiter {
            id,
            key: addr,
            tag,
            woken: false,
        });
        while !take_woken(&mut state.mutex_waiters, id) {
            state = self.sleep(state, None, released).0;
        }
        state
    }

    /// ArbitrateLock: wait for `owner` to hand over the mutex at `addr`,
    /// unless the word no longer says it has waiters.
    pub fn arbitrate_lock(&self, mem: &dyn MemoryAccess, owner: u32, addr: u64, tag: u32) -> u64 {
        let mut state = self.lock();
        if mem.load_u32(addr) != Some(owner | MUTEX_WAITERS) {
            return 0;
        }
        let id = state.new_id();
        state.mutex_waiters.push(Waiter {
            id,
            key: addr,
            tag,
            woken: false,
        });
        let mut released = false;
        while !take_woken(&mut state.mutex_waiters, id) {
            state = self.sleep(state, None, &mut released).0;
        }
        self.finish(state, released);
        0
    }

    /// ArbitrateUnlock
    pub fn arbitrate_unlock(&self, mem: &dyn MemoryAccess, addr: u64) -> u64 {
        let mut state = self.lock();
        self.unlock_mutex(&mut state, mem, addr);
        0
    }

    /// WaitProcessWideKeyAtomic: release the mutex, wait on `key`, and take
    /// the mutex back once signalled. A timed-out wait returns without the
    /// mutex, as Horizon's does: the SDK then takes it in user space.
    pub fn wait_process_wide_key(
        &self,
        mem: &dyn MemoryAccess,
        mutex: u64,
        key: u64,
        tag: u32,
        timeout_ns: i64,
    ) -> u64 {
        let deadline = deadline(timeout_ns);
        let mut state = self.lock();
        let id = state.new_id();
        state.cv_waiters.push(Waiter {
            id,
            key,
            tag,
            woken: false,
        });
        mem.store_u32(key, 1);
        self.unlock_mutex(&mut state, mem, mutex);
        let mut released = false;
        let mut result = 0;
        while !take_woken(&mut state.cv_waiters, id) {
            let (next, timed_out) = self.sleep(state, deadline, &mut released);
            state = next;
            if timed_out {
                state.cv_waiters.retain(|waiter| waiter.id != id);
                result = RESULT_TIMED_OUT;
                break;
            }
        }
        // The mutex and key words are guest memory that running threads
        // change with exclusive accesses: touch them only once this thread
        // runs again, or another thread could take the mutex at the same
        // time.
        if released {
            drop(state);
            if let Some(hooks) = self.hooks.get() {
                (hooks.acquire)();
            }
            released = false;
            state = self.lock();
        }
        if result == RESULT_TIMED_OUT {
            if pending(&state.cv_waiters, key) == 0 {
                mem.store_u32(key, 0);
            }
            self.finish(state, released);
            return result;
        }
        state = self.lock_mutex(state, mem, mutex, tag, &mut released);
        self.finish(state, released);
        result
    }

    /// SignalProcessWideKey: wake up to `count` waiters (all if <= 0).
    pub fn signal_process_wide_key(&self, mem: &dyn MemoryAccess, key: u64, count: i32) {
        let mut state = self.lock();
        wake(&mut state.cv_waiters, key, count);
        if pending(&state.cv_waiters, key) == 0 {
            mem.store_u32(key, 0);
        }
        self.changed.notify_all();
    }

    /// WaitForAddress: 0 WaitIfLessThan, 1 DecrementAndWaitIfLessThan,
    /// 2 WaitIfEqual.
    pub fn wait_for_address(
        &self,
        mem: &dyn MemoryAccess,
        addr: u64,
        kind: u32,
        value: i32,
        timeout_ns: i64,
    ) -> u64 {
        let deadline = deadline(timeout_ns);
        let mut state = self.lock();
        if kind > 2 {
            return RESULT_INVALID_ENUM;
        }
        // Check (and for kind 1 decrement) in one step.
        let mut wait = false;
        mem.update_u32(addr, &mut |current| {
            let signed = current as i32;
            wait = if kind == 2 {
                signed == value
            } else {
                signed < value
            };
            if wait && kind == 1 {
                signed.wrapping_sub(1) as u32
            } else {
                current
            }
        });
        if !wait {
            return RESULT_INVALID_STATE;
        }
        let id = state.new_id();
        state.address_waiters.push(Waiter {
            id,
            key: addr,
            tag: 0,
            woken: false,
        });
        let mut released = false;
        let mut result = 0;
        while !take_woken(&mut state.address_waiters, id) {
            let (next, timed_out) = self.sleep(state, deadline, &mut released);
            state = next;
            if timed_out {
                state.address_waiters.retain(|waiter| waiter.id != id);
                result = RESULT_TIMED_OUT;
                break;
            }
        }
        self.finish(state, released);
        result
    }

    /// SignalToAddress: 0 Signal, 1 SignalAndIncrementIfEqual,
    /// 2 SignalAndModifyByWaitingCountIfEqual; wakes up to `count` waiters
    /// (all if <= 0).
    pub fn signal_to_address(
        &self,
        mem: &dyn MemoryAccess,
        addr: u64,
        kind: u32,
        value: i32,
        count: i32,
    ) -> u64 {
        let mut state = self.lock();
        if kind != 0 {
            let waiters = pending(&state.address_waiters, addr) as i32;
            let updated = match kind {
                1 => value.wrapping_add(1),
                2 if count <= 0 => {
                    if waiters > 0 {
                        value.wrapping_sub(2)
                    } else {
                        value.wrapping_add(1)
                    }
                }
                // As Horizon: with no waiters the value rises; with waiters
                // it drops by one when this signal wakes them all and stays
                // when some remain. (nn::os light events signal a single
                // waiter this way: 1, "waiting", becomes 0, so the wake
                // consumes the signal. Raising it to 2, "signalled", as this
                // once did, let the next wait return at once, and LM3's
                // render thread then raced its job thread.)
                2 => {
                    if waiters == 0 {
                        value.wrapping_add(1)
                    } else if waiters - 1 < count {
                        value.wrapping_sub(1)
                    } else {
                        value
                    }
                }
                _ => return RESULT_INVALID_ENUM,
            };
            // Compare and update in one step.
            let previous = mem.update_u32(addr, &mut |current| {
                if current as i32 == value {
                    updated as u32
                } else {
                    current
                }
            });
            if previous.map(|current| current as i32) != Some(value) {
                return RESULT_INVALID_STATE;
            }
        }
        wake(&mut state.address_waiters, addr, count);
        self.changed.notify_all();
        0
    }
}
