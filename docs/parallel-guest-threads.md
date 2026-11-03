# Guest threads on host cores (`recomp-runtime`)

A recompiled title creates guest threads with `svcCreateThread` /
`svcStartThread`. Each guest thread runs on its own host thread. There are two
scheduling modes.

## Modes

| mode | how to select | behavior |
|------|---------------|----------|
| Serialized (default) | nothing | one guest thread runs at a time, coordinated by a run lock |
| Parallel | `RECOMP_PARALLEL=1`, or `parallel = true` under `[runtime]` in `title.toml` (then `RECOMP_PARALLEL=0` selects serialized) | guest threads really run at the same time on host cores |

Serialized mode: a thread holds the run lock while it executes guest code,
gives it up when it blocks in an SVC, and, once its time slice
(`RECOMP_TIME_SLICE_US`, default 100us) is used and another thread is waiting,
at a call or loop boundary. The lock is handed by priority
(`RECOMP_STARVATION_MS` bounds how long a lower priority thread may hold it).
Because only one thread runs, exclusive load/store pairs are atomic by
construction and no data barrier is needed.

Parallel mode: no lock is taken. `acquire`, `release`, the preemption check at a
call boundary and the check in a loop return immediately; `svcSleepThread` and
`yield` use the host scheduler. That removes the serialization cost, but guest
threads now contend for real, so the exclusive accesses and barriers have to
mean something.

### Yielding in parallel mode

A guest yield (`svcSleepThread(0)`) does not hand over the run lock in parallel
mode. It pauses for up to `RECOMP_YIELD_SPIN` `pause` iterations (default 64,
about a microsecond on current x86-64; 0 disables) and then calls the host
yield. The runtime has no wake word for the yielding code - the guest polls its
own condition after the yield returns - so the spin is a fixed pause budget,
not a check of another thread's state.

The point is to replace `sched_yield` syscalls in a poll loop when the wait is
shorter than the budget. It does not make a longer wait finish sooner, and on a
host core shared with the thread being waited for it can delay the handoff by
up to the budget. Use `RECOMP_YIELD_SPIN=0` as the A/B baseline; keep the
default (or disable the spin) if frame time regresses.

## Exclusive accesses

The lifter used to lower `LDXR`/`STXR` to a plain load and a store that always
reported success. With threads running at the same time that would silently lose
lock acquisitions, so the pair now has the hardware's semantics.

| instruction | lowered to |
|-------------|------------|
| `LDXR`, `LDAXR`, `LDXP`, `LDAXP` | `recomp_abi::mem_load_exclusive(addr, size, pair)`: read the word(s) and record a reservation |
| `STXR`, `STLXR`, `STXP`, `STLXP` | `recomp_abi::mem_store_exclusive(addr, size, pair, value, value2)`: compare-and-swap; the status register gets 0 when the store happened, 1 when it did not |
| `CLREX` | `recomp_abi::clear_exclusive()` |
| `LDAR`, `STLR` | a plain access: x86-64's total store order already gives loads acquire and stores release ordering |
| `DMB`, `DSB`, `ISB` | `recomp_abi::memory_fence()`: a `SeqCst` fence |

The reservation belongs to the thread, as the hardware's local exclusion does
(`Cell<Option<Reservation>>` in a thread-local). A store-exclusive takes the
address's lock for the compare and the write together, so no other thread's
exclusive store can slip between them. Guest memory itself stays shared without
synchronization: the fast access path reads published mapping tables and raw
region pointers, like a machine's RAM.

Locks are chosen from the address, 64 shards over `address >> 6`, so unrelated
spinlocks at different addresses do not contend with each other.

### Encoding

Load/store exclusive sit in the group with bits 29:24 = `001000`:

| bits | field | meaning |
|------|-------|---------|
| 31:30 | option, size | bit 30 selects 32-bit or 64-bit |
| 23 | non-exclusive | set for `LDAR` / `STLR`, which order an access instead of reserving it |
| 22 | direction | 1 load, 0 store |
| 21 | pair | `LDXP`, `LDAXP`, `STXP`, `STLXP` |
| 15 | ordering | the acquire/release variants (`LDAXR`, `STLXR`, ...) |

Encodings in the tests were assembled with `llvm-mc`, not hand-derived.

Barriers and `CLREX` share the system class with `CRn = 3`:
`word & 0xFFFF_F000 == 0xD503_3000`, and bits 7:5 say which: 2 is `CLREX`, 5
`DMB`, 6 `DSB`, 7 `ISB`. Anything else in that class (hints, cache maintenance)
is still ignored.

## What the guest threads share

| state | guarding |
|-------|----------|
| guest memory | lock-free shared (published mapping table, raw region pointers) |
| exclusive monitor | thread-local; the compare-and-swap is sharded by address |
| kernel objects, waits, thread table | `Mutex` + `Condvar` in `recomp-services::kernel` |
| `nvdrv` driver state | `Mutex` |
| GPU mappings, channels, syncpoints | `Mutex` in `recomp-services::gpu`, held by the GPU worker while it decodes |
| IPC sessions | the registry `Mutex` is held only to look a session up; a call holds just its object's lock |
| VI buffer queue | `Mutex` |
| `fs:` / RomFS window | `Mutex` |

## Validation

- `recomp-runtime::memory` tests: a store-exclusive writes only when the value
  is untouched; a plain store between the pair makes it fail; `CLREX` makes it
  fail; a pair reserves both words; four host threads race an SDK-shaped
  spinlock and every one of 4000 increments lands.
- `recomp-pipeline` lift test: `LDXR`/`STXR`/`LDXP`/`STXP` become exclusive ops,
  `LDAR` stays a plain load, `CLREX` and `DMB`/`ISB` emit the fence ops.
- `recomp-pipeline` codegen test: the emitted project calls the exclusive and
  fence ABI entries.
- `recomp-runtime::memory` test: kernel word updates (`update_u32_exclusive`)
  and exclusive pairs racing on one word lose no increment.
- Luigi's Mansion 3: parallel mode used to stop around second 60 with the render
  job thread (`t30008`) reading the NVN slot-index global at `0x97d35b0` as
  `-1`. The cause was the kernel, not NVN: `svcSignalToAddress` with
  SignalAndModifyByWaitingCountIfEqual left a light event signalled after waking
  its only waiter (it should drop the word from 1 to 0), so the render thread's
  next wait for the job thread returned at once and the two raced. With that
  fixed, and the kernel's own read-modify-writes of mutex and arbitration words
  made atomic against guest exclusive stores (below), parallel mode runs LM3
  through its boot loading screen (all of `global.data`) and into loading the
  title screen in about four minutes. Serialized mode reaches the same point far
  later: the loader gets a fraction of the one core.

## Kernel updates of guest words

The kernel changes mutex words (handing a mutex on, marking waiters) and
address-arbitration words (decrement-and-wait, signal-and-modify) that guest
threads change with exclusive pairs. A plain load and store there could lose an
owner's user-space release made between them, leaving the mutex owned by a
thread that had already let it go and every later locker waiting for ever.
`MemoryAccess::update_u32` does the read-modify-write under the address's
exclusive lock, the same lock a store-exclusive's compare-and-swap takes, so a
guest store-exclusive racing it sees the change and retries.

## Limits

- Serial mode remains the default; parallel mode is opt-in until more of a
  title's behavior under real concurrency is validated (LM3 runs in it now and
  needs it to load in reasonable time).
- Guest code that races non-atomic shared data without barriers races on the
  host too, exactly as it would on hardware; a title with such a bug in its own
  code may behave differently. Most of LM3's CPU/GPU ordering comes from GPU
  syncpoints; submissions now run on the GPU worker and their waits block
  (`docs/async-gpu-submissions.md`), so the interleaving parallel mode produces
  is much closer to the one hardware produces.
- A reservation survives a kernel wait in serialized mode. Hardware clears the
  monitor at a context switch, so a thread that reserves, blocks, then stores
  would succeed here where it would fail there.
