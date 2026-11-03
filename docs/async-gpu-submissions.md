# GPU submissions on their own thread (`recomp-services::gpu`)

A title submits command lists with `SubmitGPFIFO` on `/dev/nvhost-gpu` and then
waits for the fence with `EventWait` on `/dev/nvhost-ctrl`. These are the
guest's device names, not host paths: the host backend never opens them. On
hardware the
submitting thread is not the thing that runs the lists: the GPU engine consumes
the GPFIFO on its own and the syncpoint only advances when a list is done. The
port now mirrors that.

## What changed

Before: `nvdrv::ioctl` decoded the command lists on the submitting thread and
raised the syncpoint to whatever the waiter asked for, so every wait returned
immediately. Most of a title's CPU/GPU ordering comes from those waits, so
titles ran in an order hardware never produces.

Now:

| piece | behavior |
|-------|----------|
| `SubmitGPFIFO` | allocates the next syncpoint value, queues the entries, returns the fence. Returns before the lists are decoded. |
| GPU worker thread (`recomp-gpu`) | drains the queue in submission order and decodes each submission with that channel's `CommandProcessor` |
| `Syncpoints` | two counters per syncpoint: `issued` (values promised to submissions) and `reached` (values the worker has completed) |
| `EventWait` / `EventWaitAsync` | block until `reached >= threshold`, or until the timeout passes; the value read at the end is written back |
| `AllocGPFIFOEx2` | reports the `issued` value, the value the next submission will reach |

A timeout of `-1` waits forever, `0` only polls. A timed-out wait still returns
`NV_SUCCESS` with the value it found: the caller decides whether to wait again,
as on hardware.

## Fences

`allocate_fence(syncpoint)` increments the per-syncpoint `issued` counter, so
two submissions on one channel get increasing values and a waiter can ask for
the value a specific submission promises. The submission carries `(syncpoint,
value)`; when the worker finishes decoding it, `complete` writes `value` into
`reached` and wakes the kernel's waiters (`Kernel::notify_waiters`).

The fence's syncpoint is the one the caller passed in the ioctl struct, or the
channel's own syncpoint when it passed none.

## Who owns what

| state | owner |
|-------|-------|
| `CommandProcessor` per channel | the worker thread, behind a `Mutex` (one submission at a time) |
| GPU address space mappings (`GPU VA -> CPU address`) | `gpu::mappings()`, shared by the driver and the worker |
| syncpoint counters | `gpu::Syncpoints`, `Mutex` + the kernel's change condvar |
| guest memory for command lists | `gpu::attach_memory`, an `Arc<dyn MemoryAccess>` the runtime installs at startup |

The worker clones the mapping table once per batch rather than holding the
driver's lock while decoding, so mapping buffers never waits on a submission.

## Lock order

A syncpoint wait holds the kernel state lock while it rechecks its condition
(`Kernel::wait_for`), so a wake has to take that lock too: `notify_waiters`
locks the kernel state before notifying, which means a waiter cannot slip
between checking and sleeping.

To keep that from cycling, the worker releases every GPU lock before completing
a fence: the channel lock is held only while decoding one submission, and the
syncpoint lock is released before `notify_waiters`. Nothing takes a GPU lock
while holding the kernel state lock except the wait's own check.

## IPC while a call blocks

`Server` keeps its session registry in a `Mutex` of its own and no longer holds
a hub-wide lock during a service call: it picks the session object, drops the
registry lock, and then holds only that object's lock. A thread blocked in
nvdrv's `EventWait` therefore leaves every other session usable - on hardware
each IPC request is handled concurrently, and a title that waits on a syncpoint
another thread will signal needs the other thread to be able to submit.

The wait also releases the run lock through the kernel's blocking hooks, so in
serialized mode the other guest threads advance while one waits for the GPU.

## Modes

| variable | behavior |
|----------|----------|
| nothing | submissions run on the worker thread (hardware-like) |
| `RECOMP_GPU_THREAD=0` | submissions are decoded on the thread that submitted them, and the fence completes at once (the older behavior) |

`RECOMP_TRACE_GPU=1` prints each submission the worker takes and each syncpoint
wait, with whether it was reached or timed out. `RECOMP_DUMP_WAITERS_AT` lists
GPU syncpoint waits along with the kernel's mutex, condition-variable and
address waiters.

## Validation

- `gpu::a_worker_submission_completes_its_fence`: a bound class plus a short
  semaphore release; the submission returns before the payload is written, the
  wait blocks until the fence is reached, and the payload has arrived by then.
- `gpu::a_syncpoint_poll_returns_the_value_it_found`: a zero-timeout wait on a
  syncpoint nothing promised returns 0, not the threshold.
- `cmif::a_blocked_service_call_does_not_stop_other_sessions`: one session
  blocks inside its call while another session is served and unblocks it.
- `cmif::nvdrv_tracks_memory_handles_and_gpu_mappings`: a `ctrl` `EventWait`
  with a zero timeout reports 0 rather than the threshold it was asked for.

## Limits

- Semaphore releases that decrement a syncpoint (`SemaphoreControl` with the
  decrement flag) still only write the payload; they do not advance a syncpoint.
  NVN's fences go through `SubmitGPFIFO`, so this is not yet needed.
- One worker runs every channel, in submission order across channels. Hardware
  interleaves channels; per-channel order is preserved, cross-channel order is
  the order the guest submitted them.
- A submission whose channel is unknown, or whose command list is unmapped,
  still completes its fence: a guest must not wait for a GPU that was never
  given the work.
- Presenting is still not done: the VI queue counts frames, and the renderer
  backend counts draws.
