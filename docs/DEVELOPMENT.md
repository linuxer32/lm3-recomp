# Development

This project now includes an exploratory Rust workspace and uses Nix + devenv for the dev shell.

## Nix + devenv
- Install `nix` and `devenv`.
- This repo uses devenv's flake integration. Enter the shell from the repo root:

```
nix develop --impure
```

If you use direnv, run:
```
direnv allow
```

If `devenv.nix` changes, run `direnv reload` so nix-direnv refreshes the cached shell.

## Back Pressure Hooks (prek + pre-commit)
Back pressure keeps feedback close to the change. This repo uses pre-commit hooks for fast checks, and `prek` is a drop-in replacement that reads the same `.pre-commit-config.yaml`.

Install hooks (config sets `default_install_hook_types` to `pre-commit` + `pre-push`):
```
prek install
```
```
pre-commit install
```

Run hooks on demand:
```
prek run --all-files
```
```
pre-commit run --all-files
```

macOS note: the Nix dev shell ships `prek` only (to avoid Swift/.NET builds); install `pre-commit` separately if you need it.

Configured hooks:
- Pre-commit: `trailing-whitespace`, `end-of-file-fixer`, `check-merge-conflict`, `check-yaml`, `check-toml`, `check-json`, `check-added-large-files`, `detect-private-key`, `check-executables-have-shebangs`, `check-symlinks`, `check-case-conflict`, `cargo fmt --check`.
- Pre-push: `cargo clippy --workspace --all-targets --all-features -D warnings`, `cargo test --workspace`.

## Testing Expectations
- Always run the full test suite (`cargo test`) after changes.
- PRs must include a Testing section that lists the command and outcome.
- Do not mark tests as "Not run" unless you have explicit approval to skip them.
- Include the test outcome in status updates so reviews never default to "Not run".

## Workspace Commands
- Run all tests:

```
cargo test
```

- Run baseline validation and emit reports:

```
cargo run -p recomp-validation -- --out-dir artifacts/validation
```

- Run validation from an artifact index (XCI intake + pipeline + captures):

```
cargo run -p recomp-validation -- artifacts --artifact-index /path/to/validation/artifacts.json
```

- ISA unit tests live in `crates/recomp-isa` and validate arithmetic, shifts, load/store alignment, and flag updates.
- Service dispatch, timing trace, and graphics checksum tests live in their respective crates.

- Run the sample pipeline:

```
cargo run -p recomp-cli -- run \
  --module samples/minimal/module.json \
  --config samples/minimal/title.toml \
  --provenance samples/minimal/provenance.toml \
  --out-dir out/minimal
```

- Run the local automation loop (single-command intake/build/capture/validation):

```
cargo run -p recomp-cli -- automate --config samples/automation.toml
```

`samples/automation.toml` now supports optional `[analysis]` and `[policy]` sections for
research-stage outputs and policy metadata.

- Build the emitted project:

```
cargo build --manifest-path out/minimal/Cargo.toml
```

- Build a generated title for Windows x86_64 (mingw) and stage `SDL3.dll`
  beside the executable:

```
scripts/build-windows-gnu.sh --project-dir out/<title> \
  --sdl3-dir /path/to/SDL3-3.4.18/x86_64-w64-mingw32
```

The script needs `rustup target add x86_64-pc-windows-gnu`, mingw-w64 and
SDL3's mingw devel package; see `docs/windows-build.md`. The release build is
a GUI executable (`--features windows-subsystem`), while debug builds keep a
console.

- Inspect the manifest:

```
cat out/minimal/manifest.json
```

- Package a bundle (code + metadata; assets supplied separately):

```
cargo run -p recomp-cli -- package \
  --project-dir out/minimal \
  --provenance samples/minimal/provenance.toml \
  --out-dir out/bundle-minimal
```

  The bundle's `code/` includes `chunks/` and `.cargo/config.toml` when the
  emitted project has them. Runtime files are never bundled: put a RomFS
  image, `scripts/*.rhai` and `segments/*.bin` beside the executable (or in
  its `assets/` directory), or point `RECOMP_ASSETS_DIR` at them.
  `RECOMP_ROMFS`, `RECOMP_SCRIPT` and `RECOMP_SAVE_DIR` override individual
  paths. See `docs/packaging.md`.

- Run homebrew intake (NRO + optional NSO inputs):

```
cargo run -p recomp-cli -- homebrew-intake \
  --module path/to/homebrew.nro \
  --nso path/to/optional.nso \
  --provenance path/to/provenance.toml \
  --out-dir out/homebrew-intake
```

- Lift homebrew intake output into a lifted module:

```
cargo run -p recomp-cli -- homebrew-lift \
  --module-json out/homebrew-intake/module.json \
  --out-dir out/homebrew-lift
```

The default mode decodes the A64 integer subset (see `docs/exploratory-pipeline.md`); instructions
it cannot lift end their block in a trap that fails at runtime with the guest pc and encoding.
`--entry-address` and `--extra-entry` take hex addresses (`0x` optional). Use `--mode stub` to emit
a placeholder lifted module without decoding instructions.

- Retail NSO sets (`--module <rtld> --nso <main> --nso <subsdk*> --nso <sdk>`) are loaded back to back
  from 0x0800_0000 in that order, like Horizon's loader. Intake them with `--link-at-runtime` and boot
  from rtld's entry (the base of the first module): rtld links the modules itself and rejects
  pre-relocated images (lazy-binding GOT slots must hold their PLT0 offsets), then calls the SDK's
  `nn::init::Start`. Relocations are still resolved during intake to discover code pointers; each
  module's `DT_INIT`/`DT_FINI` are seeded as well, since rtld calls them through the dynamic table.
- When guest code faults, the runtime prints the guest call stack (`fn_<address>` frames, innermost
  first) along with the error. Set `RECOMP_TRACE_CALLS=1` to log every indirect call (virtual calls,
  callbacks) with its target, `x0`-`x3`, returned `x0` and thread, and every SVC with its arguments
  and result (`RECOMP_TRACE_SVC=1` traces only the SVCs). `RECOMP_SVC_STATS=1`
  counts calls per kernel SVC id and prints the top ids (count, share and rate) every
  10 s; it is off by default. `RECOMP_WATCH_CALLS=<addr>,...` logs each
  call to those functions (direct calls too) with `x0`-`x3`, the return address and thread.
  `RECOMP_WATCH_STRINGS=<addr>:x<N>` prints the C string in `xN` whenever the function at
  `addr` is called (file paths, names); `RECOMP_WATCH_MEMORY=<addr>:x<N>+<off>:<len>` hex-dumps the
  bytes around the pointer stored at `xN+off` (a script's program counter), and `&x<N>+<off>` in
  place of `x<N>+<off>` dumps `len` bytes at `xN+off` itself (an object's fields). `RECOMP_TRACE_GPU=1`
  prints every GPU method the command processor decodes, `RECOMP_TRACE_MACRO=1` the macro
  invocations, macro runs and the state methods a macro sends (a direct trace never shows a
  macro-driven draw), and `RECOMP_DUMP_GFX=1` the draws and clears the processor produced with a
  per-submission summary, plus each shader's program header once per program; `RECOMP_DUMP_SHADERS=1`
  adds the first disassembled instructions of each program, and `RECOMP_CAPTURE_SHADERS=<dir>` writes
  each program to `<dir>/<address>-<stage>.bin` (its constant buffer 1 to `.cb1`) for the shader translator's tools
  (`docs/shader-translation.md`). `RECOMP_TRACE_NVDRV=1` traces every
  nvdrv ioctl. In a title built with `graphics = "vulkan"` (`[runtime]` in `title.toml`),
  `RECOMP_VULKAN=1` installs the host Vulkan backend (`crates/recomp-gfx-vulkan`): the guest's draws are rendered with
  translated shaders and its display image is shown in an SDL3 window (`docs/vulkan-host-backend.md`); without it the
  recording backend stays in use. F1 opens a menu over the game (fullscreen, docked mode, frame rate,
  an FPS counter) and F11 toggles fullscreen. `RECOMP_VULKAN_CAPTURE=<dir>` saves every
  `RECOMP_VULKAN_CAPTURE_EVERY`th (default 300) presented frame from frame `RECOMP_VULKAN_CAPTURE_FROM` on (`RECOMP_VULKAN_CAPTURE_WINDOW=1`:
  the window's image, menu included), and `RECOMP_VULKAN_DUMP=<dir>` traces the draws of submission `RECOMP_VULKAN_DUMP_AT` (default 300)
  and then saves every render target, as PPM images read back by the renderer.
  `RECOMP_WATCH_STORE=<addr>[:<len>]` is a data watchpoint: every guest store into the range is
  logged with its block, thread and guest stack. Faults name the guest block they hit (generated code
  records each block's address as it starts) and print the registers as of the last call and the
  object `x0` points to with the start of its vtable.
- Guest services live in `crates/recomp-services`: `hipc.rs` frames HIPC/CMIF messages, `server.rs`
  runs sessions and domains, and `services.rs`, `am.rs`, `fs.rs`, `account.rs` and `system.rs`
  (time, set, psm, mm:u), `hid.rs`, `network.rs`/`online.rs` (an offline console), `audio.rs`,
  `opus.rs`, `nvdrv.rs` and `vi.rs` implement services by `sm:` name. `hwopus` decoders measure each
  packet (bytes consumed and samples from its TOC byte) and return silence; `audout` plays appended
  buffers in real time on a clock thread (2 channels of 16-bit samples at 48 kHz), handing each
  buffer's samples (times the out's volume) to the host's `AudioSink` as it starts, and releases
  them, zero-filling the unused part of a release query's buffer (the SDK returns the first tag
  whatever the count). LM3 mixes its own sound and plays it through `audout`; the Vulkan backend
  registers an SDL audio stream as the sink, and `RECOMP_AUDIO_DUMP=<file>` also writes what plays
  as raw 16-bit stereo at 48 kHz. `hid` samples the host controller every 5 ms into player 1's
  full key and the handheld state LIFOs (`hid::set_controller`, fed by the window thread);
  `RECOMP_HID_PRESS=a@20,plus@30,stick_up@70+2` holds those buttons for 150 ms at those seconds
  (or for the time after `+`; `stick_left`/`up`/`right`/`down` push the left stick all the way, which
  walks a character, and `rstick_left`/`up`/`right`/`down` turn the camera), for testing without a
  controller. `RECOMP_SAVE_DIR=<dir>` keeps save data
  there instead of next to the RomFS image: test runs on a copy leave the player's saves alone
  (a scripted walk can reach an autosave). `RECOMP_TRACE_FS=1` logs RomFS reads by file and
  writes to save data. `RECOMP_TRACE_HOST1X=1` logs each movie frame NVDEC decodes and VIC
  converts (`docs/video-decoding.md`). The user picker applet returns the one user, the controller applet the
  handheld controller. `nvdrv` tracks memory handles, GPU mappings, channels and
  syncpoints; submitted command lists go through `gpu.rs`, which decodes the method stream and performs
  the semaphore releases software polls for. Its default renderer records draws and clears; a host
  backend can take the first channel by registering a renderer factory (`docs/vulkan-host-backend.md`,
  `RECOMP_VULKAN=1`). Unknown ioctls are logged with their decoded number. `vi` provides one 1280x720 display whose
  vsync event fires at 60 Hz and a buffer queue (preallocated GraphicBuffers per slot) paced like the
  console's compositor: a queued frame is shown at a vsync once the one before it has had its swap
  interval, the buffer it replaces is released then, and DequeueBuffer blocks until a buffer is free.
  It also resolves each queued buffer to its GPU addresses (nvmap id and surface offset in the
  GraphicBuffer) and keeps the crop rectangle, which the Vulkan backend uses to present the frame.
  QueueBuffer's reply ends with the 64-bit number of the frame queued (the client keeps it,
  `Surface::getQueuedFrameNumber`), and GetBufferHistory reports the last 8 frames, newest first, as
  Ryujinx does (status, count, packed 28-byte entries: frame number, queue and presentation times in
  nanoseconds, state 2 queued or 3 shown). LM3 polls the history every frame.
  Library applets (`ILibraryAppletCreator::CreateLibraryApplet`) complete as soon as they start,
  with the output a user accepting the defaults would leave (controller support: one player on the
  handheld controller), and `GetLibraryAppletLaunchableEvent` is signalled. LM3's title screen asks
  for controller support and hides every frame (a frame-skip counter it re-arms while the request is
  busy) until it returns: this was the hang after its title setup. Account lists fill the whole
  output array (the user, then invalid zero ids).
  IPC SVCs run outside the SVC dispatcher's locks and a contended service object is waited for with
  the scheduler released, so a service may block (DequeueBuffer does). Every SVC and IPC request is
  logged only with `RECOMP_TRACE_IPC=1`: titles make millions (LM3 polls hid and audio in tight
  loops), and in parallel mode the lines also serialize the threads on stdout. There is one local user, passed to the
  title as the preselected account; time is UTC from the host clock, published through the time
  service's shared memory as the SDK expects. `svcMapMemory` (thread stacks) and
  `svcMapSharedMemory` map regions at run time.
  `fsp-srv` serves the `[runtime.ipc] romfs_image` as the title's data storage (a RomFS image file,
  or a directory such as an emulator's RomFS dump, laid out as a RomFS image in memory with file data
  read on demand; use the dump with the title's update applied, since the update's code expects the
  update's data, e.g. its `*.patch` files) and keeps save data
  in a `save/` directory next to that image (one subdirectory per save type, user and index).
  `RECOMP_ROMFS` selects another image and `RECOMP_ASSETS_DIR` another asset root; a copy beside
  the executable wins over the build-machine path recorded at generation time. Requests
  for unknown services or commands print `UNIMPLEMENTED ...` with the name and command id.
- Guest threads (`crates/recomp-runtime/src/threads.rs`) each run on their own host thread, one at
  a time by default: a thread holds the run lock while it executes guest code and gives it up when it
  blocks in the kernel, or after a 100 us slice at a call boundary or a loop back-edge (checked every
  4096 iterations, so a loop without calls cannot keep the others waiting) when others wait. The lock
  is handed on like Horizon's scheduler: a thread woken with a higher priority than the running one
  takes over at its next call, as does a woken thread of equal priority created for another core
  (on hardware it would start at once there), and a preempted thread resumes first once that thread
  blocks; yields
  and expired slices go to the longest-waiting thread (lower priorities would run on other cores),
  and no thread waits more than 10 ms. `RECOMP_DUMP_WAITERS_AT=<seconds>` lists the threads blocked in the kernel at that time with
  the mutex, condition-variable and address words they wait on, and the GPU syncpoints they are
  waiting for (for deadlocks). Short slices matter: on hardware each core's threads advance together, and games rely on
  short windows staying short (LM3's second render worker reads the swapchain index its first worker
  resets and re-acquires; 2 ms slices let it in between). `RECOMP_TIME_SLICE_US` (0: never) and
  `RECOMP_STARVATION_MS` override the two times. With `RECOMP_PARALLEL=1` no lock is taken and guest
  threads run at the same time on host cores; the exclusive load/store pair then has its real
  semantics (a per-thread reservation, a compare-and-swap guarded by an address-sharded lock, and
  `CLREX` dropping the reservation), and `DMB`/`DSB`/`ISB` become host fences. See
  `docs/parallel-guest-threads.md`. The system counter (CNTPCT_EL0, `svcGetSystemTick`) runs at the
  19.2 MHz CNTFRQ_EL0 reports, from one start every thread shares, and the services that timestamp
  in nanoseconds (buffer queue history, GPU reports) use the same clock (`recomp_services::clock`):
  the SDK converts between the two (`nn::os::ConvertToTick`). The counter used to return nanoseconds
  from each thread's first read, so guest time ran 52 times fast; LM3 spent its loading budget per
  frame in a fiftieth of the time, animated as fast as its time step allowed, and took 64 s to reach
  its title screen (now 14 s, with the native zlib in `[host_functions]`). On x86-64 the clock reads
  the invariant TSC (calibrated once against `Instant`) instead of `clock_gettime`/vDSO: titles
  poll `NvOsGetTimeUS` often enough that the syscall showed on their busiest thread.
  `[host_functions]` can also replace `NvOsGetTimeUS` itself (`nv_os_get_time_us`), which skips the
  SDK's system-counter conversion; LM3 maps `fn_000000000a7fab00` to it.
  LM3 needs parallel mode to load in reasonable time: serialized,
  its loader gets a fraction of one core, so its `title.toml` sets `parallel = true` under
  `[runtime]`, which makes the generated `main` default to it (`RECOMP_PARALLEL=0` opts out). GPU command lists run on their own worker thread, so a thread
  waiting on a syncpoint blocks until the worker reaches it and gives up the scheduler while it
  waits; `RECOMP_GPU_THREAD=0` decodes them on the submitting thread instead
  (`docs/async-gpu-submissions.md`). Events, thread objects, mutex and
  condition-variable arbitration and address waits live in `crates/recomp-services/src/kernel.rs`.
  A new guest thread starts with its creator's FPCR, as Horizon clones it (`KThread::Initialize`),
  and FPCR.FZ sets the host thread's MXCSR flush-to-zero and denormals-are-zero bits, so host float
  operations (the SIMD evaluator's included) flush as the hardware does; LM3's main thread sets FZ
  9 s in, so threads it creates later inherit it. The first FPCR value each thread uses is printed.
  Each thread gets a 0x200-byte slot of the `tls` region (0x40000 bytes from the layout script, so
  512 threads). When guest code stops making calls for 5 s the runtime says so and counts the threads
  blocked in the kernel. Generate the matching
  `[runtime.memory_layout]` for `title.toml` with `scripts/nso-memory-layout.py out/module.json`.
- `RECOMP_PROFILE=<file>` samples where the program spends its time: every millisecond of CPU
  time, SIGPROF records the running thread's host program counter (`RECOMP_PROFILE_AT=<seconds>`
  starts late, past loading). `scripts/profile-report.py <executable> <file> [top]` names the
  hottest functions overall and per thread; generated guest functions keep their `fn_<address>`
  names, code in libraries by the nearest exported symbol (`[libc.so.6] sched_yield`; internal
  functions such as memcpy's variants show as the export before them). Linux and Android sample;
  on Windows the sampler is a no-op: `RECOMP_PROFILE` is reported as unsupported at startup and
  the title runs normally.
  `RECOMP_SIMD_STATS=1`
  prints, every 10 s, which floating-point and SIMD operations reached the evaluator out of line,
  with their decoded fields (`exec` by instruction word): what to inline next.
  On Android, pull the profile and run the report on the local copy of `librecomp_game.so` (the
  profile's map line names the device's path; the report uses the executable given for a library of
  the same name). `RECOMP_TRACE_CHECKED=1` prints, every 10 s, how many guest loads and stores took
  the checked runtime path (not the direct fastmem access), by 256 MiB guest address range, with a
  sample of the guest blocks making them.
- Vulkan debugging on a device without validation layers: `RECOMP_VULKAN_FLUSH_DRAWS=<n>` submits
  and waits every n draws (clears, 2D copies and dispatches alone too), from presented frame
  `RECOMP_VULKAN_FLUSH_FROM` on, timing each submission and naming slow ones;
  `RECOMP_VULKAN_LOST_LOG=<file>` then writes the operations kept (thousands, each with its
  submission number and CLOCK_MONOTONIC start and end, the clock of the kernel log) when the device
  is lost, and every 10000 submissions to `<file>.<n>`, to match a kernel `kgsl` GPU fault to an
  operation. `RECOMP_VULKAN_SHADER_DUMP=<dir>` writes failed pipelines' SPIR-V
  (`RECOMP_VULKAN_SHADER_DUMP_ALL=1`: every pipeline's), `RECOMP_VULKAN_SKIP_FS=<address>,...`
  skips draws by fragment program (0: depth-only draws), `RECOMP_VULKAN_SKIP_KIND=<kind>,...`
  skips draws by kind (`all`, `triangles`, `strip`, `fan`, `quads`, `points`, `lines`,
  `restart`, `indexed`, `instanced`, `depth`, `nodepth`, `depthonly`, `single`, `mrt`, `mrt2`,
  `mrt3`, `mrt4`, `gap`, `integer`, `nofs`; `clear` and `blit` skip clears and 2D copies), and
  `RECOMP_PRESENT_MODE=fifo|mailbox|immediate` asks for a present mode.
- `RECOMP_DOCKED=1` (or a script's `setting("docked", 1)`) reports the console docked: `am`'s
  operation and performance modes, `apm`'s performance mode and a 1920x1080 default display
  resolution, so titles render for a TV (LM3 then presents 1920x1080 frames at the same frame
  rate). The menu's docked switch (`set_docked` in a script) changes the mode while the title runs
  and sends `am`'s mode-change messages. `RECOMP_FPS=60` (or 120, or the menu) presents at swap
  interval 1 whatever the title asks for; a script's `on_frame_rate` patches the title's own frame
  limiter to match. `RECOMP_GPU_TIME=real` writes the GPU timer at its real rate into timestamped semaphore
  releases and reports; by default it runs 256 times slower, like yuzu's and Ryujinx's "fast GPU
  time", so titles that time their frames on the GPU keep their full resolution. Either way it
  counts GPU ticks (384 per 625 ns), not nanoseconds.

- Large lifted programs are emitted as a workspace of chunk crates of at most 4 MiB of source
  each (`CHUNK_BYTES` in `crates/recomp-pipeline/src/output.rs`). Chunk boundaries are
  content-defined (a chunk of at least 2 MiB ends before a function whose name hashes to a cut
  point), so a lifter change rebuilds only the chunks whose functions changed. Emitted projects build their dev
  profile at `-O2` with `panic = "abort"` and without overflow checks or debug assertions (lifted
  code wraps its arithmetic itself): at `-O0` a retail title's machine code passed the 2 GiB that
  x86-64 PC-relative relocations reach, and the link failed. `-O2` threads more of the jumps between
  a function's blocks than `-O1` for about 20% more compile time; chunks over 8 MiB (a single huge
  function, the critical path of a full build) stay at `-O1`. A chunk takes roughly 0.75 GB and about
  16 s to compile, so `cargo build -jN` needs about `0.75 * N` GB.
  The `run` step itself parses the whole lifted module.json and can need ~8x its size in RAM.
  Chunk crates depend only on `recomp-abi` (and through it `recomp-simd`), so a change to the
  runtime, the services or the Vulkan backend relinks the program in seconds; a change to those two
  crates or to the lifter's output rebuilds every chunk (LM3: ~23 min at `-j8`, 9.7 GB peak).

- Guest calls: the guest register files are thread-locals in `recomp-abi` (`cpu_regs`,
  `cpu_store`), so the spills generated code makes around every call are inline thread-local
  accesses rather than calls into the runtime. `recomp_abi::dispatch` looks the target up in the
  hashed table the generated `main` registers (`register_targets`) and calls it directly when the
  runtime has published it: in parallel mode with no call hooks (`RECOMP_WATCH_*`,
  `RECOMP_TRACE_CALLS`), where the runtime has nothing to do at a call. Each thread remembers the
  last target it resolved, keyed by the published table, so repeated virtual calls (a job loop
  pulling the same handler, a vtable method called per object) skip the table probe; republishing
  the table for a hook reload invalidates every thread's entry. Otherwise, and for addresses
  the table lacks, calls enter the runtime's `dispatch`. A call to an address the program has a
  function for is emitted as `recomp_abi::call(address, || unsafe { fn_<address>() })`: a direct
  call while the runtime allows them (`allow_direct_calls`: the table is published and no script
  hook is set, since a hook's trampoline is only in the table), `dispatch` otherwise. Chunk
  functions are `#[no_mangle]`, and a chunk declares the ones it calls in other chunks in an
  `extern "Rust"` block that the linker resolves (chunks call each other in cycles, so they
  cannot depend on each other). The table is several MB for LM3; its lookups were a twentieth of
  the busiest threads' time. The direct-call fast path and last-target hits do not touch the
  runtime's progress counter: table resolutions report it in batches and generated loops report
  it through `backedge`, which keeps the parallel-mode hang monitor working without a
  thread-local read-modify-write on every call (serialized mode never takes these fast paths).
  A prototype guest-context ABI (`[runtime] guest_context = true`) changes generated functions
  to `unsafe fn(ctx: *mut recomp_abi::GuestContext) -> GuestResult` and has them read and write
  the shared register fields directly, so the prologue no longer copies a whole snapshot.
  Direct calls still use the clobber masks: the fast path stores and reloads only the callee's
  `required | clobber` fields, while the full-spill fallback for hooks and serialized calls
  writes the whole live set. It is off by default; see
  [performance-guest-context-prototype.md](performance-guest-context-prototype.md).

- Guest memory is one host reservation (`crates/recomp-runtime/src/fastmem.rs`): the runtime
  reserves the 39-bit guest address space without committing memory, maps each region at
  reservation + guest address, and publishes the base with `recomp_abi::set_memory_base`.
  Generated loads and stores are then one host access instead of a call into the runtime: each
  generated function takes `let __mem = recomp_abi::memory();` (the base and the end of the
  directly accessed range) as it starts, so an access is a compare and the host load or store;
  reading the shared base at every access had cost about ten instructions each. An access
  to an unmapped guest address hits an inaccessible page; the SIGSEGV handler prints the same
  `memory access error` line (guest address, block, thread) and exits, and hands other faults to
  the handler installed before it. The host page size is detected at runtime (4 KiB Horizon
  pages may share a 16 KiB Android host page); commits and protections are host-page aligned.
  `RECOMP_FASTMEM=0` keeps every access on the checked runtime path, as does `RECOMP_WATCH_STORE`.
  A host whose user address space cannot hold all 512 GiB (39-bit Android) reserves only the
  window the title's memory layout reaches (rounded up to a GiB; 8 GiB for LM3): generated code
  keeps the direct path below the window's end (published as `recomp_abi::memory_limit`) and the
  checked path at or above it. `RECOMP_FASTMEM_GIB=<n>` forces a window of that size, which
  reproduces the bounded path on a host that could reserve the whole space. On Windows the
  space is one `VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)` allocation and regions are
  `MEM_COMMIT`ed at the same 1:1 guest addresses. A first-chance vectored exception handler
  commits and protects a page that faults inside the reservation and resumes a read or write;
  a guest access with no region therefore reads zeroes there instead of stopping the process.
  Execute faults and faults outside the reservation are left to the next handler. See
  [android-memory.md](android-memory.md) for the Android behavior, thread-stack defaults, and
  remaining limits.
- Writes to guest memory in the reservation can be watched without reading it
  (`crates/recomp-runtime/src/write_watch.rs`, `MemoryAccess::written_since`): userfaultfd's
  asynchronous write protection (Linux 6.7; a user-mode-only descriptor, which needs no privilege)
  lets a write to a watched page through and marks the page, with no signal, so kernel writes into
  guest memory work as before; `PAGEMAP_SCAN` reports a range's marked pages and protects them
  again in one call. A page mapped again is not watched, and the scan then counts the whole range
  as written. The GPU worker's vertex cache uses it; `RECOMP_WRITE_WATCH=0` turns it off, and the
  runtime says once when the kernel does not offer it. On Windows `written_since` returns `None`,
  so the worker hashes the ranges it copied; startup does not block on a watcher.

- Check the lifter's integer lowering against a capstone reference model (needs `pip install capstone`):

```
cargo build --release -p recomp-cli
scripts/lift-difftest.py target/release/recomp out/lift-difftest 1000 1
```

## Scripts (changes without a compile)

`RECOMP_SCRIPT=<file.rhai>` loads a [Rhai](https://rhai.rs) script into the runtime
(`crates/recomp-runtime/src/script.rs`). A title's `[runtime] script = "scripts/game.rhai"`
(relative to its config file) is the default the generated `main` sets, so the game has its
script however it is started; an empty `RECOMP_SCRIPT` loads none. (Started without it, LM3's
menu presented 60 frames a second while the game's own limiter, which the script patches, kept
it at 30.) Saving the file reloads it while the game runs, so
settings, memory patches, cheats and replacements for single guest functions need neither a
regenerated project nor a rebuild. The script's top level registers things; the functions it
defines are called later:

```
setting("pad_layout", "positional");   // sets RECOMP_PAD_LAYOUT; settings apply at start only
hook(0x8022b50, |cpu| { log(`called from ${cpu.lr.to_hex()}`); false });  // before the function
hook_after(0x87ad1e0, |cpu| { store("jobs", load("jobs") + 1); });       // after it returns
fn on_start() { log("ready"); }             // at start and after a reload
fn on_frame(frame) { if frame % 300 == 0 { log(`jobs ${load("jobs")}`); } }
```

`on_start` runs before the game's own startup code, when its data is not yet initialized or
relocated (LM3's vtables still read 0), so patch data from `on_frame`, for example once at a
frame number or each frame for a cheat.

- `hook(address, |cpu| ...)` runs before the function at `address`; returning `true` skips the
  function (set `x0` first for its result). `hook_after` runs after it returns. Hooks run on the
  calling guest thread. `cpu.x(n)`/`cpu.set_x(n, v)` read and write x0-x30, `cpu.s(n)`/
  `cpu.set_s(n, v)` the single-precision registers, `cpu.lr` and `cpu.sp` the link register and
  stack pointer. Up to 64 functions can be hooked per session; the address must be the start of a
  recompiled function.
- `read_u8/u16/u32/u64/f32(address)`, `write_u8/u16/u32/u64/f32(address, value)` and
  `read_string(address)` go through the runtime's checked memory path: a wrong address is a script
  error, not a crash. `patch_u32`/`patch_f32(address, value)` also write read-only data and code
  pages (a constant in rodata, as an IPS/pchtxt mod changes it).
- `set_docked(bool)` and `set_frame_rate(fps)` change the console mode and the frame rate as the
  F1 menu does.
- `store(key, value)`/`load(key)` keep values across calls and reloads (`load` of a missing key is
  0); `log(...)` and `print(...)` write `[script]` lines to stderr.
- `setting(name, value)` sets the environment variable `RECOMP_<NAME>` while the script loads at
  start, before the runtime, services and window read their settings.
- `on_frame(frame)` runs on the script's own thread each time a frame is shown (frames are skipped
  while it is still busy), so a slow script never delays presentation.
- `on_frame_rate(fps)` runs at start and whenever the frame rate changes (30, 60 or 120), to patch
  the title's own frame timing:

```
fn on_frame_rate(fps) {   // LM3 1.4.0's frame limiter budget, as the FPS mod sets it
    patch_u32(0x93CD4DC, if fps >= 240 { 0x40855555 } else if fps >= 180 { 0x40B1C71C } else if fps >= 120 { 0x41055555 } else if fps >= 60 { 0x41855555 } else { 0x42055555 });
}
```

Hooked functions go through trampolines in the call table the runtime publishes
(`recomp_abi::publish_targets`), and nothing the generated chunks link against changes. While a
hook is set, the direct calls generated code makes go through the table too, since only the table
has the trampoline: in LM3's gameplay that costs a few frames a second, so measure without hooks. A script error leaves the previous script in place (at a
reload) or the original function running (in a hook), and is printed once.
