//! The GPU channel of a running port: the GPFIFO interface the guest submits
//! command lists through, backed by [`recomp_gfx::maxwell::CommandProcessor`].
//!
//! A GPFIFO entry names a command list: GPU address in bits 39:0, length in
//! words in bits 62:42. The processor decodes the list, keeps the Maxwell 3D
//! register file, runs the guest's macros, and reports what software waits for
//! by writing semaphore payloads into guest memory.
//!
//! Submissions run on a worker thread, as they do on hardware: the thread that
//! submits gets its fence back at once and then waits on the syncpoint, which
//! only advances when the worker has decoded the lists. `RECOMP_GPU_THREAD=0`
//! decodes them on the submitting thread instead, which is the older behavior.
//!
//! Rendering is the host backend's job. Until one exists, the renderer counts
//! calls and, with `RECOMP_DUMP_GFX=1`, prints the draws and clears the
//! processor produced, along with each shader's program header. With
//! `RECOMP_DUMP_SHADERS=1` the first instructions of each program are printed
//! too, disassembled. `RECOMP_TRACE_GPU=1` prints every method write.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use recomp_gfx::maxwell::compute::ComputeDispatch;
use recomp_gfx::maxwell::render::{
    ClearColorCall, ClearDepthStencilCall, DrawCall, GpuMemory, Renderer,
};
use recomp_gfx::maxwell::sph;
use recomp_gfx::maxwell::{CommandProcessor, ProcessorStats};
use recomp_gfx::GraphicsError;

use crate::ipc::MemoryAccess;

/// Bytes of a shader program to disassemble when dumping shaders.
const PROGRAM_DUMP_BYTES: u64 = 0x600;

/// Instructions printed per shader program when dumping shaders.
const INSTRUCTIONS_DUMPED: usize = 24;

fn tracing() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("RECOMP_TRACE_GPU").is_some())
}

/// Print the draws and clears the processor turns state writes into.
fn dumping() -> bool {
    static DUMP: OnceLock<bool> = OnceLock::new();
    *DUMP.get_or_init(|| std::env::var_os("RECOMP_DUMP_GFX").is_some())
}

/// `RECOMP_CAPTURE_SHADERS=<dir>`: write each shader program the title draws
/// with to `<dir>/<address>-<stage>.bin` (its header and code, for developing
/// the shader translator offline; the files are the title's own data, so
/// keep them out of the repository), and its constant buffer 1 (the
/// program's own constants: `BRX` jump tables) to `<address>-<stage>.cb1`.
fn capture_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RECOMP_CAPTURE_SHADERS").map(std::path::PathBuf::from)
}

/// Bytes captured per program at most: programs end well before this, and
/// the translator follows control flow, so trailing bytes are harmless.
const PROGRAM_CAPTURE_BYTES: u64 = 0x10000;

/// Print each shader program's header, and its first instructions.
fn dumping_shaders() -> bool {
    static DUMP: OnceLock<bool> = OnceLock::new();
    *DUMP.get_or_init(|| std::env::var_os("RECOMP_DUMP_SHADERS").is_some())
}

/// Factory function a host backend registers to build its renderer.
type RendererFactory = fn() -> Option<Box<dyn Renderer + Send>>;

/// A host backend registers a renderer factory here, so `recomp-services`
/// itself stays free of display dependencies. The factory is asked once per
/// channel; returning `None` falls back to the recording backend.
static RENDERER_FACTORY: OnceLock<RendererFactory> = OnceLock::new();

pub fn register_renderer(factory: RendererFactory) {
    let _ = RENDERER_FACTORY.set(factory);
}

/// The recording backend, with the dump flags read from the environment.
pub fn console_renderer() -> Box<dyn Renderer + Send> {
    Box::new(EarlyBackend {
        dump: dumping(),
        dump_shaders: dumping_shaders(),
        capture: capture_dir(),
        ..EarlyBackend::default()
    })
}

/// The renderer a new channel uses: the registered host backend when one is
/// installed, otherwise the recording backend.
fn channel_renderer() -> Box<dyn Renderer + Send> {
    RENDERER_FACTORY
        .get()
        .and_then(|factory| factory())
        .unwrap_or_else(console_renderer)
}

/// Print macro invocations, macro runs, and the state methods a macro sends.
/// `RECOMP_TRACE_GPU` only shows direct method writes, so this is what makes a
/// macro-driven draw visible.
fn macro_tracing() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("RECOMP_TRACE_MACRO").is_some())
}

/// The GPU timer as long semaphore releases and timestamp reports write
/// it: ticks of 625/384 ns (NVN converts with that ratio). Titles with
/// dynamic resolution time their frames with these reports; host GPU work
/// here starts only after the guest's command lists are decoded, so the
/// real elapsed time reads as a slow GPU and LM3 lowers its resolution
/// every frame. By default the timer runs 256 times slower, as yuzu's and
/// Ryujinx's "fast GPU time" does, so every frame measures as fast and the
/// title keeps its full resolution. `RECOMP_GPU_TIME=real` keeps the real
/// rate.
fn gpu_time() -> u64 {
    static REAL: OnceLock<bool> = OnceLock::new();
    let real =
        *REAL.get_or_init(|| std::env::var("RECOMP_GPU_TIME").is_ok_and(|mode| mode == "real"));
    let nanoseconds = crate::clock::now_ns();
    gpu_ticks(if real { nanoseconds } else { nanoseconds / 256 })
}

/// Nanoseconds as GPU timer ticks (384/625 per nanosecond), without
/// overflowing.
fn gpu_ticks(nanoseconds: u64) -> u64 {
    (u128::from(nanoseconds) * 384 / 625) as u64
}

/// GPU address to CPU address through the address space's mappings
/// (GPU address -> (CPU address, size)).
pub fn translate(mappings: &BTreeMap<u64, (u64, u64)>, va: u64) -> Option<u64> {
    let (start, (cpu, size)) = mappings.range(..=va).next_back()?;
    (va < start + size).then(|| cpu + (va - start))
}

/// Guest memory as the command processor sees it: GPU addresses translated
/// through the channel's mappings.
struct GpuAdapter<'a> {
    mem: &'a dyn MemoryAccess,
    mappings: &'a BTreeMap<u64, (u64, u64)>,
}

impl GpuMemory for GpuAdapter<'_> {
    fn read_gpu(&self, address: u64, len: usize) -> Option<Vec<u8>> {
        self.mem.read_bytes(translate(self.mappings, address)?, len)
    }

    fn resolve(&self, address: u64) -> Option<u64> {
        translate(self.mappings, address)
    }

    fn read_gpu_with(&self, address: u64, len: usize, read: &mut dyn FnMut(&[u8])) -> bool {
        translate(self.mappings, address).is_some_and(|cpu| self.mem.read_with(cpu, len, read))
    }

    fn written_since(
        &self,
        address: u64,
        len: usize,
        since: Option<u64>,
        written: &mut dyn FnMut(usize, usize),
    ) -> Option<u64> {
        self.mem
            .written_since(translate(self.mappings, address)?, len, since, written)
    }

    fn write_gpu(&mut self, address: u64, bytes: &[u8]) -> bool {
        let Some(cpu) = translate(self.mappings, address) else {
            // A semaphore, report or copy the title waits for would never
            // arrive: say so once per 64 KiB.
            static REPORTED: std::sync::Mutex<std::collections::BTreeSet<u64>> =
                std::sync::Mutex::new(std::collections::BTreeSet::new());
            if REPORTED
                .lock()
                .is_ok_and(|mut seen| seen.insert(address >> 16))
            {
                println!("[recomp-gpu] write to unmapped GPU address {address:#x}");
            }
            return false;
        };
        if !self.mem.write_bytes(cpu, bytes) {
            println!("[recomp-gpu] write to GPU address {address:#x} (CPU {cpu:#x}) failed");
            return false;
        }
        true
    }

    fn gpu_time(&self) -> u64 {
        gpu_time()
    }

    fn increment_syncpoint(&mut self, syncpoint: u32) {
        let shared = shared();
        {
            let mut syncpoints = locked(&shared.syncpoints);
            let value = syncpoints.reached.get(&syncpoint).copied().unwrap_or(0) + 1;
            syncpoints.reached.insert(syncpoint, value);
        }
        crate::kernel::kernel().notify_waiters();
    }
}

/// The backend until a real renderer exists: it counts what was asked for and
/// prints the calls when dumping. Recording every draw would grow without
/// bound, so nothing is kept, except the shader program headers already shown.
#[derive(Debug, Default)]
struct EarlyBackend {
    dump: bool,
    dump_shaders: bool,
    draws: u64,
    indexed_draws: u64,
    color_clears: u64,
    depth_clears: u64,
    submissions: u64,
    /// Program addresses whose header has been printed, so a title that draws
    /// with the same shaders every frame does not repeat the description.
    seen_shaders: BTreeMap<u64, String>,
    /// Where to write each program once (`RECOMP_CAPTURE_SHADERS`).
    capture: Option<std::path::PathBuf>,
    captured: std::collections::BTreeSet<u64>,
}

impl Renderer for EarlyBackend {
    fn draw(&mut self, memory: &dyn GpuMemory, draw: &DrawCall) -> Result<(), GraphicsError> {
        self.draws += 1;
        if draw.parameters.indexed {
            self.indexed_draws += 1;
        }
        if self.dump {
            let parameters = &draw.parameters;
            let index_buffer = draw
                .index_buffer
                .map(|buffer| format!("{:#x}", buffer.address))
                .unwrap_or_else(|| "-".to_string());
            println!(
                "[recomp-gfx] draw {:?} count={} first_vertex={} first_index={} indexed={} ib={index_buffer} instances={} targets={} depth={:#x} viewport={:?} shader_v={:?} shader_f={:?}",
                parameters.topology,
                parameters.count,
                parameters.first_vertex,
                parameters.first_index,
                parameters.indexed,
                parameters.instance_index,
                draw.color_target_count,
                draw.depth_target.address,
                draw.viewport,
                draw.vertex_shader(),
                draw.fragment_shader(),
            );
            self.describe_shaders(memory, draw);
        }
        if self.capture.is_some() {
            self.capture_shaders(memory, draw);
        }
        Ok(())
    }

    fn clear_color(
        &mut self,
        _memory: &dyn GpuMemory,
        clear: &ClearColorCall,
    ) -> Result<(), GraphicsError> {
        self.color_clears += 1;
        if self.dump {
            println!(
                "[recomp-gfx] clear color target {} layer {} mask {:#x} color {:?} at {:#x} {}x{}",
                clear.index,
                clear.layer,
                clear.component_mask,
                clear.color,
                clear.target.address,
                clear.target.width_or_stride,
                clear.target.height,
            );
        }
        Ok(())
    }

    fn clear_depth_stencil(
        &mut self,
        _memory: &dyn GpuMemory,
        clear: &ClearDepthStencilCall,
    ) -> Result<(), GraphicsError> {
        self.depth_clears += 1;
        if self.dump {
            println!(
                "[recomp-gfx] clear depth={} stencil={} depth_value={} stencil={} at {:#x} {}x{}",
                clear.clear_depth,
                clear.clear_stencil,
                clear.depth,
                clear.stencil,
                clear.target.address,
                clear.width,
                clear.height,
            );
        }
        Ok(())
    }

    fn dispatch(
        &mut self,
        memory: &dyn GpuMemory,
        dispatch: &ComputeDispatch,
    ) -> Result<(), GraphicsError> {
        if self.dump {
            println!(
                "[recomp-gfx] dispatch {:#x} grid {:?} block {:?} shared {:#x}",
                dispatch.program_address,
                dispatch.grid,
                dispatch.block,
                dispatch.shared_memory_size
            );
        }
        if self.capture.is_some() {
            self.capture_program(
                memory,
                dispatch.program_address,
                "compute",
                dispatch.constant_buffers[1],
            );
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), GraphicsError> {
        self.submissions += 1;
        if self.dump {
            println!(
                "[recomp-gfx] submission {} finished: {} draws ({} indexed), {} color clears, {} depth clears",
                self.submissions,
                self.draws,
                self.indexed_draws,
                self.color_clears,
                self.depth_clears
            );
        }
        Ok(())
    }
}

impl EarlyBackend {
    /// Write each program a draw uses that has not been written yet: as many
    /// bytes from its start as are mapped, up to `PROGRAM_CAPTURE_BYTES`.
    fn capture_shaders(&mut self, memory: &dyn GpuMemory, draw: &DrawCall) {
        for (stage, address) in draw.shader_addresses.iter().enumerate() {
            let Some(address) = *address else {
                continue;
            };
            let constants = draw
                .constant_buffer(stage, 1)
                .map(|binding| (binding.address, binding.size));
            self.capture_program(memory, address, &stage.to_string(), constants);
        }
    }

    /// Write one program to `<address>-<stage>.bin`, and its constant buffer
    /// 1 (address and size) to `<address>-<stage>.cb1`, once.
    fn capture_program(
        &mut self,
        memory: &dyn GpuMemory,
        address: u64,
        stage: &str,
        constants: Option<(u64, u32)>,
    ) {
        let Some(dir) = self.capture.clone() else {
            return;
        };
        if self.captured.insert(address) {
            let mut bytes = Vec::new();
            while (bytes.len() as u64) < PROGRAM_CAPTURE_BYTES {
                match memory.read_gpu(address + bytes.len() as u64, 0x100) {
                    Some(chunk) => bytes.extend_from_slice(&chunk),
                    None => break,
                }
            }
            let path = dir.join(format!("{address:x}-{stage}.bin"));
            match std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, &bytes)) {
                Ok(()) => println!(
                    "[recomp-gfx] captured shader {address:#x} (stage {stage}, {} bytes) to {}",
                    bytes.len(),
                    path.display()
                ),
                Err(error) => println!("[recomp-gfx] capturing {}: {error}", path.display()),
            }
            let constants = constants
                .and_then(|(address, size)| memory.read_gpu(address, size.min(0x1_0000) as usize));
            if let Some(constants) = constants {
                let path = path.with_extension("cb1");
                if let Err(error) = std::fs::write(&path, constants) {
                    println!("[recomp-gfx] capturing {}: {error}", path.display());
                }
            }
        }
    }

    /// Print the program header of each shader a draw uses, once per program.
    /// With `RECOMP_DUMP_SHADERS=1` the first instructions follow, disassembled.
    fn describe_shaders(&mut self, memory: &dyn GpuMemory, draw: &DrawCall) {
        for (stage, address) in draw.shader_addresses.iter().enumerate() {
            let Some(address) = *address else {
                continue;
            };
            if self.seen_shaders.contains_key(&address) {
                continue;
            }
            let Some(header) = sph::ShaderProgramHeader::read(memory, address) else {
                self.seen_shaders.insert(address, String::new());
                println!("[recomp-gfx] shader {address:#x} (stage {stage}): no header");
                continue;
            };
            println!(
                "[recomp-gfx] shader {address:#x} (stage {stage}) {}",
                header.describe()
            );
            if self.dump_shaders {
                // The listing stops at the first unknown opcode: print that
                // word's raw bits so the pattern table can be extended from a
                // real program's encodings.
                for instruction in sph::disassemble(memory, address, PROGRAM_DUMP_BYTES)
                    .into_iter()
                    .take(INSTRUCTIONS_DUMPED)
                {
                    if instruction.known() {
                        println!("[recomp-gfx]   {}", instruction.format());
                    } else {
                        println!(
                            "[recomp-gfx]   {:#06x}: unknown {:016x}",
                            instruction.address, instruction.word
                        );
                        break;
                    }
                }
            }
            self.seen_shaders.insert(address, header.describe());
        }
    }
}

/// Per-channel state the command processor keeps between submissions.
pub struct Channel {
    processor: CommandProcessor<Box<dyn Renderer + Send>>,
    /// Total command lists run, for tracing and progress reports.
    pub submitted_lists: u64,
    /// Command list words, reused across lists (a title runs thousands).
    words: Vec<u32>,
}

impl std::fmt::Debug for Channel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Channel")
            .field("bound", &self.processor.bound)
            .field("stats", &self.processor.stats)
            .field("submitted_lists", &self.submitted_lists)
            .finish()
    }
}

impl Default for Channel {
    fn default() -> Self {
        Self::new()
    }
}

impl Channel {
    pub fn new() -> Self {
        Self {
            processor: CommandProcessor::new(channel_renderer()),
            submitted_lists: 0,
            words: Vec::new(),
        }
    }

    /// What the channel has processed so far.
    pub fn stats(&self) -> ProcessorStats {
        self.processor.stats
    }

    /// The classes bound to each subchannel, for tracing and diagnostics.
    pub fn bound(&self) -> [u32; 8] {
        self.processor.bound
    }

    /// Show the last frame again (see `Renderer::refresh`).
    pub fn refresh(&mut self) {
        if let Err(error) = self.processor.renderer.refresh() {
            println!("[recomp-gpu] refresh failed: {error}");
        }
    }

    /// Show a frame the title queued to its display (see `present`).
    pub fn present(&mut self, addresses: &[u64], crop: Option<(u32, u32)>) {
        // Give the renderer the backing addresses: the guest maps one
        // display buffer at several GPU addresses, and its targets are keyed
        // by the backing.
        let mappings = mappings();
        let addresses: Vec<u64> = addresses
            .iter()
            .filter_map(|address| translate(&mappings, *address))
            .collect();
        if let Err(error) = self.processor.renderer.present(&addresses, crop) {
            println!("[recomp-gpu] present failed: {error}");
        }
    }

    /// Run the command lists of `entries`.
    pub fn submit(
        &mut self,
        mem: &dyn MemoryAccess,
        mappings: &BTreeMap<u64, (u64, u64)>,
        entries: &[u64],
    ) {
        self.processor.trace =
            tracing() || trace_after().is_some_and(|after| self.submitted_lists >= after);
        self.processor.trace_macro = macro_tracing();
        let started = std::time::Instant::now();

        for entry in entries {
            let va = entry & ((1 << 40) - 1);
            let words = ((entry >> 42) & 0x1F_FFFF) as usize;
            if words == 0 {
                continue;
            }
            self.submitted_lists += 1;
            let Some(cpu) = translate(mappings, va) else {
                println!("[recomp-gpu] command list at unmapped GPU address {va:#x}");
                continue;
            };
            let Some(bytes) = mem.read_bytes(cpu, words * 4) else {
                println!("[recomp-gpu] command list at {va:#x} ({cpu:#x}) is not readable");
                continue;
            };
            // Reuse the word buffer: the list is decoded synchronously.
            self.words.clear();
            self.words.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_le_bytes(*word)),
            );

            let mut memory = GpuAdapter { mem, mappings };
            self.processor.submit_list(&mut memory, &self.words);
            // A list that ends with a macro started but not yet run still runs
            // it, unless the macro's parameters continue in the next list (an
            // indirect draw's parameters are an entry of their own).
            if !self.processor.continues_method() {
                self.processor.run_pending_macros(&mut memory);
            }
        }

        let recorded = std::time::Instant::now();
        if let Err(error) = self.processor.finish_submission() {
            println!("[recomp-gpu] submission failed: {error}");
        }
        if gpu_stats() {
            busy_report(recorded - started, recorded.elapsed());
        }
        if gpu_stats()
            && self.submitted_lists / 64 != (self.submitted_lists - entries.len() as u64) / 64
        {
            let stats = &self.processor.stats;
            println!(
                "[recomp-gpu] channel stats: {} lists, {} methods, {} 3D state, {} macros, {} draws, {} clears, {} dispatches ({} on the CPU, {} instructions), {} inline words, {} semaphores, {} blits, {} other, bound {:x?}",
                stats.command_lists,
                stats.method_writes,
                stats.state_writes,
                stats.macros,
                stats.draws,
                stats.color_clears + stats.depth_clears,
                stats.compute_dispatches,
                stats.cpu_dispatches,
                stats.cpu_instructions,
                stats.inline_copy_words,
                stats.semaphores,
                stats.blits,
                stats.other_class_writes,
                self.processor.bound,
            );
        }
    }
}

/// Time guest threads spent waiting on syncpoints (the GPU finishing their
/// work), for `RECOMP_GPU_STATS`.
static SYNCPOINT_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// With `RECOMP_GPU_STATS`, every 2 s: how much of the time the GPU worker
/// spent decoding and recording command lists, and submitting and waiting
/// for the host GPU (a worker busy most of the time limits the frame rate).
fn busy_report(recording: std::time::Duration, submitting: std::time::Duration) {
    struct Busy {
        since: std::time::Instant,
        recording: std::time::Duration,
        submitting: std::time::Duration,
        submissions: u32,
    }
    static BUSY: Mutex<Option<Busy>> = Mutex::new(None);
    let mut busy = locked(&BUSY);
    let busy = busy.get_or_insert_with(|| Busy {
        since: std::time::Instant::now(),
        recording: std::time::Duration::ZERO,
        submitting: std::time::Duration::ZERO,
        submissions: 0,
    });
    busy.recording += recording;
    busy.submitting += submitting;
    busy.submissions += 1;
    let elapsed = busy.since.elapsed();
    if elapsed >= std::time::Duration::from_secs(2) {
        let share = |part: std::time::Duration| 100.0 * part.as_secs_f64() / elapsed.as_secs_f64();
        let waited = std::time::Duration::from_nanos(
            SYNCPOINT_WAIT_NS.swap(0, std::sync::atomic::Ordering::Relaxed),
        );
        println!(
            "[recomp-gpu] worker busy: {:.0}% decoding and recording, {:.0}% submitting and waiting ({} submissions in {:.1} s); guest threads waited on syncpoints {:.0}% of the time (summed)",
            share(busy.recording),
            share(busy.submitting),
            busy.submissions,
            elapsed.as_secs_f64(),
            share(waited)
        );
        *busy = Busy {
            since: std::time::Instant::now(),
            recording: std::time::Duration::ZERO,
            submitting: std::time::Duration::ZERO,
            submissions: 0,
        };
    }
}

/// `RECOMP_TRACE_GPU_AFTER=<lists>`: trace methods like `RECOMP_TRACE_GPU`
/// once a channel has run that many command lists (skipping the millions a
/// title's loading sends).
fn trace_after() -> Option<u64> {
    static AFTER: OnceLock<Option<u64>> = OnceLock::new();
    *AFTER.get_or_init(|| {
        std::env::var("RECOMP_TRACE_GPU_AFTER")
            .ok()
            .and_then(|lists| lists.parse().ok())
    })
}

/// `RECOMP_GPU_STATS=1`: print each channel's processor counters every 64
/// command lists (what a title's submissions contain).
fn gpu_stats() -> bool {
    static STATS: OnceLock<bool> = OnceLock::new();
    *STATS.get_or_init(|| std::env::var_os("RECOMP_GPU_STATS").is_some())
}

/// What the GPU worker does next, in the order the title asked for it.
#[derive(Debug)]
enum Work {
    Submit(Submission),
    Present(Present),
}

/// A frame queued to the display: shown once its fences are reached, as the
/// console's compositor waits for them.
#[derive(Debug)]
struct Present {
    /// The GPU addresses the buffer is mapped at.
    addresses: Vec<u64>,
    crop: Option<(u32, u32)>,
    /// (syncpoint, value) pairs the frame's GPU work reaches.
    fences: Vec<(u32, u32)>,
}

/// A submission waiting for the GPU worker: the lists and the fence they complete.
#[derive(Debug)]
struct Submission {
    /// The channel that submitted, named by the syncpoint it signals.
    channel: u32,
    entries: Vec<u64>,
    /// The syncpoint this submission reaches, and the value it reaches.
    fence: (u32, u32),
}

/// Syncpoint counters, per syncpoint: the values handed out to submissions and
/// the values the worker has reached. A wait blocks until the reached value
/// reaches its threshold.
#[derive(Debug, Default)]
struct Syncpoints {
    issued: BTreeMap<u32, u32>,
    reached: BTreeMap<u32, u32>,
}

/// What the submitting threads and the GPU worker thread share.
struct Shared {
    queue: Mutex<VecDeque<Work>>,
    /// Signals the worker that the queue has work (or that memory arrived).
    work: Condvar,
    syncpoints: Mutex<Syncpoints>,
    /// GPU address -> (CPU address, size), owned here so the worker can read
    /// command lists while the driver keeps mapping and unmapping buffers.
    mappings: Mutex<BTreeMap<u64, (u64, u64)>>,
    /// The guest memory, installed by the runtime when the process starts.
    memory: Mutex<Option<Arc<dyn MemoryAccess>>>,
    /// One command processor per channel, keyed by its syncpoint.
    channels: Mutex<BTreeMap<u32, Channel>>,
    /// Threads blocked in a syncpoint wait, as (syncpoint, threshold).
    wait: Mutex<Vec<(u32, u32)>>,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            work: Condvar::new(),
            syncpoints: Mutex::new(Syncpoints::default()),
            mappings: Mutex::new(BTreeMap::new()),
            memory: Mutex::new(None),
            channels: Mutex::new(BTreeMap::new()),
            wait: Mutex::new(Vec::new()),
        }
    }
}

fn locked<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Submissions run on the worker thread unless `RECOMP_GPU_THREAD=0`.
pub fn async_submissions() -> bool {
    static ASYNC: OnceLock<bool> = OnceLock::new();
    *ASYNC.get_or_init(|| std::env::var("RECOMP_GPU_THREAD").map_or(true, |value| value != "0"))
}

/// The GPU state, with the worker thread started once.
fn shared() -> &'static Arc<Shared> {
    static GPU: OnceLock<Arc<Shared>> = OnceLock::new();
    GPU.get_or_init(|| {
        let shared = Arc::new(Shared::default());
        if async_submissions() {
            let worker = Arc::clone(&shared);
            match std::thread::Builder::new()
                .name("recomp-gpu".to_string())
                .spawn(move || run(worker))
            {
                Ok(_) => {}
                Err(error) => eprintln!("[recomp-gpu] could not start the GPU worker: {error}"),
            }
        }
        shared
    })
}

/// The guest memory the worker decodes command lists from. The runtime installs
/// it before the guest runs.
pub fn attach_memory(memory: Arc<dyn MemoryAccess>) {
    *locked(&shared().memory) = Some(memory);
    shared().work.notify_one();
}

/// The GPU address space's mappings, shared by the driver and the worker.
pub fn mappings() -> std::sync::MutexGuard<'static, BTreeMap<u64, (u64, u64)>> {
    locked(&shared().mappings)
}

/// The GPU addresses memory at CPU address `cpu` is mapped at.
pub fn gpu_addresses_of(cpu: u64) -> Vec<u64> {
    mappings()
        .iter()
        .filter(|(_, (base, size))| (*base..base + size).contains(&cpu))
        .map(|(va, (base, _))| va + (cpu - base))
        .collect()
}

/// Show the frame in the display buffer mapped at one of `addresses` once
/// the GPU reaches `fences`: vi's QueueBuffer. Frames are shown in the order
/// they were queued, after the submissions queued before them.
pub fn present(addresses: Vec<u64>, crop: Option<(u32, u32)>, fences: Vec<(u32, u32)>) {
    let request = Present {
        addresses,
        crop,
        fences,
    };
    if async_submissions() {
        locked(&shared().queue).push_back(Work::Present(request));
        shared().work.notify_one();
    } else {
        show(shared(), &request);
    }
}

fn show(shared: &Shared, request: &Present) {
    for channel in locked(&shared.channels).values_mut() {
        channel.present(&request.addresses, request.crop);
    }
}

/// Show the waiting frames whose fences are reached, in order. A frame that
/// waits behind more than a few others is shown anyway: its fence is on a
/// syncpoint the worker does not advance.
fn show_ready(shared: &Shared, waiting: &mut VecDeque<Present>) {
    while let Some(front) = waiting.front() {
        let reached = front
            .fences
            .iter()
            .all(|(syncpoint, value)| syncpoint_value(*syncpoint) >= *value);
        if !reached && waiting.len() <= 3 {
            break;
        }
        let request = waiting.pop_front().expect("front exists");
        show(shared, &request);
    }
}

/// The value the next submission on `syncpoint` will reach.
pub fn allocate_fence(syncpoint: u32) -> u32 {
    let mut syncpoints = locked(&shared().syncpoints);
    let value = syncpoints.issued.get(&syncpoint).copied().unwrap_or(0) + 1;
    syncpoints.issued.insert(syncpoint, value);
    value
}

/// The value submissions have been asked to reach, as AllocGPFIFOEx2 reports it.
pub fn syncpoint_pending(syncpoint: u32) -> u32 {
    locked(&shared().syncpoints)
        .issued
        .get(&syncpoint)
        .copied()
        .unwrap_or(0)
}

/// The value `syncpoint` has reached.
pub fn syncpoint_value(syncpoint: u32) -> u32 {
    locked(&shared().syncpoints)
        .reached
        .get(&syncpoint)
        .copied()
        .unwrap_or(0)
}

/// Advance `syncpoint` by one, waking its waiters: the multimedia engines
/// (`host1x`) run on the submitting thread and signal as they finish.
pub fn signal_syncpoint(syncpoint: u32) {
    {
        let mut syncpoints = locked(&shared().syncpoints);
        let value = syncpoints.reached.get(&syncpoint).copied().unwrap_or(0) + 1;
        syncpoints.reached.insert(syncpoint, value);
    }
    crate::kernel::kernel().notify_waiters();
}

/// Threads blocked waiting for a GPU syncpoint.
pub fn waiting() -> usize {
    locked(&shared().wait).len()
}

/// The blocked syncpoint waits, as (syncpoint, threshold), for hang reports.
pub fn waiters() -> Vec<(u32, u32)> {
    locked(&shared().wait).clone()
}

/// Submit `entries` on `channel` and promise the fence reaches `fence`. With the
/// worker running this returns before the command lists are decoded, which is
/// what makes a later wait on the fence actually block.
pub fn submit(channel: u32, mem: &dyn MemoryAccess, entries: Vec<u64>, fence: (u32, u32)) {
    if async_submissions() {
        locked(&shared().queue).push_back(Work::Submit(Submission {
            channel,
            entries,
            fence,
        }));
        shared().work.notify_one();
    } else {
        submit_inline(channel, mem, &entries, fence);
    }
}

/// Decode a submission on the calling thread (`RECOMP_GPU_THREAD=0`).
fn submit_inline(channel: u32, mem: &dyn MemoryAccess, entries: &[u64], fence: (u32, u32)) {
    let mappings = locked(&shared().mappings).clone();
    {
        let mut channels = locked(&shared().channels);
        channels
            .entry(channel)
            .or_default()
            .submit(mem, &mappings, entries);
    }
    complete(shared(), fence);
}

/// A submission's lists are done: advance its syncpoint and wake its waiters.
///
/// The kernel's waiters are notified after the syncpoint lock is released, and
/// with no GPU lock held, so a thread waiting in the kernel cannot deadlock
/// against the worker.
fn complete(shared: &Shared, fence: (u32, u32)) {
    let (syncpoint, value) = fence;
    {
        let mut syncpoints = locked(&shared.syncpoints);
        let reached = syncpoints.reached.get(&syncpoint).copied().unwrap_or(0);
        syncpoints.reached.insert(syncpoint, value.max(reached));
    }
    crate::kernel::kernel().notify_waiters();
}

/// Wait until `syncpoint` reaches `threshold`. `timeout_ns` of -1 waits forever,
/// 0 only polls. Returns the value read at the end, which is below the threshold
/// when the wait timed out.
pub fn wait_for_syncpoint(syncpoint: u32, threshold: u32, timeout_ns: i64) -> u32 {
    let shared = shared();
    let entry = (syncpoint, threshold);
    locked(&shared.wait).push(entry);
    let started = std::time::Instant::now();
    let reached =
        crate::kernel::kernel().wait_for(timeout_ns, || syncpoint_value(syncpoint) >= threshold);
    SYNCPOINT_WAIT_NS.fetch_add(
        started.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    let mut waits = locked(&shared.wait);
    if let Some(index) = waits.iter().position(|wait| *wait == entry) {
        waits.remove(index);
    }
    drop(waits);
    let value = syncpoint_value(syncpoint);
    if tracing() {
        println!(
            "[recomp-gpu] syncpoint {syncpoint} waited for {threshold}: {}",
            if reached { "reached" } else { "timed out" },
        );
    }
    value
}

/// How long the worker waits for work before asking the renderers to show
/// their last frame again.
const IDLE_REFRESH: std::time::Duration = std::time::Duration::from_millis(100);

/// The GPU worker: take submissions in order, decode them, complete their fences.
fn run(shared: Arc<Shared>) {
    let mut batch: Vec<Work> = Vec::new();
    let mut waiting: VecDeque<Present> = VecDeque::new();
    let mut refreshed = std::time::Instant::now();
    // Let a window show its last frame again (the renderer decides whether
    // it is due): the title may submit work and present nothing for a long
    // time, or submit nothing at all.
    let refresh = |refreshed: &mut std::time::Instant| {
        if refreshed.elapsed() >= IDLE_REFRESH {
            for channel in locked(&shared.channels).values_mut() {
                channel.refresh();
            }
            *refreshed = std::time::Instant::now();
        }
    };
    loop {
        let mut queue = locked(&shared.queue);
        while queue.is_empty() {
            let (next, timeout) = shared
                .work
                .wait_timeout(queue, IDLE_REFRESH)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            queue = next;
            if timeout.timed_out() && queue.is_empty() {
                drop(queue);
                refresh(&mut refreshed);
                queue = locked(&shared.queue);
            }
        }
        batch.clear();
        batch.extend(queue.drain(..));
        drop(queue);

        let Some(memory) = locked(&shared.memory).clone() else {
            // No guest memory yet: nothing can be decoded, so complete the
            // fences rather than leave a thread waiting for a GPU that is not
            // wired up.
            for work in &batch {
                if let Work::Submit(submission) = work {
                    complete(&shared, submission.fence);
                }
            }
            continue;
        };
        let mappings = locked(&shared.mappings).clone();

        for work in batch.drain(..) {
            let submission = match work {
                Work::Submit(submission) => submission,
                Work::Present(request) => {
                    waiting.push_back(request);
                    show_ready(&shared, &mut waiting);
                    continue;
                }
            };
            if tracing() {
                println!(
                    "[recomp-gpu] worker: channel {:#x}, {} command lists, fence ({:#x}, {})",
                    submission.channel,
                    submission.entries.len(),
                    submission.fence.0,
                    submission.fence.1,
                );
            }
            {
                // The channel's processor is locked for one submission only:
                // completing a fence takes other locks, which a waiting thread
                // may hold part of.
                let mut channels = locked(&shared.channels);
                channels.entry(submission.channel).or_default().submit(
                    &*memory,
                    &mappings,
                    &submission.entries,
                );
            }
            complete(&shared, submission.fence);
            show_ready(&shared, &mut waiting);
        }
        refresh(&mut refreshed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn gpu_ticks_run_at_384_per_625_nanoseconds() {
        assert_eq!(gpu_ticks(625), 384);
        assert_eq!(gpu_ticks(1_000_000_000), 614_400_000);
        // Long uptimes do not overflow.
        assert_eq!(
            gpu_ticks(u64::MAX / 2),
            (u128::from(u64::MAX / 2) * 384 / 625) as u64
        );
    }

    /// Guest memory backed by a plain buffer, at the same addresses the GPU
    /// mappings point at.
    #[derive(Default, Debug)]
    struct TestMemory {
        bytes: Mutex<Vec<u8>>,
    }

    impl TestMemory {
        fn new(size: usize) -> Self {
            Self {
                bytes: Mutex::new(vec![0; size]),
            }
        }
    }

    impl MemoryAccess for TestMemory {
        fn load_u8(&self, addr: u64) -> Option<u8> {
            self.bytes.lock().unwrap().get(addr as usize).copied()
        }

        fn load_u32(&self, addr: u64) -> Option<u32> {
            let guard = self.bytes.lock().unwrap();
            let raw: [u8; 4] = guard
                .get(addr as usize..addr as usize + 4)?
                .try_into()
                .ok()?;
            Some(u32::from_le_bytes(raw))
        }

        fn load_u64(&self, addr: u64) -> Option<u64> {
            let guard = self.bytes.lock().unwrap();
            let raw: [u8; 8] = guard
                .get(addr as usize..addr as usize + 8)?
                .try_into()
                .ok()?;
            Some(u64::from_le_bytes(raw))
        }

        fn store_u32(&self, addr: u64, value: u32) -> bool {
            self.write_bytes(addr, &value.to_le_bytes())
        }

        fn store_u64(&self, addr: u64, value: u64) -> bool {
            self.write_bytes(addr, &value.to_le_bytes())
        }

        fn read_bytes(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
            let guard = self.bytes.lock().unwrap();
            guard
                .get(addr as usize..addr as usize + len)
                .map(|bytes| bytes.to_vec())
        }

        fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
            let mut guard = self.bytes.lock().unwrap();
            let start = addr as usize;
            let end = start + bytes.len();
            if end > guard.len() {
                return false;
            }
            guard[start..end].copy_from_slice(bytes);
            true
        }
    }

    /// A command header: method, subchannel, count and submission mode.
    fn header(method: u32, subchannel: usize, count: usize, mode: u32) -> u32 {
        method | ((subchannel as u32) << 13) | ((count as u32) << 16) | (mode << 29)
    }

    /// A command list written into guest memory, with the GPFIFO entry that
    /// points at it.
    fn list_in(memory: &TestMemory, gpu_va: u64, list: &[u32]) -> u64 {
        let bytes: Vec<u8> = list.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert!(memory.write_bytes(gpu_va, &bytes));
        gpu_va | ((list.len() as u64) << 42)
    }

    #[test]
    fn a_bound_class_and_semaphore_release_reach_guest_memory() {
        let mem = TestMemory::new(0x2000);
        let mut mappings = BTreeMap::new();
        mappings.insert(0x1000, (0x1000, 0x1000));

        let list = vec![
            header(0x0, 0, 1, 0),
            0xB197,
            header(0x4, 0, 1, 0),
            0,
            header(0x5, 0, 1, 0),
            0x1200,
            header(0x6, 0, 1, 0),
            0x2a,
            // Operation 2 is a release, short form.
            header(0x7, 0, 1, 0),
            2,
        ];
        let entry = list_in(&mem, 0x1000, &list);

        let mut channel = Channel::new();
        channel.submit(&mem, &mappings, &[entry]);

        assert_eq!(channel.bound()[0], 0xB197);
        assert_eq!(mem.load_u32(0x1200), Some(0x2a));
        assert_eq!(channel.stats().semaphores, 1);
        assert_eq!(channel.stats().method_writes, 5);
    }

    #[test]
    fn a_3d_report_writes_the_payload_through_the_mapping() {
        let mem = TestMemory::new(0x2000);
        let mut mappings = BTreeMap::new();
        mappings.insert(0x1000, (0x1000, 0x1000));

        // Bind the 3D class, set the report address and payload, then query.
        let list = vec![
            header(0x0, 0, 1, 0),
            0xB197,
            header(0x6C0, 0, 3, 0),
            0,
            0x1400,
            0x55,
            header(0x6C3, 0, 1, 0),
            0,
        ];
        let entry = list_in(&mem, 0x1000, &list);

        let mut channel = Channel::new();
        channel.submit(&mem, &mappings, &[entry]);

        assert_eq!(mem.load_u32(0x1400), Some(0x55));
        assert_eq!(channel.stats().semaphores, 1);
        assert_eq!(channel.stats().state_writes, 4);
    }

    #[test]
    fn an_unmapped_command_list_is_reported_not_run() {
        let mem = TestMemory::new(0x1000);
        let mut channel = Channel::new();
        channel.submit(&mem, &BTreeMap::new(), &[0x2000 | (4 << 42)]);
        assert_eq!(channel.stats().method_writes, 0);
        assert_eq!(channel.submitted_lists, 1);
    }

    /// A submission on the worker thread: the fence is not reached until the
    /// lists are decoded, and a wait blocks until it is.
    #[test]
    fn a_worker_submission_completes_its_fence() {
        let mem = Arc::new(TestMemory::new(0x2000));
        mappings().insert(0x1000, (0x1000, 0x1000));
        attach_memory(mem.clone());

        // Bind the 3D class and release a short semaphore.
        let list = vec![
            header(0x0, 0, 1, 0),
            0xB197,
            header(0x4, 0, 1, 0),
            0,
            header(0x5, 0, 1, 0),
            0x1200,
            header(0x6, 0, 1, 0),
            0x2a,
            header(0x7, 0, 1, 0),
            2,
        ];
        let entry = list_in(&mem, 0x1000, &list);

        let fence = allocate_fence(0x10);
        submit(0x10, &*mem, vec![entry], (0x10, fence));
        assert_eq!(syncpoint_pending(0x10), fence);

        // The wait returns only once the worker has decoded the list.
        let value = wait_for_syncpoint(0x10, fence, 5_000_000_000);
        assert_eq!(value, fence, "the fence is reached");
        assert_eq!(mem.load_u32(0x1200), Some(0x2a), "the payload is written");
        assert_eq!(waiting(), 0, "no thread is left blocked");
    }

    /// A wait on a value nothing has promised blocks, and a poll does not.
    #[test]
    fn a_syncpoint_poll_returns_the_value_it_found() {
        assert_eq!(wait_for_syncpoint(0x99, 1, 0), 0);
        assert_eq!(syncpoint_pending(0x99), 0);
    }
}
