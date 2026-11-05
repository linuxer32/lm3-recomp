//! The Maxwell 3D command processor.
//!
//! A command list is a stream of method writes. Each write goes to the class
//! bound to its subchannel; the 3D class (0xB197) keeps its state in the
//! register file in [`state`], and the methods that act rather than store
//! ([`methods::DRAW_END`], [`methods::CLEAR`], ...) become renderer calls.
//!
//! Methods from `0xE00` up are macro invocations, run by [`mme`]. Macros are
//! how games actually draw: the macro program writes the state registers and
//! the draw methods itself, so the processor does not need to know which
//! macro it is running.
//!
//! The processor is synchronous with the guest: a submission is walked on the
//! thread that submitted it, and guest memory is read through [`GpuMemory`]
//! while the state is stable.

use std::collections::VecDeque;

use crate::GraphicsError;

pub mod blocklinear;
pub mod compute;
pub mod interpret;
pub mod methods;
pub mod mme;
pub mod render;
pub mod sph;
pub mod state;
pub mod translate;
pub mod twod;

use mme::{FifoWord, MacroInterpreter, MacroState, MACRO_CODE_WORDS, MACRO_COUNT};
use render::{
    ClearColorCall, ClearDepthStencilCall, ConstantBufferBinding, ConstantBufferBindings, DrawCall,
    DrawParameters, GpuMemory, IndexType, PrimitiveTopology, Renderer, CONSTANT_BUFFER_SLOTS,
};
use state::MaxwellState;

use methods as m;

/// The class of the 3D engine, as bound by method 0 of a subchannel.
pub const CLASS_MAXWELL_3D: u32 = 0xB197;

/// Channel ("puller") methods, below the class state block.
mod puller {
    use crate::maxwell::methods as m;

    pub const BIND_OBJECT: u32 = 0x0;
    pub const SEMAPHORE_ADDRESS_HIGH: u32 = 0x4;
    pub const SEMAPHORE_ADDRESS_LOW: u32 = 0x5;
    pub const SEMAPHORE_SEQUENCE: u32 = 0x6;
    pub const SEMAPHORE_TRIGGER: u32 = 0x7;
    /// The syncpoint operation (after its payload, method 0x1C): bit 0
    /// increments (else waits), the syncpoint index is in bits 8:19.
    pub const SYNCPOINT_OPERATION: u32 = 0x1D;
    pub const LOAD_MME_INSTRUCTION_RAM_POINTER: u32 =
        m::LOAD_MME_INSTRUCTION_RAM_POINTER as u32 / 4;
    pub const LOAD_MME_INSTRUCTION_RAM: u32 = m::LOAD_MME_INSTRUCTION_RAM as u32 / 4;
    pub const LOAD_MME_START_ADDRESS_RAM_POINTER: u32 =
        m::LOAD_MME_START_ADDRESS_RAM_POINTER as u32 / 4;
    pub const LOAD_MME_START_ADDRESS_RAM: u32 = m::LOAD_MME_START_ADDRESS_RAM as u32 / 4;
    pub const LAST: u32 = 0x5f;
}

/// The last method of the class state block, in words.
const CLASS_METHOD_LIMIT: u32 = (m::STATE_SIZE / 4) as u32;

/// Words of macro code copied out of the RAM for one call. Macro programs are
/// tens of instructions; this covers the largest known one with room to spare.
const MACRO_WINDOW: usize = 512;

/// What a submission did, for tracing and for the first speed measurements.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ProcessorStats {
    pub command_lists: u64,
    pub method_writes: u64,
    /// Writes into the 3D register file.
    pub state_writes: u64,
    /// Macro invocations, and the instructions they executed.
    pub macros: u64,
    pub macro_steps: u64,
    pub draws: u64,
    /// Draws that read an index buffer or inline index elements.
    pub indexed_draws: u64,
    pub color_clears: u64,
    pub depth_clears: u64,
    pub semaphores: u64,
    /// Words written by inline-to-memory copies.
    pub inline_copy_words: u64,
    /// Writes to a subchannel whose class is not the 3D or compute engine.
    pub other_class_writes: u64,
    pub compute_dispatches: u64,
    /// Dispatches run on the CPU (`interpret`), and the instructions they ran.
    pub cpu_dispatches: u64,
    pub cpu_instructions: u64,
    /// 2D engine copies.
    pub blits: u64,
}

/// Say once per syncpoint which command increments it.
fn report_syncpoint(syncpoint: u32, by: &str) {
    static REPORTED: std::sync::Mutex<std::collections::BTreeSet<u32>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    if REPORTED.lock().is_ok_and(|mut seen| seen.insert(syncpoint)) {
        println!("[recomp-gpu] syncpoint {syncpoint} incremented by {by}");
    }
}

/// Say once per program what a compute dispatch run on the CPU cost (the
/// interpreter is slow: big dispatches show in the frame rate).
fn report_cpu_dispatch(dispatch: &compute::ComputeDispatch, instructions: u64) {
    static REPORTED: std::sync::Mutex<std::collections::BTreeSet<u64>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    if REPORTED
        .lock()
        .is_ok_and(|mut seen| seen.insert(dispatch.program_address))
    {
        println!(
            "[recomp-gpu] compute program {:#x} runs on the CPU: grid {:?}, block {:?}, {instructions} instructions",
            dispatch.program_address, dispatch.grid, dispatch.block
        );
    }
}

/// Say once per engine class and method that the method is not run yet
/// (compute, DMA copies and the 2D engine are not implemented).
fn report_other_class(class: u32, method: u32) {
    static REPORTED: std::sync::Mutex<std::collections::BTreeSet<(u32, u32)>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    if REPORTED
        .lock()
        .is_ok_and(|mut seen| seen.insert((class, method)))
    {
        println!("[recomp-gpu] class {class:#06x} method {method:#05x} not run yet");
    }
}

/// The inline-to-memory methods, shared by the 3D, compute and I2M classes
/// (method numbers, 4-byte units).
mod inline {
    pub const FIRST: u32 = 0x060;
    pub const LINE_LENGTH_IN: u32 = 0x060;
    pub const LINE_COUNT: u32 = 0x061;
    pub const OFFSET_OUT_UPPER: u32 = 0x062;
    pub const OFFSET_OUT: u32 = 0x063;
    pub const PITCH_OUT: u32 = 0x064;
    pub const LAUNCH_DMA: u32 = 0x06C;
    pub const LOAD_INLINE_DATA: u32 = 0x06D;
    pub const LAST: u32 = 0x06D;
    /// The class of the stand-alone engine.
    pub const CLASS_INLINE_TO_MEMORY: u32 = 0xA140;
}

/// An inline-to-memory copy: `LAUNCH_DMA` starts it, and the data words that
/// follow (`LOAD_INLINE_DATA`) are written to `OFFSET_OUT`, `LINE_COUNT`
/// lines of `LINE_LENGTH_IN` bytes `PITCH_OUT` apart. NVN uploads compute
/// launch descriptors and constant data this way.
#[derive(Debug, Default)]
struct InlineCopy {
    registers: [u32; 14],
    /// The copy being filled: its bytes so far, and whether its destination
    /// is pitch-linear.
    data: Vec<u8>,
    active: bool,
    linear: bool,
}

impl InlineCopy {
    fn register(&self, method: u32) -> u32 {
        self.registers[(method - inline::FIRST) as usize]
    }

    fn size(&self) -> usize {
        self.register(inline::LINE_LENGTH_IN) as usize * self.register(inline::LINE_COUNT) as usize
    }

    /// Write one method; a copy's last data word writes it to memory.
    fn write(&mut self, memory: &mut dyn GpuMemory, method: u32, data: u32) {
        self.registers[(method - inline::FIRST) as usize] = data;
        match method {
            inline::LAUNCH_DMA => {
                self.data.clear();
                self.active = true;
                // Bit 0 of LAUNCH_DMA: the destination is pitch-linear.
                self.linear = data & 1 != 0;
            }
            inline::LOAD_INLINE_DATA if self.active => {
                self.data.extend_from_slice(&data.to_le_bytes());
                if self.data.len() >= self.size() {
                    self.finish(memory);
                }
            }
            _ => {}
        }
    }

    fn finish(&mut self, memory: &mut dyn GpuMemory) {
        self.active = false;
        let address = (u64::from(self.register(inline::OFFSET_OUT_UPPER) & 0xFF) << 32)
            | u64::from(self.register(inline::OFFSET_OUT));
        let (length, lines) = (
            self.register(inline::LINE_LENGTH_IN) as usize,
            self.register(inline::LINE_COUNT) as usize,
        );
        if !self.linear && lines > 1 {
            report_other_class(inline::CLASS_INLINE_TO_MEMORY, inline::LAUNCH_DMA);
            return;
        }
        let pitch = u64::from(self.register(inline::PITCH_OUT));
        for line in 0..lines {
            let bytes = &self.data[line * length..(line + 1) * length];
            memory.write_gpu(address + line as u64 * pitch, bytes);
        }
    }
}

/// The draw state the hardware keeps between method writes: which draw is next
/// and whether it reads the index buffer.
#[derive(Debug, Default, Clone)]
struct DrawFlags {
    indexed: bool,
    topology: PrimitiveTopology,
    /// The instance index of an instanced draw started by `DrawBegin`.
    instance_index: u32,
    /// Index elements pushed inline by the vertex-buffer element methods.
    inline_indices: Vec<u32>,
    /// The constant buffers bound per stage and slot. The bind methods act:
    /// each binds the buffer the constant buffer state names at that moment.
    constant_buffers: ConstantBufferBindings,
}

/// A submission's GPU memory, readable and writable: the processor writes
/// semaphores and inline constant-buffer data into it.
pub struct CommandProcessor<R: Renderer> {
    pub state: MaxwellState,
    pub renderer: R,
    /// Print every method write to stdout.
    pub trace: bool,
    /// Print macro invocations, macro runs, and the state methods macros send.
    /// Separate from `trace`, which is already very loud.
    pub trace_macro: bool,
    /// The class bound to each subchannel.
    pub bound: [u32; 8],
    pub stats: ProcessorStats,

    /// Puller semaphore address (high, low) and sequence.
    semaphore: [u32; 3],
    macro_code: Vec<u32>,
    macro_position: [u32; MACRO_COUNT],
    macro_fifo: Vec<VecDeque<FifoWord>>,
    /// The first argument of a macro started but not yet run.
    macro_pending: [Option<u32>; MACRO_COUNT],
    instruction_pointer: u32,
    start_pointer: u32,
    draw: DrawFlags,
    /// The compute class's registers.
    compute: compute::ComputeState,
    inline_copy: InlineCopy,
    /// The 2D class's registers.
    twod: twod::TwodState,
    /// A method whose data words continue in the next command list: the
    /// pusher's state carries over from one GPFIFO entry to the next (NVN
    /// puts an indirect draw's parameters, the indirect buffer, in an entry
    /// of their own after the macro call's header).
    pending: Option<PendingMethod>,
}

/// A method command's progress through its data words.
#[derive(Debug, Clone, Copy)]
struct PendingMethod {
    subchannel: usize,
    method: u32,
    /// The header's mode: increasing, non-increasing or increase-once.
    mode: u32,
    count: usize,
    /// Data words written so far.
    written: usize,
}

impl<R: Renderer> CommandProcessor<R> {
    pub fn new(renderer: R) -> Self {
        Self {
            state: MaxwellState::default(),
            renderer,
            trace: false,
            trace_macro: false,
            bound: [0; 8],
            stats: ProcessorStats::default(),
            semaphore: [0; 3],
            macro_code: vec![0; MACRO_CODE_WORDS],
            macro_position: [0; MACRO_COUNT],
            macro_fifo: vec![VecDeque::new(); MACRO_COUNT],
            macro_pending: [None; MACRO_COUNT],
            instruction_pointer: 0,
            start_pointer: 0,
            draw: DrawFlags::default(),
            compute: compute::ComputeState::default(),
            inline_copy: InlineCopy::default(),
            twod: twod::TwodState::default(),
            pending: None,
        }
    }

    /// Whether the last list ended inside a method's data, which the next
    /// list continues.
    pub fn continues_method(&self) -> bool {
        self.pending.is_some()
    }

    /// Write a method's data words from `list[start..]`, as many as the
    /// list holds; returns where the method's data stopped.
    fn method_data(
        &mut self,
        memory: &mut dyn GpuMemory,
        pending: &mut PendingMethod,
        list: &[u32],
        start: usize,
    ) -> usize {
        let available = (pending.count - pending.written).min(list.len() - start);
        for (index, data) in list[start..start + available].iter().enumerate() {
            let offset = pending.written + index;
            let target = match pending.mode {
                0 | 1 => pending.method + offset as u32,
                5 if offset > 0 => pending.method + 1,
                _ => pending.method,
            };
            self.write_method(
                memory,
                pending.subchannel,
                target,
                *data,
                offset + 1 == pending.count,
            );
        }
        pending.written += available;
        start + available
    }

    /// Start the dispatch the compute registers describe.
    fn launch_compute(&mut self, memory: &mut dyn GpuMemory) {
        self.stats.compute_dispatches += 1;
        let Some(dispatch) = self.compute.dispatch(memory) else {
            if self.trace {
                println!("[recomp-gpu] compute launch descriptor not readable");
            }
            return;
        };
        // Programs that only compute integers and access memory run here,
        // against guest memory, where the title reads their results.
        let mut code = Vec::new();
        while code.len() < 0x1_0000 {
            match memory.read_gpu(dispatch.program_address + code.len() as u64, 0x100) {
                Some(chunk) => code.extend_from_slice(&chunk),
                None => break,
            }
        }
        match interpret::run(memory, &dispatch, &code) {
            Ok(steps) => {
                self.stats.cpu_dispatches += 1;
                self.stats.cpu_instructions += steps;
                report_cpu_dispatch(&dispatch, steps);
                return;
            }
            Err(refused) => {
                if self.trace {
                    println!(
                        "[recomp-gpu] compute program {:#x} left to the renderer: {}",
                        dispatch.program_address, refused.0
                    );
                }
            }
        }
        if let Err(error) = self.renderer.dispatch(memory, &dispatch) {
            if self.trace {
                println!("[recomp-gpu] dispatch failed: {error}");
            }
        }
    }

    /// Run a command list: header words followed by their data words.
    pub fn submit_list(&mut self, memory: &mut dyn GpuMemory, list: &[u32]) {
        self.stats.command_lists += 1;

        let mut index = 0;
        // Data words of a method whose header was in an earlier list.
        if let Some(mut pending) = self.pending.take() {
            index = self.method_data(memory, &mut pending, list, 0);
            if pending.written < pending.count {
                self.pending = Some(pending);
                return;
            }
        }
        while index < list.len() {
            let header = list[index];
            index += 1;

            let method = header & 0x1fff;
            let subchannel = ((header >> 13) & 7) as usize;
            let count = ((header >> 16) & 0x1fff) as usize;

            match header >> 29 {
                // Inline: the data word is in the header itself.
                4 => self.write_method(memory, subchannel, method, count as u32, true),
                // Increasing, non-increasing, increase-once (and the old
                // encodings of the first two).
                mode @ (0..=3 | 5) => {
                    let mut pending = PendingMethod {
                        subchannel,
                        method,
                        mode,
                        count,
                        written: 0,
                    };
                    index = self.method_data(memory, &mut pending, list, index);
                    if pending.written < pending.count {
                        self.pending = Some(pending);
                    }
                }
                _ => {
                    if self.trace {
                        println!("[recomp-gpu] unknown command header {header:#010x}");
                    }
                }
            }
        }
    }

    /// Run a macro that is waiting for its arguments, if one is.
    pub fn run_pending_macros(&mut self, memory: &mut dyn GpuMemory) {
        for index in 0..MACRO_COUNT {
            if let Some(arg0) = self.macro_pending[index].take() {
                self.run_macro(memory, index, arg0);
            }
        }
    }

    /// Called when a submission finishes. Backends present here.
    pub fn finish_submission(&mut self) -> Result<(), GraphicsError> {
        self.renderer.finish()
    }

    fn write_method(
        &mut self,
        memory: &mut dyn GpuMemory,
        subchannel: usize,
        method: u32,
        data: u32,
        is_last_call: bool,
    ) {
        self.stats.method_writes += 1;

        if self.trace {
            println!(
                "[recomp-gpu] sub {subchannel} class {:#06x} method {method:#05x} = {data:#010x}",
                self.bound[subchannel]
            );
        }

        if method <= puller::LAST {
            self.write_puller(memory, subchannel, method, data);
            return;
        }

        let class = self.bound[subchannel];
        if (inline::FIRST..=inline::LAST).contains(&method)
            && matches!(
                class,
                CLASS_MAXWELL_3D | compute::CLASS_KEPLER_COMPUTE | inline::CLASS_INLINE_TO_MEMORY
            )
        {
            if method == inline::LOAD_INLINE_DATA {
                self.stats.inline_copy_words += 1;
            }
            self.inline_copy.write(memory, method, data);
            if class == CLASS_MAXWELL_3D {
                self.state.write(method.wrapping_mul(4) as usize, data);
            }
            return;
        }

        if class == twod::CLASS_FERMI_TWOD {
            self.twod.write(method, data);
            if method == twod::method::SRC_Y0_INT {
                self.stats.blits += 1;
                let blit = self.twod.blit();
                let ran = match self.renderer.blit(memory, &blit) {
                    Ok(ran) => ran,
                    Err(error) => {
                        if self.trace {
                            println!("[recomp-gpu] 2D copy failed: {error}");
                        }
                        false
                    }
                };
                if !ran && !twod::copy_in_memory(memory, &blit) {
                    report_other_class(class, method);
                }
            } else if !twod::TwodState::known(method) {
                report_other_class(class, method);
            }
            return;
        }

        if self.bound[subchannel] == compute::CLASS_KEPLER_COMPUTE {
            self.compute.write(method, data);
            match method {
                compute::method::LAUNCH => self.launch_compute(memory),
                compute::method::REPORT_SEMAPHORE_OPERATION => {
                    let (address, payload, timestamped) = self.compute.report(data);
                    self.stats.semaphores += 1;
                    if timestamped {
                        memory.write_gpu(address, &u64::from(payload).to_le_bytes());
                        memory.write_gpu(address + 8, &memory.gpu_time().to_le_bytes());
                    } else {
                        memory.write_gpu(address, &payload.to_le_bytes());
                    }
                }
                _ if !compute::method::known(method) => {
                    report_other_class(compute::CLASS_KEPLER_COMPUTE, method);
                }
                _ => {}
            }
            return;
        }

        if self.bound[subchannel] != CLASS_MAXWELL_3D {
            self.stats.other_class_writes += 1;
            report_other_class(self.bound[subchannel], method);
            return;
        }

        if method < CLASS_METHOD_LIMIT {
            self.stats.state_writes += 1;
            let offset = method.wrapping_mul(4) as usize;
            let mut bus = MacroBus {
                state: &mut self.state,
                renderer: &mut self.renderer,
                memory,
                draw: &mut self.draw,
                stats: &mut self.stats,
                trace: self.trace,
                // A direct method write is already printed by `trace`.
                trace_macro: false,
            };
            bus.write_reg(offset, data);
        } else {
            // Macro invocation: even starts it, odd pushes an argument. The
            // macro runs when the packet that invoked it is done.
            let index = ((method >> 1) & (MACRO_COUNT - 1) as u32) as usize;
            if self.trace_macro {
                if method & 1 != 0 {
                    println!("[recomp-gfx] macro {index} push {data:#010x}");
                } else {
                    println!("[recomp-gfx] macro {index} start arg {data:#010x}");
                }
            }
            if method & 1 != 0 {
                self.macro_fifo[index].push_back(FifoWord {
                    gpu_va: 0,
                    word: data,
                });
            } else {
                self.macro_pending[index] = Some(data);
            }

            if is_last_call {
                if let Some(arg0) = self.macro_pending[index].take() {
                    self.run_macro(memory, index, arg0);
                }
            }
        }
    }

    fn write_puller(
        &mut self,
        memory: &mut dyn GpuMemory,
        subchannel: usize,
        method: u32,
        data: u32,
    ) {
        match method {
            puller::BIND_OBJECT => self.bound[subchannel] = data & 0xffff,
            puller::SEMAPHORE_ADDRESS_HIGH => self.semaphore[0] = data,
            puller::SEMAPHORE_ADDRESS_LOW => self.semaphore[1] = data,
            puller::SEMAPHORE_SEQUENCE => self.semaphore[2] = data,
            // Operation in bits 2:0 (2 = release); bit 24 asks for the long
            // form, with a GPU timestamp after the payload.
            puller::SEMAPHORE_TRIGGER if data & 0x7 == 2 => {
                let address = u64::from(self.semaphore[0]) << 32 | u64::from(self.semaphore[1]);
                self.stats.semaphores += 1;
                if data & (1 << 24) != 0 {
                    memory.write_gpu(address, &(u64::from(self.semaphore[2])).to_le_bytes());
                    memory.write_gpu(address + 8, &memory.gpu_time().to_le_bytes());
                } else {
                    memory.write_gpu(address, &self.semaphore[2].to_le_bytes());
                }
            }
            // An increment; a wait needs nothing here, as the commands before
            // it have been processed.
            puller::SYNCPOINT_OPERATION if data & 1 != 0 => {
                let syncpoint = (data >> 8) & 0xFFF;
                report_syncpoint(syncpoint, "the puller");
                memory.increment_syncpoint(syncpoint);
            }
            // Macro code RAM.
            puller::LOAD_MME_INSTRUCTION_RAM_POINTER => self.instruction_pointer = data,
            puller::LOAD_MME_INSTRUCTION_RAM => {
                let pointer = self.instruction_pointer as usize;
                if pointer < self.macro_code.len() {
                    self.macro_code[pointer] = data;
                }
                self.instruction_pointer = self.instruction_pointer.wrapping_add(1);
            }
            puller::LOAD_MME_START_ADDRESS_RAM_POINTER => self.start_pointer = data,
            puller::LOAD_MME_START_ADDRESS_RAM => {
                let pointer = self.start_pointer as usize;
                if pointer < MACRO_COUNT {
                    self.macro_position[pointer] = data;
                    self.macro_pending[pointer] = None;
                    self.macro_fifo[pointer].clear();
                }
                self.start_pointer = self.start_pointer.wrapping_add(1);
            }
            _ => {}
        }
    }

    /// Runs macro `index` with `arg0`, writing through the state block.
    fn run_macro(&mut self, memory: &mut dyn GpuMemory, index: usize, arg0: u32) {
        let position = self.macro_position[index] as usize;
        let end = (position + MACRO_WINDOW).min(self.macro_code.len());
        let code: Vec<u32> = self.macro_code[position..end].to_vec();

        let mut interpreter = MacroInterpreter::default();
        std::mem::swap(&mut interpreter.fifo, &mut self.macro_fifo[index]);

        let steps = {
            let mut bus = MacroBus {
                state: &mut self.state,
                renderer: &mut self.renderer,
                memory,
                draw: &mut self.draw,
                stats: &mut self.stats,
                trace: self.trace,
                trace_macro: self.trace_macro,
            };
            interpreter.execute(&code, &mut bus, arg0)
        };

        // Arguments the macro did not fetch stay queued, as on hardware.
        std::mem::swap(&mut interpreter.fifo, &mut self.macro_fifo[index]);

        if self.trace_macro {
            println!("[recomp-gfx] macro {index} ran at {position:#x}: {steps} steps");
        }
        self.stats.macros += 1;
        self.stats.macro_steps += steps as u64;
    }
}

/// State access for one method write or one macro instruction. The class
/// methods that act, act here.
struct MacroBus<'a, R: Renderer> {
    state: &'a mut MaxwellState,
    renderer: &'a mut R,
    memory: &'a mut dyn GpuMemory,
    draw: &'a mut DrawFlags,
    stats: &'a mut ProcessorStats,
    trace: bool,
    /// Print the state methods a macro sends. Set only when the bus is driven
    /// by the interpreter, not by a direct method write.
    trace_macro: bool,
}

impl<R: Renderer> MacroState for MacroBus<'_, R> {
    fn read_reg(&self, offset: usize) -> u32 {
        self.state.read(offset)
    }

    fn write_reg(&mut self, offset: usize, value: u32) {
        if self.trace_macro {
            let method = offset / 4;
            println!("[recomp-gfx] macro send method {method:#05x} = {value:#010x}");
        }
        self.state.write(offset, value);

        match offset {
            m::DRAW_BEGIN => self.draw_begin(value),
            m::DRAW_END => self.draw_end(value),
            m::INDEX_BUFFER_COUNT => self.draw.indexed = true,
            m::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST => self.draw_vertex_array(value, false),
            m::DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT => {
                self.draw_vertex_array(value, true)
            }
            m::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST
            | m::DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST
            | m::DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST => {
                self.draw_index_buffer(value, false)
            }
            m::DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT
            | m::DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT
            | m::DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT => {
                self.draw_index_buffer(value, true)
            }
            m::VB_ELEMENT_U8 => self.push_inline_indices(value, 4),
            m::VB_ELEMENT_U16 => self.push_inline_indices(value, 2),
            m::VB_ELEMENT_U32 => self.push_inline_indices(value, 1),
            m::CLEAR => self.clear(value),
            m::SEMAPHORE_CONTROL => self.report(value),
            // A word to any of the 16 data registers lands at the update
            // offset: NVN's incrementing updates walk CB_DATA0, 1, 2...
            // Taking only the first left the rest of each update unwritten,
            // and draws read what an earlier draw left there (LM3's lobby
            // pendant glint card read leftover words as its color).
            offset
                if (m::UNIFORM_BUFFER_UPDATE_DATA
                    ..m::UNIFORM_BUFFER_UPDATE_DATA + 4 * m::UNIFORM_BUFFER_UPDATE_DATA_WORDS)
                    .contains(&offset) =>
            {
                self.constant_buffer_update(value)
            }
            m::UNIFORM_BUFFER_BIND_VERTEX
            | m::UNIFORM_BUFFER_BIND_TESS_CONTROL
            | m::UNIFORM_BUFFER_BIND_TESS_EVALUATION
            | m::UNIFORM_BUFFER_BIND_GEOMETRY
            | m::UNIFORM_BUFFER_BIND_FRAGMENT => self.bind_constant_buffer(offset, value),
            // Barriers, cache invalidations and counter resets affect the
            // hardware's caches, which this processor does not have.
            m::SYNCPOINT_ACTION => {
                self.stats.semaphores += 1;
                let syncpoint = value & 0xFFFF;
                report_syncpoint(syncpoint, "SYNCPOINT_ACTION");
                self.memory.increment_syncpoint(syncpoint);
            }
            _ => {}
        }
    }
}

impl<R: Renderer> MacroBus<'_, R> {
    fn draw_begin(&mut self, value: u32) {
        // Bit 26 increments the instance, bit 27 clear resets it.
        let increment_instance = value & (1 << 26) != 0;
        let reset_instance = value & (1 << 27) == 0;

        if increment_instance {
            self.draw.instance_index += 1;
        } else if reset_instance {
            self.draw.instance_index = 0;
        }

        self.draw.topology = if self.state.read(m::PRIMITIVE_TYPE_OVERRIDE_ENABLE) != 0 {
            PrimitiveTopology::from_override(self.state.read(m::PRIMITIVE_TYPE_OVERRIDE))
        } else {
            PrimitiveTopology::from_u32(value & 0xffff).unwrap_or(PrimitiveTopology::Triangles)
        };
    }

    /// Finishes a draw: the count comes from the state unless the draw pushed
    /// its parameters inline, which the vertex array methods do.
    fn draw_end(&mut self, _value: u32) {
        let first_index = self.state.index_buffer().first;
        let index_count = self.state.read(m::INDEX_BUFFER_COUNT);
        let draw_state = self.state.vertex_buffer_draw_state();

        let inline_count = self.draw.inline_indices.len() as u32;
        if inline_count != 0 {
            let indices = std::mem::take(&mut self.draw.inline_indices);
            let parameters = DrawParameters {
                topology: self.draw.topology,
                count: inline_count,
                instance_count: 1,
                instance_index: self.draw.instance_index,
                first_index: 0,
                first_vertex: self.state.first_vertex(),
                first_instance: self.state.first_instance(),
                indexed: true,
            };
            self.draw_draw(
                DrawCall::from_state_with_constant_buffers(
                    self.state,
                    parameters,
                    self.draw.constant_buffers,
                )
                .with_inline_indices(indices),
            );
        } else if self.draw.indexed {
            let parameters = DrawParameters {
                topology: self.draw.topology,
                count: index_count,
                instance_count: 1,
                instance_index: self.draw.instance_index,
                first_index,
                first_vertex: self.state.first_vertex(),
                first_instance: self.state.first_instance(),
                indexed: true,
            };
            self.draw_draw(DrawCall::from_state_with_constant_buffers(
                self.state,
                parameters,
                self.draw.constant_buffers,
            ));
        } else {
            let parameters = DrawParameters {
                topology: self.draw.topology,
                count: draw_state.count,
                instance_count: 1,
                instance_index: self.draw.instance_index,
                first_index: 0,
                first_vertex: draw_state.first,
                first_instance: self.state.first_instance(),
                indexed: false,
            };
            self.draw_draw(DrawCall::from_state_with_constant_buffers(
                self.state,
                parameters,
                self.draw.constant_buffers,
            ));
        }

        self.draw.indexed = false;
    }

    /// A non-indexed draw with the topology, first vertex and count packed
    /// into one argument.
    fn draw_vertex_array(&mut self, value: u32, instanced: bool) {
        self.draw_begin((value >> 28) & 0xf);
        if instanced {
            self.draw.instance_index += 1;
        }

        let parameters = DrawParameters {
            topology: self.draw.topology,
            count: (value >> 16) & 0xfff,
            instance_count: 1,
            instance_index: self.draw.instance_index,
            first_index: 0,
            first_vertex: value & 0xffff,
            first_instance: self.state.first_instance(),
            indexed: false,
        };
        self.draw_draw(DrawCall::from_state_with_constant_buffers(
            self.state,
            parameters,
            self.draw.constant_buffers,
        ));
    }

    /// An indexed draw with the topology, first index and count packed into
    /// one argument. The index type comes from the pushed element method.
    fn draw_index_buffer(&mut self, value: u32, instanced: bool) {
        self.draw_begin((value >> 28) & 0xf);
        if instanced {
            self.draw.instance_index += 1;
        }

        let index = self.state.index_buffer();
        let parameters = DrawParameters {
            topology: self.draw.topology,
            count: (value >> 16) & 0xfff,
            instance_count: 1,
            instance_index: self.draw.instance_index,
            first_index: value & 0xffff,
            first_vertex: self.state.first_vertex(),
            first_instance: self.state.first_instance(),
            indexed: true,
        };
        let mut draw = DrawCall::from_state_with_constant_buffers(
            self.state,
            parameters,
            self.draw.constant_buffers,
        );
        if let Some(index_buffer) = draw.index_buffer.as_mut() {
            index_buffer.index_type =
                IndexType::from_u32(index.index_type).unwrap_or(IndexType::U32);
        }
        self.draw_draw(draw);
    }

    fn push_inline_indices(&mut self, value: u32, elements: usize) {
        // The elements are packed little-endian into the word.
        let bytes = value.to_le_bytes();
        let width = 4 / elements;
        for chunk in bytes.chunks(width) {
            let index = match width {
                1 => u32::from(chunk[0]),
                2 => u32::from(u16::from_le_bytes([chunk[0], chunk[1]])),
                _ => u32::from_le_bytes(chunk.try_into().expect("four byte chunk")),
            };
            self.draw.inline_indices.push(index);
        }
    }

    /// A bind method: bit 0 binds (or unbinds) the slot in bits 4:8 of the
    /// stage the method belongs to.
    fn bind_constant_buffer(&mut self, offset: usize, value: u32) {
        let stage = (offset - m::UNIFORM_BUFFER_BIND_VERTEX) / 0x20;
        let slot = ((value >> 4) & 0x1f) as usize;
        if slot >= CONSTANT_BUFFER_SLOTS {
            return;
        }
        let buffer = self.state.uniform_buffer();
        self.draw.constant_buffers[stage][slot] =
            (value & 1 != 0).then_some(ConstantBufferBinding {
                address: buffer.address,
                size: buffer.size,
            });
    }

    fn draw_draw(&mut self, draw: DrawCall) {
        self.stats.draws += 1;
        if draw.parameters.indexed {
            self.stats.indexed_draws += 1;
        }
        if let Err(error) = self.renderer.draw(self.memory, &draw) {
            if self.trace {
                println!("[recomp-gpu] draw failed: {error}");
            }
        }
    }

    /// Clears the color and depth-stencil targets. Which buffers to clear is
    /// in the argument: bit 0 depth, bit 1 stencil, bits 5:2 the component
    /// mask, bits 9:6 the color target, bits 19:10 the layer.
    fn clear(&mut self, value: u32) {
        let clear_depth = value & 1 != 0;
        let clear_stencil = value & 2 != 0;
        let component_mask = (value >> 2) & 0xf;
        let index = ((value >> 6) & 0xf) as usize;
        let layer = (value >> 10) & 0x3ff;

        if component_mask != 0 {
            let target = self.state.color_target(index);
            self.stats.color_clears += 1;
            if let Err(error) = self.renderer.clear_color(
                self.memory,
                &ClearColorCall {
                    index,
                    layer,
                    layer_count: target.depth,
                    component_mask,
                    color: self.state.clear_color(),
                    target,
                },
            ) {
                if self.trace {
                    println!("[recomp-gpu] color clear failed: {error}");
                }
            }
        }

        if clear_depth || clear_stencil {
            let size = self.state.depth_target_size();
            self.stats.depth_clears += 1;
            if let Err(error) = self.renderer.clear_depth_stencil(
                self.memory,
                &ClearDepthStencilCall {
                    clear_depth,
                    clear_stencil,
                    depth: self.state.clear_depth(),
                    stencil: self.state.clear_stencil(),
                    target: self.state.depth_target(),
                    width: size.width,
                    height: size.height,
                    layer_count: size.depth,
                },
            ) {
                if self.trace {
                    println!("[recomp-gpu] depth clear failed: {error}");
                }
            }
        }
    }

    /// A GPU report: release a semaphore, or write a counter with a timestamp.
    fn report(&mut self, value: u32) {
        let address = self.state.read_va(m::SEMAPHORE_ADDRESS);
        let payload = self.state.read(m::SEMAPHORE_PAYLOAD);
        self.stats.semaphores += 1;

        match value & 3 {
            0 => {
                self.memory.write_gpu(address, &payload.to_le_bytes());
            }
            1 => {
                self.memory
                    .write_gpu(address, &(u64::from(payload)).to_le_bytes());
                self.memory
                    .write_gpu(address + 8, &self.memory.gpu_time().to_le_bytes());
            }
            // A counter report (queries): the counter in bits 27:23, one word
            // (bit 28) or the 64-bit counter and a timestamp.
            2 => {
                let counter = match (value >> 23) & 0x1F {
                    0 => u64::from(payload),
                    // Pixels that passed the depth test (occlusion queries):
                    // not counted here, so counted as growing with every
                    // report, which makes every end-minus-begin visible.
                    0x02 | 0x15 => self.stats.semaphores << 12,
                    _ => 0,
                };
                if value & (1 << 28) != 0 {
                    self.memory
                        .write_gpu(address, &(counter as u32).to_le_bytes());
                } else {
                    self.memory.write_gpu(address, &counter.to_le_bytes());
                    self.memory
                        .write_gpu(address + 8, &self.memory.gpu_time().to_le_bytes());
                }
            }
            _ => {}
        }
    }

    /// Inline constant-buffer data: the word lands at the bound buffer's
    /// current offset, which then advances.
    fn constant_buffer_update(&mut self, value: u32) {
        let buffer = self.state.uniform_buffer();
        self.stats.inline_copy_words += 1;
        if buffer.address != 0 {
            self.memory.write_gpu(
                buffer.address + u64::from(buffer.offset),
                &value.to_le_bytes(),
            );
        }
        self.state
            .write(m::UNIFORM_BUFFER_STATE + 0x0c, buffer.offset + 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mme::{add_imm, ASG_MOVE_AND_SEND, ASG_MOVE_AND_SET_MADDR};
    use render::DrawRecorder;

    /// A block of guest GPU memory at address zero.
    struct TestMemory {
        bytes: Vec<u8>,
        writes: Vec<(u64, Vec<u8>)>,
    }

    impl TestMemory {
        fn new(size: usize) -> Self {
            Self {
                bytes: vec![0; size],
                writes: Vec::new(),
            }
        }
    }

    impl GpuMemory for TestMemory {
        fn read_gpu(&self, address: u64, len: usize) -> Option<Vec<u8>> {
            self.bytes
                .get(address as usize..address as usize + len)
                .map(|bytes| bytes.to_vec())
        }

        fn write_gpu(&mut self, address: u64, bytes: &[u8]) -> bool {
            let start = address as usize;
            let Some(slice) = self.bytes.get_mut(start..start + bytes.len()) else {
                return false;
            };
            slice.copy_from_slice(bytes);
            self.writes.push((address, bytes.to_vec()));
            true
        }
    }

    /// A method number, from a byte offset in the state block.
    fn word(offset: usize) -> u32 {
        (offset / 4) as u32
    }

    /// A command header: method, subchannel, count and submission mode.
    fn header(method: u32, subchannel: usize, count: usize, mode: u32) -> u32 {
        method | ((subchannel as u32) << 13) | ((count as u32) << 16) | (mode << 29)
    }

    /// The words that load `code` into macro RAM index 0.
    fn load_macro(code: &[u32]) -> Vec<u32> {
        let mut words = vec![
            header(word(m::LOAD_MME_INSTRUCTION_RAM_POINTER), 0, 1, 0),
            0,
            // Non-increasing: every word goes to the same method, and the
            // pointer inside the puller advances.
            header(word(m::LOAD_MME_INSTRUCTION_RAM), 0, code.len(), 2),
        ];
        words.extend_from_slice(code);
        words.push(header(word(m::LOAD_MME_START_ADDRESS_RAM_POINTER), 0, 1, 0));
        words.push(0);
        words.push(header(word(m::LOAD_MME_START_ADDRESS_RAM), 0, 1, 0));
        words.push(0);
        words
    }

    fn bound_3d() -> Vec<u32> {
        vec![header(puller::BIND_OBJECT, 0, 1, 0), CLASS_MAXWELL_3D]
    }

    #[test]
    fn a_macro_draws_through_the_state() {
        // The macro LM3 uses for a plain draw: send the topology to
        // DrawBegin, the vertex range to the draw state, then DrawEnd.
        let code = [
            add_imm(
                ASG_MOVE_AND_SET_MADDR,
                0,
                0,
                word(m::DRAW_BEGIN) as i32,
                false,
            ),
            add_imm(ASG_MOVE_AND_SEND, 0, 1, 0, false),
            add_imm(
                mme::ASG_MOVE_AND_SET_MADDR_THEN_FETCH_AND_SEND,
                0,
                0,
                word(m::VERTEX_BUFFER_DRAW_STATE) as i32,
                false,
            ),
            add_imm(
                mme::ASG_MOVE_AND_SET_MADDR_THEN_FETCH_AND_SEND,
                0,
                0,
                (word(m::VERTEX_BUFFER_DRAW_STATE) + 1) as i32,
                false,
            ),
            add_imm(
                ASG_MOVE_AND_SET_MADDR,
                0,
                0,
                word(m::DRAW_END) as i32,
                false,
            ),
            add_imm(ASG_MOVE_AND_SEND, 0, 0, 0, true),
        ];

        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);

        let mut list = bound_3d();
        list.extend(load_macro(&code));
        // Macro 0 is method 0xE00; the arguments follow as odd methods, in one
        // increase-once packet: the macro runs after the last word.
        list.push(header(0xe00, 0, 3, 5));
        list.extend([4, 7, 12]);
        processor.submit_list(&mut memory, &list);

        assert_eq!(processor.stats.macros, 1);
        assert_eq!(processor.stats.draws, 1);

        let draw = &processor.renderer.draws[0];
        assert_eq!(draw.parameters.topology, PrimitiveTopology::Triangles);
        assert_eq!(draw.parameters.first_vertex, 7);
        assert_eq!(draw.parameters.count, 12);
        assert!(!draw.parameters.indexed);
        assert_eq!(processor.state.vertex_buffer_draw_state().count, 12);
    }

    #[test]
    fn a_direct_draw_end_uses_the_state_it_follows() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);

        let mut list = bound_3d();
        list.extend([
            header(word(m::VERTEX_BUFFER_DRAW_STATE), 0, 2, 0),
            3,
            9,
            header(word(m::DRAW_BEGIN), 0, 1, 0),
            5,
            header(word(m::DRAW_END), 0, 1, 0),
            0,
        ]);
        processor.submit_list(&mut memory, &list);

        let draw = &processor.renderer.draws[0];
        assert_eq!(draw.parameters.topology, PrimitiveTopology::TriangleStrip);
        assert_eq!(draw.parameters.first_vertex, 3);
        assert_eq!(draw.parameters.count, 9);
    }

    #[test]
    fn bind_methods_give_draws_their_constant_buffers() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);

        // Bind 0x100 bytes at 0x1_0000_2000 to fragment slot 3, then 0x40
        // bytes at 0x3000 to vertex slot 0: the draw sees both.
        let mut list = bound_3d();
        list.extend([
            header(word(m::UNIFORM_BUFFER_STATE), 0, 3, 1),
            0x100,
            0x1,
            0x2000,
            header(word(m::UNIFORM_BUFFER_BIND_FRAGMENT), 0, 1, 0),
            (3 << 4) | 1,
            header(word(m::UNIFORM_BUFFER_STATE), 0, 3, 1),
            0x40,
            0x0,
            0x3000,
            header(word(m::UNIFORM_BUFFER_BIND_VERTEX), 0, 1, 0),
            1,
            header(word(m::VERTEX_BUFFER_DRAW_STATE), 0, 2, 0),
            0,
            3,
            header(word(m::DRAW_BEGIN), 0, 1, 0),
            4,
            header(word(m::DRAW_END), 0, 1, 0),
            0,
        ]);
        processor.submit_list(&mut memory, &list);

        let draw = &processor.renderer.draws[0];
        assert_eq!(
            draw.constant_buffer(5, 3),
            Some(ConstantBufferBinding {
                address: 0x1_0000_2000,
                size: 0x100
            })
        );
        assert_eq!(
            draw.constant_buffer(1, 0),
            Some(ConstantBufferBinding {
                address: 0x3000,
                size: 0x40
            })
        );
        assert_eq!(draw.constant_buffer(5, 0), None);
    }

    #[test]
    fn the_index_buffer_count_method_makes_the_draw_indexed() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);

        processor.state.write(m::INDEX_BUFFER_STATE + 4, 0x2000);
        processor.state.write(m::INDEX_BUFFER_STATE + 0x14, 4);

        let mut list = bound_3d();
        list.extend([
            header(word(m::DRAW_BEGIN), 0, 1, 0),
            4,
            header(word(m::INDEX_BUFFER_COUNT), 0, 1, 0),
            6,
            header(word(m::DRAW_END), 0, 1, 0),
            0,
        ]);
        processor.submit_list(&mut memory, &list);

        let draw = &processor.renderer.draws[0];
        assert!(draw.parameters.indexed);
        assert_eq!(draw.parameters.count, 6);
        assert_eq!(draw.parameters.first_index, 4);
        assert_eq!(draw.index_buffer.map(|buffer| buffer.address), Some(0x2000));
    }

    #[test]
    fn inline_index_elements_become_an_indexed_draw() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);

        let mut list = bound_3d();
        list.extend([
            header(word(m::DRAW_BEGIN), 0, 1, 0),
            4,
            header(word(m::VB_ELEMENT_U8), 0, 1, 0),
            0x1234_5678,
            header(word(m::DRAW_END), 0, 1, 0),
            0,
        ]);
        processor.submit_list(&mut memory, &list);

        let draw = &processor.renderer.draws[0];
        assert!(draw.parameters.indexed);
        assert_eq!(draw.parameters.count, 4);
        // The elements are packed least significant byte first.
        assert_eq!(draw.inline_indices, vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn a_clear_method_reaches_the_renderer() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);

        processor.state.write(m::RT_COLOR_STATE + 4, 0x1000);
        processor.state.write(m::RT_COLOR_STATE + 0x08, 1280);
        processor.state.write(m::RT_COLOR_STATE + 0x0c, 720);
        processor.state.write(m::RT_COLOR_STATE + 0x10, 1);
        processor.state.write(m::RT_CONTROL, 1);
        processor.state.write(m::CLEAR_COLORS, 1.0f32.to_bits());

        // Depth and stencil in bits 1:0, the component mask in bits 5:2, the
        // color target in bits 9:6. Target zero is left as the zero field.
        let clear_color = 0xf << 2;
        let clear_depth = 1;

        let mut list = bound_3d();
        list.extend([
            header(word(m::CLEAR), 0, 1, 0),
            clear_color,
            header(word(m::CLEAR), 0, 1, 0),
            clear_depth,
        ]);
        processor.submit_list(&mut memory, &list);

        let clear = &processor.renderer.color_clears[0];
        assert_eq!(clear.component_mask, 0xf);
        assert_eq!(clear.index, 0);
        assert_eq!(clear.target.address, 0x1000);
        assert_eq!(clear.color[0], 1.0);
        assert!(processor.renderer.depth_clears[0].clear_depth);
        assert_eq!(processor.stats.color_clears, 1);
        assert_eq!(processor.stats.depth_clears, 1);
    }

    #[test]
    fn a_puller_semaphore_release_writes_guest_memory() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);

        let list = vec![
            header(puller::SEMAPHORE_ADDRESS_HIGH, 0, 1, 0),
            0,
            header(puller::SEMAPHORE_ADDRESS_LOW, 0, 1, 0),
            0x100,
            header(puller::SEMAPHORE_SEQUENCE, 0, 1, 0),
            0x2a,
            // Operation 2 is a release; bit 24 asks for the long form.
            header(puller::SEMAPHORE_TRIGGER, 0, 1, 0),
            2 | (1 << 24),
        ];
        processor.submit_list(&mut memory, &list);

        assert_eq!(memory.writes[0], (0x100, vec![0x2a, 0, 0, 0, 0, 0, 0, 0]));
        assert_eq!(memory.writes.len(), 2);
        assert_eq!(processor.stats.semaphores, 1);
    }

    #[test]
    fn a_3d_report_semaphore_reaches_guest_memory() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x4000);

        processor.state.write_va(m::SEMAPHORE_ADDRESS, 0x2000);
        processor.state.write(m::SEMAPHORE_PAYLOAD, 0x55);

        let mut list = bound_3d();
        list.extend([header(word(m::SEMAPHORE_CONTROL), 0, 1, 0), 0]);
        processor.submit_list(&mut memory, &list);

        assert_eq!(memory.writes, vec![(0x2000, vec![0x55, 0, 0, 0])]);
    }

    #[test]
    fn inline_constant_buffer_data_advances_the_offset() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);

        // 0x100 bytes at 0x800: size, address high, address low.
        processor.state.write(m::UNIFORM_BUFFER_STATE, 0x100);
        processor.state.write(m::UNIFORM_BUFFER_STATE + 8, 0x800);

        let mut list = bound_3d();
        list.extend([
            // Non-increasing: every word updates the buffer at the offset the
            // previous word left, rather than at the next method.
            header(word(m::UNIFORM_BUFFER_UPDATE_DATA), 0, 2, 2),
            0xaa,
            0xbb,
        ]);
        processor.submit_list(&mut memory, &list);

        assert_eq!(
            memory.read_gpu(0x800, 8),
            Some(vec![0xaa, 0, 0, 0, 0xbb, 0, 0, 0])
        );
        assert_eq!(processor.state.uniform_buffer().offset, 8);
        assert_eq!(processor.stats.inline_copy_words, 2);
    }

    #[test]
    fn inline_constant_buffer_data_fills_every_data_register() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);

        processor.state.write(m::UNIFORM_BUFFER_STATE, 0x100);
        processor.state.write(m::UNIFORM_BUFFER_STATE + 8, 0x800);
        processor.state.write(m::UNIFORM_BUFFER_STATE + 0x0c, 0x10);

        let mut list = bound_3d();
        // Increasing: the words go to CB_DATA0, CB_DATA1 and CB_DATA2, and
        // all three land one after another at the update offset.
        list.extend([
            header(word(m::UNIFORM_BUFFER_UPDATE_DATA), 0, 3, 1),
            0xaa,
            0xbb,
            0xcc,
        ]);
        processor.submit_list(&mut memory, &list);

        assert_eq!(
            memory.read_gpu(0x810, 12),
            Some(vec![0xaa, 0, 0, 0, 0xbb, 0, 0, 0, 0xcc, 0, 0, 0])
        );
        assert_eq!(processor.state.uniform_buffer().offset, 0x1c);
        assert_eq!(processor.stats.inline_copy_words, 3);
    }

    #[test]
    fn method_data_continues_in_the_next_list() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);
        // An inline copy's header and first data word in one list, the
        // second word in the next (as an indirect buffer would be).
        let first = vec![
            header(puller::BIND_OBJECT, 1, 1, 0),
            compute::CLASS_KEPLER_COMPUTE,
            header(inline::LINE_LENGTH_IN, 1, 4, 1),
            8,
            1,
            0,
            0x300,
            header(inline::LAUNCH_DMA, 1, 1, 0),
            1,
            header(inline::LOAD_INLINE_DATA, 1, 2, 3),
            0xAAAA_0001,
        ];
        processor.submit_list(&mut memory, &first);
        assert_eq!(memory.read_u32(0x300), Some(0));
        processor.submit_list(&mut memory, &[0xBBBB_0002]);
        assert_eq!(memory.read_u32(0x300), Some(0xAAAA_0001));
        assert_eq!(memory.read_u32(0x304), Some(0xBBBB_0002));
    }

    #[test]
    fn inline_copies_through_the_compute_class_reach_memory() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);
        let list = vec![
            header(puller::BIND_OBJECT, 1, 1, 0),
            compute::CLASS_KEPLER_COMPUTE,
            // 8 bytes in one line to 0x200, then the data.
            header(inline::LINE_LENGTH_IN, 1, 4, 1),
            8,
            1,
            0,
            0x200,
            header(inline::LAUNCH_DMA, 1, 1, 0),
            1,
            header(inline::LOAD_INLINE_DATA, 1, 2, 3),
            0x1122_3344,
            0x5566_7788,
        ];
        processor.submit_list(&mut memory, &list);
        assert_eq!(memory.read_u32(0x200), Some(0x1122_3344));
        assert_eq!(memory.read_u32(0x204), Some(0x5566_7788));
        assert_eq!(processor.stats.inline_copy_words, 2);
    }

    #[test]
    fn writes_to_another_class_are_not_the_3d_state() {
        let mut processor = CommandProcessor::new(DrawRecorder::default());
        let mut memory = TestMemory::new(0x1000);

        let list = vec![
            header(puller::BIND_OBJECT, 1, 1, 0),
            // The DMA copy engine.
            0xB0B5,
            header(word(m::DRAW_END), 1, 1, 0),
            0,
        ];
        processor.submit_list(&mut memory, &list);

        assert_eq!(processor.stats.other_class_writes, 1);
        assert_eq!(processor.stats.state_writes, 0);
        assert_eq!(processor.renderer.draws.len(), 0);
    }
}
