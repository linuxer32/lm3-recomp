//! host1x: the console's multimedia engines, behind `/dev/nvhost-nvdec`,
//! `/dev/nvhost-vic` and `/dev/nvhost-nvjpg`. Titles play movies with them:
//! NVDEC decodes a frame into a 4:2:0 surface, VIC converts it to the RGBA
//! surface the title draws as a texture. LM3's title card after the King Boo
//! chase is such a movie (WebM, VP9).
//!
//! A submit hands over command buffers: host1x opcodes that pick an engine
//! (class) and write its registers. The engines' registers are reached
//! through their host interface (THI): `METHOD0` names an engine method and
//! `METHOD1` writes it. Writing `EXECUTE` runs the engine. Buffers are
//! addressed by the engines through their own 32-bit address space, mapped
//! per memory handle (`MapCommandBuffer`); their registers hold those
//! addresses shifted right by 8.
//!
//! Submits run at once, on the calling thread: decoding a 1600x900 VP9 frame
//! with libvpx and converting it take a few milliseconds. Decoded frames are
//! kept on the host, by output surface, for VIC to read (NVDEC's own
//! block-linear output is not written to guest memory). VIC's output is
//! written to guest memory, and its range is recorded
//! (`recomp_gfx::engine_writes`) so the renderer reads textures there again.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use recomp_gfx::maxwell::blocklinear::Volume;

use crate::gpu;
use crate::ipc::MemoryAccess;
use crate::vp9;

/// Engine classes.
const CLASS_HOST1X: u32 = 0x01;
const CLASS_VIC: u32 = 0x5D;
const CLASS_NVJPG: u32 = 0xC0;
const CLASS_NVDEC: u32 = 0xF0;

/// Host interface registers.
const THI_INCR_SYNCPT: u32 = 0x00;
const THI_METHOD0: u32 = 0x10;
const THI_METHOD1: u32 = 0x11;

/// host1x class registers.
const HOST1X_INCR_SYNCPT: u32 = 0x00;
const HOST1X_LOAD_SYNCPT_PAYLOAD32: u32 = 0x4E;
const HOST1X_WAIT_SYNCPT32: u32 = 0x50;

/// Engine methods (word indices; Ryujinx's `NvdecRegisters` and
/// `VicRegisters`).
const EXECUTE: usize = 0xC0;
const NVDEC_APPLICATION_ID: usize = 0x80;
const NVDEC_PICTURE_SETUP: usize = 0x101;
const NVDEC_BITSTREAM: usize = 0x102;
/// `SetPictureLumaOffset[17]`: references, then the output.
const NVDEC_LUMA: usize = 0x10C;
const NVDEC_VP9_PROBS: usize = 0x170;
/// `SetSurfacexSlotx[8][8]`: per slot, eight surfaces of three planes.
const VIC_SURFACES: usize = 0x100;
const VIC_CONFIG: usize = 0x1C2;
const VIC_OUTPUT: usize = 0x1C8;
const REGISTERS: usize = 0x200;

const APPLICATION_VP9: u32 = 9;

/// VIC pixel formats.
const A8B8G8R8: u64 = 31;
const A8R8G8B8: u64 = 32;
const X8B8G8R8: u64 = 35;
const NV12: u64 = 68;

/// VIC's configuration: pipe, output, output surface, output matrix, four
/// clear rectangles (144 bytes), then eight slots of 176.
const VIC_SLOT_START: usize = 144;
const VIC_SLOT_BYTES: usize = 176;
const VIC_CONFIG_BYTES: usize = VIC_SLOT_START + 8 * VIC_SLOT_BYTES;

/// Engine addresses handed out to mapped buffers start here.
const ENGINE_ADDRESS_START: u32 = 0x10_0000;
/// Frames kept for VIC at most (a decoder's surface pool).
const FRAMES_KEPT: usize = 32;

/// Decoded frames and engine state shared by every channel.
struct Host1x {
    /// Engine address -> (CPU address, size) of each buffer mapped.
    mappings: BTreeMap<u32, (u64, u32)>,
    /// Per memory handle, its CPU address and engine address.
    handles: HashMap<u32, (u64, u32)>,
    next: u32,
    nvdec: Vec<u32>,
    vic: Vec<u32>,
    /// Per engine class, the method `METHOD1` writes.
    method: HashMap<u32, u32>,
    /// `LOAD_SYNCPT_PAYLOAD_32`, which a syncpoint wait waits for.
    payload: u32,
    vp9: vp9::Decoder,
    /// Decoded frames by the register value naming their output surface.
    frames: HashMap<u32, vp9::Frame>,
    warned: HashSet<String>,
    decoded: u64,
    converted: u64,
}

impl Host1x {
    fn new() -> Self {
        Self {
            mappings: BTreeMap::new(),
            handles: HashMap::new(),
            next: ENGINE_ADDRESS_START,
            nvdec: vec![0; REGISTERS],
            vic: vec![0; REGISTERS],
            method: HashMap::new(),
            payload: 0,
            vp9: vp9::Decoder::default(),
            frames: HashMap::new(),
            warned: HashSet::new(),
            decoded: 0,
            converted: 0,
        }
    }

    fn warn(&mut self, message: String) {
        if self.warned.insert(message.clone()) {
            println!("[recomp-host1x] {message}");
        }
    }

    /// The CPU address of engine address `address`.
    fn cpu_address(&self, address: u64) -> Option<u64> {
        let address = u32::try_from(address).ok()?;
        let (start, (cpu, size)) = self.mappings.range(..=address).next_back()?;
        (address - start < *size).then(|| cpu + u64::from(address - start))
    }

    /// `len` bytes at the engine address a register (shifted by 8) names.
    fn read(&self, mem: &dyn MemoryAccess, register: u32, len: usize) -> Option<Vec<u8>> {
        mem.read_bytes(self.cpu_address(u64::from(register) << 8)?, len)
    }

    fn step(&mut self, mem: &dyn MemoryAccess, stream: &mut Stream, word: u32) {
        if stream.mask != 0 {
            let bit = stream.mask.trailing_zeros();
            stream.mask &= !(1 << bit);
            self.write(mem, stream.class, stream.offset + bit, word);
            return;
        }
        if stream.count != 0 {
            stream.count -= 1;
            self.write(mem, stream.class, stream.offset, word);
            if stream.incrementing {
                stream.offset += 1;
            }
            return;
        }
        let offset = (word >> 16) & 0xFFF;
        match word >> 28 {
            // SetClass
            0 => {
                stream.class = (word >> 6) & 0x3FF;
                stream.offset = offset;
                stream.mask = word & 0x3F;
            }
            // Incr, NonIncr
            opcode @ (1 | 2) => {
                stream.offset = offset;
                stream.count = word & 0xFFFF;
                stream.incrementing = opcode == 1;
            }
            // Mask
            3 => {
                stream.offset = offset;
                stream.mask = word & 0xFFFF;
            }
            // Imm
            4 => self.write(mem, stream.class, offset, word & 0xFFFF),
            opcode => self.warn(format!("host1x opcode {opcode} not run yet")),
        }
    }

    fn write(&mut self, mem: &dyn MemoryAccess, class: u32, offset: u32, data: u32) {
        match (class, offset) {
            (CLASS_HOST1X, HOST1X_INCR_SYNCPT) => gpu::signal_syncpoint(data & 0xFF),
            (CLASS_HOST1X, HOST1X_LOAD_SYNCPT_PAYLOAD32) => self.payload = data,
            (CLASS_HOST1X, HOST1X_WAIT_SYNCPT32) => {
                // Engines here never wait on the GPU for long; a bounded
                // wait keeps a wrong threshold from hanging the title.
                gpu::wait_for_syncpoint(data & 0xFF, self.payload, 50_000_000);
            }
            (CLASS_HOST1X, _) => {}
            (_, THI_INCR_SYNCPT) => gpu::signal_syncpoint(data & 0xFF),
            (_, THI_METHOD0) => {
                self.method.insert(class, data);
            }
            (_, THI_METHOD1) => {
                let method = self.method.get(&class).copied().unwrap_or(0) as usize;
                self.engine_write(mem, class, method, data);
            }
            _ => {}
        }
    }

    fn engine_write(&mut self, mem: &dyn MemoryAccess, class: u32, method: usize, data: u32) {
        let registers = match class {
            CLASS_NVDEC => &mut self.nvdec,
            CLASS_VIC => &mut self.vic,
            CLASS_NVJPG => {
                self.warn("NVJPG not run yet".to_string());
                return;
            }
            _ => {
                self.warn(format!("host1x class {class:#x} not run yet"));
                return;
            }
        };
        if let Some(register) = registers.get_mut(method) {
            *register = data;
        }
        if method == EXECUTE {
            match class {
                CLASS_NVDEC => self.nvdec_execute(mem),
                _ => self.vic_execute(mem),
            }
        }
    }

    fn nvdec_execute(&mut self, mem: &dyn MemoryAccess) {
        let application = self.nvdec[NVDEC_APPLICATION_ID];
        if application != APPLICATION_VP9 {
            self.warn(format!("NVDEC codec {application} not decoded yet"));
            return;
        }
        let Some(picture) = self
            .read(mem, self.nvdec[NVDEC_PICTURE_SETUP], vp9::PICTURE_BYTES)
            .and_then(|bytes| vp9::Picture::parse(&bytes))
        else {
            self.warn("NVDEC picture setup not readable".to_string());
            return;
        };
        let Some(probs) = self
            .read(mem, self.nvdec[NVDEC_VP9_PROBS], vp9::PROBS_BYTES)
            .and_then(|bytes| vp9::Probs::parse(&bytes))
        else {
            self.warn("NVDEC VP9 probabilities not readable".to_string());
            return;
        };
        let Some(bitstream) = self.read(
            mem,
            self.nvdec[NVDEC_BITSTREAM],
            picture.bitstream_size as usize,
        ) else {
            self.warn("NVDEC bitstream not readable".to_string());
            return;
        };
        let surfaces = [
            self.nvdec[NVDEC_LUMA],
            self.nvdec[NVDEC_LUMA + 1],
            self.nvdec[NVDEC_LUMA + 2],
            self.nvdec[NVDEC_LUMA + 3],
        ];
        if tracing() {
            println!(
                "[recomp-host1x] NVDEC VP9 {}x{} flags {:#x} {} bytes, surfaces {surfaces:x?}",
                picture.sizes[3].0, picture.sizes[3].1, picture.flags, picture.bitstream_size
            );
        }
        let request = vp9::Request {
            picture,
            probs,
            bitstream,
            surfaces,
        };
        let decoded = self.vp9.queue(request);
        self.keep(decoded);
    }

    /// Keep a decoded frame for VIC, by its output surface.
    fn keep(&mut self, decoded: Result<Option<(u32, vp9::Frame)>, String>) {
        match decoded {
            Ok(Some((surface, frame))) => {
                if self.frames.len() >= FRAMES_KEPT {
                    self.frames.clear();
                }
                self.frames.insert(surface, frame);
                self.decoded += 1;
                if self.decoded == 1 {
                    println!("[recomp-host1x] first video frame decoded (VP9, libvpx)");
                }
            }
            Ok(None) => {}
            Err(reason) => self.warn(format!("VP9 frames not decoded: {reason}")),
        }
    }

    fn vic_execute(&mut self, mem: &dyn MemoryAccess) {
        let Some(config) = self.read(mem, self.vic[VIC_CONFIG], VIC_CONFIG_BYTES) else {
            self.warn("VIC configuration not readable".to_string());
            return;
        };
        let bits = |bit: usize, len: usize| config_bits(&config, bit, len);
        // Output surface configuration (byte 32) and target rectangle
        // (output configuration, byte 16).
        let format = bits(32 * 8, 7);
        let block_linear = bits(32 * 8 + 11, 4) != 0;
        let gobs_in_y = 1u64 << bits(32 * 8 + 15, 4);
        let width = bits(32 * 8 + 32, 14) as usize + 1;
        let height = bits(32 * 8 + 46, 14) as usize + 1;
        let (left, right) = (
            bits(16 * 8 + 64, 14) as usize,
            bits(16 * 8 + 80, 14) as usize + 1,
        );
        let (top, bottom) = (
            bits(16 * 8 + 96, 14) as usize,
            bits(16 * 8 + 112, 14) as usize + 1,
        );
        let (target_x, target_y) = (left.min(right), top.min(bottom));
        let target_width = left.abs_diff(right).min(width.saturating_sub(target_x));
        let target_height = top.abs_diff(bottom).min(height.saturating_sub(target_y));

        let mut pixels = vec![0u8; width * height * 4];
        for slot in 0..8 {
            let base = (VIC_SLOT_START + slot * VIC_SLOT_BYTES) * 8;
            if bits(base, 1) == 0 {
                continue;
            }
            let surface = base + 64 * 8;
            let input_format = bits(surface, 7);
            if input_format != NV12 {
                self.warn(format!("VIC input format {input_format} not converted yet"));
                continue;
            }
            let luma = self.vic[VIC_SURFACES + slot * 24];
            if !self.frames.contains_key(&luma) {
                // The frame still waiting for its successor.
                let decoded = self.vp9.flush(luma);
                self.keep(decoded);
            }
            let Some(frame) = self.frames.get(&luma) else {
                self.warn(format!(
                    "VIC input surface {luma:#x} was not decoded here (shown black)"
                ));
                continue;
            };
            let clamp = (bits(base + 128, 10) as i64, bits(base + 138, 10) as i64);
            let matrix = Matrix::parse(&config, base + 96 * 8);
            convert(
                frame,
                &matrix,
                clamp,
                (target_x, target_y, target_width, target_height),
                width,
                &mut pixels,
            );
        }

        // Into the output surface's layout and byte order.
        if format == A8R8G8B8 {
            for pixel in pixels.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        } else if format != A8B8G8R8 && format != X8B8G8R8 {
            self.warn(format!("VIC output format {format} not written yet"));
            return;
        }
        let row_bytes = width * 4;
        let bytes = if block_linear {
            let volume = Volume::new(row_bytes as u64, height as u64, gobs_in_y, 1);
            let mut tiled = vec![0u8; volume.size(1) as usize];
            for y in 0..height {
                let row = &pixels[y * row_bytes..(y + 1) * row_bytes];
                for (chunk, x) in row.chunks(16).zip((0..row_bytes).step_by(16)) {
                    let at = volume.offset(x as u64, y as u64, 0) as usize;
                    if let Some(target) = tiled.get_mut(at..at + chunk.len()) {
                        target.copy_from_slice(chunk);
                    }
                }
            }
            tiled
        } else {
            let pitch = row_bytes.next_multiple_of(256);
            let mut linear = vec![0u8; pitch * height];
            for y in 0..height {
                linear[y * pitch..y * pitch + row_bytes]
                    .copy_from_slice(&pixels[y * row_bytes..(y + 1) * row_bytes]);
            }
            linear
        };
        let Some(cpu) = self.cpu_address(u64::from(self.vic[VIC_OUTPUT]) << 8) else {
            self.warn("VIC output surface not mapped".to_string());
            return;
        };
        if !mem.write_bytes(cpu, &bytes) {
            self.warn("VIC output surface not writable".to_string());
            return;
        }
        for address in gpu::gpu_addresses_of(cpu) {
            recomp_gfx::engine_writes::record(address, bytes.len() as u64);
        }
        self.converted += 1;
        if tracing() {
            println!(
                "[recomp-host1x] VIC {width}x{height} format {format} {} into {cpu:#x}",
                if block_linear {
                    "block linear"
                } else {
                    "pitch"
                }
            );
        }
    }
}

/// The opcode stream's state while a command buffer runs.
#[derive(Debug, Default)]
struct Stream {
    class: u32,
    offset: u32,
    count: u32,
    mask: u32,
    incrementing: bool,
}

/// `len` bits at bit `bit` of a VIC configuration (little-endian words).
fn config_bits(config: &[u8], bit: usize, len: usize) -> u64 {
    (0..len).fold(0, |value, index| {
        let at = bit + index;
        let set = config
            .get(at / 8)
            .is_some_and(|byte| byte >> (at % 8) & 1 != 0);
        value | (u64::from(set) << index)
    })
}

/// A VIC color matrix: 20-bit signed coefficients, a right shift, and the
/// enable bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Matrix {
    enabled: bool,
    shift: u32,
    /// Per output row (r, g, b): the factors of the three inputs and the
    /// offset.
    rows: [[i64; 4]; 3],
}

impl Matrix {
    fn parse(config: &[u8], bit: usize) -> Self {
        let signed = |at: usize| {
            let value = config_bits(config, bit + at, 20) as i64;
            (value << 44) >> 44
        };
        // Coefficient (row, column) sits at a column's word, 20 bits a row.
        let coefficient = |row: usize, column: usize| signed(column * 64 + row * 20);
        Self {
            enabled: config_bits(config, bit + 127, 1) != 0,
            shift: config_bits(config, bit + 60, 4) as u32,
            rows: [0, 1, 2].map(|row| [0, 1, 2, 3].map(|column| coefficient(row, column))),
        }
    }

    /// 10-bit (y, u, v) through the matrix (Ryujinx's `MatrixMultiply`).
    fn apply(&self, input: [i64; 3]) -> [i64; 3] {
        if !self.enabled {
            return input;
        }
        self.rows.map(|row| {
            let sum = input[0] * row[0] + input[1] * row[1] + input[2] * row[2];
            ((sum >> self.shift) + row[3]) >> 8
        })
    }
}

/// Convert a frame's pixels in the target rectangle to RGBA in `pixels`
/// (`width` pixels a row): 10-bit components through the slot's matrix,
/// clamped, then 8-bit.
fn convert(
    frame: &vp9::Frame,
    matrix: &Matrix,
    (low, high): (i64, i64),
    (x0, y0, width, height): (usize, usize, usize, usize),
    row_pixels: usize,
    pixels: &mut [u8],
) {
    let (low, high) = if high > low { (low, high) } else { (0, 1023) };
    let chroma_width = frame.chroma_width() as usize;
    let x_end = (x0 + width).min(frame.width as usize);
    let y_end = (y0 + height).min(frame.height as usize);
    for y in y0..y_end {
        let row = &frame.y[y * frame.width as usize..];
        let chroma = (y / 2) * chroma_width;
        for (x, luma) in row.iter().enumerate().take(x_end).skip(x0) {
            let yuv = [
                i64::from(*luma) << 2,
                i64::from(frame.u[chroma + x / 2]) << 2,
                i64::from(frame.v[chroma + x / 2]) << 2,
            ];
            let rgb = matrix.apply(yuv);
            let at = (y * row_pixels + x) * 4;
            for (component, value) in rgb.into_iter().enumerate() {
                pixels[at + component] = (value.clamp(low, high) >> 2) as u8;
            }
            pixels[at + 3] = 255;
        }
    }
}

fn host1x() -> std::sync::MutexGuard<'static, Host1x> {
    static STATE: OnceLock<Mutex<Host1x>> = OnceLock::new();
    STATE
        .get_or_init(|| Mutex::new(Host1x::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Submits and engine runs are logged with `RECOMP_TRACE_HOST1X=1`.
fn tracing() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("RECOMP_TRACE_HOST1X").is_some())
}

/// The engine address of the buffer of memory handle `handle` (at CPU
/// address `address`, `size` bytes), mapped on first use.
pub fn map(handle: u32, address: u64, size: u32) -> u32 {
    let mut host1x = host1x();
    if let Some((cpu, engine)) = host1x.handles.get(&handle) {
        if *cpu == address {
            return *engine;
        }
    }
    let engine = host1x.next;
    let span = size.max(1).next_multiple_of(0x1_0000);
    host1x.next = host1x.next.saturating_add(span);
    host1x.mappings.insert(engine, (address, size.max(1)));
    host1x.handles.insert(handle, (address, engine));
    engine
}

/// Run a command buffer's words.
pub fn submit(mem: &dyn MemoryAccess, words: &[u32]) {
    let mut host1x = host1x();
    let mut stream = Stream::default();
    for word in words {
        host1x.step(mem, &mut stream, *word);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_bits_cross_bytes() {
        let config = [0b1010_0000, 0b0000_0011, 0xFF];
        assert_eq!(config_bits(&config, 5, 5), 0b11101);
        assert_eq!(config_bits(&config, 16, 8), 0xFF);
        assert_eq!(config_bits(&config, 20, 8), 0x0F);
    }

    #[test]
    fn matrices_turn_yuv_into_rgb() {
        // Identity on y with no chroma, coefficient 1 << 8 and no shift:
        // every output is the luma.
        let mut config = vec![0u8; 32];
        let mut set = |bit: usize, len: usize, value: u64| {
            for index in 0..len {
                if value >> index & 1 != 0 {
                    config[(bit + index) / 8] |= 1 << ((bit + index) % 8);
                }
            }
        };
        for row in 0..3 {
            set(row * 20, 20, 256);
        }
        set(127, 1, 1);
        let matrix = Matrix::parse(&config, 0);
        assert!(matrix.enabled);
        assert_eq!(matrix.apply([400, 512, 512]), [400, 400, 400]);
        // A negative coefficient sign-extends.
        let mut config = vec![0u8; 32];
        let negative = (1u64 << 20) - 256;
        for index in 0..20 {
            if negative >> index & 1 != 0 {
                config[(64 + index) / 8] |= 1 << ((64 + index) % 8);
            }
        }
        assert_eq!(Matrix::parse(&config, 0).rows[0][1], -256);
    }

    #[test]
    fn opcodes_write_engine_methods() {
        // SetClass VIC, then METHOD0 = EXECUTE - 1 and METHOD1 = 7 by Incr.
        let mut host1x = Host1x::new();
        let mut stream = Stream::default();
        #[derive(Debug)]
        struct NoMemory;
        impl MemoryAccess for NoMemory {
            fn load_u8(&self, _: u64) -> Option<u8> {
                None
            }
            fn load_u32(&self, _: u64) -> Option<u32> {
                None
            }
            fn load_u64(&self, _: u64) -> Option<u64> {
                None
            }
            fn store_u32(&self, _: u64, _: u32) -> bool {
                false
            }
            fn store_u64(&self, _: u64, _: u64) -> bool {
                false
            }
            fn read_bytes(&self, _: u64, _: usize) -> Option<Vec<u8>> {
                None
            }
            fn write_bytes(&self, _: u64, _: &[u8]) -> bool {
                false
            }
        }
        for word in [CLASS_VIC << 6, (1 << 28) | (0x10 << 16) | 2, 0x1C2, 0xABCD] {
            host1x.step(&NoMemory, &mut stream, word);
        }
        assert_eq!(host1x.vic[VIC_CONFIG], 0xABCD);
    }

    #[test]
    fn engine_addresses_map_handles() {
        let first = map(0x7001, 0x1_0000_0000, 0x2_0000);
        let second = map(0x7002, 0x2_0000_0000, 0x10);
        assert_ne!(first, second);
        assert_eq!(map(0x7001, 0x1_0000_0000, 0x2_0000), first);
        let host1x = host1x();
        assert_eq!(
            host1x.cpu_address(u64::from(first) + 0x100),
            Some(0x1_0000_0100)
        );
        assert_eq!(host1x.cpu_address(u64::from(second) + 0x10), None);
    }
}
