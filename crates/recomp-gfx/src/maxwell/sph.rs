//! The shader program header (SPH) and the instruction stream it describes.
//!
//! A guest shader program is `HEADER_SIZE` bytes of header followed by 64-bit
//! instruction words. The header says which stage the program is for, what it
//! writes, and the sizes of its local memory. Instruction addresses are
//! program-relative, so branch displacements count from the program's start:
//! inside the instruction stream, every word at a stream-relative multiple of
//! `SCHEDULE_PERIOD` is scheduler padding, not an instruction, and the first
//! instruction is at `HEADER_SIZE + INSTRUCTION_SIZE`.
//!
//! The field layout and instruction encodings here follow Ryujinx's
//! `ShaderHeader` and instruction table (MIT licensed; the notice is in
//! `THIRD_PARTY_NOTICES.md`). This module parses the
//! header and names instructions; translating them into host shader code is
//! separate work.

use crate::maxwell::render::GpuMemory;

/// Bytes of the program header, including the padding words inside it.
pub const HEADER_SIZE: usize = 0x50;

/// Bytes per instruction word.
pub const INSTRUCTION_SIZE: usize = 8;

/// Instruction addresses are program-relative, and words inside the
/// instruction stream whose stream-relative address is a multiple of this are
/// scheduler padding, not instructions. The stream starts after the header, so
/// the first instruction is at `HEADER_SIZE + INSTRUCTION_SIZE`.
pub const SCHEDULE_PERIOD: u64 = 32;

/// Bits of the instruction word that name the instruction.
pub const OPCODE_BITS: usize = 14;

/// Where the opcode field starts in the word.
pub const OPCODE_SHIFT: u32 = 64 - OPCODE_BITS as u32;

/// The GPR that reads as zero, and the predicate that is always true.
pub const REGISTER_ZERO: u8 = 255;
pub const PREDICATE_TRUE: u8 = 7;

/// true when the word at this program-relative address is scheduler padding.
/// Words in the header are header data, not padding; inside the instruction
/// stream, padding sits at every stream-relative multiple of `SCHEDULE_PERIOD`.
pub fn is_schedule_word(address: u64) -> bool {
    let header = HEADER_SIZE as u64;
    address >= header && (address - header).is_multiple_of(SCHEDULE_PERIOD)
}

/// The opcode field of an instruction word.
pub fn opcode_of(word: u64) -> u16 {
    (word >> OPCODE_SHIFT) as u16
}

/// The stage a program header names, as the hardware encodes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Encoded as 0. The 3D pipeline has no compute shader, so a header that
    /// says compute is read as vertex (`Stage::as_graphics`).
    Compute,
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry,
    Fragment,
}

impl Stage {
    pub fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Compute,
            1 => Self::Vertex,
            2 => Self::TessellationControl,
            3 => Self::TessellationEvaluation,
            4 => Self::Geometry,
            5 => Self::Fragment,
            _ => return None,
        })
    }

    /// The 3D state block this stage's registers live in: 0 vertex A,
    /// 1 vertex B, 2 tessellation control, 3 tessellation evaluation,
    /// 4 geometry, 5 fragment. `None` for compute, which has its own block.
    pub fn register_index(self) -> Option<usize> {
        match self {
            Self::Compute => None,
            Self::Vertex => Some(1),
            Self::TessellationControl => Some(2),
            Self::TessellationEvaluation => Some(3),
            Self::Geometry => Some(4),
            Self::Fragment => Some(5),
        }
    }

    /// The stage a graphics program header is read as.
    pub fn as_graphics(self) -> Self {
        if self == Self::Compute {
            Self::Vertex
        } else {
            self
        }
    }
}

/// The topology a geometry or tessellation program emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTopology {
    PointList,
    LineStrip,
    TriangleStrip,
}

impl OutputTopology {
    pub fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            1 => Self::PointList,
            6 => Self::LineStrip,
            7 => Self::TriangleStrip,
            _ => return None,
        })
    }
}

/// How one component of an input attribute is interpolated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PixelImap {
    #[default]
    Unused,
    Constant,
    Perspective,
    ScreenLinear,
}

impl PixelImap {
    pub fn from_u32(value: u32) -> Self {
        match value & 3 {
            1 => Self::Constant,
            2 => Self::Perspective,
            3 => Self::ScreenLinear,
            _ => Self::Unused,
        }
    }
}

/// The interpolation modes of one attribute's x, y, z and w.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImapPixel {
    pub x: PixelImap,
    pub y: PixelImap,
    pub z: PixelImap,
    pub w: PixelImap,
}

/// The header of a shader program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderProgramHeader {
    pub sph_type: u32,
    pub version: u32,
    pub stage: Stage,
    pub mrt_enable: bool,
    pub kills_pixels: bool,
    pub does_global_store: bool,
    pub sass_version: u32,
    pub gp_passthrough: bool,
    pub does_load_or_store: bool,
    pub does_fp64: bool,
    pub stream_out_mask: u32,
    pub local_memory_low_size: u32,
    pub per_patch_attribute_count: u32,
    pub local_memory_high_size: u32,
    pub threads_per_input_primitive: u32,
    pub local_memory_crs_size: u32,
    pub output_topology: Option<OutputTopology>,
    pub max_output_vertex_count: u32,
    pub store_req_start: u32,
    pub store_req_end: u32,
    /// Interpolation of the 32 input attributes, used by fragment programs.
    pub imap: [ImapPixel; 32],
    /// Output mask targets, used by fragment programs.
    pub omap_targets: u32,
    pub omap_sample_mask: bool,
    pub omap_depth: bool,
}

fn bits(word: u32, offset: u32, len: u32) -> u32 {
    (word >> offset) & ((1u32 << len) - 1)
}

fn bit(word: u32, offset: u32) -> bool {
    (word >> offset) & 1 != 0
}

fn bit64(word: u64, offset: u32) -> bool {
    (word >> offset) & 1 != 0
}

impl ShaderProgramHeader {
    /// Parse a header from a program's bytes. `None` when there are fewer than
    /// `HEADER_SIZE` bytes to read.
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < HEADER_SIZE {
            return None;
        }
        let word = |index: usize| {
            let at = index * 4;
            Some(u32::from_le_bytes(bytes[at..at + 4].try_into().ok()?))
        };
        let common0 = word(0)?;
        let common1 = word(1)?;
        let common2 = word(2)?;
        let common3 = word(3)?;
        let common4 = word(4)?;
        let stage = Stage::from_u32(bits(common0, 10, 4))?;

        // The interpolation words hold four attributes, two bits per component.
        let mut imap = [ImapPixel::default(); 32];
        for (index, entry) in imap.iter_mut().enumerate() {
            let packed = word(6 + index / 4)? >> (index % 4 * 8);
            *entry = ImapPixel {
                x: PixelImap::from_u32(packed),
                y: PixelImap::from_u32(packed >> 2),
                z: PixelImap::from_u32(packed >> 4),
                w: PixelImap::from_u32(packed >> 6),
            };
        }

        Some(Self {
            sph_type: bits(common0, 0, 5),
            version: bits(common0, 5, 5),
            stage,
            mrt_enable: bit(common0, 14),
            kills_pixels: bit(common0, 15),
            does_global_store: bit(common0, 16),
            sass_version: bits(common0, 17, 4),
            gp_passthrough: bit(common0, 24),
            does_load_or_store: bit(common0, 26),
            does_fp64: bit(common0, 27),
            stream_out_mask: bits(common0, 28, 4),
            local_memory_low_size: bits(common1, 0, 24),
            per_patch_attribute_count: bits(common1, 24, 8),
            local_memory_high_size: bits(common2, 0, 24),
            threads_per_input_primitive: bits(common2, 24, 8),
            local_memory_crs_size: bits(common3, 0, 24),
            output_topology: OutputTopology::from_u32(bits(common3, 24, 4)),
            max_output_vertex_count: bits(common4, 0, 12),
            store_req_start: bits(common4, 12, 8),
            store_req_end: bits(common4, 24, 8),
            imap,
            omap_targets: word(18)?,
            omap_sample_mask: bit(word(19)?, 0),
            omap_depth: bit(word(19)?, 1),
        })
    }

    /// Read the header of the program at `address`.
    pub fn read(memory: &dyn GpuMemory, address: u64) -> Option<Self> {
        Self::parse(&memory.read_gpu(address, HEADER_SIZE)?)
    }

    /// One line describing the program, for traces and dumps.
    pub fn describe(&self) -> String {
        format!(
            "sph type {} version {} stage {:?} sass {} local {}+{}+{} topology {:?} max_out {} imap_used {} global_store {}",
            self.sph_type,
            self.version,
            self.stage,
            self.sass_version,
            self.local_memory_low_size,
            self.local_memory_high_size,
            self.local_memory_crs_size,
            self.output_topology,
            self.max_output_vertex_count,
            self.imap
                .iter()
                .filter(|imap| *imap != &ImapPixel::default())
                .count(),
            self.does_global_store,
        )
    }
}

/// How an instruction reads its operand B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    /// The instruction has no operand B.
    None,
    /// A register in bits 20:27.
    Register(u8),
    /// A 20-bit immediate in bits 20:39.
    Immediate(u32),
    /// A 32-bit immediate in bits 20:52.
    Immediate32(u64),
    /// A constant-buffer read: slot in bits 34:38, offset in bits 20:33.
    Constant { slot: u8, offset: u16 },
    /// A branch displacement in bits 20:31, relative to the next instruction.
    Branch(i32),
}

/// A named instruction with the operand fields the encodings share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction {
    /// Address within the program, including the header's bytes.
    pub address: u64,
    /// The raw word.
    pub word: u64,
    /// The mnemonic, or "unknown" when no known pattern matches.
    pub name: &'static str,
    /// Destination register (bits 0:7), `None` when the instruction writes no GPR.
    pub dest: Option<u8>,
    /// Source A (bits 8:15).
    pub src_a: Option<u8>,
    /// Source B, in the form the instruction uses.
    pub src_b: Operand,
    /// Source C (bits 39:46).
    pub src_c: Option<u8>,
    /// Predicate the instruction is conditional on (bits 16:18; 7 is always true).
    pub pred: u8,
    pub pred_inv: bool,
    /// Predicate the instruction writes, for the compare instructions (bits 3:5).
    pub dest_pred: Option<u8>,
    pub dest_pred_inv: bool,
}

impl Instruction {
    pub fn known(&self) -> bool {
        self.name != "unknown"
    }

    fn register(name: u8) -> String {
        if name == REGISTER_ZERO {
            "rz".to_string()
        } else {
            format!("r{name}")
        }
    }

    /// Text for a listing, one instruction per line.
    pub fn format(&self) -> String {
        let mut text = format!("{:#06x}: {}", self.address, self.name);
        if let Some(dest) = self.dest {
            text.push_str(&format!(" {}", Self::register(dest)));
        }
        if let Some(src_a) = self.src_a {
            text.push_str(&format!(" {}", Self::register(src_a)));
        }
        match self.src_b {
            Operand::None => {}
            Operand::Register(reg) => text.push_str(&format!(" {}", Self::register(reg))),
            Operand::Immediate(value) => text.push_str(&format!(" {value:#x}")),
            Operand::Immediate32(value) => text.push_str(&format!(" {value:#x}")),
            Operand::Constant { slot, offset } => {
                text.push_str(&format!(" c{slot}[{offset:#x}]"));
            }
            Operand::Branch(offset) => {
                let target = self.address.wrapping_add(offset as u64).wrapping_add(8);
                text.push_str(&format!(" {:#06x}", target));
            }
        }
        if let Some(src_c) = self.src_c {
            text.push_str(&format!(" {}", Self::register(src_c)));
        }
        if let Some(dest_pred) = self.dest_pred {
            text.push_str(&format!(
                " p{dest_pred}{}",
                if self.dest_pred_inv { "!" } else { "" }
            ));
        }
        if self.pred != PREDICATE_TRUE {
            text.push_str(&format!(
                " if p{}{}",
                self.pred,
                if self.pred_inv { "!" } else { "" }
            ));
        }
        text
    }
}

/// How operand B is encoded, once the opcode is known.
#[derive(Debug, Clone, Copy)]
enum OperandKind {
    None,
    Register,
    Immediate,
    Immediate32,
    Constant,
    Branch,
}

/// An instruction pattern: which opcode bits must match, and what the
/// instruction's operands look like.
#[derive(Debug, Clone, Copy)]
struct Pattern {
    name: &'static str,
    /// The opcode bits that are fixed.
    bits: u16,
    /// Mask of the fixed bits.
    mask: u16,
    /// How many opcode bits are free. Fewer means a more specific pattern.
    wild: u8,
    rd: bool,
    ra: bool,
    rc: bool,
    src_b: OperandKind,
    /// The instruction writes a predicate instead of a GPR destination.
    pd: bool,
}

const fn pattern(
    encoding: &str,
    name: &'static str,
    src_b: OperandKind,
    rd: bool,
    ra: bool,
    rc: bool,
    pd: bool,
) -> Pattern {
    let bytes = encoding.as_bytes();
    let mut bits = 0u16;
    let mut mask = 0u16;
    let mut wild = 0u8;
    let mut index = 0;
    while index < OPCODE_BITS {
        if index >= bytes.len() {
            // A shorter encoding leaves the remaining opcode bits free.
            wild += 1;
        } else {
            match bytes[index] {
                b'1' => {
                    bits |= 1 << (OPCODE_BITS - 1 - index);
                    mask |= 1 << (OPCODE_BITS - 1 - index);
                }
                // A fixed zero must match too, so only `x` is free.
                b'0' => mask |= 1 << (OPCODE_BITS - 1 - index),
                b'x' => wild += 1,
                _ => {}
            }
        }
        index += 1;
    }
    Pattern {
        name,
        bits,
        mask,
        wild,
        rd,
        ra,
        rc,
        src_b,
        pd,
    }
}

/// The encodings this disassembler knows: the fixed top bits of the instruction
/// word, with `x` for a free bit. A pattern with fewer free bits wins, as the
/// The encodings this disassembler knows, copied from Ryujinx's instruction
/// table: the fixed top bits of the instruction word, with `x` for a free bit.
/// A pattern with fewer free bits wins, as the hardware's instruction table
/// does. Operand B and the written predicate come from each entry's operand
/// properties.
const PATTERNS: &[Pattern] = &[
    pattern(
        "000000010000xx",
        "MOV32I",
        OperandKind::Immediate32,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0000001xxxxxxx",
        "LOP3",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "000001xxxxxxxx",
        "LOP32I",
        OperandKind::Immediate32,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "000010xxxxxxxx",
        "FADD32I",
        OperandKind::Immediate32,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "000011xxxxxxxx",
        "FFMA32I",
        OperandKind::Immediate32,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "000100xxxxxxxx",
        "IMAD32I",
        OperandKind::Immediate32,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "000101xxxxxxxx",
        "ISCADD32I",
        OperandKind::Immediate32,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "000110xxxxxxxx",
        "LEAHI",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0001110xxxxxxx",
        "IADD32I",
        OperandKind::Immediate32,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "00011110xxxxxx",
        "FMUL32I",
        OperandKind::Immediate32,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "00011111xxxxxx",
        "IMUL32I",
        OperandKind::Immediate32,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "001000xxxxxxxx",
        "VADD",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0010100xxxxxxx",
        "HFMA2",
        OperandKind::Immediate32,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0010101xxxxxxx",
        "HMUL232I",
        OperandKind::Immediate32,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0010110xxxxxxx",
        "HADD232I",
        OperandKind::Immediate32,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011000xxxxxxx",
        "FSET",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011001x0xxxxx",
        "DSET",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011001x1xxxxx",
        "FFMA",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011010x0xxxxx",
        "IMAD",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011010x1xxxxx",
        "IMADSP",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011011x00xxxx",
        "XMAD",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011011x0100xx",
        "ICMP",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011011x0101xx",
        "ISET",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011011x0110xx",
        "ISETP",
        OperandKind::Immediate,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "0011011x0111xx",
        "DFMA",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011011x1000xx",
        "DSETP",
        OperandKind::Immediate,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "0011011x1010xx",
        "FCMP",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011011x1011xx",
        "FSETP",
        OperandKind::Immediate,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "0011011x1100xx",
        "PRMT",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011011x11010x",
        "LEA",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011011x11110x",
        "BFI",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011011x11111x",
        "SHF.L",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011100x00000x",
        "BFE",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x00001x",
        "POPC",
        OperandKind::Immediate,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0011100x00010x",
        "IADD",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x00011x",
        "ISCADD",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x00100x",
        "IMNMX",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x00101x",
        "SHR",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x00110x",
        "FLO",
        OperandKind::Immediate,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0011100x00111x",
        "IMUL",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x01000x",
        "LOP",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x01001x",
        "SHL",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x01010x",
        "DMNMX",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x01011x",
        "FADD",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x01100x",
        "FMNMX",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x01101x",
        "FMUL",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x01110x",
        "DADD",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x10000x",
        "DMUL",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x10001x",
        "FCHK",
        OperandKind::Immediate,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "0011100x10010x",
        "RRO",
        OperandKind::Immediate,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0011100x10011x",
        "MOV",
        OperandKind::Immediate,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0011100x10100x",
        "SEL",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x10101x",
        "F2F",
        OperandKind::Immediate,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0011100x10110x",
        "F2I",
        OperandKind::Immediate,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0011100x10111x",
        "I2F",
        OperandKind::Immediate,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0011100x1100xx",
        "IADD3",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011100x11100x",
        "I2I",
        OperandKind::Immediate,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0011100x11101x",
        "P2R",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x11110x",
        "R2P",
        OperandKind::Immediate,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "0011100x11111x",
        "SHF.R",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0011101xxxxxxx",
        "VMNMX",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "001111xxxxxxxx",
        "LOP3",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0100000xxxxxxx",
        "VSET",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "01001000xxxxxx",
        "FSET",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "010010010xxxxx",
        "DSET",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "010010011xxxxx",
        "FFMA",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010010100xxxxx",
        "IMAD",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010010101xxxxx",
        "IMADSP",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010010110100xx",
        "ICMP",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010010110101xx",
        "ISET",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "010010110110xx",
        "ISETP",
        OperandKind::Constant,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "010010110111xx",
        "DFMA",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010010111000xx",
        "DSETP",
        OperandKind::Constant,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "010010111010xx",
        "FCMP",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010010111011xx",
        "FSETP",
        OperandKind::Constant,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "010010111100xx",
        "PRMT",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0100101111010x",
        "LEA",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100101111110x",
        "BFI",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0100110000000x",
        "BFE",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110000001x",
        "POPC",
        OperandKind::Constant,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0100110000010x",
        "IADD",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110000011x",
        "ISCADD",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110000100x",
        "IMNMX",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110000101x",
        "SHR",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110000110x",
        "FLO",
        OperandKind::Constant,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0100110000111x",
        "IMUL",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110001000x",
        "LOP",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110001001x",
        "SHL",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110001010x",
        "DMNMX",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110001011x",
        "FADD",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110001100x",
        "FMNMX",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110001101x",
        "FMUL",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110001110x",
        "DADD",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110010000x",
        "DMUL",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110010001x",
        "FCHK",
        OperandKind::Constant,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "0100110010010x",
        "RRO",
        OperandKind::Constant,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0100110010011x",
        "MOV",
        OperandKind::Constant,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0100110010100x",
        "SEL",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110010101x",
        "F2F",
        OperandKind::Constant,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0100110010110x",
        "F2I",
        OperandKind::Constant,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0100110010111x",
        "I2F",
        OperandKind::Constant,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "010011001100xx",
        "IADD3",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0100110011100x",
        "I2I",
        OperandKind::Constant,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0100110011101x",
        "P2R",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0100110011110x",
        "R2P",
        OperandKind::Constant,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "0100111xxxxxxx",
        "XMAD",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010100000xxxxx",
        "VABSDIFF4",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101000010000x",
        "MUFU",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101000010001x",
        "PSET",
        OperandKind::None,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0101000010010x",
        "PSETP",
        OperandKind::None,
        false,
        false,
        false,
        true,
    ),
    pattern(
        "0101000010011x",
        "CSET",
        OperandKind::None,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0101000010100x",
        "CSETP",
        OperandKind::None,
        false,
        false,
        false,
        true,
    ),
    pattern(
        "0101000010110x",
        "NOP",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "0101000011001x",
        "CS2R",
        OperandKind::None,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0101000011010x",
        "LEPC",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "0101000011011x",
        "VOTE",
        OperandKind::None,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0101000011100x",
        "VOTEVTG",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "0101000011110x",
        "VSETP",
        OperandKind::Register,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "0101000011111x",
        "FSWZADD",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "010100010xxxxx",
        "XMAD",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010100011xxxxx",
        "FFMA",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010100100xxxxx",
        "IMAD",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010100101xxxxx",
        "IMADSP",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010100110100xx",
        "ICMP",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010100110111xx",
        "DFMA",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010100111010xx",
        "FCMP",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010100111100xx",
        "PRMT",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101001111011x",
        "IDP",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101001111110x",
        "BFI",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101001111111x",
        "IDP",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "01010100xxxxxx",
        "VABSDIFF",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "01010110xxxxxx",
        "VSHR",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "01010111xxxxxx",
        "VSHL",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "01011000xxxxxx",
        "FSET",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "010110010xxxxx",
        "DSET",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "010110011xxxxx",
        "FFMA",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010110100xxxxx",
        "IMAD",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010110101xxxxx",
        "IMADSP",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101101100xxxx",
        "XMAD",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010110110100xx",
        "ICMP",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010110110101xx",
        "ISET",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "010110110110xx",
        "ISETP",
        OperandKind::Register,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "010110110111xx",
        "DFMA",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010110111000xx",
        "DSETP",
        OperandKind::Register,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "010110111010xx",
        "FCMP",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "010110111011xx",
        "FSETP",
        OperandKind::Register,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "010110111100xx",
        "PRMT",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101101111010x",
        "LEA",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101101111011x",
        "LEAHI",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101101111100x",
        "LOP3",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101101111110x",
        "BFI",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101101111111x",
        "SHF.L",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101110000000x",
        "BFE",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110000001x",
        "POPC",
        OperandKind::Register,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0101110000010x",
        "IADD",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110000011x",
        "ISCADD",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110000100x",
        "IMNMX",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110000101x",
        "SHR",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110000110x",
        "FLO",
        OperandKind::Register,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0101110000111x",
        "IMUL",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110001000x",
        "LOP",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110001001x",
        "SHL",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110001010x",
        "DMNMX",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110001011x",
        "FADD",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110001100x",
        "FMNMX",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110001101x",
        "FMUL",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110001110x",
        "DADD",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110010000x",
        "DMUL",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110010001x",
        "FCHK",
        OperandKind::Register,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "0101110010010x",
        "RRO",
        OperandKind::Register,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0101110010011x",
        "MOV",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110010100x",
        "SEL",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110010101x",
        "F2F",
        OperandKind::Register,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0101110010110x",
        "F2I",
        OperandKind::Register,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0101110010111x",
        "I2F",
        OperandKind::Register,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "010111001100xx",
        "IADD3",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101110011100x",
        "I2I",
        OperandKind::Register,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "0101110011101x",
        "P2R",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110011110x",
        "R2P",
        OperandKind::Register,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "0101110011111x",
        "SHF.R",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101110100000x",
        "HFMA2",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0101110100001x",
        "HMUL2",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110100010x",
        "HADD2",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110100011x",
        "HSET2",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0101110100100x",
        "HSETP2",
        OperandKind::Register,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "01011111xxxxxx",
        "VMAD",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "01100xxx1xxxxx",
        "HFMA2",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "01110xxx0xxxxx",
        "HFMA2",
        OperandKind::Immediate,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "01110xxx1xxxxx",
        "HFMA2",
        OperandKind::Constant,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "0111100x0xxxxx",
        "HMUL2",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0111100x1xxxxx",
        "HMUL2",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0111101x0xxxxx",
        "HADD2",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0111101x1xxxxx",
        "HADD2",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0111110x0xxxxx",
        "HSET2",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0111110x1xxxxx",
        "HSET2",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "0111111x0xxxxx",
        "HSETP2",
        OperandKind::Immediate,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "0111111x1xxxxx",
        "HSETP2",
        OperandKind::Constant,
        false,
        true,
        false,
        true,
    ),
    pattern(
        "100xxxxxxxxxxx",
        "LD",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "101xxxxxxxxxxx",
        "ST",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "11000xxxxxxxxx",
        "TEX",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "110010xxxxxxxx",
        "TLD4",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101000xxxxxxx",
        "TEXSF16",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101001xxxxxxx",
        "TLDSF16",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101100xxxxxxx",
        "TEXS",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101101xxxxxxx",
        "TLDS",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "11011100xxxxxx",
        "TLD",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "11011101xxxxxx",
        "TLDB",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101111001xxxx",
        "TXDB",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "110111100xxxxx",
        "TXD",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101111010xxxx",
        "TEXB",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101111011xxxx",
        "TLD4B",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101111100xxxx",
        "TLD4S",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101111101000x",
        "TXA",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101111101001x",
        "TXQ",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101111101010x",
        "TXQB",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101111101011x",
        "TMML",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101111101100x",
        "TMMLB",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1101111110xxxx",
        "TLD4SF16",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "11100000xxxxxx",
        "IPA",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "111000100000xx",
        "JMX",
        OperandKind::Branch,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "111000100001xx",
        "JMP",
        OperandKind::Register,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "111000100010xx",
        "JCAL",
        OperandKind::Branch,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000100011xx",
        "PEXIT",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000100100xx",
        "BRA",
        OperandKind::Branch,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000100101xx",
        "BRX",
        OperandKind::Register,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "111000100110xx",
        "CAL",
        OperandKind::Branch,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000100111xx",
        "PRET",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000101000xx",
        "PLONGJMP",
        OperandKind::Branch,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000101001xx",
        "SSY",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000101010xx",
        "PBK",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000101011xx",
        "PCNT",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000101100xx",
        "GETCRSPTR",
        OperandKind::None,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "111000101101xx",
        "GETLMEMBASE",
        OperandKind::None,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "111000101110xx",
        "SETCRSPTR",
        OperandKind::None,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "111000101111xx",
        "SETLMEMBASE",
        OperandKind::None,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "111000110000xx",
        "EXIT",
        OperandKind::Branch,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000110001xx",
        "LONGJMP",
        OperandKind::Branch,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000110010xx",
        "RET",
        OperandKind::Branch,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000110011xx",
        "KIL",
        OperandKind::Branch,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000110100xx",
        "BRK",
        OperandKind::Branch,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000110101xx",
        "CONT",
        OperandKind::Branch,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000110110xx",
        "RTT",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000110111xx",
        "SAM",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000111000xx",
        "RAM",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000111001xx",
        "IDE",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "111000111010xx",
        "BPT",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "1110101001110x",
        "SUATOMB",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "1110101011010x",
        "SUATOMCASB",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "11101010x0xxxx",
        "SUATOM",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110101100000x",
        "SULDB",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "1110101100001x",
        "SULD",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110101100010x",
        "SULDDB",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "1110101100011x",
        "SULDD",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110101100100x",
        "SUSTB",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "1110101100101x",
        "SUST",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110101100110x",
        "SUSTDB",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "1110101100111x",
        "SUSTD",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110101101010x",
        "SUREDB",
        OperandKind::None,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "1110101101011x",
        "SURED",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110101110xxxx",
        "SUATOMB2",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "1110101111100x",
        "OUT",
        OperandKind::Constant,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110101111101x",
        "CCTLT",
        OperandKind::None,
        false,
        false,
        true,
        false,
    ),
    pattern(
        "1110101111110x",
        "CCTLT",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "1110101111111x",
        "RED",
        OperandKind::None,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "1110101x1xxxxx",
        "SUATOMCAS",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "11101100xxxxxx",
        "ATOMS",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "11101101xxxxxx",
        "ATOM",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "111011100xxxxx",
        "ATOMSCAS",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110111010100x",
        "STP",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "1110111011010x",
        "LDG",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110111011011x",
        "STG",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "111011101111xx",
        "ATOMCAS",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110111100010x",
        "SHFL",
        OperandKind::Register,
        true,
        true,
        true,
        false,
    ),
    pattern(
        "1110111101000x",
        "LDL",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110111101001x",
        "LDS",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110111101010x",
        "STL",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110111101011x",
        "STS",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "11101111011xxx",
        "CCTL",
        OperandKind::None,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "1110111110000x",
        "CCTLL",
        OperandKind::None,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "1110111110010x",
        "LDC",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110111110011x",
        "MEMBAR",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "1110111110100x",
        "AL2P",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110111111010x",
        "ISBERD",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110111111011x",
        "ALD",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110111111101x",
        "PIXLD",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1110111111110x",
        "AST",
        OperandKind::None,
        false,
        true,
        true,
        false,
    ),
    pattern(
        "1111000010101x",
        "BAR",
        OperandKind::None,
        false,
        true,
        false,
        false,
    ),
    pattern(
        "1111000010111x",
        "B2R",
        OperandKind::None,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1111000011000x",
        "R2B",
        OperandKind::Register,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "1111000011001x",
        "S2R",
        OperandKind::None,
        true,
        false,
        false,
        false,
    ),
    pattern(
        "1111000011110x",
        "DEPBAR",
        OperandKind::None,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "1111000011111x",
        "SYNC",
        OperandKind::Branch,
        false,
        false,
        false,
        false,
    ),
    pattern(
        "1111011x11100x",
        "OUT",
        OperandKind::Immediate,
        true,
        true,
        false,
        false,
    ),
    pattern(
        "1111101111100x",
        "OUT",
        OperandKind::Register,
        true,
        true,
        false,
        false,
    ),
];

/// Decode an instruction word, without an address.
pub fn decode(word: u64) -> Instruction {
    decode_at(0, word)
}

/// Decode the instruction word at `address` within a program.
pub fn decode_at(address: u64, word: u64) -> Instruction {
    let opcode = opcode_of(word);
    // The most specific matching pattern wins, so a fixed encoding is not
    // shadowed by a wildcard one that covers the same value.
    let found = PATTERNS
        .iter()
        .filter(|entry| (opcode & entry.mask) == entry.bits)
        .min_by_key(|entry| entry.wild);

    let src_b = match found.map(|entry| entry.src_b) {
        None | Some(OperandKind::None) => Operand::None,
        Some(OperandKind::Register) => Operand::Register(((word >> 20) & 0xFF) as u8),
        Some(OperandKind::Immediate) => Operand::Immediate(((word >> 20) & 0xF_FFFF) as u32),
        Some(OperandKind::Immediate32) => Operand::Immediate32((word >> 20) & 0xFFFF_FFFF),
        Some(OperandKind::Constant) => Operand::Constant {
            slot: ((word >> 34) & 0x1F) as u8,
            offset: ((word >> 20) & 0x3FFF) as u16,
        },
        Some(OperandKind::Branch) => Operand::Branch(branch_offset(word)),
    };

    let pd = found.is_some_and(|entry| entry.pd);
    Instruction {
        address,
        word,
        name: found.map_or("unknown", |entry| entry.name),
        dest: found.filter(|entry| entry.rd).map(|_| (word & 0xFF) as u8),
        src_a: found
            .filter(|entry| entry.ra)
            .map(|_| ((word >> 8) & 0xFF) as u8),
        src_b,
        src_c: found
            .filter(|entry| entry.rc)
            .map(|_| ((word >> 39) & 0xFF) as u8),
        pred: ((word >> 16) & 7) as u8,
        pred_inv: bit64(word, 19),
        dest_pred: pd.then_some(((word >> 3) & 7) as u8),
        dest_pred_inv: pd && (word & 7) != 0,
    }
}

/// The branch displacement of a branch instruction: bits 20:31, signed.
pub fn branch_offset(word: u64) -> i32 {
    (((word >> 20) as i32) << 8) >> 8
}

/// Disassemble the program at `address`, from the first instruction after the
/// header up to `size` bytes of program. Scheduling words and anything outside
/// the program are skipped. Addresses in the result are program-relative.
pub fn disassemble(memory: &dyn GpuMemory, address: u64, size: u64) -> Vec<Instruction> {
    let mut instructions = Vec::new();
    let mut offset = HEADER_SIZE as u64;
    while offset + INSTRUCTION_SIZE as u64 <= size {
        if !is_schedule_word(offset) {
            if let Some(raw) = memory.read_gpu(address + offset, INSTRUCTION_SIZE) {
                if let Ok(word) = raw.try_into() {
                    instructions.push(decode_at(offset, u64::from_le_bytes(word)));
                }
            }
        }
        offset += INSTRUCTION_SIZE as u64;
    }
    instructions
}

/// A listing of the program, one instruction per line. It stops at the first
/// word no known pattern names, so the listing ends where the instructions do.
pub fn listing(memory: &dyn GpuMemory, address: u64, size: u64) -> Vec<String> {
    disassemble(memory, address, size)
        .iter()
        .take_while(|instruction| instruction.known())
        .map(Instruction::format)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxwell::render::GpuMemory;

    /// Program bytes for the tests.
    struct Program {
        bytes: Vec<u8>,
    }

    impl Program {
        fn new(size: usize) -> Self {
            Program {
                bytes: vec![0; size],
            }
        }

        fn put(&mut self, address: usize, value: &[u8]) {
            self.bytes[address..address + value.len()].copy_from_slice(value);
        }
    }

    impl GpuMemory for Program {
        fn read_gpu(&self, address: u64, len: usize) -> Option<Vec<u8>> {
            self.bytes
                .get(address as usize..address as usize + len)
                .map(Vec::from)
        }
    }

    /// The opcode value an encoding string names.
    fn opcode(encoding: &str) -> u16 {
        let mut value = 0u16;
        for (index, byte) in encoding.as_bytes().iter().enumerate().take(OPCODE_BITS) {
            if *byte == b'1' {
                value |= 1 << (OPCODE_BITS - 1 - index);
            }
        }
        value
    }

    fn word(encoding: &str, fields: &[(u64, u32)]) -> u64 {
        let mut word = u64::from(opcode(encoding)) << OPCODE_SHIFT;
        for (value, shift) in fields {
            word |= value << shift;
        }
        word
    }

    fn header_bytes() -> Vec<u8> {
        let mut bytes = vec![0u8; HEADER_SIZE];
        // Type 2, version 1, vertex stage (bits 10:13), global store (bit 16).
        bytes[0..4].copy_from_slice(&(2u32 | (1 << 5) | (1 << 10) | (1 << 16)).to_le_bytes());
        // Local memory low size 0x100.
        bytes[4..8].copy_from_slice(&0x100u32.to_le_bytes());
        // Triangle strip output, word 3 bits 24:27.
        bytes[12..16].copy_from_slice(&(7u32 << 24).to_le_bytes());
        // Attribute 0: perspective in x, screen linear in y.
        bytes[24..28].copy_from_slice(&(2u32 | (3 << 2)).to_le_bytes());
        // Omap targets in word 18, depth written in word 19 bit 1.
        bytes[72..76].copy_from_slice(&3u32.to_le_bytes());
        bytes[76..80].copy_from_slice(&(1u32 << 1).to_le_bytes());
        bytes
    }

    #[test]
    fn header_reports_stage_sizes_and_imap() {
        let header = ShaderProgramHeader::parse(&header_bytes()).expect("header");
        assert_eq!(header.stage, Stage::Vertex);
        assert_eq!(header.stage.register_index(), Some(1));
        assert_eq!(header.sph_type, 2);
        assert_eq!(header.version, 1);
        assert!(header.does_global_store);
        assert_eq!(header.local_memory_low_size, 0x100);
        assert_eq!(header.output_topology, Some(OutputTopology::TriangleStrip));
        assert_eq!(
            header.imap[0],
            ImapPixel {
                x: PixelImap::Perspective,
                y: PixelImap::ScreenLinear,
                z: PixelImap::Unused,
                w: PixelImap::Unused,
            }
        );
        assert_eq!(
            header.imap[1],
            ImapPixel::default(),
            "later attributes are unused"
        );
        assert_eq!(header.omap_targets, 3);
        assert!(header.omap_depth);
        assert!(!header.omap_sample_mask);
    }

    #[test]
    fn header_needs_all_its_bytes() {
        assert!(ShaderProgramHeader::parse(&[0u8; HEADER_SIZE - 1]).is_none());
    }

    #[test]
    fn compute_headers_read_as_vertex_for_graphics() {
        let mut bytes = header_bytes();
        bytes[0..4].copy_from_slice(&0u32.to_le_bytes());
        let header = ShaderProgramHeader::parse(&bytes).expect("header");
        assert_eq!(header.stage, Stage::Compute);
        assert_eq!(header.stage.as_graphics(), Stage::Vertex);
        assert_eq!(header.stage.register_index(), None);
    }

    #[test]
    fn scheduling_words_are_not_instructions() {
        assert!(
            !is_schedule_word(0),
            "header words are header data, not padding"
        );
        assert!(!is_schedule_word(0x40));
        assert!(
            is_schedule_word(0x50),
            "the stream opens with a scheduling word"
        );
        assert!(!is_schedule_word(0x58));
        assert!(!is_schedule_word(0x60));
        assert!(is_schedule_word(0x70));
    }

    #[test]
    fn integer_add_from_registers() {
        let word = word("0101110000010", &[(5, 0), (6, 8), (2, 16), (7, 20)]);
        let instruction = decode_at(8, word);
        assert_eq!(instruction.name, "IADD");
        assert_eq!(instruction.dest, Some(5));
        assert_eq!(instruction.src_a, Some(6));
        assert_eq!(instruction.src_b, Operand::Register(7));
        assert_eq!(instruction.pred, 2);
        assert_eq!(instruction.format(), "0x0008: IADD r5 r6 r7 if p2");
    }

    #[test]
    fn immediate_and_constant_forms_win_over_the_register_form() {
        let immediate = word("0011100000010", &[(5, 0), (6, 8), (7, 16), (0x1234, 20)]);
        let instruction = decode(immediate);
        assert_eq!(instruction.name, "IADD");
        assert_eq!(instruction.src_b, Operand::Immediate(0x1234));

        // Constant form: offset in bits 20:33, slot in bits 34:38.
        let constant = word(
            "0100110000010",
            &[(5, 0), (6, 8), (7, 16), (0x40, 20), (3, 34)],
        );
        let instruction = decode(constant);
        assert_eq!(instruction.name, "IADD");
        assert_eq!(
            instruction.src_b,
            Operand::Constant {
                slot: 3,
                offset: 0x40
            }
        );
        assert_eq!(instruction.format(), "0x0000: IADD r5 r6 c3[0x40]");
    }

    #[test]
    fn zero_register_and_true_predicate_are_not_shown() {
        let word = word(
            "0101110000010",
            &[
                (REGISTER_ZERO as u64, 0),
                (6, 8),
                (PREDICATE_TRUE as u64, 16),
                (0, 20),
            ],
        );
        assert_eq!(decode(word).format(), "0x0000: IADD rz r6 r0");
    }

    #[test]
    fn compare_instructions_report_the_predicate_they_write() {
        // ISETP, register form: destination predicate in bits 3:5, inverted
        // when the low bits are non-zero.
        let word = word("010110110110", &[(2, 0), (1, 8), (4, 3), (7, 16)]);
        let instruction = decode(word);
        assert_eq!(instruction.name, "ISETP");
        assert_eq!(instruction.dest, None);
        assert_eq!(instruction.dest_pred, Some(4));
        assert!(instruction.dest_pred_inv);
        assert_eq!(instruction.pred, PREDICATE_TRUE);
        assert_eq!(instruction.format(), "0x0000: ISETP r1 r0 p4!");
    }

    #[test]
    fn branch_target_is_relative_to_the_next_instruction() {
        let word = word("111000100100", &[(7, 16), (1, 20)]);
        assert_eq!(branch_offset(word), 1);
        let instruction = decode_at(0x30, word);
        assert_eq!(instruction.name, "BRA");
        assert_eq!(instruction.src_b, Operand::Branch(1));
        assert_eq!(instruction.format(), "0x0030: BRA 0x0039");
    }

    #[test]
    fn unknown_opcode_is_reported_as_unknown() {
        let instruction = decode(0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(instruction.name, "unknown");
        assert!(!instruction.known());
        assert_eq!(instruction.dest, None);
    }

    #[test]
    fn listing_skips_scheduling_words() {
        let mut program = Program::new(0x68);
        // A vertex fetch (ALD) at 0x58: the header occupies 0x00..0x50, and the
        // word at 0x50 is the stream's first scheduling word, so the first
        // instruction sits at 0x58.
        let word = word("1110111111011", &[(9, 0), (PREDICATE_TRUE as u64, 16)]);
        program.put(0x58, &word.to_le_bytes());
        assert_eq!(listing(&program, 0, 0x68), vec!["0x0058: ALD r9 r0"]);
    }
}
