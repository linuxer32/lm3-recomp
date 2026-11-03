//! Maxwell shader programs translated to SPIR-V for the host renderer.
//!
//! A program is the 0x50-byte header ([`sph::ShaderProgramHeader`]) followed
//! by 64-bit instructions, with a scheduling word heading every 32-byte
//! bundle. Translation has two passes:
//!
//! 1. Control flow: instructions reachable from the entry are found by
//!    following branches; each `SYNC` is matched to the `SSY` whose region
//!    holds it (the innermost one in program order), and the program is cut
//!    into blocks at branch targets and after terminators. A branch target
//!    that lands on a scheduling word means the bundle's first instruction.
//! 2. Code: the blocks become cases of a `switch` inside a loop, selected by
//!    a block index variable, which keeps arbitrary branches structured as
//!    SPIR-V requires. Registers are 32-bit function variables (floats are
//!    bit casts), predicates are booleans, constant buffer `c[s][o]` is a
//!    uniform buffer at binding `s`, and each texture handle a program reads
//!    from the texture constant buffer becomes a combined image sampler.
//!
//! Instruction operand fields follow the hardware encodings as Ryujinx and
//! yuzu decode them; only the instructions real programs have needed so far
//! are translated, and anything else is reported with its address.

use std::collections::{BTreeMap, BTreeSet};

use crate::maxwell::sph::{
    self, Instruction, Operand, PixelImap, ShaderProgramHeader, Stage, HEADER_SIZE,
    INSTRUCTION_SIZE, PREDICATE_TRUE, REGISTER_ZERO,
};
use crate::spirv::{
    built_in, capability, decoration, dim, glsl, image_operands, mode, model, op, scope, semantics,
    storage, Builder, Id,
};

/// Descriptor set of a vertex program's resources; fragment programs use the
/// next set, so one pipeline layout holds both.
pub const VERTEX_SET: u32 = 0;
pub const FRAGMENT_SET: u32 = 1;
/// Textures are bound after the constant buffers (bindings 0..18).
pub const TEXTURE_BINDING_BASE: u32 = 32;

/// The push constants programs take their resolution scale from (16
/// bytes), by member: the scale the draw renders at over the guest's
/// resolution (`FragCoord` is divided by it), the scale of rendered images
/// bound as textures, and which texture bindings are such images (a bit
/// per binding, in the order of `TranslatedShader::textures`) for vertex
/// programs and for the others. Texel fetches multiply their coordinates
/// by a texture's scale and size queries divide by it, so programs see the
/// guest's positions and sizes.
pub const RESCALE_TARGET_SCALE: u32 = 0;
pub const RESCALE_TEXTURE_SCALE: u32 = 1;
pub const RESCALE_VERTEX_MASK: u32 = 2;
pub const RESCALE_OTHER_MASK: u32 = 3;
pub const RESCALE_BYTES: u32 = 16;
/// Words in a constant buffer as the host declares it (64 KiB).
const CONSTANT_BUFFER_VEC4S: u32 = 4096;

/// What a vertex attribute's components hold, from its format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttributeType {
    #[default]
    Float,
    SignedInt,
    UnsignedInt,
}

/// What a program is translated against, besides its own code.
#[derive(Debug, Clone, Default)]
pub struct Environment {
    pub vertex_attributes: [AttributeType; 32],
    /// The component type of each color target a fragment program writes
    /// (by draw buffer): integer targets take the registers' bits unchanged.
    pub color_targets: [AttributeType; 8],
    /// The clip distances (bit per distance) the draw enables: a vertex
    /// program's writes to the others are dropped, as the hardware ignores
    /// them.
    pub clip_distances: u8,
    /// A compute program's threads per block, from its dispatch.
    pub local_size: [u32; 3],
    /// A compute program's shared memory per block, in bytes.
    pub shared_memory_bytes: u32,
    /// The program's constant buffer 1 as bound when it is translated: NVN
    /// keeps a program's own constants there, `BRX` jump tables among them.
    pub constant_buffer_1: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureKind {
    D1,
    D2,
    D2Array,
    D3,
    Cube,
}

/// One combined image sampler a program samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureBinding {
    pub binding: u32,
    /// Byte offset of the texture handle in the texture constant buffer.
    pub handle_offset: u32,
    pub kind: TextureKind,
    /// Sampled with a depth comparison.
    pub shadow: bool,
}

#[derive(Debug, Clone)]
pub struct TranslatedShader {
    pub stage: Stage,
    pub words: Vec<u32>,
    pub descriptor_set: u32,
    /// Constant buffer slots read; slot `s` is at binding `s`.
    pub uniform_buffers: Vec<u32>,
    /// For each slot in `uniform_buffers`, the bytes from its start the
    /// program can read: past its last fixed offset, or the whole buffer when
    /// it indexes the buffer at run time. Uploading only these keeps a draw
    /// from copying 64 KiB per bound buffer.
    pub uniform_extents: Vec<u32>,
    pub textures: Vec<TextureBinding>,
    /// Generic attributes read, by location (vertex inputs or varyings).
    pub inputs: Vec<u32>,
    /// Generic attributes a vertex program writes, by location.
    pub outputs: Vec<u32>,
    /// Color targets a fragment program writes.
    pub color_outputs: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranslateError {
    NoHeader,
    UnsupportedStage(Stage),
    UnknownInstruction { address: u64, word: u64 },
    Unsupported { address: u64, detail: String },
    MissingInstruction { address: u64 },
}

impl std::fmt::Display for TranslateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoHeader => write!(f, "program has no header"),
            Self::UnsupportedStage(stage) => write!(f, "unsupported stage {stage:?}"),
            Self::UnknownInstruction { address, word } => {
                write!(f, "unknown instruction {word:016x} at {address:#x}")
            }
            Self::Unsupported { address, detail } => write!(f, "{detail} at {address:#x}"),
            Self::MissingInstruction { address } => {
                write!(f, "control reaches {address:#x}, outside the program")
            }
        }
    }
}

impl std::error::Error for TranslateError {}

fn unsupported(instruction: &Instruction, detail: impl Into<String>) -> TranslateError {
    TranslateError::Unsupported {
        address: instruction.address,
        detail: format!("{}: {}", instruction.name, detail.into()),
    }
}

fn field(word: u64, offset: u32, len: u32) -> u32 {
    ((word >> offset) & ((1u64 << len) - 1)) as u32
}

fn flag(word: u64, offset: u32) -> bool {
    (word >> offset) & 1 != 0
}

fn register(word: u64, offset: u32) -> u8 {
    field(word, offset, 8) as u8
}

fn word_at(program: &[u8], offset: u64) -> Option<u64> {
    let at = usize::try_from(offset).ok()?;
    Some(u64::from_le_bytes(
        program.get(at..at + INSTRUCTION_SIZE)?.try_into().ok()?,
    ))
}

/// The instruction slot after `offset`, skipping a scheduling word.
fn next_slot(offset: u64) -> u64 {
    let next = offset + INSTRUCTION_SIZE as u64;
    if sph::is_schedule_word(next) {
        next + INSTRUCTION_SIZE as u64
    } else {
        next
    }
}

/// The target of a branch at `offset`: relative to the next slot. A target on
/// a bundle's scheduling word is the bundle's first instruction.
fn branch_target(offset: u64, word: u64) -> u64 {
    let target =
        (offset as i64 + INSTRUCTION_SIZE as i64 + i64::from(sph::branch_offset(word))) as u64;
    if sph::is_schedule_word(target) {
        target + INSTRUCTION_SIZE as u64
    } else {
        target
    }
}

/// The first instruction: the slot after the stream's opening scheduling word.
fn entry_offset() -> u64 {
    next_slot(HEADER_SIZE as u64 - INSTRUCTION_SIZE as u64)
}

fn always(instruction: &Instruction) -> bool {
    instruction.pred == PREDICATE_TRUE && !instruction.pred_inv
}

fn never(instruction: &Instruction) -> bool {
    instruction.pred == PREDICATE_TRUE && instruction.pred_inv
}

/// `SSY`, `PBK` and `PCNT` are not predicated: their predicate bits are not
/// a condition.
fn conditional(instruction: &Instruction) -> bool {
    !matches!(instruction.name, "SSY" | "PBK" | "PCNT") && !always(instruction)
}

/// The instruction that pushes the target a returning instruction (`SYNC`,
/// `BRK`, `CONT`) goes back to.
fn pusher(returning: &str) -> &'static str {
    match returning {
        "BRK" => "PBK",
        "CONT" => "PCNT",
        _ => "SSY",
    }
}

/// The value of an IEEE half-precision bit pattern.
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let fraction = f32::from(bits & 0x3FF);
    sign * match exponent {
        0 => fraction * 2f32.powi(-24),
        0x1F if fraction == 0.0 => f32::INFINITY,
        0x1F => f32::NAN,
        _ => (1.0 + fraction / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// A half-precision instruction's operation.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HalfOperation {
    Add,
    Multiply,
    FusedMultiplyAdd,
}

/// Where a half-precision instruction's operands B and C come from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HalfForm {
    Register,
    /// Two 10-bit immediates (a half's top bits).
    Immediate,
    /// B from a constant buffer.
    Constant,
    /// B from register C, C from a constant buffer (`HFMA2` only).
    RegisterConstant,
    /// Both halves of a 32-bit immediate.
    Immediate32,
}

/// The instruction slot before `offset`, skipping a scheduling word.
fn previous_slot(offset: u64) -> u64 {
    let previous = offset - INSTRUCTION_SIZE as u64;
    if sph::is_schedule_word(previous) {
        previous - INSTRUCTION_SIZE as u64
    } else {
        previous
    }
}

/// The nearest instruction before `offset`, in straight-line code, that
/// writes `register`.
fn last_write(program: &[u8], offset: u64, register: u8) -> Option<Instruction> {
    let mut at = offset;
    for _ in 0..96 {
        if at <= entry_offset() {
            return None;
        }
        at = previous_slot(at);
        let instruction = sph::decode_at(at, word_at(program, at)?);
        if instruction.dest == Some(register) {
            return Some(instruction);
        }
        let leaves = always(&instruction)
            && matches!(
                instruction.name,
                "BRA" | "BRX" | "EXIT" | "KIL" | "SYNC" | "BRK" | "CONT"
            );
        if leaves {
            return None;
        }
    }
    None
}

/// The cases of a `BRX`, a switch through a jump table: the compiler clamps
/// the case number (`IMNMX.U32 rX, rX, last`), scales it (`SHL rX, rX, 2`)
/// and loads the case's code offset from the program's constants
/// (`LDC rX, c[1][rX + table]`); `BRX` jumps that far past its own target.
/// Each case is the loaded offset and the instruction it reaches.
fn jump_table(
    program: &[u8],
    brx: &Instruction,
    constants: &[u8],
) -> Result<Vec<(u32, u64)>, TranslateError> {
    let not_a_table = || unsupported(brx, "BRX not through a jump table");
    let load = last_write(program, brx.address, register(brx.word, 8)).ok_or_else(not_a_table)?;
    if load.name != "LDC" || field(load.word, 36, 5) != 1 || field(load.word, 44, 2) != 0 {
        return Err(not_a_table());
    }
    let table = i64::from(field(load.word, 20, 16) as u16 as i16);
    let scale =
        last_write(program, load.address, register(load.word, 8)).ok_or_else(not_a_table)?;
    if scale.name != "SHL" || scale.src_b != Operand::Immediate(2) {
        return Err(not_a_table());
    }
    let clamp =
        last_write(program, scale.address, register(scale.word, 8)).ok_or_else(not_a_table)?;
    // An immediate, unsigned minimum: the immediate is bits 20:38 and 56
    // (its sign), the selecting predicate (bits 39:42) is PT.
    let immediate = matches!(clamp.src_b, Operand::Immediate(_)) && !flag(clamp.word, 56);
    let unsigned_minimum = !flag(clamp.word, 48) && field(clamp.word, 39, 4) == 7;
    let last = field(clamp.word, 20, 19);
    if clamp.name != "IMNMX" || !immediate || !unsigned_minimum || last >= 0x400 {
        return Err(not_a_table());
    }
    let base =
        brx.address as i64 + INSTRUCTION_SIZE as i64 + i64::from(sph::branch_offset(brx.word));
    (0..=i64::from(last))
        .map(|case| {
            let at = usize::try_from(table + case * 4).map_err(|_| not_a_table())?;
            let offset = constants
                .get(at..at + 4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes")))
                .ok_or_else(|| unsupported(brx, "BRX jump table not in constant buffer 1"))?;
            let mut target = (base + i64::from(offset)) as u64;
            if sph::is_schedule_word(target) {
                target += INSTRUCTION_SIZE as u64;
            }
            Ok((offset, target))
        })
        .collect()
}

/// Entries of a bindless handle array a program may pick from: its length
/// is not in the program, so (as Ryujinx does) four are bound.
const HANDLE_ARRAY_ENTRIES: u32 = 4;

/// Where a bindless texture instruction's handle (register B) was loaded
/// from: NVN's texture handle buffer, c[2], at a fixed offset or indexed
/// (`SHL rI, rX, 3; LDC rH, c[2][rI + base]`, 8 bytes per entry). The
/// register holding the byte offset into the array (zero register when
/// fixed) and the array's offset.
fn handle_array(program: &[u8], instruction: &Instruction) -> Option<(u8, u32)> {
    let load = last_write(program, instruction.address, register(instruction.word, 20))?;
    if load.name != "LDC" || field(load.word, 36, 5) != 2 || field(load.word, 44, 2) != 0 {
        return None;
    }
    let base = u32::try_from(field(load.word, 20, 16) as u16 as i16).ok()?;
    // The index register holds the byte offset of a handle past `base`:
    // an array index scaled by 8, or an offset the program read from its
    // own constants (LM3's materials keep their handles' offsets there).
    let index = register(load.word, 8);
    Some((index, base))
}

struct Block {
    offsets: Vec<u64>,
    /// Where control continues when the last instruction does not leave.
    fallthrough: Option<u64>,
}

struct Flow {
    instructions: BTreeMap<u64, Instruction>,
    blocks: Vec<Block>,
    block_at: BTreeMap<u64, usize>,
    sync_targets: BTreeMap<u64, u64>,
    /// Each `BRX`'s cases (`jump_table`).
    jump_tables: BTreeMap<u64, Vec<(u32, u64)>>,
    /// Each bindless texture instruction's handle array (`handle_array`).
    handle_arrays: BTreeMap<u64, (u8, u32)>,
}

impl Flow {
    fn analyze(program: &[u8], constants: &[u8]) -> Result<Self, TranslateError> {
        let mut instructions = BTreeMap::new();
        let mut jump_tables = BTreeMap::new();
        let mut handle_arrays = BTreeMap::new();
        let mut pending = vec![entry_offset()];
        while let Some(offset) = pending.pop() {
            if instructions.contains_key(&offset) {
                continue;
            }
            let word = word_at(program, offset)
                .ok_or(TranslateError::MissingInstruction { address: offset })?;
            let instruction = sph::decode_at(offset, word);
            if !instruction.known() {
                return Err(TranslateError::UnknownInstruction {
                    address: offset,
                    word,
                });
            }
            let next = next_slot(offset);
            let live = !never(&instruction);
            match instruction.name {
                "EXIT" | "KIL" if live && always(&instruction) => {}
                "BRA" if live => {
                    pending.push(branch_target(offset, word));
                    if conditional(&instruction) {
                        pending.push(next);
                    }
                }
                "BRX" if live => {
                    let cases = jump_table(program, &instruction, constants)?;
                    pending.extend(cases.iter().map(|(_, target)| *target));
                    if conditional(&instruction) {
                        pending.push(next);
                    }
                    jump_tables.insert(offset, cases);
                }
                "TEXB" | "TLD4B" => {
                    if let Some(array) = handle_array(program, &instruction) {
                        handle_arrays.insert(offset, array);
                    }
                    pending.push(next);
                }
                "SSY" | "PBK" | "PCNT" => {
                    pending.push(branch_target(offset, word));
                    pending.push(next);
                }
                "SYNC" | "BRK" | "CONT" if live && always(&instruction) => {}
                _ => pending.push(next),
            }
            instructions.insert(offset, instruction);
        }

        // Each SYNC returns to the join point of the innermost SSY region
        // that holds it, in program order; BRK and CONT likewise to their
        // PBK and PCNT regions' targets.
        let mut sync_targets = BTreeMap::new();
        let mut regions: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
        for (offset, instruction) in &instructions {
            for stack in regions.values_mut() {
                while stack.last().is_some_and(|target| offset >= target) {
                    stack.pop();
                }
            }
            match instruction.name {
                "SSY" | "PBK" | "PCNT" => regions
                    .entry(instruction.name)
                    .or_default()
                    .push(branch_target(*offset, instruction.word)),
                "SYNC" | "BRK" | "CONT" if !never(instruction) => {
                    let kind = pusher(instruction.name);
                    let target = *regions
                        .get(kind)
                        .and_then(|stack| stack.last())
                        .ok_or_else(|| {
                            unsupported(
                                instruction,
                                format!("{} outside any {kind} region", instruction.name),
                            )
                        })?;
                    sync_targets.insert(*offset, target);
                }
                _ => {}
            }
        }

        // Blocks start at the entry, at branch and join targets, and after
        // instructions that may leave.
        let mut leaders = BTreeSet::from([entry_offset()]);
        for (offset, instruction) in &instructions {
            if never(instruction) {
                continue;
            }
            match instruction.name {
                "BRA" => {
                    leaders.insert(branch_target(*offset, instruction.word));
                    leaders.insert(next_slot(*offset));
                }
                "BRX" => {
                    leaders.extend(jump_tables[offset].iter().map(|(_, target)| *target));
                    leaders.insert(next_slot(*offset));
                }
                "SSY" | "PBK" | "PCNT" => {
                    leaders.insert(branch_target(*offset, instruction.word));
                }
                "SYNC" | "BRK" | "CONT" | "EXIT" | "KIL" => {
                    leaders.insert(next_slot(*offset));
                }
                _ => {}
            }
        }
        leaders.extend(sync_targets.values().copied());

        let mut blocks: Vec<Block> = Vec::new();
        let mut block_at = BTreeMap::new();
        let mut previous: Option<u64> = None;
        for (offset, instruction) in &instructions {
            let continues = previous.is_some_and(|previous| next_slot(previous) == *offset);
            if leaders.contains(offset) || !continues || blocks.is_empty() {
                block_at.insert(*offset, blocks.len());
                blocks.push(Block {
                    offsets: Vec::new(),
                    fallthrough: None,
                });
            }
            let block = blocks.last_mut().expect("a block was started");
            block.offsets.push(*offset);
            let leaves = !never(instruction)
                && always(instruction)
                && matches!(
                    instruction.name,
                    "BRA" | "BRX" | "EXIT" | "KIL" | "SYNC" | "BRK" | "CONT"
                );
            block.fallthrough = (!leaves).then(|| next_slot(*offset));
            previous = Some(*offset);
        }
        Ok(Self {
            instructions,
            blocks,
            block_at,
            sync_targets,
            jump_tables,
            handle_arrays,
        })
    }
}

/// A loaded sampled image, with the image type and kind behind it.
#[derive(Clone, Copy)]
struct Texture {
    value: Id,
    image: Id,
    kind: TextureKind,
    /// Its place among the program's texture bindings (`RESCALE_MASKS`).
    index: u32,
}

/// Common types.
struct Types {
    boolean: Id,
    uint: Id,
    int: Id,
    float: Id,
    vec2: Id,
    vec3: Id,
    vec4: Id,
    uvec4: Id,
    ivec4: Id,
}

struct Translator<'a> {
    b: Builder,
    t: Types,
    header: ShaderProgramHeader,
    environment: &'a Environment,
    main: Id,
    registers: BTreeMap<u8, Id>,
    predicates: BTreeMap<u8, Id>,
    constant_buffers: BTreeMap<u32, Id>,
    /// Bytes read from the start of each constant buffer (see
    /// `TranslatedShader::uniform_extents`).
    constant_extents: BTreeMap<u32, u32>,
    constant_array: Option<Id>,
    /// Each texture binding with its variable, sampled-image and image types.
    textures: Vec<(TextureBinding, Id, Id, Id)>,
    inputs: BTreeMap<u32, Id>,
    outputs: BTreeMap<u32, Id>,
    color_outputs: BTreeMap<u32, Id>,
    builtins: BTreeMap<u32, Id>,
    interface: Vec<Id>,
    /// Shared memory (compute programs), declared on first use.
    shared: Option<Id>,
    /// Local memory (per invocation) and its size in words, declared on
    /// first use.
    local: Option<(Id, u32)>,
    /// Bindless instructions' handle arrays (`Flow::handle_arrays`).
    handle_arrays: BTreeMap<u64, (u8, u32)>,
    /// The renderer's scales (`rescale`), declared on first use.
    rescale: Option<Id>,
    pc: Id,
    switch_merge: Id,
    loop_merge: Id,
    /// Set by KIL; the fragment is discarded once the dispatch loop ends.
    killed: Option<Id>,
}

impl<'a> Translator<'a> {
    fn new(header: ShaderProgramHeader, environment: &'a Environment) -> Self {
        let mut b = Builder::new();
        let float = b.f32();
        let uint = b.u32();
        let int = b.i32();
        let t = Types {
            boolean: b.bool(),
            uint,
            int,
            float,
            vec2: b.vector(float, 2),
            vec3: b.vector(float, 3),
            vec4: b.vector(float, 4),
            uvec4: b.vector(uint, 4),
            ivec4: b.vector(int, 4),
        };
        let main = b.begin_main();
        Self {
            b,
            t,
            header,
            environment,
            main,
            registers: BTreeMap::new(),
            predicates: BTreeMap::new(),
            constant_buffers: BTreeMap::new(),
            constant_extents: BTreeMap::new(),
            constant_array: None,
            textures: Vec::new(),
            handle_arrays: BTreeMap::new(),
            rescale: None,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            color_outputs: BTreeMap::new(),
            builtins: BTreeMap::new(),
            interface: Vec::new(),
            shared: None,
            local: None,
            pc: 0,
            switch_merge: 0,
            loop_merge: 0,
            killed: None,
        }
    }

    fn is_fragment(&self) -> bool {
        self.header.stage == Stage::Fragment
    }

    fn is_compute(&self) -> bool {
        self.header.stage == Stage::Compute
    }

    /// A component of a compute invocation's id in its block
    /// (`LocalInvocationId`) or of its block's id (`WorkgroupId`).
    fn invocation_id(&mut self, which: u32, component: u32) -> Id {
        let uvec3 = self.b.vector(self.t.uint, 3);
        let var = self.builtin(which, uvec3, storage::INPUT);
        let value = self.b.load(uvec3, var);
        self.b
            .inst(op::COMPOSITE_EXTRACT, self.t.uint, &[value, component])
    }

    /// The program's shared memory, as 32-bit words.
    fn shared_memory(&mut self) -> Id {
        if let Some(id) = self.shared {
            return id;
        }
        let words = (self.environment.shared_memory_bytes / 4).max(1);
        let array = self.b.array(self.t.uint, words);
        let id = self.b.global_variable(array, storage::WORKGROUP);
        self.shared = Some(id);
        id
    }

    /// The program's local memory (`LDL`/`STL`: per-invocation scratch, as
    /// spilled arrays use), as 32-bit words: the header's low and high
    /// sizes, at most 16 KiB.
    fn local_memory(&mut self) -> (Id, u32) {
        if let Some(local) = self.local {
            return local;
        }
        let bytes = self.header.local_memory_low_size + self.header.local_memory_high_size;
        let words = (bytes / 4).clamp(1, 4096);
        let array = self.b.array(self.t.uint, words);
        let id = self.b.local_variable(array);
        self.local = Some((id, words));
        (id, words)
    }

    /// A pointer to the local memory word at byte `address`; an address
    /// past the end reads and writes word 0 (the hardware's would be
    /// undefined, and a host access out of bounds must not happen).
    fn local_word(&mut self, address: Id) -> Id {
        let (local, words) = self.local_memory();
        let two = self.b.const_u32(2);
        let index = self
            .b
            .inst(op::SHIFT_RIGHT_LOGICAL, self.t.uint, &[address, two]);
        let limit = self.b.const_u32(words);
        let inside = self
            .b
            .inst(op::U_LESS_THAN, self.t.boolean, &[index, limit]);
        let zero = self.b.const_u32(0);
        let index = self.b.inst(op::SELECT, self.t.uint, &[inside, index, zero]);
        let pointer = self.b.pointer(storage::FUNCTION, self.t.uint);
        self.b.inst(op::ACCESS_CHAIN, pointer, &[local, index])
    }

    /// A pointer to the shared memory word at byte `address`.
    fn shared_word(&mut self, address: Id) -> Id {
        let shared = self.shared_memory();
        let two = self.b.const_u32(2);
        let index = self
            .b
            .inst(op::SHIFT_RIGHT_LOGICAL, self.t.uint, &[address, two]);
        let pointer = self.b.pointer(storage::WORKGROUP, self.t.uint);
        self.b.inst(op::ACCESS_CHAIN, pointer, &[shared, index])
    }

    fn descriptor_set(&self) -> u32 {
        if self.is_fragment() {
            FRAGMENT_SET
        } else {
            VERTEX_SET
        }
    }

    // ---- registers and predicates ----

    fn register_var(&mut self, index: u8) -> Id {
        if let Some(id) = self.registers.get(&index) {
            return *id;
        }
        let id = self.b.local_variable(self.t.uint);
        self.registers.insert(index, id);
        id
    }

    fn get_u(&mut self, index: u8) -> Id {
        if index == REGISTER_ZERO {
            return self.b.const_u32(0);
        }
        let var = self.register_var(index);
        self.b.load(self.t.uint, var)
    }

    fn get_f(&mut self, index: u8) -> Id {
        let value = self.get_u(index);
        self.b.bitcast(self.t.float, value)
    }

    fn set_u(&mut self, index: u8, value: Id) {
        if index == REGISTER_ZERO {
            return;
        }
        let var = self.register_var(index);
        self.b.store(var, value);
    }

    fn set_f(&mut self, index: u8, value: Id) {
        let bits = self.b.bitcast(self.t.uint, value);
        self.set_u(index, bits);
    }

    fn get_p(&mut self, index: u8, invert: bool) -> Id {
        let value = if index == PREDICATE_TRUE {
            self.b.const_bool(true)
        } else {
            let var = match self.predicates.get(&index) {
                Some(id) => *id,
                None => {
                    let id = self.b.local_variable(self.t.boolean);
                    self.predicates.insert(index, id);
                    id
                }
            };
            self.b.load(self.t.boolean, var)
        };
        if invert {
            self.b.inst(op::LOGICAL_NOT, self.t.boolean, &[value])
        } else {
            value
        }
    }

    fn set_p(&mut self, index: u8, value: Id) {
        if index == PREDICATE_TRUE {
            return;
        }
        let var = match self.predicates.get(&index) {
            Some(id) => *id,
            None => {
                let id = self.b.local_variable(self.t.boolean);
                self.predicates.insert(index, id);
                id
            }
        };
        self.b.store(var, value);
    }

    // ---- constant buffers ----

    fn constant_buffer(&mut self, slot: u32) -> Id {
        if let Some(id) = self.constant_buffers.get(&slot) {
            return *id;
        }
        let array = match self.constant_array {
            Some(array) => array,
            None => {
                let array = self.b.array(self.t.uvec4, CONSTANT_BUFFER_VEC4S);
                self.b.decorate(array, decoration::ARRAY_STRIDE, &[16]);
                self.constant_array = Some(array);
                array
            }
        };
        let block = self.b.struct_type(&[array]);
        self.b.decorate(block, decoration::BLOCK, &[]);
        self.b.member_decorate(block, 0, decoration::OFFSET, &[0]);
        let var = self.b.global_variable(block, storage::UNIFORM);
        let set = self.descriptor_set();
        self.b.decorate(var, decoration::DESCRIPTOR_SET, &[set]);
        self.b.decorate(var, decoration::BINDING, &[slot]);
        self.constant_buffers.insert(slot, var);
        var
    }

    /// The word at byte `offset` of constant buffer `slot`.
    fn cbuf_u(&mut self, slot: u32, offset: u32) -> Id {
        let extent = self.constant_extents.entry(slot).or_insert(0);
        *extent = (*extent).max(offset % (CONSTANT_BUFFER_VEC4S * 16) + 4);
        let var = self.constant_buffer(slot);
        let zero = self.b.const_u32(0);
        let vector = self.b.const_u32((offset / 16) % CONSTANT_BUFFER_VEC4S);
        let component = self.b.const_u32((offset / 4) % 4);
        let pointer = self.b.pointer(storage::UNIFORM, self.t.uint);
        let element = self
            .b
            .inst(op::ACCESS_CHAIN, pointer, &[var, zero, vector, component]);
        self.b.load(self.t.uint, element)
    }

    /// The constant-buffer word holding byte `address` (computed at run time).
    fn cbuf_u_at(&mut self, slot: u32, address: Id) -> Id {
        self.constant_extents
            .insert(slot, CONSTANT_BUFFER_VEC4S * 16);
        let var = self.constant_buffer(slot);
        let zero = self.b.const_u32(0);
        let (four, two) = (self.b.const_u32(4), self.b.const_u32(2));
        let vector = self
            .b
            .inst(op::SHIFT_RIGHT_LOGICAL, self.t.uint, &[address, four]);
        let wrap = self.b.const_u32(CONSTANT_BUFFER_VEC4S - 1);
        let vector = self.b.inst(op::BITWISE_AND, self.t.uint, &[vector, wrap]);
        let word = self
            .b
            .inst(op::SHIFT_RIGHT_LOGICAL, self.t.uint, &[address, two]);
        let three = self.b.const_u32(3);
        let component = self.b.inst(op::BITWISE_AND, self.t.uint, &[word, three]);
        let pointer = self.b.pointer(storage::UNIFORM, self.t.uint);
        let element = self
            .b
            .inst(op::ACCESS_CHAIN, pointer, &[var, zero, vector, component]);
        self.b.load(self.t.uint, element)
    }

    // ---- operands ----

    /// Operand B as raw bits: a register, a 20-bit immediate (sign in bit 56),
    /// a 32-bit immediate, or a constant buffer word.
    fn operand_b_u(&mut self, instruction: &Instruction) -> Result<Id, TranslateError> {
        let word = instruction.word;
        Ok(match instruction.src_b {
            Operand::Register(_) => self.get_u(register(word, 20)),
            Operand::Immediate(_) => {
                let mut value = field(word, 20, 19);
                if flag(word, 56) {
                    value |= 0xFFF8_0000;
                }
                self.b.const_u32(value)
            }
            Operand::Immediate32(_) => self.b.const_u32(field(word, 20, 32)),
            Operand::Constant { .. } => {
                let (slot, offset) = (field(word, 34, 5), field(word, 20, 14) * 4);
                self.cbuf_u(slot, offset)
            }
            _ => return Err(unsupported(instruction, "operand B form")),
        })
    }

    /// Operand B as a float: a 20-bit immediate holds the top bits of an f32.
    fn operand_b_f(&mut self, instruction: &Instruction) -> Result<Id, TranslateError> {
        let word = instruction.word;
        if let Operand::Immediate(_) = instruction.src_b {
            let bits = (field(word, 20, 19) << 12) | (u32::from(flag(word, 56)) << 31);
            return Ok(self.b.const_f32(f32::from_bits(bits)));
        }
        if let Operand::Immediate32(_) = instruction.src_b {
            return Ok(self.b.const_f32(f32::from_bits(field(word, 20, 32))));
        }
        let bits = self.operand_b_u(instruction)?;
        Ok(self.b.bitcast(self.t.float, bits))
    }

    fn cbuf_f(&mut self, word: u64) -> Id {
        let bits = self.cbuf_u(field(word, 34, 5), field(word, 20, 14) * 4);
        self.b.bitcast(self.t.float, bits)
    }

    fn fmod(&mut self, value: Id, abs: bool, neg: bool) -> Id {
        let mut value = value;
        if abs {
            value = self.b.ext(self.t.float, glsl::F_ABS, &[value]);
        }
        if neg {
            value = self.b.inst(op::F_NEGATE, self.t.float, &[value]);
        }
        value
    }

    fn saturate(&mut self, value: Id, sat: bool) -> Id {
        if !sat {
            return value;
        }
        let zero = self.b.const_f32(0.0);
        let one = self.b.const_f32(1.0);
        self.b.ext(self.t.float, glsl::F_CLAMP, &[value, zero, one])
    }

    fn fmul(&mut self, a: Id, b: Id) -> Id {
        self.b.inst(op::F_MUL, self.t.float, &[a, b])
    }

    fn fadd(&mut self, a: Id, b: Id) -> Id {
        self.b.inst(op::F_ADD, self.t.float, &[a, b])
    }

    fn select_u(&mut self, condition: Id, a: Id, b: Id) -> Id {
        self.b.inst(op::SELECT, self.t.uint, &[condition, a, b])
    }

    fn select_f(&mut self, condition: Id, a: Id, b: Id) -> Id {
        self.b.inst(op::SELECT, self.t.float, &[condition, a, b])
    }

    // ---- attributes ----

    fn interface_variable(&mut self, pointee: Id, class: u32) -> Id {
        let var = self.b.global_variable(pointee, class);
        self.interface.push(var);
        var
    }

    fn builtin(&mut self, which: u32, pointee: Id, class: u32) -> Id {
        if let Some(id) = self.builtins.get(&which) {
            return *id;
        }
        let var = self.interface_variable(pointee, class);
        self.b.decorate(var, decoration::BUILT_IN, &[which]);
        if which == built_in::FRAG_DEPTH {
            self.b.execution_mode(self.main, mode::DEPTH_REPLACING);
        }
        // Integer fragment inputs must not be interpolated.
        if self.is_fragment() && class == storage::INPUT && pointee == self.t.uint {
            self.b.decorate(var, decoration::FLAT, &[]);
        }
        self.builtins.insert(which, var);
        var
    }

    fn generic_input(&mut self, location: u32) -> (Id, Id) {
        let ty = if self.is_fragment() {
            self.t.vec4
        } else {
            match self.environment.vertex_attributes[location as usize] {
                AttributeType::Float => self.t.vec4,
                AttributeType::SignedInt => self.t.ivec4,
                AttributeType::UnsignedInt => self.t.uvec4,
            }
        };
        if let Some(id) = self.inputs.get(&location) {
            return (*id, ty);
        }
        let var = self.interface_variable(ty, storage::INPUT);
        self.b.decorate(var, decoration::LOCATION, &[location]);
        if self.is_fragment() {
            let imap = self.header.imap[location as usize];
            let mode = [imap.x, imap.y, imap.z, imap.w]
                .into_iter()
                .find(|mode| *mode != PixelImap::Unused)
                .unwrap_or(PixelImap::Perspective);
            match mode {
                PixelImap::Constant => self.b.decorate(var, decoration::FLAT, &[]),
                PixelImap::ScreenLinear => self.b.decorate(var, decoration::NO_PERSPECTIVE, &[]),
                _ => {}
            }
        }
        self.inputs.insert(location, var);
        (var, ty)
    }

    fn component_pointer(&mut self, var: Id, class: u32, scalar: Id, component: u32) -> Id {
        let pointer = self.b.pointer(class, scalar);
        let index = self.b.const_u32(component);
        self.b.inst(op::ACCESS_CHAIN, pointer, &[var, index])
    }

    /// Read input attribute word at byte `address` as raw bits.
    fn read_attribute(
        &mut self,
        instruction: &Instruction,
        address: u32,
    ) -> Result<Id, TranslateError> {
        let component = (address / 4) % 4;
        match address {
            0x80..=0x27F => {
                let location = (address - 0x80) / 16;
                let (var, ty) = self.generic_input(location);
                let scalar = if ty == self.t.vec4 {
                    self.t.float
                } else if ty == self.t.ivec4 {
                    self.t.int
                } else {
                    self.t.uint
                };
                let pointer = self.component_pointer(var, storage::INPUT, scalar, component);
                let value = self.b.load(scalar, pointer);
                Ok(if scalar == self.t.uint {
                    value
                } else {
                    self.b.bitcast(self.t.uint, value)
                })
            }
            0x70..=0x7F if self.is_fragment() => {
                let var = self.builtin(built_in::FRAG_COORD, self.t.vec4, storage::INPUT);
                let pointer = self.component_pointer(var, storage::INPUT, self.t.float, component);
                let mut value = self.b.load(self.t.float, pointer);
                if component < 2 {
                    // The guest's pixel position at any resolution scale.
                    let scale = self.rescale_member(RESCALE_TARGET_SCALE, self.t.float);
                    value = self.b.inst(op::F_DIV, self.t.float, &[value, scale]);
                }
                Ok(self.b.bitcast(self.t.uint, value))
            }
            0x2F8 | 0x2FC if !self.is_fragment() => {
                let which = if address == 0x2F8 {
                    built_in::INSTANCE_INDEX
                } else {
                    built_in::VERTEX_INDEX
                };
                let var = self.builtin(which, self.t.int, storage::INPUT);
                let value = self.b.load(self.t.int, var);
                Ok(self.b.bitcast(self.t.uint, value))
            }
            _ => Err(unsupported(
                instruction,
                format!("input attribute {address:#x}"),
            )),
        }
    }

    /// Write output attribute word at byte `address` from raw bits.
    fn generic_output(&mut self, location: u32) -> Id {
        if let Some(id) = self.outputs.get(&location) {
            return *id;
        }
        let var = self.interface_variable(self.t.vec4, storage::OUTPUT);
        self.b.decorate(var, decoration::LOCATION, &[location]);
        self.outputs.insert(location, var);
        var
    }

    /// The clip distance array (8 floats), when the draw enables a clip
    /// distance the program writes.
    fn clip_distance_output(&mut self) -> Id {
        if let Some(id) = self.builtins.get(&built_in::CLIP_DISTANCE) {
            return *id;
        }
        self.b.capability(capability::CLIP_DISTANCE);
        let array = self.b.array(self.t.float, 8);
        self.builtin(built_in::CLIP_DISTANCE, array, storage::OUTPUT)
    }

    /// A vertex program's outputs start as (0, 0, 0, 1), as on the hardware:
    /// programs may write only some components (a full-screen pass writes
    /// position x and y). Clip distances start at 0 (inside): every
    /// element of the array clips, written or not.
    fn initialize_outputs(&mut self, flow: &Flow) {
        let mut position = false;
        let mut clip = false;
        let mut locations = BTreeSet::new();
        for instruction in flow.instructions.values() {
            if instruction.name != "AST" {
                continue;
            }
            let base = field(instruction.word, 20, 10);
            for component in 0..=field(instruction.word, 47, 2) {
                match base + component * 4 {
                    0x70..=0x7F => position = true,
                    address @ 0x80..=0x27F => {
                        locations.insert((address - 0x80) / 16);
                    }
                    address @ 0x2C0..=0x2DF => {
                        clip |=
                            self.environment.clip_distances & (1 << ((address - 0x2C0) / 4)) != 0;
                    }
                    _ => {}
                }
            }
        }
        let (zero, one) = (self.b.const_f32(0.0), self.b.const_f32(1.0));
        let default = self.vec(&[zero, zero, zero, one]);
        if clip {
            let var = self.clip_distance_output();
            for index in 0..8 {
                let pointer = self.component_pointer(var, storage::OUTPUT, self.t.float, index);
                self.b.store(pointer, zero);
            }
        }
        if position {
            let var = self.builtin(built_in::POSITION, self.t.vec4, storage::OUTPUT);
            self.b.store(var, default);
        }
        for location in locations {
            let var = self.generic_output(location);
            self.b.store(var, default);
        }
    }

    fn write_attribute(
        &mut self,
        instruction: &Instruction,
        address: u32,
        value: Id,
    ) -> Result<(), TranslateError> {
        let component = (address / 4) % 4;
        let float = self.b.bitcast(self.t.float, value);
        match address {
            0x80..=0x27F => {
                let var = self.generic_output((address - 0x80) / 16);
                let pointer = self.component_pointer(var, storage::OUTPUT, self.t.float, component);
                self.b.store(pointer, float);
            }
            0x70..=0x7F => {
                let var = self.builtin(built_in::POSITION, self.t.vec4, storage::OUTPUT);
                let pointer = self.component_pointer(var, storage::OUTPUT, self.t.float, component);
                self.b.store(pointer, float);
            }
            0x6C => {
                let var = self.builtin(built_in::POINT_SIZE, self.t.float, storage::OUTPUT);
                self.b.store(var, float);
            }
            // Clip distances 0-7; disabled ones do not clip.
            0x2C0..=0x2DF => {
                let index = (address - 0x2C0) / 4;
                if self.environment.clip_distances & (1 << index) != 0 {
                    let var = self.clip_distance_output();
                    let pointer = self.component_pointer(var, storage::OUTPUT, self.t.float, index);
                    self.b.store(pointer, float);
                }
            }
            _ => {
                return Err(unsupported(
                    instruction,
                    format!("output attribute {address:#x}"),
                ))
            }
        }
        Ok(())
    }

    // ---- textures ----

    /// The sampled image for a texture handle, with its image type.
    fn texture(&mut self, handle_offset: u32, kind: TextureKind, shadow: bool) -> Texture {
        let found = self.textures.iter().position(|(binding, _, _, _)| {
            binding.handle_offset == handle_offset
                && binding.kind == kind
                && binding.shadow == shadow
        });
        let index = found.unwrap_or(self.textures.len()) as u32;
        let (var, sampled, image) = match found.map(|index| self.textures[index]) {
            Some((_, var, sampled, image)) => (var, sampled, image),
            None => {
                let (image_dim, arrayed) = match kind {
                    TextureKind::D1 => {
                        self.b.capability(capability::SAMPLED_1D);
                        (dim::D1, false)
                    }
                    TextureKind::D2 => (dim::D2, false),
                    TextureKind::D2Array => (dim::D2, true),
                    TextureKind::D3 => (dim::D3, false),
                    TextureKind::Cube => (dim::CUBE, false),
                };
                let image = self.b.image(image_dim, shadow, arrayed);
                let sampled = self.b.sampled_image(image);
                let var = self.b.global_variable(sampled, storage::UNIFORM_CONSTANT);
                let binding = TextureBinding {
                    binding: TEXTURE_BINDING_BASE + self.textures.len() as u32,
                    handle_offset,
                    kind,
                    shadow,
                };
                let set = self.descriptor_set();
                self.b.decorate(var, decoration::DESCRIPTOR_SET, &[set]);
                self.b
                    .decorate(var, decoration::BINDING, &[binding.binding]);
                self.textures.push((binding, var, sampled, image));
                (var, sampled, image)
            }
        };
        Texture {
            value: self.b.load(sampled, var),
            image,
            kind,
            index,
        }
    }

    // ---- resolution scale ----

    /// The renderer's scales for this draw or dispatch, as push constants
    /// (`RESCALE_*`): when it renders at a multiple of the guest's
    /// resolution, programs still see the guest's pixel positions, texel
    /// coordinates and sizes.
    fn rescale(&mut self) -> Id {
        if let Some(var) = self.rescale {
            return var;
        }
        let members = [self.t.float, self.t.float, self.t.uint, self.t.uint];
        let block = self.b.struct_type(&members);
        self.b.decorate(block, decoration::BLOCK, &[]);
        for member in 0..members.len() as u32 {
            self.b
                .member_decorate(block, member, decoration::OFFSET, &[member * 4]);
        }
        let var = self.b.global_variable(block, storage::PUSH_CONSTANT);
        self.rescale = Some(var);
        var
    }

    fn rescale_member(&mut self, member: u32, ty: Id) -> Id {
        let var = self.rescale();
        let pointer = self.b.pointer(storage::PUSH_CONSTANT, ty);
        let index = self.b.const_u32(member);
        let element = self.b.inst(op::ACCESS_CHAIN, pointer, &[var, index]);
        self.b.load(ty, element)
    }

    /// The scale `texture` is bound at: the renderer's when it is a
    /// rendered image at that scale (its bit in the stage's mask), else 1.
    fn texture_scale(&mut self, texture: &Texture) -> Id {
        let one = self.b.const_f32(1.0);
        if texture.index >= 32 {
            return one;
        }
        let member = if self.header.stage == Stage::Vertex {
            RESCALE_VERTEX_MASK
        } else {
            RESCALE_OTHER_MASK
        };
        let mask = self.rescale_member(member, self.t.uint);
        let shift = self.b.const_u32(texture.index);
        let bit = self
            .b
            .inst(op::SHIFT_RIGHT_LOGICAL, self.t.uint, &[mask, shift]);
        let low = self.b.const_u32(1);
        let bit = self.b.inst(op::BITWISE_AND, self.t.uint, &[bit, low]);
        let zero = self.b.const_u32(0);
        let scaled = self.b.inst(op::I_NOT_EQUAL, self.t.boolean, &[bit, zero]);
        let scale = self.rescale_member(RESCALE_TEXTURE_SCALE, self.t.float);
        self.b.inst(op::SELECT, self.t.float, &[scaled, scale, one])
    }

    /// An integer texel coordinate times `scale` (rounded down).
    fn scale_texel(&mut self, value: Id, scale: Id) -> Id {
        let value = self.b.inst(op::CONVERT_S_TO_F, self.t.float, &[value]);
        let value = self.b.inst(op::F_MUL, self.t.float, &[value, scale]);
        self.b.inst(op::CONVERT_F_TO_S, self.t.int, &[value])
    }

    /// An integer size divided by `scale` (rounded down).
    fn unscale_size(&mut self, value: Id, scale: Id) -> Id {
        let value = self.b.inst(op::CONVERT_S_TO_F, self.t.float, &[value]);
        let value = self.b.inst(op::F_DIV, self.t.float, &[value, scale]);
        self.b.inst(op::CONVERT_F_TO_S, self.t.int, &[value])
    }

    fn vec(&mut self, parts: &[Id]) -> Id {
        let ty = match parts.len() {
            2 => self.t.vec2,
            3 => self.t.vec3,
            _ => self.t.vec4,
        };
        self.b.inst(op::COMPOSITE_CONSTRUCT, ty, parts)
    }

    /// The array layer a coordinate register holds: its low 16 bits.
    fn layer(&mut self, index: u8) -> Id {
        let bits = self.get_u(index);
        let mask = self.b.const_u32(0xFFFF);
        let low = self.b.inst(op::BITWISE_AND, self.t.uint, &[bits, mask]);
        self.b.inst(op::CONVERT_U_TO_F, self.t.float, &[low])
    }

    /// `coords` moved by a texel `offset` (an ivec2): Vulkan only takes a
    /// non-constant offset on gathers, so sampling adds offset / size, with
    /// the size of the level sampled (level 0 for implicit-lod sampling).
    fn offset_coords(&mut self, texture: Texture, coords: Id, offset: Id, lod: Option<Id>) -> Id {
        let kind = texture.kind;
        let image_type = texture.image;
        self.b.capability(capability::IMAGE_QUERY);
        let image = self.b.inst(op::IMAGE, image_type, &[texture.value]);
        let level = match lod {
            Some(lod) => self.b.inst(op::CONVERT_F_TO_S, self.t.int, &[lod]),
            None => self.b.const_i32(0),
        };
        let ivec2 = self.b.vector(self.t.int, 2);
        let size_type = if kind == TextureKind::D2Array {
            self.b.vector(self.t.int, 3)
        } else {
            ivec2
        };
        let size = self
            .b
            .inst(op::IMAGE_QUERY_SIZE_LOD, size_type, &[image, level]);
        let (width, height) = (
            self.b.inst(op::COMPOSITE_EXTRACT, self.t.int, &[size, 0]),
            self.b.inst(op::COMPOSITE_EXTRACT, self.t.int, &[size, 1]),
        );
        let size2 = self
            .b
            .inst(op::COMPOSITE_CONSTRUCT, ivec2, &[width, height]);
        let size_f = self.b.inst(op::CONVERT_S_TO_F, self.t.vec2, &[size2]);
        // The offset is in the guest's texels.
        let scale = self.texture_scale(&texture);
        let scale2 = self.vec(&[scale, scale]);
        let size_f = self.b.inst(op::F_DIV, self.t.vec2, &[size_f, scale2]);
        let offset_f = self.b.inst(op::CONVERT_S_TO_F, self.t.vec2, &[offset]);
        let delta = self.b.inst(op::F_DIV, self.t.vec2, &[offset_f, size_f]);
        let (dx, dy) = (
            self.b
                .inst(op::COMPOSITE_EXTRACT, self.t.float, &[delta, 0]),
            self.b
                .inst(op::COMPOSITE_EXTRACT, self.t.float, &[delta, 1]),
        );
        let x = self
            .b
            .inst(op::COMPOSITE_EXTRACT, self.t.float, &[coords, 0]);
        let y = self
            .b
            .inst(op::COMPOSITE_EXTRACT, self.t.float, &[coords, 1]);
        let (x, y) = (self.fadd(x, dx), self.fadd(y, dy));
        if kind == TextureKind::D2Array {
            let layer = self
                .b
                .inst(op::COMPOSITE_EXTRACT, self.t.float, &[coords, 2]);
            self.vec(&[x, y, layer])
        } else {
            self.vec(&[x, y])
        }
    }

    /// Sample with optional explicit lod, bias, offset and depth reference.
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &mut self,
        texture: Texture,
        coords: Id,
        lod: Option<Id>,
        bias: Option<Id>,
        offset: Option<Id>,
        dref: Option<Id>,
    ) -> Id {
        let coords = match offset {
            Some(offset) => self.offset_coords(texture, coords, offset, lod),
            None => coords,
        };
        let texture = texture.value;
        let mut mask = 0;
        let mut extra = Vec::new();
        if let Some(bias) = bias {
            mask |= image_operands::BIAS;
            extra.push(bias);
        }
        if let Some(lod) = lod {
            mask |= image_operands::LOD;
            extra.push(lod);
        }
        let mut operands = vec![texture, coords];
        if let Some(dref) = dref {
            operands.push(dref);
        }
        if mask != 0 {
            operands.push(mask);
            operands.extend(extra);
        }
        let (opcode, ty) = match (dref.is_some(), lod.is_some()) {
            (false, false) => (op::IMAGE_SAMPLE_IMPLICIT_LOD, self.t.vec4),
            (false, true) => (op::IMAGE_SAMPLE_EXPLICIT_LOD, self.t.vec4),
            (true, false) => (op::IMAGE_SAMPLE_DREF_IMPLICIT_LOD, self.t.float),
            (true, true) => (op::IMAGE_SAMPLE_DREF_EXPLICIT_LOD, self.t.float),
        };
        let result = self.b.inst(opcode, ty, &operands);
        if dref.is_some() {
            // A depth comparison gives one value; it fills r, g and b.
            let one = self.b.const_f32(1.0);
            self.vec(&[result, result, result, one])
        } else {
            result
        }
    }

    fn gather(
        &mut self,
        texture: Texture,
        coords: Id,
        component: u32,
        offset: Option<Id>,
        dref: Option<Id>,
    ) -> Id {
        let mut operands = vec![texture.value, coords];
        let opcode = match dref {
            Some(dref) => {
                operands.push(dref);
                op::IMAGE_DREF_GATHER
            }
            None => {
                operands.push(self.b.const_u32(component));
                op::IMAGE_GATHER
            }
        };
        if let Some(offset) = offset {
            self.b.capability(capability::IMAGE_GATHER_EXTENDED);
            operands.push(image_operands::OFFSET);
            operands.push(offset);
        }
        self.b.inst(opcode, self.t.vec4, &operands)
    }

    /// A 2D texel offset from a register: x in bits 0:3, y in bits 4:7, signed.
    fn offset_2d(&mut self, index: u8, stride: u32) -> Id {
        let bits = self.get_u(index);
        let signed = self.b.bitcast(self.t.int, bits);
        let width = self.b.const_u32(stride.min(6));
        let x_at = self.b.const_u32(0);
        let y_at = self.b.const_u32(if stride == 4 { 4 } else { 8 });
        let x = self
            .b
            .inst(op::BIT_FIELD_S_EXTRACT, self.t.int, &[signed, x_at, width]);
        let y = self
            .b
            .inst(op::BIT_FIELD_S_EXTRACT, self.t.int, &[signed, y_at, width]);
        let ivec2 = self.b.vector(self.t.int, 2);
        self.b.inst(op::COMPOSITE_CONSTRUCT, ivec2, &[x, y])
    }

    /// An ivec2 of the signed 6-bit values at bits `shift` and `shift + 8`
    /// of a register (one texel of a per-texel offset list).
    fn offset_pair(&mut self, index: u8, shift: u32) -> Id {
        let bits = self.get_u(index);
        let signed = self.b.bitcast(self.t.int, bits);
        let width = self.b.const_u32(6);
        let x_at = self.b.const_u32(shift);
        let y_at = self.b.const_u32(shift + 8);
        let x = self
            .b
            .inst(op::BIT_FIELD_S_EXTRACT, self.t.int, &[signed, x_at, width]);
        let y = self
            .b
            .inst(op::BIT_FIELD_S_EXTRACT, self.t.int, &[signed, y_at, width]);
        let ivec2 = self.b.vector(self.t.int, 2);
        self.b.inst(op::COMPOSITE_CONSTRUCT, ivec2, &[x, y])
    }

    /// Store the components `mask` selects, in order, to `destinations`.
    fn store_components(&mut self, result: Id, mask: u32, destinations: &[u8]) {
        let mut next = 0;
        for component in 0..4 {
            if mask & (1 << component) == 0 {
                continue;
            }
            let Some(destination) = destinations.get(next).copied() else {
                break;
            };
            next += 1;
            let value = self
                .b
                .inst(op::COMPOSITE_EXTRACT, self.t.float, &[result, component]);
            self.set_f(destination, value);
        }
    }

    // ---- program structure ----

    fn emit_outputs_and_return(&mut self) {
        if self.is_fragment() {
            let mut source = 0u8;
            for target in 0..8u32 {
                let mask = (self.header.omap_targets >> (target * 4)) & 0xF;
                if mask == 0 {
                    continue;
                }
                let kind = self.environment.color_targets[target as usize];
                let float = kind == AttributeType::Float;
                let mut parts = [0; 4];
                for (component, part) in parts.iter_mut().enumerate() {
                    *part = if mask & (1 << component) != 0 {
                        let value = if float {
                            self.get_f(source)
                        } else {
                            self.get_u(source)
                        };
                        source += 1;
                        value
                    } else if float {
                        self.b.const_f32(0.0)
                    } else {
                        self.b.const_u32(0)
                    };
                }
                let (value, ty) = match kind {
                    AttributeType::Float => (self.vec(&parts), self.t.vec4),
                    AttributeType::UnsignedInt => (
                        self.b.inst(op::COMPOSITE_CONSTRUCT, self.t.uvec4, &parts),
                        self.t.uvec4,
                    ),
                    AttributeType::SignedInt => {
                        let bits = self.b.inst(op::COMPOSITE_CONSTRUCT, self.t.uvec4, &parts);
                        (self.b.bitcast(self.t.ivec4, bits), self.t.ivec4)
                    }
                };
                let var = match self.color_outputs.get(&target) {
                    Some(id) => *id,
                    None => {
                        let var = self.interface_variable(ty, storage::OUTPUT);
                        self.b.decorate(var, decoration::LOCATION, &[target]);
                        self.color_outputs.insert(target, var);
                        var
                    }
                };
                self.b.store(var, value);
            }
            if self.header.omap_depth {
                let depth = self.get_f(source + 1);
                let var = self.builtin(built_in::FRAG_DEPTH, self.t.float, storage::OUTPUT);
                self.b.store(var, depth);
            }
        }
        self.b.inst_void(op::RETURN, &[]);
    }

    /// Run `body` only when the instruction's predicate holds.
    fn guarded(
        &mut self,
        instruction: &Instruction,
        body: impl FnOnce(&mut Self) -> Result<bool, TranslateError>,
    ) -> Result<(), TranslateError> {
        if !conditional(instruction) {
            body(self)?;
            return Ok(());
        }
        let condition = self.get_p(instruction.pred, instruction.pred_inv);
        let then = self.b.id();
        let after = self.b.id();
        self.b.selection_merge(after);
        self.b.branch_conditional(condition, then, after);
        self.b.label(then);
        // The body reports whether it already ended the block (return/kill).
        if !body(self)? {
            self.b.branch(after);
        }
        self.b.label(after);
        Ok(())
    }

    fn goto(
        &mut self,
        flow: &Flow,
        target: u64,
        instruction: &Instruction,
    ) -> Result<(), TranslateError> {
        let index = *flow.block_at.get(&target).ok_or_else(|| {
            unsupported(
                instruction,
                format!("branch to {target:#x}, not a block start"),
            )
        })?;
        let value = self.b.const_u32(index as u32);
        self.b.store(self.pc, value);
        self.b.branch(self.switch_merge);
        Ok(())
    }

    fn emit_program(&mut self, flow: &Flow) -> Result<(), TranslateError> {
        self.handle_arrays = flow.handle_arrays.clone();
        if !self.is_fragment() {
            self.initialize_outputs(flow);
        }
        self.pc = self.b.local_variable(self.t.uint);
        let header = self.b.id();
        let dispatch = self.b.id();
        let continue_target = self.b.id();
        self.loop_merge = self.b.id();
        self.switch_merge = self.b.id();
        let default = self.b.id();
        let cases: Vec<Id> = flow.blocks.iter().map(|_| self.b.id()).collect();

        let zero = self.b.const_u32(0);
        self.b.store(self.pc, zero);
        if self.is_fragment() && flow.instructions.values().any(|ins| ins.name == "KIL") {
            let killed = self.b.local_variable(self.t.boolean);
            let alive = self.b.const_bool(false);
            self.b.store(killed, alive);
            self.killed = Some(killed);
        }
        self.b.branch(header);
        self.b.label(header);
        self.b.loop_merge(self.loop_merge, continue_target);
        self.b.branch(dispatch);
        self.b.label(dispatch);
        let selector = self.b.load(self.t.uint, self.pc);
        self.b.selection_merge(self.switch_merge);
        let targets: Vec<(u32, Id)> = cases
            .iter()
            .enumerate()
            .map(|(index, label)| (index as u32, *label))
            .collect();
        self.b.switch(selector, default, &targets);

        for (index, block) in flow.blocks.iter().enumerate() {
            self.b.label(cases[index]);
            let mut ended = false;
            for offset in &block.offsets {
                let instruction = flow.instructions[offset];
                if never(&instruction) {
                    continue;
                }
                ended = self.emit_instruction(flow, &instruction)?;
                if ended {
                    break;
                }
            }
            if !ended {
                match block.fallthrough {
                    Some(next) => {
                        let last = flow.instructions
                            [block.offsets.last().expect("block has instructions")];
                        self.goto(flow, next, &last)?;
                    }
                    None => self.emit_outputs_and_return(),
                }
            }
        }

        self.b.label(default);
        self.b.branch(self.loop_merge);
        self.b.label(self.switch_merge);
        self.b.branch(continue_target);
        self.b.label(continue_target);
        self.b.branch(header);
        self.b.label(self.loop_merge);
        if let Some(killed) = self.killed {
            let killed = self.b.load(self.t.boolean, killed);
            let (discard, end) = (self.b.id(), self.b.id());
            self.b.selection_merge(end);
            self.b.branch_conditional(killed, discard, end);
            self.b.label(discard);
            self.b.inst_void(op::KILL, &[]);
            self.b.label(end);
        }
        self.b.inst_void(op::RETURN, &[]);
        Ok(())
    }

    /// KIL: leave the dispatch loop and discard after it. OpKill inside the
    /// loop, reached by some of a quad's invocations while the others had
    /// returned, hung the Adreno 650 (LM3's alpha-tested shadow casters at
    /// the title screen ran for two seconds and lost the device).
    fn kill(&mut self) {
        match self.killed {
            Some(killed) => {
                let dead = self.b.const_bool(true);
                self.b.store(killed, dead);
                self.b.branch(self.loop_merge);
            }
            None => self.b.inst_void(op::KILL, &[]),
        }
    }

    /// Emit one instruction; true when it ended the block.
    fn emit_instruction(&mut self, flow: &Flow, ins: &Instruction) -> Result<bool, TranslateError> {
        let word = ins.word;
        let ins = *ins;
        match ins.name {
            "EXIT" => {
                if conditional(&ins) {
                    self.guarded(&ins, |t| {
                        t.emit_outputs_and_return();
                        Ok(true)
                    })?;
                    return Ok(false);
                }
                self.emit_outputs_and_return();
                return Ok(true);
            }
            "KIL" => {
                if conditional(&ins) {
                    self.guarded(&ins, |t| {
                        t.kill();
                        Ok(true)
                    })?;
                    return Ok(false);
                }
                self.kill();
                return Ok(true);
            }
            "BRA" | "SYNC" | "BRK" | "CONT" => {
                let target = if ins.name == "BRA" {
                    if field(word, 0, 5) != 0xF {
                        return Err(unsupported(&ins, "branch on a condition code"));
                    }
                    branch_target(ins.address, word)
                } else {
                    flow.sync_targets[&ins.address]
                };
                if conditional(&ins) {
                    let condition = self.get_p(ins.pred, ins.pred_inv);
                    let taken = *flow.block_at.get(&target).ok_or_else(|| {
                        unsupported(&ins, format!("branch to {target:#x}, not a block start"))
                    })?;
                    let next = next_slot(ins.address);
                    let fall = *flow
                        .block_at
                        .get(&next)
                        .ok_or_else(|| unsupported(&ins, "fall-through is not a block start"))?;
                    let taken = self.b.const_u32(taken as u32);
                    let fall = self.b.const_u32(fall as u32);
                    let index = self.select_u(condition, taken, fall);
                    self.b.store(self.pc, index);
                    self.b.branch(self.switch_merge);
                } else {
                    self.goto(flow, target, &ins)?;
                }
                return Ok(true);
            }
            "BRX" => {
                if field(word, 0, 5) != 0xF {
                    return Err(unsupported(&ins, "branch on a condition code"));
                }
                // The case whose offset register A holds; an offset not in
                // the table ends the program (the dispatch's default).
                let value = self.get_u(register(word, 8));
                let mut index = self.b.const_u32(u32::MAX);
                for (offset, target) in flow.jump_tables[&ins.address].iter().rev() {
                    let block = *flow.block_at.get(target).ok_or_else(|| {
                        unsupported(&ins, format!("branch to {target:#x}, not a block start"))
                    })?;
                    let offset = self.b.const_u32(*offset);
                    let matches = self.b.inst(op::I_EQUAL, self.t.boolean, &[value, offset]);
                    let block = self.b.const_u32(block as u32);
                    index = self.select_u(matches, block, index);
                }
                if conditional(&ins) {
                    let condition = self.get_p(ins.pred, ins.pred_inv);
                    let next = next_slot(ins.address);
                    let fall = *flow
                        .block_at
                        .get(&next)
                        .ok_or_else(|| unsupported(&ins, "fall-through is not a block start"))?;
                    let fall = self.b.const_u32(fall as u32);
                    index = self.select_u(condition, index, fall);
                }
                self.b.store(self.pc, index);
                self.b.branch(self.switch_merge);
                return Ok(true);
            }
            "SSY" | "PBK" | "PCNT" | "NOP" | "DEPBAR" => {}
            _ => self.guarded(&ins, |t| t.emit_operation(&ins).map(|()| false))?,
        }
        Ok(false)
    }

    /// An instruction that computes (no control flow).
    fn emit_operation(&mut self, ins: &Instruction) -> Result<(), TranslateError> {
        let word = ins.word;
        let rd = register(word, 0);
        let ra = register(word, 8);
        let rc = register(word, 39);
        match ins.name {
            "MOV" => {
                let value = match ins.src_b {
                    Operand::None => self.get_u(register(word, 20)),
                    _ => self.operand_b_u(ins)?,
                };
                self.set_u(rd, value);
            }
            "MOV32I" => {
                let value = self.b.const_u32(field(word, 20, 32));
                self.set_u(rd, value);
            }
            "RRO" => {
                // Range reduction feeds MUFU.SIN/COS/EX2, which take the plain
                // value here.
                let value = self.operand_b_f(ins)?;
                self.set_f(rd, value);
            }
            "FADD" => {
                let a = self.get_f(ra);
                let a = self.fmod(a, flag(word, 46), flag(word, 48));
                let b = self.operand_b_f(ins)?;
                let b = self.fmod(b, flag(word, 49), flag(word, 45));
                let sum = self.fadd(a, b);
                let value = self.saturate(sum, flag(word, 50));
                self.set_f(rd, value);
            }
            "FADD32I" => {
                let a = self.get_f(ra);
                let a = self.fmod(a, flag(word, 54), flag(word, 56));
                let b = self.b.const_f32(f32::from_bits(field(word, 20, 32)));
                let b = self.fmod(b, flag(word, 57), flag(word, 53));
                let value = self.fadd(a, b);
                self.set_f(rd, value);
            }
            "FMUL" => {
                let a = self.get_f(ra);
                let b = self.operand_b_f(ins)?;
                let b = self.fmod(b, false, flag(word, 48));
                let mut product = self.fmul(a, b);
                let scale = match field(word, 41, 3) {
                    0 => None,
                    1 => Some(0.5),
                    2 => Some(0.25),
                    3 => Some(0.125),
                    4 => Some(8.0),
                    5 => Some(4.0),
                    6 => Some(2.0),
                    _ => return Err(unsupported(ins, "FMUL scale 7")),
                };
                if let Some(scale) = scale {
                    let scale = self.b.const_f32(scale);
                    product = self.fmul(product, scale);
                }
                let value = self.saturate(product, flag(word, 50));
                self.set_f(rd, value);
            }
            "FMUL32I" => {
                let a = self.get_f(ra);
                let b = self.b.const_f32(f32::from_bits(field(word, 20, 32)));
                let product = self.fmul(a, b);
                let value = self.saturate(product, flag(word, 55));
                self.set_f(rd, value);
            }
            "FFMA" => {
                let a = self.get_f(ra);
                let (b, c) = match ins.src_b {
                    Operand::Register(_) => {
                        let b = self.get_f(register(word, 20));
                        (b, self.get_f(rc))
                    }
                    Operand::Constant { .. } => {
                        let b = self.cbuf_f(word);
                        (b, self.get_f(rc))
                    }
                    Operand::Immediate(_) => {
                        let b = self.operand_b_f(ins)?;
                        (b, self.get_f(rc))
                    }
                    // The RC form: B is register C, C comes from the constant buffer.
                    _ => {
                        let b = self.get_f(rc);
                        (b, self.cbuf_f(word))
                    }
                };
                let b = self.fmod(b, false, flag(word, 48));
                let c = self.fmod(c, false, flag(word, 49));
                let fused = self.b.ext(self.t.float, glsl::FMA, &[a, b, c]);
                let value = self.saturate(fused, flag(word, 50));
                self.set_f(rd, value);
            }
            "FMNMX" => {
                let a = self.get_f(ra);
                let a = self.fmod(a, flag(word, 46), flag(word, 48));
                let b = self.operand_b_f(ins)?;
                let b = self.fmod(b, flag(word, 49), flag(word, 45));
                let min = self.get_p(field(word, 39, 3) as u8, flag(word, 42));
                // NaN-aware: the other operand wins, as the hardware does.
                let low = self.b.ext(self.t.float, glsl::N_MIN, &[a, b]);
                let high = self.b.ext(self.t.float, glsl::N_MAX, &[a, b]);
                let value = self.b.inst(op::SELECT, self.t.float, &[min, low, high]);
                self.set_f(rd, value);
            }
            "LDC" => {
                // A constant-buffer load at register A plus a signed offset.
                if field(word, 44, 2) != 0 {
                    return Err(unsupported(ins, "LDC address modes"));
                }
                let slot = field(word, 36, 5);
                let offset = self
                    .b
                    .const_u32(field(word, 20, 16) as u16 as i16 as i32 as u32);
                let base = self.get_u(ra);
                let address = self.b.inst(op::I_ADD, self.t.uint, &[base, offset]);
                let size = field(word, 48, 3);
                match size {
                    4 => {
                        let value = self.cbuf_u_at(slot, address);
                        self.set_u(rd, value);
                    }
                    5 => {
                        let low = self.cbuf_u_at(slot, address);
                        let four = self.b.const_u32(4);
                        let next = self.b.inst(op::I_ADD, self.t.uint, &[address, four]);
                        let high = self.cbuf_u_at(slot, next);
                        self.set_u(rd, low);
                        self.set_u(rd.wrapping_add(1), high);
                    }
                    0..=3 => {
                        let value = self.cbuf_u_at(slot, address);
                        let three = self.b.const_u32(3);
                        let byte = self.b.inst(op::BITWISE_AND, self.t.uint, &[address, three]);
                        let shift =
                            self.b
                                .inst(op::SHIFT_LEFT_LOGICAL, self.t.uint, &[byte, three]);
                        let bits = self.b.const_u32(if size < 2 { 8 } else { 16 });
                        let extract = if size % 2 == 1 {
                            op::BIT_FIELD_S_EXTRACT
                        } else {
                            op::BIT_FIELD_U_EXTRACT
                        };
                        let value = self.b.inst(extract, self.t.uint, &[value, shift, bits]);
                        self.set_u(rd, value);
                    }
                    _ => return Err(unsupported(ins, "LDC size")),
                }
            }
            "SEL" => {
                // A when the predicate holds, else B.
                let a = self.get_u(ra);
                let b = self.operand_b_u(ins)?;
                let choose_a = self.get_p(field(word, 39, 3) as u8, flag(word, 42));
                let value = self.b.inst(op::SELECT, self.t.uint, &[choose_a, a, b]);
                self.set_u(rd, value);
            }
            "FLO" => {
                // The position of operand B's highest set bit (signed: the
                // highest bit unlike the sign), -1 when there is none;
                // inverted first with bit 40, and as a shift (31 - position)
                // with bit 41.
                let mut b = self.operand_b_u(ins)?;
                if flag(word, 40) {
                    b = self.b.inst(op::NOT, self.t.uint, &[b]);
                }
                let find = if flag(word, 48) {
                    glsl::FIND_S_MSB
                } else {
                    glsl::FIND_U_MSB
                };
                let mut value = self.b.ext(self.t.uint, find, &[b]);
                if flag(word, 41) {
                    let none = self.b.const_u32(u32::MAX);
                    let found = self.b.inst(op::I_NOT_EQUAL, self.t.boolean, &[value, none]);
                    let thirty_one = self.b.const_u32(31);
                    let shift = self.b.inst(op::I_SUB, self.t.uint, &[thirty_one, value]);
                    value = self.b.inst(op::SELECT, self.t.uint, &[found, shift, none]);
                }
                self.set_u(rd, value);
            }
            "IMNMX" => {
                let a = self.get_u(ra);
                let b = self.operand_b_u(ins)?;
                let min = self.get_p(field(word, 39, 3) as u8, flag(word, 42));
                let (low, high) = if flag(word, 48) {
                    (glsl::S_MIN, glsl::S_MAX)
                } else {
                    (glsl::U_MIN, glsl::U_MAX)
                };
                let low = self.b.ext(self.t.uint, low, &[a, b]);
                let high = self.b.ext(self.t.uint, high, &[a, b]);
                let value = self.b.inst(op::SELECT, self.t.uint, &[min, low, high]);
                self.set_u(rd, value);
            }
            "MUFU" => {
                let x = self.get_f(ra);
                let x = self.fmod(x, flag(word, 46), flag(word, 48));
                let one = self.b.const_f32(1.0);
                let value = match field(word, 20, 4) {
                    0 => self.b.ext(self.t.float, glsl::COS, &[x]),
                    1 => self.b.ext(self.t.float, glsl::SIN, &[x]),
                    2 => self.b.ext(self.t.float, glsl::EXP2, &[x]),
                    3 => self.b.ext(self.t.float, glsl::LOG2, &[x]),
                    4 => self.b.inst(op::F_DIV, self.t.float, &[one, x]),
                    5 => self.b.ext(self.t.float, glsl::INVERSE_SQRT, &[x]),
                    8 => self.b.ext(self.t.float, glsl::SQRT, &[x]),
                    other => return Err(unsupported(ins, format!("MUFU operation {other}"))),
                };
                let value = self.saturate(value, flag(word, 50));
                self.set_f(rd, value);
            }
            "PSETP" => {
                // PSETP: combine three predicates with two boolean ops and
                // write the result and its inverse. LM3's save screen uses
                // it; without it the whole draw was skipped.
                let a = self.get_p(field(word, 12, 3) as u8, flag(word, 15));
                let b = self.get_p(field(word, 29, 3) as u8, flag(word, 32));
                let c = self.get_p(field(word, 39, 3) as u8, flag(word, 42));
                let (first, _) = self.combine(ins, field(word, 24, 2), a, b)?;
                let (first, second) = self.combine(ins, field(word, 45, 2), first, c)?;
                self.set_p(field(word, 3, 3) as u8, first);
                self.set_p(field(word, 0, 3) as u8, second);
            }
            "FSETP" => {
                let a = self.get_f(ra);
                let a = self.fmod(a, flag(word, 7), flag(word, 43));
                let b = self.operand_b_f(ins)?;
                let b = self.fmod(b, flag(word, 44), flag(word, 6));
                let compare = self.float_compare(ins, field(word, 48, 4), a, b)?;
                let other = self.get_p(field(word, 39, 3) as u8, flag(word, 42));
                let (first, second) = self.combine(ins, field(word, 45, 2), compare, other)?;
                self.set_p(field(word, 3, 3) as u8, first);
                self.set_p(field(word, 0, 3) as u8, second);
            }
            "FSET" => {
                let a = self.get_f(ra);
                let a = self.fmod(a, flag(word, 54), flag(word, 43));
                let b = self.operand_b_f(ins)?;
                let b = self.fmod(b, flag(word, 44), flag(word, 53));
                let compare = self.float_compare(ins, field(word, 48, 4), a, b)?;
                let other = self.get_p(field(word, 39, 3) as u8, flag(word, 42));
                let (result, _) = self.combine(ins, field(word, 45, 2), compare, other)?;
                let value = self.bool_to_value(result, flag(word, 52));
                self.set_u(rd, value);
            }
            "ISETP" => {
                let a = self.get_u(ra);
                let b = self.operand_b_u(ins)?;
                let compare = self.int_compare(ins, field(word, 49, 3), flag(word, 48), a, b)?;
                let other = self.get_p(field(word, 39, 3) as u8, flag(word, 42));
                let (first, second) = self.combine(ins, field(word, 45, 2), compare, other)?;
                self.set_p(field(word, 3, 3) as u8, first);
                self.set_p(field(word, 0, 3) as u8, second);
            }
            "ISET" => {
                let a = self.get_u(ra);
                let b = self.operand_b_u(ins)?;
                let compare = self.int_compare(ins, field(word, 49, 3), flag(word, 48), a, b)?;
                let other = self.get_p(field(word, 39, 3) as u8, flag(word, 42));
                let (result, _) = self.combine(ins, field(word, 45, 2), compare, other)?;
                let value = self.bool_to_value(result, flag(word, 44));
                self.set_u(rd, value);
            }
            "IADD" => {
                let mut a = self.get_u(ra);
                let mut b = self.operand_b_u(ins)?;
                let (neg_b, neg_a) = (flag(word, 48), flag(word, 49));
                if neg_a && neg_b {
                    let one = self.b.const_u32(1);
                    b = self.b.inst(op::I_ADD, self.t.uint, &[b, one]);
                } else {
                    if neg_a {
                        a = self.b.inst(op::S_NEGATE, self.t.uint, &[a]);
                    }
                    if neg_b {
                        b = self.b.inst(op::S_NEGATE, self.t.uint, &[b]);
                    }
                }
                let value = self.b.inst(op::I_ADD, self.t.uint, &[a, b]);
                self.set_u(rd, value);
            }
            "XMAD" => self.xmad(ins)?,
            "IADD32I" => {
                let mut a = self.get_u(ra);
                if flag(word, 56) {
                    a = self.b.inst(op::S_NEGATE, self.t.uint, &[a]);
                }
                let b = self.b.const_u32(field(word, 20, 32));
                let value = self.b.inst(op::I_ADD, self.t.uint, &[a, b]);
                self.set_u(rd, value);
            }
            "ISCADD" => {
                let mut a = self.get_u(ra);
                let mut b = self.operand_b_u(ins)?;
                let (neg_b, neg_a) = (flag(word, 48), flag(word, 49));
                if neg_a && neg_b {
                    let one = self.b.const_u32(1);
                    b = self.b.inst(op::I_ADD, self.t.uint, &[b, one]);
                } else {
                    if neg_a {
                        a = self.b.inst(op::S_NEGATE, self.t.uint, &[a]);
                    }
                    if neg_b {
                        b = self.b.inst(op::S_NEGATE, self.t.uint, &[b]);
                    }
                }
                let shift = self.b.const_u32(field(word, 39, 5));
                let scaled = self
                    .b
                    .inst(op::SHIFT_LEFT_LOGICAL, self.t.uint, &[a, shift]);
                let value = self.b.inst(op::I_ADD, self.t.uint, &[scaled, b]);
                self.set_u(rd, value);
            }
            "LOP" => {
                // LOP (register, constant or immediate B): a logic operation,
                // and optionally a predicate saying whether the result is
                // zero.
                let mut a = self.get_u(ra);
                let mut b = self.operand_b_u(ins)?;
                if flag(word, 39) {
                    a = self.b.inst(op::NOT, self.t.uint, &[a]);
                }
                if flag(word, 40) {
                    b = self.b.inst(op::NOT, self.t.uint, &[b]);
                }
                if flag(word, 43) {
                    return Err(unsupported(ins, "LOP.X"));
                }
                let value = match field(word, 41, 2) {
                    0 => self.b.inst(op::BITWISE_AND, self.t.uint, &[a, b]),
                    1 => self.b.inst(op::BITWISE_OR, self.t.uint, &[a, b]),
                    2 => self.b.inst(op::BITWISE_XOR, self.t.uint, &[a, b]),
                    _ => b,
                };
                let predicate = field(word, 48, 3) as u8;
                if predicate != PREDICATE_TRUE {
                    let zero = self.b.const_u32(0);
                    let result = match field(word, 44, 2) {
                        0 => self.b.const_bool(false),
                        1 => self.b.const_bool(true),
                        2 => self.b.inst(op::I_EQUAL, self.t.boolean, &[value, zero]),
                        _ => self.b.inst(op::I_NOT_EQUAL, self.t.boolean, &[value, zero]),
                    };
                    self.set_p(predicate, result);
                }
                self.set_u(rd, value);
            }
            "S2R" => {
                // Special registers: graphics programs read the ones below;
                // thread and block ids are compute's.
                let register = field(word, 20, 8);
                let value = match register {
                    // SR_TID_X..Z and SR_CTAID_X..Z.
                    0x21..=0x23 if self.is_compute() => {
                        self.invocation_id(built_in::LOCAL_INVOCATION_ID, register - 0x21)
                    }
                    0x25..=0x27 if self.is_compute() => {
                        self.invocation_id(built_in::WORKGROUP_ID, register - 0x25)
                    }
                    // SR_TID: x in bits 0:15, y in 16:25, z in 26:31.
                    0x20 if self.is_compute() => {
                        let x = self.invocation_id(built_in::LOCAL_INVOCATION_ID, 0);
                        let y = self.invocation_id(built_in::LOCAL_INVOCATION_ID, 1);
                        let z = self.invocation_id(built_in::LOCAL_INVOCATION_ID, 2);
                        let (sixteen, twenty_six) = (self.b.const_u32(16), self.b.const_u32(26));
                        let y = self
                            .b
                            .inst(op::SHIFT_LEFT_LOGICAL, self.t.uint, &[y, sixteen]);
                        let z = self
                            .b
                            .inst(op::SHIFT_LEFT_LOGICAL, self.t.uint, &[z, twenty_six]);
                        let xy = self.b.inst(op::BITWISE_OR, self.t.uint, &[x, y]);
                        self.b.inst(op::BITWISE_OR, self.t.uint, &[xy, z])
                    }
                    // SR_LANEID, SR_THREAD_KILL, SR_TID*, SR_CTAID*, the
                    // clocks: a single invocation, lane 0, not a helper.
                    0x00 | 0x13 | 0x20..=0x23 | 0x25..=0x27 | 0x50..=0x53 => self.b.const_u32(0),
                    // SR_Y_DIRECTION: +1.0, y is not flipped.
                    0x12 => self.b.const_u32(1f32.to_bits()),
                    // SR_EQMASK, SR_LEMASK: lane 0's bit; LT, GT, GE: the
                    // others for one lane.
                    0x38 | 0x3A | 0x3C => self.b.const_u32(1),
                    0x39 | 0x3B => self.b.const_u32(0),
                    other => {
                        return Err(unsupported(
                            ins,
                            format!("S2R of special register {other:#x}"),
                        ))
                    }
                };
                self.set_u(rd, value);
            }
            "BAR" => {
                // BAR.SYNC and BAR.SYNCALL wait for the block; arrivals
                // (which only count) have nothing to wait for here.
                if matches!(field(word, 32, 3), 0 | 4) {
                    let workgroup = self.b.const_u32(scope::WORKGROUP);
                    let semantics = self.b.const_u32(
                        semantics::ACQUIRE_RELEASE
                            | semantics::WORKGROUP_MEMORY
                            | semantics::IMAGE_MEMORY,
                    );
                    self.b
                        .inst_void(op::CONTROL_BARRIER, &[workgroup, workgroup, semantics]);
                }
            }
            "LDS" | "STS" => {
                // Shared memory at register A plus a signed 24-bit offset.
                let offset = self
                    .b
                    .const_u32(((field(word, 20, 24) << 8) as i32 >> 8) as u32);
                let base = self.get_u(ra);
                let address = self.b.inst(op::I_ADD, self.t.uint, &[base, offset]);
                let words = match field(word, 48, 3) {
                    4 => 1,
                    5 => 2,
                    6 => 4,
                    _ => return Err(unsupported(ins, "sub-word shared memory access")),
                };
                for index in 0..words {
                    let four = self.b.const_u32(4 * index);
                    let at = self.b.inst(op::I_ADD, self.t.uint, &[address, four]);
                    let pointer = self.shared_word(at);
                    let register = rd.wrapping_add(index as u8);
                    if ins.name == "LDS" {
                        let value = self.b.load(self.t.uint, pointer);
                        self.set_u(register, value);
                    } else {
                        let value = self.get_u(register);
                        self.b.store(pointer, value);
                    }
                }
            }
            "LDL" | "STL" => {
                // Local memory at register A plus a signed 24-bit offset;
                // sizes 0-3 are U8, S8, U16, S16 in the addressed word.
                let offset = self
                    .b
                    .const_u32(((field(word, 20, 24) << 8) as i32 >> 8) as u32);
                let base = self.get_u(ra);
                let address = self.b.inst(op::I_ADD, self.t.uint, &[base, offset]);
                let size = field(word, 48, 3);
                if size <= 3 {
                    let bits = self.b.const_u32(if size <= 1 { 8 } else { 16 });
                    let three = self.b.const_u32(3);
                    let low = self.b.inst(op::BITWISE_AND, self.t.uint, &[address, three]);
                    let shift = self
                        .b
                        .inst(op::SHIFT_LEFT_LOGICAL, self.t.uint, &[low, three]);
                    let pointer = self.local_word(address);
                    let current = self.b.load(self.t.uint, pointer);
                    if ins.name == "LDL" {
                        let value = if size % 2 == 1 {
                            let current = self.b.bitcast(self.t.int, current);
                            let value = self.b.inst(
                                op::BIT_FIELD_S_EXTRACT,
                                self.t.int,
                                &[current, shift, bits],
                            );
                            self.b.bitcast(self.t.uint, value)
                        } else {
                            self.b.inst(
                                op::BIT_FIELD_U_EXTRACT,
                                self.t.uint,
                                &[current, shift, bits],
                            )
                        };
                        self.set_u(rd, value);
                    } else {
                        let value = self.get_u(rd);
                        let merged = self.b.inst(
                            op::BIT_FIELD_INSERT,
                            self.t.uint,
                            &[current, value, shift, bits],
                        );
                        self.b.store(pointer, merged);
                    }
                } else {
                    let words = match size {
                        4 => 1,
                        5 => 2,
                        _ => 4,
                    };
                    for index in 0..words {
                        let four = self.b.const_u32(4 * index);
                        let at = self.b.inst(op::I_ADD, self.t.uint, &[address, four]);
                        let pointer = self.local_word(at);
                        let register = rd.wrapping_add(index as u8);
                        if ins.name == "LDL" {
                            let value = self.b.load(self.t.uint, pointer);
                            self.set_u(register, value);
                        } else {
                            let value = self.get_u(register);
                            self.b.store(pointer, value);
                        }
                    }
                }
            }
            "LOP32I" => {
                let mut a = self.get_u(ra);
                let mut b = self.b.const_u32(field(word, 20, 32));
                if flag(word, 55) {
                    a = self.b.inst(op::NOT, self.t.uint, &[a]);
                }
                if flag(word, 56) {
                    b = self.b.inst(op::NOT, self.t.uint, &[b]);
                }
                let value = match field(word, 53, 2) {
                    0 => self.b.inst(op::BITWISE_AND, self.t.uint, &[a, b]),
                    1 => self.b.inst(op::BITWISE_OR, self.t.uint, &[a, b]),
                    2 => self.b.inst(op::BITWISE_XOR, self.t.uint, &[a, b]),
                    _ => b,
                };
                self.set_u(rd, value);
            }
            "BFE" => {
                let mut base = self.get_u(ra);
                if flag(word, 40) {
                    base = self.b.inst(op::BIT_REVERSE, self.t.uint, &[base]);
                }
                let spec = self.operand_b_u(ins)?;
                let value = self.bitfield_extract(base, spec, flag(word, 48));
                self.set_u(rd, value);
            }
            "BFI" => {
                // Operand B (or register C in the RC form) is the field spec;
                // the base is register C or the constant buffer.
                let (spec, base) = match ins.src_b {
                    Operand::Register(_) => (self.get_u(register(word, 20)), self.get_u(rc)),
                    Operand::Immediate(_) | Operand::Constant { .. } => {
                        let spec = self.operand_b_u(ins)?;
                        (spec, self.get_u(rc))
                    }
                    _ => {
                        let spec = self.get_u(rc);
                        let base = self.cbuf_u(field(word, 34, 5), field(word, 20, 14) * 4);
                        (spec, base)
                    }
                };
                let insert = self.get_u(ra);
                let value = self.bitfield_insert(base, insert, spec);
                self.set_u(rd, value);
            }
            "F2I" => {
                if field(word, 10, 2) != 2 {
                    return Err(unsupported(ins, "F2I from a non-32-bit float"));
                }
                let x = self.operand_b_f(ins)?;
                let x = self.fmod(x, flag(word, 49), flag(word, 45));
                let rounded = match field(word, 39, 2) {
                    0 => self.b.ext(self.t.float, glsl::ROUND_EVEN, &[x]),
                    1 => self.b.ext(self.t.float, glsl::FLOOR, &[x]),
                    2 => self.b.ext(self.t.float, glsl::CEIL, &[x]),
                    _ => self.b.ext(self.t.float, glsl::TRUNC, &[x]),
                };
                let signed = flag(word, 12);
                let (low, high) = match (field(word, 8, 2), signed) {
                    (1, true) => (-32768.0, 32767.0),
                    (1, false) => (0.0, 65535.0),
                    (2, true) => (-2_147_483_648.0, 2_147_483_520.0),
                    (2, false) => (0.0, 4_294_967_040.0),
                    _ => return Err(unsupported(ins, "F2I to a 64-bit integer")),
                };
                let low = self.b.const_f32(low);
                let high = self.b.const_f32(high);
                let clamped = self
                    .b
                    .ext(self.t.float, glsl::F_CLAMP, &[rounded, low, high]);
                let converted = if signed {
                    let value = self.b.inst(op::CONVERT_F_TO_S, self.t.int, &[clamped]);
                    self.b.bitcast(self.t.uint, value)
                } else {
                    self.b.inst(op::CONVERT_F_TO_U, self.t.uint, &[clamped])
                };
                // NaN converts to zero.
                let nan = self.b.inst(op::IS_NAN, self.t.boolean, &[x]);
                let zero = self.b.const_u32(0);
                let value = self.select_u(nan, zero, converted);
                self.set_u(rd, value);
            }
            "HADD2" | "HMUL2" | "HFMA2" | "HADD232I" | "HMUL232I" => self.half_arithmetic(ins)?,
            "F2F" => {
                // Between 16- and 32-bit floats (a half is a register's low
                // 16 bits, or its high ones with bit 41), or rounding to an
                // integral value at the same size (bit 42; the mode in bits
                // 39:40).
                let (destination, source) = (field(word, 8, 2), field(word, 10, 2));
                if !matches!(destination, 1 | 2) || !matches!(source, 1 | 2) {
                    return Err(unsupported(ins, "F2F with a 64-bit float"));
                }
                let x = if source == 1 {
                    let bits = self.operand_b_u(ins)?;
                    let pair = self.b.ext(self.t.vec2, glsl::UNPACK_HALF_2X16, &[bits]);
                    let half = u32::from(flag(word, 41));
                    self.b
                        .inst(op::COMPOSITE_EXTRACT, self.t.float, &[pair, half])
                } else {
                    self.operand_b_f(ins)?
                };
                let mut x = self.fmod(x, flag(word, 49), flag(word, 45));
                if source == destination && flag(word, 42) {
                    let rounding = [glsl::ROUND_EVEN, glsl::FLOOR, glsl::CEIL, glsl::TRUNC];
                    x = self
                        .b
                        .ext(self.t.float, rounding[field(word, 39, 2) as usize], &[x]);
                }
                let x = self.saturate(x, flag(word, 50));
                if destination == 1 {
                    let zero = self.b.const_f32(0.0);
                    let pair = self.vec(&[x, zero]);
                    let bits = self.b.ext(self.t.uint, glsl::PACK_HALF_2X16, &[pair]);
                    self.set_u(rd, bits);
                } else {
                    self.set_f(rd, x);
                }
            }
            "I2F" => {
                if field(word, 8, 2) != 2 {
                    return Err(unsupported(ins, "I2F to a non-32-bit float"));
                }
                let mut value = self.operand_b_u(ins)?;
                let signed = flag(word, 13);
                let width = match field(word, 10, 2) {
                    0 => 8,
                    1 => 16,
                    2 => 32,
                    _ => return Err(unsupported(ins, "I2F from a 64-bit integer")),
                };
                if width < 32 {
                    let offset = self.b.const_u32(field(word, 41, 2) * width);
                    let count = self.b.const_u32(width);
                    let opcode = if signed {
                        op::BIT_FIELD_S_EXTRACT
                    } else {
                        op::BIT_FIELD_U_EXTRACT
                    };
                    value = self.b.inst(opcode, self.t.uint, &[value, offset, count]);
                }
                let converted = if signed {
                    let signed_value = self.b.bitcast(self.t.int, value);
                    self.b
                        .inst(op::CONVERT_S_TO_F, self.t.float, &[signed_value])
                } else {
                    self.b.inst(op::CONVERT_U_TO_F, self.t.float, &[value])
                };
                let value = self.fmod(converted, flag(word, 49), flag(word, 45));
                self.set_f(rd, value);
            }
            "IPA" => {
                if !self.is_fragment() {
                    return Err(unsupported(ins, "IPA outside a fragment program"));
                }
                if flag(word, 38) && register(word, 8) != REGISTER_ZERO {
                    return Err(unsupported(ins, "indexed IPA"));
                }
                let address = field(word, 30, 8) * 4;
                let bits = self.read_attribute(ins, address)?;
                let mut value = self.b.bitcast(self.t.float, bits);
                if (0x80..0x280).contains(&address) {
                    let imap = self.header.imap[((address - 0x80) / 16) as usize];
                    let mode = [imap.x, imap.y, imap.z, imap.w][((address / 4) % 4) as usize];
                    if mode == PixelImap::Perspective {
                        // The hardware hands back attribute / w here; the
                        // program multiplies by w itself.
                        let frag_coord =
                            self.builtin(built_in::FRAG_COORD, self.t.vec4, storage::INPUT);
                        let pointer =
                            self.component_pointer(frag_coord, storage::INPUT, self.t.float, 3);
                        let inverse_w = self.b.load(self.t.float, pointer);
                        value = self.fmul(value, inverse_w);
                    }
                }
                match field(word, 54, 2) {
                    0 => {}
                    1 => {
                        let multiplier = self.get_f(register(word, 20));
                        value = self.fmul(value, multiplier);
                    }
                    other => {
                        return Err(unsupported(ins, format!("IPA interpolation mode {other}")))
                    }
                }
                let value = self.saturate(value, flag(word, 51));
                self.set_f(rd, value);
            }
            "ALD" => {
                if register(word, 8) != REGISTER_ZERO || flag(word, 32) || flag(word, 31) {
                    return Err(unsupported(ins, "indexed, output or patch ALD"));
                }
                let base = field(word, 20, 10);
                for component in 0..=field(word, 47, 2) {
                    let bits = self.read_attribute(ins, base + component * 4)?;
                    self.set_u(rd.wrapping_add(component as u8), bits);
                }
            }
            "AST" => {
                if register(word, 8) != REGISTER_ZERO || flag(word, 31) {
                    return Err(unsupported(ins, "indexed or patch AST"));
                }
                let base = field(word, 20, 10);
                for component in 0..=field(word, 47, 2) {
                    let bits = self.get_u(rd.wrapping_add(component as u8));
                    self.write_attribute(ins, base + component * 4, bits)?;
                }
            }
            "TEXS" => self.texs(ins)?,
            "TXQ" => self.txq(ins)?,
            "TLDS" => self.tlds(ins)?,
            "TLD4S" => self.tld4s(ins, false)?,
            "TLD4SF16" => self.tld4s(ins, true)?,
            "TEX" => self.tex(ins, false)?,
            "TEXB" => self.tex(ins, true)?,
            "TXD" => self.txd(ins)?,
            "LOP3" => {
                // A three-input logic function given by its truth table: bit
                // (a << 2 | b << 1 | c) of the table is the result for those
                // input bits.
                let register_form = matches!(ins.src_b, Operand::Register(_));
                let table = if register_form {
                    field(word, 28, 8)
                } else {
                    field(word, 48, 8)
                };
                let a = self.get_u(ra);
                let b = self.operand_b_u(ins)?;
                let c = self.get_u(rc);
                let not = |t: &mut Self, value: Id| t.b.inst(op::NOT, t.t.uint, &[value]);
                let (na, nb, nc) = (not(self, a), not(self, b), not(self, c));
                let mut value = self.b.const_u32(0);
                for minterm in 0..8u32 {
                    if table & (1 << minterm) == 0 {
                        continue;
                    }
                    let pick =
                        |bit: u32, yes: Id, no: Id| if minterm & bit != 0 { yes } else { no };
                    let ab = self.b.inst(
                        op::BITWISE_AND,
                        self.t.uint,
                        &[pick(4, a, na), pick(2, b, nb)],
                    );
                    let term = self
                        .b
                        .inst(op::BITWISE_AND, self.t.uint, &[ab, pick(1, c, nc)]);
                    value = self.b.inst(op::BITWISE_OR, self.t.uint, &[value, term]);
                }
                if register_form {
                    let predicate = field(word, 48, 3) as u8;
                    if predicate != PREDICATE_TRUE {
                        let zero = self.b.const_u32(0);
                        let result = match field(word, 36, 2) {
                            0 => self.b.const_bool(false),
                            1 => self.b.const_bool(true),
                            2 => self.b.inst(op::I_EQUAL, self.t.boolean, &[value, zero]),
                            _ => self.b.inst(op::I_NOT_EQUAL, self.t.boolean, &[value, zero]),
                        };
                        self.set_p(predicate, result);
                    }
                }
                self.set_u(rd, value);
            }
            "TLD4" => self.tld4(ins, false)?,
            "TLD4B" => self.tld4(ins, true)?,
            "SHFL" => self.shfl(ins)?,
            "FSWZADD" => self.fswzadd(ins)?,
            "SHR" | "SHL" => {
                // Shifts by register, constant or immediate: the amount is
                // masked to 0..31 (.W), or else 32 and more shift everything
                // out (the sign for an arithmetic right shift).
                let left = ins.name == "SHL";
                let signed = !left && flag(word, 48);
                if field(word, 43, 2) != 0 {
                    return Err(unsupported(ins, "extended shift"));
                }
                let mut a = self.get_u(ra);
                if !left && flag(word, 40) {
                    a = self.b.inst(op::BIT_REVERSE, self.t.uint, &[a]);
                }
                let mut amount = self.operand_b_u(ins)?;
                let wrap = flag(word, 39);
                if wrap {
                    let mask = self.b.const_u32(31);
                    amount = self.b.inst(op::BITWISE_AND, self.t.uint, &[amount, mask]);
                }
                let shift = match (left, signed) {
                    (true, _) => op::SHIFT_LEFT_LOGICAL,
                    (false, true) => op::SHIFT_RIGHT_ARITHMETIC,
                    (false, false) => op::SHIFT_RIGHT_LOGICAL,
                };
                let mut value = self.b.inst(shift, self.t.uint, &[a, amount]);
                if !wrap {
                    let limit = self.b.const_u32(32);
                    let in_range = self
                        .b
                        .inst(op::U_LESS_THAN, self.t.boolean, &[amount, limit]);
                    let out = if signed {
                        let sign = self.b.const_u32(31);
                        self.b
                            .inst(op::SHIFT_RIGHT_ARITHMETIC, self.t.uint, &[a, sign])
                    } else {
                        self.b.const_u32(0)
                    };
                    value = self.select_u(in_range, value, out);
                }
                self.set_u(rd, value);
            }
            _ => return Err(unsupported(ins, "not translated yet")),
        }
        Ok(())
    }

    fn float_compare(
        &mut self,
        ins: &Instruction,
        compare: u32,
        a: Id,
        b: Id,
    ) -> Result<Id, TranslateError> {
        let opcode = match compare {
            0 => return Ok(self.b.const_bool(false)),
            1 => op::F_ORD_LESS_THAN,
            2 => op::F_ORD_EQUAL,
            3 => op::F_ORD_LESS_THAN_EQUAL,
            4 => op::F_ORD_GREATER_THAN,
            5 => op::F_ORD_NOT_EQUAL,
            6 => op::F_ORD_GREATER_THAN_EQUAL,
            7 | 8 => {
                // NUM (both ordered) and NAN (either unordered).
                let a_nan = self.b.inst(op::IS_NAN, self.t.boolean, &[a]);
                let b_nan = self.b.inst(op::IS_NAN, self.t.boolean, &[b]);
                let any = self.b.inst(op::LOGICAL_OR, self.t.boolean, &[a_nan, b_nan]);
                return Ok(if compare == 8 {
                    any
                } else {
                    self.b.inst(op::LOGICAL_NOT, self.t.boolean, &[any])
                });
            }
            9 => op::F_UNORD_LESS_THAN,
            10 => op::F_UNORD_EQUAL,
            11 => op::F_UNORD_LESS_THAN_EQUAL,
            12 => op::F_UNORD_GREATER_THAN,
            13 => op::F_UNORD_NOT_EQUAL,
            14 => op::F_UNORD_GREATER_THAN_EQUAL,
            15 => return Ok(self.b.const_bool(true)),
            _ => return Err(unsupported(ins, "compare operation")),
        };
        Ok(self.b.inst(opcode, self.t.boolean, &[a, b]))
    }

    fn int_compare(
        &mut self,
        ins: &Instruction,
        compare: u32,
        signed: bool,
        a: Id,
        b: Id,
    ) -> Result<Id, TranslateError> {
        let opcode = match (compare, signed) {
            (0, _) => return Ok(self.b.const_bool(false)),
            (7, _) => return Ok(self.b.const_bool(true)),
            (1, true) => op::S_LESS_THAN,
            (1, false) => op::U_LESS_THAN,
            (2, _) => op::I_EQUAL,
            (3, true) => op::S_LESS_THAN_EQUAL,
            (3, false) => op::U_LESS_THAN_EQUAL,
            (4, true) => op::S_GREATER_THAN,
            (4, false) => op::U_GREATER_THAN,
            (5, _) => op::I_NOT_EQUAL,
            (6, true) => op::S_GREATER_THAN_EQUAL,
            (6, false) => op::U_GREATER_THAN_EQUAL,
            _ => return Err(unsupported(ins, "compare operation")),
        };
        Ok(self.b.inst(opcode, self.t.boolean, &[a, b]))
    }

    /// Combine a comparison with a predicate (AND/OR/XOR): the result, and
    /// the result for the negated comparison.
    fn combine(
        &mut self,
        ins: &Instruction,
        operation: u32,
        compare: Id,
        other: Id,
    ) -> Result<(Id, Id), TranslateError> {
        let opcode = match operation {
            0 => op::LOGICAL_AND,
            1 => op::LOGICAL_OR,
            2 => op::LOGICAL_NOT_EQUAL,
            _ => return Err(unsupported(ins, "boolean operation")),
        };
        let inverted = self.b.inst(op::LOGICAL_NOT, self.t.boolean, &[compare]);
        let first = self.b.inst(opcode, self.t.boolean, &[compare, other]);
        let second = self.b.inst(opcode, self.t.boolean, &[inverted, other]);
        Ok((first, second))
    }

    /// A boolean as a register value: 1.0f with `float`, else all ones.
    fn bool_to_value(&mut self, value: Id, float: bool) -> Id {
        let on = if float {
            self.b.const_u32(1.0f32.to_bits())
        } else {
            self.b.const_u32(u32::MAX)
        };
        let off = self.b.const_u32(0);
        self.select_u(value, on, off)
    }

    /// `BFE`: `count` bits (spec bits 8:15) from `offset` (bits 0:7).
    fn bitfield_extract(&mut self, base: Id, spec: Id, signed: bool) -> Id {
        let u = self.t.uint;
        let eight = self.b.const_u32(8);
        let zero = self.b.const_u32(0);
        let max = self.b.const_u32(32);
        let offset = self
            .b
            .inst(op::BIT_FIELD_U_EXTRACT, u, &[spec, zero, eight]);
        let count = self
            .b
            .inst(op::BIT_FIELD_U_EXTRACT, u, &[spec, eight, eight]);
        // Clamp so the SPIR-V extract stays defined (offset + count <= 32).
        let offset_c = self.b.ext(u, glsl::U_MIN, &[offset, max]);
        let room = self.b.inst(op::I_SUB, u, &[max, offset_c]);
        let count_c = self.b.ext(u, glsl::U_MIN, &[count, room]);
        let opcode = if signed {
            op::BIT_FIELD_S_EXTRACT
        } else {
            op::BIT_FIELD_U_EXTRACT
        };
        let mut result = self.b.inst(opcode, u, &[base, offset_c, count_c]);
        if signed {
            // Past bit 31 the sign bit repeats.
            let signed_base = self.b.bitcast(self.t.int, base);
            let zero_i = self.b.const_i32(0);
            let negative = self
                .b
                .inst(op::S_LESS_THAN, self.t.boolean, &[signed_base, zero_i]);
            let ones = self.b.const_u32(u32::MAX);
            let fill = self.select_u(negative, ones, zero);
            let beyond = self
                .b
                .inst(op::U_GREATER_THAN_EQUAL, self.t.boolean, &[offset, max]);
            result = self.select_u(beyond, fill, result);
        }
        let empty = self.b.inst(op::I_EQUAL, self.t.boolean, &[count, zero]);
        self.select_u(empty, zero, result)
    }

    /// `BFI`: insert `insert`'s low bits into `base` at the spec's field.
    fn bitfield_insert(&mut self, base: Id, insert: Id, spec: Id) -> Id {
        let u = self.t.uint;
        let eight = self.b.const_u32(8);
        let zero = self.b.const_u32(0);
        let max = self.b.const_u32(32);
        let offset = self
            .b
            .inst(op::BIT_FIELD_U_EXTRACT, u, &[spec, zero, eight]);
        let count = self
            .b
            .inst(op::BIT_FIELD_U_EXTRACT, u, &[spec, eight, eight]);
        let offset_c = self.b.ext(u, glsl::U_MIN, &[offset, max]);
        let room = self.b.inst(op::I_SUB, u, &[max, offset_c]);
        let count_c = self.b.ext(u, glsl::U_MIN, &[count, room]);
        let inserted = self
            .b
            .inst(op::BIT_FIELD_INSERT, u, &[base, insert, offset_c, count_c]);
        let beyond = self
            .b
            .inst(op::U_GREATER_THAN_EQUAL, self.t.boolean, &[offset, max]);
        self.select_u(beyond, base, inserted)
    }

    // ---- texture instructions ----

    /// `TEXS`: a 2D-family sample with the type in bits 53:56 and up to four
    /// results split over two destination register pairs.
    fn texs(&mut self, ins: &Instruction) -> Result<(), TranslateError> {
        let word = ins.word;
        let (src_a, src_b) = (register(word, 8), register(word, 20));
        let handle = field(word, 36, 13) * 4;
        let zero = self.b.const_f32(0.0);
        let encoding = field(word, 53, 4);
        let (kind, shadow) = match encoding {
            0 => (TextureKind::D1, false),
            1..=3 => (TextureKind::D2, false),
            4..=6 => (TextureKind::D2, true),
            7 | 8 => (TextureKind::D2Array, false),
            9 => (TextureKind::D2Array, true),
            10 | 11 => (TextureKind::D3, false),
            12 | 13 => (TextureKind::Cube, false),
            _ => return Err(unsupported(ins, format!("TEXS encoding {encoding}"))),
        };
        let texture = self.texture(handle, kind, shadow);
        let f = |t: &mut Self, r: u8| t.get_f(r);
        let result = match encoding {
            0 => {
                let x = f(self, src_a);
                self.sample(texture, x, Some(zero), None, None, None)
            }
            1 | 2 => {
                let (x, y) = (f(self, src_a), f(self, src_b));
                let coords = self.vec(&[x, y]);
                let lod = (encoding == 2).then_some(zero);
                self.sample(texture, coords, lod, None, None, None)
            }
            3 => {
                let (x, y, lod) = (f(self, src_a), f(self, src_a + 1), f(self, src_b));
                let coords = self.vec(&[x, y]);
                self.sample(texture, coords, Some(lod), None, None, None)
            }
            4 | 6 => {
                let (x, y, dref) = (f(self, src_a), f(self, src_a + 1), f(self, src_b));
                let coords = self.vec(&[x, y]);
                let lod = (encoding == 6).then_some(zero);
                self.sample(texture, coords, lod, None, None, Some(dref))
            }
            5 => {
                let (x, y) = (f(self, src_a), f(self, src_a + 1));
                let (lod, dref) = (f(self, src_b), f(self, src_b + 1));
                let coords = self.vec(&[x, y]);
                self.sample(texture, coords, Some(lod), None, None, Some(dref))
            }
            7 | 8 => {
                let layer = self.layer(src_a);
                let (x, y) = (f(self, src_a + 1), f(self, src_b));
                let coords = self.vec(&[x, y, layer]);
                let lod = (encoding == 8).then_some(zero);
                self.sample(texture, coords, lod, None, None, None)
            }
            9 => {
                let layer = self.layer(src_a);
                let (x, y) = (f(self, src_a + 1), f(self, src_b));
                let dref = f(self, src_b + 1);
                let coords = self.vec(&[x, y, layer]);
                self.sample(texture, coords, Some(zero), None, None, Some(dref))
            }
            10..=12 => {
                let (x, y, z) = (f(self, src_a), f(self, src_a + 1), f(self, src_b));
                let coords = self.vec(&[x, y, z]);
                let lod = (encoding == 11).then_some(zero);
                self.sample(texture, coords, lod, None, None, None)
            }
            _ => {
                let (x, y, z) = (f(self, src_a), f(self, src_a + 1), f(self, src_b));
                let lod = f(self, src_b + 1);
                let coords = self.vec(&[x, y, z]);
                self.sample(texture, coords, Some(lod), None, None, None)
            }
        };
        self.store_scalar_result(ins, result)
    }

    /// `TEXS` and `TLDS` results: the components the swizzle field picks,
    /// into destination A and A + 1, then B and B + 1.
    fn store_scalar_result(&mut self, ins: &Instruction, result: Id) -> Result<(), TranslateError> {
        let word = ins.word;
        let (dest_a, dest_b) = (register(word, 0), register(word, 28));
        // The written components: one table when there is no second
        // destination pair, another when there is.
        const RG: [u32; 8] = [0x1, 0x2, 0x4, 0x8, 0x3, 0x9, 0xA, 0xC];
        const RGBA: [u32; 5] = [0x7, 0xB, 0xD, 0xE, 0xF];
        let swizzle = field(word, 50, 3) as usize;
        let mask = if dest_b == REGISTER_ZERO {
            *RG.get(swizzle)
                .ok_or_else(|| unsupported(ins, "TEXS component selection"))?
        } else {
            *RGBA
                .get(swizzle)
                .ok_or_else(|| unsupported(ins, "TEXS component selection"))?
        };
        let destinations = [
            dest_a,
            dest_a.wrapping_add(1),
            dest_b,
            dest_b.wrapping_add(1),
        ];
        self.store_components(result, mask, &destinations);
        Ok(())
    }

    /// `TLDS`: a texel fetch at integer coordinates, from register A (then
    /// A + 1) and B as the target says, at level zero or the level in B.
    fn tlds(&mut self, ins: &Instruction) -> Result<(), TranslateError> {
        let word = ins.word;
        let (src_a, src_b) = (register(word, 8), register(word, 20));
        let target = field(word, 53, 4);
        let kind = match target {
            0 | 1 => TextureKind::D1,
            2 | 5 => TextureKind::D2,
            7 => TextureKind::D3,
            8 => TextureKind::D2Array,
            _ => return Err(unsupported(ins, format!("TLDS target {target}"))),
        };
        let texture = self.texture(field(word, 36, 13) * 4, kind, false);
        // Texel coordinates are the guest's: a rendered image at a
        // resolution scale has that many more texels (3D ones are not).
        let scale = (kind != TextureKind::D3).then(|| self.texture_scale(&texture));
        let i = |t: &mut Self, r: u8| {
            let value = t.get_u(r);
            t.b.bitcast(t.t.int, value)
        };
        let texel = |t: &mut Self, r: u8| {
            let value = i(t, r);
            match scale {
                Some(scale) => t.scale_texel(value, scale),
                None => value,
            }
        };
        let zero = self.b.const_i32(0);
        let (coords, lod) = match target {
            0 => (texel(self, src_a), zero),
            1 => (texel(self, src_a), i(self, src_b)),
            2 => {
                let (x, y) = (texel(self, src_a), texel(self, src_b));
                (self.ivec(&[x, y]), zero)
            }
            5 => {
                let (x, y, lod) = (texel(self, src_a), texel(self, src_a + 1), i(self, src_b));
                (self.ivec(&[x, y]), lod)
            }
            7 => {
                let (x, y, z) = (i(self, src_a), i(self, src_a + 1), i(self, src_b));
                (self.ivec(&[x, y, z]), zero)
            }
            _ => {
                let (x, y, layer) = (texel(self, src_b), texel(self, src_b + 1), i(self, src_a));
                (self.ivec(&[x, y, layer]), zero)
            }
        };
        let image = self.b.inst(op::IMAGE, texture.image, &[texture.value]);
        let result = self.b.inst(
            op::IMAGE_FETCH,
            self.t.vec4,
            &[image, coords, image_operands::LOD, lod],
        );
        self.store_scalar_result(ins, result)
    }

    fn ivec(&mut self, parts: &[Id]) -> Id {
        let ty = self.b.vector(self.t.int, parts.len() as u32);
        self.b.inst(op::COMPOSITE_CONSTRUCT, ty, parts)
    }

    /// `TXQ`'s size query (query 1): width, height, depth and mip levels at
    /// the level in register A, the components the mask picks written to
    /// consecutive registers. A depth query reads a 3D binding, any other a
    /// 2D one (the texture header is not known when translating).
    fn txq(&mut self, ins: &Instruction) -> Result<(), TranslateError> {
        let word = ins.word;
        if field(word, 22, 6) != 1 {
            return Err(unsupported(ins, "TXQ queries other than the size"));
        }
        let mask = field(word, 31, 4);
        let kind = if mask & 4 != 0 {
            TextureKind::D3
        } else {
            TextureKind::D2
        };
        let texture = self.texture(field(word, 36, 13) * 4, kind, false);
        self.b.capability(capability::IMAGE_QUERY);
        let image = self.b.inst(op::IMAGE, texture.image, &[texture.value]);
        let level = self.get_u(register(word, 8));
        let level = self.b.bitcast(self.t.int, level);
        let dimensions = if kind == TextureKind::D3 { 3 } else { 2 };
        let size_type = self.b.vector(self.t.int, dimensions);
        let size = self
            .b
            .inst(op::IMAGE_QUERY_SIZE_LOD, size_type, &[image, level]);
        let mut values = Vec::new();
        let scale = self.texture_scale(&texture);
        for component in 0..dimensions {
            let mut value = self
                .b
                .inst(op::COMPOSITE_EXTRACT, self.t.int, &[size, component]);
            if component < 2 {
                // The guest's size of a rendered image at a resolution scale.
                value = self.unscale_size(value, scale);
            }
            values.push(self.b.bitcast(self.t.uint, value));
        }
        if dimensions == 2 {
            values.push(self.b.const_u32(1));
        }
        let levels = self.b.inst(op::IMAGE_QUERY_LEVELS, self.t.int, &[image]);
        values.push(self.b.bitcast(self.t.uint, levels));
        let mut destination = register(word, 0);
        for (component, value) in values.into_iter().enumerate() {
            if mask & (1 << component) != 0 && destination != REGISTER_ZERO {
                self.set_u(destination, value);
                destination += 1;
            }
        }
        Ok(())
    }

    /// A register's (or constant's) two half-precision lanes, as a swizzle
    /// picks them: both halves, the whole word as a 32-bit float in both
    /// lanes, the low half twice or the high half twice.
    fn half_pair(&mut self, bits: Id, swizzle: u32) -> [Id; 2] {
        if swizzle == 1 {
            let value = self.b.bitcast(self.t.float, bits);
            return [value, value];
        }
        let pair = self.b.ext(self.t.vec2, glsl::UNPACK_HALF_2X16, &[bits]);
        let low = self.b.inst(op::COMPOSITE_EXTRACT, self.t.float, &[pair, 0]);
        let high = self.b.inst(op::COMPOSITE_EXTRACT, self.t.float, &[pair, 1]);
        match swizzle {
            0 => [low, high],
            2 => [low, low],
            _ => [high, high],
        }
    }

    fn half_immediate(&mut self, low: u32, high: u32) -> [Id; 2] {
        [low, high].map(|bits| self.b.const_f32(half_to_f32(bits as u16)))
    }

    fn half_modifiers(&mut self, pair: [Id; 2], abs: bool, neg: bool) -> [Id; 2] {
        pair.map(|value| self.fmod(value, abs, neg))
    }

    /// `HADD2`, `HMUL2` and `HFMA2` in all their forms: both half-precision
    /// lanes computed as 32-bit floats, then written as the output format
    /// says: packed halves, the first lane as a 32-bit float, or one half
    /// merged into the destination's other half.
    fn half_arithmetic(&mut self, ins: &Instruction) -> Result<(), TranslateError> {
        use HalfForm::{Constant, Immediate, Immediate32, Register, RegisterConstant};
        use HalfOperation::{Add, FusedMultiplyAdd as Fma, Multiply};
        let word = ins.word;
        let f = |bit| flag(word, bit);
        let (rd, ra) = (register(word, 0), register(word, 8));
        let (operation, form) = match (
            field(word, 51, 13),
            field(word, 57, 7),
            field(word, 59, 5),
            f(55),
        ) {
            (0b0_1011_1010_0000, ..) => (Fma, Register),
            (0b0_1011_1010_0001, ..) => (Multiply, Register),
            (0b0_1011_1010_0010, ..) => (Add, Register),
            (_, 0b001_0100, ..) => (Fma, Immediate32),
            (_, 0b001_0101, ..) => (Multiply, Immediate32),
            (_, 0b001_0110, ..) => (Add, Immediate32),
            (_, 0b011_1100, _, constant) => (Multiply, if constant { Constant } else { Immediate }),
            (_, 0b011_1101, _, constant) => (Add, if constant { Constant } else { Immediate }),
            (_, _, 0b0_1110, constant) => (Fma, if constant { Constant } else { Immediate }),
            (_, _, 0b0_1100, true) => (Fma, RegisterConstant),
            _ => return Err(unsupported(ins, "this half-precision form")),
        };
        // Absolute and negate of A and B, negate of C, saturate.
        let (a_abs, a_neg, b_abs, b_neg, c_neg, saturate) = match (operation, form) {
            (Add, Register) => (f(44), f(43), f(30), f(31), false, f(32)),
            (Add, Immediate) | (Multiply, Immediate) => (f(44), f(43), false, false, false, f(52)),
            (Add, Constant) => (f(44), f(43), f(54), f(56), false, f(52)),
            (Add, _) => (false, f(56), false, false, false, f(52)),
            (Multiply, Register) => (f(44), false, f(30), f(31), false, f(32)),
            (Multiply, Constant) => (f(44), false, f(54), f(43), false, f(52)),
            (Multiply, _) => (false, false, false, false, false, f(52)),
            (Fma, Register) => (false, false, false, f(31), f(30), f(32)),
            (Fma, Immediate) => (false, false, false, false, f(51), f(52)),
            (Fma, Constant | RegisterConstant) => (false, false, false, f(56), f(51), f(52)),
            (Fma, Immediate32) => (false, false, false, false, f(51), false),
        };
        let a_swizzle = if form == Immediate32 && operation != Fma {
            field(word, 53, 2)
        } else {
            field(word, 47, 2)
        };
        let a_bits = self.get_u(ra);
        let a = self.half_pair(a_bits, a_swizzle);
        let a = self.half_modifiers(a, a_abs, a_neg);
        let constant = |t: &mut Self| t.cbuf_u(field(word, 34, 5), field(word, 20, 14) * 4);
        let b = match form {
            Register => {
                let bits = self.get_u(register(word, 20));
                self.half_pair(bits, field(word, 28, 2))
            }
            Immediate => self.half_immediate(
                field(word, 20, 10) << 6,
                (field(word, 30, 9) | (field(word, 56, 1) << 9)) << 6,
            ),
            Constant => {
                let bits = constant(self);
                self.half_pair(bits, 1)
            }
            RegisterConstant => {
                let bits = self.get_u(register(word, 39));
                self.half_pair(bits, field(word, 53, 2))
            }
            Immediate32 => self.half_immediate(field(word, 20, 16), field(word, 36, 16)),
        };
        let b = self.half_modifiers(b, b_abs, b_neg);
        let mut lanes = [a[0], a[1]];
        if operation == Fma {
            let c = match form {
                Register => {
                    let bits = self.get_u(register(word, 39));
                    self.half_pair(bits, field(word, 35, 2))
                }
                Immediate | Constant => {
                    let bits = self.get_u(register(word, 39));
                    self.half_pair(bits, field(word, 53, 2))
                }
                RegisterConstant => {
                    let bits = constant(self);
                    self.half_pair(bits, 1)
                }
                Immediate32 => {
                    let bits = self.get_u(rd);
                    self.half_pair(bits, 0)
                }
            };
            let c = self.half_modifiers(c, false, c_neg);
            for lane in 0..2 {
                lanes[lane] = self
                    .b
                    .ext(self.t.float, glsl::FMA, &[a[lane], b[lane], c[lane]]);
            }
        } else {
            for lane in 0..2 {
                lanes[lane] = if operation == Add {
                    self.fadd(a[lane], b[lane])
                } else {
                    self.fmul(a[lane], b[lane])
                };
            }
        }
        let lanes = lanes.map(|lane| self.saturate(lane, saturate));
        let format = if form == Immediate32 {
            0
        } else {
            field(word, 49, 2)
        };
        if format == 1 {
            self.set_f(rd, lanes[0]);
            return Ok(());
        }
        let pair = self.vec(&lanes);
        let packed = self.b.ext(self.t.uint, glsl::PACK_HALF_2X16, &[pair]);
        let value = match format {
            0 => packed,
            // Merge into the low half (2) or the high half (3).
            _ => {
                let kept_mask = if format == 2 { 0xFFFF_0000 } else { 0xFFFF };
                let kept_mask = self.b.const_u32(kept_mask);
                let new_mask = self.b.inst(op::NOT, self.t.uint, &[kept_mask]);
                let old = self.get_u(rd);
                let kept = self.b.inst(op::BITWISE_AND, self.t.uint, &[old, kept_mask]);
                let new = self
                    .b
                    .inst(op::BITWISE_AND, self.t.uint, &[packed, new_mask]);
                self.b.inst(op::BITWISE_OR, self.t.uint, &[kept, new])
            }
        };
        self.set_u(rd, value);
        Ok(())
    }

    /// `TLD4S`: a 2D gather of one component, four results; with `f16` the
    /// four results are packed as half pairs, the
    /// first two in destination A, the others in destination B.
    fn tld4s(&mut self, ins: &Instruction, f16: bool) -> Result<(), TranslateError> {
        let word = ins.word;
        let (dest_a, src_a, src_b, dest_b) = (
            register(word, 0),
            register(word, 8),
            register(word, 20),
            register(word, 28),
        );
        let handle = field(word, 36, 13) * 4;
        let (aoffi, dc) = (flag(word, 51), flag(word, 50));
        let component = field(word, 52, 2);
        let texture = self.texture(handle, TextureKind::D2, dc);
        let result = if aoffi {
            let (x, y) = (self.get_f(src_a), self.get_f(src_a + 1));
            let coords = self.vec(&[x, y]);
            let offset = self.offset_2d(src_b, 6);
            let dref = if dc {
                Some(self.get_f(src_b + 1))
            } else {
                None
            };
            self.gather(texture, coords, component, Some(offset), dref)
        } else if dc {
            let (x, y) = (self.get_f(src_a), self.get_f(src_a + 1));
            let coords = self.vec(&[x, y]);
            let dref = self.get_f(src_b);
            self.gather(texture, coords, component, None, Some(dref))
        } else {
            let (x, y) = (self.get_f(src_a), self.get_f(src_b));
            let coords = self.vec(&[x, y]);
            self.gather(texture, coords, component, None, None)
        };
        if f16 {
            for (pair, destination) in [dest_a, dest_b].into_iter().enumerate() {
                let pair = pair as u32 * 2;
                let low = self
                    .b
                    .inst(op::COMPOSITE_EXTRACT, self.t.float, &[result, pair]);
                let high = self
                    .b
                    .inst(op::COMPOSITE_EXTRACT, self.t.float, &[result, pair + 1]);
                let halves = self.vec(&[low, high]);
                let packed = self.b.ext(self.t.uint, glsl::PACK_HALF_2X16, &[halves]);
                self.set_u(destination, packed);
            }
            return Ok(());
        }
        let destinations = [
            dest_a,
            dest_a.wrapping_add(1),
            dest_b,
            dest_b.wrapping_add(1),
        ];
        self.store_components(result, 0xF, &destinations);
        Ok(())
    }

    /// The coordinates and array layer of a `TEX`/`TLD4` texture type, read
    /// from consecutive registers.
    fn coordinates(
        &mut self,
        ins: &Instruction,
        kind_bits: u32,
        base: u8,
    ) -> Result<(TextureKind, Id), TranslateError> {
        let f = |t: &mut Self, r: u8| t.get_f(r);
        Ok(match kind_bits {
            0 => (TextureKind::D1, f(self, base)),
            2 => {
                let (x, y) = (f(self, base), f(self, base + 1));
                (TextureKind::D2, self.vec(&[x, y]))
            }
            3 => {
                let layer = self.layer(base);
                let (x, y) = (f(self, base + 1), f(self, base + 2));
                (TextureKind::D2Array, self.vec(&[x, y, layer]))
            }
            4 => {
                let (x, y, z) = (f(self, base), f(self, base + 1), f(self, base + 2));
                (TextureKind::D3, self.vec(&[x, y, z]))
            }
            6 => {
                let (x, y, z) = (f(self, base), f(self, base + 1), f(self, base + 2));
                (TextureKind::Cube, self.vec(&[x, y, z]))
            }
            other => return Err(unsupported(ins, format!("texture type {other}"))),
        })
    }

    /// `TEX`: type in bits 28:30, component mask in bits 31:34, coordinates
    /// from register A, then lod/bias, offset and depth reference from
    /// register B on.
    fn tex(&mut self, ins: &Instruction, bindless: bool) -> Result<(), TranslateError> {
        let word = ins.word;
        let (dest, coord, mut meta) = (register(word, 0), register(word, 8), register(word, 20));
        let kind_bits = field(word, 28, 3);
        let mask = field(word, 31, 4);
        let dc = flag(word, 50);
        let (aoffi, blod) = if bindless {
            (flag(word, 36), field(word, 37, 3))
        } else {
            (flag(word, 54), field(word, 55, 3))
        };
        let (kind, coords) = self.coordinates(ins, kind_bits, coord)?;
        let (handles, selector) = self.texture_handles(ins, bindless, &mut meta)?;
        let zero = self.b.const_f32(0.0);
        let (lod, bias) = match blod {
            0 => (None, None),
            1 => (Some(zero), None),
            2 | 3 => {
                let value = self.get_f(meta);
                meta = meta.wrapping_add(1);
                if blod == 2 {
                    (None, Some(value))
                } else {
                    (Some(value), None)
                }
            }
            other => return Err(unsupported(ins, format!("TEX lod mode {other}"))),
        };
        let offset = if aoffi {
            let index = meta;
            meta = meta.wrapping_add(1);
            Some(self.offset_2d(index, 4))
        } else {
            None
        };
        let dref = if dc { Some(self.get_f(meta)) } else { None };
        let mut results = Vec::new();
        for handle in handles {
            let texture = self.texture(handle, kind, dc);
            results.push(self.sample(texture, coords, lod, bias, offset, dref));
        }
        let result = self.pick_handle(&results, selector);
        let destinations: Vec<u8> = (0..4).map(|index| dest.wrapping_add(index)).collect();
        self.store_components(result, mask, &destinations);
        Ok(())
    }

    /// `XMAD`: a 16 x 16-bit multiply plus a 32-bit addend. A and B are a
    /// 16-bit half of their registers (signed or not); `PSL` shifts the
    /// product left 16 bits; C is the whole addend register, its low or high
    /// half, or it plus B shifted left 16 bits; `MRG` replaces the result's
    /// high half with B's low half.
    fn xmad(&mut self, ins: &Instruction) -> Result<(), TranslateError> {
        let word = ins.word;
        let (rd, ra) = (register(word, 0), register(word, 8));
        if flag(word, 38) {
            return Err(unsupported(ins, "XMAD.X"));
        }
        let cbuf = |t: &mut Self| t.cbuf_u(field(word, 34, 5), field(word, 20, 14) * 4);
        // B, C, whether B is its high half, and the C mode.
        let (b, c, b_high, c_mode) = match ins.src_b {
            Operand::Register(_) => {
                let b = self.get_u(register(word, 20));
                let c = self.get_u(register(word, 39));
                (b, c, flag(word, 35), field(word, 50, 3))
            }
            Operand::Immediate(_) => {
                let b = self.b.const_u32(field(word, 20, 16));
                let c = self.get_u(register(word, 39));
                (b, c, false, field(word, 50, 3))
            }
            Operand::Constant { .. } => {
                let b = cbuf(self);
                let c = self.get_u(register(word, 39));
                (b, c, flag(word, 52), field(word, 50, 2))
            }
            _ => {
                // B from the register field at 39, C from the constant buffer.
                let b = self.get_u(register(word, 39));
                let c = cbuf(self);
                (b, c, flag(word, 52), field(word, 50, 2))
            }
        };
        let a = self.get_u(ra);
        let half = |t: &mut Self, value: Id, high: bool, signed: bool| {
            let sixteen = t.b.const_u32(16);
            match (high, signed) {
                (true, true) => {
                    t.b.inst(op::SHIFT_RIGHT_ARITHMETIC, t.t.uint, &[value, sixteen])
                }
                (true, false) => {
                    t.b.inst(op::SHIFT_RIGHT_LOGICAL, t.t.uint, &[value, sixteen])
                }
                (false, true) => {
                    let zero = t.b.const_u32(0);
                    t.b.inst(op::BIT_FIELD_S_EXTRACT, t.t.uint, &[value, zero, sixteen])
                }
                (false, false) => {
                    let mask = t.b.const_u32(0xFFFF);
                    t.b.inst(op::BITWISE_AND, t.t.uint, &[value, mask])
                }
            }
        };
        let a16 = half(self, a, flag(word, 53), flag(word, 48));
        let b16 = half(self, b, b_high, flag(word, 49));
        let sixteen = self.b.const_u32(16);
        let mut product = self.b.inst(op::I_MUL, self.t.uint, &[a16, b16]);
        if flag(word, 36) {
            product = self
                .b
                .inst(op::SHIFT_LEFT_LOGICAL, self.t.uint, &[product, sixteen]);
        }
        let addend = match c_mode {
            0 => c,
            1 => {
                let mask = self.b.const_u32(0xFFFF);
                self.b.inst(op::BITWISE_AND, self.t.uint, &[c, mask])
            }
            2 => self
                .b
                .inst(op::SHIFT_RIGHT_LOGICAL, self.t.uint, &[c, sixteen]),
            4 => {
                let shifted = self
                    .b
                    .inst(op::SHIFT_LEFT_LOGICAL, self.t.uint, &[b, sixteen]);
                self.b.inst(op::I_ADD, self.t.uint, &[c, shifted])
            }
            _ => return Err(unsupported(ins, "XMAD.CSFU")),
        };
        let mut value = self.b.inst(op::I_ADD, self.t.uint, &[product, addend]);
        if flag(word, 37) {
            let mask = self.b.const_u32(0xFFFF);
            let low = self.b.inst(op::BITWISE_AND, self.t.uint, &[value, mask]);
            let high = self
                .b
                .inst(op::SHIFT_LEFT_LOGICAL, self.t.uint, &[b, sixteen]);
            value = self.b.inst(op::BITWISE_OR, self.t.uint, &[low, high]);
        }
        self.set_u(rd, value);
        Ok(())
    }

    /// `TXD`: a sample with explicit gradients. Register B holds them
    /// interleaved per coordinate: d/dx of u, d/dy of u, d/dx of v, ...
    fn txd(&mut self, ins: &Instruction) -> Result<(), TranslateError> {
        let word = ins.word;
        let (dest, coord, gradients) = (register(word, 0), register(word, 8), register(word, 20));
        let kind_bits = field(word, 28, 3);
        let mask = field(word, 31, 4);
        if flag(word, 35) {
            return Err(unsupported(ins, "TXD with texel offsets"));
        }
        let handle = field(word, 36, 13) * 4;
        let (kind, coords) = self.coordinates(ins, kind_bits, coord)?;
        let components = match kind {
            TextureKind::D2 | TextureKind::D2Array => 2,
            TextureKind::D1 => 1,
            TextureKind::D3 | TextureKind::Cube => 3,
        };
        let texture = self.texture(handle, kind, false);
        let mut dx = Vec::new();
        let mut dy = Vec::new();
        for index in 0..components {
            dx.push(self.get_f(gradients.wrapping_add(2 * index)));
            dy.push(self.get_f(gradients.wrapping_add(2 * index + 1)));
        }
        let (dx, dy) = if components == 1 {
            (dx[0], dy[0])
        } else {
            (self.vec(&dx), self.vec(&dy))
        };
        let result = self.b.inst(
            op::IMAGE_SAMPLE_EXPLICIT_LOD,
            self.t.vec4,
            &[texture.value, coords, image_operands::GRAD, dx, dy],
        );
        let destinations: Vec<u8> = (0..4).map(|index| dest.wrapping_add(index)).collect();
        self.store_components(result, mask, &destinations);
        Ok(())
    }

    /// `TLD4`: a gather with type, mask and component like `TEX`, and an
    /// optional offset register.
    fn tld4(&mut self, ins: &Instruction, bindless: bool) -> Result<(), TranslateError> {
        let word = ins.word;
        let (dest, coord, mut meta) = (register(word, 0), register(word, 8), register(word, 20));
        let kind_bits = field(word, 28, 3);
        let mask = field(word, 31, 4);
        let dc = flag(word, 50);
        let (component, offsets) = if bindless {
            (field(word, 38, 2), field(word, 36, 2))
        } else {
            (field(word, 56, 2), field(word, 54, 2))
        };
        let (kind, coords) = self.coordinates(ins, kind_bits, coord)?;
        if kind != TextureKind::D2 && kind != TextureKind::D2Array {
            return Err(unsupported(ins, "TLD4 of a non-2D texture"));
        }
        let (handles, selector) = self.texture_handles(ins, bindless, &mut meta)?;
        let mut results = Vec::new();
        for handle in handles {
            let texture = self.texture(handle, kind, dc);
            results.push(self.tld4_with(texture, coords, component, offsets, meta, dc, ins)?);
        }
        let result = self.pick_handle(&results, selector);
        let destinations: Vec<u8> = (0..4).map(|index| dest.wrapping_add(index)).collect();
        self.store_components(result, mask, &destinations);
        Ok(())
    }

    /// One `TLD4` gather (or four, with per-texel offsets) from `texture`,
    /// its offsets and depth reference read from register `meta` on.
    #[allow(clippy::too_many_arguments)]
    fn tld4_with(
        &mut self,
        texture: Texture,
        coords: Id,
        component: u32,
        offsets: u32,
        mut meta: u8,
        dc: bool,
        ins: &Instruction,
    ) -> Result<Id, TranslateError> {
        Ok(match offsets {
            0 | 1 => {
                let offset = if offsets == 1 {
                    let index = meta;
                    meta = meta.wrapping_add(1);
                    Some(self.offset_2d(index, 6))
                } else {
                    None
                };
                let dref = if dc { Some(self.get_f(meta)) } else { None };
                self.gather(texture, coords, component, offset, dref)
            }
            2 => {
                // Per-texel offsets: two registers of four signed 6-bit
                // values, (x, y) for each of the four texels. Each texel comes
                // from its own gather.
                let (first, second) = (meta, meta.wrapping_add(1));
                let dref = if dc {
                    Some(self.get_f(meta.wrapping_add(2)))
                } else {
                    None
                };
                let mut texels = [0; 4];
                for (texel, value) in texels.iter_mut().enumerate() {
                    let source = if texel < 2 { first } else { second };
                    let shift = (texel as u32 % 2) * 16;
                    let offset = self.offset_pair(source, shift);
                    let gathered = self.gather(texture, coords, component, Some(offset), dref);
                    *value = self.b.inst(
                        op::COMPOSITE_EXTRACT,
                        self.t.float,
                        &[gathered, texel as u32],
                    );
                }
                self.vec(&texels)
            }
            _ => return Err(unsupported(ins, "TLD4 offset type 3")),
        })
    }

    /// The texture handles (offsets in the handle buffer) an instruction may
    /// sample: its immediate one, or for a bindless one the entries of the
    /// handle array its register B was loaded from (`handle_array`), with
    /// the byte offset into the array that picks one at run time. Register
    /// B then moves past the handle.
    fn texture_handles(
        &mut self,
        ins: &Instruction,
        bindless: bool,
        meta: &mut u8,
    ) -> Result<(Vec<u32>, Option<Id>), TranslateError> {
        if !bindless {
            return Ok((vec![field(ins.word, 36, 13) * 4], None));
        }
        let (index, base) = *self
            .handle_arrays
            .get(&ins.address)
            .ok_or_else(|| unsupported(ins, "bindless handle not from the handle buffer"))?;
        *meta = meta.wrapping_add(1);
        if index == REGISTER_ZERO {
            return Ok((vec![base], None));
        }
        let selector = self.get_u(index);
        let handles = (0..HANDLE_ARRAY_ENTRIES)
            .map(|entry| base + entry * 8)
            .collect();
        Ok((handles, Some(selector)))
    }

    /// The result of the handle the selector (a byte offset into the handle
    /// array) picks; the first for offsets past the bound entries.
    fn pick_handle(&mut self, results: &[Id], selector: Option<Id>) -> Id {
        let Some(offset) = selector else {
            return results[0];
        };
        let bvec4 = self.b.vector(self.t.boolean, 4);
        let mut result = results[0];
        for (entry, value) in results.iter().enumerate().skip(1) {
            let expected = self.b.const_u32(entry as u32 * 8);
            let matches = self
                .b
                .inst(op::I_EQUAL, self.t.boolean, &[offset, expected]);
            let matches = self.b.inst(op::COMPOSITE_CONSTRUCT, bvec4, &[matches; 4]);
            result = self
                .b
                .inst(op::SELECT, self.t.vec4, &[matches, *value, result]);
        }
        result
    }

    /// `SHFL` within a 2x2 quad (segment mask 0x1C, clamp 3), in fragment
    /// programs: the butterflies that exchange values with a neighbour, and
    /// the index form reading one lane (LM3's half-resolution passes read
    /// lane 0's coordinates; their draws were skipped without it).
    fn shfl(&mut self, ins: &Instruction) -> Result<(), TranslateError> {
        let word = ins.word;
        let (dest, source) = (register(word, 0), register(word, 8));
        let index_immediate = flag(word, 28);
        let clamp_immediate = flag(word, 29);
        let shuffle_mode = field(word, 30, 2);
        // The lane: an immediate, or RZ (lane 0) for the register form.
        let lane = if index_immediate {
            Some(field(word, 20, 5))
        } else {
            (register(word, 20) == REGISTER_ZERO).then_some(0)
        };
        let clamp = field(word, 34, 13);
        // Segments of four lanes (a 2x2 quad of pixels): a butterfly
        // swaps with a neighbour, an index reads one lane of the quad.
        let quad = clamp_immediate && clamp == 0x1C03;
        let (xor, index) = match (quad, shuffle_mode, lane) {
            // Lane XOR 1: horizontal neighbour, 2: vertical, 3: diagonal.
            (true, 3, Some(mask @ 1..=3)) => (Some(mask), None),
            // LM3's half-resolution passes read lane 0's coordinates.
            (true, 0, Some(index)) => (None, Some(index & 3)),
            _ => {
                return Err(unsupported(
                    ins,
                    "SHFL other than a quad butterfly or a quad lane read",
                ))
            }
        };
        if !self.is_fragment() {
            return Err(unsupported(ins, "SHFL outside a fragment program"));
        }
        // Not a subgroup shuffle: the Adreno 650 runs subgroup operations
        // in compute shaders only (`supportedStages` is COMPUTE), so one in
        // a fragment shader is invalid there. Fine derivatives exchange
        // values across a quad on any device: the value is moved as its two
        // 16-bit halves, as floats, which the differences carry exactly.
        let (x_odd, y_odd) = self.quad_position();
        let value = self.get_u(source);
        let (low_mask, sixteen) = (self.b.const_u32(0xFFFF), self.b.const_u32(16));
        let low = self
            .b
            .inst(op::BITWISE_AND, self.t.uint, &[value, low_mask]);
        let high = self
            .b
            .inst(op::SHIFT_RIGHT_LOGICAL, self.t.uint, &[value, sixteen]);
        let mut halves = [low, high];
        for half in &mut halves {
            let mut part = self.b.inst(op::CONVERT_U_TO_F, self.t.float, &[*half]);
            part = match (xor, index) {
                (Some(mask), _) => {
                    if mask & 1 != 0 {
                        part = self.quad_neighbour(part, x_odd, false);
                    }
                    if mask & 2 != 0 {
                        part = self.quad_neighbour(part, y_odd, true);
                    }
                    part
                }
                (_, Some(lane)) => {
                    // The lane's column first, then its row.
                    let across = self.quad_neighbour(part, x_odd, false);
                    let in_column = self.parity_is(x_odd, lane & 1 != 0);
                    let column = self.select_f(in_column, part, across);
                    let down = self.quad_neighbour(column, y_odd, true);
                    let in_row = self.parity_is(y_odd, lane & 2 != 0);
                    self.select_f(in_row, column, down)
                }
                (None, None) => return Err(unsupported(ins, "SHFL without a lane")),
            };
            *half = self.b.inst(op::CONVERT_F_TO_U, self.t.uint, &[part]);
        }
        let high = self
            .b
            .inst(op::SHIFT_LEFT_LOGICAL, self.t.uint, &[halves[1], sixteen]);
        let shuffled = self.b.inst(op::BITWISE_OR, self.t.uint, &[halves[0], high]);
        self.set_u(dest, shuffled);
        // The in-bounds predicate is always true inside a quad.
        let in_bounds = self.b.const_bool(true);
        self.set_p(field(word, 48, 3) as u8, in_bounds);
        Ok(())
    }

    /// Whether this invocation's pixel is in an odd column and an odd row:
    /// its place in its 2x2 quad (lane `x + 2y`, as quad lanes are named).
    fn quad_position(&mut self) -> (Id, Id) {
        let var = self.builtin(built_in::FRAG_COORD, self.t.vec4, storage::INPUT);
        let (zero, one) = (self.b.const_u32(0), self.b.const_u32(1));
        let odd = |t: &mut Self, component: u32| {
            let pointer = t.component_pointer(var, storage::INPUT, t.t.float, component);
            let coordinate = t.b.load(t.t.float, pointer);
            let pixel = t.b.inst(op::CONVERT_F_TO_U, t.t.uint, &[coordinate]);
            let bit = t.b.inst(op::BITWISE_AND, t.t.uint, &[pixel, one]);
            t.b.inst(op::I_NOT_EQUAL, t.t.boolean, &[bit, zero])
        };
        (odd(self, 0), odd(self, 1))
    }

    /// `odd`, or its negation when `want_odd` is false.
    fn parity_is(&mut self, odd: Id, want_odd: bool) -> Id {
        if want_odd {
            odd
        } else {
            self.b.inst(op::LOGICAL_NOT, self.t.boolean, &[odd])
        }
    }

    /// `value` as the quad neighbour across x (or y when `vertical`) holds
    /// it: a fine derivative is the difference across the pair, so the
    /// neighbour's value is `value + d` on even pixels and `value - d` on
    /// odd ones.
    fn quad_neighbour(&mut self, value: Id, odd: Id, vertical: bool) -> Id {
        self.b.capability(capability::DERIVATIVE_CONTROL);
        let derivative = if vertical {
            op::DPDY_FINE
        } else {
            op::DPDX_FINE
        };
        let difference = self.b.inst(derivative, self.t.float, &[value]);
        let plus = self.b.inst(op::F_ADD, self.t.float, &[value, difference]);
        let minus = self.b.inst(op::F_SUB, self.t.float, &[value, difference]);
        self.select_f(odd, minus, plus)
    }

    /// `FSWZADD`: per lane of the 2x2 quad, two bits of the mask choose
    /// a + b, -a + b, a - b or b (the derivative helper programs use with
    /// SHFL).
    fn fswzadd(&mut self, ins: &Instruction) -> Result<(), TranslateError> {
        let word = ins.word;
        if !self.is_fragment() {
            return Err(unsupported(ins, "FSWZADD outside a fragment program"));
        }
        let a = self.get_f(register(word, 8));
        let b = self.get_f(register(word, 20));
        // The lane in the quad from the pixel's position (subgroup lane ids
        // are not available to fragment shaders on every device; see SHFL).
        let (x_odd, y_odd) = self.quad_position();
        let (zero_u, one_u, two_u) = (
            self.b.const_u32(0),
            self.b.const_u32(1),
            self.b.const_u32(2),
        );
        let column = self
            .b
            .inst(op::SELECT, self.t.uint, &[x_odd, one_u, zero_u]);
        let row = self
            .b
            .inst(op::SELECT, self.t.uint, &[y_odd, two_u, zero_u]);
        let quad_lane = self.b.inst(op::BITWISE_OR, self.t.uint, &[column, row]);
        let three = self.b.const_u32(3);
        let one = self.b.const_u32(1);
        let shift = self
            .b
            .inst(op::SHIFT_LEFT_LOGICAL, self.t.uint, &[quad_lane, one]);
        let mask = self.b.const_u32(field(word, 28, 8));
        let shifted = self
            .b
            .inst(op::SHIFT_RIGHT_LOGICAL, self.t.uint, &[mask, shift]);
        let index = self.b.inst(op::BITWISE_AND, self.t.uint, &[shifted, three]);
        // Factors for a: (1, -1, 1, 0); for b: (1, 1, -1, 1).
        let is = |t: &mut Self, value: u32| {
            let value = t.b.const_u32(value);
            t.b.inst(op::I_EQUAL, t.t.boolean, &[index, value])
        };
        let (plus, minus, zero) = (
            self.b.const_f32(1.0),
            self.b.const_f32(-1.0),
            self.b.const_f32(0.0),
        );
        let index_is_1 = is(self, 1);
        let index_is_2 = is(self, 2);
        let index_is_3 = is(self, 3);
        let a_factor_tail = self.select_f(index_is_3, zero, plus);
        let a_factor = self.select_f(index_is_1, minus, a_factor_tail);
        let b_factor = self.select_f(index_is_2, minus, plus);
        let a_term = self.b.inst(op::F_MUL, self.t.float, &[a, a_factor]);
        let b_term = self.b.inst(op::F_MUL, self.t.float, &[b, b_factor]);
        let value = self.b.inst(op::F_ADD, self.t.float, &[a_term, b_term]);
        self.set_f(register(word, 0), value);
        Ok(())
    }

    fn finish(mut self, stage: Stage) -> TranslatedShader {
        let execution_model = if self.is_fragment() {
            model::FRAGMENT
        } else if self.is_compute() {
            model::GL_COMPUTE
        } else {
            model::VERTEX
        };
        if self.is_fragment() {
            self.b.execution_mode(self.main, mode::ORIGIN_UPPER_LEFT);
        }
        if self.is_compute() {
            let size = self.environment.local_size.map(|size| size.max(1));
            self.b
                .execution_mode_with(self.main, mode::LOCAL_SIZE, &size);
        }
        let interface = self.interface.clone();
        self.b.entry_point(execution_model, self.main, &interface);
        let descriptor_set = self.descriptor_set();
        TranslatedShader {
            stage,
            words: self.b.finish(),
            descriptor_set,
            uniform_buffers: self.constant_buffers.keys().copied().collect(),
            uniform_extents: self
                .constant_buffers
                .keys()
                .map(|slot| {
                    self.constant_extents
                        .get(slot)
                        .copied()
                        .unwrap_or(16)
                        .next_multiple_of(16)
                })
                .collect(),
            textures: self
                .textures
                .iter()
                .map(|(binding, _, _, _)| *binding)
                .collect(),
            inputs: self.inputs.keys().copied().collect(),
            outputs: self.outputs.keys().copied().collect(),
            color_outputs: self.color_outputs.keys().copied().collect(),
        }
    }
}

/// Translate a compute program (code only: compute programs have no
/// header) to a SPIR-V module, for the block size and shared memory in
/// `environment`.
pub fn translate_compute(
    code: &[u8],
    environment: &Environment,
) -> Result<TranslatedShader, TranslateError> {
    // The control flow and code passes read a program as header and code;
    // instruction bundles are aligned to the code's start either way.
    let mut program = vec![0u8; HEADER_SIZE];
    program[0] = 1;
    program.extend_from_slice(code);
    let mut header = ShaderProgramHeader::parse(&program).ok_or(TranslateError::NoHeader)?;
    header.stage = Stage::Compute;
    let flow = Flow::analyze(&program, &environment.constant_buffer_1)?;
    let mut translator = Translator::new(header, environment);
    translator.emit_program(&flow)?;
    Ok(translator.finish(Stage::Compute))
}

/// A fragment program that writes nothing (a header and `EXIT`): what a
/// draw without a fragment program (a depth-only pass) runs on a host that
/// needs one.
pub fn empty_fragment_program() -> Vec<u8> {
    let mut header = [0u32; HEADER_SIZE / 4];
    // SPH type 2 (fragment), version 3, shader type 5, SASS version 3.
    header[0] = 2 | (3 << 5) | (5 << 10) | (3 << 17);
    let mut program: Vec<u8> = header.iter().flat_map(|word| word.to_le_bytes()).collect();
    program.extend_from_slice(&0u64.to_le_bytes());
    program.extend_from_slice(&0xE300_0000_0007_000Fu64.to_le_bytes());
    program
}

/// Translate one program (header and code) to a SPIR-V module.
pub fn translate(
    program: &[u8],
    environment: &Environment,
) -> Result<TranslatedShader, TranslateError> {
    let header = ShaderProgramHeader::parse(program).ok_or(TranslateError::NoHeader)?;
    let stage = header.stage;
    if !matches!(stage, Stage::Vertex | Stage::Fragment) {
        return Err(TranslateError::UnsupportedStage(stage));
    }
    let flow = Flow::analyze(program, &environment.constant_buffer_1)?;
    let mut translator = Translator::new(header, environment);
    translator.emit_program(&flow)?;
    Ok(translator.finish(stage))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PT: u64 = 7 << 16;

    /// The 14-bit opcode field of a word, from its pattern bits.
    fn opcode(bits: &str) -> u64 {
        u64::from_str_radix(bits, 2).expect("opcode bits") << 50
    }

    fn ald(rd: u8, address: u32, components: u64) -> u64 {
        opcode("11101111110110")
            | ((components - 1) << 47)
            | (0xFF << 39)
            | (u64::from(address) << 20)
            | PT
            | (0xFF << 8)
            | u64::from(rd)
    }

    fn ast(source: u8, address: u32, components: u64) -> u64 {
        opcode("11101111111100")
            | ((components - 1) << 47)
            | (0xFF << 39)
            | (u64::from(address) << 20)
            | PT
            | (0xFF << 8)
            | u64::from(source)
    }

    fn exit() -> u64 {
        opcode("11100011000000") | PT | 0xF
    }

    fn ipa(rd: u8, address: u32) -> u64 {
        opcode("11100000000000")
            | (u64::from(address / 4) << 30)
            | (0xFF << 20)
            | PT
            | (0xFF << 8)
            | u64::from(rd)
    }

    fn mov32i(rd: u8, value: u32) -> u64 {
        opcode("00000001000000") | (u64::from(value) << 20) | (0xF << 12) | PT | u64::from(rd)
    }

    /// `FSETP.LT P0, PT, Ra, Rb`.
    fn fsetp_lt(ra: u8, rb: u8) -> u64 {
        opcode("01011011101100")
            | (1 << 48)
            | (7 << 39)
            | (u64::from(rb) << 20)
            | PT
            | (u64::from(ra) << 8)
            | 7
    }

    fn ssy(offset: u32) -> u64 {
        opcode("11100010100100") | (u64::from(offset) << 20)
    }

    fn sync(predicate: u64) -> u64 {
        opcode("11110000111110") | (predicate << 16) | 0xF
    }

    fn fmul32i(rd: u8, ra: u8, value: f32, predicate: u64) -> u64 {
        opcode("00011110000000")
            | (u64::from(value.to_bits()) << 20)
            | (predicate << 16)
            | (u64::from(ra) << 8)
            | u64::from(rd)
    }

    /// A program: a header for `stage` (1 vertex, 5 fragment), then the
    /// instructions, with a zero scheduling word heading each bundle.
    fn program(stage: u32, omap_targets: u32, instructions: &[u64]) -> Vec<u8> {
        let mut header = [0u32; 20];
        let sph_type = if stage == 5 { 2 } else { 1 };
        header[0] = sph_type | (3 << 5) | (stage << 10) | (3 << 17);
        // Attribute 0 interpolated with perspective in all four components.
        header[6] = 0xAA;
        header[18] = omap_targets;
        let mut bytes: Vec<u8> = header.iter().flat_map(|word| word.to_le_bytes()).collect();
        for (slot, word) in instructions.iter().enumerate() {
            if slot % 3 == 0 {
                bytes.extend_from_slice(&0u64.to_le_bytes());
            }
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// Check a module with `spirv-val` when it is installed.
    fn validate(name: &str, words: &[u32]) {
        let path =
            std::env::temp_dir().join(format!("recomp-gfx-{name}-{}.spv", std::process::id()));
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        std::fs::write(&path, bytes).expect("write module");
        let output = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.1"])
            .arg(&path)
            .output();
        let _ = std::fs::remove_file(&path);
        if let Ok(output) = output {
            assert!(
                output.status.success(),
                "{name}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn sel_and_flo_translate() {
        // FSETP P0; SEL R2, R0, R1, P0; FLO.SH R3, ~R2; FLO.S R4, R0.
        let sel = opcode("01011100101000") | (1 << 20) | PT | 2;
        let flo_sh = opcode("01011100001100") | (1 << 41) | (1 << 40) | (2 << 20) | PT | 3;
        let flo_s = opcode("01011100001100") | (1 << 48) | PT | 4;
        let bytes = program(1, 0, &[fsetp_lt(0, 1), sel, flo_sh, flo_s, exit()]);
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        validate("sel-flo", &shader.words);
    }

    #[test]
    fn psetp_combines_predicates() {
        // PSETP P1, P2, P2, P3, P4: (P2 OR P3) AND P4, and its inverse.
        let psetp = opcode("01010000100100")
            | (1 << 3)
            | 2
            | (2 << 12)
            | (1 << 24)
            | (3 << 29)
            | (4 << 39)
            | PT;
        let bytes = program(5, 0, &[psetp, exit()]);
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        validate("psetp", &shader.words);
    }

    #[test]
    fn brx_switches_through_its_jump_table() {
        // 0x58 MOV32I R0, 2; 0x60 IMNMX.U32 R0, R0, 3; 0x68 SHL R0, R0, 2;
        // 0x78 LDC R0, c1[R0]; 0x80 BRX R0 (base: the code start, 0x50);
        // cases 0 and 2 run 0x88, cases 1 and 3 run 0xa0.
        let imnmx = 0x3820_0380_0037_0000;
        let shl = 0x3848_0000_0027_0000;
        let ldc = 0xef94_0010_0007_0000;
        let brx = opcode("11100010010100") | (0xFF_FFC8 << 20) | PT | 0xF;
        let instructions = [
            mov32i(0, 2),
            imnmx,
            shl,
            ldc,
            brx,
            mov32i(1, 1.0f32.to_bits()),
            exit(),
            mov32i(1, 2.0f32.to_bits()),
            exit(),
        ];
        let bytes = program(1, 0, &instructions);
        let table: Vec<u8> = [0x38u32, 0x50, 0x38, 0x50]
            .iter()
            .flat_map(|offset| offset.to_le_bytes())
            .collect();
        let flow = Flow::analyze(&bytes, &table).expect("analyzes");
        assert_eq!(
            flow.jump_tables[&0x80],
            vec![(0x38, 0x88), (0x50, 0xa0), (0x38, 0x88), (0x50, 0xa0)]
        );
        let environment = Environment {
            constant_buffer_1: table,
            ..Environment::default()
        };
        let shader = translate(&bytes, &environment).expect("translates");
        validate("brx", &shader.words);
        // Without its constants the table cannot be read.
        assert!(translate(&bytes, &Environment::default()).is_err());
    }

    #[test]
    fn f2f_converts_halves_and_rounds() {
        // F2F.F16.F32 R2, R1; F2F.F32.F16 R3, R2.H1; F2F.F32.F32.FLOOR R4, R1.
        let f2f = opcode("01011100101010") | PT;
        let to_half = f2f | (2 << 10) | (1 << 8) | (1 << 20) | 2;
        let from_high_half = f2f | (1 << 10) | (2 << 8) | (1 << 41) | (2 << 20) | 3;
        let floor = f2f | (2 << 10) | (2 << 8) | (1 << 42) | (1 << 39) | (1 << 20) | 4;
        let bytes = program(
            1,
            0,
            &[
                mov32i(1, 1.5f32.to_bits()),
                to_half,
                from_high_half,
                floor,
                exit(),
            ],
        );
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        validate("f2f", &shader.words);
    }

    #[test]
    fn half_precision_forms_translate() {
        // HADD2 R2, R0, R1; HADD2.MRG_H1 R2, R0, R1; HMUL2 R3, R2, 1.0 1.0;
        // HFMA2.F32 R4, R3, c0[0x0], R1; HADD2.F16 R5, R4, 0x3c003c00 (32I);
        // HFMA2 R6, R5, R2, c0[0x4] (register then constant).
        let hadd2 = opcode("01011101000100") | (1 << 20) | PT | 2;
        let merge = hadd2 | (3 << 49);
        let hmul2 = opcode("01111000000000") | (0xF0 << 30) | (0xF0 << 20) | (2 << 8) | PT | 3;
        let hfma2_c = opcode("01110000100000") | (1 << 49) | (1 << 39) | (3 << 8) | PT | 4;
        let hadd2_32i = opcode("00101100000000") | (0x3C00_3C00 << 20) | (4 << 8) | PT | 5;
        let hfma2_rc = opcode("01100000100000") | (2 << 39) | (1 << 20) | (5 << 8) | PT | 6;
        let bytes = program(
            1,
            0,
            &[hadd2, merge, hmul2, hfma2_c, hadd2_32i, hfma2_rc, exit()],
        );
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        validate("half", &shader.words);
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0xC000), -2.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn bindless_textures_bind_their_handle_array() {
        // SHL R11, R16, 3; LDC R16, c2[R11 + 0x48]; LDC R6, c2[R11 + 0x48];
        // TEX.B.DC R17 (2D array, offset) and TLD4.B.DC R4, handles in R16
        // and R6 (LM3's shadow lookups).
        let instructions = [
            0x3848_0000_0037_100b,
            0xef94_0020_0487_0b10,
            0xef94_0020_0487_0b06,
            0xdebe_0010_b107_0811,
            0xdefe_0007_b067_0804,
            exit(),
        ];
        let bytes = program(5, 0, &instructions);
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        validate("bindless", &shader.words);
        let handles: BTreeSet<u32> = shader
            .textures
            .iter()
            .map(|binding| binding.handle_offset)
            .collect();
        assert_eq!(handles, BTreeSet::from([0x48, 0x50, 0x58, 0x60]));
        // A handle loaded from anywhere else is refused.
        let unknown = program(5, 0, &[instructions[3], exit()]);
        assert!(translate(&unknown, &Environment::default()).is_err());
    }

    #[test]
    fn the_empty_fragment_program_writes_nothing() {
        let shader =
            translate(&empty_fragment_program(), &Environment::default()).expect("translates");
        assert_eq!(shader.stage, Stage::Fragment);
        assert!(shader.color_outputs.is_empty());
        validate("empty-fragment", &shader.words);
    }

    #[test]
    fn xmad_forms_translate() {
        // XMAD R0, R1, R2, R3 (register); XMAD.PSL.CBCC R4, R1.H1, R2, R0;
        // XMAD.MRG R5, R1, 0x1234, R4 (immediate); XMAD R6, R1, c1[0x10], R5.
        let register = opcode("01011011000000") | (3 << 39) | (2 << 20) | (1 << 8) | PT;
        let cbcc = opcode("01011011000000")
            | (4 << 50)
            | (1 << 53)
            | (1 << 36)
            | (2 << 20)
            | (1 << 8)
            | PT
            | 4;
        let immediate =
            opcode("00110110000000") | (1 << 37) | (4 << 39) | (0x1234 << 20) | (1 << 8) | PT | 5;
        let constant =
            opcode("01001110000000") | (5 << 39) | (1 << 34) | (4 << 20) | (1 << 8) | PT | 6;
        let bytes = program(1, 0, &[register, cbcc, immediate, constant, exit()]);
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        validate("xmad", &shader.words);
    }

    #[test]
    fn compute_programs_use_ids_shared_memory_and_barriers() {
        // S2R R0, SR_TID_X; S2R R1, SR_CTAID_X; STS [R0], R1; BAR.SYNC 0;
        // LDS R2, [R0 + 4]; EXIT.
        let sts = opcode("11101111010110") | (4 << 48) | PT | 1;
        let bar = opcode("11110000101010") | PT;
        let lds = opcode("11101111010010") | (4 << 48) | (4 << 20) | PT | 2;
        let bytes = program(5, 0, &[s2r(0, 0x21), s2r(1, 0x25), sts, bar, lds, exit()]);
        let environment = Environment {
            local_size: [8, 4, 1],
            shared_memory_bytes: 0x100,
            ..Environment::default()
        };
        let shader = translate_compute(&bytes[HEADER_SIZE..], &environment).expect("translates");
        assert_eq!(shader.stage, Stage::Compute);
        validate("compute", &shader.words);
    }

    #[test]
    fn shfl_reads_a_lane_of_the_quad() {
        // LM3's half-resolution pass: IPA R10, a[0x70]; SHFL.IDX R2, R10,
        // RZ, 0x1C03 (lane 0 of each 2x2 quad).
        let shfl = 0xef17_700c_2ff7_0a02;
        let bytes = program(5, 0, &[ipa(10, 0x70), shfl, exit()]);
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        // Fine derivatives (OpDPdxFine/OpDPdyFine, 4 words) move the value
        // across the quad: no subgroup operation, which the Adreno 650 only
        // runs in compute shaders.
        assert!(shader.words.contains(&((4 << 16) | 210)));
        assert!(shader.words.contains(&((4 << 16) | 211)));
        assert!(!shader
            .words
            .iter()
            .any(|word| matches!(word & 0xFFFF, 345 | 365 | 366)));
        validate("shfl-quad-broadcast", &shader.words);
    }

    #[test]
    fn kil_discards_after_the_dispatch_loop() {
        // IPA R0, a[0x70]; IPA R1, a[0x74]; FSETP.LT P0, PT, R0, R1;
        // @P0 KIL; EXIT: an alpha test's shape.
        let kil = opcode("11100011001100") | 0xF;
        let bytes = program(
            5,
            0,
            &[ipa(0, 0x70), ipa(1, 0x74), fsetp_lt(0, 1), kil, exit()],
        );
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        validate("kil", &shader.words);
        // One OpKill (opcode 252), after the dispatch loop's OpLoopMerge
        // (246) and its merge block: none inside the loop.
        let words = &shader.words;
        let kills: Vec<usize> = (0..words.len())
            .filter(|&at| words[at] == ((1 << 16) | 252))
            .collect();
        assert_eq!(kills.len(), 1);
        let loop_merge = (0..words.len())
            .find(|&at| words[at] == ((4 << 16) | 246))
            .expect("a dispatch loop");
        let merge_label = words[loop_merge + 1];
        let merge_at = (0..words.len())
            .find(|&at| words[at] == ((2 << 16) | 248) && words[at + 1] == merge_label)
            .expect("the loop's merge block");
        assert!(kills[0] > merge_at, "OpKill is after the loop");
    }

    #[test]
    fn fswzadd_takes_its_quad_lane_from_the_pixel() {
        // IPA R0, a[0x70]; IPA R1, a[0x74]; FSWZADD R2, R0, R1, 0x99 (the
        // pattern derivative helpers use): no subgroup lane id, which the
        // Adreno 650 does not give fragment shaders.
        let fswzadd = opcode("01010000111110") | (0x99 << 28) | (1 << 20) | 2;
        let bytes = program(5, 0, &[ipa(0, 0x70), ipa(1, 0x74), fswzadd, exit()]);
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        assert!(!shader
            .words
            .iter()
            .any(|word| matches!(word & 0xFFFF, 345 | 365 | 366)));
        // BuiltIn SubgroupLocalInvocationId (41) is never declared.
        assert!(!shader
            .words
            .windows(4)
            .any(|words| words[0] == ((4 << 16) | 71) && words[2] == 11 && words[3] == 41));
        validate("fswzadd", &shader.words);
    }

    #[test]
    fn screen_positions_texel_fetches_and_sizes_take_the_resolution_scale() {
        // IPA R0, a[0x70]; IPA R1, a[0x74] (the pixel position); TLDS.LZ
        // R2, R0, R1 (2D, handle 0x10); TXQ R4, size of the same texture.
        let tlds = (0b1101101u64 << 57) | (2 << 53) | (4 << 36) | (0xFF << 28) | (1 << 20) | PT | 2;
        let txq =
            (0b1101111101001u64 << 51) | (4 << 36) | (0x3 << 31) | (1 << 22) | PT | (0xFF << 8) | 4;
        let bytes = program(5, 0, &[ipa(0, 0x70), ipa(1, 0x74), tlds, txq, exit()]);
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        assert_eq!(shader.textures.len(), 1);
        // One push-constant block: OpVariable (4 words, opcode 59) in the
        // PushConstant storage class.
        let push_constants = shader
            .words
            .windows(4)
            .filter(|words| words[0] == (4 << 16) | 59 && words[3] == storage::PUSH_CONSTANT)
            .count();
        assert_eq!(push_constants, 1);
        validate("rescale", &shader.words);

        // A program that uses none of them declares no block.
        let bytes = program(1, 0, &[ald(0, 0x80, 4), ast(0, 0x70, 4), exit()]);
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        assert!(!shader
            .words
            .windows(4)
            .any(|words| words[0] == (4 << 16) | 59 && words[3] == storage::PUSH_CONSTANT));
    }

    #[test]
    fn a_vertex_program_passes_its_attribute_through() {
        let bytes = program(1, 0, &[ald(0, 0x80, 4), ast(0, 0x70, 4), exit()]);
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        assert_eq!(shader.stage, Stage::Vertex);
        assert_eq!(shader.descriptor_set, VERTEX_SET);
        assert_eq!(shader.inputs, [0]);
        assert!(shader.outputs.is_empty(), "the position is a built-in");
        validate("vertex", &shader.words);

        // Writing only x and y of the position and of varying 0 still gives
        // every output defaults for the rest: (0, 0, 0, 1).
        let bytes = program(
            1,
            0,
            &[ald(0, 0x80, 4), ast(0, 0x70, 2), ast(2, 0x80, 2), exit()],
        );
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        assert_eq!(shader.outputs, [0]);
        assert!(
            shader.words.contains(&1.0f32.to_bits()),
            "the (0, 0, 0, 1) default is a constant of the module"
        );
        validate("partial-vertex", &shader.words);
    }

    #[test]
    fn clip_distances_clip_only_when_the_draw_enables_them() {
        // AST R0, R1 -> clip distances 0 and 1 (attributes 0x2C0, 0x2C4).
        let bytes = program(
            1,
            0,
            &[ald(0, 0x80, 4), ast(0, 0x70, 4), ast(0, 0x2C0, 2), exit()],
        );
        // OpCapability ClipDistance.
        let declares = |words: &[u32]| words.windows(2).any(|pair| pair == [0x0002_0011, 32]);
        let enabled = Environment {
            clip_distances: 0b01,
            ..Environment::default()
        };
        let shader = translate(&bytes, &enabled).expect("translates");
        assert!(declares(&shader.words), "an enabled distance is written");
        validate("clip-distance", &shader.words);
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        assert!(!declares(&shader.words), "disabled distances are dropped");
        validate("no-clip-distance", &shader.words);
    }

    /// LDL/STL rd, [ra + offset] of `size` (4: 32 bits, 0: U8, 3: S16).
    fn local(store: bool, rd: u8, ra: u8, offset: u32, size: u64) -> u64 {
        opcode(if store {
            "11101111010100"
        } else {
            "11101111010000"
        }) | (size << 48)
            | (u64::from(offset) << 20)
            | PT
            | (u64::from(ra) << 8)
            | u64::from(rd)
    }

    #[test]
    fn local_memory_loads_and_stores_words_and_parts_of_words() {
        // STL [RZ + 0x10], R0; LDL R1, [RZ + 0x10]; STL.U8 [RZ + 0x21], R0;
        // LDL.S16 R2, [RZ + 0x22]; then position = (R1, R2, ...).
        let mut bytes = program(
            1,
            0,
            &[
                ald(0, 0x80, 4),
                local(true, 0, 0xFF, 0x10, 4),
                local(false, 1, 0xFF, 0x10, 4),
                local(true, 0, 0xFF, 0x21, 0),
                local(false, 2, 0xFF, 0x22, 3),
                ast(1, 0x70, 2),
                exit(),
            ],
        );
        // 0x40 bytes of local memory (SPH common word 1).
        bytes[4..8].copy_from_slice(&0x40u32.to_le_bytes());
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        validate("local-memory", &shader.words);
    }

    #[test]
    fn a_fragment_program_writes_its_color_registers() {
        let bytes = program(
            5,
            0xF,
            &[
                ipa(0, 0x80),
                mov32i(1, 0),
                mov32i(2, 0),
                mov32i(3, 1.0f32.to_bits()),
                exit(),
            ],
        );
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        assert_eq!(shader.descriptor_set, FRAGMENT_SET);
        assert_eq!(shader.inputs, [0]);
        assert_eq!(shader.color_outputs, [0]);
        validate("fragment", &shader.words);
    }

    #[test]
    fn sync_returns_to_the_join_point_of_its_ssy() {
        // 0x58 FSETP; 0x60 SSY; 0x68 @P0 FMUL32I; (0x70); 0x78 @P0 SYNC;
        // 0x80 FMUL32I; 0x88 SYNC; (0x90); 0x98 EXIT. The SSY's offset lands
        // on the bundle at 0x90, so the join point is its first instruction.
        let bytes = program(
            5,
            0x1,
            &[
                fsetp_lt(0, 1),
                ssy(0x28),
                fmul32i(0, 0, 2.0, 0),
                sync(0),
                fmul32i(0, 0, 3.0, 7),
                sync(7),
                exit(),
            ],
        );
        assert_eq!(branch_target(0x60, ssy(0x28)), 0x98);
        let flow = Flow::analyze(&bytes, &[]).expect("analyzes");
        assert_eq!(
            flow.sync_targets,
            BTreeMap::from([(0x78, 0x98), (0x88, 0x98)])
        );
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        validate("sync", &shader.words);
    }

    fn pbk(offset: u32) -> u64 {
        opcode("11100010101000") | (u64::from(offset) << 20)
    }

    fn brk(predicate: u64) -> u64 {
        opcode("11100011010000") | (predicate << 16) | 0xF
    }

    /// `LOP.op Rd, Ra, Rb` setting `dst_pred` from the result by `pred_op`
    /// (2: zero, 3: not zero).
    fn lop(rd: u8, ra: u8, rb: u8, op: u64, pred_op: u64, dst_pred: u64) -> u64 {
        opcode("01011100010000")
            | (dst_pred << 48)
            | (pred_op << 44)
            | (op << 41)
            | (u64::from(rb) << 20)
            | PT
            | (u64::from(ra) << 8)
            | u64::from(rd)
    }

    fn s2r(rd: u8, special: u64) -> u64 {
        opcode("11110000110010") | (special << 20) | PT | u64::from(rd)
    }

    #[test]
    fn brk_leaves_the_loop_of_its_pbk() {
        // 0x58 FSETP; 0x60 PBK (to 0x98); 0x68 @P0 BRK; (0x70); 0x78 FMUL32I;
        // 0x80 BRK; 0x88 EXIT; (0x90); 0x98 LOP.AND R1 (P1: not zero);
        // 0xa0 S2R R2, SR_Y_DIRECTION; 0xa8 EXIT.
        let bytes = program(
            5,
            0x1,
            &[
                fsetp_lt(0, 1),
                pbk(0x28),
                brk(0),
                fmul32i(0, 0, 2.0, 7),
                brk(7),
                exit(),
                lop(1, 0, 3, 0, 3, 1),
                s2r(2, 0x12),
                exit(),
            ],
        );
        let flow = Flow::analyze(&bytes, &[]).expect("analyzes");
        assert_eq!(
            flow.sync_targets,
            BTreeMap::from([(0x68, 0x98), (0x80, 0x98)])
        );
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        assert!(
            shader.words.contains(&1.0f32.to_bits()),
            "SR_Y_DIRECTION reads +1.0"
        );
        validate("break", &shader.words);
    }

    #[test]
    fn title_screen_instructions_translate() {
        // Words from the title screen's programs: IADD32I R24, R23, -1;
        // IMNMX.U32 R24, R24, 3; LDC R24, c1[R24]; LOP3 R0, R8, R10, R9
        // (a | b | c); TXD R0, R4, R0 (two components, handle 16).
        let bytes = program(
            5,
            0x1,
            &[
                0x1c0f_ffff_fff7_1718,
                0x3820_0380_0037_1818,
                0xef94_0010_0007_1818,
                0x5be7_048f_e0a7_0800,
                0xde3a_0101_a007_0400,
                exit(),
            ],
        );
        let shader = translate(&bytes, &Environment::default()).expect("translates");
        validate("title", &shader.words);
    }

    #[test]
    fn integer_color_targets_get_the_registers_bits() {
        let bytes = program(5, 0x1, &[mov32i(0, 7), exit()]);
        let mut environment = Environment::default();
        environment.color_targets[0] = AttributeType::UnsignedInt;
        let shader = translate(&bytes, &environment).expect("translates");
        validate("uint-target", &shader.words);
        environment.color_targets[0] = AttributeType::SignedInt;
        let shader = translate(&bytes, &environment).expect("translates");
        validate("sint-target", &shader.words);
    }

    #[test]
    fn an_unknown_word_is_reported_with_its_address() {
        let bytes = program(5, 0x1, &[0x0000_0000_0000_0000, exit()]);
        match translate(&bytes, &Environment::default()) {
            Err(TranslateError::UnknownInstruction { address, .. }) => assert_eq!(address, 0x58),
            other => panic!("expected an unknown instruction, got {other:?}"),
        }
    }
}
