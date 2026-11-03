//! Compute programs that only compute integers and access memory (constant
//! buffers and global memory), run on the CPU against guest memory.
//!
//! Such programs resolve queries and counters the title's CPU code then
//! reads (LM3 sums occlusion query counters into predicates this way), so
//! their results must land in guest memory, where a host GPU dispatch would
//! not put them. Every thread of the grid runs in turn; programs that use
//! shared memory, barriers, textures or floating point are left to the host
//! GPU (`run` refuses them before running anything).
//!
//! Field positions follow the hardware encodings as Ryujinx and yuzu decode
//! them, as in `translate`.

use super::compute::ComputeDispatch;
use super::render::GpuMemory;
use super::sph::{self, Instruction, Operand, HEADER_SIZE, INSTRUCTION_SIZE, PREDICATE_TRUE};

const REGISTER_ZERO: u8 = 255;
/// Instructions one thread may run before it is stopped (a runaway loop).
const STEP_LIMIT: u64 = 1 << 20;

fn field(word: u64, start: u32, len: u32) -> u32 {
    ((word >> start) & ((1u64 << len) - 1)) as u32
}

fn flag(word: u64, bit: u32) -> bool {
    word & (1 << bit) != 0
}

fn sext(value: u32, bits: u32) -> u32 {
    (((value << (32 - bits)) as i32) >> (32 - bits)) as u32
}

/// The instructions this interpreter runs.
fn supported(instruction: &Instruction) -> bool {
    matches!(
        instruction.name,
        "MOV"
            | "MOV32I"
            | "S2R"
            | "ISCADD"
            | "SHF.R"
            | "SHF.L"
            | "LOP32I"
            | "LOP"
            | "IADD"
            | "IADD32I"
            | "IADD3"
            | "ISETP"
            | "XMAD"
            | "SHL"
            | "SHR"
            | "LDC"
            | "LDG"
            | "STG"
            | "BRA"
            | "EXIT"
            | "NOP"
            | "DEPBAR"
            | "SSY"
            | "SYNC"
            | "PBK"
            | "BRK"
            | "IMNMX"
            | "BFE"
            | "SEL"
    )
}

/// The slot after `offset`, skipping scheduling words.
fn next_slot(offset: u64) -> u64 {
    let mut next = offset + INSTRUCTION_SIZE as u64;
    if sph::is_schedule_word(next) {
        next += INSTRUCTION_SIZE as u64;
    }
    next
}

/// A branch's target: relative to the next slot; a target on a scheduling
/// word means the bundle's first instruction.
fn branch_target(offset: u64, word: u64) -> u64 {
    let displacement = sext(field(word, 20, 24), 24) as i32;
    let mut target = (offset as i64 + INSTRUCTION_SIZE as i64 + i64::from(displacement)) as u64;
    if sph::is_schedule_word(target) {
        target += INSTRUCTION_SIZE as u64;
    }
    target
}

/// The program behind a zeroed header, as the decoder's offsets expect.
fn with_header(code: &[u8]) -> Vec<u8> {
    let mut program = vec![0u8; HEADER_SIZE];
    program.extend_from_slice(code);
    program
}

fn word_at(program: &[u8], offset: u64) -> Option<u64> {
    let at = usize::try_from(offset).ok()?;
    Some(u64::from_le_bytes(
        program.get(at..at + INSTRUCTION_SIZE)?.try_into().ok()?,
    ))
}

/// Up to `len` bytes from `address`, stopping where the mapping ends.
fn read_mapped(memory: &dyn GpuMemory, address: u64, len: usize) -> Vec<u8> {
    if let Some(bytes) = memory.read_gpu(address, len) {
        return bytes;
    }
    let mut bytes = Vec::new();
    while bytes.len() < len {
        let chunk = (len - bytes.len()).min(0x100);
        match memory.read_gpu(address + bytes.len() as u64, chunk) {
            Some(more) => bytes.extend_from_slice(&more),
            None => break,
        }
    }
    bytes
}

/// Why a program is left to the host GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused(pub String);

/// Every instruction reachable from the entry, or why the program cannot
/// run here.
fn reachable(program: &[u8]) -> Result<std::collections::BTreeMap<u64, Instruction>, Refused> {
    let mut found = std::collections::BTreeMap::new();
    let mut pending = vec![next_slot(HEADER_SIZE as u64 - INSTRUCTION_SIZE as u64)];
    while let Some(offset) = pending.pop() {
        if found.contains_key(&offset) {
            continue;
        }
        let word =
            word_at(program, offset).ok_or_else(|| Refused(format!("no code at {offset:#x}")))?;
        let instruction = sph::decode_at(offset, word);
        if !supported(&instruction) {
            return Err(Refused(format!("{} at {offset:#x}", instruction.name)));
        }
        let unconditional = instruction.pred == PREDICATE_TRUE && !instruction.pred_inv;
        match instruction.name {
            "EXIT" | "SYNC" | "BRK" if unconditional => {}
            "BRA" => {
                pending.push(branch_target(offset, word));
                if !unconditional {
                    pending.push(next_slot(offset));
                }
            }
            "SSY" | "PBK" => {
                pending.push(branch_target(offset, word));
                pending.push(next_slot(offset));
            }
            _ => pending.push(next_slot(offset)),
        }
        found.insert(offset, instruction);
    }
    Ok(found)
}

/// One thread's state.
struct Thread {
    registers: [u32; 256],
    predicates: [bool; 8],
    /// The carry (0..=2 after a three-input add) and zero flags.
    carry: u32,
    zero: bool,
    /// Join points of the open SSY and PBK regions.
    stack: Vec<u64>,
}

impl Thread {
    fn get(&self, index: u8) -> u32 {
        if index == REGISTER_ZERO {
            0
        } else {
            self.registers[index as usize]
        }
    }

    fn set(&mut self, index: u8, value: u32) {
        if index != REGISTER_ZERO {
            self.registers[index as usize] = value;
        }
    }

    /// Register `base + index` of a multi-register operand (RZ stays RZ).
    fn get_at(&self, base: u8, index: u32) -> u32 {
        if base == REGISTER_ZERO {
            0
        } else {
            self.get(base.wrapping_add(index as u8))
        }
    }

    fn set_at(&mut self, base: u8, index: u32, value: u32) {
        if base != REGISTER_ZERO {
            self.set(base.wrapping_add(index as u8), value);
        }
    }

    fn predicate(&self, index: u32, negate: bool) -> bool {
        let value = index == u32::from(PREDICATE_TRUE) || self.predicates[index as usize];
        value != negate
    }

    fn set_predicate(&mut self, index: u32, value: bool) {
        if index != u32::from(PREDICATE_TRUE) {
            self.predicates[index as usize] = value;
        }
    }
}

struct Machine<'a> {
    memory: &'a mut dyn GpuMemory,
    /// The program's instructions, decoded once, by 8-byte slot.
    code: Vec<Option<Instruction>>,
    /// The bound constant buffers' contents (fixed for the dispatch).
    constants: [Vec<u8>; 8],
}

impl Machine<'_> {
    fn cbuf(&self, slot: u32, offset: u32) -> u32 {
        let at = offset as usize;
        self.constants
            .get(slot as usize)
            .and_then(|bytes| bytes.get(at..at + 4))
            .map_or(0, |word| {
                u32::from_le_bytes(word.try_into().unwrap_or_default())
            })
    }

    /// Operand B: a register, a sign-extended 20-bit immediate, a 32-bit
    /// immediate or a constant buffer word.
    fn operand_b(&self, thread: &Thread, instruction: &Instruction) -> u32 {
        let word = instruction.word;
        match instruction.src_b {
            Operand::Register(_) => thread.get(field(word, 20, 8) as u8),
            Operand::Immediate(_) => {
                let value = field(word, 20, 19);
                if flag(word, 56) {
                    value | 0xFFF8_0000
                } else {
                    value
                }
            }
            Operand::Immediate32(_) => field(word, 20, 32),
            Operand::Constant { .. } => self.cbuf(field(word, 34, 5), field(word, 20, 14) * 4),
            _ => thread.get(field(word, 39, 8) as u8),
        }
    }

    /// Run one thread to its exit; the instructions it ran.
    fn run_thread(&mut self, ids: ([u32; 3], [u32; 3])) -> Result<u64, Refused> {
        let mut thread = Thread {
            registers: [0; 256],
            predicates: [false; 8],
            carry: 0,
            zero: false,
            stack: Vec::new(),
        };
        let mut pc = next_slot(HEADER_SIZE as u64 - INSTRUCTION_SIZE as u64);
        for steps in 1..=STEP_LIMIT {
            let instruction = self
                .code
                .get((pc / INSTRUCTION_SIZE as u64) as usize)
                .copied()
                .flatten()
                .ok_or_else(|| Refused(format!("no code at {pc:#x}")))?;
            let next = next_slot(pc);
            if !thread.predicate(u32::from(instruction.pred), instruction.pred_inv) {
                pc = next;
                continue;
            }
            match self.step(&mut thread, &instruction, pc, ids)? {
                Flow::Next => pc = next,
                Flow::Jump(target) => pc = target,
                Flow::Exit => return Ok(steps),
            }
        }
        Err(Refused("a thread ran too long".to_string()))
    }

    fn step(
        &mut self,
        thread: &mut Thread,
        instruction: &Instruction,
        pc: u64,
        (tid, ctaid): ([u32; 3], [u32; 3]),
    ) -> Result<Flow, Refused> {
        let word = instruction.word;
        let (rd, ra) = (field(word, 0, 8) as u8, field(word, 8, 8) as u8);
        let a = thread.get(ra);
        match instruction.name {
            "MOV" => {
                let value = self.operand_b(thread, instruction);
                thread.set(rd, value);
            }
            "MOV32I" => thread.set(rd, field(word, 20, 32)),
            "S2R" => {
                let value = match field(word, 20, 8) {
                    0x00 => tid[0] % 32,
                    0x20 => tid[0] | (tid[1] << 16) | (tid[2] << 26),
                    special @ 0x21..=0x23 => tid[(special - 0x21) as usize],
                    special @ 0x25..=0x27 => ctaid[(special - 0x25) as usize],
                    special => return Err(Refused(format!("S2R of {special:#x}"))),
                };
                thread.set(rd, value);
            }
            "NOP" | "DEPBAR" => {}
            "EXIT" => return Ok(Flow::Exit),
            "BRA" => {
                if field(word, 0, 5) != 0xF {
                    return Err(Refused("BRA on a condition code".to_string()));
                }
                return Ok(Flow::Jump(branch_target(pc, word)));
            }
            "SSY" | "PBK" => thread.stack.push(branch_target(pc, word)),
            "SYNC" | "BRK" => {
                let target = thread
                    .stack
                    .pop()
                    .ok_or_else(|| Refused(format!("{} without its region", instruction.name)))?;
                return Ok(Flow::Jump(target));
            }
            "ISCADD" => {
                let b = self.operand_b(thread, instruction);
                let (a, b) = negate_pair(a, b, field(word, 48, 2));
                thread.set(rd, (a << field(word, 39, 5)).wrapping_add(b));
            }
            "SHF.R" | "SHF.L" => {
                let shift = match instruction.src_b {
                    Operand::Register(_) => thread.get(field(word, 20, 8) as u8),
                    _ => field(word, 20, 6),
                };
                let high = thread.get(field(word, 39, 8) as u8);
                let wide = field(word, 37, 2) >= 2;
                let shift = if flag(word, 50) {
                    shift & if wide { 63 } else { 31 }
                } else {
                    shift.min(if wide { 64 } else { 32 })
                };
                let value = (u64::from(high) << 32) | u64::from(a);
                let result = if instruction.name == "SHF.R" {
                    if shift >= 64 {
                        0
                    } else {
                        (value >> shift) as u32
                    }
                } else if shift >= 64 {
                    0
                } else {
                    ((value << shift) >> 32) as u32
                };
                thread.set(rd, result);
            }
            "LOP32I" => {
                let a = if flag(word, 55) { !a } else { a };
                let b = field(word, 20, 32);
                let b = if flag(word, 56) { !b } else { b };
                let value = logic(field(word, 53, 2), a, b);
                thread.set(rd, value);
            }
            "LOP" => {
                let b = self.operand_b(thread, instruction);
                let a = if flag(word, 39) { !a } else { a };
                let b = if flag(word, 40) { !b } else { b };
                let value = logic(field(word, 41, 2), a, b);
                let predicate = field(word, 48, 3);
                if predicate != u32::from(PREDICATE_TRUE) {
                    let result = match field(word, 44, 2) {
                        0 => false,
                        1 => true,
                        2 => value == 0,
                        _ => value != 0,
                    };
                    thread.set_predicate(predicate, result);
                }
                thread.set(rd, value);
            }
            "IADD" | "IADD32I" | "IADD3" => self.add(thread, instruction)?,
            "ISETP" => {
                let b = self.operand_b(thread, instruction);
                let signed = flag(word, 48);
                let extended = flag(word, 43);
                let compare = compare(
                    field(word, 49, 3),
                    a,
                    b,
                    signed,
                    extended.then_some((thread.carry != 0, thread.zero)),
                );
                let other = thread.predicate(field(word, 39, 3), flag(word, 42));
                let combine = |value: bool| match field(word, 45, 2) {
                    0 => value && other,
                    1 => value || other,
                    _ => value != other,
                };
                thread.set_predicate(field(word, 3, 3), combine(compare));
                thread.set_predicate(field(word, 0, 3), combine(!compare));
            }
            "IMNMX" => {
                let b = self.operand_b(thread, instruction);
                let minimum = thread.predicate(field(word, 39, 3), flag(word, 42));
                let value = if flag(word, 48) {
                    let (a, b) = (a as i32, b as i32);
                    (if minimum { a.min(b) } else { a.max(b) }) as u32
                } else if minimum {
                    a.min(b)
                } else {
                    a.max(b)
                };
                thread.set(rd, value);
            }
            "SEL" => {
                let b = self.operand_b(thread, instruction);
                let choose_a = thread.predicate(field(word, 39, 3), flag(word, 42));
                thread.set(rd, if choose_a { a } else { b });
            }
            "BFE" => {
                let b = self.operand_b(thread, instruction);
                let (position, length) = (b & 0xFF, (b >> 8) & 0xFF);
                let signed = flag(word, 48);
                let value = if length == 0 || position >= 32 {
                    0
                } else {
                    let length = length.min(32 - position);
                    let bits = (u64::from(a) >> position) & ((1u64 << length) - 1);
                    if signed {
                        sext(bits as u32, length)
                    } else {
                        bits as u32
                    }
                };
                thread.set(rd, value);
            }
            "XMAD" => {
                let value = self.xmad(thread, instruction)?;
                thread.set(rd, value);
            }
            "SHL" => {
                let b = self.operand_b(thread, instruction);
                let shift = if flag(word, 39) { b & 31 } else { b };
                thread.set(rd, if shift >= 32 { 0 } else { a << shift });
            }
            "SHR" => {
                let b = self.operand_b(thread, instruction);
                let shift = if flag(word, 39) { b & 31 } else { b };
                let value = if flag(word, 48) {
                    ((a as i32) >> shift.min(31)) as u32
                } else if shift >= 32 {
                    0
                } else {
                    a >> shift
                };
                thread.set(rd, value);
            }
            "LDC" => {
                if field(word, 44, 2) != 0 {
                    return Err(Refused("LDC address modes".to_string()));
                }
                let slot = field(word, 36, 5);
                let address = a.wrapping_add(sext(field(word, 20, 16), 16));
                match field(word, 48, 3) {
                    4 => thread.set(rd, self.cbuf(slot, address)),
                    5 => {
                        thread.set_at(rd, 0, self.cbuf(slot, address));
                        thread.set_at(rd, 1, self.cbuf(slot, address.wrapping_add(4)));
                    }
                    size => return Err(Refused(format!("LDC size {size}"))),
                }
            }
            "LDG" | "STG" => {
                let words = match field(word, 48, 3) {
                    4 => 1,
                    5 => 2,
                    6 => 4,
                    size => return Err(Refused(format!("{} size {size}", instruction.name))),
                };
                let base = if flag(word, 45) {
                    (u64::from(thread.get_at(ra, 1)) << 32) | u64::from(a)
                } else {
                    u64::from(a)
                };
                let address =
                    base.wrapping_add(i64::from(sext(field(word, 20, 24), 24) as i32) as u64);
                for index in 0..words {
                    let at = address + u64::from(index) * 4;
                    if instruction.name == "LDG" {
                        let value = self.memory.read_u32(at).unwrap_or(0);
                        thread.set_at(rd, index, value);
                    } else {
                        let value = thread.get_at(rd, index);
                        self.memory.write_gpu(at, &value.to_le_bytes());
                    }
                }
            }
            other => return Err(Refused(other.to_string())),
        }
        Ok(Flow::Next)
    }

    /// `IADD`, `IADD32I`, `IADD3`: negated operands add one each unless the
    /// add is extended (`.X`), which adds the carry instead; `.CC` keeps the
    /// carry out (up to 2 for three operands) and whether the result is zero.
    fn add(&mut self, thread: &mut Thread, instruction: &Instruction) -> Result<(), Refused> {
        let word = instruction.word;
        let rd = field(word, 0, 8) as u8;
        let a = thread.get(field(word, 8, 8) as u8);
        let (operands, negations, extended, write_cc): (Vec<u32>, Vec<bool>, bool, bool) =
            match instruction.name {
                "IADD3" => {
                    let b = self.operand_b(thread, instruction);
                    let c = thread.get(field(word, 39, 8) as u8);
                    if matches!(instruction.src_b, Operand::Register(_))
                        && (field(word, 31, 8) != 0)
                    {
                        return Err(Refused("IADD3 with parts or shifts".to_string()));
                    }
                    (
                        vec![a, b, c],
                        vec![flag(word, 51), flag(word, 50), flag(word, 49)],
                        flag(word, 48),
                        flag(word, 47),
                    )
                }
                "IADD32I" => {
                    // Bits 55 and 56: negate the immediate, negate A.
                    if flag(word, 55) && flag(word, 56) {
                        return Err(Refused("IADD32I.PO".to_string()));
                    }
                    (
                        vec![a, field(word, 20, 32)],
                        vec![flag(word, 56), flag(word, 55)],
                        flag(word, 53),
                        flag(word, 52),
                    )
                }
                _ => {
                    let b = self.operand_b(thread, instruction);
                    let mode = field(word, 48, 2);
                    if mode == 3 {
                        return Err(Refused("IADD.PO".to_string()));
                    }
                    (
                        vec![a, b],
                        vec![mode == 2, mode == 1],
                        flag(word, 43),
                        flag(word, 47),
                    )
                }
            };
        let mut wide: u64 = 0;
        for (value, negate) in operands.iter().zip(&negations) {
            wide += u64::from(if *negate { !*value } else { *value });
            if *negate && !extended {
                wide += 1;
            }
        }
        if extended {
            wide += u64::from(thread.carry);
        }
        let result = wide as u32;
        if write_cc {
            let zero = result == 0;
            thread.zero = if extended { zero && thread.zero } else { zero };
            thread.carry = (wide >> 32) as u32;
        }
        thread.set(rd, result);
        Ok(())
    }

    /// `XMAD` as `translate` lowers it.
    fn xmad(&self, thread: &Thread, instruction: &Instruction) -> Result<u32, Refused> {
        let word = instruction.word;
        if flag(word, 38) {
            return Err(Refused("XMAD.X".to_string()));
        }
        let a = thread.get(field(word, 8, 8) as u8);
        let cbuf = || self.cbuf(field(word, 34, 5), field(word, 20, 14) * 4);
        let (b, c, b_high, c_mode) = match instruction.src_b {
            Operand::Register(_) => (
                thread.get(field(word, 20, 8) as u8),
                thread.get(field(word, 39, 8) as u8),
                flag(word, 35),
                field(word, 50, 3),
            ),
            Operand::Immediate(_) => (
                field(word, 20, 16),
                thread.get(field(word, 39, 8) as u8),
                false,
                field(word, 50, 3),
            ),
            Operand::Constant { .. } => (
                cbuf(),
                thread.get(field(word, 39, 8) as u8),
                flag(word, 52),
                field(word, 50, 2),
            ),
            _ => (
                thread.get(field(word, 39, 8) as u8),
                cbuf(),
                flag(word, 52),
                field(word, 50, 2),
            ),
        };
        let half = |value: u32, high: bool, signed: bool| match (high, signed) {
            (true, true) => ((value as i32) >> 16) as u32,
            (true, false) => value >> 16,
            (false, true) => sext(value & 0xFFFF, 16),
            (false, false) => value & 0xFFFF,
        };
        let mut product =
            half(a, flag(word, 53), flag(word, 48)).wrapping_mul(half(b, b_high, flag(word, 49)));
        if flag(word, 36) {
            product <<= 16;
        }
        let addend = match c_mode {
            0 => c,
            1 => c & 0xFFFF,
            2 => c >> 16,
            4 => c.wrapping_add(b << 16),
            _ => return Err(Refused("XMAD.CSFU".to_string())),
        };
        let mut value = product.wrapping_add(addend);
        if flag(word, 37) {
            value = (value & 0xFFFF) | (b << 16);
        }
        Ok(value)
    }
}

enum Flow {
    Next,
    Jump(u64),
    Exit,
}

/// Negations of `ISCADD`'s operands (bit 0 negates B, bit 1 A; both is
/// plus one).
fn negate_pair(a: u32, b: u32, mode: u32) -> (u32, u32) {
    match mode {
        1 => (a, b.wrapping_neg()),
        2 => (a.wrapping_neg(), b),
        3 => (a, b.wrapping_add(1)),
        _ => (a, b),
    }
}

/// `LOP` and `LOP32I` operations: and, or, xor, pass B.
fn logic(operation: u32, a: u32, b: u32) -> u32 {
    match operation {
        0 => a & b,
        1 => a | b,
        2 => a ^ b,
        _ => b,
    }
}

/// An `ISETP` comparison (F, LT, EQ, LE, GT, NE, GE, T). An extended
/// comparison continues a 64-bit one whose low words set the carry (no
/// borrow) and zero flags.
fn compare(operation: u32, a: u32, b: u32, signed: bool, low: Option<(bool, bool)>) -> bool {
    let ordering = if signed {
        (a as i32).cmp(&(b as i32))
    } else {
        a.cmp(&b)
    };
    use std::cmp::Ordering::{Equal, Greater, Less};
    let Some((carry, zero)) = low else {
        return match operation {
            0 => false,
            1 => ordering == Less,
            2 => ordering == Equal,
            3 => ordering != Greater,
            4 => ordering == Greater,
            5 => ordering != Equal,
            6 => ordering != Less,
            _ => true,
        };
    };
    let equal = ordering == Equal;
    match operation {
        0 => false,
        1 => ordering == Less || (equal && !carry),
        2 => equal && zero,
        3 => ordering == Less || (equal && (zero || !carry)),
        4 => ordering == Greater || (equal && !zero && carry),
        5 => !equal || !zero,
        6 => ordering == Greater || (equal && carry),
        _ => true,
    }
}

/// Run a dispatch on the CPU: every thread of every block. Refuses (before
/// running anything) programs with instructions this module does not run;
/// otherwise the instructions run, over all threads.
pub fn run(
    memory: &mut dyn GpuMemory,
    dispatch: &ComputeDispatch,
    code: &[u8],
) -> Result<u64, Refused> {
    let program = with_header(code);
    let mut decoded = vec![None; program.len() / INSTRUCTION_SIZE];
    for (offset, instruction) in reachable(&program)? {
        decoded[(offset / INSTRUCTION_SIZE as u64) as usize] = Some(instruction);
    }
    let [grid, block] = [dispatch.grid, dispatch.block].map(|size| size.map(|count| count.max(1)));
    let threads = u64::from(grid[0])
        * u64::from(grid[1])
        * u64::from(grid[2])
        * u64::from(block[0])
        * u64::from(block[1])
        * u64::from(block[2]);
    if threads > 1 << 20 {
        return Err(Refused(format!("{threads} threads")));
    }
    let constants = dispatch.constant_buffers.map(|buffer| {
        buffer.map_or_else(Vec::new, |(address, size)| {
            read_mapped(&*memory, address, size.min(0x1_0000) as usize)
        })
    });
    let mut steps = 0;
    let mut machine = Machine {
        memory,
        code: decoded,
        constants,
    };
    for cz in 0..grid[2] {
        for cy in 0..grid[1] {
            for cx in 0..grid[0] {
                for tz in 0..block[2] {
                    for ty in 0..block[1] {
                        for tx in 0..block[0] {
                            steps += machine.run_thread(([tx, ty, tz], [cx, cy, cz]))?;
                        }
                    }
                }
            }
        }
    }
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxwell::state::PoolState;

    const PT: u64 = 7 << 16;

    /// Guest memory as one buffer from address 0.
    struct Bytes(Vec<u8>);

    impl GpuMemory for Bytes {
        fn read_gpu(&self, address: u64, len: usize) -> Option<Vec<u8>> {
            let at = usize::try_from(address).ok()?;
            self.0.get(at..at + len).map(<[u8]>::to_vec)
        }

        fn write_gpu(&mut self, address: u64, bytes: &[u8]) -> bool {
            let at = address as usize;
            match self.0.get_mut(at..at + bytes.len()) {
                Some(target) => {
                    target.copy_from_slice(bytes);
                    true
                }
                None => false,
            }
        }
    }

    /// Instructions laid out in bundles: a scheduling word, then three.
    fn code(instructions: &[u64]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for bundle in instructions.chunks(3) {
            bytes.extend_from_slice(&0u64.to_le_bytes());
            for word in bundle {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn every_thread_stores_its_global_index() {
        let s2r = |rd: u64, special: u64| 0xF0C8_0000_0000_0000 | (special << 20) | PT | rd;
        // R0 = ctaid.x * 4 + tid.x; R2:R3 = c0[0x10] + R0 * 4; [R2:R3] = R0.
        let program = code(&[
            s2r(0, 0x25),
            s2r(1, 0x21),
            0x5C18_0000_0000_0000 | (2 << 39) | (1 << 20) | PT,
            0x3848_0000_0000_0000 | (2 << 20) | PT | 2,
            0x4C10_0000_0000_0000 | (1 << 47) | (4 << 20) | PT | (2 << 8) | 2,
            0x4C10_0000_0000_0000 | (1 << 43) | (5 << 20) | PT | (0xFF << 8) | 3,
            0xEEDC_2000_0000_0000 | PT | (2 << 8),
            0xE300_0000_0007_000F,
        ]);
        let mut memory = Bytes(vec![0; 0x1000]);
        memory.write_gpu(0x110, &0x400u64.to_le_bytes());
        let pool = PoolState {
            address: 0,
            limit: 0,
        };
        let mut constant_buffers = [None; 8];
        constant_buffers[0] = Some((0x100, 0x100));
        let dispatch = ComputeDispatch {
            program_address: 0,
            grid: [2, 1, 1],
            block: [4, 1, 1],
            shared_memory_size: 0,
            constant_buffers,
            texture_pool: pool,
            sampler_pool: pool,
            texture_buffer_index: 0,
        };
        run(&mut memory, &dispatch, &program).expect("runs");
        for index in 0..8u64 {
            assert_eq!(memory.read_u32(0x400 + index * 4), Some(index as u32));
        }
    }

    #[test]
    fn programs_with_other_instructions_are_refused_before_running() {
        // FADD, then a store that must not happen.
        let program = code(&[
            0x5C58_0000_0000_0000 | PT,
            0xEEDC_2000_0000_0000 | PT,
            0xE300_0000_0007_000F,
        ]);
        let mut memory = Bytes(vec![0; 0x100]);
        let pool = PoolState {
            address: 0,
            limit: 0,
        };
        let dispatch = ComputeDispatch {
            program_address: 0,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_memory_size: 0,
            constant_buffers: [None; 8],
            texture_pool: pool,
            sampler_pool: pool,
            texture_buffer_index: 0,
        };
        assert!(run(&mut memory, &dispatch, &program).is_err());
        assert!(memory.0.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn extended_comparisons_finish_64_bit_ones() {
        // Equal high words: the low words decide through the flags their
        // subtraction left (carry = no borrow).
        let low_greater = Some((true, false));
        let low_equal = Some((true, true));
        let low_less = Some((false, false));
        assert!(compare(4, 5, 5, false, low_greater));
        assert!(!compare(4, 5, 5, false, low_equal));
        assert!(compare(6, 5, 5, false, low_equal));
        assert!(compare(1, 5, 5, false, low_less));
        assert!(compare(2, 5, 5, false, low_equal));
        assert!(!compare(2, 5, 5, false, low_greater));
        // Different high words decide alone.
        assert!(compare(4, 6, 5, false, low_less));
    }
}
