//! The Maxwell macro execution engine (MME).
//!
//! The GPU keeps small programs in a macro instruction RAM and a table of
//! start addresses. Programs are loaded with methods `LOAD_MME_INSTRUCTION_RAM_POINTER`
//! / `LOAD_MME_INSTRUCTION_RAM` and `LOAD_MME_START_ADDRESS_RAM_POINTER` /
//! `LOAD_MME_START_ADDRESS_RAM`. Methods from `0xE00` up invoke them: the
//! macro index is `(method >> 1) & 0x7F`, an odd method pushes the word as an
//! argument, an even method starts the macro with the word as its first
//! argument (`R1`).
//!
//! A macro is a small RISC program with eight registers, an immediate, a
//! carry flag, and a method-address register. Instructions are pipelined: an
//! instruction fetched one step earlier is what executes, so the instruction
//! after an exit is still run (the delay slot). Branches take a delay slot
//! too, unless the instruction says otherwise.
//!
//! This is the general path: every macro runs here. A backend can recognize
//! the common macros (draws, clears, state updates) and do them directly, but
//! the interpreter is what makes an unrecognized macro work.

use std::collections::VecDeque;

/// Number of macros the hardware tracks.
pub const MACRO_COUNT: usize = 0x80;
/// Macro code RAM, in words. The hardware size is not documented; this is the
/// size the emulator implementations use, and it is far past what games load.
pub const MACRO_CODE_WORDS: usize = 256 * 256;
/// Step budget for one macro call. Real macros are tens of instructions; the
/// budget only stops a program whose code was never fully loaded.
pub const MAX_STEPS: usize = 4096;

/// A macro argument, with the GPU address the word was written at. Indirect
/// draws use the address to find the buffer the parameters live in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FifoWord {
    pub gpu_va: u64,
    pub word: u32,
}

/// Register access for macro code. Offsets are byte offsets in the state
/// block of the class the macro runs on.
pub trait MacroState {
    fn read_reg(&self, offset: usize) -> u32;
    fn write_reg(&mut self, offset: usize, value: u32);
}

/// ALU operation, in bits 2:0 of the instruction.
pub const ALU_REG: u32 = 0;
pub const ALU_ADD_IMMEDIATE: u32 = 1;
pub const ALU_BITFIELD_REPLACE: u32 = 2;
pub const ALU_BITFIELD_EXTRACT_LSL_IMM: u32 = 3;
pub const ALU_BITFIELD_EXTRACT_LSL_REG: u32 = 4;
pub const ALU_READ_IMMEDIATE: u32 = 5;

/// Assignment operation, in bits 6:4 of the instruction.
pub const ASG_IGNORE_AND_FETCH: u32 = 0;
pub const ASG_MOVE: u32 = 1;
pub const ASG_MOVE_AND_SET_MADDR: u32 = 2;
pub const ASG_FETCH_AND_SEND: u32 = 3;
pub const ASG_MOVE_AND_SEND: u32 = 4;
pub const ASG_FETCH_AND_SET_MADDR: u32 = 5;
pub const ASG_MOVE_AND_SET_MADDR_THEN_FETCH_AND_SEND: u32 = 6;
pub const ASG_MOVE_AND_SET_MADDR_THEN_SEND_HIGH: u32 = 7;

/// The interpreter's state for one macro program. The argument FIFO survives
/// between calls; the registers and pipeline do not.
#[derive(Debug, Default)]
pub struct MacroInterpreter {
    pub fifo: VecDeque<FifoWord>,
    gprs: [u32; 8],
    meth_addr: u32,
    meth_incr: u32,
    carry: bool,
    opcode: u32,
    pipe_op: u32,
    ignore_exit_flag: bool,
    pc: usize,
    steps: usize,
}

impl MacroInterpreter {
    /// Runs the program at the start of `code` until it exits, with `arg0` as
    /// its first argument. Returns the number of instructions executed.
    pub fn execute(&mut self, code: &[u32], state: &mut dyn MacroState, arg0: u32) -> usize {
        self.reset();
        self.gprs[1] = arg0;
        self.pc = 0;
        self.fetch(code);

        while self.step(code, state) {
            if self.steps >= MAX_STEPS {
                break;
            }
        }

        // The delay slot: the instruction after the exit was already fetched
        // and still runs.
        self.step(code, state);
        self.steps
    }

    fn reset(&mut self) {
        self.gprs = [0; 8];
        self.meth_addr = 0;
        self.meth_incr = 0;
        self.carry = false;
        self.opcode = 0;
        self.pipe_op = 0;
        self.ignore_exit_flag = false;
        self.pc = 0;
        self.steps = 0;
    }

    /// Runs one instruction (the one fetched in the previous step). True
    /// continues, false exits.
    fn step(&mut self, code: &[u32], state: &mut dyn MacroState) -> bool {
        self.steps += 1;
        let base_addr = self.pc.wrapping_sub(1);
        self.fetch(code);

        if self.opcode & 7 < 7 {
            // The operation produces a value.
            let asg_op = (self.opcode >> 4) & 7;
            let result = self.alu_result(state);

            match asg_op {
                ASG_IGNORE_AND_FETCH => {
                    let parameter = self.fetch_param();
                    self.set_dst(parameter);
                }
                ASG_MOVE => self.set_dst(result),
                ASG_MOVE_AND_SET_MADDR => {
                    self.set_dst(result);
                    self.set_meth_addr(result);
                }
                ASG_FETCH_AND_SEND => {
                    let parameter = self.fetch_param();
                    self.set_dst(parameter);
                    self.send(state, result);
                }
                ASG_MOVE_AND_SEND => {
                    self.set_dst(result);
                    self.send(state, result);
                }
                ASG_FETCH_AND_SET_MADDR => {
                    let parameter = self.fetch_param();
                    self.set_dst(parameter);
                    self.set_meth_addr(result);
                }
                ASG_MOVE_AND_SET_MADDR_THEN_FETCH_AND_SEND => {
                    let parameter = self.fetch_param();
                    self.set_dst(result);
                    self.set_meth_addr(result);
                    self.send(state, parameter);
                }
                ASG_MOVE_AND_SET_MADDR_THEN_SEND_HIGH => {
                    self.set_dst(result);
                    self.set_meth_addr(result);
                    self.send(state, (result >> 12) & 0x3f);
                }
                _ => {}
            }
        } else {
            // Branch on the A register.
            let on_not_zero = (self.opcode >> 4) & 1 != 0;
            let taken = if on_not_zero {
                self.gpr_a() != 0
            } else {
                self.gpr_a() == 0
            };

            if taken {
                self.pc = base_addr.wrapping_add(self.imm() as usize);

                if self.opcode & 0x20 != 0 {
                    self.fetch(code);
                } else {
                    // The delay slot instruction's exit flag is ignored.
                    self.ignore_exit_flag = true;
                }

                return true;
            }
        }

        let exit = self.opcode & 0x80 != 0 && !self.ignore_exit_flag;
        self.ignore_exit_flag = false;
        !exit
    }

    /// Shifts the pipeline: the fetched instruction becomes the one running.
    fn fetch(&mut self, code: &[u32]) {
        self.opcode = self.pipe_op;
        self.pipe_op = code.get(self.pc).copied().unwrap_or(0);
        self.pc += 1;
    }

    fn alu_result(&mut self, state: &mut dyn MacroState) -> u32 {
        match self.opcode & 7 {
            ALU_REG => self.alu_reg((self.opcode >> 17) & 0x1f, self.gpr_a(), self.gpr_b()),
            ALU_ADD_IMMEDIATE => self.gpr_a().wrapping_add(self.imm()),
            ALU_BITFIELD_REPLACE | ALU_BITFIELD_EXTRACT_LSL_IMM | ALU_BITFIELD_EXTRACT_LSL_REG => {
                let src_bit = (self.opcode >> 17) & 0x1f;
                let size = (self.opcode >> 22) & 0x1f;
                let dst_bit = (self.opcode >> 27) & 0x1f;
                let mask = (1u32 << size).wrapping_sub(1);

                let mut dst = self.gpr_a();
                let mut src = self.gpr_b();

                match self.opcode & 7 {
                    ALU_BITFIELD_REPLACE => {
                        src = (src >> src_bit) & mask;
                        dst &= !(mask << dst_bit);
                        dst | (src << dst_bit)
                    }
                    ALU_BITFIELD_EXTRACT_LSL_IMM => {
                        src = (src >> dst) & mask;
                        src << dst_bit
                    }
                    _ => {
                        src = (src >> src_bit) & mask;
                        src.wrapping_shl(dst)
                    }
                }
            }
            ALU_READ_IMMEDIATE => {
                // The register plus the immediate is a method number.
                let method = self.gpr_a().wrapping_add(self.imm());
                state.read_reg(method.wrapping_mul(4) as usize)
            }
            _ => 0,
        }
    }

    fn alu_reg(&mut self, op: u32, a: u32, b: u32) -> u32 {
        // Operands are signed 32-bit; the hardware widens them to 64 bits and
        // the carry flag is set from the widened result.
        let a = a as i32 as i64 as u64;
        let b = b as i32 as i64 as u64;

        match op {
            0 => {
                let result = a.wrapping_add(b);
                self.carry = result > u64::from(u32::MAX);
                result as u32
            }
            1 => {
                let result = a.wrapping_add(b).wrapping_add(u64::from(self.carry));
                self.carry = result > u64::from(u32::MAX);
                result as u32
            }
            2 => {
                let result = a.wrapping_sub(b);
                self.carry = result < 0x1_0000_0000;
                result as u32
            }
            3 => {
                let result = a.wrapping_sub(b).wrapping_sub(u64::from(!self.carry));
                self.carry = result < 0x1_0000_0000;
                result as u32
            }
            8 => (a as u32) ^ (b as u32),
            9 => (a as u32) | (b as u32),
            10 => (a as u32) & (b as u32),
            11 => (a as u32) & !(b as u32),
            12 => !((a as u32) & (b as u32)),
            _ => 0,
        }
    }

    /// The immediate is the top 18 bits, sign-extended.
    fn imm(&self) -> u32 {
        (self.opcode as i32 >> 14) as u32
    }

    fn set_dst(&mut self, value: u32) {
        self.gprs[((self.opcode >> 8) & 7) as usize] = value;
    }

    fn gpr_a(&self) -> u32 {
        self.gpr(((self.opcode >> 11) & 7) as usize)
    }

    fn gpr_b(&self) -> u32 {
        self.gpr(((self.opcode >> 14) & 7) as usize)
    }

    /// Register zero reads as zero whatever was written to it.
    fn gpr(&self, index: usize) -> u32 {
        if index == 0 {
            0
        } else {
            self.gprs[index]
        }
    }

    fn set_meth_addr(&mut self, value: u32) {
        self.meth_addr = value & 0xfff;
        self.meth_incr = (value >> 12) & 0x3f;
    }

    fn fetch_param(&mut self) -> u32 {
        self.fifo.pop_front().map(|word| word.word).unwrap_or(0)
    }

    fn send(&mut self, state: &mut dyn MacroState, value: u32) {
        state.write_reg(self.meth_addr.wrapping_mul(4) as usize, value);
        self.meth_addr = self.meth_addr.wrapping_add(self.meth_incr);
    }
}

/// An ALU instruction between registers. The instruction encoders are public
/// so tests and tools can assemble macro programs.
pub fn alu_reg(alu: u32, asg: u32, dst: usize, a: usize, b: usize, exit: bool) -> u32 {
    ALU_REG
        | (asg << 4)
        | ((dst as u32) << 8)
        | ((a as u32) << 11)
        | ((b as u32) << 14)
        | (alu << 17)
        | (if exit { 0x80 } else { 0 })
}

/// `Rdst = Ra + immediate`, optionally with the assignment operations.
pub fn add_imm(asg: u32, dst: usize, a: usize, imm: i32, exit: bool) -> u32 {
    ALU_ADD_IMMEDIATE
        | (asg << 4)
        | ((dst as u32) << 8)
        | ((a as u32) << 11)
        | ((imm as u32) << 14)
        | (if exit { 0x80 } else { 0 })
}

/// `Rdst = method[Ra + immediate]`.
pub fn read_imm(asg: u32, dst: usize, a: usize, imm: i32, exit: bool) -> u32 {
    ALU_READ_IMMEDIATE
        | (asg << 4)
        | ((dst as u32) << 8)
        | ((a as u32) << 11)
        | ((imm as u32) << 14)
        | (if exit { 0x80 } else { 0 })
}

/// Branch to `pc + offset` when `Ra` is zero (or non-zero), with the
/// delay slot suppressed or not.
pub fn branch(a: usize, on_not_zero: bool, offset: i32, no_delays: bool, exit: bool) -> u32 {
    7 | (if on_not_zero { 1 << 4 } else { 0 })
        | (if no_delays { 1 << 5 } else { 0 })
        | ((a as u32) << 11)
        | ((offset as u32) << 14)
        | (if exit { 0x80 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state block that records writes.
    struct TestState {
        words: Vec<u32>,
        writes: Vec<(usize, u32)>,
    }

    impl TestState {
        fn new() -> Self {
            Self {
                words: vec![0; 0x1000],
                writes: Vec::new(),
            }
        }
    }

    impl MacroState for TestState {
        fn read_reg(&self, offset: usize) -> u32 {
            self.words.get(offset / 4).copied().unwrap_or(0)
        }

        fn write_reg(&mut self, offset: usize, value: u32) {
            self.writes.push((offset, value));
            if let Some(word) = self.words.get_mut(offset / 4) {
                *word = value;
            }
        }
    }

    #[test]
    fn sends_the_method_address_and_increment_together() {
        // The method address register packs the address in bits 11:0 and the
        // increment in bits 17:12, so one instruction sets both.
        let code = [
            add_imm(ASG_MOVE_AND_SET_MADDR, 0, 0, 0x674 | (1 << 12), false),
            add_imm(ASG_MOVE_AND_SEND, 0, 1, 0, true),
        ];

        let mut interpreter = MacroInterpreter::default();
        let mut state = TestState::new();
        interpreter.execute(&code, &mut state, 7);

        // The exit instruction sends once; its delay slot is the instruction
        // past the code and produces nothing. The method number is a word
        // offset, so `CLEAR` (byte 0x19d0) is word 0x674.
        assert_eq!(state.writes, vec![(0x674 * 4, 7)]);
    }

    #[test]
    fn sends_pushed_arguments_in_order() {
        let code = [
            add_imm(
                ASG_MOVE_AND_SET_MADDR_THEN_FETCH_AND_SEND,
                0,
                0,
                0x674,
                false,
            ),
            add_imm(
                ASG_MOVE_AND_SET_MADDR_THEN_FETCH_AND_SEND,
                0,
                0,
                0x675,
                false,
            ),
            add_imm(ASG_MOVE, 0, 0, 0, true),
        ];

        let mut interpreter = MacroInterpreter::default();
        interpreter.fifo.push_back(FifoWord {
            gpu_va: 0x1000,
            word: 1,
        });
        interpreter.fifo.push_back(FifoWord {
            gpu_va: 0x1004,
            word: 2,
        });

        let mut state = TestState::new();
        interpreter.execute(&code, &mut state, 0);

        assert_eq!(state.writes, vec![(0x674 * 4, 1), (0x675 * 4, 2)]);
        assert!(interpreter.fifo.is_empty());
    }

    #[test]
    fn reads_a_method_relative_to_a_register() {
        let mut state = TestState::new();
        state.words[0x674] = 0xdead;

        let code = [
            add_imm(ASG_MOVE_AND_SET_MADDR, 0, 0, 0x100, false),
            read_imm(ASG_MOVE, 2, 0, 0x674, false),
            add_imm(ASG_MOVE_AND_SEND, 0, 2, 0, true),
        ];

        let mut interpreter = MacroInterpreter::default();
        interpreter.execute(&code, &mut state, 0);

        assert_eq!(state.writes, vec![(0x100 * 4, 0xdead)]);
        assert_eq!(interpreter.gprs[2], 0xdead);
    }

    #[test]
    fn branch_runs_the_delay_slot_and_then_the_target() {
        let code = [
            add_imm(ASG_MOVE_AND_SET_MADDR, 0, 0, 0x60, false),
            branch(1, true, 2, false, true),
            add_imm(ASG_MOVE_AND_SEND, 2, 0, 1, false),
            add_imm(ASG_MOVE, 3, 0, 2, true),
        ];

        let mut interpreter = MacroInterpreter::default();
        let mut state = TestState::new();
        interpreter.execute(&code, &mut state, 1);

        // The branch is taken, so its exit flag is ignored; the delay slot
        // sends, and the target instruction moves and exits.
        assert_eq!(state.writes, vec![(0x60 * 4, 1)]);
        assert_eq!(interpreter.gprs[2], 1);
        assert_eq!(interpreter.gprs[3], 2);
    }

    #[test]
    fn branch_not_taken_falls_through() {
        let code = [
            add_imm(ASG_MOVE_AND_SET_MADDR, 0, 0, 0x60, false),
            branch(1, true, 2, false, false),
            add_imm(ASG_MOVE_AND_SEND, 0, 1, 1, true),
            add_imm(ASG_MOVE_AND_SEND, 0, 1, 2, false),
        ];

        let mut interpreter = MacroInterpreter::default();
        let mut state = TestState::new();
        interpreter.execute(&code, &mut state, 0);

        // R1 is zero, so the branch is not taken and the fall-through runs.
        // The exit instruction sends, then its delay slot sends again.
        assert_eq!(state.writes, vec![(0x60 * 4, 1), (0x60 * 4, 2)]);
    }

    #[test]
    fn register_zero_reads_as_zero() {
        let code = [
            alu_reg(0, ASG_MOVE, 2, 0, 0, false),
            add_imm(ASG_MOVE, 3, 2, 5, true),
        ];

        let mut interpreter = MacroInterpreter::default();
        let mut state = TestState::new();
        interpreter.execute(&code, &mut state, 0);

        assert_eq!(interpreter.gprs[2], 0);
        assert_eq!(interpreter.gprs[3], 5);
    }

    #[test]
    fn bitfield_replace_edits_only_the_field() {
        // Replace bits 11:8 of R2 with bits 3:0 of R3. The source bit field
        // is zero, so it is left out of the instruction.
        let instruction = ALU_BITFIELD_REPLACE
            | (ASG_MOVE << 4)
            | (2 << 8)
            | (2 << 11)
            | (3 << 14)
            | (4 << 22)
            | (8 << 27)
            | 0x80;

        let mut state = TestState::new();
        state.words[0x100] = 0xdead_beef;
        state.words[0x101] = 0xf00d;

        let code = [
            read_imm(ASG_MOVE, 2, 0, 0x100, false),
            read_imm(ASG_MOVE, 3, 0, 0x101, false),
            instruction,
        ];

        let mut interpreter = MacroInterpreter::default();
        interpreter.execute(&code, &mut state, 0);

        // The field's old value (0xe) is replaced by 0xd, the rest is kept.
        assert_eq!(interpreter.gprs[2], 0xdead_bdef);
    }

    #[test]
    fn carry_comes_from_the_widened_result() {
        let mut interpreter = MacroInterpreter::default();
        let mut state = TestState::new();
        state.words[0x100] = 0xffff_ffff;

        // Operands are signed and widened, so -1 plus 0x10000 stays inside 32
        // bits and leaves the carry clear: add-with-carry adds nothing.
        let code = [
            read_imm(ASG_MOVE, 2, 0, 0x100, false),
            alu_reg(0, ASG_MOVE, 3, 2, 1, false),
            alu_reg(1, ASG_MOVE, 4, 3, 0, true),
        ];

        interpreter.execute(&code, &mut state, 0x1_0000);
        assert_eq!(interpreter.gprs[3], 0x0000_ffff);
        assert_eq!(interpreter.gprs[4], 0x0000_ffff);

        // Subtract sets the carry when there is no borrow, and clears it when
        // the result goes below zero; subtract-with-borrow then reads it back.
        state.words[0x100] = 5;
        let code = [
            read_imm(ASG_MOVE, 2, 0, 0x100, false),
            alu_reg(2, ASG_MOVE, 3, 1, 2, false),
            alu_reg(3, ASG_MOVE, 4, 1, 2, true),
        ];

        interpreter.execute(&code, &mut state, 0);
        assert_eq!(interpreter.gprs[3], (-5_i32) as u32);
        // Borrowed, so the second subtraction takes off one more.
        assert_eq!(interpreter.gprs[4], (-6_i32) as u32);

        state.words[0x100] = 0;
        interpreter.execute(&code, &mut state, 5);
        assert_eq!(interpreter.gprs[3], 5);
        // No borrow, so the second subtraction is just 5 again.
        assert_eq!(interpreter.gprs[4], 5);
    }
}
