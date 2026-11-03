use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Module {
    pub arch: String,
    #[serde(default)]
    pub segments: Vec<ModuleSegment>,
    /// Guest address of the configured entry function, when known.
    #[serde(default)]
    pub entry_address: Option<u64>,
    pub functions: Vec<Function>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModuleSegment {
    pub name: String,
    pub base: u64,
    pub size: u64,
    pub permissions: ModuleSegmentPermissions,
    #[serde(default)]
    pub init_path: Option<String>,
    #[serde(default)]
    pub init_size: Option<u64>,
    #[serde(default)]
    pub zero_fill: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModuleSegmentPermissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    #[serde(default)]
    pub ops: Vec<Op>,
    #[serde(default)]
    pub blocks: Vec<Block>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Block {
    pub label: String,
    pub start: u64,
    #[serde(default)]
    pub ops: Vec<Op>,
    pub terminator: Terminator,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    ConstI64 {
        dst: String,
        imm: i64,
    },
    AddI64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    MovI64 {
        dst: String,
        src: String,
    },
    SubI64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    AndI64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    OrI64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    XorI64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    CmpI64 {
        lhs: String,
        rhs: String,
    },
    CmnI64 {
        lhs: String,
        rhs: String,
    },
    TestI64 {
        lhs: String,
        rhs: String,
    },
    LslI64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    LsrI64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    AsrI64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    MulI64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    /// High 64 bits of the 128-bit unsigned product (UMULH).
    MulHiU64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    /// High 64 bits of the 128-bit signed product (SMULH).
    MulHiS64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    /// Unsigned division; division by zero yields zero (UDIV).
    UdivI64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    /// Signed division truncating toward zero; division by zero yields zero
    /// and MIN / -1 wraps (SDIV).
    SdivI64 {
        dst: String,
        lhs: String,
        rhs: String,
    },
    /// `dst = cond ? lhs : rhs` on the current NZCV flags (CSEL family).
    SelectI64 {
        dst: String,
        cond: String,
        lhs: String,
        rhs: String,
    },
    ClzI64 {
        dst: String,
        src: String,
    },
    RbitI64 {
        dst: String,
        src: String,
    },
    /// Reverse the byte order of all 64 bits.
    RevI64 {
        dst: String,
        src: String,
    },
    /// 128-bit constant into a vector register. Vector registers are named
    /// `v0`-`v31`; vector temps start with `vtmp`.
    VConst {
        dst: String,
        lo: i64,
        hi: i64,
    },
    /// Load `bytes` (1, 2, 4, 8 or 16) into a vector register, zeroing the
    /// rest of it.
    VLoad {
        dst: String,
        addr: String,
        #[serde(default)]
        offset: i64,
        bytes: u8,
    },
    /// Store the low `bytes` of a vector register.
    VStore {
        src: String,
        addr: String,
        #[serde(default)]
        offset: i64,
        bytes: u8,
    },
    /// Copy the low `bytes` of `src` and zero the rest (16 copies it all).
    VMov {
        dst: String,
        src: String,
        bytes: u8,
    },
    /// Replicate the low `esize` bytes of a general register across the low
    /// 8 bytes (`q` false, upper half zeroed) or all 16 bytes of `dst`.
    VDup {
        dst: String,
        src: String,
        esize: u8,
        q: bool,
    },
    /// Write a general register into lane `index` (of `esize` bytes) of
    /// `dst`, keeping the other lanes.
    VInsGpr {
        dst: String,
        src: String,
        esize: u8,
        index: u8,
    },
    /// Read lane `index` (of `esize` bytes) of `src` into a general register,
    /// zero- or sign-extended to 64 bits.
    VExtract {
        dst: String,
        src: String,
        esize: u8,
        index: u8,
        signed: bool,
    },
    /// Copy lane `src_index` of `src` into lane `dst_index` of `dst`.
    VInsLane {
        dst: String,
        src: String,
        esize: u8,
        dst_index: u8,
        src_index: u8,
    },
    /// Bitwise `and`, `orr`, `eor`, `bic` or `orn` of two vectors; `q` false
    /// zeroes the upper 8 bytes of the result.
    VBitwise {
        dst: String,
        lhs: String,
        rhs: String,
        kind: String,
        q: bool,
    },
    /// `dst = (mask & set) | (!mask & clear)` (BSL/BIT/BIF); `q` as above.
    VBitSelect {
        dst: String,
        mask: String,
        set: String,
        clear: String,
        q: bool,
    },
    PcRel {
        dst: String,
        pc: i64,
        offset: i64,
    },
    LoadI8 {
        dst: String,
        addr: String,
        #[serde(default)]
        offset: i64,
    },
    LoadI16 {
        dst: String,
        addr: String,
        #[serde(default)]
        offset: i64,
    },
    LoadI32 {
        dst: String,
        addr: String,
        #[serde(default)]
        offset: i64,
    },
    LoadI64 {
        dst: String,
        addr: String,
        #[serde(default)]
        offset: i64,
    },
    StoreI8 {
        src: String,
        addr: String,
        #[serde(default)]
        offset: i64,
    },
    StoreI16 {
        src: String,
        addr: String,
        #[serde(default)]
        offset: i64,
    },
    StoreI32 {
        src: String,
        addr: String,
        #[serde(default)]
        offset: i64,
    },
    StoreI64 {
        src: String,
        addr: String,
        #[serde(default)]
        offset: i64,
    },
    /// LDXR, LDAXR and the pair forms: read the value(s) and reserve the
    /// address for the store-exclusive that follows.
    ExclusiveLoad {
        dst: String,
        /// The second register of a pair load.
        #[serde(default)]
        dst2: Option<String>,
        addr: String,
        /// Access width in bytes.
        size: u64,
        #[serde(default)]
        pair: bool,
        #[serde(default)]
        offset: i64,
    },
    /// STXR, STLXR and the pair forms. `status` receives the success flag:
    /// 0 when the store happened, 1 when the reservation was lost.
    ExclusiveStore {
        src: String,
        /// The second value of a pair store.
        #[serde(default)]
        src2: Option<String>,
        status: String,
        addr: String,
        /// Access width in bytes.
        size: u64,
        #[serde(default)]
        pair: bool,
        #[serde(default)]
        offset: i64,
    },
    /// CLREX: drop this thread's reservation.
    ClearExclusive,
    /// DMB: guest threads run on host threads, so the order guest code
    /// expects between its accesses has to be real.
    MemoryFence,
    /// After LDAR: later accesses stay after the load (acquire).
    AcquireFence,
    /// Before STLR: earlier accesses stay before the store (release).
    ReleaseFence,
    Br {
        target: String,
    },
    BrCond {
        cond: String,
        then: String,
        #[serde(rename = "else")]
        else_target: String,
    },
    Call {
        target: String,
    },
    Syscall {
        name: String,
        args: Vec<String>,
    },
    /// Read an AArch64 system register (TPIDR_EL0, CNTPCT_EL0, ...).
    ReadSysReg {
        dst: String,
        reg: String,
    },
    /// Write an AArch64 system register.
    WriteSysReg {
        reg: String,
        src: String,
    },
    /// MSR NZCV, Xt: restore condition flags from bits 31..28.
    ReadNzcv {
        dst: String,
    },
    WriteNzcv {
        src: String,
    },
    /// ADC/ADCS/SBC/SBCS: `lhs + rhs + C`, with `rhs` inverted when
    /// `subtract`, over 32 or 64 bits; `set_flags` also updates NZCV.
    AddCarry {
        dst: String,
        lhs: String,
        rhs: String,
        subtract: bool,
        set_flags: bool,
        is_32: bool,
    },
    /// A floating-point or SIMD instruction evaluated at run time by
    /// `recomp_abi::simd::exec`: the vector registers it reads, the general
    /// register it reads (absent for none or the zero register) and its
    /// destination (`vN` or `xN`; absent when it only sets flags).
    Simd {
        word: u32,
        vector_reads: Vec<String>,
        #[serde(default)]
        general_read: Option<String>,
        #[serde(default)]
        dst: Option<String>,
        #[serde(default)]
        reads_flags: bool,
        #[serde(default)]
        writes_flags: bool,
    },
    Ret,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Terminator {
    Br {
        target: String,
    },
    BrCond {
        cond: String,
        then: String,
        #[serde(rename = "else")]
        else_target: String,
    },
    /// Branch on bits of a register (CBZ/CBNZ, TBZ/TBNZ): to `then` when
    /// `reg & mask` is zero (`zero`) or non-zero, else to `else`. Unlike
    /// `br_cond` it neither reads nor writes the flags.
    BrTest {
        reg: String,
        mask: u64,
        zero: bool,
        then: String,
        #[serde(rename = "else")]
        else_target: String,
    },
    Call {
        target: String,
        next: String,
        /// Guest address the call returns to (BL/BLR set X30 to it). When
        /// set, a callee that returns with a different X30 makes this
        /// function return too, as `ret` to X30 would on hardware.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        return_address: Option<u64>,
    },
    BrIndirect {
        reg: String,
    },
    /// An indirect branch through a recovered jump table: jumps to the case
    /// whose guest address equals `reg`.
    Switch {
        reg: String,
        cases: Vec<(u64, String)>,
    },
    CallIndirect {
        reg: String,
        next: String,
        /// As for `Call`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        return_address: Option<u64>,
    },
    /// An instruction the lifter cannot translate; raises a runtime error
    /// instead of silently returning with partial guest state.
    Trap {
        pc: u64,
        word: u32,
    },
    Ret,
}

impl Module {
    pub fn validate_arch(&self) -> Result<(), String> {
        if self.arch != "aarch64" {
            return Err(format!("unsupported arch: {}", self.arch));
        }
        Ok(())
    }
}
