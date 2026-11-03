# Exploratory Pipeline Notes

This document captures the initial exploratory pipeline that mirrors proven static recomp workflows while staying aligned with the specs.

## Scope
- Parse a small JSON module format as a stand-in for lifted code.
- Apply a TOML config to select syscall stub behaviors.
- Emit a Rust project that links against a runtime ABI crate.

## Why This Shape
- Mirrors the deterministic, file-driven flow seen in static recomp projects.
- Keeps a strict boundary between inputs and outputs.
- Produces compilable artifacts for validation and iteration.

## Inputs
- `module.json` describes a lifted module, functions, and operations.
- `title.toml` provides the title name, entry function, ABI version, and stub map.
- `provenance.toml` records lawful input provenance and format metadata.
- Homebrew intake emits a separate `module.json` + `manifest.json` with segment blobs and assets extracted from NRO/NSO inputs; this homebrew module.json is not consumed by the translator until a lifter produces a lifted module.json.
- The `homebrew-lift` command decodes the A64 integer subset (branches, add/sub, add/sub with carry, logical, move-wide, bitfield/EXTR, conditional select/compare, multiply/divide, CLZ/RBIT/REV, single-register, pair and literal loads/stores, exclusives and load-acquire/store-release lowered for a single host thread, SVC, hints, barriers and cache maintenance) plus SIMD&FP data movement (B/H/S/D/Q loads and stores and pairs, LD1-LD4/ST1-ST4 in their multiple, single-lane and replicate forms, DUP, UMOV/SMOV/INS, MOVI/MVNI/ORR/BIC immediate, FMOV, vector AND/ORR/EOR/BIC/ORN/BSL/BIT/BIF). Calls set X30 to the return address.
- All other floating-point and Advanced SIMD data processing (scalar FP, AdvSIMD vector and scalar, AES, SHA-1/SHA-256) becomes a `simd` op: the generated code passes the registers it uses to `recomp_abi::simd::exec`, which evaluates the encoding with the `recomp-simd` crate (Arm pseudocode semantics with FPCR.DN/FZ = 0 and round-to-nearest-even). The lifter asks the same crate which registers and flags an instruction touches, so the operands passed always match what the evaluator reads. Emitted projects build `recomp-simd` with `opt-level = 3` even in dev builds. `exec` decodes the word on every call, which made floating-point code tens of times slower than integer code (LM3 spent 38 s of its boot in it), so the code generator first asks `recomp_simd::fast::specialize` for a direct call: the common scalar FP operations, multiply-add, compares and selects, conversions, FMOV forms, the vector three-same FP operations, FP by element, EXT and scalar DUP become calls like `recomp_abi::simd::fast::scalar_fp2(2u32, false, v1, v0)` with the decoded fields as constants (91.6% of LM3's 555,005 FP/SIMD instructions). Generated code reaches these
through `__recomp_simd_*` symbols the runtime defines (`recomp-abi` does not
depend on `recomp-simd`), so changing the evaluator relinks a program instead
of rebuilding its chunks. The hottest of them (scalar FADD/FSUB/FMUL/FDIV/
FNMUL, the scalar fused multiply-adds, the single-precision three-same vector
operations and FP by element) have inline versions in `recomp_abi::simd::fast`
that fold, with the constant fields, into a few host instructions and leave
NaN results to the evaluator (`exact`); direct-register code calls
`vector3_into`/`vector_element_into`, which store the result register in place
with one 16-byte access, and the four-lane NaN guard is one compare plus one
`vptest` over the lanes the operation writes. A test runs millions of random and
special inputs through both and requires identical bits. Through the symbols,
LM3's busiest job threads spent a third of their time in calls. The rest of
the specialized calls (FCMP, FCSEL, FMOV/FABS/FNEG/FSQRT, the integer
conversions, EXT, DUP, vector FMAX/FMIN and their NM and pairwise forms,
FRECPS, FRSQRTS and the vector compares) are inline too, and ZIP, UZP and TRN
are specialized (`permute`): `RECOMP_SIMD_STATS=1` counted 18 million
evaluator calls a second in LM3's gameplay for them, a quarter of them ZIP
through `exec`. Built for x86-64 with AVX and FMA (`target_cpu = "native"`
on any recent x86 CPU), the single-precision vector operations, FP by
element, EXT, scalar DUP, ZIP, UZP and TRN run on host 128-bit vectors
(`recomp_abi::simd::fast::host`): a register's lanes go through one
instruction together and stay in a vector register between operations.
Lane by lane, LM3's job code moved every lane between general and vector
registers and back, about 30 instructions per FMLA by element; its hottest
job function went from 408 scalar fused multiply-adds and 1498 moves
between register files to 102 packed ones and 316 moves. The byte moves
use `pshufb` masks from `const` tables (a `static`'s contents are not
visible across crates, which left a load per use), the evaluator's results
come back as vectors through cold wrappers (an integer result to merge with
sent the fast path's vector through general registers), and 16-byte loads
and stores are one access of a host vector type (`Memory::load_u128`). The
same equivalence test run with `RUSTFLAGS="-C target-cpu=native"` checks
these paths. A title's
`[runtime] target_cpu = "native"` writes the project's `.cargo/config.toml`
so the program is compiled for the host CPU: with FMA each fused
multiply-add is one instruction instead of a libm call (Rust never fuses a
separate multiply and add, so results stay exact). For a cross-compiled
title, `[runtime] target = "aarch64-linux-android"` selects the Cargo
target and `native` resolves to that target's `generic` CPU instead of the
build host's; see `docs/arm64-portability.md`. For single-precision lanes the specialized vector,
by-element and scalar arithmetic computes every lane with host operations
(hardware FMA when the CPU has it) and falls back to the exact evaluator only
when a lane comes out NaN: for any other result the host computes exactly what
the Arm pseudocode does, and the exact path differs only in NaN rules. LM3's
title-screen setup spent 90% of its busy time in the per-lane dispatch before. A test runs millions of random encodings through both paths, values weighted to zeros, infinities, NaNs and denormals, and requires identical results. CRC32, DC ZVA and MTE/PAC encodings are not lifted yet: a block that reaches one ends in a `trap` terminator that fails at runtime with the guest pc and encoding instead of silently returning. Use `--extra-entry <addr>` to seed functions reached only through computed pointers, and `--mode stub` to emit a placeholder lifted module when decoding is not possible.
- CBZ/CBNZ/TBZ/TBNZ become `br_test` terminators, which branch on bits of a register and leave NZCV untouched: compilers branch on flags set before them (`subs; stlxr; cbnz retry; b.ne`, the reference-count release pattern). A test of XZR is an unconditional branch.
- NZCV crosses calls like the registers do: a lifted function that uses flags starts from the thread's flags (`recomp_abi::cpu_nzcv`), stores its flags before every call, return and tail call (`cpu_set_nzcv`), and reloads them after each call. Compiled code branches on flags a callee set: LLVM's machine outliner moves `add; cmp; str; ret` into a shared helper and the caller does `bl helper; b.hi overflow`. While flags were function-local, LM3's per-frame GPU allocator saw a stale comparison after Story > New Game and took its overflow path, a fatal trap. Functions that never touch flags pass them through untouched.
- Indirect branches through compiler jump tables (`ldr`/`ldrsw` of an entry, `add` to a base, `br`) become `switch` terminators whose cases are the table's targets. The entry count comes from the bounds check guarding the index, followed through register moves and through ANDs of the same source with the same mask (also an earlier AND into the index register itself); the table address may be formed anywhere back to the function's entry (compilers hoist it into a callee-saved register). The generated code matches the computed target at run time and dispatches anything else normally, so an over-approximated case list cannot misdirect control. An index reloaded from a stack slot is followed back to the store that spilled it, and the table address through register copies and past epilogue reloads of callee-saved registers. When no bounds check is recognized (or the one found does not fit the table), entries are read while their targets stay inside the function: between its entry and the next known function start, or between the known starts around the `br`. When the table register reaches the `br` along several paths (a state machine passing it between registers), every table address the function forms in that register is tried, and the switch covers the targets of each one whose entries all land in the function. Table-shaped branches that still cannot be recovered are listed in the lift warnings (`unresolved jump table: ...`), since they fail when reached. SIMD and floating-point data processing counts as writing no general register (except conversions to integers, FMOV to a general register and SMOV/UMOV): taking `mov v8.16b, v3.16b` for a write of `x8` once hid a table's `add x8` from its `br x8`, and LM3 crashed on its main menu.
- `scripts/lift-difftest.py` checks the lowering (integer, memory and SIMD data movement) against a capstone-based reference model on random encodings, register states and memory.
- Generated chunk crates depend only on `crates/recomp-abi`, a small stable interface the runtime implements at link time, so runtime changes relink the program instead of recompiling every chunk. The pipeline rewrites only files whose content changed.
- Homebrew RomFS assets are emitted as a file tree under `assets/romfs/`; runtime implementations should mount or map this directory when wiring up RomFS access.

Example stub map:
```
[stubs]
svc_log = "log"
svc_sleep = "noop"
svc_fatal = "panic"
```

Example runtime config:
```
[runtime]
performance_mode = "handheld"
```

Host implementations of lifted functions (`crates/recomp-runtime/src/host_functions.rs`):
```
[host_functions]
fn_00000000090415a0 = "zlib_inflate_fast"
```
The generated function then calls the runtime's `__recomp_host_<name>` first and runs its lifted
code only when that returns false, so the host version can leave rare cases to the original
(`zlib_inflate_fast` hands back any invalid code or out-of-window distance before writing stream
state, and the guest's zlib sets the error). Only the chunk holding the function is rebuilt. LM3
inflates every file it loads with its statically linked zlib (1.2.11 layout); the lifted
`inflate_fast` was the one busy loop of its loading screen, which now takes 7 s instead of 20.

## Outputs
- A self-contained Rust crate in the output directory.
- The crate depends on `recomp-runtime` via a relative path.
- The emitted `main.rs` invokes the entry function and records the ABI version.
- A `manifest.json` file with input hashes, provenance hash, and generated file list.

## Process Automation Ideas (Crimsonland Write-up)
- Build a deterministic analysis regen loop: drive a headless decompiler to export XML and decompile artifacts; treat exports as generated outputs and keep only inputs plus a rename/type map under version control.
- Maintain a structured `name_map.json` (or equivalent) where each rename/type entry includes address and evidence; reapply it to regenerate names and types consistently.
- Detect bundled third-party libraries via version strings and inject known headers/typedefs before decompilation to improve type recovery.
- Create a long-running runtime analysis session with log tailing so behavioral observations can be captured while keeping the debugger attached.
- Use runtime hooks to capture validation fixtures (framebuffer dumps, deterministic samples) and store them alongside provenance for later regression checks.
- Consider agent-assisted rename and pattern discovery backed by a curated knowledge base; only promote renames with evidence.

## Next Steps
- Add a real input parser for Switch binaries.
- Expand the lifter to cover more AArch64 instructions and control flow.
- Expand runtime services and ABI validation.

## Further Reading
- `docs/static-recompilation-flow.md` for a hypothetical end-to-end flow and verification plan.
