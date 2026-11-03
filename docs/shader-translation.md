# Shader translation (Maxwell to SPIR-V)

`crates/recomp-gfx/src/maxwell/translate.rs` turns a guest shader program into
a SPIR-V module the host Vulkan renderer can run. `crates/recomp-gfx/src/spirv.rs`
is the small module writer it uses (SPIR-V 1.0, or 1.3 when a program needs
quad operations).

## Program layout

A program is the 0x50-byte shader program header (`sph::ShaderProgramHeader`)
followed by 64-bit instructions. Every 32-byte bundle of the instruction stream
starts with a scheduling word, so the first instruction is at 0x58 and the
words at 0x70, 0x90, ... are skipped.

Compute programs have no header: their code starts at the dispatch's program
address, bundles aligned to it. `translate_compute` reads them as a zeroed
header plus code, so both passes are shared; the program becomes a `GLCompute`
entry with `LocalSize` from the launch descriptor (`Environment::local_size`),
`S2R` of the thread and block ids reads `LocalInvocationId` and `WorkgroupId`,
`LDS`/`STS` access a `Workgroup` array of the descriptor's shared memory size,
and `BAR.SYNC` is a workgroup control barrier.

Branch offsets (`BRA`, `SSY`) are relative to the next slot. A target can land
on a bundle's scheduling word; that means the bundle's first instruction (LM3's
vertex program has an `SSY` whose join point is computed this way).

## Passes

1. **Control flow.** Instructions reachable from the entry are found by
   following branches. Each `SYNC` returns to the join point of the innermost
   `SSY` region holding it, in program order. `SSY` ignores its predicate bits;
   `SYNC`, `BRA`, `EXIT` and `KIL` may be predicated. Blocks start at the
   entry, at branch and join targets, and after instructions that may leave.
2. **Code.** The blocks become the cases of a `switch` on a block index inside
   a loop, which keeps any branching structured as SPIR-V requires.
   Predicated instructions become structured `if`s inside their case.

## Resources

| guest | host |
|-------|------|
| registers R0-R254 | 32-bit function variables; floats are bit casts |
| predicates P0-P6 | boolean variables; PT is the constant true |
| constant buffer `c[s][o]` | a uniform buffer of 4096 `uvec4`s at binding `s` |
| texture handle at byte `o` of the texture constant buffer | a combined image sampler at binding 32 + n |
| generic attribute N (`a[0x80 + 16N]`) | location N (vertex input, varying) |
| `a[0x70..0x7C]` | `Position` (vertex), `FragCoord` (fragment) |
| `a[0x2C0 + 4N]` (vertex output) | `ClipDistance[N]` when the draw enables distance N (`SET_USER_CLIP_ENABLE`, 0x1510); writes to disabled ones are dropped and every element starts at 0 (inside), since Vulkan clips by all elements of the array |
| local memory (`LDL`/`STL`) | a `Function` array of 32-bit words, the header's local size (at most 16 KiB); addresses past it use word 0 |
| fragment color registers at `EXIT` | outputs per target, from R0 on, as the header's output map lists them |

Vertex programs use descriptor set 0 and fragment programs set 1, so one
pipeline layout holds both. `TranslatedShader` reports the constant buffer
slots, texture handles, inputs and outputs a program uses, which is what the
renderer binds.

Hardware behaviours the translation keeps:

- Vertex outputs start as (0, 0, 0, 1): programs often write only some
  components (LM3's full-screen pass writes position x and y).
- `IPA` returns the interpolated value divided by w for perspective
  attributes, and programs multiply by w themselves; the translation does the
  same (as yuzu does), so the host's perspective-correct value comes out.
- `FMNMX` is NaN-aware (the other operand wins).
- `F2I` clamps to the destination range and turns NaN into zero.
- `RRO` passes its value through to `MUFU`.
- `PSETP` combines three predicates with two boolean operations and writes
  the result and its inverse (`P` and the predicate in bits 0:2).

## Instructions

Translated: `MOV`, `MOV32I`, `FADD`, `FADD32I`, `FMUL`, `FMUL32I`, `FFMA`,
`FMNMX`, `MUFU`, `FSETP`, `FSET`, `ISETP`, `ISET`, `PSETP`, `IADD`, `IADD32I`, `ISCADD`,
`IMNMX`, `SEL`, `FLO`, `LOP32I`, `LOP3` (any truth table), `BFE`, `BFI`,
`SHL`, `SHR`, `F2I`, `I2F`, `F2F` (16/32-bit floats, rounding to integral
values), `HADD2`, `HMUL2`, `HFMA2` (all forms; both half lanes computed as
32-bit floats, as Ryujinx does), `RRO`, `IPA`, `ALD`, `AST`, `LDC` (default
address mode), `XMAD` (16 x 16-bit multiply-add in all four operand forms;
not `.X` or `CSFU`), `LDL`, `STL` (8 to 128 bits), and in compute programs
`LDS`, `STS` (32 to 128 bits) and `BAR`; `TEXS`, `TLDS` (texel fetches; not the offset or multisample
targets), `TLD4S` (and `.F16`, results packed as halves), `TEX`, `TXD`
(explicit gradients), `TLD4`, `TXQ` (the size query), `SHFL` (within a
quad), `FSWZADD`, `LOP` (with its predicate result), `S2R` (the
special registers graphics programs read), `BRA`, `BRX`, `SSY`, `SYNC`,
`PBK`, `BRK`, `PCNT`, `CONT`, `EXIT`, `KIL`, `NOP`, `DEPBAR`. Anything else is
reported with its address. `BRK` and `CONT` return to the innermost `PBK` and
`PCNT` region in program order, as `SYNC` does to its `SSY`.

`BRX` is a switch through a jump table: the compiler clamps the case number
(`IMNMX.U32`), scales it (`SHL 2`) and loads the case's code offset from
constant buffer 1 (`LDC c[1][rX + table]`), where NVN keeps a program's own
constants. The translator reads the table from the bound buffer
(`Environment::constant_buffer_1`, which the Vulkan backend fills when it
translates) and makes every case a block; the jump picks the block whose
table entry matches.

`TEX.B` and `TLD4.B` take their texture handle from a register. When it was
loaded from NVN's texture handle buffer (c[2]), at a fixed offset or indexed
(`LDC rH, c[2][rI + base]`, up to 96 instructions earlier), the handle
array's first four entries are bound and sampled, and the result of the entry
the index picks is kept (Ryujinx also binds four: the array's length is not
in the program). The index may be an array index scaled by 8
(`SHL rI, rX, 3`) or a byte offset the program read from its constants: LM3's
materials keep their handles' offsets in a constant buffer, and its 32x32
probe renders (about 5% of the lobby's draws) were skipped until that form
was accepted.

Programs read the renderer's resolution scale from push constants
(`RESCALE_*`, declared only by programs that need it): `FragCoord`'s x and
y are divided by the draw's scale, and texel fetches and size queries of a
texture the renderer bound as a scaled render target multiply or divide by
its scale, so programs work in the guest's pixels at any resolution scale
(see `docs/vulkan-host-backend.md`).

Fragment outputs take each color target's component type
(`Environment::color_targets`): integer targets get the registers' bits as
`uvec4`/`ivec4`.

Texture notes:

- Vulkan accepts a non-constant texel offset only on gathers, so a sample with
  an offset register adds offset / size to the coordinates (`ImageQuery`).
- `TLD4` with per-texel offsets is four gathers, one texel from each.
- `SHFL` butterflies within a quad (segment mask 0x1C, clamp 3) are
  `OpGroupNonUniformQuadSwap`: lane XOR 1 is horizontal, 2 vertical, 3
  diagonal; the index form with a constant lane (an immediate, or RZ) is
  `OpGroupNonUniformQuadBroadcast`. LM3's half-resolution passes read lane
  0's coordinates that way, and its rooms lost about 20 draws a frame
  without it. This needs SPIR-V 1.3 and Vulkan 1.1.

## Developing against a title's programs

Programs are the title's own data and stay out of the repository:

```
RECOMP_CAPTURE_SHADERS=<dir> ./recomp-<title>     # writes <address>-<stage>.bin (and .cb1) per program
cargo run -p recomp-gfx --example shader_listing -- <dir>
cargo run -p recomp-gfx --example translate_shaders -- <dir>
```

`translate_shaders` writes `<name>.spv` next to each capture and checks it with
`spirv-val --target-env vulkan1.1` when that is installed. The unit tests in
`translate.rs` build programs from synthetic instruction words and validate the
modules the same way.

All 102 programs LM3 runs up to and on its title screen translate and
validate (with the captured constant buffer 1, `<name>.cb1`, which
`translate_shaders` reads for `BRX` jump tables).

## Not done yet

- Geometry and tessellation programs; indexed attribute access; `SHFL`
  across more than a quad; `LOP.X`; `BRX` not through a jump table; bindless handles from
  anywhere but c[2], or more than four entries past their base;
  `TXQ` queries other than the size;
  `TEXS.F16`/`TLDS.F16`; 64-bit floats; `LDC`'s indexed address modes; `TXD`
  with texel offsets.
- Validation shows the modules are well-formed. The Vulkan backend
  (`docs/vulkan-host-backend.md`) now runs them: LM3's loading screen renders
  correctly; scenes using depth are the next check.
