# Maxwell 3D command processor (`recomp-gfx`)

This note describes the GPU command processor added in
`crates/recomp-gfx/src/maxwell/`. It decodes the command lists a recompiled
Switch title submits to its GPU channel, keeps the Maxwell 3D register file,
runs the guest's macro programs, and turns the methods that act into renderer
calls.

## Scope and provenance

The register layout and method offsets are reconstructed from two sources:

- A method trace captured while running the recompiled Luigi's Mansion 3
  target (`RECOMP_TRACE_GPU`), which gave the subchannels bound, the MME RAM
  contents, macro start addresses, and the packets games actually submit.
- Ryujinx's `ThreedClassState` (MIT). A sparse copy was used as a reference for
  field positions and semantics; no GPL-licensed emulator code was imported.

The state block is 0x3800 bytes (0xE00 words). Method numbers are word offsets,
so the byte offset of a register is `method * 4`. Offsets 0x60..0xE00 words are
3D state; 0xE00 and above are macro invocations. No proprietary binaries, keys
or assets are stored in the repository; captures and fixtures stay under `/tmp`.

## Module layout

| file          | contents                                                  |
|---------------|-----------------------------------------------------------|
| `methods.rs`  | 208 method byte offsets, `STATE_SIZE`, reserved ranges    |
| `state.rs`    | register file, packed 64-bit addresses, typed accessors   |
| `mme.rs`      | macro interpreter plus public instruction encoders        |
| `render.rs`   | `GpuMemory`, `Renderer`, snapshot `DrawCall`/clear calls  |
| `mod.rs`      | `CommandProcessor`, puller methods, state dispatch, stats |

The crate has no host dependencies. A backend implements
`maxwell::render::Renderer`; the Vulkan/window backend lives outside this crate
and will be added with its own feature gate.

## Command list decode

A command list is a stream of words: a header followed by its data words.

| bits    | field        | note                                     |
|---------|--------------|------------------------------------------|
| 11:0    | method       | word offset in the class state block     |
| 15:13   | subchannel   | the class the write goes to              |
| 28:16   | count        | data words that follow                   |
| 31:29   | SecOp        | submission mode                          |

Modes, matching the `SecOp` enumeration in the GPFIFO format:

| SecOp | name           | handled as                                       |
|-------|----------------|--------------------------------------------------|
| 0     | `Grp0UseTert`  | increasing methods                               |
| 1     | `IncMethod`    | increasing methods                               |
| 2     | `Grp2UseTert`  | non-increasing (same method, repeated)           |
| 3     | `NonIncMethod` | non-increasing                                   |
| 4     | `ImmdDataMethod` | inline: the data word is the header's count field |
| 5     | `OneInc`       | first word at the method, the rest at method + 1 |

The class bound to a subchannel comes from puller method 0 (`BIND_OBJECT`);
the 3D engine is class 0xB197. Writes to a subchannel bound to another class
are counted, not interpreted.

## Macros

Games draw through macros, not by pushing the draw methods directly. A method
at 0xE00 or above is a macro call:

- index = `(method >> 1) & 0x7f`
- odd methods push the word onto that macro's argument FIFO
- even methods start the macro with the word as its first argument (`R1`)
- the macro runs when the packet that invoked it is finished

Code is loaded with `LOAD_MME_INSTRUCTION_RAM_POINTER` /
`LOAD_MME_INSTRUCTION_RAM` and the start addresses with
`LOAD_MME_START_ADDRESS_RAM_POINTER` / `LOAD_MME_START_ADDRESS_RAM`, all in the
puller range. Every macro runs in the interpreter, so the processor does not
need to identify macros by hashing their code: draws, clears and state updates
are triggered by the state methods the macro writes.

Interpreter semantics that matter for correctness:

- Instructions are pipelined: the instruction fetched one step earlier is the
  one that executes, so the instruction after an exit still runs (the delay
  slot), and a taken branch suppresses the delay slot's exit flag unless the
  instruction's no-delay bit is set.
- `R0` reads as zero whatever was written to it.
- The immediate is bits 31:14, sign-extended.
- ALU operands are signed 32-bit and widened: add sets the carry from the
  widened sum, subtract from the absence of a borrow. Add-with-carry and
  subtract-with-borrow read the flag back.
- The method address register packs the address in bits 11:0 and the increment
  in bits 17:12; a send writes `meth_addr * 4` and advances by the increment.
- Bitfield operations replace or extract a field without touching the rest of
  the destination register.
- A step budget stops a program whose code was never fully loaded from spinning.

## Methods that act

A state write is stored in the register file; these offsets also do something:

| method        | effect                                            |
|---------------|---------------------------------------------------|
| `DRAW_BEGIN`  | topology (or override), instance index            |
| `DRAW_END`    | builds a `DrawCall` from the state, calls the renderer |
| `INDEX_BUFFER_COUNT` | marks the next draw as indexed               |
| `DRAW_VERTEX_ARRAY_*` / `DRAW_INDEX_BUFFER_*` | single-word draws, first and count packed |
| `VB_ELEMENT_U8/16/32` | inline index elements, least significant first |
| `CLEAR`       | color and/or depth-stencil clear                  |
| `SEMAPHORE_CONTROL` | report: release a payload, optionally with a timestamp |
| `UNIFORM_BUFFER_UPDATE_DATA` | inline constant-buffer word              |
| `SYNCPOINT_ACTION` | counted; the processor has no hardware caches   |

Inline constant-buffer updates mirror Ryujinx's `ConstantBufferUpdater`: the
word is written to guest memory at the bound buffer's address plus its offset,
and the offset register advances by four. Direct draws take their range from
`VERTEX_BUFFER_DRAW_STATE`, indexed draws from `INDEX_BUFFER_COUNT` and the
bound index buffer.

`DrawCall` is a snapshot: parameters, index buffer or inline indices, shader
addresses per stage, vertex attributes and buffers, render target state,
viewport, scissor, depth and blend state. The renderer receives a snapshot and
the memory accessor, so it never has to reach into the processor.

## Shader programs (`maxwell::sph`)

A shader program is `0x50` bytes of header followed by 64-bit instruction
words. The stage registers name it by an offset from `SHADER_BASE_ADDRESS`, so
the header is read at that address. `sph::ShaderProgramHeader` parses it:

| field | where | meaning |
|-------|-------|---------|
| `sph_type`, `version` | word 0 bits 4:0, 9:5 | header form and version |
| `stage` | word 0 bits 13:10 | compute, vertex, tessellation control/evaluation, geometry, fragment |
| `mrt_enable`, `kills_pixels`, `does_global_store` | word 0 bits 14, 15, 16 | what the program writes |
| `sass_version` | word 0 bits 20:17 | instruction encoding generation |
| `gp_passthrough`, `does_load_or_store`, `does_fp64` | word 0 bits 24, 26, 27 | pipeline and feature flags |
| `stream_out_mask` | word 0 bits 31:28 | transform-feedback outputs |
| local memory sizes | words 1, 2, 3 bits 23:0 | low, high and CRS local memory |
| `per_patch_attribute_count`, `threads_per_input_primitive` | words 1, 2 bits 31:24 | tessellation input |
| `output_topology` | word 3 bits 27:24 | point list, line strip, triangle strip |
| `max_output_vertex_count`, `store_req_start`, `store_req_end` | word 4 | geometry output limits |
| `imap` | words 6..13 | interpolation of the 32 input attributes, two bits per component |
| `omap_targets`, sample mask, depth | words 18, 19 | fragment output mask |

`Stage::register_index` maps the encoded stage onto the state block indices the
processor already uses (vertex A 0, vertex B 1, ..., fragment 5). A header
encoding compute is read as vertex for graphics stages, as the hardware does.

Instructions share their operand fields:

| field | bits |
|-------|------|
| destination register | 7:0 (255 reads as zero) |
| source A | 15:8 |
| predicate the instruction runs under | 18:16, inverted at bit 19 (7 is always true) |
| operand B | from bit 20: register 27:20, immediate 39:20, constant slot 38:34 plus offset 33:20, branch displacement 31:20 |
| source C | 46:39 |
| predicate a compare instruction writes | 5:3, inverted when bits 2:0 are non-zero |

The opcode is bits 63:50. Inside the instruction stream - the bytes after the
header - every word whose stream-relative address is a multiple of 32 is
scheduler padding, not an instruction, so the first instruction sits at `0x58`
and a listing skips the padding at `0x50`, `0x70`, ... Addresses in a listing
are relative to the program's start, and a branch target is the displacement
added to the next instruction's address.

`sph::PATTERNS` is the full encoding table copied from Ryujinx's `InstTable`
(294 entries): arithmetic, compare, conversion, attribute, constant-buffer,
texture, uniform, vote, and control-flow instructions, with operand B and the
written predicate taken from each entry's operand properties. A pattern with
fewer free bits wins, as the hardware's table does; any other opcode is
reported as `unknown`, and a listing stops at the first unknown word.
`RECOMP_DUMP_GFX` prints each program's header once, per address a draw
references; `RECOMP_DUMP_SHADERS` also prints the first disassembled
instructions, and prints the raw bits of the first unknown word so the table
can be extended from a real program's encodings.

## Integration with the runtime

`recomp-services::gpu::Channel` owns a `CommandProcessor` for each guest GPU
channel. `nvdrv`'s `SubmitGPFIFO` reads the entry list, translates each command
list's GPU address through the channel's mappings, and hands the words to
`submit_list`. Guest memory reaches the processor through a small adapter:
reads and writes go through the address space's mappings, and the clock comes
from the host timer, so semaphore releases and inline constant-buffer updates
land where software waits for them.

Submissions run on the GPU worker thread (`docs/async-gpu-submissions.md`): the
submitting thread gets its fence back at once and `EventWait` blocks until the
worker reaches the syncpoint value. `RECOMP_GPU_THREAD=0` decodes the lists on
the submitting thread, which is the mode the deterministic tests use.

| variable | effect |
|----------|--------|
| `RECOMP_TRACE_GPU=1` | print every method write |
| `RECOMP_TRACE_MACRO=1` | print macro invocations, macro runs, and the state methods a macro sends |
| `RECOMP_DUMP_GFX=1` | print the draws and clears the processor produced, a per-submission summary, and each shader's program header once per program |
| `RECOMP_DUMP_SHADERS=1` | also print the first disassembled instructions of each program |

`RECOMP_TRACE_GPU` only shows direct method writes, so a macro-driven draw is
invisible there. `RECOMP_TRACE_MACRO` prints the interpreter's sends, which is
how a macro that writes `INDEX_BUFFER_COUNT` and `DRAW_END` is identified.

With `RECOMP_DUMP_GFX=1` on the recompiled Luigi's Mansion 3 target, a 200s
run decoded 11,444 submissions and produced 103,053 draws and 91,542 clears:
triangle draws of three vertices, quad draws of 20 and 24 vertices, and clears
of the color and depth-stencil targets at their real addresses and sizes.

None of those draws were indexed, even though the captured macro RAM holds
macros that send `INDEX_BUFFER_COUNT`. The processor now counts indexed draws
separately (`ProcessorStats::indexed_draws`) and `RECOMP_TRACE_MACRO` prints the
interpreter's sends, so the macro traffic can be checked against the draws it
produces.

## Validation so far

`cargo test -p recomp-gfx` covers the interpreter (method sends, argument
fetch order, branches and delay slots, register zero, bitfields, carry) and the
processor (macro-driven draw, direct `DRAW_END`, indexed draw from the count
method, inline index elements, clears, puller and 3D semaphore reports, inline
constant-buffer offset advance, other-class writes).

The processor's decode was checked against the captured LM3 trace: five bound
subchannels, real macro traffic, and the plain draw macro invoked as
`method 0xE32, count 3, mode 5` (topology in the start argument, first vertex
and count pushed as arguments), which writes `DRAW_BEGIN`,
`VERTEX_BUFFER_DRAW_STATE` and `DRAW_END`.

With `RECOMP_DUMP_SHADERS=1` on LM3, the headers of 14 programs parse as sane
vertex and fragment programs (sph type 1/2, version 3, sass 3, imap counts
matching their attribute use), and the listings name real instructions: vertex
programs fetch attributes with `ALD`, write them with `AST` and end with
`EXIT`; fragment programs interpolate with `IPA`, sample with `TLD4S`, and
compute with `MOV`/`FMNMX`. Four words at program tails report `unknown` -
data after the last instruction, not encodings the table misses. The listing
stops there, as the hardware decoder does.

## Not done yet

- Real draws on the Vulkan backend (`docs/vulkan-host-backend.md`): the window
  and presenter exist and show the guest's clear colors, but draws are still
  only recorded, and VI framebuffer upload/presentation is pending.
- Using the translated shaders: `maxwell::translate` turns programs into
  SPIR-V (`docs/shader-translation.md`), and the Vulkan backend does not build
  pipelines from them yet.
- Syncpoint-decrementing semaphore releases advancing syncpoints, as hardware
  does. Submissions advance their own fence; NVN's fences go through
  `SubmitGPFIFO`.
- Uniform buffer, texture and sampler bindings becoming host descriptors.
