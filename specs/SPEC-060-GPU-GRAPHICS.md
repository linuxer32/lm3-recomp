# SPEC-060: GPU and Graphics Strategy

## Status
Draft v0.8

## Rationale
- Added a stub graphics API surface and command stream model for early conformance tests.
- Added a deterministic command stream checksum helper for validation.
- Added a Maxwell 3D command processor (`recomp-gfx::maxwell`): the register file, macro interpreter, and renderer interface that a host backend drives (`docs/maxwell-gpu-processor.md`).
- Chose interpreter-driven macro execution over hash-based high-level replacement, so an unrecognized macro still produces the state writes it performs.
- Moved submissions onto a GPU worker thread with real syncpoint waits, so a title's CPU and GPU threads order the way hardware orders them (`docs/async-gpu-submissions.md`).
- Added shader program header parsing and instruction disassembly (`recomp-gfx::maxwell::sph`), following Ryujinx's `ShaderHeader` and instruction table.
- Added a host Vulkan backend (`crates/recomp-gfx-vulkan`, `RECOMP_VULKAN=1`): an SDL3 window, a device and swapchain through `ash`, and per-submission presentation of the guest's clear color (`docs/vulkan-host-backend.md`). `recomp-gfx` and `recomp-services` stay display-free; the backend registers through a renderer factory.

## Purpose
Define the graphics strategy and GPU abstractions required for correctness.

## Goals
- Separate CPU recompilation from GPU behavior where possible.
- Define a stable graphics API boundary for the runtime.
- Decode GPFIFO command lists and keep the Maxwell 3D state the title sets.
- Run the title's GPU macros so draws, clears, and semaphore reports are observed.
- Emit host renderer calls from state snapshots, not from engine internals.

## Non-Goals
- A full GPU hardware emulator.
- A cross-platform high-level renderer replacement, unless required later.
- Emulating hardware caches, counters, and memory barriers that have no host equivalent.

## Strategy Options
- GPU command stream interpretation in the runtime.
- Shader translation pipeline.
- Hybrid approach with selective high-level replacements.

## Requirements
- Accurate handling of synchronization between CPU and GPU (semaphore release and report paths write guest memory).
- Feature coverage for a Maxwell-class GPU baseline: draw state, render targets, viewport, scissor, depth, blend, constant-buffer updates.
- Submissions run apart from the submitting thread: `SubmitGPFIFO` hands the lists to a GPU worker and returns its fence, and a syncpoint wait blocks until the worker reaches the value (`docs/async-gpu-submissions.md`).
- Syncpoint counters separate the values promised to submissions from the values reached, so a fence names the submission it waits for.
- A submission always completes its fence, including one whose channel or command list could not be decoded: a guest must not wait for work the GPU never received.
- Decode on the submitting thread remains available (`RECOMP_GPU_THREAD=0`) for deterministic rendering tests.
- Shader program headers are parsed and instruction streams disassembled (`recomp-gfx::maxwell::sph`), so a backend sees the stage, sizes, and instruction sequence before translation.
- Keep the command processor dependency-free so it can be tested without a display; host backends sit behind the renderer interface.
- Derive state layout from published documentation and MIT-licensed references; no proprietary assets in the repository.

## Deliverables
- A graphics runtime API surface.
- A documented set of supported GPU features.
- A Maxwell 3D command processor with macro execution and a renderer interface.
- A GPU worker thread with a submission queue and syncpoint waits that block.
- A shader program header parser and instruction disassembler (`recomp-gfx::maxwell::sph`).
- A host backend that presents frames: `crates/recomp-gfx-vulkan` (SDL3 window, Vulkan swapchain, `RECOMP_VULKAN=1`) presents each submission's clear; draws and framebuffer upload remain behind it.
- Shader translation from the guest program headers to a host shader format.

## Open Questions
- Which GPU feature set is required by the first target title?
- What level of shader translation is needed?
- Should syncpoint-decrementing semaphore releases advance syncpoints, as hardware does?

## Acceptance Criteria
- Render a minimal test scene generated from recompiled code.
- A graphics conformance test set for core features.
- Draws extracted from a real title's command lists match the state those lists set (topology, counts, targets, shader addresses).
- A submission's fence is not reached until its command lists have been decoded, and a wait on it returns once the worker completes them.

## References
- https://en.wikipedia.org/wiki/Nintendo_Switch
- https://www.nvidia.com/content/tegra/embedded-systems/pdf/tegra-x1-whitepaper.pdf
- https://switchbrew.org/wiki/Hardware
- https://github.com/Ryujinx/Ryujinx (MIT; Maxwell 3D state and macro semantics)
