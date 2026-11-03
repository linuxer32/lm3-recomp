# SPEC-040: Runtime and ABI Contract

## Status
Draft v0.5

## Rationale
- Added a minimal runtime ABI stub crate with versioning for early linkage tests.
- Added runtime-facing scaffolds for services, timing, and graphics.
- Added ABI entries for exclusive accesses (`mem_load_exclusive`, `mem_store_exclusive`, `clear_exclusive`) and memory barriers (`memory_fence`), so guest threads that run at the same time contend for real (`docs/parallel-guest-threads.md`).

## Purpose
Define the runtime library and ABI contract between recompiled code and host platform.

## Goals
- Provide a stable ABI for recompiled code.
- Encapsulate platform services and hardware behaviors in the runtime.
- Keep the generated code's dependency surface to this crate, so a runtime change relinks rather than regenerates.

## Non-Goals
- Full hardware-level emulation.
- Exposing a public SDK for game development.

## ABI Requirements
- Register and flag state representation.
- Memory model, alignment, and endianness assumptions.
- Atomic and synchronization primitives.
- Exclusive load/store pairs keep the hardware's reservation and success-flag semantics: the load records what the thread observed, the store writes only if the address still holds it, and returns 0 for success and 1 for failure.
- A store-exclusive's compare and write are indivisible against another thread's, guarded by an address-sharded lock; guest memory itself stays lock-free shared.
- `DMB`, `DSB` and `ISB` map to a host full memory barrier; `LDAR` and `STLR` map to plain accesses, since x86-64's total store order already orders them.

## Runtime Services
- Memory allocation and mapping.
- System services and IPC stubs.
- Graphics and audio dispatch layers.
- Timing, interrupts, and event scheduling.
- Thread scheduling: one thread at a time by default, or genuinely parallel with `RECOMP_PARALLEL=1`.

## Deliverables
- A runtime library with a stable ABI.
- ABI documentation and versioning rules.

## Open Questions
- Which services must be implemented for a first milestone?
- How should ABI changes be versioned and validated?
- Should the exclusive monitor be cleared at a thread context switch, as hardware does?

## Acceptance Criteria
- A unit test that links and executes a minimal recompiled program.
- A defined ABI version embedded in produced outputs.

## References
- https://andrewkelley.me/post/jamulator.html
