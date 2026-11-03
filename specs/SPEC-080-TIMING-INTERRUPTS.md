# SPEC-080: Timing, Scheduling, and Interrupts

## Status
Draft v0.6

## Rationale
- Added a deterministic scheduler scaffold for timing and replay tests.
- Added a trace recorder for deterministic event playback validation.
- Added the two guest-thread scheduling modes: a priority run lock by default, and genuinely parallel host threads with `RECOMP_PARALLEL=1` (`docs/parallel-guest-threads.md`).
- Added the GPU worker thread and blocking syncpoint waits, which release the scheduler while they block (`docs/async-gpu-submissions.md`).

## Purpose
Define timing models and interrupt handling needed for correctness.

## Goals
- Provide deterministic scheduling for CPU, GPU, and IO interactions.
- Model interrupts and timing-sensitive behaviors with precision.
- Let guest threads advance together on host cores once their synchronization is real.

## Non-Goals
- Cycle-accurate hardware emulation for all subsystems.

## Timing Model
- A unified scheduler driving CPU, GPU, and IO events.
- Configurable time steps and event priorities.
- Emulation-style coordination for concurrency where static scheduling is insufficient.
- Serialized thread mode: one guest thread runs at a time, switching at SVC blocking and at call or loop boundaries once a time slice (`RECOMP_TIME_SLICE_US`) is used, handed by priority with a starvation bound (`RECOMP_STARVATION_MS`).
- Parallel mode: no run lock; the host scheduler places guest threads on cores, and exclusive accesses and data barriers carry their real semantics.
- Engine work runs outside guest threads: the GPU worker consumes command lists and advances syncpoints, and a thread waiting on a syncpoint releases the scheduler until it is woken.

## Interrupts
- A defined set of interrupt sources used by titles.
- Deterministic injection points in the scheduler.

## Deliverables
- A timing subsystem in the runtime.
- Instrumentation for tracing and replay.

## Open Questions
- Which subsystems require strict timing for correct gameplay?
- How to validate timing against reference traces?

## Acceptance Criteria
- Deterministic playback of a trace across two runs.
- Stable frame pacing for a minimal test scene.
- With parallel guest threads, no lost lock acquisition: competing `LDXR`/`STXR` loops increment a shared counter exactly once per successful store.
- A thread blocked on a GPU syncpoint is woken when the worker completes the submission, and other guest threads keep running while it waits.

## References
- https://andrewkelley.me/post/jamulator.html
