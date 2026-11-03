# Performance plan: stable >180 fps

Status: investigation. This is a research target, not a promise.

## Baseline (2026-10-31)

LM3 in the lobby, docked 1920x1080, i5-12400F + RX 7900 XT:

| Setting | Throughput over 30 s |
|---|---:|
| 60 FPS | 60-61 fps |
| 120 FPS | 89-92 fps |
| 240 FPS target | not reachable |

The GPU worker is 30-40% busy at the 120 setting. The guest main thread runs
at ~90-99% and the job thread `guest-30008` at ~99%, idling about a third of
the time waiting for the main thread. The profile is dominated by the title's
lifted FP/SIMD code and call dispatch. Real >180 fps needs roughly 2x the
current per-frame CPU throughput.

The test monitor reports a maximum refresh of 180 Hz, so 240 is not displayable
on this desk setup even if the runtime produced it.

## Workstreams and results

1. Hot-path report (done): the main thread is ~77% generated game code and
   ~14% SIMD evaluator; `rsqrt_estimate`'s linear search was ~8% of main.
   Thread scaling was ruled out: the idling job thread is waiting on the
   main thread, not the reverse.
2. Register spills: the leaf inliner was built and tested but regressed the
   integrated build, so it was removed. The conservative clobber-mask version
   (`perf/clobber`) spills only a direct callee's inputs and reloads only its
   outputs, with a full-spill fallback for hooks, indirect calls and unknown
   callees.
3. SIMD: 15 additional instruction families are specialized at generation
   time, FRECPE/FRSQRTE use const tables, and AVX2/FMA 4-lane paths cover the
   hot vector three-same/element operations. Integer ADD/SUB is enabled too.
4. 180 Hz: `vi` now paces 60/120/180/240 and the F1 menu exposes 180/240.
   The test monitor's maximum is 180 Hz. The frame-limiter question is being
   checked separately: the official 240 mod reuses the 120 frame constant, so
   presentation may go above the simulation rate.
5. Toolchain: `opt-level=3` on the hot chunks is still an open experiment;
   full LTO/PGO is deferred because the generated workspace takes tens of
   minutes per full rebuild.
6. Context ABI: prototype in progress. Suyu's static AOT passes one
   `GuestContext*` per block and lets the C compiler allocate temporaries; our
   current generated code snapshots the whole register file through TLS.
   This is the 25-40% structural bet.
7. Yield spin (`perf/yield`): in parallel mode a yield pauses for
   `RECOMP_YIELD_SPIN` iterations (default 64, 0 disables) before the host
   yield. The A/B baseline is `RECOMP_YIELD_SPIN=0`; it reduced CPU in the
   yield path but moved the lobby by ~1-2 fps. Not yet measured on the final
   direct-register build.
8. Direct registers (`perf/ctxdirect`): the `direct_registers` flag lowers
   every guest register access to `(*ctx).x[N]` / `(*ctx).v[N]` /
   `(*ctx).nzcv` with no prologue register loads and no call spill/reload.
   LM3 measures 163.8 fps at the nominal 180 budget and 179.8 fps with the
   early 5.0 ms pacing target; the binary is ~407 MB. See
   `docs/performance-guest-context-prototype.md`. A later operand-cache and
   dispatch-cache experiment was reverted after a refreshed PGO build
   regressed to 77 fps; the known-good binary is the one above.

## Measured intermediate results (2026-10-31, lobby, same temp save)

| Build | Setting | Throughput |
|---|---|---:|
| old binary (TSC + FABD only) | 120 | 89.7 fps |
| + clobber spills, AVX2/ADD-SUB, 180 Hz pacing, dispatch cache | 120 | 100.9 fps |
| same build | 180 (5.5556 ms budget) | 116.7 fps |
| + FPX | 180 | 116.2 fps (no measurable change) |
| + guest-context ABI + clobber | 120 | 106.5 fps |
| + guest-context ABI + clobber | 180 (5.5556 ms) | 138.4 fps |
| + block-TLS elision + `-O3`/`codegen-units=1` | 180 | 142.1 fps |
| + direct per-instruction context registers | 180 (5.5556 ms) | 163.8 fps |
| + PGO (profile-use, clean harness) | 180 (5.5556 ms) | 168.2 fps |
| + early 5.0 ms pacing target | 180 | **179.84 fps (60 s), 179.37 fps (30 s)** |

The 180 run was verified with a limiter hook: the budget read back as
5.5556 ms and dt was the measured wall time (6.6-8.6 ms under load), not a
duplicated 120 frame. The engine is variable-dt and has no 120 cap.

The guest-context ABI then removed the per-function `[i64; 32]` snapshot while
keeping the clobber fast path: 106.5 fps at 120 and **138.4 fps at the true
180 budget** in the same lobby (+18.5% over the pre-context build, +54% over
the old binary). The generated binary shrank from 1.2 GB to ~0.9 GB. The
remaining profile is dominated by the job thread's yield wait (~35% of its
samples), indirect dispatch (~3.6%), and the title's lifted logic; the next
levers are per-block TLS elision, `-O3`, and eventually per-instruction
`(*ctx).x[N]` access.

The dispatch cache was removed after the profile showed `recomp_abi::miss`
at 5% of samples (7.5% of the job thread); its hit rate was too low. The
remaining main-thread costs are the host `NvOsGetTimeUS` poll (~6.3%),
NvWsi callbacks (~6.8%) and the title's lifted code; the job thread still
spends ~32% in `sched_yield` waiting for work. A full rebuild at `-O3` with
`codegen-units=1` is the current toolchain experiment.

## Heavy-room profile (2026-10-31)

The heavy laundry gameplay room runs 173.8 fps at the 180/5.0 ms target. Its
profile differs from the lobby: `__recomp_rt_svc_id` is **13% of all samples
and 38% of the job thread**; the job worker polls `svcSleepThread(0)`/events.
Next costs are `recomp_abi::dispatch` (~5%), the job game functions
`85b2930`/`84e25d0`/`84df6e0`/`84ddbb0`/`84e3760`, and the Vulkan worker's
draw path (~17% of that worker). The SVC entry is the first target; runtime
fast paths can be relinked without rebuilding chunks. A codegen experiment
that lowered SleepThread/GetSystemTick to direct helpers found the per-tick
helpers 4x faster, but `SleepThread(0)` is dominated by the job thread's
yield/pause loop; removing the dispatcher alone is only ~0.3-0.7%. The
remaining heavy-room work is therefore the yield behavior, dispatch, and the
job-thread game functions, not the SVC wrapper.

## Vector NaN guard (2026-10-31, `perf/nanguard`)

Every single-precision vector operation checks its result for NaN before it
may return the host value: Arm's NaN selection, quieting, signalling and
default-NaN rules are not the host's. The check used to be a `vcmpunordps` /
`vmovmskps` / `test` sequence per instruction, and the `u128` result then
went through two general registers and two stores. The heavy-room job
functions (`85b2930` and the rest) spent 20-36% of their samples there.

The four-lane guard is now one `vcmpunordps` plus one `vptest` on its
all-ones mask, with no general-register transfer; the one- and two-lane forms
keep a `movemask` test, which lowers better at that width. Direct-register code calls
`vector3_into`/`vector_element_into`, which store the result register in
place with one 16-byte access instead of returning bits to be split. The
portable (non-AVX/FMA) path packs the four lanes and detects NaN with one
word-wide bit test instead of an `is_nan` branch per lane. Tests compare the
fast paths with `exec` on zeros, infinities, quiet and signalling NaNs,
denormals and the largest finite values; they also count fallbacks to prove
that finite inputs stay on the host path.

## Native AOT reference

Suyu's static AOT path (public source, `mk8-recomp`) reaches about 59.8 fps
against 60 for its JIT on Mario Kart 8 Deluxe. The reusable ideas are:
- a context/register file instead of per-function TLS snapshots;
- exact native FP with a gate and cold exact fallback;
- a flat block index and bounded block chaining;
- fast guest memory (SwitchRecomp already has a 1:1 fast-memory reservation);
- `-O2`/`-O3` generated code and cold fallback bodies kept out of line.

Suyu's per-title static recompiler emits C and relies on the host compiler
for register allocation. It does not use per-function clobber masks; that is
our lower-risk interim step.

## Rules

- Subagents may not commit, push, or edit `switch-recomp-work/` (the user's
  generated title workspace).
- Each subagent owns an explicit file scope; unrelated changes are reverted.
- Unit tests are required for code changes. Full-suite testing and game
  benchmarks happen in the main session after integration.
- Benchmarks use a fixed save/scene, a 30 s window of `presented` lines, and
  an A/B/A/B run order. Report actual throughput and CPU frame time, not the
  FPS overlay's short-window value. Do not use the elevator/hallway FPS as
  the target metric.
