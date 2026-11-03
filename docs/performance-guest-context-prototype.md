# Guest-context codegen prototype

Status: prototype, enabled per title with `[runtime] guest_context = true`.
Off by default; the generated code and runtime behavior are unchanged unless
the flag is set. The per-instruction register-access prototype is a second,
nested flag (`[runtime] direct_registers = true`, requires
`guest_context = true`) and is also off by default.

## Why

Generated functions currently copy the whole `[i64; 32]` register snapshot
from the thread-local (`cpu_regs`) as they start, and spill the live set with
slice literals (`cpu_store`/`cpu_vstore`) around calls. Suyu's Arm
recompiler instead passes one `GuestContext*` and reads and writes its fields
directly, which lets the host compiler keep values in registers. This
prototype moves the generated ABI to that shape without migrating the op
lowering.

## ABI

`recomp-abi` owns the register storage:

```rust
#[repr(C)]
pub struct GuestContext {
    pub x: [i64; 32],   // x0-x30 and sp (31)
    pub v: [u128; 32],  // v0-v31
    pub nzcv: u8,       // N 8, Z 4, C 2, V 1
}

pub fn guest_context() -> *mut GuestContext; // this thread's file
```

The existing `cpu_regs`, `cpu_store`, `cpu_get`, `cpu_set`, `cpu_vregs`,
`cpu_vstore`, `cpu_nzcv` and `cpu_replace` now read and write that one
`GuestContext` in the thread local, so the runtime, SVCs, scripts, thread
starts and debug reports see the same storage. There is one copy of the
register file, not a second ABI.

With the flag on, the emitter writes:

```rust
#[no_mangle]
pub unsafe fn fn_0000000000002000(
    ctx: *mut recomp_abi::GuestContext,
) -> recomp_abi::GuestResult {
    let mut x0: i64 = unsafe { (*ctx).x[0] };
    // ...
    unsafe { (*ctx).x[0] = x0; } // selective, before a direct call
    recomp_abi::call_direct(|| unsafe { fn_0000000000003000(ctx) })?;
    x0 = unsafe { (*ctx).x[0] };
    if x30 as u64 != 0x... { /* unwind */ }
    // ...
}
```

The prologue loads only the registers the function uses (no `[i64; 32]`
copy), and spills and reloads are direct field stores and loads instead of a
slice literal passed to `cpu_store`/`cpu_vstore`. NZCV is the context's
`nzcv` field. The `x30` return-address check, flag hand-off, vector moves,
linear-function spills and split-function parts all keep their previous
semantics.

## Interaction with clobber-aware direct calls

The clobber fast path is preserved in context mode. At a direct call whose
callee's read/write masks are known, generated code still checks
`recomp_abi::direct_calls_enabled()` and then:

- stores only `callee.required | callee.clobber` to the context fields,
  plus NZCV when the callee needs it;
- calls `recomp_abi::call_direct(|| unsafe { fn_callee(ctx) })`;
- reloads only the callee's clobbered registers (and flags) from the
  context.

The `else` branch is the full-live-set spill into the context followed by
`recomp_abi::dispatch` (rewritten by the emitter to `recomp_abi::call` with
the context), which keeps hooks, serialized execution and unknown callees
seeing the whole register file. Indirect calls, tail branches, `Switch`
fallbacks and host implementations still take that full-spill path.

So context mode reduces the entry cost from a 256-byte snapshot to per-field
loads, and keeps the selective call spill. It does not yet remove the
per-call reload of the callee's outputs.

## What is migrated

- `recomp-abi`: `GuestContext`, `guest_context()`, and the existing CPU
  accessors backed by it.
- `recomp-pipeline` codegen behind `[runtime] guest_context = true`:
  function signatures, prologue, call/return spills, reloads, NZCV, vector
  registers, linear functions, split functions, cross-chunk `extern`
  declarations, direct-call closures, chunked and single-file dispatch
  wrappers, the program entry call, and the clobber fast/slow paths.
- Tests: baseline (off) output and clobber fast path unchanged; context-mode
  caller/callee shape with flags, vectors, nested direct calls and the `x30`
  unwind rule; selective fast-path masks; full-spill fallback; indirect call
  and tail-branch fallbacks; linear and split-function emission; chunk
  externs, wrappers and entry.

## Direct per-instruction register access (`direct_registers`)

The above ABI still lowers ops to whole-function locals: the prologue loads
every register the function uses from `(*ctx)` and calls spill/reload the live
set. `[runtime] direct_registers = true` (with `guest_context = true`) removes
that layer. Every guest register read and write in an op line becomes the
context field itself:

```rust
// x2 = x0 + x1
unsafe { (*ctx).x[2] = (unsafe { (*ctx).x[0] })
    .wrapping_add((unsafe { (*ctx).x[1] })); }
// cmp x2, x1
let (res, overflow) = (unsafe { (*ctx).x[2] }).overflowing_sub(unsafe { (*ctx).x[1] });
// ...
unsafe { (*ctx).nzcv = (((res < 0) as u8) << 3) | ...; }
// b.ne
if !((unsafe { (*ctx).nzcv } & 4) != 0) { ... }
```

Temporaries and constant-folded values stay Rust locals. There is no prologue
register load, no `x30`/flag local, and no call spill or reload: the context
already holds the whole register file. A direct call is just
`recomp_abi::call(addr, || unsafe { fn_callee(ctx) })`; the runtime's direct
fast path, hooks, serialized execution and the dispatch-table wrappers all see
the same storage through `ctx`/`guest_context()`. Indirect calls, tail
branches and `Switch` fallbacks read their target register from `ctx` and
dispatch without a snapshot. The clobber fast path is intentionally bypassed
in this mode, because there is nothing left to spill selectively.

The X30 return-address check reads `(*ctx).x[30]`; split-function parts share
the caller's context, so flags stay in `(*ctx).nzcv` and only `(part, block)`
travels through the trampoline.

### What is migrated

All `Op` renderers in `crates/recomp-pipeline/src/pipeline.rs` use one
`RegisterAccess` abstraction:

- integer ALU: const/mov/add/sub/and/or/xor/shift/mul/mul-hi/div/select/clz/
  rbit/rev/pc-rel;
- compare/test/carry ops and their NZCV packing, `read_nzcv`/`write_nzcv`;
- loads, stores, exclusive load/store, clear-exclusive, memory fence;
- vector const/load/store/mov/dup/ins/extract/lane/bitwise/bit-select and the
  SIMD fast path and evaluator fallback (vector reads, general reads, flag
  reads and writes);
- syscalls, sysreg reads/writes, direct and indirect calls, tail branches,
  `Switch`, `Ret`, `BrCond`/`BrTest` conditions;
- linear functions and split-function parts/trampolines.

`scan_line` also recognizes `(*ctx).x[N]`, `(*ctx).v[N]` and `(*ctx).nzcv`, so
the clobber analysis still sees the accesses if it is re-enabled later.

### Verification

- Synthetic pipeline tests assert the direct-mode shape (no register locals,
  context-field ALU/load/store/vector/flags, no spill around a direct call,
  context X30 check) and that the same module without `direct_registers`
  keeps the local ABI.
- An end-to-end generated project (nested `entry -> f1 -> f2` calls, integer
  CMP flags crossing both returns, vector registers crossing both calls, a
  specialized SIMD conversion, an SBCS carry op, a WZCV/RNZCV round-trip, a
  guest store/load, and the X30 return check) was generated twice, once per
  mode, compiled against `recomp-abi`/`recomp-runtime`, and run with
  identically asserted results. The same test also runs with direct calls
  disabled and the generated dispatch table registered, exercising the
  serialized/hook fallback.
- Linear functions with an `Op::Call` and split functions (>64 blocks, flags
  crossing a part boundary) were generated and run in direct mode.
- A generated project containing every `Op` variant and terminator (indirect
  call/branch, `Switch`, `BrTest`, `Trap`) compiles in direct mode.
- The non-direct outputs (snapshot ABI and `guest_context` without
  `direct_registers`) are byte-identical to the baseline commit for the same
  module.

### Remaining work and expected gain

- The full op set is migrated for this prototype, so the remaining work is
  validation and measurement, not op coverage: benchmark LM3 at the true 180
  budget with the flag on, then check the code generated for hot chunks.
- The likely limit is host register allocation. Direct field access is a
  memory load/store per use unless LLVM forwards it; a raw `*mut` field does
  not give the compiler the aliasing information a local does. If the
  benchmark regresses, the next step is to bind multi-use operands to
  instruction temps and keep field stores out of the hot loop.
- Host implementations (`host_functions`) still reach the register file
  through the thread-local accessors, not the passed `ctx`. That is unchanged
  from the context ABI and only matters if a caller passes a context other
  than `guest_context()`.
- Estimated gain: this removes the remaining prologue and call-boundary
  register traffic, but replaces it with per-op context accesses. Expect a
  modest single-digit percent end-to-end if LLVM keeps the values in host
  registers, and a possible regression if it does not. The 142 fps -> 180 fps
  gap is about 21%; this is not expected to close all of it on its own.

## What is not migrated (from the original ABI step)

- The dispatch table still has type `fn() -> GuestResult`. Removing the
  wrappers means changing `recomp_abi::Targets`, `recomp-runtime::GuestFn`,
  `DispatchFn`, script trampolines and thread entry together.
- No full-title regeneration or benchmark was run for this prototype.

## Full-migration estimates

- State: one context of 65 slots (32 GPR, 32 vector, NZCV), 769 bytes of
  payload (784 with alignment). Accessors and the runtime boundary already
  use it, so no per-thread duplication is needed.
- Functions: every generated function changes signature and is recompiled;
  for LM3 that is a regeneration of all 550 chunk crates. Call sites do not
  need hand edits: `direct_calls` already rewrites every direct call, and
  indirect sites inherit the pointer through the table.
- Call-site shape: the clobber fast path changes from `cpu_store`/`cpu_vstore`
  slice literals to per-field stores; the slow path likewise. The number of
  stores is the callee's `required | clobber` set (fast path) or the caller's
  live set (fallback), not the whole 32-register file.
- Removing wrappers is one runtime/ABI change: `GuestFn`/
  `Targets`/`DispatchFn`/script trampolines take `*mut GuestContext` and every
  caller passes `guest_context()`. Then indirect calls stop paying the
  wrapper and TLS lookup. This is the main remaining ABI work.
- Per-instruction field access is the larger win and the larger change: all
  `Op` renderers, condition helpers and SIMD stubs now emit `(*ctx).x[N]`
  expressions behind `direct_registers`, with locals only for instruction
  temporaries. The op migration is complete for this prototype; the
  Suyu-style allocation benefit still has to be proven by an LM3 benchmark.

## Biggest blockers

1. Dispatch/hook boundary: `fn()` table entries and script trampolines must
   carry the context before wrappers can go away.
2. Runtime entry points: `cpu_get`/`cpu_set` are used by SVCs, sysregs, watch
   filters, threads and the script API. They work while the context is the
   thread local; an explicit per-thread context would need to be plumbed
   through those paths.
3. Codegen churn: direct field access touches every op renderer and needs a
   careful review of register aliasing and temporaries, not just signatures.
   The prototype covers the whole op set but has only synthetic and
   generated-project tests, not a retail regeneration.
4. Performance proof: this prototype removes the entry snapshot and the
   slice-literal spill, and the direct-register flag also removes the call
   reload of the callee's clobber set. It needs an LM3 benchmark in the main
   session before deciding whether to make either flag the default.
