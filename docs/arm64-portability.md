# ARM64 Host Portability

Status: first milestone. The runtime and recompiled chunks build and run
the exact scalar fallbacks on AArch64. Performance parity with the x86-64
AVX2/FMA host paths is not part of this milestone.

## Build targets

The generated program and the ABI/SIMD crates are ordinary Rust targets; no
SDL, Vulkan, or linker setup is needed to type-check the portable crates:

```
rustup target add aarch64-linux-android aarch64-unknown-linux-gnu
cargo check --target aarch64-linux-android -p recomp-simd -p recomp-abi
cargo check --target aarch64-unknown-linux-gnu -p recomp-simd -p recomp-abi
```

The host tests and clippy still run on the build host (x86-64 here) and
exercise the portable paths directly; the same test suite can be run on an
AArch64 host without changes.

## Host-specific code audit

Everything that assumes x86 is behind a `cfg`, so an AArch64 build never
compiles or calls it:

| Location | x86-only use | AArch64 behavior |
| --- | --- | --- |
| `crates/recomp-simd/src/fast.rs` | `mod x86` (AVX2/FMA 128-bit integer and floating-point paths), `is_x86_feature_detected!` | Whole-register paths are not compiled; `int_three_scalar`, `vector3_f32` and the per-lane evaluator compute every lane. |
| `crates/recomp-simd/src/lib.rs` | `fused_mul_add_*_host` (`#[target_feature(enable = "fma")]`), FMA detection, MXCSR flush-to-zero test | `mul_add` is baseline FMADD on AArch64; the MXCSR test is `#[cfg(target_arch = "x86_64")]`. |
| `crates/recomp-abi/src/lib.rs` | `simd::fast::host` (`avx` + `fma` target features), `Wide = __m128` for 16-byte memory views | Host module is not compiled; `Wide = u128` keeps 16-byte loads/stores as unaligned little-endian bytes. |
| `crates/recomp-runtime` | SDL/Vulkan backend | Outside this port's scope; not changed. |

The ABI surface generated chunks call (`recomp_abi::*`, including
`recomp_abi::simd::fast::*` and `recomp_abi::simd::exec`) is portable Rust.
A generated project test checks that emitted source contains no
`core::arch`/`std::arch` intrinsic, `__m128`, `_mm_*` or `#[target_feature]`.

AArch64 has Advanced SIMD (NEON) in the baseline target, but this milestone
deliberately uses the scalar/per-lane paths: they are the existing exact
implementation, including Arm's NaN and signed-zero rules. A future NEON
backend can replace them without changing the ABI.

## Target CPU selection

`-C target-cpu=native` is only safe when the program is built for the build
host. For a title built for another architecture it would either fail or
select host features the target device does not have. The pipeline now
accepts the target triple in the title config:

```toml
[runtime]
# Cross-compile the generated project for Android on ARM64.
target = "aarch64-linux-android"
# `native` resolves to `generic` for this target (AArch64 baseline has NEON).
target_cpu = "native"
```

The emitted `.cargo/config.toml` becomes:

```toml
[build]
target = "aarch64-linux-android"

[target."aarch64-linux-android"]
rustflags = ["-C", "target-cpu=generic"]
```

Rules:

- No `target`: `target_cpu = "native"` keeps its old meaning, the build host.
- `target` on AArch64 (or any non-x86 architecture): `native` resolves to
  `generic`; an explicit CPU such as `cortex-a76` is kept as written.
- `target` on x86-64 while building on x86-64: `native` is kept.
- `target` without `target_cpu`: only the target triple is written; Cargo's
  default CPU is used.
- Invalid target/CPU strings (quotes, whitespace, empty) are rejected while
  parsing the title config.

A hand-written override is possible in the generated project's
`.cargo/config.toml`; the pipeline rewrites only files whose first line is
its own header, so a manual config without that header is left alone. To
compile without changing the title config:

```
RUSTFLAGS="-C target-cpu=generic" \
    cargo build --manifest-path out/<title>/Cargo.toml \
    --target aarch64-linux-android
```

## Exactness

The portable paths are the reference implementation used by the existing
equivalence tests (`exec` against the specialized calls on millions of
random and edge-case encodings). On AArch64:

- `mul_add` is a hardware fused multiply-add, matching the single-rounding
  Arm semantics; NaN results still go through the evaluator's Arm NaN rules.
- Vector loads/stores are `u128` unaligned accesses, so guest byte order and
  alignment behavior are unchanged.
- No x86 feature detection or MXCSR state is compiled or executed.

## Caveats

- The scalar/per-lane paths are slower than the x86 AVX2 paths; this
  milestone targets correctness, not frame-rate parity.
- The x86 AVX2/FMA code remains the default on x86-64 hosts.
- SDL/Vulkan windowing and the runtime backend still need their own Android
  port; `cargo check` of `recomp-simd` and `recomp-abi` does not build them.
- No game run is part of this milestone.
