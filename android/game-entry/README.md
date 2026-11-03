# Game-library entry contract (Android)

The Android bootstrap (`libmain.so`) never links the generated title. It loads
`librecomp_game.so` at run time and calls one exported symbol. The ARM64/generated
title workstream supplies that library; this directory records the exact
contract so the two branches can be developed independently.

## Required artifact

* ELF shared object, `aarch64`, Android API 24 or later.
* Packaged as `lib/arm64-v8a/librecomp_game.so`, or placed at an absolute path
  in `RECOMP_GAME_LIB`.
* `crate-type = ["cdylib"]` with `[lib] name = "recomp_game"`.
* Exported symbol:

```rust
#[no_mangle]
pub extern "C" fn recomp_android_main(
    argc: core::ffi::c_int,
    argv: *mut *mut core::ffi::c_char,
) -> core::ffi::c_int;
```

Return `0` for a normal exit and nonzero for a fault. The arguments are the
same `argc`/`argv` SDL3's activity passed to `SDL_main`; they may be ignored.

## Generated project adaptation

`recomp-pipeline` currently emits a binary crate with `fn main() ->
Result<(), recomp_abi::Fault>` in `src/main.rs`. To turn that project into the
Android game library:

1. Change `Cargo.toml`:

   ```toml
   [lib]
   name = "recomp_game"
   crate-type = ["cdylib"]
   path = "src/lib.rs"
   ```

   and rename/copy `src/main.rs` to `src/lib.rs`.

2. Rename the generated function `fn main` to `fn recomp_main` (keep its body
   and return type).

3. Append the wrapper from [`shim.rs`](shim.rs). It catches a fault and turns
   it into a nonzero status instead of unwinding across the C ABI.

4. On Android, `recomp-gfx-vulkan` selects the SDL3 `build-from-source` feature
   through its target-specific dependency. The NDK must be in the environment
   when the library is built (see `android/README.md`).

A future pipeline flag can emit this shape directly; this wrapper is the
manual bridge until then. `scripts/check-game-lib.sh PATH` validates the
symbol, architecture, and `DT_NEEDED` entries before packaging.

## Environment prepared by the bootstrap

The bootstrap sets these before calling `recomp_android_main`; each value is
only set when the app did not already provide one.

| Variable | Value |
|---|---|
| `RECOMP_ASSETS` | app external files `assets/` (or internal files `assets/`) |
| `RECOMP_SAVE_DIR` | app internal files `save/` |
| `RECOMP_CACHE_DIR` | app cache directory |
| `RECOMP_ROMFS` | `$RECOMP_ASSETS/romfs.bin` when that file exists |
| `RECOMP_VULKAN` | `1` |
| `RECOMP_GAME_LIB` | optional override read by the bootstrap, not set by it |

The generated `main` passes a build-host RomFS path to `init_ipc_services`.
`recomp-runtime` now prefers `RECOMP_ROMFS` when it is set, so an on-device
`assets/romfs.bin` works without regenerating the title. Other relative asset
paths must be resolved against `RECOMP_ASSETS` by the caller.
