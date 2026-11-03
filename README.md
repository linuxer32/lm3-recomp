# lm3-recomp

lm3-recomp is an experimental static-recompilation toolkit for Nintendo
Switch software. It turns a legally obtained game dump into a native program
that runs on Linux, Windows, and Android.

This repository contains **code and documentation only**. It does not include
game ROMs, XCI/NSP/NCA files, encryption keys, decrypted assets, save data, or
generated game code. You must dump your own games and keep that data outside
this repository. lm3-recomp is not affiliated with Nintendo.

## Status

| Platform | Status | Notes |
|---|---|---|
| **Linux** | **Most stable** | Primary development target. Full SDL3 + Vulkan backend; recommended for playing and testing. |
| **Windows** | **Works, less mature** | Builds and boots through the mingw port. Slower than Linux in testing; some features are still Linux-only. |
| **Android** | **Experimental** | The APK and game-library scaffold build and launch, but new games currently crash. Expect debugging, not a finished experience. |

Linux is currently the fastest and most reliable platform. Windows is usable
but still behind Linux. Android support is early-stage and not ready for
general use.

## Requirements

Common:

- Rust stable and Cargo
- A C/C++ toolchain for native dependencies
- Vulkan loader and a working GPU driver (the offline shader precompiler
  requires Vulkan 1.3)
- SDL3 3.4.18 for the desktop window/audio backend
- Optional: `libvpx` for VP9 cutscenes
- Optional: Nix with `devenv` (`nix develop --impure`, or `direnv allow`)

Platform-specific:

- **Linux:** SDL3 development package, Vulkan loader/driver.
- **Windows:** `rustup target add x86_64-pc-windows-gnu`, mingw-w64, and the
  SDL3 3.4.18 mingw development package. `libvpx-1.dll` is needed for VP9
  cutscenes.
- **Android:** Android SDK 35, NDK 28.2, JDK 17+, and
  `rustup target add aarch64-linux-android`.

## Build the toolkit

From the repository root:

```sh
cargo build --workspace --release
cargo test --workspace
```

The main command-line tool:

```sh
cargo run -p recomp-cli -- --help
```

The synthetic NSO sample can be used to check the pipeline without any game
data:

```sh
python3 samples/nso-boot/generate.py
```

Then follow `samples/nso-boot/README.md`.

## Game data you must supply

### ExeFS — needed at build time

The ExeFS contains the game's executable code. Dump the whole ExeFS directory
and keep the original file names. A typical ExeFS looks like this:

```text
exefs/
  main          # NSO
  rtld          # NSO
  sdk           # NSO
  subsdk1       # NSO; some titles have more subsdk* files
  main.npdm     # process metadata
```

The exact module set and entry addresses are title-specific. The recompiler
needs these NSOs to build the generated Rust project. You do not need the
original executable at run time once the generated program is built, but you
do need the generated `segments/` files produced by the pipeline.

### RomFS — needed at run time

The RomFS contains the game's assets: models, textures, audio, scripts, and
data files. Use a **complete decompressed RomFS dump** from the same game
version/update as the code. A partial or mismatched dump can crash or fail to
load.

The runtime accepts either:

- a decompressed RomFS **directory**, or
- a `romfs.bin` **image file**.

Point the runtime at it with `RECOMP_ROMFS`, or set it in the title's
`title.toml`:

```toml
[runtime.ipc]
romfs_image = "/absolute/path/to/game-data/<title>/romfs"
```

### Keys — only for encrypted dumps

`prod.keys` and, when needed, `title.keys` are required only if you are
extracting/decrypting an XCI, NSP, or NCA yourself. They are **not** needed by
the generated program at run time, and they must never be committed or
distributed.

If you already have a decrypted ExeFS and RomFS dump, you can skip the keys
entirely.

### Save data and scripts

- Save data is created at run time in a `save/` directory, or wherever
  `RECOMP_SAVE_DIR` points. It is not part of the game dump.
- A Rhai script such as `scripts/game.rhai` is optional. `RECOMP_SCRIPT`
  selects one.
- `segments/*.bin` are generated from the title during the build. They are
  title data too; keep them outside public repositories.

### Recommended folder layout

Keep all game data outside the source tree:

```text
game-data/
  <title>/
    exefs/
      main
      rtld
      sdk
      subsdk1
      main.npdm
    romfs/                  # decompressed RomFS, or romfs.bin
    keys/                   # only for encrypted dumps; never commit
      prod.keys
      title.keys
```

`game-data/` is already ignored by `.gitignore`. Generated projects go under
`out/<title>/` and are also ignored.

### If you start from an XCI or NSP

The runtime does not load XCI/NSP files directly. Extract them first with an
external tool you trust, using your own lawfully obtained keys. The
`xci-intake` command is a wrapper around that workflow; see
`docs/xci-intake.md`. Keep the extracted ExeFS, RomFS, and keys outside the
repository.

## Generate a title project

The pipeline has three broad steps: intake, lift, then generated-project
creation. The exact entry addresses and flags are title-specific; use
`docs/DEVELOPMENT.md`, `docs/xci-intake.md`, and the title runbooks as your
reference.

A typical flow, with placeholders:

```sh
# 1. Intake: extract the ExeFS modules and metadata.
cargo run -p recomp-cli -- homebrew-intake \
  --module game-data/<title>/exefs/rtld \
  --nso game-data/<title>/exefs/main \
  --nso game-data/<title>/exefs/subsdk1 \
  --nso game-data/<title>/exefs/sdk \
  --provenance game-data/<title>/provenance.toml \
  --out-dir out/<title>/intake --link-at-runtime

# 2. Lift the guest code to Rust.
cargo run -p recomp-cli -- homebrew-lift \
  --module-json out/<title>/intake/module.json \
  --out-dir out/<title>/lift \
  --entry entry --entry-address <entry-address>

# 3. Emit the generated project.
cargo run -p recomp-cli -- run \
  --module out/<title>/lift/module.json \
  --config game-data/<title>/title.toml \
  --provenance game-data/<title>/provenance.toml \
  --out-dir out/<title>/project \
  --runtime-path crates/recomp-runtime
```

A `provenance.toml` is required. It records the lawful source of your inputs
and their hashes. Do not put keys or game files in it. The files under
`samples/minimal/` are small templates for `title.toml` and
`provenance.toml`.

For the desktop SDL3/Vulkan backend, set `graphics = "vulkan"` under
`[runtime]` in `title.toml`. `target_cpu = "native"` is recommended when the
build and play machines are the same.

## Build and run on Linux

Build the generated project for the host machine:

```sh
cargo build --manifest-path out/<title>/project/Cargo.toml --release
```

The executable is written to `out/<title>/project/target/release/`. Run it
with the RomFS path:

```sh
RECOMP_ROMFS=/absolute/path/to/game-data/<title>/romfs \
  ./out/<title>/project/target/release/<binary>
```

Or place a `romfs` directory / `romfs.bin` next to the executable, or under
`assets/` beside it. Use `RECOMP_ASSETS_DIR` for a directory containing
`scripts/` and `segments/`; use `RECOMP_ROMFS` for the RomFS itself.

Linux is the recommended platform for performance and stability.

## Build and run on Windows

Cross-compile from Linux, or run the same script from an MSYS2 **MINGW64**
shell on Windows.

```sh
rustup target add x86_64-pc-windows-gnu

scripts/build-windows-gnu.sh \
  --project-dir out/<title>/project \
  --sdl3-dir /path/to/SDL3-3.4.18/x86_64-w64-mingw32
```

The staged output is:

```text
out/<title>/project/dist/windows/
  <title>.exe
  SDL3.dll
  libvpx-1.dll          # needed for VP9 cutscenes
  ...
```

Copy `SDL3.dll` and `libvpx-1.dll` beside the `.exe`. If `libvpx-1.dll` is
elsewhere, pass `LIBVPX_DLL=/path/to/libvpx-1.dll` to the build script.
`vulkan-1.dll` normally comes from the GPU driver in `System32`; it can also
be copied beside the executable for an offline bundle.

Place the RomFS next to the executable, under `assets/`, or point
`RECOMP_ROMFS` at it. Use a `--debug` build if you need console logs. Windows
works but is slower than Linux in testing; see `docs/windows-build.md` for
details.

## Build and run on Android

Android is experimental. The scaffold builds an APK and loads a game shared
library, but new titles currently crash. Treat it as a development target.

Prerequisites are in `android/README.md`. A quick scaffold check:

```sh
rustup target add aarch64-linux-android

# Build the stub game library and APK.
android/scripts/build-native.sh --stub-game
android/scripts/build-apk.sh --skip-native --stub-game
```

For a generated title, the project must be built as an ARM64 `cdylib` that
exports `recomp_android_main`; see `android/game-entry/README.md`. Then:

```sh
android/scripts/build-native.sh --game-lib /path/to/librecomp_game.so
android/scripts/build-apk.sh --skip-native
```

The APK is written under `android/build/` (the current scaffold names it
`SwitchRecomp.apk`). Install it on a
connected device:

```sh
adb install -r android/build/SwitchRecomp.apk
```

### Game files on Android

The Android bootstrap uses external app storage:

```text
/sdcard/Android/data/dev.switchrecomp.android/files/assets/
```

Put the RomFS image there as `romfs.bin`:

```sh
adb push romfs.bin \
  /sdcard/Android/data/dev.switchrecomp.android/files/assets/romfs.bin
```

The bootstrap sets `RECOMP_ASSETS` to that directory and `RECOMP_ROMFS` to
`assets/romfs.bin` when it exists. Save data lives in app-private storage
(`files/save`). Other runtime files such as scripts or segments must be placed
under the same `assets/` directory and resolved by the game library.

## Runtime file placement

| File | Purpose | Where it goes |
|---|---|---|
| `exefs/` | Game code used to build the recompiled program | Build machine only; never commit |
| `romfs` or `romfs.bin` | Game assets needed while playing | Linux/Windows: next to executable, `assets/`, or `RECOMP_ROMFS`. Android: `assets/romfs.bin` |
| `prod.keys`, `title.keys` | Decrypting your own encrypted dump | Outside the repo; never shipped |
| `segments/*.bin` | Generated title memory data | Next to executable or under `assets/segments/`; never commit |
| `scripts/*.rhai` | Optional runtime settings/patches | Next to executable or under `assets/scripts/` |
| `save/` | Save data created at run time | Default `save/` next to RomFS, or `RECOMP_SAVE_DIR` |

Useful environment variables:

| Variable | Meaning |
|---|---|
| `RECOMP_ROMFS` | Path to a RomFS image or decompressed dump |
| `RECOMP_ASSETS_DIR` | Root for relative assets such as `scripts/` and `segments/` |
| `RECOMP_SAVE_DIR` | Save-data directory |
| `RECOMP_SCRIPT` | Path to a Rhai script; empty disables it |
| `RECOMP_VULKAN` | `0` selects the recording backend instead of the Vulkan window |
| `RECOMP_SHADER_PACK` | Portable SPIR-V shader pack |
| `RECOMP_PIPELINE_CACHE` | Driver pipeline cache; GPU/driver-specific |

## Performance notes

- Keep `target_cpu = "native"` in a generated `title.toml` if you build and
  play on the same machine.
- Shader translation can be warmed with a portable `shader-pack.bin`; driver
  pipeline compilation can be cached in `pipeline-cache.bin`. The cache is
  GPU/driver-specific, so generate it on the machine that will run the game.
  See `docs/windows-build.md` and `tools/shader-precompile/`.
- Linux currently has the best performance; Windows has additional overhead,
  and Android is not performance-ready.

## Open-source components and credits

lm3-recomp builds on open-source work. The main components are:

- **[SwitchRecomp](https://github.com/bgyss/SwitchRecomp)** — the upstream
  project this source release is based on. See its repository for the project
  source and license metadata.
- **[Ryujinx](https://github.com/Ryujinx/Ryujinx)** (MIT) — Maxwell shader
  instruction tables, shader program header layouts, 3D state and macro
  semantics, host1x/NVDEC/VIC register layouts, and VP9 probability tables.
  Several source files are derived from or follow Ryujinx as a reference.
- **[SDL3](https://www.libsdl.org/)** 3.4.18 (zlib) — desktop window, input,
  audio, and the Android activity layer.
- **[libvpx](https://chromium.googlesource.com/webm/libvpx)** (BSD) — VP9
  video decoding at run time.
- **Rust ecosystem crates** — including `ash` (Vulkan), `sdl3` /
  `sdl3-sys`, `egui` / `egui-ash-renderer` (debug overlay), `rhai` (runtime
  scripts), `clap`, `serde`, `toml`, `sha2`, `lz4_flex`,
  `texture2ddecoder`, `astc-decode`, `foldhash`, and `twox-hash`. Each
  crate remains under its own license.

The full license texts and attribution details are in
`THIRD_PARTY_NOTICES.md`. If you redistribute this project, keep that file
with the source.

## Legal

Use only game data you own and are legally allowed to dump. Do not upload
ROMs, XCI/NSP/NCA files, keys, decrypted assets, generated game projects, or
captured copyrighted media to this repository or its issue tracker. Keep
user-supplied data outside the source tree.

Read `docs/LEGAL-POLICY.md` for the full policy and `THIRD_PARTY_NOTICES.md`
for third-party licenses.

## License

Original lm3-recomp source is available under the MIT License or the Apache
License, Version 2.0, at your option. See `LICENSE-MIT` and
`LICENSE-APACHE`. Third-party components remain under their own licenses.
