# Android port: first scaffold

Status: M0 compile / M1 launch scaffold. The APK builds and starts the SDL3
activity; it loads a game shared library but does not run a title. No device or
emulator was used for this milestone, and no game assets are in the tree.

The buildable project, scripts, and operational details live in
[`android/README.md`](../android/README.md). This note records the design and
the findings that the ARM64 workstream needs.

## Target

| Item | Value |
|---|---|
| ABI | `arm64-v8a` (`aarch64-linux-android`) |
| NDK | 28.2.13676358 |
| Min SDK | 24 (Vulkan floor) |
| Target SDK | 35 |
| Window/audio/input | SDL3 3.4.18 through the `sdl3` 0.20 crate |
| GPU | `recomp-gfx-vulkan` through `ash` 0.38 |
| Package | `dev.switchrecomp.android` |

## Why `libmain.so` is a separate bootstrap

SDL3's Android model is Java-first:

1. `org.libsdl.app.SDLActivity` is the activity.
2. It loads `libSDL3.so` (the JNI bridge lives there).
3. It loads the app's main shared object (`libmain.so`) and resolves `SDL_main`.
4. `SDL_RunApp` calls `SDL_main(argc, argv)` on the Java-created thread.

The `sdl3` crate is only a Rust binding; it does not ship an Android project,
`SDLActivity`, or a Gradle plugin. `sdl3-sys` does support Android when the
`build-from-source` feature is enabled: its CMake build detects
`CARGO_CFG_TARGET_OS=android`, maps the Rust target to `arm64-v8a`, and builds
`libSDL3.so`. The same build installs the official Java sources as
`share/java/SDL3/SDL3-<version>-sources.jar`.

The scaffold therefore builds two native pieces:

* `libmain.so`: a small Rust cdylib exporting `SDL_main`, which prepares
  app-private paths and `dlopen`s the title;
* `librecomp_game.so`: the generated title, exported as
  `recomp_android_main(argc, argv)`.

This split keeps the APK shell independent of the generated code. It also makes
the milestone testable before the ARM64 code generator can emit an Android
shared library: `stub-game` implements the contract and can be packaged instead.

## Build inputs and packaging

`android/scripts/build-native.sh`:

* exports `CC_aarch64_linux_android`, `AR_aarch64_linux_android`,
  `CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER`, `CMAKE_TOOLCHAIN_FILE`,
  `ANDROID_ABI=arm64-v8a`, `ANDROID_PLATFORM=android-24`;
* builds `android/native` (SDL3 from source + bootstrap);
* copies `libmain.so` and `libSDL3.so` to
  `app/src/main/jniLibs/arm64-v8a`;
* optionally builds `stub-game` or copies a supplied `--game-lib`;
* extracts the SDL3 Java sources into the gitignored
  `app/src/main/java/org/libsdl/app`;
* validates the game library's architecture and exported symbol.

`android/scripts/build-apk.sh`:

* compiles resources with `aapt2`, injects the package for the non-Gradle path,
  compiles Java against the SDK's `android.jar`, dexes with `d8`, adds
  `classes.dex` and `lib/arm64-v8a/*.so`, zipaligns, and signs with
  `apksigner` using a generated debug key;
* needs no Gradle. The checked-in Gradle project is the Android Studio path.

APK layout:

```
classes.dex
lib/arm64-v8a/libSDL3.so
lib/arm64-v8a/libmain.so
lib/arm64-v8a/librecomp_game.so   (optional)
assets/                            (optional user assets)
```

## Contract with the ARM64 workstream

The generated project must become a `cdylib` named `recomp_game` and export:

```rust
#[no_mangle]
pub extern "C" fn recomp_android_main(
    argc: core::ffi::c_int,
    argv: *mut *mut core::ffi::c_char,
) -> core::ffi::c_int;
```

Return 0 on normal completion and nonzero on fault. The full wrapper, including
the `catch_unwind` shape, is in
[`android/game-entry/shim.rs`](../android/game-entry/shim.rs); the pipeline
currently emits `fn main`, so this is a deliberate manual bridge until an
`--android` emit mode exists. `android/scripts/check-game-lib.sh` fails the
build if the symbol or AArch64 ELF is missing.

The bootstrap exports the runtime environment before the call:

| Variable | Value |
|---|---|
| `RECOMP_ASSETS` | external app files `assets/` (internal fallback) |
| `RECOMP_SAVE_DIR` | internal app files `save/` |
| `RECOMP_CACHE_DIR` | app cache |
| `RECOMP_ROMFS` | `$RECOMP_ASSETS/romfs.bin` when present |
| `RECOMP_VULKAN` | `1` |
| `RECOMP_GAME_LIB` | optional absolute override for the game library |

`recomp-runtime::init_ipc_services` now prefers `RECOMP_ROMFS` over the host
path compiled into the generated entry. `RECOMP_SAVE_DIR` was already used by
the IPC filesystem server. Other asset lookups must root relative paths at
`RECOMP_ASSETS`.

`recomp-gfx-vulkan` gained a target-specific dependency on
`sdl3/build-from-source`, so Android builds do not need a system SDL3 while
desktop builds keep the existing pkg-config path.

## Compile and link results

* `cargo check --target aarch64-linux-android -p recomp-runtime` initially
  failed on seven Linux-only assumptions:
  - `profile.rs`: `libc::REG_RIP`, `ucontext_t.uc_mcontext.gregs`, and
    `libc::ITIMER_PROF` do not exist on Android.
  - `write_watch.rs`: bionic's `ioctl` request type is `c_int`, not the Linux
    `c_ulong`/`Ioctl` used by the original constants.
  The Android profile path is now a no-op and the ioctl constants are cast to
  `libc::Ioctl`. `cargo check` then passes with only unrelated warnings.
* `cargo check --target aarch64-linux-android -p recomp-gfx-vulkan` passes.
* A throwaway cdylib depending on `recomp-runtime` and `recomp-gfx-vulkan`
  linked successfully for `aarch64-linux-android` (8.8 MB,
  `DT_NEEDED libSDL3.so`).
* `build-native.sh` produces `libmain.so` (532 KiB, `SDL_main` exported,
  `DT_NEEDED libSDL3.so`) from NDK r28c.
* `build-apk.sh` produces a signed APK (about 3.9 MiB without a title,
  4.1 MiB with the stub), `minSdkVersion 24`, `native-code: arm64-v8a`,
  launchable `RecompActivity`; `apksigner verify` passes.
* The stub package contains `librecomp_game.so` with
  `recomp_android_main` and `recomp_android_abi`.

## Blockers / unvalidated areas

1. **Pipeline output shape** is still a binary `fn main`; the cdylib wrapper is
   manual. This is the main ARM64 coordination item.
2. **APK assets are not extracted** into `RECOMP_ASSETS`. External app storage
   works today; bundled assets need a JNI `AssetManager` extractor. SDL 3.4.18
   predates the `assets://` base-path behavior, so `SDL_GetBasePath` is not a
   substitute.
3. **Device run**: a realme GT Neo 2 runs the title through the loading
   screen in landscape (see "On-device results"); lifecycle pause/resume,
   audio, gamepads and touch are still unexercised.
4. **Profiling** samples on Android (`RECOMP_PROFILE`, aarch64 `mcontext.pc`);
   the report names `librecomp_game.so` functions from its static symbol
   table.
5. **Write watch** compiles but `userfaultfd`/`PAGEMAP_SCAN` availability on
   Android kernels is unverified; the renderer's conservative fallback is
   expected to take over.
6. **Unsigned distribution** is not set up. The APK uses a local debug key;
   release signing and AAB packaging are future work.

## On-device results (2026-11-01)

The target device is a realme GT Neo 2 (`RMX3370`, Android 13, Snapdragon 870,
Adreno 650, 4 KiB pages, arm64-v8a). It renders in landscape, presents frames
through the Vulkan backend, and reaches the loading screen with the full
7.6 GB RomFS.

Two host limits had to be handled for it:

* **Vulkan 1.1.** The physical device reports `1.1.128` and has no
  `VK_KHR_dynamic_rendering`, so the backend keeps cached render passes and
  framebuffers for devices below 1.3 (see `docs/platform-port-plan.md`).
* **39-bit user address space.** The full 512 GiB fastmem reservation fails,
  so guest memory is the window the title's layout reaches (8 GiB here) and
  generated code checks accesses against the published end.

`RECOMP_ASSETS` names the asset directory. A `recomp.env` file there is read
before startup and each `KEY=VALUE` line is set as an environment variable, so
a device test can toggle `RECOMP_FPS`, `RECOMP_SCRIPT`, `RECOMP_PROFILE`,
`RECOMP_SIMD_STATS`, `RECOMP_TRACE_FS`, `RECOMP_FASTMEM` and so on without
rebuilding the APK:

```sh
adb push recomp.env \
  /storage/emulated/0/Android/data/dev.switchrecomp.android/files/assets/recomp.env
```

`android/scripts/bench-device.sh [seconds]` restarts the activity and prints the
frame rate between the runtime's `presented <n> frames` log lines.

## Next steps

1. Add the `--android` emitted entry (cdylib + `recomp_android_main`) to the
   pipeline, then build a real generated library for `arm64-v8a`.
2. Add APK asset extraction (or an explicit `adb` asset staging workflow) to
   `RECOMP_ASSETS`.
3. Run on a Vulkan-capable arm64 device: confirm the window, surface, fastmem,
   and a single guest frame; then measure.
4. Add lifecycle pause/resume handling and save-on-background.
