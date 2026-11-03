# Android scaffold (arm64-v8a, SDL3 + Vulkan)

Status: **M0/M1 scaffold**. It builds a signed APK, launches the SDL3 activity,
and loads a generated game library. It does not run a title yet.

This is a native port of the recompiled title, not an emulator. No game assets,
keys, or proprietary binaries live in this tree. Design and coordination notes
are in [`docs/android-port.md`](../docs/android-port.md).

## What is here

```
android/
  app/                      Gradle Android application
    src/main/AndroidManifest.xml
    src/main/java/dev/switchrecomp/android/RecompActivity.java
    src/main/java/org/libsdl/app/   generated from SDL3 (gitignored)
    src/main/res/values/strings.xml
    src/main/assets/        user assets only (gitignored)
    src/main/jniLibs/       generated native payload (gitignored)
  native/                   Rust cdylib libmain.so, exports SDL_main
  stub-game/                non-proprietary test implementation of the game entry contract
  game-entry/               contract, environment variables, and shim for the ARM64 workstream
  scripts/                  native, Java, APK, and ABI checks
```

## How the SDL3 Android path works

The `sdl3` 0.20 crate is only a Rust binding: it ships no Gradle project and no
Java. The `sdl3-sys` 0.7 build script does have Android support. With the
`build-from-source` feature it compiles the pinned SDL3 3.4.18 source with the
NDK toolchain and installs:

* `libSDL3.so` for the target ABI;
* the official `org.libsdl.app` Java sources in `share/java/SDL3/SDL3-*-sources.jar`.

SDL3's `SDLActivity` loads `libSDL3.so`, then `libmain.so`, then resolves
`SDL_main` in `libmain.so` and calls `SDL_RunApp`. Therefore `libmain.so` must
be a shared library exporting:

```rust
#[no_mangle]
pub extern "C" fn SDL_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
```

That is exactly what `native/src/lib.rs` provides. It prepares storage, loads
`librecomp_game.so`, resolves `recomp_android_main`, and calls it. The title is
not linked into the bootstrap, so the game library can be rebuilt and swapped
without touching the APK's Java.

## Prerequisites

* Android SDK with platform 35 and build-tools (tested with 36.0.0).
* Android NDK 28.2.13676358 (`/opt/android-sdk/ndk/28.2.13676358`).
* Rust target:

  ```sh
  rustup target add aarch64-linux-android
  ```

* JDK 17+ (tested with 21) and `zip`, `unzip`, `keytool`.
* Gradle 8.9+ only if you use the Gradle project. The shell scripts do not
  need Gradle.

The native build exports the NDK clang, `llvm-ar`, `CMAKE_TOOLCHAIN_FILE`,
`ANDROID_ABI=arm64-v8a`, and `ANDROID_PLATFORM=android-24` for `sdl3-sys`.
No system SDL3 or Vulkan SDK is needed; SDL3 is built from source and Vulkan is
loaded by `ash` at run time.

## Build

From the repository root:

```sh
# 1. Bootstrap + SDL3 + generated SDL3 Java sources.
android/scripts/build-native.sh

# 2. Signed debug APK (no game library; shows a launch window, then exits).
android/scripts/build-apk.sh

# Or package the non-proprietary stub game library and exercise dlopen.
android/scripts/build-native.sh --stub-game
android/scripts/build-apk.sh --skip-native --stub-game

# Or package an ARM64 game library built by the generated project.
android/scripts/build-native.sh --game-lib /path/to/librecomp_game.so
android/scripts/build-apk.sh --skip-native
```

Output:

```
android/build/SwitchRecomp.apk          (default)
android/build/SwitchRecomp-stub.apk    (when --output names it)
```

The APK is signed with `android/.debug.keystore` (generated on first run,
gitignored). It is a development key, not a distribution key.

To install on a connected device (not run in this milestone):

```sh
adb install -r android/build/SwitchRecomp.apk
adb logcat -s SwitchRecomp SwitchRecompStub SDL
```

### Gradle path

```sh
cd android
gradle :app:assembleDebug
```

`app/build.gradle.kts` registers `buildNative`, which runs
`scripts/build-native.sh --release` before `preBuild`. The Gradle project uses
AGP 8.7.3, `compileSdk 35`, `minSdk 24`, and only `arm64-v8a`.

## APK/ABI layout

```
AndroidManifest.xml                 package dev.switchrecomp.android
classes.dex                         RecompActivity + org.libsdl.app.*
lib/arm64-v8a/libSDL3.so            SDL3 3.4.18, built by sdl3-sys
lib/arm64-v8a/libmain.so            Rust bootstrap, exports SDL_main
lib/arm64-v8a/librecomp_game.so     optional generated title / stub
assets/                             optional user-supplied assets
```

* `minSdkVersion 24` (Vulkan), `targetSdkVersion 35`.
* The activity is `dev.switchrecomp.android.RecompActivity`, a subclass of
  `org.libsdl.app.SDLActivity`.
* `android:extractNativeLibs="true"` keeps the native library directory
  paths predictable for the bootstrap and for `adb push`.
* The manifest requests Vulkan 1.0 and is landscape-only.

## Game-library contract

See [`game-entry/README.md`](game-entry/README.md). The short form:

* artifact: `lib/arm64-v8a/librecomp_game.so` (or `RECOMP_GAME_LIB` path);
* export: `recomp_android_main(argc, argv) -> c_int`;
* return 0 on normal exit, nonzero on fault;
* validate with `scripts/check-game-lib.sh PATH`.

The generated project currently emits a binary `fn main`. The ARM64 workstream
must emit a `cdylib` and add the wrapper in `game-entry/shim.rs` (or add an
equivalent pipeline output mode). This is the main coordination point between
`port/android` and `port/arm64`.

## Assets and saves

The bootstrap creates these paths and exports them before the game entry runs:

| Variable | Meaning |
|---|---|
| `RECOMP_ASSETS` | external app files `assets/` (fallback: internal files `assets/`) |
| `RECOMP_SAVE_DIR` | internal app files `save/` |
| `RECOMP_CACHE_DIR` | app cache directory |
| `RECOMP_ROMFS` | `$RECOMP_ASSETS/romfs.bin`, set only when the file exists |
| `RECOMP_GAME_LIB` | optional absolute override for the game library |

At runtime the external path is normally:

```
/sdcard/Android/data/dev.switchrecomp.android/files/assets
```

and the save path is:

```
/data/data/dev.switchrecomp.android/files/save
```

`recomp-runtime` prefers `RECOMP_ROMFS` over the host path compiled into the
generated entry, so a decompressed RomFS image can be copied into
`assets/romfs.bin` on device. `RECOMP_SAVE_DIR` is already honoured by the IPC
filesystem server. APK assets are packaged by `--assets DIR`, but SDL's
`assets://` namespace and POSIX `std::fs` are different worlds: this milestone
does not extract APK assets into `RECOMP_ASSETS` yet. Use the external app
directory (or add an extractor) until that is implemented.

Do not commit `app/src/main/assets/*`, `jniLibs`, keystores, or game data.
`--assets` stages a directory into the build tree only.

## Verification performed

* `cargo check --target aarch64-linux-android -p recomp-runtime` passes after
  the Android profile/write-watch cfg fixes.
* `cargo check --target aarch64-linux-android -p recomp-gfx-vulkan` passes.
* A test cdylib depending on both crates links for `aarch64-linux-android`
  (8.8 MB, `DT_NEEDED libSDL3.so`), proving the generated game library's link
  surface.
* `scripts/build-native.sh` produces `libmain.so` (`SDL_main` exported,
  `DT_NEEDED libSDL3.so`) and `libSDL3.so` (NDK r28c, API 21).
* `scripts/build-apk.sh` produces a signed, zipaligned APK with
  `arm64-v8a` native code, validated with `apksigner verify` and
  `aapt2 dump badging`.
* The stub APK packages `librecomp_game.so` exporting `recomp_android_main`
  and `recomp_android_abi`.

No emulator or physical device was started, and no game was run.

## Known blockers and next steps

1. **Generated entry shape.** The pipeline still emits a binary. A cdylib entry
   wrapper or an `--android` output mode is required before the real title can
   be packaged. See `game-entry/README.md`.
2. **APK asset extraction.** Bundled assets need a JNI/`AssetManager` extractor
   into `RECOMP_ASSETS`; currently assets are external-only. SDL 3.4.18 also
   does not expose the newer `assets://` base path semantics, so the runtime
   cannot rely on `SDL_GetBasePath`.
3. **Profiling.** `RECOMP_PROFILE` is accepted but is a no-op on Android
   (bionic/libc lacks `ITIMER_PROF` in the Rust bindings and the host PC is in
   an aarch64-specific `mcontext` field). Guest execution still works.
4. **Write watch.** userfaultfd/PAGEMAP_SCAN code compiles and falls back at
   runtime when the kernel/device denies it; it has not been tested on an
   Android kernel.
5. **Fastmem/signals.** The Linux `mmap`/`mprotect`/SIGSEGV path compiles and
   is the next thing to validate on device (Android SELinux and signal
   delivery differ from desktop Linux).
6. **Vulkan/GPU.** The renderer compiles and links for AArch64, but no device
   has presented a swapchain yet. Android surface/lifecycle handling and
   `egui` input need on-device validation.
7. **Audio/input.** SDL3's Android audio and gamepad bridges are packaged, but
   not exercised. `RECOMP_PAD_LAYOUT` and touch mapping are future work.
8. **Saves/permissions.** Saves are app-private and uninstall-sensitive. A
   user-visible export/import path is future work.
