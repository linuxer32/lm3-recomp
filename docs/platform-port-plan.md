# Platform port plan: Windows and Android

Status: first compile/boot work. The Android APK shell boots on an emulator;
Windows compiles and links the runtime, with a full title link under test. This
is a native port of the recompiled title, not an emulator.

## Targets

| Target | Host ABI | Immediate goal |
|---|---|---|
| Windows | `x86_64-pc-windows-gnu` (mingw) | Compile, link and open the Vulkan window |
| Android | `aarch64-linux-android` (NDK 28) | Build a minimal SDL3/Vulkan APK and load the game library |

The test machine has mingw-w64, clang/lld, SDL3 3.4.18 and Android NDK
27.1/28.2 installed.

## Milestones

1. **M0 compile**: every crate checks for the target; no Linux-only symbol is
   reachable from the target build.
2. **M1 boot**: the SDL3 window opens, Vulkan initializes, the title reaches
   its first rendered frame.
3. **M2 playable**: input, audio, save/romfs paths and the generated title
   run with the same guest behavior as Linux.
4. **M3 performance**: target-specific SIMD and scheduling tuning.

## Workstreams

| Area | Windows | Android |
|---|---|---|
| Guest memory/fastmem | `VirtualAlloc`/`VirtualProtect` + vectored exception handler | Linux `mmap`/`mprotect`/signals (NDK); no userfaultfd assumption |
| Write watch | Conservative fallback or Windows page guard | Kernel/NDK capability check with fallback |
| Profiling | Windows sampler or no-op | Linux/Android `SIGPROF` where permitted |
| Clock | x86 TSC | `Instant` fallback / ARM counter |
| SIMD | x86 AVX2 as today | ARM64 scalar/NEON exact fallback |
| Window/audio | SDL3 + Vulkan DLLs | SDL3 Android activity/JNI + Vulkan |
| Packaging | mingw binary + `SDL3.dll` + `vulkan-1.dll` | Gradle/APK with native `.so` and SDL3 Java classes |
| Paths | config-relative / `%APPDATA%` | app-private storage / external assets |
| Legal | no keys/assets in repo | same; user supplies dumps |

## Audit findings (2026-10-31)

P0 blockers remaining after the first runtime/ARM64 branches:

- Android guest memory: 512 GiB reservation may fail on 39-bit VA; Android
  15+ uses 16 KiB pages. `port/andmem` adds page-size detection and a checked
  fallback.
- Android app: generated projects are desktop `bin` crates; an APK needs
  `libmain.so`, SDLActivity/JNI, Gradle/Manifest, app-private save/assets, and
  an SDL main-thread activity.
- SDL3: Windows needs `SDL3.dll` and Vulkan loader packaging; Android needs
  SDL3 built from source with the NDK and its Java classes.
- Windows execution has not been tested (no Wine/Windows host); the runtime
  and workspace link for `x86_64-pc-windows-gnu`.
- Write-watch and the profiler are disabled on Windows/Android where the
  Linux kernel facilities are unavailable; vertex hashing is the fallback.

Resolved by `port/pathport`: generated Cargo manifests use `/` separators and
TOML escaping (including cross-drive absolute fallbacks), generated projects
resolve RomFS/script/segments beside the executable or through
`RECOMP_ROMFS`/`RECOMP_SCRIPT`/`RECOMP_ASSETS_DIR`, and `package` now copies
`chunks/` and `.cargo/config.toml`.

No proprietary title assets or keys are part of the port.

## Branch layout

- `port/winrt`: Windows runtime memory/exception/profile layer.
- `port/wingfx`: Windows SDL3/Vulkan build and packaging.
- `port/arm64`: ARM64 SIMD fallbacks and target CPU settings.
- `port/android`: SDL3 Android/JNI/Gradle scaffold.
- `port/pathport`: generated project path portability and packaging.
- Read-only audit: portability matrix and blockers.

Branches are integrated through a separate merge worktree; the Linux
known-good binary in `switch-recomp-work/lm3/recomp-lm3-static-180` is not
touched.

## Progress

- `port/winrt` (`3ccf5cb`): the runtime memory/fault layer compiles and links
  for `x86_64-pc-windows-gnu`. Fastmem uses a 39-bit `VirtualAlloc` reservation,
  `VirtualProtect` commits and a first-chance vectored exception handler;
  write-watch and the profiler are safely disabled on Windows. `recomp-services`
  file-at helpers are portable. No Windows execution test yet (no Wine/Windows
  host). The workspace itself checks/builds for the Windows target.
- `port/arm64` (`7cf04fb`): `recomp-simd` and `recomp-abi` cross-check for
  `aarch64-linux-android` and `aarch64-unknown-linux-gnu`; all x86 intrinsics
  are cfg-gated, exact scalar/per-lane fallbacks compile, and FMA uses the
  AArch64 `mul_add` path. Generated projects now write a target-specific
  `[build] target`/rustflags so `target_cpu = "native"` becomes `generic` on
  cross-compiled AArch64. No NEON vectors yet, so AArch64 SIMD is exact but
  slower than x86 AVX2.
- `port/wingfx` (`837a39a`): Windows SDL3/Vulkan build script, `SDL3.dll`
  staging, opt-in GUI subsystem, config-relative script/RomFS paths, and
  Windows build/packaging documentation. No Windows execution test yet.
- `port/android` (`c9d9ec2`): SDL3 Android activity/JNI bootstrap, signed APK
  scripts, `libmain.so`/`libSDL3.so`/`librecomp_game.so` layout, and runtime
  cfg fixes for the Android profiler and ioctl types.
- `port/andmem` (`8cd12c3`, docs `c56996c`): the runtime detects the host page size
  (`sysconf(_SC_PAGESIZE)`; 16 KiB on Android 15) and aligns every commit,
  release and `MAP_FIXED` address to it instead of assuming 4 KiB. Guest
  offsets stay 1:1 and 4 KiB guest pages inside one host page are zeroed on
  claim/release. A failed 512 GiB reservation (common with a 39-bit user
  address space) leaves every access on the checked path and publishes no
  direct base; no smaller rebased window is used. Android guest threads
  default to a 16 MiB host stack instead of 256 MiB (`RECOMP_THREAD_STACK_MB`
  configures 1..1024 MiB; desktop keeps 256 MiB). Write watch and profiling
  are not built for Android. See
  [android-memory.md](android-memory.md). Cross-checks: host workspace tests
  and clippy, plus `cargo check --target aarch64-linux-android -p
  recomp-runtime`. No device run yet.

## Integration status (2026-10-31)

All port branches are merged in `port/integration`:

- `port/winrt` (Windows memory, exceptions, profiler/write-watch fallback).
- `port/arm64` (AArch64 SIMD/ABI fallbacks and target settings).
- `port/wingfx` (Windows SDL3/Vulkan build, DLL packaging, subsystem).
- `port/android` (SDL3 Android activity/JNI, APK scripts, runtime cfg).
- `port/andmem` (host-page-aware guest memory, Android stack defaults).
- `port/pathport` (portable generated paths, asset lookup, bundle copying).

Host `cargo test --workspace` passes, as do `cargo clippy --workspace
--all-targets -- -D warnings` for `x86_64-pc-windows-gnu` and
`aarch64-linux-android`.

The Android APK scaffold was installed on the `bc_pixel8` x86_64 Android 16
emulator (ARM64 translation through `libndk_translation`). `SDLActivity`
initialized SDL 3.4.18, loaded `libSDL3.so` and `libmain.so`, and called the
stub `recomp_android_main` in `librecomp_game.so`; the stub read the bootstrap
environment and returned 0. This validates the Java/JNI/SDL3/native-library
launch path only: no Vulkan surface, title code, assets, audio or input have
run yet.

Two compatibility additions unblock the existing 506-chunk LM3 project without
regenerating it:

- `recomp-abi::simd::fast::vector3_into` / `vector_element_into` restore the
  in-place ABI the old chunks were generated against.
- The runtime accepts `RECOMP_ASSETS` as an alias for `RECOMP_ASSETS_DIR`, so
  the Android bootstrap's external-files directory is found by the shared
  asset resolver.

The known-good Linux binary in `switch-recomp-work/lm3/recomp-lm3-static-180`
is unchanged.

## Boot results (2026-11-01)

Both targets built the existing 506-chunk LM3 project without regenerating it,
using the compatibility entry points in `recomp-abi::simd::fast` and the
`RECOMP_ASSETS` alias in the shared asset resolver. Chunks were built at
`opt-level=0` for the first boot; the runtime and graphics crates stayed
optimized. This is a boot milestone, not a performance result.

### Windows

- `recomp-lm3-static.exe` (1.9 GiB) links for `x86_64-pc-windows-gnu` and
  imports `SDL3.dll` plus the UCRT/kernel DLLs. It was staged with the official
  SDL3 3.4.18 mingw DLL and the mingw runtime DLLs.
- With no native Windows host available, it ran under Wine 11.21 WoW64. The
  guest reached the Luigi's Mansion 3 title screen at 1920x1080, rendered over
  640 frames and created the expected Vulkan device/swapchain. This used the
  full 7.1 GiB extracted RomFS dump; a 729 MiB subset crashed during the title
  load. The title-screen
  screenshot is at `port-build/win-title.png` in the local work tree.
- This exercises the Windows `VirtualAlloc`/vectored-exception fastmem path,
  SDL3 window/audio initialization, Vulkan presentation, config-relative
  script/RomFS lookup and the portable file-at helpers.
- A native Windows machine is still untested. Wine is a compatibility layer,
  not the shipping target.

### Android

- `librecomp_game.so` (1.8 GiB, `arm64-v8a`) links from the same chunks with a
  small cdylib wrapper exporting `recomp_android_main`. A 505 MiB signed APK
  packages it with SDL3 3.4.18 and the JNI bootstrap.
- It ran on the `bc_pixel8` x86_64 Android 16 emulator through ARM64
  translation (`libndk_translation`). SDL3 initialized, Vulkan used the
  SwiftShader device with a MAILBOX swapchain, and the guest booted past
  `fsp-srv` after a 729 MiB RomFS subset was made available through
  `RECOMP_ROMFS`. The subset is enough for the loading screen but not for the
  title/gameplay; use the full 7.1 GiB dump for real runs.
- The guest reached the Nintendo Switch loading screen and presented over 384
  frames. The title screen was not reached within the test: ARM64 translation
  plus software Vulkan is very slow and the emulator is memory constrained
  (6 GiB, low-memory killer active). The loading-screen screenshot is at
  `port-build/android-title.png` in the local work tree.
- The activity currently renders in a portrait surface even though the
  manifest requests landscape; window orientation/lifecycle handling remains
  open. A real arm64 device and a 16 KiB-page kernel are still untested.
- Fastmem falls back to the checked path when the 512 GiB reservation is
  rejected; the emulator did not provide a 39-bit VA or 16 KiB-page test.

The Linux known-good binary in
`switch-recomp-work/lm3/recomp-lm3-static-180` was not modified.

## Windows performance result (2026-11-01)

The first Windows exe was a **boot build**: generated chunks at `opt-level=0`,
so its 7-9 FPS said nothing about the port. The optimized build uses
`opt-level=3`, `codegen-units=1`, `x86-64-v3`, the performance runtime/SIMD
branch, and native Windows TLS.

`thread_local!` on Windows calls `TlsGetValue`; the lifted code enters a block
on every guest function, and the hot runtime paths also use thread-local
state. Under Wine that was the dominant cost: replacing the hot thread-locals
with `#[thread_local]` (Windows only, built with `--cfg recomp_fast_tls` and
`RUSTC_BOOTSTRAP=1`) moved the fixed lobby benchmark from about **42 FPS to
166 FPS** at the 180 target. Runtime/GUI TLS added about 1 FPS more; applying
the Linux PGO profile to the Windows chunks made no measurable difference.

The measurement is under Wine on the i5-12400F / RX 7900 XT test machine:
guest main and job threads use about 3.3 cores, the GPU worker about 60%, and
the guest job thread is the limiting path. It is still short of the Linux
180 FPS build; native Windows 11 has not been measured, and its scheduler and
Vulkan path may differ from Wine. The build is a performance build, not a
promise of 180 on every Windows machine.

Build a generated project with the optimized profile and:

```sh
scripts/build-windows-gnu.sh --project-dir <project> --fast-tls
```

`--fast-tls` rebuilds every chunk because `recomp-abi` changes; the ordinary
stable `thread_local!` build still works but is much slower.

## Android optimized build (2026-11-01)

The first Android game library was also a boot build: generated chunks at
`opt-level=0`, the compatibility `_into` wrappers, and old chunks. Rebuilding
the same title with `opt-level=3`, `codegen-units=1`, the performance
runtime/SIMD branch, and the native `_into` paths produced:

```text
librecomp_game.so   395 MB  (was 1.8 GB)
APK                 129 MB  (was 505 MB)
```

The APK contains the same diagnostic bootstrap as before: missing assets and
game-start failures show a dialog with the required paths. The emulator test
still runs through ARM64 translation (`libndk_translation`) and software
Vulkan, so its FPS is not representative; it reached the loading screen and
kept rendering. A real arm64 phone is still the required performance target.

The renderer currently requires Vulkan 1.3 (dynamic rendering). Android 13
devices that launched with Android 13 must expose it, but older devices
upgraded to Android 13, including Adreno 650 phones, may only expose Vulkan
1.1/1.2. The phone check is:

```sh
adb shell cmd gpu vkjson | grep -i apiVersion
adb shell cmd gpu vkjson | grep -i VK_KHR_dynamic_rendering
```

If the device has no Vulkan 1.3 and no `VK_KHR_dynamic_rendering`, the current
renderer must be extended with a render-pass fallback or the device is
unsupported.

## Android on-device result (2026-11-01)

The real target is a realme GT Neo 2 (`RMX3370`, Android 13, Snapdragon 870,
Adreno 650, 4 KiB pages, arm64-v8a). Its Vulkan *physical device* is
`1.1.128` even though the loader reports 1.3, and it has no
`VK_KHR_dynamic_rendering`, so the renderer gained a fallback: on a device
below 1.3 the draw path uses cached `VkRenderPass`/`VkFramebuffer` objects
(one pass per attachment-format set, `GENERAL` layout, LOAD/STORE) and
pipelines name that pass instead of `VkPipelineRenderingCreateInfo`. The egui
menu also uses dynamic rendering, so it is skipped there; the title renders
without the overlay. Sparse color-attachment draws cannot be expressed as a
render pass, so they are skipped instead of handing a null view to the
driver (which crashed Adreno with `fault addr 0xe0`).

The activity is landscape: the native entry sets
`SDL_ORIENTATIONS=LandscapeLeft LandscapeRight` before SDL initializes, and
screenshots come back 2400x1080. Earlier runs were portrait because SDL's
full-sensor default followed the phone's auto-rotate setting.

The first on-device run reached the loading screen and presented thousands of
frames, but it was slow: logcat said

```text
[recomp-runtime] could not reserve the 39-bit guest address space
(Out of memory (os error 12)); memory accesses stay checked
```

A 39-bit user address space is 512 GiB in total, so the full-space fastmem
reservation cannot be made on this phone and **every** guest load and store
went through a runtime call. `fastmem` now reserves only the window the
title's memory layout reaches (`reserve_for`): the end is published to
generated code (`recomp_abi::set_memory_range`), which keeps the direct
single-instruction path below it and the checked path above it. For this
title the layout ends at 8 GiB, so the window is 8 GiB. The Android profile
also matches the Windows fast build now (`opt-level=3` for
`recomp-abi`, `recomp-gfx`, `recomp-services` and `recomp-gfx-vulkan`; they
were at 2).

`RECOMP_FASTMEM_GIB=<n>` forces a window of `n` GiB, which exercises the
bounded path on a host that could reserve the whole space;
`crates/recomp-runtime/tests/fastmem_window.rs` covers it.

The swapchain names `IDENTITY` as its image transform, not the surface's
`current_transform`. A landscape app on a portrait-identity panel gets
`ROTATE_90`, and claiming that the image is already pre-rotated (while the
guest's frame is blitted upright) shows the whole frame sideways — which is
what the phone did before the fix. `RECOMP_PRE_TRANSFORM=current`
reproduces the old behaviour for comparison. The startup line

```text
[recomp-vulkan] surface 2400x1080 transform ROTATE_90 (image IDENTITY), ...
```

names both values, so a device log says which case it is in.

### Formats the Adreno 650 cannot sample or filter

`cmd gpu vkjson` on the phone lists no BC format at all
(`textureCompressionBC` is false) and no linear filtering for the 32-bit
float formats (`R32`, `R32G32`, `R32G32B32A32_SFLOAT`) or the depth formats
`D16_UNORM`, `D32_SFLOAT` and `D32_SFLOAT_S8_UINT`. The renderer used both
anyway: BC textures were created as BC images, and samplers filtered
whatever they were given. Both are invalid on that device, and are the
suspected cause of two phone symptoms: the loading screen showed
Polterpup but not its "Loading..." text, and the device was lost a few
seconds into the title screen (black picture, audio still playing). On the
phone the decoded BC textures fixed the text; the lost device has another
cause (below).

- A BC texture the device cannot sample (`SAMPLED_IMAGE` missing from its
  optimal-tiling features) is decoded on the texture workers, as ASTC is
  on desktop GPUs (`crates/recomp-gfx-vulkan/src/bcn.rs`): BC1-3 and BC7 to
  RGBA8 (sRGB kept), BC4/BC5 to R8/RG8 (UNORM or SNORM), BC6H to RGBA16F.
  BC7 goes through `texture2ddecoder`; BC2/BC3 colors always use the
  four-color mode and BC6H keeps its half floats. Devices with BC (every
  desktop GPU) upload the blocks as before.
- A view whose format has no `SAMPLED_IMAGE_FILTER_LINEAR` gets a
  nearest-filter sampler (no linear mips, no anisotropy) unless the
  sampler compares: Vulkan only requires the feature for non-comparing
  samplers, so depth-compare shadow lookups keep their filtering.

`BCN_CHECK_DIR=<dir> cargo test -p recomp-gfx-vulkan bcn_matches_reference`
compares the decoder with another decoder's output for random blocks of
each format (`<format>.blocks` and `<format>.texels`, 4N x 4 texels); against
Pillow 13 every format matched within 1 (rounding) except BC7's reserved
mode, which the spec decodes to transparent black and Pillow to opaque.

### Load time on the phone (2026-11-02)

The first on-device profile (`RECOMP_PROFILE` now starts on Android; it was
compiled for Linux only) showed about 40% of the CPU in the runtime's
checked memory path (`memory::fast_region`, `mem_load`), and
`RECOMP_TRACE_CHECKED` counted 14 million checked accesses a second, most of
them to the TLS and main stack inside the 8 GiB window. `fastmem::map_pages`
turned direct access off for all of guest memory as soon as any region was
mapped above the window, and LM3 maps two there: thread stacks (svcMapMemory
into the stack region the kernel reports at 128 GiB) and a 4 KiB shared
memory page at about 380 GiB.

- A region wholly above the window now lives in a buffer of its own on the
  checked path and leaves direct access on (only a region straddling the
  window's end turns it off). `tests/fastmem_stack_region.rs` covers both.
- When only a window can be reserved, the kernel reports the stack region
  right after the memory layout (2 GiB at 8 GiB for LM3), inside a window
  reserved 2 GiB larger, so stack accesses stay direct.
- A region mapped above the window is named once in the log.

Result on the realme GT Neo 2: the loading screen appears 7 s after launch
(was 33 s) and the title 20 s after (was 3 min 15 s). The profile is now
real work (zlib inflate, guest code) and threads waiting.

Also: guest FPCR.FZ is mirrored into the host FPCR on AArch64 (x86 already
mirrored it into MXCSR), and a suboptimal present no longer recreates the
swapchain unless the surface was resized: Android reports every present
suboptimal while the image transform is IDENTITY on a rotated panel, and
the renderer rebuilt its swapchain, after waiting for the GPU, every frame.

### The GPU fault at the title screen

A few seconds into the title, the Adreno 650 faulted and the device was
lost. The kernel log (`adb bugreport`, KERNEL LOG section) said:

```text
kgsl kgsl-3d0: GPU PAGE FAULT: addr = 4200024100 ... (read translation fault)
kgsl kgsl-3d0: FAULTING BLOCK: UCHE: SP | VSC | VPC | PC | LRZ
kgsl kgsl-3d0: ... gpu fault ctx 28 ctx_type VK ts 5220 ...
kgsl kgsl-3d0: falut=TIMEOUTFAULT ...
```

Placeholders for every texture, no render-target snapshots, depth-only copies,
FIFO presentation and more texture bindings changed nothing.
`RECOMP_VULKAN_SKIP_KIND` (skip draws by kind) found it: skipping every draw
stopped the faults, and so did skipping the two-target draws. LM3's G-buffer
pass draws into two targets with an unused slot before them. On the Vulkan 1.1
render-pass path the pipeline was made for a pass with that slot (the scratch
image's format), but the pass the draw began dropped it: one color attachment
fewer than the framebuffer's views, so the depth attachment was a color image
(LRZ in the faulting units). The draw now begins the pass the pipeline names.

Subgroup operations in fragment shaders were ruled out as the cause, but this
driver does not support them at all (`supportedStages` is COMPUTE): SHFL and
FSWZADD now exchange values across the quad with fine derivatives instead,
valid on every device. The descriptor set now has 32 texture bindings
per stage (LM3's title has a program that samples 17, whose 17th texture read
a descriptor no one wrote), and a draw whose programs need more is skipped
with a warning.

With the pass fixed, the title still lost the device now and then, this time
with `gpu timeout` and `PREEMPTFAULT` in the kernel log and no page fault.
`RECOMP_VULKAN_FLUSH_DRAWS` (one submission per draw, slow ones logged with
their textures) showed a single draw taking about 2 s: an alpha-tested shadow
caster (fragment program 0x4007e3030). Its KIL was an `OpKill` inside the
translated program's dispatch loop, and the Adreno 650 driver hangs on a
discard inside a loop. A fragment program with KIL now sets a flag and leaves
the loop, and the one `OpKill` comes after the loop's merge. The draw takes
under a millisecond, and LM3 has run 100 s at the title (1152 frames, 557269
draws) with no fault.

### Memory ordering on an AArch64 host

With guest memory direct on the phone, LM3 sometimes exited at boot or soon
after the title: `memory access error: guest address 0x0 is not mapped (in the
guest block at 0x87ad1e0, thread 0x30008)`, a null job in its job system.
(The fatal line now reaches logcat: the runtime pauses 200 ms before exiting
on Android, where stdout and stderr go through a pipe a thread reads.)

The lifter turned LDAR and STLR into plain accesses, which x86-64's total
store order keeps in order but an AArch64 host does not, and the checked path
had been ordering them by accident. LDAR is now a load and then
`recomp_abi::acquire_fence()`, STLR `release_fence()` and then the store:
compiler barriers on x86-64, `dmb` on AArch64 (LM3 has 4286 LDAR and 1566
STLR). The runtime's exclusive load and store (LDXR/LDAXR, STXR/STLXR) fence
as well. This needs the module lifted again and the project regenerated.

### Frame rate on the Adreno 650

LM3's title ran at 11 fps on the phone. With every draw skipped the game
itself ran at about 80 fps, so the time went into drawing: the GPU worker was
busy 99% of the time but on the CPU only a third of it, waiting for the GPU.
Each frame is two submissions (the scene, then the present), and with two
recording slots the next scene could not be recorded until the GPU had run
the one before: about 29 ms of recording and 62 ms of GPU work, taking turns.
There are now three slots (`gpu::SLOTS`).

Skipping clears (`RECOMP_VULKAN_SKIP_KIND=clear`) alone gave 16 fps: each
guest clear ended the rendering instance, drained the GPU with a barrier and
cleared the whole image in memory, about 23 times a frame. A clear now waits
for the instance that draws into its image and is recorded there as an
attachment clear (tile memory on a tiler), unless something else uses the
image first.

The title now runs at 15 fps, and the GPU is the limit (the worker waits for
it about 29 ms a frame; the phone's game overlay shows the GPU at 99%).
`RECOMP_GPU_STATS` counts rendering instances, barriers and clears per frame:
about 64, 143 and 37 (12 of them in instances), so fewer barriers and
instances are where more could come from.
