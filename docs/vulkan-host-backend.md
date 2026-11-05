# Vulkan Host Backend

## Purpose

`recomp-gfx` decodes the guest's Maxwell 3D streams into renderer calls, but
the workspace had no display: draws and clears were only counted by the
recording backend. `crates/recomp-gfx-vulkan` is the first host backend that
presents frames. It opens an SDL3 window, builds a Vulkan device and swapchain
through `ash`, and registers itself as the renderer factory for GPU channels.

## Crate shape

`recomp-gfx` and `recomp-services` stay dependency-free for CI: the renderer
interface (`maxwell::render::Renderer`) is the boundary, and `recomp-services`
exposes two registration points:

- `gpu::register_renderer(factory)` - a host backend installs a factory that
  returns `Box<dyn Renderer + Send>`; the first GPU channel takes it, later
  channels fall back to `gpu::console_renderer()`.
- `gpu::console_renderer()` - the recording backend with the dump flags, reused
  inside the Vulkan renderer so draw/clear accounting keeps working.

The Vulkan crate depends on `ash` and `sdl3` and is a workspace member. A title
opts in with `graphics = "vulkan"` under `[runtime]` in `title.toml`: the
pipeline then adds the crate as a path dependency of the generated project and
emits a call to `recomp_gfx_vulkan::install()`, made when the title runs with
`RECOMP_VULKAN=1`. `install()` fails loudly and leaves the recording backend in
place if the window or device cannot be created.

## Ownership and thread placement

`install()` is called on the title's main thread, but only to start the
`recomp-window` thread and wait for the surface it reports; SDL is
initialized, and the window and Vulkan surface are created, on that thread.
The window stays alive there for the run (the surface needs it and the
window's lifetime is the title process's lifetime). The `VulkanRenderer`
itself moves to the GPU worker thread when the first channel takes it from
the slot (`docs/async-gpu-submissions.md`), and every Vulkan call after that
happens on that one thread, so no external synchronization is needed. The
renderer slot is a `Mutex<Option<VulkanRenderer>>`; the factory takes the
renderer once.

SDL records the thread that initializes video as its video and event thread
(and on Windows, Win32 messages are pumped on the thread that created the
window, which is the same one), so the window, events, gamepads and audio
stay on one thread for the run. macOS is the platform that requires the
process's first thread; the title's `main` would have to move off it there,
which this backend does not do. On Windows, `SDL3.dll` must sit next to the
executable (or on `PATH`), and `vulkan-1.dll` is loaded dynamically from the
executable's directory, `System32` or the Vulkan SDK's `PATH` entry; no
Vulkan import library is linked. See `docs/windows-build.md`.

## What a submission presents

`gpu.rs` records the guest's draws into one command buffer per submission:

- **Render targets** are images kept in the `GENERAL` layout so they can be
  rendered to (dynamic rendering, Vulkan 1.3), sampled and blitted. Each guest
  address and format has one image, as large as the largest size rendered
  there: LM3's dynamic resolution renders every frame at a new size into the
  same memory (1280x720 one frame, 1264x710 the next), and its temporal
  effects read the previous frame from it. A key per size names that image; a
  draw renders into the corner its size covers. A texture header at a target's
  address samples it at the header's size: the image itself, or a copy of its
  top-left corner at that size, refreshed when the target has been written
  since (sampling normalizes coordinates to the texture's size, so reading a
  1253x705 frame from a 1264x710 image misplaced every shape by up to the
  difference, which slid away as the resolution climbed back after a load).
  Previously each size made a new, empty image (and leaked the old one).
  Draws into the same attachments add to one open rendering instance; a
  global memory barrier comes first only when a draw samples an image written
  since the last barrier, or writes again to one written in an earlier
  instance, and copies, clears and dispatches close the instance and barrier.
  A barrier and a new instance before every draw had the GPU drain about 1900
  times per LM3 lobby frame (about 50 ms of GPU time). sRGB targets are stored as their UNORM twin (`MUTABLE_FORMAT`) and
  viewed as sRGB, so blits copy the bytes the guest wrote. A disabled target
  is left out, or, when an enabled one follows (fragment outputs match
  attachments by position), kept as an unused attachment (no view, format
  `UNDEFINED`). Draws without a fragment program (depth-only passes such as
  shadow maps) run an empty one (`translate::empty_fragment_program`), which
  masks off every color target.
- **Shaders** are translated once per program (`maxwell::translate`, keyed by
  address, attribute types and a hash of the code); pipelines are built per
  shader pair, topology, vertex layout, attachment formats and blend state.
  A submission reads and hashes a program's code once (`shader_keys`), not
  for both stages of every draw: that took 2.4 of the worker's 22 ms a frame
  in LM3's lobby. The worker's maps hash with foldhash, not SipHash.
  `VK_EXT_depth_clip_control` keeps the guest's -1..1 depth range.
- **Data** (vertices, indices, constant buffers) is copied from guest memory
  into a 128 MB stream buffer that is reset each submission. It lives in
  video memory the CPU writes into when there is room for it (resizable BAR;
  else in host memory): shaders read constants many times, and reading them
  over PCIe made the GPU, not the worker, the limit of LM3's lobby.
  Robust buffer access makes reads past a constant buffer's bound size zero.
  A draw uploads only the part of each constant buffer its shaders read
  (`TranslatedShader::uniform_extents`: past the last fixed offset, or all
  64 KiB when the program indexes the buffer at run time), and a submission
  is sent early once the stream is three-quarters full or 12000 descriptor
  sets are allocated. Copying whole buffers filled the stream in LM3's title
  screen, which then skipped most of its draws ("stream full").
  A draw copies each vertex buffer from the lowest to the highest element its
  indices read (bound so that the lowest is vertex 0, with the index offset
  negated), 16- and 32-bit indices as they are (8-bit ones widened to 16), and
  a guest range already copied in the submission is bound again instead.
  16- and 32-bit indices are scanned for that range and copied straight from
  guest memory; the scan is two min/max reductions (`index_bounds`: the
  restart value never lowers the minimum and wraps to zero in the maximum of
  index + 1), which vectorize. Constant buffers and pool entries are also
  read in place. LM3's
  lobby copied 167 MB a frame when every buffer was read from its start and
  every index widened to 32 bits, and then about 73 MB.
- **Vertex cache.** Vertex ranges are kept in a 512 MB buffer in video
  memory across submissions, keyed by guest address and length. Each
  submission that draws a range hashes its guest bytes in place (XXH3, no
  copy) and uploads them again, at a new offset, only when they changed; a
  range that changed twice (per-frame vertex data) is copied into the stream
  instead. A submission that starts with the cache three-quarters full waits
  for the other slot and empties it. In LM3's gameplay 54 of the 71 MB of
  vertices a frame come from the cache: the worker spent 14 ms a frame on
  vertices and was busy 97% of the time at 31 fps (60 FPS selected), and now
  spends 7 ms and is 70% busy; the opening cutscene went from 40-52 to 60.
  The cache keeps a hash per guest page of each range. For ranges of 16 KiB
  or more it asks the runtime which of their pages were written since it
  last looked (`GpuMemory::written_since`, the runtime's write watch) and
  hashes only those again; a range with none is not read. Hashing all 55 MB
  of cached ranges every submission had taken 3 ms a frame. A check that
  hashed every watched range anyway found no change the watch missed in
  2.2 million checks through LM3's boot, title, loading and lobby.
  Quads become triangle lists through generated indices. Strips and fans
  restart where the guest turned restarting on (`PRIMITIVE_RESTART_STATE`:
  enable, index); Vulkan restarts only at all ones, so another restart index
  (and every restart of widened 8-bit indices) is rewritten to that. The
  vertex range a draw's indices name leaves the restart index out only while
  restarting is on. LM3 draws fans of 8-bit indices that restart at 255, and
  small meshes (the ones it swaps in at a distance, plate stacks) with 8-bit
  indices where 255 is a vertex like any other: leaving that vertex out of
  the range placed drew it from whatever followed, as long streaks out of
  far objects, and the fans without restart drew streaks across the screen.
- **Textures** come from the guest's texture header pool (TIC) and sampler pool
  (TSC). A header whose address is a render target samples that image. Others
  are read from guest memory and handed to `decode.rs` worker threads, which
  untile them (block linear, 64x8-byte GOBs) and decode ASTC to RGBA8 with the
  `astc-decode` crate (the host GPU has no ASTC). Every mip level of 1D, 2D,
  array, cube and cube-array textures is prepared (block-linear levels follow each other,
  each in blocks fitted to its height; 3D and pitch-linear textures keep level
  0), and a view shows the levels its header names: with level 0 alone, LM3's
  carpets and floors sparkled as they moved. Some effects are sampled with a
  cube instruction but bound as a plain 2D image (a 32x32 matcap in LM3's
  ghost and gem shaders); the image is uploaded as a cube with the single
  face repeated, rather than sampling the magenta placeholder. Prepared
  textures upload as the next submission begins, up to 32 MB per submission.
  A draw that samples a texture still being prepared waits for it and uploads
  it at once, up to 12 ms per submission in all, and samples a transparent
  placeholder after
  that: new things (the room behind a door, debris in LM3's King Boo chase)
  flickered in over 1 to 37 frames while each new texture waited for the next
  submission. In the chase the frames drawing placeholders went from 180 to
  26, most of them the first frames after loading, when 135 textures arrive
  at once. Decoding a 1024x1024 ASTC texture with its mip chain takes 30-50
  ms on the CPU, so a door opening onto a room seen the first time still
  drew new textures as placeholders for frames (a room seen again has them
  cached). A large texture with a mip chain (256x256 or more, five levels
  or more, a view of every level) also gets a preview job of its levels from
  128 texels on a side down, about 1/64 of the work; previews and textures
  without one go before whole large textures in the decoder's queue, and
  draws sample a preview (a smaller image of those levels: briefly blurry)
  until the whole texture arrives. Up to six decoder threads run (the
  host's threads less four). Loading LM3's Laundry Room and walking, draws
  sampled placeholders 2448 times in 17 submissions, and 570 times in 4
  with previews, all as the room first appears. A draw waiting for a quick
  texture (an ASTC preview, or up to 1 MiB to untile) that no decoder
  thread has started takes the job back and prepares it itself: the color
  lookup table LM3 grades each frame through otherwise waited behind the
  new room's textures, and the frame came out black.

  Textures are cached by header, and the bytes each was read from are
  watched for writes (`GpuMemory::written_since`): a texture the guest wrote
  since is prepared again, checked once per submission (about 100 checks,
  0.1 ms, in a room). Draws sample the old image until the new one is ready
  (quick ones are prepared at once), and a preparation made before the
  write is dropped (`Job::generation`). As a room is entered the first time,
  LM3 fades its look in over two seconds by rewriting nine 3D lookup tables
  in turn, one a frame; kept as first read, the tables graded every fifth
  frame or so much darker, and the room flickered. It also rebuilds a 68x22x21
  3D light grid and a few small textures every frame. Formats not handled yet
  (depth textures in guest memory) sample magenta. Samplers take the header's anisotropic filtering, at least
  `RECOMP_ANISOTROPY` (a script's `setting("anisotropy", 16)`), and sample
  only the base level when the header has no mip filter.

When a submission finishes, the renderer submits it and goes on: each
submission records into one of two slots (command buffer, stream, descriptor
pool, fence, and a pair of swapchain semaphores), and a slot waits for its
previous submission only when it begins again, so the worker records one
submission while the GPU runs the one before. Waiting for every submission
kept the worker idle for a third of each LM3 frame. Each submission starts
with a barrier (the one before may still be running), images a slot replaced
are freed when that slot begins again, and a readback waits for its
submission. Besides read-after-write, a draw that starts a new rendering
instance also waits for earlier draws that sampled its attachments. Submitting
does not present. Frames are presented the way the console's compositor
shows them: `vi`'s QueueBuffer resolves the queued GraphicBuffer to the GPU
addresses its memory is mapped at (LM3's three 1920x1080 swapchain images,
each mapped three times) and passes them, the crop rectangle and the frame's
fences to `gpu::present`. The GPU worker queues that request after the
submissions before it and shows the frame once its fences are reached: the
renderer blits the crop rectangle (LM3 renders 1280x720 into its 1920x1080
images) of the target at that address to the swapchain and presents (mailbox,
so the worker never waits for vsync). Presenting at the end of submissions
instead showed half-drawn frames: LM3 builds a frame in two submissions, and
the loading screen's dog flickered. Before the title queues its first frame,
the largest target drawn stands in. While the title presents nothing (a long
load, or a burst of work that presents nothing), the worker shows the last
frame again every 250 ms, so the window keeps its contents after being
minimized and restored. A present waits at most 50 ms for a swapchain image
and then skips the frame (and showing frames for 250 ms): an X11 window that
is minimized keeps its size but stops giving images back, and an unbounded
wait there hung the renderer for good.

The window can be resized and made fullscreen; the swapchain is recreated at
the new size and the frame is scaled to fit it, centred, with black bars when
the window's shape differs from the frame's. A title that switches its output
resolution at run time (LM3 renders 1920x1080 once docked) fills the same
window.

### Menu

`overlay.rs` draws an [egui](https://github.com/emilk/egui) menu over the
swapchain image after the frame is blitted (`egui-ash-renderer`, dynamic
rendering, so it never touches the guest's targets). F1 opens and closes it
(Escape closes it), F11 toggles fullscreen. The window thread turns SDL's
mouse events into egui input while the menu is open, and the title's
controller reads as idle until it closes. The menu has:

- Fullscreen.
- Docked mode: `system::set_docked` changes the mode and queues `am`'s
  OperationModeChanged and PerformanceModeChanged messages, as docking a
  console does; LM3 switches between 1280x720 and 1920x1080 a second later.
- Frame rate: 30 (the title's own), 60 or 120. `vi` presents at swap interval
  1 above 30 (and at a 120 Hz pace above 60) whatever interval the title
  queues, and the script's `on_frame_rate(fps)` sets the title's own frame
  timing: LM3 reads a constant at 0x93CD4DC every frame (33.33 ms by
  default) for its frame limiter, which the FPS mod by Fl4sh9174 sets to
  16.67 for 60, 8.33 for 120, 5.56 for 180 and 4.17 for 240. The mod's two code patches change the swap
  interval, which `vi` already does.
- Resolution: 1x (the title's own), 1.5x, 2x, 3x or 4x (below).
- Show FPS: frames presented per second, in the top-left corner.

`RECOMP_MENU=1`, `RECOMP_SHOW_FPS=1`, `RECOMP_FULLSCREEN=1`, `RECOMP_FPS=60`
and `RECOMP_RESOLUTION=2` (or a script's `setting(...)`) set them at start.

### Resolution scale

Render targets of at least 256x128 are made at the menu's multiple of the
size the title renders (`Image::scale`, in halves; smaller ones, which are
probes, lookup tables and the last steps of blurs, keep theirs). A draw's
render area, viewport and scissor are scaled by its attachments' scale, 2D
copies by each image's, and the window shows the larger image. Programs
still see the guest's pixels: each takes push constants
(`translate::RESCALE_*`: the draw's scale, the scale of rendered images,
and a bit per texture binding that samples one), and `FragCoord`'s x and y
are divided by the draw's scale, texel fetches (`TLDS`) multiply their
coordinates by the texture's, and size queries (`TXQ`, and the size texel
offsets are divided by) divide by it. Compute programs get the same
constants. A change takes effect as the next submission begins: render
targets made at the old scale are dropped with their contents, and the
title draws them again (a frame or two may look wrong).

In LM3's lobby (docked, 60 FPS mode) 1.5x (2880x1620) runs at 53-55
fps, 2x (3840x2160) at 51-53 and 3x (5760x3240) at 44-46, against 52-58 at
1x: the game's threads limit the frame rate, and the GPU is idle enough to
render four times the pixels. Above 2x the GPU starts to cost frames.

Draws cull the faces the guest culls (`CULL_FACE`, when culling is on) with
its front face (`FRONT_FACE`): OpenGL's winding is defined for a window whose
y goes up, and the guest, rendering with y going down as Vulkan does, sets
`Y_CONTROL`'s flip bit to turn it back, so its winding is Vulkan's with the
flip and the other one without. Its bit 0 negates the viewport's y for flipped
passes (reflections and shadow variants), reproduced as a negative Vulkan
viewport height. The front face also picks the two-sided
stencil test's face. Every draw once drew both faces: LM3's lights draw
their volume with front faces culled when the camera is inside it, and lamps
then lit no floor (or lit it while switched off), boxes of floor flickered as
the camera moved and lights came and went with its angle. Taken the yuzu way
round (flip turning the winding), LM3's back-face culling removed every
front face.

Draws also take the guest's polygon offset (`DEPTH_BIAS_STATE`, factor,
units and clamp): Vulkan's constant factor is the guest's units, its slope
factor the guest's factor, and its clamp the guest's clamp when the device
has the `depthBiasClamp` feature. Coplanar details such as a crown on a
head or decals on a floor z-fight without it.

Depth-stencil targets are images too (24-bit depth becomes 32-bit float depth:
AMD's Vulkan drivers have none), attached when the draw enables its depth
target, with depth test, write and compare state and the stencil test (both
faces' ops, comparison, reference and masks, in the guest's Direct3D-style or
OpenGL encoding) in the pipeline. LM3's light volumes are back-facing shells
with their front faces culled and set only the back face's enable, so the
back state must be kept when the front is off. LM3's deferred lighting shades
only the pixels a light or reflection volume marked in stencil; without the
test the volumes shaded their whole boxes on screen (rectangles in the lobby
and the elevator). The 12 lobby light-volume draws (fragment `0x4016cb030`)
are skipped: their volume mask is not reproduced and in the close-up they
shaded the whole volume, a giant cyan orb around Hellen Gravely's necklace;
the lobby is still lit correctly without them. The gem-glow full-screen pass
(fragment `0x400990030`) is skipped for the same close-up. The rest of the
frame is unchanged. Draws with
only a depth target (shadow maps) run, clears are executed, and a texture at a
depth target's address samples its depth. An address can hold both a color
target and a depth target at different times. A header with a depth format
samples the depth target even when a light mask color target shares the
address: LM3 keeps a light's Z16 shadow map and its filtered R16 light mask at
one address, and sampling the color mask there left the light a disc with no
window frame shadow. A 2D array whose layers the guest drew as targets of
their own, each at its layer's address (the header's block-linear layer
stride apart), samples an array image the layers are copied into, each again
when its target was drawn since; layers never drawn read as cleared. LM3
draws its lamps' shadow maps into a pool of Z16 layers (1024x1024 ones 2 MiB
apart, 512x512 ones 512 KiB apart) and samples the pool as arrays of four
sizes. Sampling the magenta placeholder there, no lamp cast a shadow and its
light reached through furniture and walls. Integer color targets get integer
fragment outputs, integer clears and no blending. Pitch-linear color targets
(LM3's save screen and save thumbnail) render into tiled host images too:
their first size word is a row stride, and the width is that stride over the
format's texel size. A 3D texture instruction on a 2D header, which the save
screen uses, uploads the 2D texture as one slice of a 3D image and samples
z = 0.

### Keep the GPU worker quick

LM3's render thread writes the swapchain image index, signals a job thread,
and on its next frame resets the index to -1 before acquiring again. The job
thread waits on the GPU before it reads the index. On hardware the GPU is
never a frame late, but if the worker stalls for about a frame (20 ms) the
render thread's next frame resets the index first and the job dereferences
`table[-1]` (`memory access error ... at 0x809ce90e8`, guest block 0x87d1b6c).
So the worker must not do slow work inline: texture decoding is on worker
threads, readbacks use cached memory and write files on a thread, and
presentation does not block on vsync. Pipeline creation is still inline and is
the next candidate to move off the worker. (The crash turned out to be a
kernel bug, light events left signalled after waking their waiter, and a
slow worker no longer crashes LM3; draws now wait a bounded time for textures
being prepared.)

## Compute dispatches

The command processor keeps the compute class's (0xB1C0) registers, and a
launch reads the 256-byte queue meta data block it names
(`maxwell::compute`): program offset in the code region, grid and block sizes,
shared memory size and the constant buffers bound (QMD version 1.7 field
positions, as NVIDIA's `cla0c0qmd.h` and Ryujinx's `ComputeQmd`: block size
in words 18-19, the valid mask in word 20, buffers from word 29; reading them
a word or more off gave LM3's title dispatch 319488 threads instead of 64 and
the wrong buffers, which cost two thirds of each frame and garbled the draws
that read its results). `Gpu::dispatch` translates the program once per address, code hash,
block size and shared size, builds a compute pipeline on the graphics
descriptor set layout (constant buffers as uniform buffers, texture handles
from the buffer `SET_BINDLESS_TEXTURE` names), and records the dispatch
between barriers. Programs that do not translate yet are reported once each.

Before that, `maxwell::interpret` tries the dispatch on the CPU, against guest
memory: programs that only compute integers and use constant buffers and
global memory (no shared memory, barriers, textures or floating point) run
there, every thread in turn, and are refused before running anything
otherwise. The program is decoded once per dispatch and the bound constant
buffers read once; `RECOMP_GPU_STATS` counts the instructions run, and each
program's first CPU dispatch is reported with its grid, block and cost. Their results are what the title's CPU code reads: LM3's first
compute program, at its title screen, sums occlusion query counters (end minus
begin, 64-bit with carry chains) into predicates.

NVN uploads compute launch descriptors and constant data with the
inline-to-memory methods (0x60-0x6D) the 3D, compute and I2M classes share:
`LAUNCH_DMA` then `LOAD_INLINE_DATA` words, written to `OFFSET_OUT` once
`LINE_LENGTH_IN` x `LINE_COUNT` bytes have arrived (pitch-linear
destinations; block-linear ones are reported). Compute report semaphores
(0x6C0-0x6C3) release like the 3D class's.

The 2D engine (0x902D, `maxwell::twod`) copies between surfaces: writing the
source origin's integer y starts a scaled copy, which the backend runs as
`vkCmdBlitImage` when the source is a render target (the destination becomes
the render target at its address). Other copies run in guest memory
(`twod::copy_in_memory`): pitch-linear or block-linear on either side (the
block from `BLOCK_SIZE`, a 3D surface's slice from `LAYER`, the same
`maxwell::blocklinear::Volume` layout textures are untiled with), nearest
sampling, formats of the same pixel size. LM3 uploads its title screen's
32x64x32 color lookup table this way, a slice at a time from a linear buffer
the CPU wrote; skipped, the table stayed zero and the color grading turned
the whole 3D background black. The DMA copy engine (0xB0B5) is still
ignored, each method reported once.

A method's data words may continue in the next command list: the pusher's
state carries over from one GPFIFO entry to the next (NVN puts an indirect
draw's parameters, the indirect buffer, in an entry of their own), so a list
that ends inside a method leaves it pending, and a macro started at the end
of a list only runs early when no method continues.

The command stream also increments syncpoints itself: the puller's syncpoint
method (0x1D, bit 0 increments, index in bits 8:19) and the 3D class's
`SYNCPOINT_ACTION` (index in the low 16 bits). Both advance the syncpoint
through `GpuMemory::increment_syncpoint`, waking waiters, besides the value a
whole submission's fence sets when it completes. LM3 increments syncpoint 0
this way and waits on it every frame; before these increments ran, each wait
lasted until its timeout and the loading screen ran at about one frame a
second (title setup at 150-340 s instead of about 50 s). Counter reports
(semaphore operation 2) write the counter, or the payload, with a timestamp
(or one word); occlusion sample counts are not measured and grow with every
report, so every query reads as visible.

## Debugging

- `RECOMP_VULKAN_CAPTURE=<dir>` saves every `RECOMP_VULKAN_CAPTURE_EVERY`th
  (default 300) presented frame as `<dir>/frame-<n>/<address>-<w>x<h>-<format>.ppm`,
  starting at frame `RECOMP_VULKAN_CAPTURE_FROM` (default 0; capturing every
  frame of a whole run fills gigabytes), read back by the renderer itself (desktop screenshot tools can interfere with
  the user's session and miss Vulkan windows). With
  `RECOMP_VULKAN_CAPTURE_WINDOW=1` it saves the window's image instead
  (`window-<w>x<h>.ppm`): the scaled frame with the menu over it. A relative
  capture or dump directory is joined to `RECOMP_BASE_DIR` when set, and on
  Windows to the executable's directory, so a shortcut launch does not write
  into the shell's working directory; elsewhere the working directory is kept.
- `RECOMP_GPU_STATS=1` prints each channel's command processor counters
  every 64 command lists (methods, 3D state writes, macros, draws, clears,
  dispatches, inline words, semaphores, copies, bound classes):
  what a title's submissions contain. Every 2 s it also prints how busy the
  GPU worker was (decoding and recording, submitting and waiting) and how long
  guest threads waited on syncpoints, and every 128 frames the time per frame
  draw recording spent finding shaders, copying vertices, building descriptors
  and pipelines, and recording commands, the time the worker waited for the
  GPU to free a submission slot, the rendering instances begun, barriers, and
  clears recorded inside and outside rendering instances, with the vertex and
  index bytes copied and the vertex bytes the vertex cache held. In LM3's lobby the worker was busy all of every frame (15 fps)
  until the vertex copies and per-draw barriers were cut; it is now idle a
  third of the time and the game's own threads are the limit. `RECOMP_TRACE_GPU_AFTER=<lists>`
  traces methods like `RECOMP_TRACE_GPU` once a channel has run that many
  lists, and `RECOMP_TRACE_IPC_AFTER=<seconds>` starts IPC tracing late
  (both skip the millions of calls a title's loading makes).
- `RECOMP_VULKAN_DUMP=<dir>` writes `trace.txt` with every draw and clear of
  submission `RECOMP_VULKAN_DUMP_AT` (default 300; the first one after it
  with at least 100 draws, as LM3 alternates drawing submissions with ones
  that only copy or compute): shaders, topology, counts,
  targets, viewport, scissor, blend, mask, the textures each draw samples
  (address, size, format, levels, header words) and 2D engine copies. It then
  saves every render target as a PPM, one per following submission.
- `RECOMP_SLOW_FRAMES=<ms>` logs each frame presented more than `ms` after
  the one before (`frame <n> took <t> ms`): with `RECOMP_TRACE_FS=1`, which
  logs save data writes, a hitch while the title saves lines up with the
  writes around it.
- `RECOMP_VULKAN_SKIP_FS=<address>,...` skips draws with those fragment
  programs: dropping groups of draws and comparing captures finds the draw
  behind a glitch.
- `RECOMP_VULKAN_CULL=off`, `invert-front` or `swap-face` overrides face
  culling for a title whose winding or cull state looks inverted.
- Unsupported formats, topologies and shader failures are reported once each
  with a `[recomp-vulkan]` line; the present line counts drawn and skipped
  draws.
- SDL's signal handlers are disabled (`SDL_NO_SIGNAL_HANDLERS`), so `timeout`
  and Ctrl-C stop the title.
- SDL runs on a thread of its own (`recomp-window`; the main thread runs the
  guest): it creates the window and surface, hands the surface to the
  renderer, then handles the window's events for the run. Closing the window
  ends the title. Each pass (4 ms) reads the gamepads and the keyboard
  (`input.rs`) into `hid`. Face buttons map by label (Xbox A is Switch A;
  `RECOMP_PAD_LAYOUT=positional` maps by position); triggers are ZL/ZR,
  Start/Back are Plus/Minus. Sticks have a 12.5% radial deadzone (the camera
  stick 25%, the standard Xbox range), so a resting off-centre stick cannot
  keep turning the camera. Keyboard: A Enter/Z, B Backspace/X, X C, Y V,
  L Q, R E, ZL 1, ZR 3, Plus P, Minus M, left stick WASD, right stick IJKL,
  D-pad arrows. Gamepads are opened as they connect; a joystick SDL has no
  gamepad mapping for is reported and not read. The thread also opens the
  default audio device (48 kHz stereo) as `audout`'s sink. F1 and F11 are the
  menu's and fullscreen's keys (see "Menu").

## Validation

LM3 with `RECOMP_VULKAN=1` on an RX 7900 XT (RADV, Vulkan 1.4): every draw is
executed (43,233 draws, 0 skipped, in a 100 s run) and the window shows the
boot loading screen: Polterpup and "Loading..." (ASTC textures) composited
over the frame's full-screen passes, in the right colors and orientation. All
54 frames captured over that run show it, and none are black.

The title screen ("Press A to Start") renders its 3D scene, logo and text at
the 30 frames per second the title asks for (swap interval 2), about 64 s
after start, with no draw skipped (each reason a draw is skipped is reported
once). The lake in front of the hotel (a `BRX` and half-precision program)
and the lighting pass that colors the hotel (its draw buffer 0 is disabled)
were the last to render.

## Not done yet

- Depth bounds.
- Video decoding beyond VP9 into RGBA (H.264, VP8, JPEG, other VIC formats;
  see `docs/video-decoding.md`). Textures the host's engines rewrite (VIC's
  movie frames, `recomp_gfx::engine_writes`) are read again at the next
  submission.
- Pitch-linear 3D textures, and noticing rewritten texture data where the
  runtime does not watch writes (`RECOMP_WRITE_WATCH=0`, memory outside the
  fastmem reservation): there textures stay as first read. Array
  and cube textures upload every layer (layer stride as the hardware lays
  them out, with shorter blocks for small surfaces); block-linear 3D textures
  upload every slice (`maxwell::blocklinear::Volume`: a block is its z
  slices, each a column of GOBs). LM3's title screen grades its colors through a 3D
  lookup table; sampling the magenta placeholder in its place turned the
  whole screen pink.
- Pipeline creation off the GPU worker (see above), and a pipeline cache on
  disk.
- 2026-10-31: the runtime clock reads the invariant TSC on x86-64 instead of
  `clock_gettime`/vDSO, `NvOsGetTimeUS` has a host implementation, and scalar
  FABD SIMD is specialized at generation time. In the lobby the 60 FPS setting
  holds 60; the 120 FPS setting measures 89-92 fps over 30 s windows on the
  i5-12400F test machine, with the GPU worker 30-40% busy. The changes removed
  the clock conversion from the main thread's profile but did not move the
  ceiling: the main thread is still dominated by the title's lifted FP/SIMD
  work and call dispatch, and the job thread (`guest-30008`) is idle about a
  third of the time waiting for it. 120 fps is therefore not reachable on this
  CPU by trimming runtime overhead alone; 240 fps is further away. The test
  monitor's maximum refresh is 180 Hz.
- At 60 frames per second LM3's title screen runs at 49-51 (docked) and
  52-56 (handheld) before the round below; it is at 60 now, as is the
  opening cutscene. Gameplay runs at 52-59 (31-35 before generated code kept
  the memory base per function and inlined the rest of the SIMD evaluator
  calls, 40-47 before the worker's per-draw work above, 51-56 before host
  vector SIMD and direct calls). The game's own threads are the limit: the
  main thread does about 13 ms of work a frame, spread over the game's
  logic, and waits a fifth of the time for its job threads (0x30008,
  `fn_87adb90`), which wait about as much for it. The worker is 73% busy since the write watch (14 ms a
  frame: 4.4 ms of vertices, most of it copying the 17 MB of vertex data
  the game writes every frame, 3.6 ms of descriptors and pipelines, 1.5 ms
  of commands, and decoding). The main thread had waited for the worker in
  NVN's fence wait (polling a semaphore and the clock), which is why the
  clock reads looked hot. The host GPU is mostly idle. Setting the value the FPS
  mod's Ryujinx cheat writes (heap + 0x3C0E3754, 2 here too) to 0 changed
  nothing. The game's speed with dropped frames is not checked yet.
