# Video decoding (NVDEC and VIC)

Titles play movies on the console's multimedia engines. LM3 plays WebM
movies (VP9, 1600x900, 30 fps) through `nn::movie`: the title card after the
King Boo chase is one, the ScareScraper loading loop another. NVDEC decodes
each frame into a 4:2:0 surface, and VIC converts it into the RGBA surface the
title then draws as a texture (with alpha: without frames, the scene behind
the movie showed through).

## Channels and command buffers

`/dev/nvhost-nvdec`, `/dev/nvhost-vic` and `/dev/nvhost-nvjpg` are engine
channels (`nvdrv::Device::Engine`); the names are the guest's, and the host
does not open them. `MapCommandBuffer` gives each memory
handle an address in the engines' own 32-bit space; `Submit` hands over
command buffers, and the fences each syncpoint increment reaches. `host1x.rs`
runs the buffers at once, on the submitting thread: host1x opcodes select a
class and write its registers, and an engine's methods are reached through its
host interface (`METHOD0` names a method, `METHOD1` writes it, `INCR_SYNCPT`
signals). Writing `EXECUTE` runs the engine. Registers that name buffers hold
engine addresses shifted right by 8. Register and structure layouts follow
Ryujinx (see `THIRD_PARTY_NOTICES.md`).

## VP9 on NVDEC

NVDEC decodes only a frame's tiles. The title's software has read the frame's
headers and hands over their values (a picture info block) and the frame's
probabilities (a 3744-byte table). `vp9.rs` writes both headers again from
those values, ahead of the tiles, and libvpx decodes the whole frame. libvpx
is loaded at run time (`libvpx.so` on Linux, `libvpx-1.dll`/`libvpx.dll` on
Windows); without it, movies are not decoded and the log says so once.
`RECOMP_LIBVPX=<file>` overrides the search path.

- **Probabilities.** Rebuilt frames refresh their frame context and do not
  adapt it backward, so libvpx's four contexts hold the probabilities frames
  ended with, and a compressed header codes the change from one of them to the
  title's. NVDEC is not told which context a frame used. Each frame takes the
  context closest to its probabilities among those every change can be coded
  from: motion vector probabilities update only to odd values, so going back
  to another context can need a change no update codes. One context for every
  frame corrupted the first frame after each hidden alternate reference.
  This decodes correctly only when the stream does not adapt probabilities
  backward either (no symbol counts are produced for the title); LM3's movies
  set `frame_parallel_decoding_mode` in every frame.
- **References.** NVDEC names references by surface; each output surface takes
  one of libvpx's eight slots (the least recently used).
- **Shown or hidden.** Only the next frame's flags say whether a frame was
  shown, and a hidden frame lends the next no motion vectors. A frame is
  decoded when the next one arrives; titles decode several frames ahead of the
  ones VIC converts, and a frame VIC wants first is decoded then, as shown.
  Coding the frame after a hidden one error resilient instead fails: that also
  clears the reference sign biases, and with them compound prediction.

Decoded frames are kept on the host, by output surface, for VIC; NVDEC's own
block-linear output is not written to guest memory.

`cargo test -p recomp-services rebuilt_frames` with `VP9_CHECK_IVF=<file>`
checks the path end to end: it does what the title's driver does with each
frame of a VP9 stream (profile 0, no backward adaptation, no segmentation) in
IVF, decodes through the rebuilt headers, and compares every shown frame with
libvpx's picture of the original. The ScareScraper loop
(`Movies/scarescraper_loading_loop.data` from byte 0x278 is WebM;
`ffmpeg -i loop.webm -c:v copy -f ivf loop.ivf`): 978 frames, 900 shown, all
identical.

## VIC

A configuration structure (read at `SetConfigStructOffset`) describes the
output surface and up to eight input slots. Each enabled NV12 slot's frame is
taken through the slot's color matrix (10-bit components, clamped to the
slot's soft clamp), and the result is written to the output surface in its
format (A8B8G8R8, X8B8G8R8 or A8R8G8B8) and layout (pitch or block linear).
The written range is recorded in `recomp_gfx::engine_writes`, and the Vulkan
renderer drops the textures there as its next submission begins, so they are
read again.

## Not done yet

H.264 and VP8 on NVDEC, NVJPG, VIC input formats other than NV12, scaling,
deinterlacing and blending between slots, segmentation and backward adaptation
in VP9 streams, and showing an existing (hidden) frame through VIC.
