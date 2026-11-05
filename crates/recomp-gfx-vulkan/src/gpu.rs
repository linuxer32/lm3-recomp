//! Drawing the guest's draws with Vulkan.
//!
//! Everything a submission does is recorded into one command buffer, which
//! the renderer submits and waits for when the submission finishes:
//!
//! - Render targets are images keyed by guest address, size and format and
//!   kept in the `GENERAL` layout, so they can be rendered to, sampled
//!   (render-to-texture is how the guest's post-processing chain works) and
//!   blitted to the window. A global barrier precedes every draw.
//! - Shaders are translated by `recomp_gfx::maxwell::translate` once per
//!   program; pipelines are built per shader pair and fixed state.
//! - Vertex ranges are kept in a vertex cache in video memory across
//!   submissions and uploaded again only when their bytes change. Index,
//!   constant buffer and changing vertex data is copied from guest memory into
//!   a host-visible stream buffer, reset each submission; descriptor sets
//!   come from a pool reset each submission. Robust buffer access makes reads
//!   past a constant buffer's size return zero.
//! - Textures come from the guest's texture header pool: a header whose
//!   address is a render target samples that image; others are untiled from
//!   guest memory and uploaded once. Formats not decoded yet (ASTC) sample a
//!   magenta placeholder.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::Path;

use ash::{vk, Device, Instance};
// The worker looks keys up several times per draw (shaders, pipelines,
// textures, vertex ranges): std's SipHash was a twentieth of its time.
use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};

use recomp_gfx::maxwell::compute::ComputeDispatch;
use recomp_gfx::maxwell::render::{
    ClearColorCall, ClearDepthStencilCall, DrawCall, GpuMemory, PrimitiveTopology, VertexAttribute,
};
use recomp_gfx::maxwell::state::{ColorTargetState, PoolState, StencilFace};
use recomp_gfx::maxwell::translate::{
    self, AttributeType, Environment, TextureBinding, TextureKind, TranslatedShader, FRAGMENT_SET,
    VERTEX_SET,
};
use recomp_gfx::maxwell::twod::Blit;

use crate::bcn::Bcn;
use crate::decode::{Decode, Decoder, Job, Mip, Prepared};
use crate::formats;
use crate::shader_pack::ShaderPack;
use crate::texture::{self, Layout, SamplerHeader, Target, TextureHeader};
use recomp_gfx::maxwell::translate::{RESCALE_BYTES, TEXTURE_BINDING_BASE};

/// Bytes of guest data (vertices, indices, constants, texels) one
/// submission may copy.
const STREAM_BYTES: u64 = 128 << 20;
/// Submissions that may be in flight at once, each recorded into a slot of
/// its own. LM3 sends two a frame (its scene, then the present): with two
/// slots the next scene could not be recorded until the GPU had run the one
/// before, so the worker and the GPU took turns (11 fps at the title on an
/// Adreno 650, the worker waiting two thirds of the time).
pub const SLOTS: usize = 3;
/// Bytes of prepared textures one submission uploads at most.
const UPLOAD_BUDGET: usize = 32 << 20;
/// How long draws of one submission wait in all for textures they sample
/// to be prepared, before sampling placeholders instead.
const TEXTURE_WAIT: std::time::Duration = std::time::Duration::from_millis(12);
/// What draws sample in place of a texture still being prepared.
const PENDING_COLOR: [u8; 4] = [0, 0, 0, 0];
/// What draws sample in place of a texture in a format not decoded yet.
const UNSUPPORTED_COLOR: [u8; 4] = [255, 0, 255, 255];
/// Bytes of the render target readback buffer at first (a 2048x2048 RGBA8
/// target); it grows for larger ones (targets at a resolution scale).
const READBACK_BYTES: u64 = 16 << 20;
/// Bytes of a program read for translation; it follows control flow, so a
/// shorter program reads past its end harmlessly.
const PROGRAM_BYTES: u64 = 0x10000;
const UNIFORM_BYTES: u32 = 0x10000;
const UNIFORM_BINDINGS: u32 = 18;
/// Texture bindings per stage. The translator gives each texture a program
/// samples its own binding, and a bindless instruction one per entry of its
/// handle array: LM3's title scene has a program that samples 17.
const TEXTURE_BINDINGS: u32 = 32;

/// Whether a shader texture binding is one the descriptor set layout declares.
/// A pipeline whose shader names a binding outside it is invalid: AMDVLK
/// dereferences the missing layout entry and crashes in
/// `vkCreateGraphicsPipelines`, so such draws are skipped before that call.
fn texture_binding_fits(binding: u32) -> bool {
    (TEXTURE_BINDING_BASE..TEXTURE_BINDING_BASE + TEXTURE_BINDINGS).contains(&binding)
}

/// Whether a shader's texture kind and a header's target can share a view.
///
/// A one-row lookup table can be sampled through a 2D instruction, and a
/// 1D-array header through a 2D-array one: both are uploaded as the flattened
/// view the shader names (see `prepare_texture`). Rejecting the header before
/// that conversion would sample the magenta placeholder instead; LM3's
/// PearlescentLUT is a one-row table of this shape.
fn texture_target_supported(kind: TextureKind, target: Option<Target>) -> bool {
    matches!(
        (kind, target),
        (TextureKind::D1, Some(Target::D1 | Target::D1Array))
            | (TextureKind::D2, Some(Target::D1 | Target::D1Array))
            | (TextureKind::D2, Some(Target::D2 | Target::D2NoMipmap))
            | (TextureKind::D2Array, Some(Target::D1Array))
            | (TextureKind::D2Array, Some(Target::D2Array))
            | (
                TextureKind::Cube,
                Some(Target::Cube | Target::D2 | Target::D2NoMipmap | Target::D2Array)
            )
            | (
                TextureKind::D3,
                Some(Target::D2 | Target::D2NoMipmap | Target::D3)
            )
    )
}

/// The address a window capture's readback names (no target is there).
const WINDOW_CAPTURE: u64 = u64::MAX;

/// Descriptor sets one submission may allocate (a submission is sent early
/// past 12000).
const DESCRIPTOR_SETS: u32 = 16384;
/// Descriptor sets allocated from the pool at once. Allocating per draw cost
/// a driver call and its lock; a draw takes two sets (vertex and fragment).
const DESCRIPTOR_BATCH: u32 = 256;
/// Descriptor writes of the two graphics stages: every constant buffer and
/// texture binding of each set.
const MAX_BUFFER_WRITES: usize = 2 * UNIFORM_BINDINGS as usize;
const MAX_IMAGE_WRITES: usize = 2 * TEXTURE_BINDINGS as usize;
/// Vertex buffer bytes one draw copies at most.
const VERTEX_BYTES: u64 = 32 << 20;
/// Bytes of the vertex cache; a submission that starts with it three
/// quarters full empties it.
const VERTEX_CACHE_BYTES: u64 = 512 << 20;
/// The vertex buffer bindings a draw binds: the guest's 16, and the zero
/// buffer constant attributes read (which takes a free slot, below).
const MAX_VERTEX_BINDINGS: usize = 17;

fn error(context: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |result| format!("{context}: {result}")
}

/// A host-visible buffer data is appended to, reset per submission.
struct Stream {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: u64,
    cursor: u64,
}

// The mapping is only touched by the thread that owns the renderer.
unsafe impl Send for Stream {}

impl Stream {
    /// Copy `bytes` in at an offset aligned to `align`; `None` when full.
    fn push(&mut self, bytes: &[u8], align: u64) -> Option<u64> {
        let offset = self.cursor.next_multiple_of(align.max(1));
        let end = offset.checked_add(bytes.len() as u64)?;
        if end > self.size {
            return None;
        }
        // SAFETY: the mapping covers `size` bytes and `end` is within it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.mapped.add(offset as usize),
                bytes.len(),
            );
        }
        self.cursor = end;
        Some(offset)
    }

    /// A constant buffer's bytes (GPU address and size) copied in from
    /// guest memory for a uniform buffer descriptor, padded to 16 bytes;
    /// zeros when none is bound or it cannot be read.
    fn push_uniform(
        &mut self,
        memory: &dyn GpuMemory,
        buffer: Option<(u64, u32)>,
        align: u64,
    ) -> Result<vk::DescriptorBufferInfo, String> {
        let size = buffer.map_or(16, |(_, size)| size.clamp(16, UNIFORM_BYTES) as usize);
        let range = size.next_multiple_of(16);
        let offset = self.reserve(range as u64, align).ok_or("stream full")?;
        // SAFETY: `reserve` handed out this range, within the mapping, and
        // nothing else refers to it yet.
        let target =
            unsafe { std::slice::from_raw_parts_mut(self.mapped.add(offset as usize), range) };
        let read = buffer.is_some_and(|(address, _)| {
            memory.read_gpu_with(address, size, &mut |bytes| {
                target[..size].copy_from_slice(bytes)
            })
        });
        target[if read { size } else { 0 }..].fill(0);
        Ok(vk::DescriptorBufferInfo {
            buffer: self.buffer,
            offset,
            range: range as u64,
        })
    }

    /// Room for `len` bytes the GPU writes, at an offset aligned to `align`.
    fn reserve(&mut self, len: u64, align: u64) -> Option<u64> {
        let offset = self.cursor.next_multiple_of(align.max(1));
        let end = offset.checked_add(len)?;
        if end > self.size {
            return None;
        }
        self.cursor = end;
        Some(offset)
    }

    /// Bytes at `offset`, after the GPU work writing them has finished.
    fn read(&self, offset: u64, len: u64) -> &[u8] {
        // SAFETY: `reserve` handed out this range, within the mapping.
        unsafe { std::slice::from_raw_parts(self.mapped.add(offset as usize), len as usize) }
    }
}

/// Guest vertex ranges kept in video memory across submissions. Most of a
/// scene's meshes are the same every frame, and copying them again was most
/// of the worker's time in LM3's rooms (66 MB a frame, 14 ms). A range is
/// checked once per submission that draws it and uploaded again only when its
/// bytes changed, at a new offset (the submission before may still read the
/// old one). A range that keeps changing (per-frame vertex data) goes to the
/// stream instead.
///
/// Each guest page of a range has a hash. Where the runtime watches guest
/// writes (`GpuMemory::written_since`), only the pages of a range of 16 KiB or
/// more written since the last check are read and hashed again, and a range
/// with none is not read at all:
/// hashing every range took 3 of the worker's 19 ms a frame in LM3's lobby
/// (55 MB). Elsewhere every page is hashed each time.
struct VertexCache {
    arena: Stream,
    /// Per (address, length): what it held and where.
    entries: HashMap<(u64, u64), CachedVertices>,
}

struct CachedVertices {
    /// XXH3 of each guest page the range touches (the first and last
    /// clipped to the range).
    pages: Box<[u64]>,
    offset: u64,
    /// Times its bytes were found changed.
    changes: u32,
    /// The write check its bytes are known at (`GpuMemory::written_since`);
    /// `None` when writes to it are not watched.
    checked: Option<u64>,
}

/// Guest page size: the unit writes are watched in.
const GUEST_PAGE: u64 = 0x1000;

/// Vertex ranges shorter than this are hashed whole each time instead of
/// watched: hashing a few KB costs about what asking the kernel does. In
/// LM3's lobby that leaves 760 of the 1250 ranges a frame (3 of 58 MB).
const WATCHED_BYTES: u64 = 16 << 10;

/// Changes after which a vertex range is streamed instead of cached.
const VERTEX_CHANGES: u32 = 2;

/// An image with its memory and one view.
#[derive(Clone, Copy)]
struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// The view shaders sample (a depth image's depth aspect).
    view: vk::ImageView,
    /// The view rendering attaches (all of a depth-stencil image's aspects).
    attachment: vk::ImageView,
    format: vk::Format,
    width: u32,
    height: u32,
    /// The guest's pixels times `scale / 2`: 2 for the guest's own size,
    /// more for render targets at a resolution scale.
    scale: u32,
}

/// `value` guest pixels at `scale` (in halves: 2 is the guest's size).
fn scaled(value: u32, scale: u32) -> u32 {
    (value * scale).div_ceil(2)
}

/// `rect` (in guest pixels) at `scale`.
fn scaled_rect(rect: vk::Rect2D, scale: u32) -> vk::Rect2D {
    let (x, y) = (rect.offset.x.max(0) as u32, rect.offset.y.max(0) as u32);
    let (left, top) = (x * scale / 2, y * scale / 2);
    vk::Rect2D {
        offset: vk::Offset2D {
            x: left as i32,
            y: top as i32,
        },
        extent: vk::Extent2D {
            width: scaled(x + rect.extent.width, scale) - left,
            height: scaled(y + rect.extent.height, scale) - top,
        },
    }
}

/// The push constants of `translate::RESCALE_*`: the scale a draw renders at,
/// the one render targets are sampled at (both in halves), and the texture
/// bindings that sample such targets for vertex programs and the others.
fn rescale_constants(draw_scale: u32, target_scale: u32, textures: [u32; 2]) -> [u8; 16] {
    let words = [
        (draw_scale as f32 / 2.0).to_bits(),
        (target_scale as f32 / 2.0).to_bits(),
        textures[0],
        textures[1],
    ];
    let mut bytes = [0; 16];
    for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *chunk = word.to_le_bytes();
    }
    bytes
}

/// Render targets smaller than this keep the guest's size at any resolution
/// scale: probes, lookup tables and the small steps of blurs, which are data
/// more than pictures.
const SCALED_MIN: (u32, u32) = (256, 128);

/// A guest texture in the cache.
#[derive(Clone, Copy)]
enum TextureState {
    /// Being prepared off the worker; created as described when it arrives.
    Pending(PendingTexture),
    /// Its small levels arrived first (see `decode.rs`); sampled until the
    /// whole texture does.
    Preview(Image, PendingTexture),
    Ready(Image),
    /// In a format not decoded yet, or not readable.
    Unsupported,
}

/// How to make a texture's image once it is prepared.
#[derive(Clone, Copy)]
struct PendingTexture {
    format: vk::Format,
    components: vk::ComponentMapping,
    width: u32,
    height: u32,
    layers: u32,
    view_type: vk::ImageViewType,
    /// Mip levels prepared, and the first and count of those the header's
    /// view uses.
    levels: u32,
    view: (u32, u32),
    /// Which preparation this is (`Gpu::texture_generation`).
    generation: u64,
}

/// A cached texture's bytes in guest memory, watched for writes
/// (`GpuMemory::written_since`). LM3 grades each frame through one of nine
/// 3D color lookup tables it rewrites in turn as a room's look fades in (on
/// entering a room the first time), and rebuilds its light grid every
/// frame: kept as first read, the tables graded every frame through stale
/// colors, some much darker, and the room flickered for two seconds.
struct TextureWatch {
    address: u64,
    len: u64,
    /// The write check its bytes are known at.
    checked: u64,
    /// Whether `checked` is a real write-watch token (otherwise the source
    /// hash is compared every submission).
    check: bool,
    /// A hash of the source bytes: a page can be reported written while the
    /// texture contents are unchanged, and re-decoding it every frame makes
    /// a door flicker. Refresh only when this actually differs.
    hash: Option<u64>,
    /// The submission it was last checked in.
    submission: u64,
}

/// Textures with a preview: at least this many texels at level 0, a full
/// view and this many levels; the preview starts at the first level no more
/// than `PREVIEW_SIZE` texels on a side.
const PREVIEW_TEXELS: u32 = 256 * 256;
const PREVIEW_LEVELS: u32 = 5;
const PREVIEW_SIZE: u32 = 128;

/// The level a texture's preview starts at (see `decode.rs`): large textures
/// with a mip chain and a view of all of it show their small levels first.
fn preview_level(
    width: u32,
    height: u32,
    depth: u32,
    levels: u32,
    view: (u32, u32),
) -> Option<u32> {
    if width * height < PREVIEW_TEXELS
        || depth != 1
        || levels < PREVIEW_LEVELS
        || view != (0, levels)
    {
        return None;
    }
    (1..levels).find(|level| (width.max(height) >> level) <= PREVIEW_SIZE)
}

/// A render target copy in flight: see [`Gpu::record_readback`].
pub struct Readback {
    key: TargetKey,
    staging: Image,
    offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TargetKey {
    address: u64,
    width: u32,
    height: u32,
    format: u32,
}

/// A draw's guest indices as placed once per submission: start, length,
/// index size, whether they are quads, the guest's restart index (while
/// restarting is on) and whether the draw's strips restart.
type IndexKey = (u64, u64, u64, bool, Option<u32>, bool);

/// A texture array whose layers the guest drew as targets of their own,
/// one at each layer's address (LM3 draws its lights' shadow maps into a
/// pool of depth layers and samples the pool as an array).
struct Layered {
    /// None while no layer was drawn.
    image: Option<Image>,
    /// `target_generation` when `sources` was found.
    generation: u64,
    /// The layers drawn, and their addresses.
    sources: Vec<(u32, u64)>,
    /// Per source, its image and write count when last copied in.
    copied: Vec<(vk::Image, u64)>,
}

struct Shader {
    module: vk::ShaderModule,
    translated: TranslatedShader,
}

/// A program as cached: its address, the attribute types it was translated
/// for, and a hash of its first bytes (the guest reuses shader memory).
type ShaderKey = (u64, [u8; 41], u64);

/// The key of the empty fragment program (`translate::empty_fragment_program`)
/// draws without a fragment program run: no guest program is at address 0.
const EMPTY_FRAGMENT: ShaderKey = (0, [0; 41], 0);

/// A compute program by address, a hash of its first bytes, block size and
/// shared memory size.
type ComputeKey = (u64, u64, [u32; 3], u32);

struct ComputeProgram {
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    translated: TranslatedShader,
}

/// Where a stage's texture handles and headers come from.
struct TextureSource {
    /// GPU address of the constant buffer holding the handles.
    handles: Option<u64>,
    texture_pool: PoolState,
    sampler_pool: PoolState,
    samplers_from_texture_index: bool,
}

/// A program's constant buffer 1 (address and size), read for translation;
/// empty when unbound or unmapped.
fn constant_buffer_1(memory: &dyn GpuMemory, binding: Option<(u64, u32)>) -> Vec<u8> {
    binding
        .and_then(|(address, size)| memory.read_gpu(address, size.min(0x1_0000) as usize))
        .unwrap_or_default()
}

/// The lowest and highest vertex 16- or 32-bit indices name; with
/// `restart`, the restart value (all ones) names none, and without it is a
/// vertex like any other. Two plain min/max reductions, which vectorize
/// (six times faster than one filtered pass): all ones never lowers the
/// minimum, and wraps to zero in the maximum of index + 1.
fn index_bounds(bytes: &[u8], size: u64, restart: bool) -> (u32, u32) {
    if !restart {
        // Two reductions directly over the guest bytes: collecting into a
        // Vec first allocated on every uncached draw and showed up as a
        // quarter of the GPU worker's index work.
        if size == 2 {
            let values = bytes.as_chunks::<2>().0;
            let low = values
                .iter()
                .map(|value| u32::from(u16::from_le_bytes(*value)))
                .min();
            let high = values
                .iter()
                .map(|value| u32::from(u16::from_le_bytes(*value)))
                .max();
            return match (low, high) {
                (Some(low), Some(high)) => (low, high),
                _ => (0, 0),
            };
        }
        let values = bytes.as_chunks::<4>().0;
        let low = values.iter().map(|value| u32::from_le_bytes(*value)).min();
        let high = values.iter().map(|value| u32::from_le_bytes(*value)).max();
        return match (low, high) {
            (Some(low), Some(high)) => (low, high),
            _ => (0, 0),
        };
    }
    let (low, high) = if size == 2 {
        let values = bytes.as_chunks::<2>().0;
        let index = |value: &[u8; 2]| u16::from_le_bytes(*value);
        (
            u32::from(values.iter().map(index).fold(u16::MAX, u16::min)),
            u32::from(
                values
                    .iter()
                    .map(|value| index(value).wrapping_add(1))
                    .fold(0, u16::max),
            ),
        )
    } else {
        let values = bytes.as_chunks::<4>().0;
        let index = |value: &[u8; 4]| u32::from_le_bytes(*value);
        (
            values.iter().map(index).fold(u32::MAX, u32::min),
            values
                .iter()
                .map(|value| index(value).wrapping_add(1))
                .fold(0, u32::max),
        )
    };
    match high {
        0 => (0, 0),
        high => (low.min(high - 1), high - 1),
    }
}

/// A texture or sampler pool entry (32 bytes), read in place.
fn pool_entry(memory: &dyn GpuMemory, address: u64) -> Option<[u8; 32]> {
    let mut entry = None;
    memory.read_gpu_with(address, 32, &mut |bytes| entry = bytes.try_into().ok());
    entry
}

/// FNV-1a over a program's first bytes: the guest reuses shader memory.
fn code_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

/// A bounded source hash for hosts without a page write watch. Small
/// textures are read whole; large ones are sampled in five 64 KiB windows so
/// a rewritable decal is caught without reading tens of MB per submission.
/// Hash only the first page of a texture. Windows GetWriteWatch commonly
/// reports page 0 because the texture header/pool shares it; a real change
/// there still differs, while a later dirty page refreshes without hashing.
fn hash_gpu(memory: &dyn GpuMemory, address: u64, len: u64) -> Option<u64> {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut read = false;
    if !memory.read_gpu_with(address, len as usize, &mut |bytes| {
        read = true;
        hash = code_hash(bytes);
    }) {
        return None;
    }
    read.then_some(hash)
}

fn texture_hash(memory: &dyn GpuMemory, address: u64, len: u64) -> Option<u64> {
    const WINDOW: u64 = 64 << 10;
    if len <= 512 << 10 {
        return hash_gpu(memory, address, len);
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let starts = [
        0,
        len / 4,
        len / 2,
        len.saturating_mul(3) / 4,
        len.saturating_sub(WINDOW),
    ];
    for start in starts {
        let mut read = false;
        if !memory.read_gpu_with(address + start, WINDOW as usize, &mut |bytes| {
            read = true;
            hash = bytes.iter().fold(hash, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
            });
        }) {
            return None;
        }
        if !read {
            return None;
        }
    }
    Some(hash)
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct RenderPassKey {
    colors: Vec<vk::Format>,
    depth: Option<vk::Format>,
    stencil: bool,
    /// The stencil test is on, so the pass loads and stores the stencil
    /// aspect. Otherwise it is DONT_CARE: Adreno has to restore a tiled
    /// stencil surface for LOAD even when nothing uses it, which is the
    /// kind of thing its tile path deadlocks on.
    stencil_ops: bool,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct FramebufferKey {
    pass: vk::RenderPass,
    views: Vec<vk::ImageView>,
    width: u32,
    height: u32,
}

/// Render passes and framebuffers for Vulkan 1.1/1.2 devices without dynamic
/// rendering. The draw path keeps every target in `GENERAL`, so one pass per
/// attachment format set with LOAD/STORE is enough; pipelines use the same
/// pass for compatibility.
struct RenderPassCache {
    device: Device,
    passes: RefCell<HashMap<RenderPassKey, vk::RenderPass>>,
    framebuffers: RefCell<HashMap<FramebufferKey, vk::Framebuffer>>,
}

impl RenderPassCache {
    /// Destroy every framebuffer that references one of `views`. A
    /// framebuffer holds the attachment views it was created with, so
    /// destroying a view a live framebuffer still names is a use-after-free
    /// — and on Adreno it faults in the render pass setup, which is how the
    /// phone lost the device at the title screen (images are reallocated
    /// there and the cache kept the old handles).
    fn forget_views(&self, views: &[vk::ImageView]) {
        let mut framebuffers = self.framebuffers.borrow_mut();
        let stale: Vec<FramebufferKey> = framebuffers
            .keys()
            .filter(|key| key.views.iter().any(|view| views.contains(view)))
            .cloned()
            .collect();
        for key in stale {
            if let Some(framebuffer) = framebuffers.remove(&key) {
                // SAFETY: the caller has waited for the GPU (`begin` retires
                // images only after its previous submission finished).
                unsafe { self.device.destroy_framebuffer(framebuffer, None) };
            }
        }
    }

    /// Forget every framebuffer (the images they name are all going away).
    fn forget_all(&self) {
        for (_, framebuffer) in self.framebuffers.borrow_mut().drain() {
            // SAFETY: as above.
            unsafe { self.device.destroy_framebuffer(framebuffer, None) };
        }
    }

    fn new(device: Device) -> Self {
        Self {
            device,
            passes: RefCell::new(HashMap::new()),
            framebuffers: RefCell::new(HashMap::new()),
        }
    }

    fn pass(
        &self,
        colors: &[vk::Format],
        depth: Option<vk::Format>,
        stencil_ops: bool,
    ) -> vk::RenderPass {
        let key = RenderPassKey {
            colors: colors.to_vec(),
            depth,
            stencil: depth.is_some_and(crate::formats::has_stencil),
            stencil_ops: stencil_ops && depth.is_some_and(crate::formats::has_stencil),
        };
        let existing = self.passes.borrow().get(&key).copied();
        if let Some(pass) = existing {
            return pass;
        }
        let mut attachments = Vec::new();
        for &format in colors {
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::GENERAL)
                    .final_layout(vk::ImageLayout::GENERAL),
            );
        }
        if let Some(format) = depth {
            let stencil = key.stencil_ops;
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(if stencil {
                        vk::AttachmentLoadOp::LOAD
                    } else {
                        vk::AttachmentLoadOp::DONT_CARE
                    })
                    .stencil_store_op(if stencil {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
                    .initial_layout(vk::ImageLayout::GENERAL)
                    .final_layout(vk::ImageLayout::GENERAL),
            );
        }
        let color_refs: Vec<_> = colors
            .iter()
            .enumerate()
            .map(|(index, _)| vk::AttachmentReference {
                attachment: index as u32,
                layout: vk::ImageLayout::GENERAL,
            })
            .collect();
        let depth_ref = depth.map(|_| vk::AttachmentReference {
            attachment: colors.len() as u32,
            layout: vk::ImageLayout::GENERAL,
        });
        let mut subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs);
        if let Some(reference) = depth_ref.as_ref() {
            subpass = subpass.depth_stencil_attachment(reference);
        }
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));
        // SAFETY: the description is fully populated and the device is live.
        let pass =
            unsafe { self.device.create_render_pass(&info, None) }.expect("render pass creation");
        self.passes.borrow_mut().insert(key, pass);
        pass
    }

    fn begin(
        &self,
        command: vk::CommandBuffer,
        rendering: &vk::RenderingInfo<'_>,
        colors: &[vk::Format],
        depth: Option<vk::Format>,
        stencil_ops: bool,
    ) {
        let pass = self.pass(colors, depth, stencil_ops);
        let attachments = unsafe {
            std::slice::from_raw_parts(
                rendering.p_color_attachments,
                rendering.color_attachment_count as usize,
            )
        };
        let mut views: Vec<vk::ImageView> = attachments.iter().map(|a| a.image_view).collect();
        if !rendering.p_depth_attachment.is_null() {
            views.push(unsafe { (*rendering.p_depth_attachment).image_view });
        }
        let key = FramebufferKey {
            pass,
            views: views.clone(),
            width: rendering.render_area.extent.width,
            height: rendering.render_area.extent.height,
        };
        let existing = self.framebuffers.borrow().get(&key).copied();
        let framebuffer = if let Some(framebuffer) = existing {
            framebuffer
        } else {
            let info = vk::FramebufferCreateInfo::default()
                .render_pass(pass)
                .attachments(&views)
                .width(key.width)
                .height(key.height)
                .layers(rendering.layer_count.max(1));
            // SAFETY: as above.
            let framebuffer = unsafe { self.device.create_framebuffer(&info, None) }
                .expect("framebuffer creation");
            self.framebuffers.borrow_mut().insert(key, framebuffer);
            framebuffer
        };
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(pass)
            .framebuffer(framebuffer)
            .render_area(rendering.render_area);
        // SAFETY: the pass and framebuffer match the attachments; the
        // command buffer is recording.
        unsafe {
            self.device
                .cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
        }
    }

    fn end(&self, command: vk::CommandBuffer) {
        // SAFETY: the caller began a render pass on this command buffer.
        unsafe { self.device.cmd_end_render_pass(command) };
    }
}

/// `RECOMP_VULKAN_DUMP_SHADER=<address>`: write that program's SPIR-V to
/// `shader-dump/`, for a shader that hangs the GPU rather than failing a
/// pipeline.
fn shader_dump_address() -> Option<u64> {
    static ADDRESS: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *ADDRESS.get_or_init(|| {
        std::env::var("RECOMP_VULKAN_DUMP_SHADER")
            .ok()
            .and_then(|value| u64::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok())
    })
}

/// How many recent draws and operations to keep for a lost device's
/// report: a frame's worth and more in `RECOMP_VULKAN_FLUSH_DRAWS` mode.
fn recent_capacity() -> usize {
    static CAPACITY: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAPACITY.get_or_init(|| {
        if std::env::var("RECOMP_VULKAN_FLUSH_DRAWS").is_ok_and(|value| value != "0") {
            16_384
        } else {
            64
        }
    })
}

/// The optimal-tiling feature flags of `format`, asked once per format.
fn format_features(
    cache: &std::cell::RefCell<HashMap<vk::Format, vk::FormatFeatureFlags>>,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
) -> vk::FormatFeatureFlags {
    if let Some(features) = cache.borrow().get(&format) {
        return *features;
    }
    // SAFETY: the instance and physical device are live; the query has no
    // other preconditions.
    let features = unsafe {
        instance
            .get_physical_device_format_properties(physical_device, format)
            .optimal_tiling_features
    };
    cache.borrow_mut().insert(format, features);
    features
}

pub struct Gpu {
    device: Device,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    /// Per-format feature flags, asked once: a driver may support a format
    /// as a color attachment but not blending into it, and a pipeline that
    /// blends anyway is invalid (Adreno answers VK_ERROR_UNKNOWN, and the
    /// draw can hang the GPU).
    format_features: std::cell::RefCell<HashMap<vk::Format, vk::FormatFeatureFlags>>,
    /// `None` for Vulkan 1.3 dynamic rendering; `Some` for the 1.1/1.2
    /// render-pass fallback.
    render_passes: Option<RenderPassCache>,
    /// `VK_EXT_depth_clip_control` is enabled: clip space depth can be
    /// -1..1, as the guest's default depth mode is.
    depth_clip_control: bool,
    /// The device enabled the `depthClamp` feature.
    depth_clamp: bool,
    /// The device enabled the `depthBiasClamp` feature.
    depth_bias_clamp: bool,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    uniform_alignment: u64,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    /// Compute pipelines: one set, laid out like a graphics stage's.
    compute_layout: vk::PipelineLayout,
    compute_programs: HashMap<ComputeKey, Option<ComputeProgram>>,
    descriptor_pool: vk::DescriptorPool,
    /// Sets allocated from `descriptor_pool` but not used yet. The pool is
    /// reset per submission; allocate a batch at a time instead of two
    /// driver calls per draw.
    descriptor_batch: Vec<vk::DescriptorSet>,
    stream: Stream,
    /// `None` when its memory could not be allocated.
    vertex_cache: Option<VertexCache>,
    /// The draws recorded most recently, in order. Adreno reports a device
    /// loss with no other detail; the last draws before it are the evidence
    /// of what triggered it.
    recent_draws: VecDeque<String>,
    /// Scratch color attachments for the unused slots of a sparse draw: a
    /// Vulkan 1.1 render pass cannot name a null view, and dropping the draw
    /// loses whatever it was drawing (LM3's UI composite writes only some
    /// targets). Keyed by render-area size.
    scratch: HashMap<(u32, u32), Image>,
    /// Images a render pass or clear wrote since the last submission. Adreno
    /// 650 deadlocks when a texture unit reads a tiled attachment the same
    /// command buffer rendered into (even with a barrier between, and even
    /// though the layout stays GENERAL), so `sampled_view` copies one of
    /// these instead of sampling it live.
    written_since_submit: std::cell::RefCell<std::collections::HashSet<vk::Image>>,
    /// Cached host memory render targets are read back into (reads from the
    /// write-combined stream are slow), created on first use.
    readback: Option<Stream>,
    /// The resolution scale large render targets are made at, in halves (2:
    /// the guest's resolution); the menu's choice
    /// (`overlay::Controls::resolution`) takes effect as a submission
    /// begins.
    scale: u32,
    shaders: HashMap<ShaderKey, Option<Shader>>,
    /// The program each address and attribute type set found in this
    /// submission (`None`: not translated), so its code is read and hashed
    /// once per submission instead of twice per draw.
    shader_keys: HashMap<(u64, [u8; 41]), Option<ShaderKey>>,
    /// Portable SPIR-V translations extracted from the game's shader library;
    /// a hit skips Maxwell-to-SPIR-V translation.
    shader_pack: Option<ShaderPack>,
    pipelines: HashMap<Vec<u64>, Option<vk::Pipeline>>,
    /// Driver-compiled shaders and pipelines, loaded from disk at startup and
    /// saved as new pipelines appear: a later run does not recompile a scene
    /// it has already visited.
    pipeline_cache: vk::PipelineCache,
    pipeline_cache_path: Option<std::path::PathBuf>,
    /// New pipelines since the last background cache save.
    pipelines_since_cache_save: u32,
    /// Counter for the temporary files the background saves use.
    pipeline_cache_saves: u64,
    /// The pipeline key of the draw being recorded, and the last key looked
    /// up: most draws repeat the previous pipeline, which then needs neither
    /// an allocation nor a hash lookup.
    pipeline_key: Vec<u64>,
    pipeline_last: Option<(Vec<u64>, Option<vk::Pipeline>)>,
    /// Graphics state bound in the command buffer, so a draw does not bind
    /// what the draw before it already did.
    bound: BoundState,
    targets: HashMap<TargetKey, Image>,
    target_at: HashMap<u64, TargetKey>,
    /// Depth-stencil targets (keyed with the zeta format), and by address.
    depths: HashMap<TargetKey, Image>,
    depth_at: HashMap<u64, TargetKey>,
    /// One image per target address and format, as large as the largest
    /// size rendered there: the guest's dynamic resolution renders frames
    /// of a new size into the same memory every frame, and temporal effects
    /// read the previous frame from it. `targets` and `depths` map every
    /// size's key to it.
    target_images: HashMap<(u64, u32), Image>,
    depth_images: HashMap<(u64, u32), Image>,
    /// Per target image, how often it has been a destination: copies made
    /// for sampling are current while this has not changed.
    writes: HashMap<vk::Image, u64>,
    /// Per target image, its copy at the size a texture header last sampled
    /// it, and the image's write count when it was made.
    resized: HashMap<vk::Image, (Image, u64)>,
    /// Arrays sampled from layers drawn as targets, by address, size and
    /// layer count.
    layered: HashMap<(u64, u32, u32, u32), Layered>,
    /// Counts the addresses targets were first drawn at, so `layered`
    /// arrays know to look for their layers again.
    target_generation: u64,
    /// Images replaced while recording: the submission may still use them,
    /// so they go once it has run (when this slot begins again).
    retired: Vec<Image>,
    /// The current slot's fence, and whether its last submission may still
    /// be running (see `Slot`).
    pub fence: vk::Fence,
    in_flight: bool,
    /// The other slots, the one submitted longest ago first, and which slot
    /// is current (0 to `SLOTS - 1`).
    spares: VecDeque<Slot>,
    slot: usize,
    /// Target images sampled since the last barrier: writing one from a new
    /// rendering instance must wait for those reads.
    read: std::cell::RefCell<HashSet<vk::Image>>,
    textures: HashMap<[u32; 8], TextureState>,
    /// Where cached textures' bytes are, to notice the guest writing them
    /// again.
    texture_watch: HashMap<[u32; 8], TextureWatch>,
    /// Numbers each preparation of a texture: a texture written again is
    /// prepared again, and a result of the preparation before is dropped.
    texture_generation: u64,
    /// Submissions begun: textures are checked for writes once in each.
    submission: u64,
    decoder: Decoder,
    /// Prepared textures waiting for upload room.
    prepared: VecDeque<Prepared>,
    /// Time draws of this submission waited for textures being prepared.
    texture_waited: std::time::Duration,
    /// Per render target, the extent draws covered (their viewports): the
    /// guest renders 1280x720 frames into 1920x1080 swapchain images.
    drawn: HashMap<TargetKey, (u32, u32)>,
    /// Samplers by header, comparing, and filtering: nearest-only ones for
    /// views in `unfilterable`.
    samplers: HashMap<([u32; 8], bool, bool), vk::Sampler>,
    /// Views whose format the device cannot filter linearly (Adreno: 32-bit
    /// floats and depth). Sampling one through a linear sampler is invalid
    /// unless the sampler compares, so those get a nearest sampler.
    unfilterable: std::cell::RefCell<HashSet<vk::ImageView>>,
    placeholders: HashMap<(TextureKind, [u8; 4]), Image>,
    pub command: vk::CommandBuffer,
    pub recording: bool,
    /// The display image this submission drew last: what the window shows.
    pub presentable: Option<TargetKey>,
    /// The area of the largest target drawn so far.
    display_area: u64,
    warned: HashSet<String>,
    pub draws: u64,
    pub skipped: u64,
    pub dispatches: u64,
    /// Descriptor sets allocated in this submission (the pool is reset at
    /// its start).
    descriptor_sets: u32,
    /// One line per draw and clear, while tracing a submission.
    pub trace: Option<Vec<String>>,
    /// The textures the draw being traced samples.
    traced_textures: Vec<String>,
    /// The least anisotropic filtering samplers get
    /// (`RECOMP_ANISOTROPY`, else 1: what the guest asks for), and the
    /// device's most (1 without the feature).
    anisotropy: f32,
    max_anisotropy: f32,
    /// Time draws spent in each phase of recording (`RECOMP_GPU_STATS`).
    phases: [std::time::Duration; 4],
    /// Time `begin` waited for a slot's last submission to run.
    gpu_waited: std::time::Duration,
    /// Vertex and index bytes draws copied since the last report, and vertex
    /// bytes the vertex cache held already.
    copied: [u64; 3],
    /// Guest vertex ranges (address, length) and index ranges (address,
    /// length, index size, quads) this submission has placed: vertices in
    /// the vertex cache or the stream, indices in the stream.
    vertex_uploads: HashMap<(u64, u64), (vk::Buffer, u64)>,
    index_uploads: HashMap<IndexKey, StreamIndices>,
    /// Reused while converting and widening a draw's indices, so uncached
    /// and inline draws do not allocate.
    index_bytes: Vec<u8>,
    index_values: Vec<u32>,
    index_data: Vec<u8>,
    index_triangles: Vec<u32>,
    /// The rendering instance draws are adding to, while one is open.
    rendering: std::cell::Cell<Option<RenderingKey>>,
    /// Images written since the last barrier (attachments and clears).
    written: std::cell::RefCell<HashSet<vk::Image>>,
    /// Clears not recorded yet, in order (see `PendingClear`).
    pending_clears: std::cell::RefCell<Vec<PendingClear>>,
    /// Rendering instances begun, barriers, and clears recorded inside and
    /// outside instances, since the last report (`RECOMP_GPU_STATS`).
    pass_counts: std::cell::Cell<[u64; 4]>,
    /// Target images the draw being recorded samples in place.
    sampled: Vec<vk::Image>,
}

/// What a rendering instance draws into: draws with the same attachments
/// and area add to an open one.
#[derive(Clone, Copy, PartialEq, Eq)]
struct RenderingKey {
    colors: [vk::ImageView; 8],
    depth: vk::ImageView,
    width: u32,
    height: u32,
}

impl RenderingKey {
    /// Which attachment `image` is: `Some(color slot)`, `None` for depth.
    fn attachment(&self, image: &Image) -> Option<Option<u32>> {
        if self.depth == image.attachment {
            return Some(None);
        }
        let slot = self
            .colors
            .iter()
            .position(|view| *view == image.attachment)?;
        Some(Some(slot as u32))
    }
}

/// A guest clear waiting for the rendering instance that draws into its
/// image, to be recorded there as an attachment clear. On a tiler that
/// clears tile memory; clearing the image before the instance wrote all of
/// it to memory behind a barrier and broke the instance, and LM3's title
/// clears about 23 targets a frame (a third of the Adreno 650's frame).
/// Anything else that may use the image first records it as before
/// (`barrier`).
#[derive(Clone, Copy)]
struct PendingClear {
    image: Image,
    /// COLOR, or the depth and stencil aspects cleared.
    aspect: vk::ImageAspectFlags,
    value: vk::ClearValue,
}

/// What one submission records into, used again once the GPU has run it.
/// The renderer has `SLOTS`: the worker records a submission while the GPU
/// runs the ones before (waiting for each submission had the worker idle
/// for a third of every LM3 frame).
struct Slot {
    command: vk::CommandBuffer,
    stream: Stream,
    descriptor_pool: vk::DescriptorPool,
    /// Signalled when the slot's last submission has run.
    fence: vk::Fence,
    /// Whether that submission may still be running.
    in_flight: bool,
    /// Images replaced while it was recorded.
    retired: Vec<Image>,
}

/// The graphics state draws have bound in the command buffer (null/None at
/// the start of a submission): binding it again is redundant.
#[derive(Default)]
struct BoundState {
    pipeline: vk::Pipeline,
    push_constants: Option<[u8; 16]>,
    viewport: Option<vk::Viewport>,
    scissor: Option<vk::Rect2D>,
    /// The buffer and offset bound to each vertex binding.
    vertex_buffers: [Option<(vk::Buffer, u64)>; MAX_VERTEX_BINDINGS],
    index_buffer: Option<(vk::Buffer, u64, vk::IndexType)>,
}

impl BoundState {
    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// A draw's indices in the stream.
#[derive(Clone, Copy)]
struct StreamIndices {
    offset: u64,
    count: u32,
    kind: vk::IndexType,
    /// The lowest and highest index (restart values aside): the vertices
    /// the draw reads.
    low: u32,
    high: u32,
}

/// The phases of recording a draw, timed for `RECOMP_GPU_STATS`.
#[derive(Clone, Copy)]
enum DrawPhase {
    /// Finding (or translating) the two programs.
    Shaders,
    /// Vertex and index data into the stream.
    Vertices,
    /// Constant buffers, textures and samplers, and the pipeline.
    DescriptorsAndPipeline,
    /// The Vulkan commands.
    Commands,
}

/// How many sets a batch allocation may take: never past the pool's limit.
fn descriptor_batch_count(allocated: u32) -> u32 {
    DESCRIPTOR_SETS
        .saturating_sub(allocated)
        .min(DESCRIPTOR_BATCH)
}

/// `RECOMP_SLOW_PIPELINES=<ms>`: report a shader translation or pipeline
/// creation that took at least this long. Off by default.
fn slow_pipeline_limit() -> Option<u128> {
    std::env::var("RECOMP_SLOW_PIPELINES")
        .ok()
        .and_then(|value| value.parse().ok())
}

/// The portable SPIR-V shader pack, when present. `RECOMP_SHADER_PACK=<file>`
/// chooses another path; an empty value disables it.
fn shader_pack_path() -> Option<std::path::PathBuf> {
    match std::env::var_os("RECOMP_SHADER_PACK") {
        Some(value) if value.is_empty() => None,
        Some(value) => Some(crate::paths::resolve(std::path::PathBuf::from(value))),
        None => Some(crate::paths::resolve("shader-pack.bin")),
    }
}

/// Where the driver's pipeline cache is kept. `RECOMP_PIPELINE_CACHE=<file>`
/// chooses another path; an empty value disables the cache. By default it is
/// beside the executable, so later runs reuse the compiled shaders.
fn pipeline_cache_path() -> Option<std::path::PathBuf> {
    match std::env::var_os("RECOMP_PIPELINE_CACHE") {
        Some(value) if value.is_empty() => None,
        Some(value) => Some(crate::paths::resolve(std::path::PathBuf::from(value))),
        None => Some(crate::paths::resolve("pipeline-cache.bin")),
    }
}

impl Gpu {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: &Device,
        command_pool: vk::CommandPool,
        depth_clip_control: bool,
        depth_clamp: bool,
        depth_bias_clamp: bool,
        dynamic_rendering: bool,
    ) -> Result<Self, String> {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let limits = unsafe { instance.get_physical_device_properties(physical_device) }.limits;

        // Compiled pipelines from the last run. A driver may reject a cache
        // from another device/driver: fall back to an empty one.
        let pipeline_cache_path = pipeline_cache_path();
        let initial_cache = pipeline_cache_path
            .as_ref()
            .and_then(|path| std::fs::read(path).ok())
            .unwrap_or_default();
        let pipeline_cache = match unsafe {
            device.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default().initial_data(&initial_cache),
                None,
            )
        } {
            Ok(cache) => cache,
            Err(_) => unsafe {
                device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
            }
            .map_err(error("pipeline cache"))?,
        };
        if !initial_cache.is_empty() {
            println!(
                "[recomp-vulkan] loaded {} bytes of pipeline cache",
                initial_cache.len()
            );
        }
        let shader_pack = shader_pack_path().and_then(|path| {
            if !path.exists() {
                return None;
            }
            match ShaderPack::load(&path) {
                Ok(pack) => Some(pack),
                Err(error) => {
                    eprintln!("[recomp-vulkan] shader pack: {error}");
                    None
                }
            }
        });

        let mut bindings = Vec::new();
        for binding in 0..UNIFORM_BINDINGS {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(
                        vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                    ),
            );
        }
        for index in 0..TEXTURE_BINDINGS {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(translate::TEXTURE_BINDING_BASE + index)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(
                        vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                    ),
            );
        }
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(error("descriptor set layout"))?;
        let set_layouts = [set_layout, set_layout];
        // The resolution scale programs read (`translate::RESCALE_*`).
        let rescale = |stage_flags| vk::PushConstantRange {
            stage_flags,
            offset: 0,
            size: RESCALE_BYTES,
        };
        let graphics_constants = [rescale(
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        )];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&graphics_constants),
                None,
            )
        }
        .map_err(error("pipeline layout"))?;
        let compute_constants = [rescale(vk::ShaderStageFlags::COMPUTE)];
        let compute_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts[..1])
                    .push_constant_ranges(&compute_constants),
                None,
            )
        }
        .map_err(error("compute pipeline layout"))?;
        // Room for every set the pool allows: each set takes all its
        // bindings. 65536 of each type ran out of uniform buffers after
        // about 3600 sets, before the early submit at 12000 sets.
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: DESCRIPTOR_SETS * UNIFORM_BINDINGS,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: DESCRIPTOR_SETS * TEXTURE_BINDINGS,
            },
        ];
        let commands = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(SLOTS as u32),
            )
        }
        .map_err(error("command buffer"))?;
        let mut slots = Vec::new();
        for command in commands {
            let descriptor_pool = unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(DESCRIPTOR_SETS)
                        .pool_sizes(&pool_sizes),
                    None,
                )
            }
            .map_err(error("descriptor pool"))?;
            // In video memory the CPU writes into (resizable BAR): shaders
            // read vertices and constants many times, and over PCIe from
            // system memory that made the GPU, not the worker, LM3's limit.
            let stream = host_buffer(
                device,
                &memory_properties,
                STREAM_BYTES,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(error("fence"))?;
            slots.push(Slot {
                command,
                stream,
                descriptor_pool,
                fence,
                in_flight: false,
                retired: Vec::new(),
            });
        }
        let mut spares = VecDeque::from(slots);
        let Slot {
            command,
            stream,
            descriptor_pool,
            fence,
            ..
        } = spares.pop_front().expect("slots");

        // Video memory the CPU writes into, like the stream.
        let vertex_cache = host_buffer(
            device,
            &memory_properties,
            VERTEX_CACHE_BYTES,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .map(|arena| VertexCache {
            arena,
            entries: HashMap::new(),
        })
        .map_err(|message| eprintln!("[recomp-vulkan] no vertex cache: {message}"))
        .ok();

        Ok(Self {
            device: device.clone(),
            instance: instance.clone(),
            physical_device,
            format_features: std::cell::RefCell::new(HashMap::new()),
            render_passes: if dynamic_rendering {
                None
            } else {
                Some(RenderPassCache::new(device.clone()))
            },
            depth_clip_control,
            depth_clamp,
            depth_bias_clamp,
            memory_properties,
            uniform_alignment: limits.min_uniform_buffer_offset_alignment.max(16),
            set_layout,
            pipeline_layout,
            compute_layout,
            compute_programs: HashMap::new(),
            descriptor_pool,
            descriptor_batch: Vec::new(),
            stream,
            vertex_cache,
            recent_draws: VecDeque::new(),
            scratch: HashMap::new(),
            written_since_submit: std::cell::RefCell::new(std::collections::HashSet::new()),
            readback: None,
            scale: crate::overlay::controls()
                .resolution
                .load(std::sync::atomic::Ordering::Acquire),
            shaders: HashMap::new(),
            shader_keys: HashMap::new(),
            shader_pack,
            pipelines: HashMap::new(),
            pipeline_cache,
            pipeline_cache_path,
            pipelines_since_cache_save: 0,
            pipeline_cache_saves: 0,
            pipeline_key: Vec::new(),
            pipeline_last: None,
            bound: BoundState::default(),
            targets: HashMap::new(),
            target_at: HashMap::new(),
            depths: HashMap::new(),
            depth_at: HashMap::new(),
            target_images: HashMap::new(),
            depth_images: HashMap::new(),
            writes: HashMap::new(),
            resized: HashMap::new(),
            layered: HashMap::new(),
            target_generation: 0,
            retired: Vec::new(),
            fence,
            in_flight: false,
            spares,
            slot: 0,
            read: std::cell::RefCell::new(HashSet::new()),
            textures: HashMap::new(),
            texture_watch: HashMap::new(),
            texture_generation: 0,
            submission: 0,
            decoder: Decoder::new(),
            prepared: VecDeque::new(),
            texture_waited: std::time::Duration::ZERO,
            drawn: HashMap::new(),
            samplers: HashMap::new(),
            unfilterable: std::cell::RefCell::new(HashSet::new()),
            placeholders: HashMap::new(),
            command,
            recording: false,
            presentable: None,
            display_area: 0,
            warned: HashSet::new(),
            draws: 0,
            skipped: 0,
            dispatches: 0,
            descriptor_sets: 0,
            trace: None,
            traced_textures: Vec::new(),
            anisotropy: std::env::var("RECOMP_ANISOTROPY")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1.0),
            max_anisotropy: if unsafe { instance.get_physical_device_features(physical_device) }
                .sampler_anisotropy
                == vk::TRUE
            {
                limits.max_sampler_anisotropy
            } else {
                1.0
            },
            phases: [std::time::Duration::ZERO; 4],
            gpu_waited: std::time::Duration::ZERO,
            copied: [0; 3],
            vertex_uploads: HashMap::new(),
            index_uploads: HashMap::new(),
            index_bytes: Vec::new(),
            index_values: Vec::new(),
            index_data: Vec::new(),
            index_triangles: Vec::new(),
            rendering: std::cell::Cell::new(None),
            written: std::cell::RefCell::new(HashSet::new()),
            pending_clears: std::cell::RefCell::new(Vec::new()),
            pass_counts: std::cell::Cell::new([0; 4]),
            sampled: Vec::new(),
        })
    }

    /// Print a message once.
    /// A 2D engine copy from a render target: the destination becomes the
    /// render target at its address, so later sampling there finds it.
    /// False (not run here) when the source is memory the GPU did not render:
    /// the command processor copies that in guest memory.
    pub fn blit(&mut self, memory: &dyn GpuMemory, blit: &Blit) -> Result<bool, String> {
        if skips("blit") {
            return Ok(false);
        }
        self.flush_clears();
        let src_address = canonical(memory, blit.src.address);
        let dst_address = canonical(memory, blit.dst.address);
        if let Some(trace) = self.trace.as_mut() {
            trace.push(format!(
                "blit {:#x} {}x{} fmt {:#x} -> {:#x} {}x{} fmt {:#x} rect {:?} from {:?} step {:?}{}",
                blit.src.address,
                blit.src.width,
                blit.src.height,
                blit.src.format,
                blit.dst.address,
                blit.dst.width,
                blit.dst.height,
                blit.dst.format,
                blit.dst_rect,
                blit.src_origin,
                blit.src_step,
                if self.target_at.contains_key(&src_address) {
                    ""
                } else {
                    " (source not rendered: copied in guest memory)"
                }
            ));
        }
        let Some(source_key) = self.target_at.get(&src_address).copied() else {
            return Ok(false);
        };
        let destination = ColorTargetState {
            address: dst_address,
            width_or_stride: blit.dst.width,
            height: blit.dst.height,
            format: blit.dst.format,
            memory_layout: if blit.dst.linear { 1 << 12 } else { 0 },
            depth: 1,
            layer_size: 0,
            base_layer: 0,
        };
        let Some(destination_key) = self.target(&destination)? else {
            return Ok(false);
        };
        self.begin()?;
        let (source, target) = (self.targets[&source_key], self.targets[&destination_key]);
        let [x, y, width, height] = blit.dst_rect;
        // Guest pixels, at each image's resolution scale.
        let clamp = |value: f64, limit: u32| value.round().clamp(0.0, f64::from(limit)) as i32;
        let source_factor = f64::from(source.scale) / 2.0;
        let src_x1 = blit.src_origin[0] + f64::from(width) * blit.src_step[0];
        let src_y1 = blit.src_origin[1] + f64::from(height) * blit.src_step[1];
        let (src_x0, src_y0, src_x1, src_y1) = (
            blit.src_origin[0] * source_factor,
            blit.src_origin[1] * source_factor,
            src_x1 * source_factor,
            src_y1 * source_factor,
        );
        let (x0, y0) = (x * target.scale / 2, y * target.scale / 2);
        let (x1, y1) = (
            scaled(x + width, target.scale),
            scaled(y + height, target.scale),
        );
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let region = vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [
                vk::Offset3D {
                    x: clamp(src_x0, source.width),
                    y: clamp(src_y0, source.height),
                    z: 0,
                },
                vk::Offset3D {
                    x: clamp(src_x1, source.width),
                    y: clamp(src_y1, source.height),
                    z: 1,
                },
            ],
            dst_subresource: layers,
            dst_offsets: [
                vk::Offset3D {
                    x: x0.min(target.width) as i32,
                    y: y0.min(target.height) as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: x1.min(target.width) as i32,
                    y: y1.min(target.height) as i32,
                    z: 1,
                },
            ],
        };
        let filter = if blit.linear_filter {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
        };
        self.barrier();
        unsafe {
            self.device.cmd_blit_image(
                self.command,
                source.image,
                vk::ImageLayout::GENERAL,
                target.image,
                vk::ImageLayout::GENERAL,
                std::slice::from_ref(&region),
                filter,
            );
        }
        self.barrier();
        Ok(true)
    }

    /// Run a compute dispatch: its program with the constant buffers and
    /// textures its launch descriptor names, between barriers.
    pub fn dispatch(
        &mut self,
        memory: &dyn GpuMemory,
        dispatch: &ComputeDispatch,
    ) -> Result<(), String> {
        let Some(key) = self.compute_program(memory, dispatch) else {
            return Ok(());
        };
        self.begin()?;
        self.flush_clears();
        let Some(program) = self.compute_programs[&key].as_ref() else {
            return Ok(());
        };
        let (pipeline, uniform_buffers, textures) = (
            program.pipeline,
            program
                .translated
                .uniform_buffers
                .iter()
                .copied()
                .zip(program.translated.uniform_extents.iter().copied())
                .collect::<Vec<_>>(),
            program.translated.textures.clone(),
        );
        let set = self.descriptor_set()?;
        let mut buffer_infos = Vec::new();
        for (slot, extent) in uniform_buffers {
            let buffer = dispatch
                .constant_buffers
                .get(slot as usize)
                .copied()
                .flatten()
                .map(|(address, size)| (address, size.min(extent)));
            buffer_infos.push((
                slot,
                self.stream
                    .push_uniform(memory, buffer, self.uniform_alignment)?,
            ));
        }
        let source = TextureSource {
            handles: dispatch
                .constant_buffers
                .get(dispatch.texture_buffer_index as usize)
                .copied()
                .flatten()
                .map(|(address, _)| address),
            texture_pool: dispatch.texture_pool,
            sampler_pool: dispatch.sampler_pool,
            samplers_from_texture_index: false,
        };
        let mut image_infos = Vec::new();
        let mut scaled_textures = 0u32;
        for binding in textures {
            let (view, sampler, scaled) = self.texture(memory, &source, &binding)?;
            let bit = binding.binding - TEXTURE_BINDING_BASE;
            if scaled && bit < 32 {
                scaled_textures |= 1 << bit;
            }
            image_infos.push((
                binding.binding,
                vk::DescriptorImageInfo {
                    sampler,
                    image_view: view,
                    image_layout: vk::ImageLayout::GENERAL,
                },
            ));
        }
        let mut writes = Vec::new();
        for (binding, info) in &buffer_infos {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(info)),
            );
        }
        for (binding, info) in &image_infos {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(info)),
            );
        }
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        self.barrier();
        let [x, y, z] = dispatch.grid.map(|count| count.max(1));
        unsafe {
            let device = &self.device;
            device.cmd_bind_pipeline(self.command, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                self.command,
                self.compute_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &rescale_constants(2, self.scale, [0, scaled_textures]),
            );
            device.cmd_dispatch(self.command, x, y, z);
        }
        self.barrier();
        self.dispatches += 1;
        Ok(())
    }

    /// The compute program a dispatch runs, translated and with its pipeline
    /// built on first use; `None` when it cannot be translated (said once).
    fn compute_program(
        &mut self,
        memory: &dyn GpuMemory,
        dispatch: &ComputeDispatch,
    ) -> Option<ComputeKey> {
        let address = dispatch.program_address;
        let hash = code_hash(&memory.read_gpu(address, 0x200).unwrap_or_default());
        let key = (address, hash, dispatch.block, dispatch.shared_memory_size);
        if !self.compute_programs.contains_key(&key) {
            let mut code = Vec::new();
            while (code.len() as u64) < PROGRAM_BYTES {
                match memory.read_gpu(address + code.len() as u64, 0x100) {
                    Some(chunk) => code.extend_from_slice(&chunk),
                    None => break,
                }
            }
            let environment = Environment {
                local_size: dispatch.block,
                shared_memory_bytes: dispatch.shared_memory_size,
                constant_buffer_1: constant_buffer_1(memory, dispatch.constant_buffers[1]),
                ..Environment::default()
            };
            let program = match translate::translate_compute(&code, &environment) {
                Ok(translated) => match self.compute_pipeline(&translated) {
                    Ok((module, pipeline)) => {
                        self.pipeline_cache_changed();
                        Some(ComputeProgram {
                            module,
                            pipeline,
                            translated,
                        })
                    }
                    Err(failure) => {
                        self.warn(format!("compute program {address:#x}: {failure}"));
                        None
                    }
                },
                Err(failure) => {
                    self.warn(format!("compute program {address:#x}: {failure}"));
                    None
                }
            };
            self.compute_programs.insert(key, program);
        }
        self.compute_programs[&key].is_some().then_some(key)
    }

    fn compute_pipeline(
        &self,
        translated: &TranslatedShader,
    ) -> Result<(vk::ShaderModule, vk::Pipeline), String> {
        // The compute layout uses the same descriptor set layout: an
        // out-of-range texture binding would crash the driver in
        // vkCreateComputePipelines just as it does for graphics.
        if let Some(binding) = translated
            .textures
            .iter()
            .find(|binding| !texture_binding_fits(binding.binding))
        {
            return Err(format!(
                "texture binding {} is outside the {TEXTURE_BINDINGS}-binding descriptor set",
                binding.binding
            ));
        }
        let module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&translated.words),
                None,
            )
        }
        .map_err(error("compute module"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.compute_layout);
        let started = std::time::Instant::now();
        let created = unsafe {
            self.device.create_compute_pipelines(
                self.pipeline_cache,
                std::slice::from_ref(&info),
                None,
            )
        };
        if let Some(limit) = slow_pipeline_limit() {
            let elapsed = started.elapsed();
            if elapsed.as_millis() >= limit {
                println!(
                    "[recomp-vulkan] compute pipeline created in {:.1} ms",
                    elapsed.as_secs_f64() * 1000.0
                );
            }
        }
        match created {
            Ok(pipelines) => Ok((module, pipelines[0])),
            Err((_, result)) => {
                unsafe { self.device.destroy_shader_module(module, None) };
                Err(format!("compute pipeline: {result}"))
            }
        }
    }

    fn warn(&mut self, message: String) {
        if self.warned.insert(message.clone()) {
            println!("[recomp-vulkan] {message}");
        }
    }

    /// Remember a recorded draw for a device-lost report.
    fn note_draw(&mut self, draw: &DrawCall) {
        let targets: Vec<String> = draw.color_targets[..draw.color_target_count]
            .iter()
            .map(|target| {
                format!(
                    "{:#x}/{:#x}/{}/{}",
                    target.address,
                    u64::from(target.format),
                    target.width_or_stride,
                    target.height
                )
            })
            .collect();
        let depth = if draw.depth_target_enable {
            format!(
                " depth {:#x}/{}x{}",
                draw.depth_target.address,
                draw.depth_target_size.width,
                draw.depth_target_size.height
            )
        } else {
            String::new()
        };
        let parameters = draw.parameters;
        self.recent_draws.push_back(format!(
            "#{} shaders {:#x}/{:#x} {:?} count {}{} first {}/{} instances {}{}{}",
            self.draws,
            draw.vertex_shader().unwrap_or(0),
            draw.fragment_shader().unwrap_or(0),
            parameters.topology,
            parameters.count,
            if draw.index_buffer.is_some() || !draw.inline_indices.is_empty() {
                " indexed"
            } else {
                ""
            },
            parameters.first_index,
            parameters.first_vertex,
            parameters.instance_count,
            if targets.is_empty() {
                String::new()
            } else {
                format!(" colors [{}]", targets.join(" "))
            },
            depth
        ));
        // A frame is about 30 draws here; keep a few frames' worth so a
        // hang can be traced to the draw that started it.
        while self.recent_draws.len() > recent_capacity() {
            self.recent_draws.pop_front();
        }
    }

    /// Add a non-draw operation (a clear, a copy) to the recent draws, for
    /// the report of a lost device.
    pub fn note_recent(&mut self, operation: String) {
        self.recent_draws.push_back(operation);
        while self.recent_draws.len() > recent_capacity() {
            self.recent_draws.pop_front();
        }
    }

    /// Note when the newest recent operation was submitted and finished
    /// (CLOCK_MONOTONIC seconds), in flush mode.
    pub fn stamp_recent(&mut self, submission: u64, submitted: f64, finished: f64) {
        if let Some(last) = self.recent_draws.back_mut() {
            last.push_str(&format!(
                " [submission {submission} {submitted:.6}..{finished:.6}]"
            ));
        }
    }

    /// The newest of the recent draws and operations.
    pub fn last_recent(&self) -> &str {
        self.recent_draws.back().map_or("nothing", String::as_str)
    }

    /// The last draws, newest last; printed when the device is lost (the
    /// last 64 of them: flush mode keeps more, for `all_recent`).
    pub fn recent_draws(&self) -> String {
        let skip = self.recent_draws.len().saturating_sub(64);
        self.recent_draws
            .iter()
            .skip(skip)
            .map(|draw| format!("\n  [recomp-vulkan]   {draw}"))
            .collect()
    }

    /// Every recent draw and operation kept, one per line.
    pub fn all_recent(&self) -> String {
        self.recent_draws
            .iter()
            .map(|draw| format!("{draw}\n"))
            .collect()
    }

    /// How many cached objects exist, printed when the device is lost (a
    /// cache that grew without bound points at a leak, not a bad draw).
    pub fn cache_sizes(&self) -> String {
        let (passes, framebuffers) = self.render_passes.as_ref().map_or((0, 0), |cache| {
            (
                cache.passes.borrow().len(),
                cache.framebuffers.borrow().len(),
            )
        });
        format!(
            "{passes} render passes, {framebuffers} framebuffers, {} shaders, {} pipelines, {} targets, {} depth targets, {} textures, {} samplers",
            self.shaders.len(),
            self.pipelines.len(),
            self.targets.len(),
            self.depths.len(),
            self.textures.len(),
            self.samplers.len()
        )
    }

    /// Start recording a submission's commands, if not started.
    pub fn begin(&mut self) -> Result<(), String> {
        if self.recording {
            return Ok(());
        }
        // This slot's previous submission (`SLOTS` back) must have run.
        let started = std::time::Instant::now();
        self.wait()?;
        self.gpu_waited += started.elapsed();
        unsafe {
            self.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(error("reset descriptor pool"))?;
            self.device
                .reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .map_err(error("reset command buffer"))?;
            self.device
                .begin_command_buffer(
                    self.command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(error("begin command buffer"))?;
        }
        // The previous submission has finished: nothing uses these now.
        for image in std::mem::take(&mut self.retired) {
            self.destroy_image(image);
        }
        // A new command buffer starts with no graphics state bound.
        self.bound.reset();
        self.descriptor_batch.clear();
        // Textures the host's engines rewrote (a movie's frames, which VIC
        // converts into the same texture) are read again.
        for (start, len) in recomp_gfx::engine_writes::take() {
            let stale: Vec<[u32; 8]> = self
                .textures
                .keys()
                .filter(|words| {
                    (start..start + len).contains(&TextureHeader { words: **words }.address())
                })
                .copied()
                .collect();
            for words in stale {
                if let Some(TextureState::Ready(image) | TextureState::Preview(image, _)) =
                    self.textures.remove(&words)
                {
                    self.retired.push(image);
                }
            }
        }
        self.stream.cursor = 0;
        let scale = crate::overlay::controls()
            .resolution
            .load(std::sync::atomic::Ordering::Acquire);
        if scale != self.scale {
            self.rescale_targets(scale);
        }
        if self
            .vertex_cache
            .as_ref()
            .is_some_and(|cache| cache.arena.cursor > VERTEX_CACHE_BYTES / 4 * 3)
        {
            // The other slot's submission may still read it.
            self.wait_submitted()?;
            if let Some(cache) = &mut self.vertex_cache {
                cache.arena.cursor = 0;
                cache.entries.clear();
            }
        }
        self.vertex_uploads.clear();
        self.index_uploads.clear();
        self.shader_keys.clear();
        self.descriptor_sets = 0;
        self.texture_waited = std::time::Duration::ZERO;
        self.submission += 1;
        self.recording = true;
        self.presentable = None;
        // Commands of the submission before may still run: nothing here
        // starts until they are done with what they read and wrote.
        self.barrier();
        self.upload_prepared()
    }

    /// Make render targets at `scale` from now on: the ones made at another
    /// are dropped (their contents with them; the title draws most of them
    /// again every frame, so a frame or two may look wrong).
    fn rescale_targets(&mut self, scale: u32) {
        let mut images: Vec<Image> = self
            .target_images
            .drain()
            .chain(self.depth_images.drain())
            .map(|(_, image)| image)
            .collect();
        images.extend(self.scratch.drain().map(|(_, image)| image));
        images.extend(self.resized.drain().map(|(_, (copy, _))| copy));
        images.extend(
            self.layered
                .drain()
                .filter_map(|(_, layered)| layered.image),
        );
        // Every target image is going away: no framebuffer may name its views.
        if let Some(cache) = &self.render_passes {
            cache.forget_all();
        }
        self.retired.extend(images);
        self.targets.clear();
        self.depths.clear();
        self.target_at.clear();
        self.depth_at.clear();
        self.writes.clear();
        self.drawn.clear();
        self.written.borrow_mut().clear();
        self.read.borrow_mut().clear();
        self.sampled.clear();
        self.presentable = None;
        self.display_area = 0;
        self.scale = scale;
    }

    /// Wait until this slot's last submission has run.
    pub fn wait(&mut self) -> Result<(), String> {
        if self.in_flight {
            unsafe {
                self.device
                    .wait_for_fences(std::slice::from_ref(&self.fence), true, u64::MAX)
                    .map_err(error("fence wait"))?;
                self.device
                    .reset_fences(std::slice::from_ref(&self.fence))
                    .map_err(error("fence reset"))?;
            }
            self.in_flight = false;
        }
        Ok(())
    }

    /// The renderer submitted this slot's commands (with `fence`); the
    /// next submission records into the other slot.
    pub fn submitted(&mut self) {
        self.written_since_submit.borrow_mut().clear();
        self.in_flight = true;
        // The next submission records into the slot submitted longest ago,
        // the likeliest to have run; this one goes to the back.
        let mut spare = self.spares.pop_front().expect("spare slots");
        std::mem::swap(&mut self.command, &mut spare.command);
        std::mem::swap(&mut self.stream, &mut spare.stream);
        std::mem::swap(&mut self.descriptor_pool, &mut spare.descriptor_pool);
        std::mem::swap(&mut self.fence, &mut spare.fence);
        std::mem::swap(&mut self.in_flight, &mut spare.in_flight);
        std::mem::swap(&mut self.retired, &mut spare.retired);
        self.spares.push_back(spare);
        self.slot = (self.slot + 1) % SLOTS;
        // The batch's sets belong to the pool just submitted.
        self.descriptor_batch.clear();
    }

    /// Wait until every submission made (the spare slots') has run.
    pub fn wait_submitted(&mut self) -> Result<(), String> {
        for spare in self.spares.iter_mut().filter(|spare| spare.in_flight) {
            unsafe {
                self.device
                    .wait_for_fences(std::slice::from_ref(&spare.fence), true, u64::MAX)
                    .map_err(error("fence wait"))?;
                self.device
                    .reset_fences(std::slice::from_ref(&spare.fence))
                    .map_err(error("fence reset"))?;
            }
            spare.in_flight = false;
        }
        Ok(())
    }

    /// Which slot is recording (0 to `SLOTS - 1`): the renderer keeps a pair
    /// of swapchain semaphores per slot.
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Upload the textures the decoder finished, up to the budget.
    fn upload_prepared(&mut self) -> Result<(), String> {
        self.prepared.extend(self.decoder.finished());
        let mut budget = UPLOAD_BUDGET;
        while let Some(next) = self.prepared.front() {
            if next.texels.len() > budget && budget < UPLOAD_BUDGET {
                break;
            }
            let prepared = self.prepared.pop_front().expect("front exists");
            budget = budget.saturating_sub(prepared.texels.len());
            self.upload_texture(prepared)?;
        }
        Ok(())
    }

    /// Make a prepared texture's image and copy its texels in, unless the
    /// header no longer waits for it.
    fn upload_texture(&mut self, prepared: Prepared) -> Result<(), String> {
        let (pending, preview) = match self.textures.get(&prepared.words).copied() {
            Some(TextureState::Pending(pending)) => (pending, None),
            Some(TextureState::Preview(image, pending)) => (pending, Some(image)),
            _ => return Ok(()),
        };
        if pending.generation != prepared.generation {
            return Ok(());
        }
        // A preview is an image of the small levels alone, made only while
        // the whole texture has not come.
        let first = prepared.first_level.min(pending.levels - 1);
        if first > 0 && preview.is_some() {
            return Ok(());
        }
        let (width, height) = (
            (pending.width >> first).max(1),
            (pending.height >> first).max(1),
        );
        let levels = pending.levels - first;
        let view = if first > 0 { (0, levels) } else { pending.view };
        let image = self.create_image_levels(
            pending.format,
            width,
            height,
            pending.layers,
            pending.view_type,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            pending.components,
            levels,
            view,
        )?;
        let offset = self
            .stream
            .push(&prepared.texels, 16)
            .ok_or("stream full")?;
        self.copy_to_image(
            offset,
            &image,
            width,
            height,
            pending.layers,
            pending.view_type,
            &prepared.levels,
        );
        let state = if first > 0 {
            TextureState::Preview(image, pending)
        } else {
            if let Some(old) = preview {
                self.retired.push(old);
            }
            TextureState::Ready(image)
        };
        self.textures.insert(prepared.words, state);
        Ok(())
    }

    /// Wait for the decoder to finish the texture `words` names and upload
    /// it now, for a draw about to sample it: up to `TEXTURE_WAIT` in all
    /// per submission. A draw that sampled the transparent placeholder
    /// instead showed new things (a room behind a door opening, debris)
    /// flickering in over the frames until the next submission uploaded
    /// them.
    fn wait_for_texture(&mut self, words: [u32; 8]) -> Result<(), String> {
        self.prepared.extend(self.decoder.finished());
        self.take_quick(words);
        loop {
            if self.upload_arrived(words)? {
                return Ok(());
            }
            let left = TEXTURE_WAIT.saturating_sub(self.texture_waited);
            if left.is_zero() {
                return Ok(());
            }
            let started = std::time::Instant::now();
            let next = self.decoder.wait(left);
            self.texture_waited += started.elapsed();
            match next {
                Some(prepared) => self.prepared.push_back(prepared),
                None => return Ok(()),
            }
        }
    }

    /// Prepare here a quick job for the texture `words` names that no
    /// worker has started, whatever is left of the wait (see `decode.rs`).
    fn take_quick(&mut self, words: [u32; 8]) {
        if let Some(job) = self.decoder.take(words) {
            self.prepared.push_back(crate::decode::prepare(job));
        }
    }

    /// Upload the texture `words` names if its preparation has arrived (or
    /// leave it for the next submission when the stream is nearly full);
    /// false when it has not arrived.
    fn upload_arrived(&mut self, words: [u32; 8]) -> Result<bool, String> {
        let generation = match self.textures.get(&words) {
            Some(TextureState::Pending(pending) | TextureState::Preview(_, pending)) => {
                pending.generation
            }
            _ => return Ok(true),
        };
        let Some(position) = self
            .prepared
            .iter()
            .position(|prepared| prepared.words == words && prepared.generation == generation)
        else {
            return Ok(false);
        };
        let room = STREAM_BYTES / 4 * 3;
        if self.stream.cursor + self.prepared[position].texels.len() as u64 > room {
            return Ok(true);
        }
        let prepared = self.prepared.remove(position).expect("found");
        self.upload_texture(prepared)?;
        Ok(true)
    }

    /// Whether the guest wrote the bytes of the cached texture `words`
    /// names since they were read: checked once per submission.
    fn rewritten(&mut self, memory: &dyn GpuMemory, words: [u32; 8]) -> bool {
        let submission = self.submission;
        let Some(watch) = self.texture_watch.get_mut(&words) else {
            return false;
        };
        if watch.submission == submission {
            return false;
        }
        watch.submission = submission;
        if !watch.check {
            // No kernel write watch on this host: compare the source bytes.
            let Some(hash) = texture_hash(memory, watch.address, watch.len) else {
                self.texture_watch.remove(&words);
                return false;
            };
            let changed = watch.hash != Some(hash);
            watch.hash = Some(hash);
            return changed;
        }
        let mut written = false;
        let checked = memory.written_since(
            watch.address,
            watch.len as usize,
            Some(watch.checked),
            &mut |_, _| written = true,
        );
        match checked {
            Some(checked) => watch.checked = checked,
            None => {
                self.texture_watch.remove(&words);
                return false;
            }
        }
        written
    }

    /// Watch the `len` bytes at `address` the texture `words` names is
    /// about to be read from.
    fn watch_texture(
        &mut self,
        memory: &dyn GpuMemory,
        words: [u32; 8],
        address: u64,
        len: u64,
        hash_only: bool,
    ) {
        // LM3 rewrites nine small 3D colour LUTs one per frame when a room's
        // look fades in (a door opening). Hash those directly: the 3D range
        // is small, and Windows' page watch is unreliable for these pool
        // textures. Regular textures use the page watch.
        if hash_only {
            let hash = texture_hash(memory, address, len);
            if let Some(hash) = hash {
                self.texture_watch.insert(
                    words,
                    TextureWatch {
                        address,
                        len,
                        checked: 0,
                        check: false,
                        hash: Some(hash),
                        submission: self.submission,
                    },
                );
            } else {
                self.texture_watch.remove(&words);
            }
            return;
        }
        match memory.written_since(address, len as usize, None, &mut |_, _| {}) {
            Some(checked) => {
                self.texture_watch.insert(
                    words,
                    TextureWatch {
                        address,
                        len,
                        checked,
                        check: true,
                        hash: None,
                        submission: self.submission,
                    },
                );
            }
            None => {
                // A page watch that cannot cover this range must not turn
                // into hashing every texture: leave it cached.
                self.texture_watch.remove(&words);
            }
        }
    }

    /// Whether the submission being recorded has used most of the stream
    /// or of the descriptor pool, so it should be sent before the next
    /// draw: a title scene's thousands of draws filled the 128 MB stream,
    /// and every draw after that was skipped.
    pub fn nearly_full(&self) -> bool {
        self.recording
            && (self.stream.cursor > STREAM_BYTES / 4 * 3 || self.descriptor_sets > 12_000)
    }

    /// Close the rendering instance draws are adding to: anything else
    /// (copies, clears, barriers, dispatches) is recorded outside one.
    fn end_rendering(&self) {
        if self.rendering.take().is_some() {
            if let Some(cache) = &self.render_passes {
                cache.end(self.command);
            } else {
                unsafe { self.device.cmd_end_rendering(self.command) };
            }
        }
    }

    pub fn end(&mut self) -> Result<(), String> {
        self.flush_clears();
        self.end_rendering();
        self.recording = false;
        unsafe { self.device.end_command_buffer(self.command) }.map_err(error("end command buffer"))
    }

    /// Everything before waits for everything after: draws sample targets
    /// earlier draws rendered.
    fn barrier(&self) {
        let pending = std::mem::take(&mut *self.pending_clears.borrow_mut());
        if !pending.is_empty() {
            // After what used their images before, before what uses them
            // next.
            self.memory_barrier();
            for clear in &pending {
                self.clear_now(clear);
            }
        }
        self.memory_barrier();
    }

    /// Record the clears still waiting for a rendering instance, before an
    /// operation that may use their images.
    fn flush_clears(&self) {
        if !self.pending_clears.borrow().is_empty() {
            self.barrier();
        }
    }

    fn count(&self, index: usize, by: u64) {
        let mut counts = self.pass_counts.get();
        counts[index] += by;
        self.pass_counts.set(counts);
    }

    /// Rendering instances begun, barriers, and clears recorded in and out
    /// of instances since the last call.
    pub fn take_pass_counts(&mut self) -> [u64; 4] {
        self.pass_counts.take()
    }

    /// A clear as an image clear, outside rendering instances.
    fn clear_now(&self, clear: &PendingClear) {
        self.end_rendering();
        self.count(3, 1);
        let range = vk::ImageSubresourceRange {
            aspect_mask: clear.aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        self.written.borrow_mut().insert(clear.image.image);
        self.written_since_submit
            .borrow_mut()
            .insert(clear.image.image);
        // SAFETY: the value's member is the one the aspect names.
        unsafe {
            if clear.aspect == vk::ImageAspectFlags::COLOR {
                self.device.cmd_clear_color_image(
                    self.command,
                    clear.image.image,
                    vk::ImageLayout::GENERAL,
                    &clear.value.color,
                    std::slice::from_ref(&range),
                );
            } else {
                self.device.cmd_clear_depth_stencil_image(
                    self.command,
                    clear.image.image,
                    vk::ImageLayout::GENERAL,
                    &clear.value.depth_stencil,
                    std::slice::from_ref(&range),
                );
            }
        }
    }

    /// Clears as attachment clears of the open rendering instance, over its
    /// whole area (`width` x `height`): `None` for the depth attachment.
    fn clear_in_instance(&self, clears: &[(Option<u32>, PendingClear)], width: u32, height: u32) {
        if clears.is_empty() {
            return;
        }
        self.count(2, clears.len() as u64);
        let attachments: Vec<vk::ClearAttachment> = clears
            .iter()
            .map(|(slot, clear)| vk::ClearAttachment {
                aspect_mask: clear.aspect,
                color_attachment: slot.unwrap_or(0),
                clear_value: clear.value,
            })
            .collect();
        let rect = vk::ClearRect {
            rect: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            },
            base_array_layer: 0,
            layer_count: 1,
        };
        for (_, clear) in clears {
            self.written.borrow_mut().insert(clear.image.image);
            self.written_since_submit
                .borrow_mut()
                .insert(clear.image.image);
        }
        unsafe {
            self.device.cmd_clear_attachments(
                self.command,
                &attachments,
                std::slice::from_ref(&rect),
            );
        }
    }

    /// Whether a clear can be recorded in a rendering instance of
    /// `width` x `height`: one that covers its image, keeping stencil if
    /// the clear sets it (`stencil_kept`; a render pass that does not test
    /// stencil neither loads nor stores it).
    fn clears_in(clear: &PendingClear, width: u32, height: u32, stencil_kept: bool) -> bool {
        width >= clear.image.width
            && height >= clear.image.height
            && (stencil_kept || !clear.aspect.contains(vk::ImageAspectFlags::STENCIL))
    }

    /// Clear an image where it costs least: in the open rendering instance
    /// when the image is one of its attachments, else in the next one that
    /// draws into it (see `PendingClear`).
    fn clear_later(&mut self, clear: PendingClear) {
        if let Some(instance) = self.rendering.get() {
            // The instance's render pass, if any, may not keep stencil.
            let stencil_kept = self.render_passes.is_none();
            // An image sampled since the last barrier waits for one.
            let read = self.read.borrow().contains(&clear.image.image);
            if let Some(slot) = instance.attachment(&clear.image).filter(|_| !read) {
                if Self::clears_in(&clear, instance.width, instance.height, stencil_kept) {
                    self.clear_in_instance(&[(slot, clear)], instance.width, instance.height);
                    return;
                }
            }
        }
        // A clear of the same image and aspects replaces a waiting one.
        let mut pending = self.pending_clears.borrow_mut();
        pending.retain(|waiting| {
            waiting.image.image != clear.image.image || !clear.aspect.contains(waiting.aspect)
        });
        pending.push(clear);
    }

    fn memory_barrier(&self) {
        self.end_rendering();
        self.count(1, 1);
        self.written.borrow_mut().clear();
        self.read.borrow_mut().clear();
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_image(
        &mut self,
        format: vk::Format,
        width: u32,
        height: u32,
        layers: u32,
        view_type: vk::ImageViewType,
        usage: vk::ImageUsageFlags,
        components: vk::ComponentMapping,
    ) -> Result<Image, String> {
        self.create_image_levels(
            format,
            width,
            height,
            layers,
            view_type,
            usage,
            components,
            1,
            (0, 1),
        )
    }

    /// An image of `levels` mip levels whose views show `view` (the first
    /// level and the count).
    #[allow(clippy::too_many_arguments)]
    fn create_image_levels(
        &mut self,
        format: vk::Format,
        width: u32,
        height: u32,
        layers: u32,
        view_type: vk::ImageViewType,
        usage: vk::ImageUsageFlags,
        components: vk::ComponentMapping,
        levels: u32,
        view: (u32, u32),
    ) -> Result<Image, String> {
        let (image_type, depth, array_layers) = match view_type {
            vk::ImageViewType::TYPE_1D => (vk::ImageType::TYPE_1D, 1, 1),
            vk::ImageViewType::TYPE_1D_ARRAY => (vk::ImageType::TYPE_1D, 1, layers),
            vk::ImageViewType::TYPE_3D => (vk::ImageType::TYPE_3D, layers, 1),
            _ => (vk::ImageType::TYPE_2D, 1, layers),
        };
        let mut flags = if matches!(
            view_type,
            vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY
        ) {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        };
        // sRGB images are stored as their UNORM twin and viewed as sRGB, so
        // blits (which convert by the image's format) copy the bytes as they
        // are: the window shows what the guest wrote.
        let storage = storage_format(format);
        if storage != format {
            flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }
        let image = unsafe {
            self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .flags(flags)
                    .image_type(image_type)
                    .format(storage)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth,
                    })
                    .mip_levels(levels)
                    .array_layers(array_layers)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(error("create image"))?;
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let memory_type = memory_type(
            &self.memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or("no device-local memory for an image")?;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }
        .map_err(error("image memory"))?;
        unsafe { self.device.bind_image_memory(image, memory, 0) }.map_err(error("bind image"))?;
        let aspects = formats::aspects(format);
        let range = vk::ImageSubresourceRange {
            aspect_mask: aspects,
            base_mip_level: view.0,
            level_count: view.1,
            base_array_layer: 0,
            layer_count: array_layers,
        };
        let make_view = |aspect_mask: vk::ImageAspectFlags| unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(view_type)
                    .format(format)
                    .components(components)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask,
                        ..range
                    }),
                None,
            )
        };
        // Shaders sample one aspect: a depth-stencil image's depth.
        let sampled = if aspects.contains(vk::ImageAspectFlags::DEPTH) {
            vk::ImageAspectFlags::DEPTH
        } else {
            aspects
        };
        let view = make_view(sampled).map_err(error("image view"))?;
        if !format_features(
            &self.format_features,
            &self.instance,
            self.physical_device,
            format,
        )
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            self.unfilterable.borrow_mut().insert(view);
        }
        let attachment = if sampled == aspects {
            view
        } else {
            make_view(aspects).map_err(error("attachment view"))?
        };
        // Into GENERAL, where every image stays.
        self.end_rendering();
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                base_mip_level: 0,
                level_count: levels,
                ..range
            });
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }
        Ok(Image {
            image,
            memory,
            view,
            attachment,
            format,
            width,
            height,
            scale: 2,
        })
    }

    /// The host image of the guest's depth-stencil target, created on first
    /// use.
    fn depth_target(
        &mut self,
        memory: &dyn GpuMemory,
        address: u64,
        format: u32,
        width: u32,
        height: u32,
    ) -> Result<Option<TargetKey>, String> {
        let address = canonical(memory, address);
        if address == 0 || width == 0 {
            return Ok(None);
        }
        let Some(host) = formats::depth_format(format) else {
            self.warn(format!("depth format {format:#x} not rendered yet"));
            return Ok(None);
        };
        let key = TargetKey {
            address,
            width,
            height,
            format,
        };
        if !self.depths.contains_key(&key) {
            let image = self.shared_target(key, host, true)?;
            self.depths.insert(key, image);
        }
        *self.writes.entry(self.depths[&key].image).or_default() += 1;
        if self.depth_at.insert(address, key).is_none() {
            self.target_generation += 1;
        }
        Ok(Some(key))
    }

    pub fn clear_depth(
        &mut self,
        memory: &dyn GpuMemory,
        clear: &ClearDepthStencilCall,
    ) -> Result<(), String> {
        if skips("clear") {
            return Ok(());
        }
        self.begin()?;
        let target = &clear.target;
        let Some(key) = self.depth_target(
            memory,
            target.address,
            target.format,
            clear.width,
            clear.height,
        )?
        else {
            return Ok(());
        };
        let image = self.depths[&key];
        let mut aspect_mask = vk::ImageAspectFlags::empty();
        if clear.clear_depth {
            aspect_mask |= vk::ImageAspectFlags::DEPTH;
        }
        if clear.clear_stencil && formats::has_stencil(image.format) {
            aspect_mask |= vk::ImageAspectFlags::STENCIL;
        }
        if aspect_mask.is_empty() {
            return Ok(());
        }
        self.clear_later(PendingClear {
            image,
            aspect: aspect_mask,
            value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: clear.depth.clamp(0.0, 1.0),
                    stencil: clear.stencil & 0xFF,
                },
            },
        });
        Ok(())
    }

    /// The host image of a guest color target, created on first use.
    fn target(&mut self, state: &ColorTargetState) -> Result<Option<TargetKey>, String> {
        if !state.enabled() {
            return Ok(None);
        }
        let Some(format) = formats::color_target_format(state.format) else {
            self.warn(format!(
                "color target format {:#x} not rendered yet",
                state.format
            ));
            return Ok(None);
        };
        // A pitch-linear target's first size word is a row stride in bytes;
        // the width is that stride over the format's texel size. LM3 renders
        // its save screen and save thumbnail into such targets.
        let mut state = *state;
        if state.memory_layout & (1 << 12) != 0 {
            let Some(width) = formats::pitch_linear_width(state.width_or_stride, state.format)
            else {
                self.warn(format!(
                    "pitch-linear color target format {:#x} not rendered yet",
                    state.format
                ));
                return Ok(None);
            };
            state.width_or_stride = width;
            state.memory_layout = 0;
        }
        let key = TargetKey {
            address: state.address,
            width: state.width_or_stride,
            height: state.height,
            format: state.format,
        };
        if !self.targets.contains_key(&key) {
            let image = self.shared_target(key, format, false)?;
            self.targets.insert(key, image);
        }
        *self.writes.entry(self.targets[&key].image).or_default() += 1;
        if self.target_at.insert(key.address, key).is_none() {
            self.target_generation += 1;
        }
        Ok(Some(key))
    }

    /// The image for the target `key` names: the one at its address and
    /// format when that is large enough, else a new one as large as both,
    /// with the old one's pixels in its corner (every key there then names
    /// the new one). Draws render into the corner their size covers.
    fn shared_target(
        &mut self,
        key: TargetKey,
        format: vk::Format,
        depth: bool,
    ) -> Result<Image, String> {
        let slot = (key.address, key.format);
        let height = key.height.max(1);
        let existing = if depth {
            self.depth_images.get(&slot)
        } else {
            self.target_images.get(&slot)
        }
        .copied();
        // The guest's size the image holds: this key's and the old image's.
        let (guest_width, guest_height) = existing.map_or((key.width, height), |old| {
            (
                (old.width * 2 / old.scale).max(key.width),
                (old.height * 2 / old.scale).max(height),
            )
        });
        let scale = if guest_width >= SCALED_MIN.0 && guest_height >= SCALED_MIN.1 {
            self.scale
        } else {
            2
        };
        if let Some(image) = existing {
            if image.scale == scale
                && image.width >= scaled(key.width, scale)
                && image.height >= scaled(height, scale)
            {
                return Ok(image);
            }
        }
        self.begin()?;
        let attachment = if depth {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
        let image = Image {
            scale,
            ..self.create_image(
                format,
                scaled(guest_width, scale),
                scaled(guest_height, scale),
                1,
                vk::ImageViewType::TYPE_2D,
                attachment
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ComponentMapping::default(),
            )?
        };
        if let Some(old) = existing {
            // Pixels at another scale do not fit: the title draws them again.
            if old.scale == scale {
                self.copy_corner(old, image, old.width, old.height);
            }
            let keys = if depth {
                &mut self.depths
            } else {
                &mut self.targets
            };
            for (other, value) in keys.iter_mut() {
                if (other.address, other.format) == slot {
                    *value = image;
                }
            }
            if let Some((copy, _)) = self.resized.remove(&old.image) {
                self.retired.push(copy);
            }
            self.writes.remove(&old.image);
            self.retired.push(old);
        }
        if depth {
            self.depth_images.insert(slot, image);
        } else {
            self.target_images.insert(slot, image);
        }
        Ok(image)
    }

    /// Copy the top-left `width` x `height` of `source` into `destination`
    /// (both in the same format).
    fn copy_corner(&mut self, source: Image, destination: Image, width: u32, height: u32) {
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: formats::aspects(source.format),
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let region = vk::ImageCopy {
            src_subresource: layers,
            src_offset: vk::Offset3D::default(),
            dst_subresource: layers,
            dst_offset: vk::Offset3D::default(),
            extent: vk::Extent3D {
                width: width.min(source.width).min(destination.width),
                height: height.min(source.height).min(destination.height),
                depth: 1,
            },
        };
        self.barrier();
        unsafe {
            self.device.cmd_copy_image(
                self.command,
                source.image,
                vk::ImageLayout::GENERAL,
                destination.image,
                vk::ImageLayout::GENERAL,
                std::slice::from_ref(&region),
            );
        }
        self.barrier();
    }

    /// A view of a target image at the size a texture header samples it:
    /// the image itself, or a copy of its top-left corner at that size.
    /// Sampling normalizes coordinates to the texture's size, so a frame
    /// that dynamic resolution rendered smaller than the image must be read
    /// at its own size, or it lands stretched and shifted.
    fn sampled_view(
        &mut self,
        image: Image,
        width: u32,
        height: u32,
    ) -> Result<vk::ImageView, String> {
        let (width, height) = (
            scaled(width.max(1), image.scale),
            scaled(height.max(1), image.scale),
        );
        // Sampling an image this submission rendered into is the pattern
        // Adreno's tile path deadlocks on: snapshot it instead. (The copy is
        // cached per write below, so one write still costs one copy.)
        let written = self.written_since_submit.borrow().contains(&image.image);
        if !written && (image.width, image.height) == (width, height) {
            self.sampled.push(image.image);
            return Ok(image.view);
        }
        let writes = self.writes.get(&image.image).copied().unwrap_or(0);
        let copy = match self.resized.get(&image.image).copied() {
            Some((copy, made_at)) if (copy.width, copy.height) == (width, height) => {
                if made_at == writes {
                    return Ok(copy.view);
                }
                copy
            }
            previous => {
                if let Some((old, _)) = previous {
                    self.retired.push(old);
                }
                Image {
                    scale: image.scale,
                    ..self.create_image(
                        image.format,
                        width,
                        height,
                        1,
                        vk::ImageViewType::TYPE_2D,
                        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                        vk::ComponentMapping::default(),
                    )?
                }
            }
        };
        self.copy_corner(image, copy, width, height);
        self.resized.insert(image.image, (copy, writes));
        Ok(copy.view)
    }

    /// Write both stages' SPIR-V to `RECOMP_VULKAN_SHADER_DUMP`, for
    /// spirv-val and spirv-dis on the host.
    fn dump_shaders(&self, vertex_key: ShaderKey, fragment_key: ShaderKey) {
        let Some(dir) = std::env::var_os("RECOMP_VULKAN_SHADER_DUMP") else {
            return;
        };
        let dir = crate::paths::resolve(std::path::PathBuf::from(dir));
        let _ = std::fs::create_dir_all(&dir);
        for (stage, key) in [("vertex", vertex_key), ("fragment", fragment_key)] {
            if let Some(Some(shader)) = self.shaders.get(&key) {
                let mut bytes = Vec::with_capacity(shader.translated.words.len() * 4);
                for word in &shader.translated.words {
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
                let path = dir.join(format!("{stage}-{:x}-{:x}.spv", key.0, key.2));
                let _ = std::fs::write(&path, bytes);
            }
        }
    }

    /// A color attachment for the unused slots of a sparse draw: the render
    /// pass needs a real view, and the draw's writes to it go nowhere.
    fn scratch_target(&mut self, width: u32, height: u32) -> Result<Image, String> {
        let size = (width.max(1), height.max(1));
        if let Some(image) = self.scratch.get(&size) {
            return Ok(*image);
        }
        let image = Image {
            scale: 2,
            ..self.create_image(
                vk::Format::R8G8B8A8_UNORM,
                size.0,
                size.1,
                1,
                vk::ImageViewType::TYPE_2D,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ComponentMapping::default(),
            )?
        };
        self.scratch.insert(size, image);
        Ok(image)
    }

    fn destroy_image(&self, image: Image) {
        // Before the views die: any framebuffer naming them must go.
        if let Some(cache) = &self.render_passes {
            cache.forget_views(&[image.view, image.attachment]);
        }
        self.unfilterable.borrow_mut().remove(&image.view);
        unsafe {
            if image.attachment != image.view {
                self.device.destroy_image_view(image.attachment, None);
            }
            self.device.destroy_image_view(image.view, None);
            self.device.destroy_image(image.image, None);
            self.device.free_memory(image.memory, None);
        }
    }

    pub fn clear(&mut self, memory: &dyn GpuMemory, clear: &ClearColorCall) -> Result<(), String> {
        if skips("clear") {
            return Ok(());
        }
        self.begin()?;
        if let Some(trace) = self.trace.as_mut() {
            let target = &clear.target;
            trace.push(format!(
                "clear {:#x} {}x{} fmt {:#x} mask {:#x} to {:?}",
                target.address,
                target.width_or_stride,
                target.height,
                target.format,
                clear.component_mask,
                clear.color
            ));
        }
        let mut state = clear.target;
        state.address = canonical(memory, state.address);
        let Some(key) = self.target(&state)? else {
            return Ok(());
        };
        let image = self.targets[&key];
        self.clear_later(PendingClear {
            image,
            aspect: vk::ImageAspectFlags::COLOR,
            value: vk::ClearValue {
                // Integer targets clear to the color's integer values.
                color: match formats::color_target_kind(key.format) {
                    AttributeType::Float => vk::ClearColorValue {
                        float32: clear.color,
                    },
                    AttributeType::UnsignedInt => vk::ClearColorValue {
                        uint32: clear.color.map(|value| value as u32),
                    },
                    AttributeType::SignedInt => vk::ClearColorValue {
                        int32: clear.color.map(|value| value as i32),
                    },
                },
            },
        });
        Ok(())
    }

    /// The translated program at `address`, cached. `constants` is the
    /// stage's constant buffer 1 (address and size), read when translating.
    fn shader(
        &mut self,
        memory: &dyn GpuMemory,
        address: u64,
        environment: &Environment,
        constants: Option<(u64, u32)>,
    ) -> Option<ShaderKey> {
        let code = |kind: AttributeType| match kind {
            AttributeType::Float => 0,
            AttributeType::SignedInt => 1,
            AttributeType::UnsignedInt => 2,
        };
        // Vertex attribute types, color target types, then the enabled
        // clip distances.
        let types: [u8; 41] = std::array::from_fn(|index| match index {
            0..32 => code(environment.vertex_attributes[index]),
            32..40 => code(environment.color_targets[index - 32]),
            _ => environment.clip_distances,
        });
        if let Some(found) = self.shader_keys.get(&(address, types)) {
            return *found;
        }
        let hash = code_hash(&memory.read_gpu(address, 0x200).unwrap_or_default());
        let key = (address, types, hash);
        if !self.shaders.contains_key(&key) {
            let prebuilt = self.shader_pack.as_ref().and_then(|pack| {
                pack.lookup(hash, &types).unwrap_or_else(|error| {
                    eprintln!("[recomp-vulkan] shader pack: {error}");
                    None
                })
            });
            if let Some(translated) = prebuilt {
                let shader = self.module(address, Ok(translated));
                self.shaders.insert(key, shader);
            } else {
                let mut program = Vec::new();
                while (program.len() as u64) < PROGRAM_BYTES {
                    match memory.read_gpu(address + program.len() as u64, 0x100) {
                        Some(chunk) => program.extend_from_slice(&chunk),
                        None => break,
                    }
                }
                let environment = Environment {
                    constant_buffer_1: constant_buffer_1(memory, constants),
                    ..environment.clone()
                };
                let started = std::time::Instant::now();
                let shader = self.module(address, translate::translate(&program, &environment));
                if let Some(limit) = slow_pipeline_limit() {
                    let elapsed = started.elapsed();
                    if elapsed.as_millis() >= limit {
                        println!(
                            "[recomp-vulkan] shader {address:#x} translated in {:.1} ms",
                            elapsed.as_secs_f64() * 1000.0
                        );
                    }
                }
                self.shaders.insert(key, shader);
            }
        }
        let found = self.shaders[&key].as_ref().map(|_| key);
        self.shader_keys.insert((address, types), found);
        found
    }

    /// The empty fragment program, translated on first use.
    fn empty_fragment(&mut self) -> Option<ShaderKey> {
        if !self.shaders.contains_key(&EMPTY_FRAGMENT) {
            let translated = translate::translate(
                &translate::empty_fragment_program(),
                &Environment::default(),
            );
            let shader = self.module(0, translated);
            self.shaders.insert(EMPTY_FRAGMENT, shader);
        }
        self.shaders[&EMPTY_FRAGMENT]
            .as_ref()
            .map(|_| EMPTY_FRAGMENT)
    }

    /// The shader module of a translation; failures are reported once.
    fn module(
        &mut self,
        address: u64,
        translation: Result<TranslatedShader, translate::TranslateError>,
    ) -> Option<Shader> {
        match translation {
            Ok(translated) => {
                if let Some(wanted) = shader_dump_address() {
                    if wanted == address {
                        let mut bytes = Vec::with_capacity(translated.words.len() * 4);
                        for word in &translated.words {
                            bytes.extend_from_slice(&word.to_le_bytes());
                        }
                        let dir = crate::paths::resolve("shader-dump");
                        let _ = std::fs::create_dir_all(&dir);
                        let _ = std::fs::write(dir.join(format!("shader-{address:x}.spv")), bytes);
                        println!("[recomp-vulkan] shader {address:#x} written to {dir:?}");
                    }
                }
                let info = vk::ShaderModuleCreateInfo::default().code(&translated.words);
                match unsafe { self.device.create_shader_module(&info, None) } {
                    Ok(module) => Some(Shader { module, translated }),
                    Err(result) => {
                        self.warn(format!("shader {address:#x}: module: {result}"));
                        None
                    }
                }
            }
            Err(failure) => {
                self.warn(format!("shader {address:#x}: {failure}"));
                None
            }
        }
    }

    /// A draw's indices in the stream: 16- and 32-bit ones as the guest
    /// wrote them, 8-bit ones widened (Vulkan has no 8-bit indices without
    /// an extension), quads as triangle pairs. None for a draw without
    /// indices that is not quads. The same guest indices go in once per
    /// submission. Vulkan restarts strips only at the all-ones index, so
    /// another restart index the guest set becomes that. The guest's
    /// restart index names no vertex while restarting is on; while it is
    /// off, all ones is a vertex too (the 256th of a mesh with 8-bit
    /// indices: leaving it out of the vertices placed drew it from
    /// whatever followed, as long streaks).
    fn stream_indices(
        &mut self,
        memory: &dyn GpuMemory,
        draw: &DrawCall,
    ) -> Result<Option<StreamIndices>, String> {
        let mut bytes = std::mem::take(&mut self.index_bytes);
        let mut values = std::mem::take(&mut self.index_values);
        let mut data = std::mem::take(&mut self.index_data);
        let mut triangles = std::mem::take(&mut self.index_triangles);
        let result = Self::stream_indices_inner(
            self,
            memory,
            draw,
            &mut bytes,
            &mut values,
            &mut data,
            &mut triangles,
        );
        self.index_bytes = bytes;
        self.index_values = values;
        self.index_data = data;
        self.index_triangles = triangles;
        result
    }

    fn stream_indices_inner(
        gpu: &mut Gpu,
        memory: &dyn GpuMemory,
        draw: &DrawCall,
        bytes: &mut Vec<u8>,
        values: &mut Vec<u32>,
        data: &mut Vec<u8>,
        triangles: &mut Vec<u32>,
    ) -> Result<Option<StreamIndices>, String> {
        bytes.clear();
        values.clear();
        data.clear();
        triangles.clear();
        let parameters = draw.parameters;
        let quads = parameters.topology == PrimitiveTopology::Quads
            || (parameters.topology == PrimitiveTopology::Patches && draw.patch_vertices == 4);
        let restart = restart_index(draw);
        let guest_restart = draw.primitive_restart;
        let key: Option<IndexKey>;
        let size: u64;
        if !draw.inline_indices.is_empty() {
            key = None;
            size = 4;
            values.extend_from_slice(&draw.inline_indices);
        } else if let Some(buffer) = draw.index_buffer {
            size = match buffer.index_type {
                recomp_gfx::maxwell::render::IndexType::U8 => 1,
                recomp_gfx::maxwell::render::IndexType::U16 => 2,
                recomp_gfx::maxwell::render::IndexType::U32 => 4,
            };
            let start = buffer.address + u64::from(parameters.first_index) * size;
            let len = u64::from(parameters.count) * size;
            let index_key = (start, len, size, quads, guest_restart, restart.is_some());
            if let Some(indices) = gpu.index_uploads.get(&index_key) {
                return Ok(Some(*indices));
            }
            // 16- and 32-bit indices go in as they are: scanned for their
            // bounds and copied straight from guest memory.
            if !quads && size >= 2 && restart.is_none_or(|index| index == all_ones(size as usize)) {
                let stream = &mut gpu.stream;
                let mut placed = None;
                let skip = guest_restart == Some(all_ones(size as usize));
                memory.read_gpu_with(start, len as usize, &mut |bytes| {
                    let (low, high) = index_bounds(bytes, size, skip);
                    placed = Some((stream.push(bytes, 4), low, high));
                });
                if let Some((offset, low, high)) = placed {
                    gpu.copied[1] += len;
                    let indices = StreamIndices {
                        offset: offset.ok_or("stream full")?,
                        count: (len / size) as u32,
                        kind: if size == 2 {
                            vk::IndexType::UINT16
                        } else {
                            vk::IndexType::UINT32
                        },
                        low,
                        high,
                    };
                    gpu.index_uploads.insert(index_key, indices);
                    return Ok(Some(indices));
                }
            }
            bytes.clear();
            let read = memory.read_gpu_with(start, len as usize, &mut |read| {
                bytes.extend_from_slice(read)
            });
            if !read {
                bytes.resize(len as usize, 0);
            }
            match size {
                1 => values.extend(bytes.iter().map(|byte| u32::from(*byte))),
                2 => values.extend(
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|index| u32::from(u16::from_le_bytes(*index))),
                ),
                _ => values.extend(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|index| u32::from_le_bytes(*index)),
                ),
            }
            key = Some(index_key);
        } else if quads {
            key = None;
            size = 4;
            values.extend(0..parameters.count);
        } else {
            return Ok(None);
        };
        // The guest's restart index, unless it is all ones of a 16- or
        // 32-bit index already, becomes all ones (which widened 8-bit
        // indices keep in 16 bits).
        let size = size as usize;
        let remapped = restart.filter(|index| size == 1 || *index != all_ones(size));
        if let Some(restart) = remapped {
            for index in values.iter_mut() {
                if *index == restart {
                    *index = u32::MAX;
                }
            }
        }
        // A restart value names no vertex.
        let skip = if remapped.is_some() {
            Some(u32::MAX)
        } else {
            guest_restart
        };
        let (low, high) = values
            .iter()
            .filter(|index| Some(**index) != skip)
            .fold((u32::MAX, 0), |(low, high), index| {
                (low.min(*index), high.max(*index))
            });
        let low = low.min(high);
        let (placed, kind, count): (&[u8], vk::IndexType, usize) = if quads {
            quads_to_triangles_into(values, triangles);
            for index in triangles.iter() {
                data.extend_from_slice(&index.to_le_bytes());
            }
            (data, vk::IndexType::UINT32, triangles.len())
        } else {
            match (size, remapped) {
                (1, _) => {
                    for index in values.iter() {
                        data.extend_from_slice(&(*index as u16).to_le_bytes());
                    }
                    (data, vk::IndexType::UINT16, values.len())
                }
                (_, Some(_)) => {
                    for index in values.iter() {
                        data.extend_from_slice(&index.to_le_bytes());
                    }
                    (data, vk::IndexType::UINT32, values.len())
                }
                (2, None) => (bytes, vk::IndexType::UINT16, values.len()),
                _ => (bytes, vk::IndexType::UINT32, values.len()),
            }
        };
        let copied = placed.len() as u64;
        let offset = gpu.stream.push(placed, 4).ok_or("stream full")?;
        gpu.copied[1] += copied;
        let indices = StreamIndices {
            offset,
            count: count as u32,
            kind,
            low,
            high,
        };
        if let Some(key) = key {
            gpu.index_uploads.insert(key, indices);
        }
        Ok(Some(indices))
    }

    /// `len` bytes of guest memory at `address` (zeros without one) for a
    /// vertex buffer binding, placed once per submission: from the vertex
    /// cache, or copied into the stream.
    fn vertex_range(
        &mut self,
        memory: &dyn GpuMemory,
        address: Option<u64>,
        len: u64,
    ) -> Result<(vk::Buffer, u64), String> {
        let key = (address.unwrap_or(0), len);
        if let Some(placed) = self.vertex_uploads.get(&key) {
            return Ok(*placed);
        }
        let placed = match address.and_then(|address| self.cached_vertices(memory, address, len)) {
            Some(placed) => placed,
            None => (self.stream.buffer, self.stream_copy(memory, address, len)?),
        };
        self.vertex_uploads.insert(key, placed);
        Ok(placed)
    }

    /// A vertex range from the vertex cache, uploaded if its bytes changed;
    /// `None` for ranges that keep changing, unreadable ones, and when the
    /// cache is full.
    fn cached_vertices(
        &mut self,
        memory: &dyn GpuMemory,
        address: u64,
        len: u64,
    ) -> Option<(vk::Buffer, u64)> {
        let cache = self.vertex_cache.as_mut()?;
        let key = (address, len);
        let (since, offset) = match cache.entries.get(&key) {
            Some(entry) if entry.changes >= VERTEX_CHANGES => return None,
            Some(entry) => (entry.checked, Some(entry.offset)),
            None => (None, None),
        };
        // The range's pages: byte ranges within it, by index.
        let first = address / GUEST_PAGE;
        let pages = ((address + len).div_ceil(GUEST_PAGE) - first) as usize;
        let page = |index: usize| {
            let start = ((first + index as u64) * GUEST_PAGE).max(address) - address;
            let end = ((first + index as u64 + 1) * GUEST_PAGE).min(address + len) - address;
            start as usize..end as usize
        };
        // The pages to hash again: those written since the last check when
        // writes are watched, else all. Watching starts before the bytes
        // are read, so a write made after that is seen next time.
        let mut stale = None;
        let checked = match since {
            _ if len < WATCHED_BYTES => None,
            Some(since) => {
                let mut written = Vec::new();
                let checked =
                    memory.written_since(address, len as usize, Some(since), &mut |at, bytes| {
                        let start = (address + at as u64) / GUEST_PAGE - first;
                        let end = (address + (at + bytes) as u64).div_ceil(GUEST_PAGE) - first;
                        written.extend(start as usize..(end as usize).min(pages));
                    });
                stale = checked.map(|_| written);
                checked
            }
            None => memory.written_since(address, len as usize, None, &mut |_, _| {}),
        };
        if let (Some(offset), Some(stale)) = (offset, &stale) {
            if stale.is_empty() {
                if let Some(entry) = cache.entries.get_mut(&key) {
                    entry.checked = checked;
                }
                self.copied[2] += len;
                return Some((cache.arena.buffer, offset));
            }
        }
        let mut placed = None;
        let mut copied = 0;
        memory.read_gpu_with(address, len as usize, &mut |bytes| {
            let hash = |index: usize| twox_hash::XxHash3_64::oneshot(&bytes[page(index)]);
            let changes = match cache.entries.get_mut(&key) {
                Some(entry) => {
                    let same = |index: usize| hash(index) == entry.pages[index];
                    let unchanged = match &stale {
                        Some(stale) => stale.iter().all(|index| same(*index)),
                        None => (0..pages).all(same),
                    };
                    if unchanged {
                        entry.checked = checked;
                        placed = Some(entry.offset);
                        return;
                    }
                    if entry.changes + 1 >= VERTEX_CHANGES {
                        entry.changes += 1;
                        return;
                    }
                    entry.changes + 1
                }
                None => 0,
            };
            placed = cache.arena.push(bytes, 16);
            if let Some(offset) = placed {
                copied = bytes.len() as u64;
                cache.entries.insert(
                    key,
                    CachedVertices {
                        pages: (0..pages).map(hash).collect(),
                        offset,
                        changes,
                        checked,
                    },
                );
            }
        });
        let buffer = cache.arena.buffer;
        if placed.is_some() {
            self.copied[0] += copied;
            self.copied[2] += len - copied;
        }
        placed.map(|offset| (buffer, offset))
    }

    /// `len` bytes of guest memory at `address` (zeros without one) in the
    /// stream.
    fn stream_copy(
        &mut self,
        memory: &dyn GpuMemory,
        address: Option<u64>,
        len: u64,
    ) -> Result<u64, String> {
        let mut pushed = None;
        if let Some(address) = address {
            let stream = &mut self.stream;
            memory.read_gpu_with(address, len as usize, &mut |bytes| {
                pushed = Some(stream.push(bytes, 16));
            });
        }
        let offset = match pushed {
            Some(offset) => offset,
            None => self.stream.push(&vec![0; len as usize], 16),
        }
        .ok_or("stream full")?;
        self.copied[0] += len;
        Ok(offset)
    }

    fn phase(&mut self, phase: DrawPhase, clock: &mut std::time::Instant) {
        let now = std::time::Instant::now();
        self.phases[phase as usize] += now - *clock;
        *clock = now;
    }

    /// Where draw recording spent its time since the last report, in ms:
    /// shaders, vertices, descriptors and pipeline, commands.
    pub fn take_copied(&mut self) -> [u64; 3] {
        std::mem::take(&mut self.copied)
    }

    pub fn take_phases(&mut self) -> [f64; 4] {
        std::mem::take(&mut self.phases).map(|time| time.as_secs_f64() * 1000.0)
    }

    /// Milliseconds `begin` waited for the GPU since the last call.
    pub fn take_gpu_waited(&mut self) -> f64 {
        std::mem::take(&mut self.gpu_waited).as_secs_f64() * 1000.0
    }

    /// Draw one guest draw; draws that use something not supported yet are
    /// skipped (and reported once).
    pub fn draw(&mut self, memory: &dyn GpuMemory, draw: &DrawCall) -> Result<(), String> {
        self.begin()?;
        let outcome = match self.record_draw(memory, draw) {
            Ok(true) => {
                self.draws += 1;
                self.note_draw(draw);
                "drawn".to_string()
            }
            Ok(false) => {
                self.skipped += 1;
                "skipped".to_string()
            }
            Err(message) => {
                self.skipped += 1;
                self.warn(message.clone());
                message
            }
        };
        if let Some(trace) = self.trace.as_mut() {
            let textures = std::mem::take(&mut self.traced_textures).join(", ");
            trace.push(format!(
                "{} textures [{textures}]",
                describe_draw(draw, &outcome)
            ));
        }
        Ok(())
    }

    fn record_draw(&mut self, memory: &dyn GpuMemory, draw: &DrawCall) -> Result<bool, String> {
        let mut clock = std::time::Instant::now();
        let parameters = draw.parameters;
        let Some(topology) = formats::topology(parameters.topology) else {
            self.warn(format!("topology {:?} not drawn yet", parameters.topology));
            return Ok(false);
        };
        let Some(vertex_address) = draw.vertex_shader() else {
            self.warn("draws without a vertex program skipped".to_string());
            return Ok(false);
        };
        if skipped_programs().contains(&draw.fragment_shader().unwrap_or(0))
            || is_subres_postprocess(draw)
            || (!skipped_kinds().is_empty() && draw_kinds(draw).iter().any(|kind| skips(kind)))
        {
            return Ok(false);
        }

        // Vertex attributes: the shader's inputs decide which are read.
        let mut environment = Environment {
            clip_distances: draw.clip_distances,
            ..Environment::default()
        };
        for (index, attribute) in draw.vertex_attributes.iter().enumerate() {
            environment.vertex_attributes[index] =
                match (VertexAttribute { raw: *attribute }).component_type() {
                    3 => AttributeType::SignedInt,
                    4 => AttributeType::UnsignedInt,
                    _ => AttributeType::Float,
                };
        }
        let constants = |stage| {
            draw.constant_buffer(stage, 1)
                .map(|binding| (binding.address, binding.size))
        };
        let Some(vertex_key) = self.shader(memory, vertex_address, &environment, constants(1))
        else {
            return Ok(false);
        };
        // Fragment outputs take each color target's component type.
        let mut fragment_environment = Environment::default();
        for buffer in 0..draw.color_target_count.min(8) {
            let target = &draw.color_targets[draw.color_target_index(buffer)];
            fragment_environment.color_targets[buffer] = formats::color_target_kind(target.format);
        }
        // Without a fragment program (depth-only passes) the empty one runs:
        // it writes no color, so every color target is masked off.
        let fragment_key = match draw.fragment_shader() {
            Some(address) => self.shader(memory, address, &fragment_environment, constants(5)),
            None => self.empty_fragment(),
        };
        let Some(fragment_key) = fragment_key else {
            return Ok(false);
        };
        self.phase(DrawPhase::Shaders, &mut clock);
        // Copied out so the shader map borrow ends before the draw's other
        // lookups, without allocating a vector per draw.
        let mut vertex_inputs = [0u32; 32];
        let vertex_input_count = self.shaders[&vertex_key]
            .as_ref()
            .map(|shader| {
                let inputs = &shader.translated.inputs;
                let count = inputs.len().min(vertex_inputs.len());
                vertex_inputs[..count].copy_from_slice(&inputs[..count]);
                count
            })
            .unwrap_or(0);
        let vertex_inputs = &vertex_inputs[..vertex_input_count];

        // Layered and 3D targets are drawn as their first slice; say which
        // draws aim at more than one.
        for buffer in 0..draw.color_target_count.min(8) {
            let target = &draw.color_targets[draw.color_target_index(buffer)];
            if target.depth > 1 {
                self.warn(format!(
                    "draw into a {}-slice target at {:#x} ({}x{}, format {:#x}, layout {:#x}, layer size {:#x}, base layer {}), vertex program {vertex_address:#x}",
                    target.depth,
                    target.address,
                    target.width_or_stride,
                    target.height,
                    target.format,
                    target.memory_layout,
                    target.layer_size,
                    target.base_layer,
                ));
            }
        }
        // Color targets, in draw buffer order: an unused attachment where a
        // disabled target precedes an enabled one (fragment outputs are
        // matched to attachments by position).
        let mut layout = [None::<(TargetKey, usize)>; 8];
        let mut layout_count = 0;
        let mut enabled_targets = 0;
        for buffer in 0..draw.color_target_count.min(8) {
            let attachment = draw.color_target_index(buffer);
            let target = &draw.color_targets[attachment];
            if !target.enabled() {
                let enabled_after = (buffer + 1..draw.color_target_count.min(8))
                    .any(|later| draw.color_targets[draw.color_target_index(later)].enabled());
                if enabled_after {
                    layout[layout_count] = None;
                    layout_count += 1;
                }
                continue;
            }
            let mut state = *target;
            state.address = canonical(memory, state.address);
            match self.target(&state)? {
                Some(key) => {
                    layout[layout_count] = Some((key, attachment));
                    layout_count += 1;
                    enabled_targets += 1;
                }
                None => return Ok(false),
            }
        }
        let layout = &layout[..layout_count];
        // The depth-stencil target, when enabled; depth-only draws (shadow
        // maps) have no color targets.
        let depth = if draw.depth_target_enable {
            self.depth_target(
                memory,
                draw.depth_target.address,
                draw.depth_target.format,
                draw.depth_target_size.width,
                draw.depth_target_size.height,
            )?
        } else {
            None
        };
        if enabled_targets == 0 && depth.is_none() {
            self.warn("draws with no color or depth target skipped".to_string());
            return Ok(false);
        }

        // Vertex input layout and data.
        // Constant attributes read a zero buffer. Its binding index must be
        // one the device accepts — `maxVertexInputBindings` is 16 on Adreno,
        // so 0..=15, and binding 16 fails pipeline creation with
        // VK_ERROR_UNKNOWN — and one this draw's real buffers do not use.
        let mut used_bindings = 0u32;
        for location in vertex_inputs {
            let attribute = VertexAttribute {
                raw: draw.vertex_attributes[*location as usize],
            };
            if !attribute.is_constant() {
                used_bindings |= 1u32 << (attribute.buffer_index() & 15);
            }
        }
        let zero_binding = (!used_bindings).trailing_zeros().min(15);
        if used_bindings & (1 << zero_binding) != 0 {
            self.warn(format!(
                "draw uses all 16 vertex buffers; a constant attribute shares binding {zero_binding}"
            ));
        }
        let mut bindings = [(0u32, 0u32); MAX_VERTEX_BINDINGS];
        let mut binding_count = 0;
        let mut attributes = [(0u32, 0u32, vk::Format::UNDEFINED, 0u32); 32];
        let mut attribute_count = 0;
        for location in vertex_inputs {
            let attribute = VertexAttribute {
                raw: draw.vertex_attributes[*location as usize],
            };
            if attribute.is_constant() {
                if !bindings[..binding_count]
                    .iter()
                    .any(|(binding, _)| *binding == zero_binding)
                {
                    bindings[binding_count] = (zero_binding, 0);
                    binding_count += 1;
                }
                attributes[attribute_count] =
                    (*location, zero_binding, vk::Format::R32G32B32A32_SFLOAT, 0);
                attribute_count += 1;
                continue;
            }
            let Some(format) =
                formats::vertex_format(attribute.format(), attribute.component_type())
            else {
                self.warn(format!(
                    "vertex attribute format {:#x}/{} not read yet",
                    attribute.format(),
                    attribute.component_type()
                ));
                return Ok(false);
            };
            let buffer = attribute.buffer_index() as u32;
            if !bindings[..binding_count]
                .iter()
                .any(|(binding, _)| *binding == buffer)
            {
                bindings[binding_count] = (buffer, draw.vertex_buffers[buffer as usize].stride());
                binding_count += 1;
            }
            attributes[attribute_count] = (*location, buffer, format, attribute.offset());
            attribute_count += 1;
        }
        let bindings = &bindings[..binding_count];
        let attributes = &attributes[..attribute_count];

        // Indices into the stream, and the range of vertices they read.
        let instances = parameters.instance_count.max(1);
        let indices = self.stream_indices(memory, draw)?;
        let first_vertex = u64::from(parameters.first_vertex);
        let (first_element, last_element) = match &indices {
            Some(indices) => (
                u64::from(indices.low) + first_vertex,
                u64::from(indices.high) + first_vertex,
            ),
            None => (
                first_vertex,
                first_vertex + u64::from(parameters.count.max(1)) - 1,
            ),
        };

        // Each buffer's elements from the first to the last the draw reads,
        // bound so that the first is vertex 0. The same range is placed once
        // per submission (passes draw the same meshes again).
        let mut vertex_buffers = [(0u32, (vk::Buffer::null(), 0u64)); MAX_VERTEX_BINDINGS];
        let mut vertex_buffer_count = 0;
        for (binding, stride) in bindings {
            if *binding == zero_binding {
                vertex_buffers[vertex_buffer_count] =
                    (*binding, self.vertex_range(memory, None, 16)?);
                vertex_buffer_count += 1;
                continue;
            }
            let buffer = draw.vertex_buffers[*binding as usize];
            let limit = draw.vertex_buffer_limits[*binding as usize];
            let stride = u64::from(*stride);
            let start = buffer.address + first_element * stride;
            // Past the last element's start: its attributes.
            let reach = attributes
                .iter()
                .filter(|attribute| attribute.1 == *binding)
                .map(|attribute| u64::from(attribute.3) + 16)
                .max()
                .unwrap_or(16);
            let end = if limit >= start {
                limit + 1
            } else {
                start + VERTEX_BYTES
            };
            let wanted = ((last_element - first_element) * stride + reach)
                .min(end - start)
                .min(VERTEX_BYTES);
            vertex_buffers[vertex_buffer_count] =
                (*binding, self.vertex_range(memory, Some(start), wanted)?);
            vertex_buffer_count += 1;
        }
        let vertex_buffers = &vertex_buffers[..vertex_buffer_count];
        self.phase(DrawPhase::Vertices, &mut clock);
        // Descriptor sets: constant buffers and textures of each stage.
        let (sets, scaled_textures) = self.descriptor_sets(
            memory,
            draw,
            &[(vertex_key, 1, VERTEX_SET), (fragment_key, 5, FRAGMENT_SET)],
        )?;

        // The pipeline for this shader pair and fixed state.
        // A Vulkan 1.1 render pass needs a real format for every attachment;
        // sparse slots get the scratch target's format (see `scratch_target`).
        let sparse = if self.render_passes.is_some() {
            vk::Format::R8G8B8A8_UNORM
        } else {
            vk::Format::UNDEFINED
        };
        let mut color_formats = [sparse; 8];
        for (format, slot) in color_formats.iter_mut().zip(layout) {
            *format = slot.map_or(sparse, |(key, _)| self.targets[&key].format);
        }
        let color_formats = &color_formats[..layout.len()];
        let depth_image = depth.map(|key| self.depths[&key]);
        let Some(pipeline) = self.pipeline(
            draw,
            vertex_key,
            fragment_key,
            topology,
            bindings,
            attributes,
            layout,
            color_formats,
            depth_image.map(|image| image.format),
        )?
        else {
            return Ok(false);
        };

        self.phase(DrawPhase::DescriptorsAndPipeline, &mut clock);
        // Record: the area every attachment covers.
        // Each target's own size: its image may be larger (see
        // `target_images`).
        let width = layout
            .iter()
            .flatten()
            .map(|(key, _)| key.width)
            .chain(depth.map(|key| key.width))
            .min()
            .unwrap_or(1);
        let height = layout
            .iter()
            .flatten()
            .map(|(key, _)| key.height.max(1))
            .chain(depth.map(|key| key.height.max(1)))
            .min()
            .unwrap_or(1);
        // The resolution scale the draw renders at: its attachments' (the
        // same unless a draw mixes a small target with a large one).
        let scale = layout
            .iter()
            .flatten()
            .map(|(key, _)| self.targets[key].scale)
            .chain(depth_image.map(|image| image.scale))
            .min()
            .unwrap_or(2);
        if layout
            .iter()
            .flatten()
            .map(|(key, _)| self.targets[key].scale)
            .chain(depth_image.map(|image| image.scale))
            .any(|other| other != scale)
        {
            self.warn(
                "draws into targets at different resolution scales render at the smallest"
                    .to_string(),
            );
        }
        let (area_width, area_height) = (scaled(width, scale), scaled(height, scale));
        // An unused attachment still needs a view on the render-pass path;
        // one scratch image per render-area size takes its writes.
        let scratch = if self.render_passes.is_some() && layout.iter().any(Option::is_none) {
            Some(self.scratch_target(area_width, area_height)?)
        } else {
            None
        };
        let mut attachments = [vk::RenderingAttachmentInfo::default(); 8];
        for (attachment, slot) in attachments.iter_mut().zip(layout) {
            *attachment = match slot {
                Some((key, _)) => vk::RenderingAttachmentInfo::default()
                    .image_view(self.targets[key].attachment)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE),
                None => match scratch {
                    Some(image) => vk::RenderingAttachmentInfo::default()
                        .image_view(image.attachment)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .store_op(vk::AttachmentStoreOp::DONT_CARE),
                    None => vk::RenderingAttachmentInfo::default(),
                },
            };
        }
        let attachments = &attachments[..layout.len()];
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: area_width,
                height: area_height,
            },
        };
        let depth_attachment = depth_image.map(|image| {
            vk::RenderingAttachmentInfo::default()
                .image_view(image.attachment)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
        });
        let mut rendering = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(attachments);
        if let (Some(attachment), Some(image)) = (depth_attachment.as_ref(), depth_image) {
            rendering = rendering.depth_attachment(attachment);
            if formats::has_stencil(image.format) {
                rendering = rendering.stencil_attachment(attachment);
            }
        }
        // In the guest's pixels, then at the scale.
        let viewport = viewport(draw, width, height);
        let scissor = scissor(draw, width, height);
        let factor = scale as f32 / 2.0;
        let host_viewport = vk::Viewport {
            x: viewport.x * factor,
            y: viewport.y * factor,
            width: viewport.width * factor,
            height: viewport.height * factor,
            ..viewport
        };
        let host_scissor = scaled_rect(scissor, scale);
        // Draws into the same attachments share one rendering instance. A
        // barrier comes first only when this draw reads an image written
        // since the last one, or starts writing again to one written in an
        // earlier instance (the GPU then runs draws back to back instead of
        // draining between every two).
        let mut instance = RenderingKey {
            colors: [vk::ImageView::null(); 8],
            depth: depth_image.map_or(vk::ImageView::null(), |image| image.attachment),
            width: area_width,
            height: area_height,
        };
        for (slot, attachment) in attachments.iter().enumerate().take(8) {
            instance.colors[slot] = attachment.image_view;
        }
        let mut written_images = [vk::Image::null(); 9];
        let mut written_count = 0;
        for (key, _) in layout.iter().flatten() {
            written_images[written_count] = self.targets[key].image;
            written_count += 1;
        }
        if let Some(image) = depth_image {
            written_images[written_count] = image.image;
            written_count += 1;
        }
        let written_images = &written_images[..written_count];
        // Clears waiting for this draw's attachments go in the instance it
        // begins; any other one is recorded first (the draw may read its
        // image).
        let stencil_ops = depth_image.is_some() && draw.stencil.is_some();
        let mut clears = Vec::new();
        if self.rendering.get() != Some(instance) {
            let stencil_kept = self.render_passes.is_none() || stencil_ops;
            self.pending_clears.borrow_mut().retain(|clear| {
                match instance.attachment(&clear.image) {
                    Some(slot) if Self::clears_in(clear, area_width, area_height, stencil_kept) => {
                        clears.push((slot, *clear));
                        false
                    }
                    _ => true,
                }
            });
        }
        self.flush_clears();
        let continuing = self.rendering.get() == Some(instance);
        let hazard = {
            let (written, read) = (self.written.borrow(), self.read.borrow());
            self.sampled.iter().any(|image| written.contains(image))
                || (!continuing
                    && written_images
                        .iter()
                        .any(|image| written.contains(image) || read.contains(image)))
        };
        if hazard {
            self.barrier();
        }
        if self.rendering.get() != Some(instance) {
            self.end_rendering();
            if let Some(cache) = &self.render_passes {
                // The pass that the pipeline names must be the pass the
                // draw begins: the same formats, a sparse slot's scratch
                // format included, and keyed by whether stencil is tested.
                // Leaving the sparse slots out began a pass with fewer
                // color attachments than the framebuffer's views: the depth
                // attachment became a color image, and the Adreno 650 read
                // a wild address and lost the device at LM3's title screen.
                cache.begin(
                    self.command,
                    &rendering,
                    color_formats,
                    depth_image.map(|image| image.format),
                    stencil_ops,
                );
            } else {
                unsafe { self.device.cmd_begin_rendering(self.command, &rendering) };
            }
            self.rendering.set(Some(instance));
            self.count(0, 1);
            self.clear_in_instance(&clears, area_width, area_height);
        }
        self.written
            .borrow_mut()
            .extend(written_images.iter().copied());
        self.written_since_submit
            .borrow_mut()
            .extend(written_images.iter().copied());
        self.read.borrow_mut().extend(self.sampled.iter().copied());
        self.sampled.clear();
        let push_constants = rescale_constants(scale, self.scale, scaled_textures);
        unsafe {
            let device = &self.device;
            let command = self.command;
            let state = &mut self.bound;
            if state.pipeline != pipeline {
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, pipeline);
                state.pipeline = pipeline;
            }
            // Every draw gets its own sets (constant buffer data is placed at
            // a new stream offset), so they are bound every time.
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &sets,
                &[],
            );
            for (binding, (buffer, offset)) in vertex_buffers {
                let slot = *binding as usize;
                let binding_state = (*buffer, *offset);
                if state.vertex_buffers[slot] != Some(binding_state) {
                    device.cmd_bind_vertex_buffers(command, *binding, &[*buffer], &[*offset]);
                    state.vertex_buffers[slot] = Some(binding_state);
                }
            }
            if state.push_constants != Some(push_constants) {
                device.cmd_push_constants(
                    command,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &push_constants,
                );
                state.push_constants = Some(push_constants);
            }
            if !state
                .viewport
                .is_some_and(|viewport| same_viewport(&viewport, &host_viewport))
            {
                device.cmd_set_viewport(command, 0, std::slice::from_ref(&host_viewport));
                state.viewport = Some(host_viewport);
            }
            if state.scissor != Some(host_scissor) {
                device.cmd_set_scissor(command, 0, std::slice::from_ref(&host_scissor));
                state.scissor = Some(host_scissor);
            }
            // The vertex buffers start at the first element read.
            match indices {
                Some(indices) => {
                    let bound = (self.stream.buffer, indices.offset, indices.kind);
                    if state.index_buffer != Some(bound) {
                        device.cmd_bind_index_buffer(
                            command,
                            self.stream.buffer,
                            indices.offset,
                            indices.kind,
                        );
                        state.index_buffer = Some(bound);
                    }
                    device.cmd_draw_indexed(
                        command,
                        indices.count,
                        instances,
                        0,
                        -(indices.low as i32),
                        parameters.first_instance,
                    );
                }
                None => device.cmd_draw(
                    command,
                    parameters.count,
                    instances,
                    0,
                    parameters.first_instance,
                ),
            }
        }
        self.phase(DrawPhase::Commands, &mut clock);

        // Remember how much of each target draws cover.
        let (left, right) = (
            viewport.x.min(viewport.x + viewport.width),
            viewport.x.max(viewport.x + viewport.width),
        );
        let (top, bottom) = (
            viewport.y.min(viewport.y + viewport.height),
            viewport.y.max(viewport.y + viewport.height),
        );
        if left < right && top < bottom {
            for (key, _) in layout.iter().flatten() {
                let extent = (
                    (right.ceil() as u32).min(key.width),
                    (bottom.ceil() as u32).min(key.height.max(1)),
                );
                let entry = self.drawn.entry(*key).or_insert((0, 0));
                *entry = (entry.0.max(extent.0), entry.1.max(extent.1));
            }
        }

        // Until the guest queues frames to VI (`present`), the window shows
        // the largest target drawn: its display images.
        let Some(&(drawn, _)) = layout.iter().flatten().next() else {
            return Ok(true);
        };
        let area = u64::from(drawn.width) * u64::from(drawn.height);
        if area >= self.display_area {
            self.display_area = area;
            self.presentable = Some(drawn);
        }
        Ok(true)
    }

    /// One descriptor set from the current pool, from a batch allocated at
    /// once (the pool is reset per submission; sets are never freed
    /// individually).
    fn descriptor_set(&mut self) -> Result<vk::DescriptorSet, String> {
        if let Some(set) = self.descriptor_batch.pop() {
            return Ok(set);
        }
        let count = descriptor_batch_count(self.descriptor_sets);
        if count == 0 {
            return Err("descriptor pool full".to_string());
        }
        let layouts = [self.set_layout; DESCRIPTOR_BATCH as usize];
        let mut sets = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts[..count as usize]),
            )
        }
        .map_err(error("descriptor sets"))?;
        self.descriptor_sets += count;
        let set = sets.swap_remove(0);
        self.descriptor_batch.extend(sets);
        Ok(set)
    }

    /// Allocate and fill one descriptor set per stage; with each set's
    /// texture bindings that are render targets at the resolution scale (a
    /// bit per binding, `translate::RESCALE_*`). The infos and writes are
    /// stack arrays: allocating vectors here showed up as allocator time in
    /// every draw.
    fn descriptor_sets(
        &mut self,
        memory: &dyn GpuMemory,
        draw: &DrawCall,
        stages: &[(ShaderKey, usize, u32)],
    ) -> Result<([vk::DescriptorSet; 2], [u32; 2]), String> {
        let mut scaled_textures = [0u32; 2];
        let sets = [self.descriptor_set()?, self.descriptor_set()?];
        let mut buffer_infos = [(
            vk::DescriptorSet::null(),
            0u32,
            vk::DescriptorBufferInfo::default(),
        ); MAX_BUFFER_WRITES];
        let mut image_infos = [(
            vk::DescriptorSet::null(),
            0u32,
            vk::DescriptorImageInfo::default(),
        ); MAX_IMAGE_WRITES];
        let mut buffer_count = 0;
        let mut image_count = 0;
        // The texture bindings of the stage being read; copied out so the
        // shader map borrow ends before the textures are looked up.
        let mut textures = [None::<TextureBinding>; TEXTURE_BINDINGS as usize];
        for (key, stage, set_index) in stages {
            let set = sets[*set_index as usize];
            let Some(shader) = self.shaders[key].as_ref() else {
                continue;
            };
            let translated = &shader.translated;
            // A binding the set layout does not have reads a descriptor no
            // one wrote (zeros on some GPUs, stale pool memory on others):
            // such a draw is skipped instead, with a warning.
            if let Some(binding) = translated
                .textures
                .iter()
                .map(|texture| texture.binding)
                .find(|binding| !texture_binding_fits(*binding))
            {
                return Err(format!(
                    "shader {key:?} samples {} textures (binding {binding}); the descriptor set has {TEXTURE_BINDINGS}",
                    translated.textures.len()
                ));
            }
            for (index, &slot) in translated.uniform_buffers.iter().enumerate() {
                let extent = translated.uniform_extents[index];
                // Only the bytes the program can read.
                let buffer = draw
                    .constant_buffer(*stage, slot as usize)
                    .map(|binding| (binding.address, binding.size.min(extent)));
                let info = self
                    .stream
                    .push_uniform(memory, buffer, self.uniform_alignment)?;
                if buffer_count < buffer_infos.len() {
                    buffer_infos[buffer_count] = (set, slot, info);
                    buffer_count += 1;
                }
            }
            let mut texture_count = 0;
            for &binding in &translated.textures {
                if texture_count < textures.len() {
                    textures[texture_count] = Some(binding);
                    texture_count += 1;
                }
            }
            let source = TextureSource {
                handles: draw
                    .constant_buffer(*stage, draw.texture_buffer_index as usize)
                    .map(|buffer| buffer.address),
                texture_pool: draw.texture_pool,
                sampler_pool: draw.sampler_pool,
                samplers_from_texture_index: draw.samplers_from_texture_index,
            };
            for binding in textures[..texture_count].iter().flatten() {
                let (view, sampler, scaled) = self.texture(memory, &source, binding)?;
                let bit = binding.binding - TEXTURE_BINDING_BASE;
                if scaled && bit < 32 {
                    scaled_textures[*set_index as usize] |= 1 << bit;
                }
                if image_count < image_infos.len() {
                    image_infos[image_count] = (
                        set,
                        binding.binding,
                        vk::DescriptorImageInfo {
                            sampler,
                            image_view: view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                    );
                    image_count += 1;
                }
            }
        }
        let mut writes = [vk::WriteDescriptorSet::default(); MAX_BUFFER_WRITES + MAX_IMAGE_WRITES];
        let mut write_count = 0;
        for (set, binding, info) in &buffer_infos[..buffer_count] {
            writes[write_count] = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(*binding)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(info));
            write_count += 1;
        }
        for (set, binding, info) in &image_infos[..image_count] {
            writes[write_count] = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(*binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(info));
            write_count += 1;
        }
        unsafe {
            self.device
                .update_descriptor_sets(&writes[..write_count], &[])
        };
        Ok((sets, scaled_textures))
    }

    /// The image view and sampler a texture binding samples, and whether it
    /// is a render target at the resolution scale.
    fn texture(
        &mut self,
        memory: &dyn GpuMemory,
        source: &TextureSource,
        binding: &TextureBinding,
    ) -> Result<(vk::ImageView, vk::Sampler, bool), String> {
        let handle = source
            .handles
            .and_then(|address| memory.read_u32(address + u64::from(binding.handle_offset)))
            .unwrap_or(0);
        let texture_index = handle & 0xF_FFFF;
        let sampler_index = if source.samplers_from_texture_index {
            texture_index
        } else {
            (handle >> 20) & 0xFFF
        };
        let header = pool_entry(
            memory,
            source.texture_pool.address + u64::from(texture_index) * 32,
        )
        .and_then(|bytes| TextureHeader::parse(&bytes));
        let sampler_header = pool_entry(
            memory,
            source.sampler_pool.address + u64::from(sampler_index) * 32,
        )
        .and_then(|bytes| SamplerHeader::parse(&bytes))
        .unwrap_or(SamplerHeader { words: [0; 8] });

        if self.trace.is_some() {
            self.traced_textures
                .push(header.map_or("none".to_string(), |header| {
                    let target = if self.target_at.contains_key(&header.address()) {
                        " target"
                    } else if self.depth_at.contains_key(&header.address()) {
                        " depth"
                    } else {
                        ""
                    };
                    let (base, last) = header.view_levels();
                    format!(
                        "{:#x} {}x{} fmt {:#x} levels {} view {base}-{last} words {:08x?}{target}",
                        header.address(),
                        header.width(),
                        header.height(),
                        header.format(),
                        header.levels(),
                        header.words
                    )
                }));
        }
        let (view, scaled) = match header {
            Some(header) => self.texture_view(memory, &header, binding.kind)?,
            None => (
                self.placeholder(binding.kind, UNSUPPORTED_COLOR)?.view,
                false,
            ),
        };
        let filterable = !self.unfilterable.borrow().contains(&view);
        let sampler = self.sampler(&sampler_header, binding.shadow, filterable)?;
        Ok((view, sampler, scaled))
    }

    /// A view of a 2D array whose layers the guest drew as targets of their
    /// own (at the addresses the header's layer stride puts them), or None
    /// when no layer was drawn. Each layer drawn is copied into a layer of
    /// one array image, again whenever its target was drawn since; layers
    /// never drawn read as cleared (depth 1, or zero).
    fn layered_view(
        &mut self,
        memory: &dyn GpuMemory,
        header: &TextureHeader,
    ) -> Result<Option<(vk::ImageView, bool)>, String> {
        let layers = header.array_layers().max(1);
        let Some(stride) = array_layer_stride(header) else {
            return Ok(None);
        };
        let (width, height) = (header.width().max(1), header.height().max(1));
        let key = (header.address(), width, height, layers);
        // The layers drawn, looked for again once targets were drawn at a
        // new address. Most arrays are plain textures, with none.
        let generation = self.target_generation;
        if self
            .layered
            .get(&key)
            .is_none_or(|layered| layered.generation != generation)
        {
            let sources: Vec<(u32, u64)> = (0..layers)
                .map(|layer| {
                    (
                        layer,
                        canonical(memory, header.address() + u64::from(layer) * stride),
                    )
                })
                .filter(|(_, address)| {
                    self.depth_at.contains_key(address) || self.target_at.contains_key(address)
                })
                .collect();
            let layered = self.layered.entry(key).or_insert(Layered {
                image: None,
                generation,
                sources: Vec::new(),
                copied: Vec::new(),
            });
            layered.copied = vec![(vk::Image::null(), u64::MAX); sources.len()];
            layered.sources = sources;
            layered.generation = generation;
        }
        // Each layer's image now: a target's address can name a target of
        // another size since (dynamic resolution).
        let images: Vec<Option<Image>> = self.layered[&key]
            .sources
            .iter()
            .map(|(_, address)| match self.depth_at.get(address) {
                Some(target) => self.depths.get(target).copied(),
                None => self
                    .target_at
                    .get(address)
                    .and_then(|target| self.targets.get(target))
                    .copied(),
            })
            .collect();
        let array = match self.layered[&key].image {
            Some(image) => image,
            None => {
                // Made at the first layer's format and scale.
                let Some(first) = images.iter().flatten().next().copied() else {
                    return Ok(None);
                };
                let image = Image {
                    scale: first.scale,
                    ..self.create_image(
                        first.format,
                        scaled(width, first.scale),
                        scaled(height, first.scale),
                        layers,
                        vk::ImageViewType::TYPE_2D_ARRAY,
                        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                        vk::ComponentMapping::default(),
                    )?
                };
                self.clear_image(image, layers);
                self.layered.get_mut(&key).expect("found above").image = Some(image);
                image
            }
        };
        // Copy in the layers drawn since they were last copied.
        let mut copies = Vec::new();
        for (index, source) in images.iter().enumerate() {
            let Some(source) = source else {
                continue;
            };
            let writes = self.writes.get(&source.image).copied().unwrap_or(0);
            let fits = source.format == array.format
                && source.scale == array.scale
                && source.width >= array.width
                && source.height >= array.height;
            if fits && self.layered[&key].copied[index] != (source.image, writes) {
                let layer = self.layered[&key].sources[index].0;
                copies.push((index, layer, *source, writes));
            }
        }
        if !copies.is_empty() {
            self.barrier();
            let aspect_mask = formats::aspects(array.format);
            for (_, layer, source, _) in &copies {
                let subresource = |base_array_layer| vk::ImageSubresourceLayers {
                    aspect_mask,
                    mip_level: 0,
                    base_array_layer,
                    layer_count: 1,
                };
                let region = vk::ImageCopy {
                    src_subresource: subresource(0),
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: subresource(*layer),
                    dst_offset: vk::Offset3D::default(),
                    extent: vk::Extent3D {
                        width: array.width,
                        height: array.height,
                        depth: 1,
                    },
                };
                unsafe {
                    self.device.cmd_copy_image(
                        self.command,
                        source.image,
                        vk::ImageLayout::GENERAL,
                        array.image,
                        vk::ImageLayout::GENERAL,
                        std::slice::from_ref(&region),
                    );
                }
            }
            self.barrier();
            let layered = self.layered.get_mut(&key).expect("found above");
            for (index, _, source, writes) in copies {
                layered.copied[index] = (source.image, writes);
            }
        }
        Ok(Some((array.view, array.scale != 2)))
    }

    /// Clear every layer of a new image: depth to 1 (and stencil to 0), or
    /// color to zero.
    fn clear_image(&mut self, image: Image, layers: u32) {
        self.barrier();
        let aspect_mask = formats::aspects(image.format);
        let range = vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: layers,
        };
        unsafe {
            if aspect_mask.contains(vk::ImageAspectFlags::DEPTH) {
                self.device.cmd_clear_depth_stencil_image(
                    self.command,
                    image.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                    std::slice::from_ref(&range),
                );
            } else {
                self.device.cmd_clear_color_image(
                    self.command,
                    image.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue::default(),
                    std::slice::from_ref(&range),
                );
            }
        }
        self.barrier();
    }

    /// A view of a texture header's image: a render target at its address,
    /// the texture prepared from guest memory, or a placeholder while it is
    /// prepared (or when its format is not decoded yet).
    fn texture_view(
        &mut self,
        memory: &dyn GpuMemory,
        header: &TextureHeader,
        kind: TextureKind,
    ) -> Result<(vk::ImageView, bool), String> {
        // The texture kinds a shader samples and the header kinds it can
        // name: 1D tables, 2D images, arrays and cube maps are uploaded.
        let supported = texture_target_supported(kind, header.target());
        if !supported {
            self.warn(format!(
                "{kind:?} textures ({:?} headers, format {:#x}) not uploaded yet",
                header.target(),
                header.format()
            ));
            return Ok((self.placeholder(kind, UNSUPPORTED_COLOR)?.view, false));
        }
        // Render targets have 2D views: they stand in for 2D textures only.
        // A guest address can hold both a color target and a depth target
        // at different times; the header's format says which one a depth
        // texture samples. LM3 keeps a light's shadow map and its filtered
        // light mask at one address: taking the color target for the Z16
        // header sampled the mask, a disc with no occluders, and the window
        // frame cast no shadow.
        let address = canonical(memory, header.address());
        let rendered = if kind == TextureKind::D2 {
            self.target_at
                .get(&address)
                .and_then(|key| self.targets.get(key))
                .copied()
        } else {
            None
        };
        let depth = if kind == TextureKind::D2 {
            self.depth_at
                .get(&address)
                .and_then(|key| self.depths.get(key))
                .copied()
        } else {
            None
        };
        let depth_texture = formats::depth_texel_bytes(header.format()).is_some();
        let image = if depth_texture {
            depth.or(rendered)
        } else {
            rendered.or(depth)
        };
        if let Some(image) = image {
            let view = self.sampled_view(image, header.width(), header.height())?;
            return Ok((view, image.scale != 2));
        }
        if kind == TextureKind::D2Array {
            if let Some(found) = self.layered_view(memory, header)? {
                return Ok(found);
            }
        }
        let cached = self.textures.get(&header.words).copied();
        let mut state = match cached {
            Some(state) if !self.rewritten(memory, header.words) => state,
            _ => {
                // Written again: draws sample what it held until the new
                // bytes are ready, unless they are quick to prepare here.
                let old = match cached {
                    Some(TextureState::Ready(image) | TextureState::Preview(image, _)) => {
                        Some(image)
                    }
                    _ => None,
                };
                let state = match (self.prepare_texture(memory, header, kind), old) {
                    (TextureState::Pending(pending), Some(image)) => {
                        TextureState::Preview(image, pending)
                    }
                    (state, old) => {
                        self.retired.extend(old);
                        state
                    }
                };
                self.textures.insert(header.words, state);
                if old.is_some() {
                    self.take_quick(header.words);
                    self.upload_arrived(header.words)?;
                }
                self.textures[&header.words]
            }
        };
        if matches!(state, TextureState::Pending(_)) {
            self.wait_for_texture(header.words)?;
            state = self.textures[&header.words];
        }
        let view = match state {
            TextureState::Ready(image) | TextureState::Preview(image, _) => image.view,
            TextureState::Pending(_) => self.placeholder(kind, PENDING_COLOR)?.view,
            TextureState::Unsupported => self.placeholder(kind, UNSUPPORTED_COLOR)?.view,
        };
        Ok((view, false))
    }

    /// Read level 0 of a 2D texture and hand it to the decoder.
    fn prepare_texture(
        &mut self,
        memory: &dyn GpuMemory,
        header: &TextureHeader,
        kind: TextureKind,
    ) -> TextureState {
        let (format, block_width, block_height, block_bytes, decode) =
            if let Some(layout) = formats::texture_format(header.format(), header.srgb()) {
                // BCn the device cannot sample (mobile GPUs) is decoded here.
                let decoded = Bcn::of(layout.format).filter(|_| {
                    !format_features(
                        &self.format_features,
                        &self.instance,
                        self.physical_device,
                        layout.format,
                    )
                    .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
                });
                let (format, decode) = match decoded {
                    Some((kind, plain)) => (plain, Decode::Bcn(kind)),
                    None => (layout.format, Decode::None),
                };
                (format, layout.block, layout.block, layout.bytes, decode)
            } else if let Some((width, height)) = formats::astc_footprint(header.format()) {
                let format = if header.srgb() {
                    vk::Format::R8G8B8A8_SRGB
                } else {
                    vk::Format::R8G8B8A8_UNORM
                };
                (format, width, height, 16, Decode::Astc(width, height))
            } else {
                self.warn(format!(
                    "texture format {:#x} not decoded yet",
                    header.format()
                ));
                return TextureState::Unsupported;
            };
        let (width, mut height) = (header.width(), header.height());
        if kind == TextureKind::D1 || matches!(header.target(), Some(Target::D1 | Target::D1Array))
        {
            // A 1D image has one row, whatever the header's height says.
            height = 1;
        }
        // Some effects are sampled with a cube instruction but bound as a
        // plain 2D image (a 32x32 matcap in LM3's ghost and gem shaders).
        // Vulkan needs six faces for a cube view, so repeat the single face
        // when the header has no array to take them from.
        let cube_from_2d = kind == TextureKind::Cube
            && matches!(
                header.target(),
                Some(Target::D2 | Target::D2NoMipmap | Target::D2Array)
            );
        let source_layers = header.array_layers().max(1);
        // LM3 samples a 2D texture with a 3D instruction on its save screen;
        // upload it as one slice of a 3D image, so the sample reads z = 0.
        let one_slice = kind == TextureKind::D3
            && matches!(header.target(), Some(Target::D2 | Target::D2NoMipmap));
        let (view_type, layers) = if one_slice {
            (vk::ImageViewType::TYPE_3D, 1)
        } else if kind == TextureKind::D1 {
            match header.target() {
                Some(Target::D1Array) => (vk::ImageViewType::TYPE_1D_ARRAY, header.array_layers()),
                _ => (vk::ImageViewType::TYPE_1D, 1),
            }
        } else if matches!(header.target(), Some(Target::D1)) {
            (vk::ImageViewType::TYPE_2D, 1)
        } else if matches!(header.target(), Some(Target::D1Array)) {
            (vk::ImageViewType::TYPE_2D_ARRAY, header.array_layers())
        } else if cube_from_2d {
            (vk::ImageViewType::CUBE, 6)
        } else {
            match header.target() {
                Some(Target::D2Array) => (vk::ImageViewType::TYPE_2D_ARRAY, header.array_layers()),
                Some(Target::Cube) => (vk::ImageViewType::CUBE, 6),
                Some(Target::CubeArray) => (vk::ImageViewType::CUBE_ARRAY, header.array_layers()),
                Some(Target::D3) => (vk::ImageViewType::TYPE_3D, 1),
                _ => (vk::ImageViewType::TYPE_2D, 1),
            }
        };
        // A 3D texture is one layer of `depth` slices.
        let depth = if one_slice {
            1
        } else if view_type == vk::ImageViewType::TYPE_3D {
            header.depth()
        } else {
            1
        };
        let mut gobs_in_z = 1;
        let rows = u64::from(height.div_ceil(block_height));
        let row_bytes = u64::from(width.div_ceil(block_width)) * u64::from(block_bytes);
        let layout = header.layout();
        let mut layer_stride = 0;
        let mut mips = vec![Mip {
            offset: 0,
            row_bytes,
            rows,
            gobs_in_y: 1,
            width,
            height,
        }];
        let source = match header.layout() {
            Layout::BlockLinear { gobs_in_y } => {
                // Small surfaces use shorter blocks than the header names;
                // layers follow each other at whole-block strides.
                let fitted = texture::fitted_gobs_in_y(rows, u64::from(gobs_in_y));
                mips[0].gobs_in_y = fitted as u32;
                gobs_in_z =
                    texture::fitted_gobs_in_z(u64::from(depth), u64::from(header.gobs_in_z()));
                let mut level_size =
                    texture::Volume::new(row_bytes, rows, fitted, gobs_in_z).size(u64::from(depth));
                // Every mip level of 2D, array and cube textures (3D ones
                // keep level 0): without them, detailed textures far away
                // sparkle as they move.
                let levels = if depth == 1 {
                    header.levels().min(32 - width.max(height).leading_zeros())
                } else {
                    1
                };
                if levels > 1 {
                    let rows_at =
                        |level: u32| u64::from((height >> level).max(1).div_ceil(block_height));
                    let row_bytes_at = |level: u32| {
                        u64::from((width >> level).max(1).div_ceil(block_width))
                            * u64::from(block_bytes)
                    };
                    let (starts, end) =
                        texture::mip_levels(levels, u64::from(gobs_in_y), rows_at, row_bytes_at);
                    mips = starts
                        .iter()
                        .enumerate()
                        .map(|(level, &(offset, gobs))| {
                            let level = level as u32;
                            Mip {
                                offset,
                                row_bytes: row_bytes_at(level),
                                rows: rows_at(level),
                                gobs_in_y: gobs as u32,
                                width: (width >> level).max(1),
                                height: (height >> level).max(1),
                            }
                        })
                        .collect();
                    level_size = end;
                }
                layer_stride = if cube_from_2d && source_layers < 6 {
                    0
                } else if layers > 1 {
                    texture::layer_stride(
                        header.levels(),
                        u64::from(gobs_in_y),
                        |level| u64::from((height >> level).max(1).div_ceil(block_height)),
                        |level| {
                            u64::from((width >> level).max(1).div_ceil(block_width))
                                * u64::from(block_bytes)
                        },
                    )
                } else {
                    level_size
                };
                let size = layer_stride * u64::from(layers - 1) + level_size;
                self.watch_texture(memory, header.words, header.address(), size, depth > 1);
                memory.read_gpu(header.address(), size as usize)
            }
            Layout::Pitch { .. } if depth > 1 => {
                self.warn("pitch-linear 3D textures not uploaded yet".to_string());
                None
            }
            Layout::Pitch { stride } => {
                let size = u64::from(stride) * rows.saturating_sub(1) + row_bytes;
                self.watch_texture(memory, header.words, header.address(), size, depth > 1);
                let mut texels = Vec::with_capacity((row_bytes * rows) as usize);
                for row in 0..rows {
                    let line = memory
                        .read_gpu(
                            header.address() + row * u64::from(stride),
                            row_bytes as usize,
                        )
                        .unwrap_or_else(|| vec![0; row_bytes as usize]);
                    texels.extend_from_slice(&line);
                }
                Some(texels)
            }
        };
        let Some(source) = source else {
            return TextureState::Unsupported;
        };
        // Volume textures are few (lookup tables); say what each holds, the
        // first time: some are written again every frame.
        if depth > 1
            && self
                .warned
                .insert(format!("3D texture at {:#x}", header.address()))
        {
            let written = source.iter().filter(|byte| **byte != 0).count();
            println!(
                "[recomp-vulkan] 3D texture at {:#x}: {width}x{height}x{depth}, format {:#x}, {written} of {} bytes nonzero",
                header.address(),
                header.format(),
                source.len()
            );
        }
        let swizzle = header.swizzle();
        let components = vk::ComponentMapping {
            r: formats::swizzle(swizzle[0]),
            g: formats::swizzle(swizzle[1]),
            b: formats::swizzle(swizzle[2]),
            a: formats::swizzle(swizzle[3]),
        };
        let levels = mips.len() as u32;
        let (first, last) = header.view_levels();
        let first = first.min(levels - 1);
        let view = (first, last.clamp(first, levels - 1) - first + 1);
        let source: std::sync::Arc<[u8]> = source.into();
        self.texture_generation += 1;
        let generation = self.texture_generation;
        let job = |mips: Vec<Mip>, first_level: u32| Job {
            words: header.words,
            generation,
            source: std::sync::Arc::clone(&source),
            layout,
            mips,
            first_level,
            decode,
            layers,
            layer_stride,
            depth,
            gobs_in_z: gobs_in_z as u32,
        };
        let preview = preview_level(width, height, depth, levels, view);
        // Previews, and textures without one, go before whole large ones.
        if let Some(level) = preview {
            self.decoder
                .submit(job(mips[level as usize..].to_vec(), level), true);
        }
        self.decoder.submit(job(mips, 0), preview.is_none());
        TextureState::Pending(PendingTexture {
            format,
            components,
            width,
            height,
            // `create_image` takes a 3D image's depth here.
            layers: layers.max(depth),
            view_type,
            levels,
            view,
            generation,
        })
    }

    /// Copy tightly packed texels to `image`: `layers` array layers, or
    /// for a 3D image that many slices, of each mip level (`levels` gives
    /// where each starts after `offset`).
    #[allow(clippy::too_many_arguments)]
    fn copy_to_image(
        &self,
        offset: u64,
        image: &Image,
        width: u32,
        height: u32,
        layers: u32,
        view_type: vk::ImageViewType,
        levels: &[u64],
    ) {
        let (layers, depth) = if view_type == vk::ImageViewType::TYPE_3D {
            (1, layers)
        } else {
            (layers, 1)
        };
        let regions: Vec<vk::BufferImageCopy> = levels
            .iter()
            .enumerate()
            .map(|(level, start)| vk::BufferImageCopy {
                buffer_offset: offset + start,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: level as u32,
                    base_array_layer: 0,
                    layer_count: layers,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: (width >> level).max(1),
                    height: (height >> level).max(1),
                    depth,
                },
            })
            .collect();
        self.barrier();
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.command,
                self.stream.buffer,
                image.image,
                vk::ImageLayout::GENERAL,
                &regions,
            );
        }
        // Draws sample it next.
        self.barrier();
    }

    /// A 1x1 image of `kind` and `color`, sampled in place of a texture.
    fn placeholder(&mut self, kind: TextureKind, color: [u8; 4]) -> Result<Image, String> {
        if let Some(image) = self.placeholders.get(&(kind, color)) {
            return Ok(*image);
        }
        let (view_type, layers) = match kind {
            TextureKind::D1 => (vk::ImageViewType::TYPE_1D, 1),
            TextureKind::D2 => (vk::ImageViewType::TYPE_2D, 1),
            TextureKind::D2Array => (vk::ImageViewType::TYPE_2D_ARRAY, 1),
            TextureKind::D3 => (vk::ImageViewType::TYPE_3D, 1),
            TextureKind::Cube => (vk::ImageViewType::CUBE, 6),
        };
        let image = self.create_image(
            vk::Format::R8G8B8A8_UNORM,
            1,
            1,
            layers,
            view_type,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ComponentMapping::default(),
        )?;
        let texel: Vec<u8> = color.repeat(layers as usize);
        let offset = self.stream.push(&texel, 16).ok_or("stream full")?;
        let region = vk::BufferImageCopy {
            buffer_offset: offset,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: layers,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        };
        self.barrier();
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.command,
                self.stream.buffer,
                image.image,
                vk::ImageLayout::GENERAL,
                std::slice::from_ref(&region),
            );
        }
        // Draws sample it next.
        self.barrier();
        self.placeholders.insert((kind, color), image);
        Ok(image)
    }

    /// The sampler of a header; `shadow` samplers compare (with the
    /// header's function, or less-or-equal), others never do, as Vulkan
    /// requires of the instructions that use them. One for a view the
    /// device cannot filter (not `filterable`) only filters linearly when
    /// it compares, which Vulkan allows for any depth format.
    fn sampler(
        &mut self,
        header: &SamplerHeader,
        shadow: bool,
        filterable: bool,
    ) -> Result<vk::Sampler, String> {
        let key = (header.words, shadow, filterable || shadow);
        if let Some(sampler) = self.samplers.get(&key) {
            return Ok(*sampler);
        }
        let [u, v, w] = header.address_modes();
        let (mag_linear, min_linear) = header.linear();
        let linear_allowed = key.2;
        let (mag_linear, min_linear) = (mag_linear && linear_allowed, min_linear && linear_allowed);
        let filter = |linear: bool| {
            if linear {
                vk::Filter::LINEAR
            } else {
                vk::Filter::NEAREST
            }
        };
        let (min_lod, mut max_lod) = header.lod_range();
        // Without a mip filter only the base level is sampled.
        if header.mip_filter() == 1 {
            max_lod = min_lod;
        }
        let anisotropy = if linear_allowed {
            header
                .max_anisotropy()
                .max(self.anisotropy)
                .min(self.max_anisotropy)
        } else {
            1.0
        };
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(filter(mag_linear))
            .min_filter(filter(min_linear))
            .mipmap_mode(if header.mip_filter() == 3 && linear_allowed {
                vk::SamplerMipmapMode::LINEAR
            } else {
                vk::SamplerMipmapMode::NEAREST
            })
            .address_mode_u(formats::address_mode(u))
            .address_mode_v(formats::address_mode(v))
            .address_mode_w(formats::address_mode(w))
            .min_lod(min_lod)
            .max_lod(max_lod.max(min_lod))
            .anisotropy_enable(anisotropy > 1.0)
            .max_anisotropy(anisotropy)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK);
        if shadow {
            let compare = header.depth_compare().unwrap_or(3);
            info = info
                .compare_enable(true)
                .compare_op(vk::CompareOp::from_raw(compare as i32));
        }
        let sampler =
            unsafe { self.device.create_sampler(&info, None) }.map_err(error("sampler"))?;
        self.samplers.insert(key, sampler);
        Ok(sampler)
    }

    #[allow(clippy::too_many_arguments)]
    fn pipeline(
        &mut self,
        draw: &DrawCall,
        vertex_key: ShaderKey,
        fragment_key: ShaderKey,
        topology: vk::PrimitiveTopology,
        bindings: &[(u32, u32)],
        attributes: &[(u32, u32, vk::Format, u32)],
        targets: &[Option<(TargetKey, usize)>],
        color_formats: &[vk::Format],
        depth_format: Option<vk::Format>,
    ) -> Result<Option<vk::Pipeline>, String> {
        // Per attachment: blend enable, equation and write mask (none for an
        // unused attachment). A fixed array: at most eight color targets.
        let mut blends = [(false, draw.blend_equations[0], 0u32); 8];
        for (blend, (buffer, slot)) in blends.iter_mut().zip(targets.iter().enumerate()) {
            let Some((_, attachment)) = slot else {
                continue;
            };
            let written = self.shaders[&fragment_key]
                .as_ref()
                .is_some_and(|shader| shader.translated.color_outputs.contains(&(buffer as u32)));
            let mask = if written {
                draw.color_masks[*attachment]
            } else {
                0
            };
            *blend = (
                draw.blend_enables[*attachment],
                draw.blend_equations[*attachment],
                mask,
            );
        }

        let restart = restart_index(draw).is_some();
        let key = &mut self.pipeline_key;
        key.clear();
        key.extend([
            vertex_key.0,
            vertex_key.2,
            fragment_key.0,
            fragment_key.2,
            topology.as_raw() as u64,
            u64::from(draw.depth_mode),
            depth_format.map_or(0, |format| format.as_raw() as u64),
            u64::from(draw.depth_test_enable)
                | (u64::from(draw.depth_write_enable) << 1)
                | (u64::from(draw.depth_test_func) << 8),
        ]);
        let mut cull = cull_mode(draw.cull_enable, draw.cull_face);
        let mut front = front_face(draw.front_face, draw.y_control);
        // Diagnostics for a title with an inverted face order or a culled
        // floor: `RECOMP_VULKAN_CULL=off|invert-front|swap-face`.
        match std::env::var("RECOMP_VULKAN_CULL").as_deref() {
            Ok("off") => cull = vk::CullModeFlags::NONE,
            Ok("invert-front") => {
                front = match front {
                    vk::FrontFace::CLOCKWISE => vk::FrontFace::COUNTER_CLOCKWISE,
                    _ => vk::FrontFace::CLOCKWISE,
                }
            }
            Ok("swap-face") => {
                cull = match cull {
                    vk::CullModeFlags::FRONT => vk::CullModeFlags::BACK,
                    vk::CullModeFlags::BACK => vk::CullModeFlags::FRONT,
                    other => other,
                }
            }
            _ => {}
        }
        key.push(
            u64::from(cull.as_raw())
                | (u64::from(front.as_raw() as u32) << 8)
                | (u64::from(restart) << 16),
        );
        match draw.depth_bias {
            Some((factor, units, clamp)) => key.extend([
                1,
                u64::from(factor.to_bits()),
                u64::from(units.to_bits()),
                u64::from(clamp.to_bits()),
            ]),
            None => key.extend([0, 0, 0, 0]),
        }
        key.extend(vertex_key.1.iter().map(|value| u64::from(*value)));
        if let Some((front, back)) = draw.stencil {
            for face in [front, back] {
                key.extend([
                    1 << 63,
                    u64::from(face.fail) | (u64::from(face.depth_fail) << 32),
                    u64::from(face.pass) | (u64::from(face.func) << 32),
                    u64::from(face.reference)
                        | (u64::from(face.compare_mask) << 16)
                        | (u64::from(face.write_mask) << 32),
                ]);
            }
        }
        for (binding, stride) in bindings {
            key.push((u64::from(*binding) << 32) | u64::from(*stride));
        }
        for (location, binding, format, offset) in attributes {
            key.push(
                (u64::from(*location) << 48)
                    | (u64::from(*binding) << 40)
                    | (u64::from(*offset) << 8),
            );
            key.push(format.as_raw() as u64);
        }
        for format in color_formats {
            key.push(format.as_raw() as u64 | (1 << 40));
        }
        for (enable, equation, mask) in &blends[..targets.len()] {
            key.push(u64::from(*enable) | (u64::from(*mask) << 8));
            key.extend([
                u64::from(equation.color_op),
                u64::from(equation.color_src),
                u64::from(equation.color_dst),
                u64::from(equation.alpha_op),
                u64::from(equation.alpha_src),
                u64::from(equation.alpha_dst),
                u64::from(equation.separate_alpha),
            ]);
        }
        // Most draws repeat the pipeline of the draw before: that needs no
        // hash lookup. The key vector is reused, so a miss does not allocate
        // one per draw either.
        if let Some((last_key, pipeline)) = &self.pipeline_last {
            if last_key.as_slice() == self.pipeline_key.as_slice() {
                return Ok(*pipeline);
            }
        }
        if let Some(pipeline) = self.pipelines.get(self.pipeline_key.as_slice()).copied() {
            if let Some((last_key, last_pipeline)) = &mut self.pipeline_last {
                last_key.clone_from(&self.pipeline_key);
                *last_pipeline = pipeline;
            } else {
                self.pipeline_last = Some((self.pipeline_key.clone(), pipeline));
            }
            return Ok(pipeline);
        }

        // A shader that samples a texture binding the descriptor set layout
        // does not declare makes an invalid pipeline. AMDVLK dereferences the
        // missing layout entry and crashes in vkCreateGraphicsPipelines, so
        // skip the draw before that call instead.
        let out_of_range = [(vertex_key, "vertex"), (fragment_key, "fragment")]
            .into_iter()
            .find_map(|(key, stage)| {
                let shader = self.shaders.get(&key)?.as_ref()?;
                let binding = shader
                    .translated
                    .textures
                    .iter()
                    .find(|binding| !texture_binding_fits(binding.binding))?;
                Some((key.0, stage, binding.binding))
            });
        if let Some((address, stage, binding)) = out_of_range {
            self.warn(format!(
                "pipeline for {stage} shader {address:#x} skipped: texture binding {binding} is outside the {TEXTURE_BINDINGS}-binding descriptor set"
            ));
            return Ok(None);
        }

        // Build (or fail) this pipeline: the key leaves the reusable buffer
        // and becomes the map key and the last-lookup key.
        let key = std::mem::take(&mut self.pipeline_key);
        let (Some(vertex), Some(fragment)) = (
            self.shaders[&vertex_key].as_ref(),
            self.shaders[&fragment_key].as_ref(),
        ) else {
            return Ok(None);
        };
        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex.module)
                .name(entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment.module)
                .name(entry),
        ];
        let mut binding_descriptions =
            [vk::VertexInputBindingDescription::default(); MAX_VERTEX_BINDINGS];
        for (description, (binding, stride)) in binding_descriptions.iter_mut().zip(bindings) {
            *description = vk::VertexInputBindingDescription {
                binding: *binding,
                stride: *stride,
                input_rate: vk::VertexInputRate::VERTEX,
            };
        }
        let binding_descriptions = &binding_descriptions[..bindings.len()];
        let mut attribute_descriptions = [vk::VertexInputAttributeDescription::default(); 32];
        for (description, (location, binding, format, offset)) in
            attribute_descriptions.iter_mut().zip(attributes)
        {
            *description = vk::VertexInputAttributeDescription {
                location: *location,
                binding: *binding,
                format: *format,
                offset: *offset,
            };
        }
        let attribute_descriptions = &attribute_descriptions[..attributes.len()];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(attribute_descriptions);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(topology)
            .primitive_restart_enable(restart);
        let mut depth_clip = vk::PipelineViewportDepthClipControlCreateInfoEXT::default()
            .negative_one_to_one(draw.depth_mode == 0);
        let mut viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        if self.depth_clip_control {
            viewport_state = viewport_state.push_next(&mut depth_clip);
        }
        // Without depth clip control, -1..1 depth would be clipped at 0.
        let mut rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            // `depthClamp` is a device feature: enabling it without asking
            // for it makes Adreno refuse the pipeline (VK_ERROR_UNKNOWN),
            // and that path is only taken without VK_EXT_depth_clip_control.
            .depth_clamp_enable(self.depth_clamp && draw.depth_mode == 0)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(cull)
            .front_face(front)
            .line_width(1.0);
        // Polygon offset keeps coplanar details (a crown on a head, decals
        // and shadows on a floor) from z-fighting. Vulkan's constant factor
        // is the guest's units and its slope factor the guest's factor.
        if let Some((factor, units, clamp)) = draw.depth_bias {
            rasterization = rasterization
                .depth_bias_enable(true)
                .depth_bias_slope_factor(factor)
                .depth_bias_constant_factor(units)
                .depth_bias_clamp(if self.depth_bias_clamp { clamp } else { 0.0 });
        }
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // Integer attachments cannot blend.
        let mut attachments = [vk::PipelineColorBlendAttachmentState::default(); 8];
        for (index, (((enable, equation, mask), slot), attachment)) in blends[..targets.len()]
            .iter()
            .zip(targets)
            .zip(attachments.iter_mut())
            .enumerate()
        {
            let float = slot.is_some_and(|(key, _)| {
                formats::color_target_kind(key.format) == AttributeType::Float
            });
            let blendable = format_features(
                &self.format_features,
                &self.instance,
                self.physical_device,
                color_formats[index],
            )
            .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND);
            let enable = *enable && float && blendable;
            let mut write = vk::ColorComponentFlags::empty();
            for (bit, flag) in [
                (0x1, vk::ColorComponentFlags::R),
                (0x10, vk::ColorComponentFlags::G),
                (0x100, vk::ColorComponentFlags::B),
                (0x1000, vk::ColorComponentFlags::A),
            ] {
                if mask & bit != 0 {
                    write |= flag;
                }
            }
            let (alpha_op, alpha_src, alpha_dst) = if equation.separate_alpha {
                (equation.alpha_op, equation.alpha_src, equation.alpha_dst)
            } else {
                (equation.color_op, equation.color_src, equation.color_dst)
            };
            *attachment = vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::Bool32::from(enable),
                src_color_blend_factor: formats::blend_factor(equation.color_src),
                dst_color_blend_factor: formats::blend_factor(equation.color_dst),
                color_blend_op: formats::blend_op(equation.color_op),
                src_alpha_blend_factor: formats::blend_factor(alpha_src),
                dst_alpha_blend_factor: formats::blend_factor(alpha_dst),
                alpha_blend_op: formats::blend_op(alpha_op),
                color_write_mask: write,
            };
        }
        let attachments = &attachments[..targets.len()];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(color_formats);
        if let Some(format) = depth_format {
            rendering = rendering.depth_attachment_format(format);
            if formats::has_stencil(format) {
                rendering = rendering.stencil_attachment_format(format);
            }
        }
        let testing = depth_format.is_some() && draw.depth_test_enable;
        let mut depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(testing)
            .depth_write_enable(testing && draw.depth_write_enable)
            .depth_compare_op(formats::compare_op(draw.depth_test_func));
        // Deferred lighting marks the pixels inside a light or reflection
        // volume in stencil and shades only those: without the test, a
        // volume shaded its whole box on screen (LM3's lobby and elevator
        // rectangles).
        if let (Some((front, back)), Some(format)) = (draw.stencil, depth_format) {
            if formats::has_stencil(format) {
                let face = |face: StencilFace| vk::StencilOpState {
                    fail_op: formats::stencil_op(face.fail),
                    pass_op: formats::stencil_op(face.pass),
                    depth_fail_op: formats::stencil_op(face.depth_fail),
                    compare_op: formats::compare_op(face.func),
                    compare_mask: face.compare_mask,
                    write_mask: face.write_mask,
                    reference: face.reference,
                };
                depth_stencil = depth_stencil
                    .stencil_test_enable(true)
                    .front(face(front))
                    .back(face(back));
            }
        }
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic)
            .layout(self.pipeline_layout);
        if let Some(cache) = &self.render_passes {
            let stencil_ops = depth_format.is_some() && draw.stencil.is_some();
            let pass = cache.pass(color_formats, depth_format, stencil_ops);
            info = info.render_pass(pass).subpass(0);
        } else {
            info = info.push_next(&mut rendering);
        }
        let started = std::time::Instant::now();
        let created = unsafe {
            self.device.create_graphics_pipelines(
                self.pipeline_cache,
                std::slice::from_ref(&info),
                None,
            )
        };
        if let Some(limit) = slow_pipeline_limit() {
            let elapsed = started.elapsed();
            if elapsed.as_millis() >= limit {
                println!(
                    "[recomp-vulkan] pipeline {:#x}/{:#x} created in {:.1} ms",
                    vertex_key.0,
                    fragment_key.0,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
        }
        let pipeline = match created {
            Ok(pipelines) => {
                // `RECOMP_VULKAN_SHADER_DUMP_ALL=1`: every pipeline's, too.
                if std::env::var_os("RECOMP_VULKAN_SHADER_DUMP_ALL").is_some() {
                    self.dump_shaders(vertex_key, fragment_key);
                }
                self.pipeline_cache_changed();
                Some(pipelines[0])
            }
            Err((_, result)) => {
                // Adreno answers VK_ERROR_UNKNOWN and says nothing else;
                // the state that produced it is all there is to go on.
                // `RECOMP_VULKAN_SHADER_DUMP=<dir>` writes both stages'
                // SPIR-V, so the host can validate them with spirv-val.
                self.dump_shaders(vertex_key, fragment_key);
                let features: Vec<String> = color_formats
                    .iter()
                    .map(|format| {
                        let flags = format_features(
                            &self.format_features,
                            &self.instance,
                            self.physical_device,
                            *format,
                        );
                        format!("{format:?} {flags:?}")
                    })
                    .collect();
                self.warn(format!(
                    "pipeline for shaders {:#x}/{:#x}: {result} ({topology:?}, colors {color_formats:?} features [{}], depth {depth_format:?}, {} bindings, {} attributes, restart {restart})",
                    vertex_key.0,
                    fragment_key.0,
                    features.join(", "),
                    bindings.len(),
                    attributes.len()
                ));
                None
            }
        };
        self.pipelines.insert(key.clone(), pipeline);
        self.pipeline_last = Some((key, pipeline));
        Ok(pipeline)
    }

    /// Record a blit of the presentable target into a swapchain image (or a
    /// clear to `fallback` when nothing was drawn), which ends ready to
    /// present.
    pub fn show_in(
        &self,
        swapchain_image: vk::Image,
        extent: vk::Extent2D,
        fallback: [f32; 4],
        frame: Option<(TargetKey, Option<(u32, u32)>)>,
    ) {
        self.flush_clears();
        let image = frame.and_then(|(key, _)| self.targets.get(&key));
        // The part of the image the guest's frame covers: its crop
        // rectangle, else what draws covered (else the whole target).
        let shown = frame
            .and_then(|(key, crop)| crop.or_else(|| self.drawn.get(&key).copied()))
            .filter(|(width, height)| *width > 0 && *height > 0)
            .map(|(width, height)| match image {
                // The crop is in the guest's pixels.
                Some(image) => (
                    scaled(width, image.scale).min(image.width),
                    scaled(height, image.scale).min(image.height),
                ),
                None => (width, height),
            });
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let to_transfer = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(swapchain_image)
            .subresource_range(range);
        let to_present = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(crate::PRESENT_LAYOUT)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(swapchain_image)
            .subresource_range(range);
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        self.barrier();
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_transfer),
            );
            match image {
                Some(image) => {
                    let (width, height) = shown.unwrap_or((image.width, image.height));
                    // The frame keeps its aspect ratio in a resized or
                    // fullscreen window: black bars fill the rest.
                    let scale = (extent.width as f32 / width as f32)
                        .min(extent.height as f32 / height as f32);
                    let (fit_width, fit_height) = (
                        ((width as f32 * scale).round() as u32).clamp(1, extent.width),
                        ((height as f32 * scale).round() as u32).clamp(1, extent.height),
                    );
                    let (left, top) = (
                        (extent.width - fit_width) / 2,
                        (extent.height - fit_height) / 2,
                    );
                    if (fit_width, fit_height) != (extent.width, extent.height) {
                        self.device.cmd_clear_color_image(
                            self.command,
                            swapchain_image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                            std::slice::from_ref(&range),
                        );
                    }
                    // `VK_FILTER_LINEAR` needs SAMPLED_IMAGE_FILTER_LINEAR
                    // on the source format; integer and some float formats
                    // do not have it and a linear blit from them is invalid.
                    let filter = if format_features(
                        &self.format_features,
                        &self.instance,
                        self.physical_device,
                        image.format,
                    )
                    .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
                    {
                        vk::Filter::LINEAR
                    } else {
                        vk::Filter::NEAREST
                    };
                    let blit = vk::ImageBlit {
                        src_subresource: layers,
                        src_offsets: [
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: width as i32,
                                y: height as i32,
                                z: 1,
                            },
                        ],
                        dst_subresource: layers,
                        dst_offsets: [
                            vk::Offset3D {
                                x: left as i32,
                                y: top as i32,
                                z: 0,
                            },
                            vk::Offset3D {
                                x: (left + fit_width) as i32,
                                y: (top + fit_height) as i32,
                                z: 1,
                            },
                        ],
                    };
                    self.device.cmd_blit_image(
                        self.command,
                        image.image,
                        vk::ImageLayout::GENERAL,
                        swapchain_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        std::slice::from_ref(&blit),
                        filter,
                    );
                }
                None => self.device.cmd_clear_color_image(
                    self.command,
                    swapchain_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: fallback },
                    std::slice::from_ref(&range),
                ),
            }
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_present),
            );
        }
    }

    /// The render target the guest last used at one of `addresses`.
    pub fn target_at_any(&self, addresses: &[u64]) -> Option<TargetKey> {
        // Presentation addresses are canonical CPU addresses (see
        // `recomp-services::gpu::Channel::present`).
        addresses
            .iter()
            .find_map(|address| self.target_at.get(address).copied())
            .filter(|key| self.targets.contains_key(key))
    }

    /// The render targets created so far.
    /// One key per render target image (the largest size drawn there).
    pub fn target_keys(&self) -> Vec<TargetKey> {
        let mut largest: HashMap<vk::Image, TargetKey> = HashMap::new();
        for (key, image) in &self.targets {
            let entry = largest.entry(image.image).or_insert(*key);
            if u64::from(key.width) * u64::from(key.height)
                > u64::from(entry.width) * u64::from(entry.height)
            {
                *entry = *key;
            }
        }
        largest.into_values().collect()
    }

    /// Record a copy of a render target, converted to RGBA8 by a blit, into
    /// the stream; [`Gpu::write_readback`] saves it once the submission has
    /// finished. One target per submission keeps the GPU worker from
    /// stalling the guest (LM3 races its render thread against a job that
    /// waits on the GPU, so long stalls crash it).
    pub fn record_readback(&mut self, key: TargetKey) -> Result<Readback, String> {
        self.flush_clears();
        let source = *self.targets.get(&key).ok_or("no such target")?;
        self.record_readback_of(
            source.image,
            vk::ImageLayout::GENERAL,
            source.width,
            source.height,
            key,
        )
    }

    /// Record a copy of the window's image (`swapchain_image`, in the
    /// present layout): the frame as shown, with the menu and black bars
    /// (`RECOMP_VULKAN_CAPTURE_WINDOW`). Its file is named `window-...`.
    pub fn record_window_readback(
        &mut self,
        swapchain_image: vk::Image,
        extent: vk::Extent2D,
    ) -> Result<Readback, String> {
        self.flush_clears();
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let transition = |from: vk::ImageLayout, to: vk::ImageLayout| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .old_layout(from)
                .new_layout(to)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(swapchain_image)
                .subresource_range(range)
        };
        let barrier = |barrier: vk::ImageMemoryBarrier| unsafe {
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        };
        barrier(transition(
            crate::PRESENT_LAYOUT,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ));
        let readback = self.record_readback_of(
            swapchain_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            extent.width,
            extent.height,
            TargetKey {
                address: WINDOW_CAPTURE,
                width: extent.width,
                height: extent.height,
                format: 0,
            },
        );
        let barrier = |barrier: vk::ImageMemoryBarrier| unsafe {
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        };
        barrier(transition(
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            crate::PRESENT_LAYOUT,
        ));
        readback
    }

    fn record_readback_of(
        &mut self,
        source_image: vk::Image,
        source_layout: vk::ImageLayout,
        width: u32,
        height: u32,
        key: TargetKey,
    ) -> Result<Readback, String> {
        let staging = self.create_image(
            vk::Format::R8G8B8A8_UNORM,
            width,
            height,
            1,
            vk::ImageViewType::TYPE_2D,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ComponentMapping::default(),
        )?;
        let len = u64::from(width) * u64::from(height) * 4;
        if let Some(small) = self.readback.take_if(|buffer| buffer.size < len) {
            // A capture tool's path: waiting for the GPU here is fine.
            unsafe {
                self.device
                    .device_wait_idle()
                    .map_err(error("readback wait"))?;
                self.device.destroy_buffer(small.buffer, None);
                self.device.free_memory(small.memory, None);
            }
        }
        if self.readback.is_none() {
            self.readback = Some(host_buffer(
                &self.device,
                &self.memory_properties,
                len.next_power_of_two().max(READBACK_BYTES),
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_CACHED,
            )?);
        }
        let readback = self.readback.as_mut().expect("created above");
        readback.cursor = 0;
        let offset = readback
            .reserve(len, 16)
            .ok_or("target too large to read back")?;
        let readback_buffer = readback.buffer;
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let corner = vk::Offset3D {
            x: width as i32,
            y: height as i32,
            z: 1,
        };
        let blit = vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [vk::Offset3D::default(), corner],
            dst_subresource: layers,
            dst_offsets: [vk::Offset3D::default(), corner],
        };
        let copy = vk::BufferImageCopy {
            buffer_offset: offset,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: layers,
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        self.barrier();
        unsafe {
            self.device.cmd_blit_image(
                self.command,
                source_image,
                source_layout,
                staging.image,
                vk::ImageLayout::GENERAL,
                std::slice::from_ref(&blit),
                vk::Filter::NEAREST,
            );
        }
        self.barrier();
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                self.command,
                staging.image,
                vk::ImageLayout::GENERAL,
                readback_buffer,
                std::slice::from_ref(&copy),
            );
        }
        Ok(Readback {
            key,
            staging,
            offset,
        })
    }

    /// Write a finished readback to `dir` as a PPM image named after the
    /// target (on a thread of its own: the guest must not wait for it), and
    /// free its staging image.
    pub fn write_readback(&mut self, readback: Readback, dir: &Path) {
        let Readback {
            key,
            staging,
            offset,
        } = readback;
        let (width, height) = (staging.width, staging.height);
        let len = u64::from(width) * u64::from(height) * 4;
        let pixels = self
            .readback
            .as_ref()
            .map(|buffer| buffer.read(offset, len).to_vec())
            .unwrap_or_default();
        self.unfilterable.borrow_mut().remove(&staging.view);
        unsafe {
            self.device.destroy_image_view(staging.view, None);
            self.device.destroy_image(staging.image, None);
            self.device.free_memory(staging.memory, None);
        }
        let path = if key.address == WINDOW_CAPTURE {
            dir.join(format!("window-{}x{}.ppm", key.width, key.height))
        } else {
            dir.join(format!(
                "{:x}-{}x{}-{:x}.ppm",
                key.address, key.width, key.height, key.format
            ))
        };
        std::thread::spawn(move || {
            let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
            for texel in pixels.as_chunks::<4>().0 {
                ppm.extend_from_slice(&texel[..3]);
            }
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(error) = std::fs::write(&path, ppm) {
                println!("[recomp-vulkan] {}: {error}", path.display());
            }
        });
    }

    /// A new pipeline was compiled: save the driver cache now and then, so a
    /// crash or a forced close does not lose everything learned this run.
    fn pipeline_cache_changed(&mut self) {
        self.pipelines_since_cache_save += 1;
        if self.pipelines_since_cache_save >= 64 {
            self.pipelines_since_cache_save = 0;
            self.save_pipeline_cache(false);
        }
    }

    /// Write the driver's compiled-pipeline cache. Periodic saves copy the
    /// data here and write it on another thread; `destroy` writes it before
    /// the device goes away.
    fn save_pipeline_cache(&mut self, synchronous: bool) {
        let Some(path) = self.pipeline_cache_path.clone() else {
            return;
        };
        let Ok(data) = (unsafe { self.device.get_pipeline_cache_data(self.pipeline_cache) }) else {
            return;
        };
        if data.is_empty() {
            return;
        }
        if synchronous {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&path, &data);
            return;
        }
        self.pipeline_cache_saves += 1;
        let temporary = path.with_extension(format!("tmp{}", self.pipeline_cache_saves));
        std::thread::spawn(move || {
            if let Some(parent) = temporary.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(error) = std::fs::write(&temporary, &data) {
                eprintln!(
                    "[recomp-vulkan] pipeline cache {}: {error}",
                    temporary.display()
                );
                return;
            }
            // Windows refuses to rename over an existing file.
            let _ = std::fs::remove_file(&path);
            if let Err(error) = std::fs::rename(&temporary, &path) {
                eprintln!("[recomp-vulkan] pipeline cache {}: {error}", path.display());
            }
        });
    }

    pub fn destroy(&mut self) {
        self.save_pipeline_cache(true);
        unsafe {
            let device = &self.device;
            for pipeline in self.pipelines.values().flatten() {
                device.destroy_pipeline(*pipeline, None);
            }
            for shader in self.shaders.values().flatten() {
                device.destroy_shader_module(shader.module, None);
            }
            for program in self.compute_programs.values().flatten() {
                device.destroy_pipeline(program.pipeline, None);
                device.destroy_shader_module(program.module, None);
            }
            device.destroy_pipeline_cache(self.pipeline_cache, None);
            device.destroy_pipeline_layout(self.compute_layout, None);
            for sampler in self.samplers.values() {
                device.destroy_sampler(*sampler, None);
            }
            let images = self
                .target_images
                .values()
                .chain(self.depth_images.values())
                .chain(self.resized.values().map(|(image, _)| image))
                .chain(
                    self.layered
                        .values()
                        .filter_map(|layered| layered.image.as_ref()),
                )
                .chain(&self.retired)
                .chain(self.textures.values().filter_map(|state| match state {
                    TextureState::Ready(image) | TextureState::Preview(image, _) => Some(image),
                    _ => None,
                }))
                .chain(self.placeholders.values());
            for image in images {
                if image.attachment != image.view {
                    device.destroy_image_view(image.attachment, None);
                }
                device.destroy_image_view(image.view, None);
                device.destroy_image(image.image, None);
                device.free_memory(image.memory, None);
            }
            for image in self.spares.iter().flat_map(|spare| &spare.retired) {
                if image.attachment != image.view {
                    device.destroy_image_view(image.attachment, None);
                }
                device.destroy_image_view(image.view, None);
                device.destroy_image(image.image, None);
                device.free_memory(image.memory, None);
            }
            device.destroy_fence(self.fence, None);
            for spare in &self.spares {
                device.destroy_descriptor_pool(spare.descriptor_pool, None);
                device.destroy_fence(spare.fence, None);
            }
            for stream in std::iter::once(&self.stream)
                .chain(self.spares.iter().map(|spare| &spare.stream))
                .chain(self.readback.as_ref())
                .chain(self.vertex_cache.as_ref().map(|cache| &cache.arena))
            {
                device.destroy_buffer(stream.buffer, None);
                device.free_memory(stream.memory, None);
            }
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// The format an image of view format `format` is created with.
fn storage_format(format: vk::Format) -> vk::Format {
    match format {
        vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
        other => other,
    }
}

/// A mapped, host-coherent buffer, in memory with `preferred` properties
/// when there is such memory.
fn host_buffer(
    device: &Device,
    properties: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Result<Stream, String> {
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default().size(size).usage(usage),
            None,
        )
    }
    .map_err(error("host buffer"))?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let required = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let allocate = |flags: vk::MemoryPropertyFlags| {
        let memory_type = memory_type(properties, requirements.memory_type_bits, flags)?;
        unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }
        .ok()
    };
    // The preferred kind when there is one with room (without resizable
    // BAR, device-local host-visible memory is only 256 MB), else any.
    let memory = allocate(required | preferred)
        .or_else(|| allocate(required))
        .ok_or("no host-visible memory")?;
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.map_err(error("bind host buffer"))?;
    let mapped = unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) }
        .map_err(error("map host buffer"))? as *mut u8;
    Ok(Stream {
        buffer,
        memory,
        mapped,
        size,
        cursor: 0,
    })
}

fn memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|index| {
        bits & (1 << index) != 0
            && properties.memory_types[*index as usize]
                .property_flags
                .contains(flags)
    })
}

/// The backing address of a guest GPU address: GPU mappings of one memory
/// share it. Targets are keyed by it, since the guest may clear a swapchain
/// image through one mapping and draw it through another.
fn canonical(memory: &dyn GpuMemory, address: u64) -> u64 {
    memory.resolve(address).unwrap_or(address)
}

/// LM3's lobby bloom downsample/blur chain: full-screen triangles at
/// 960x540 and below, drawn without blending or a depth test. Their shader
/// addresses move between runs, so the size identifies the chain. The gem's
/// bloom spreads into a giant cyan orb around Hellen Gravely's necklace;
/// skip the chain until its mask/scale is reproduced.
fn is_subres_postprocess(draw: &DrawCall) -> bool {
    // Keep the 960x540 and 480x270 scene-color passes: they carry Hellen's
    // color grading. Skipping them left the final composite sampling a stale
    // target, so her color flickered between frames.
    const COLOR_TARGETS: [u64; 2] = [0x5a95_10000, 0x5a9c_30000];
    draw.parameters.count == 3
        && !draw.depth_test_enable
        && draw.blend_enables.iter().all(|enabled| !enabled)
        && draw.color_targets.iter().any(|target| {
            target.enabled()
                && target.width_or_stride <= 960
                && target.height <= 540
                && !COLOR_TARGETS.contains(&target.address)
        })
}

/// `RECOMP_VULKAN_SKIP_FS=<address>,...`: skip draws with these fragment
/// programs (to find which draw makes a glitch); 0 names depth-only draws.
fn skipped_programs() -> &'static [u64] {
    static SKIPPED: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    SKIPPED.get_or_init(|| {
        std::env::var("RECOMP_VULKAN_SKIP_FS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|address| {
                u64::from_str_radix(address.trim().trim_start_matches("0x"), 16).ok()
            })
            .collect()
    })
}

/// `RECOMP_VULKAN_SKIP_KIND=<kind>,...`: skip draws of these kinds (and
/// clears, `clear`, or 2D copies, `blit`), to find what makes a GPU fault.
/// Kinds are `draw_kinds`'.
fn skips(kind: &str) -> bool {
    skipped_kinds().iter().any(|skipped| skipped == kind)
}

fn skipped_kinds() -> &'static [String] {
    static KINDS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    KINDS.get_or_init(|| {
        std::env::var("RECOMP_VULKAN_SKIP_KIND")
            .unwrap_or_default()
            .split(',')
            .filter(|kind| !kind.is_empty())
            .map(str::to_string)
            .collect()
    })
}

/// The kinds a draw is, for `RECOMP_VULKAN_SKIP_KIND`.
fn draw_kinds(draw: &DrawCall) -> Vec<&'static str> {
    let parameters = draw.parameters;
    let mut kinds = vec!["all"];
    kinds.push(match parameters.topology {
        PrimitiveTopology::Triangles => "triangles",
        PrimitiveTopology::TriangleStrip => "strip",
        PrimitiveTopology::TriangleFan => "fan",
        PrimitiveTopology::Quads => "quads",
        PrimitiveTopology::Points => "points",
        _ => "lines",
    });
    if draw.primitive_restart.is_some() {
        kinds.push("restart");
    }
    if draw.index_buffer.is_some() || !draw.inline_indices.is_empty() {
        kinds.push("indexed");
    }
    if parameters.instance_count > 1 {
        kinds.push("instanced");
    }
    kinds.push(if draw.depth_target_enable {
        "depth"
    } else {
        "nodepth"
    });
    kinds.push(match draw.color_target_count {
        0 => "depthonly",
        1 => "single",
        _ => "mrt",
    });
    let targets = &draw.color_targets[..draw.color_target_count.min(8)];
    if targets.len() > 1 {
        kinds.push(match targets.len() {
            2 => "mrt2",
            3 => "mrt3",
            _ => "mrt4",
        });
    }
    if targets
        .iter()
        .any(|target| target.address == 0 || target.format == 0)
    {
        kinds.push("gap");
    }
    if targets
        .iter()
        .any(|target| formats::color_target_kind(target.format) != AttributeType::Float)
    {
        kinds.push("integer");
    }
    if draw.fragment_shader().is_none() {
        kinds.push("nofs");
    }
    kinds
}

/// One line describing a draw, for traces.
fn describe_draw(draw: &DrawCall, outcome: &str) -> String {
    let parameters = draw.parameters;
    let targets: Vec<String> = (0..draw.color_target_count.min(8))
        .map(|buffer| {
            let target = draw.color_targets[draw.color_target_index(buffer)];
            format!(
                "{:#x} {}x{} fmt {:#x}",
                target.address, target.width_or_stride, target.height, target.format
            )
        })
        .collect();
    format!(
        "draw {outcome}: vs {:#x} fs {:#x} {:?} count {} first {} instances {} indexed {} \
         targets [{}] viewport {:?}+{:?} scissor {:?} blend {} mask {:#x} depth_mode {}",
        draw.vertex_shader().unwrap_or(0),
        draw.fragment_shader().unwrap_or(0),
        parameters.topology,
        parameters.count,
        parameters.first_vertex,
        parameters.instance_count,
        draw.index_buffer.is_some() || !draw.inline_indices.is_empty(),
        targets.join(", "),
        draw.viewport.scale,
        draw.viewport.translate,
        draw.scissor,
        draw.blend_enables[0],
        draw.color_masks[0],
        draw.depth_mode,
    )
}

/// Quads (four indices) as two triangles each, into a reused buffer.
pub fn quads_to_triangles_into(indices: &[u32], triangles: &mut Vec<u32>) {
    for [a, b, c, d] in indices.as_chunks::<4>().0 {
        triangles.extend_from_slice(&[*a, *b, *c, *a, *c, *d]);
    }
}

/// The faces a draw culls (`CULL_FACE`, OpenGL's enums), when culling is
/// on. LM3 draws a light's volume with its front faces culled when the
/// camera is inside it; drawing both sides (as every draw once did) lit and
/// unlit boxes of floor as the camera moved.
fn cull_mode(enable: bool, face: u32) -> vk::CullModeFlags {
    match face {
        _ if !enable => vk::CullModeFlags::NONE,
        0x404 => vk::CullModeFlags::FRONT,
        0x405 => vk::CullModeFlags::BACK,
        0x408 => vk::CullModeFlags::FRONT_AND_BACK,
        _ => vk::CullModeFlags::NONE,
    }
}

/// The guest index that restarts the draw's strips or fans, when the
/// guest turned restarting on and the draw reads indices. Lists are left
/// alone: Vulkan restarts them only with an extra feature.
fn restart_index(draw: &DrawCall) -> Option<u32> {
    let strips = matches!(
        draw.parameters.topology,
        PrimitiveTopology::LineStrip
            | PrimitiveTopology::TriangleStrip
            | PrimitiveTopology::TriangleFan
            | PrimitiveTopology::LineStripAdjacency
            | PrimitiveTopology::TriangleStripAdjacency
    );
    let indexed = draw.index_buffer.is_some() || !draw.inline_indices.is_empty();
    draw.primitive_restart.filter(|_| strips && indexed)
}

/// Bytes from one layer of a block-linear 2D array to the next, for the
/// formats a target can be drawn in (one texel per block); None for
/// another layout or format.
fn array_layer_stride(header: &TextureHeader) -> Option<u64> {
    let Layout::BlockLinear { gobs_in_y } = header.layout() else {
        return None;
    };
    let bytes = match formats::texture_format(header.format(), header.srgb()) {
        Some(layout) if layout.block == 1 => layout.bytes,
        Some(_) => return None,
        None => formats::depth_texel_bytes(header.format())?,
    };
    let (width, height) = (header.width().max(1), header.height().max(1));
    Some(texture::layer_stride(
        header.levels().max(1),
        u64::from(gobs_in_y),
        |level| u64::from((height >> level).max(1)),
        |level| u64::from((width >> level).max(1)) * u64::from(bytes),
    ))
}

/// The index with every bit of a `size`-byte index set.
fn all_ones(size: usize) -> u32 {
    if size >= 4 {
        u32::MAX
    } else {
        (1 << (8 * size)) - 1
    }
}

/// The winding of front faces (`FRONT_FACE`, OpenGL's enums); it also picks
/// the two-sided stencil test's face. OpenGL's winding is in a window whose
/// y goes up; the guest renders with y going down, as Vulkan does, and sets
/// `Y_CONTROL`'s flip to turn the winding back, so the guest's winding is
/// Vulkan's with the flip, and the other one without. (Taken the other way
/// round, LM3's back-face culling removed every front face.)
fn front_face(front_face: u32, y_control: u32) -> vk::FrontFace {
    let counter_clockwise = front_face != 0x900;
    if counter_clockwise == (y_control & 0x10 != 0) {
        vk::FrontFace::COUNTER_CLOCKWISE
    } else {
        vk::FrontFace::CLOCKWISE
    }
}

/// The guest viewport transform as a Vulkan viewport: the guest maps
/// x_ndc to `scale * x + translate`, which is a viewport `2 * scale` wide
/// starting at `translate - scale` (negative heights flip y, as the guest's
/// negative y scale does).
pub fn viewport(draw: &DrawCall, width: u32, height: u32) -> vk::Viewport {
    let transform = draw.viewport;
    let [sx, mut sy, sz] = transform.scale;
    // Y_CONTROL bit 0 negates y in the guest's viewport transform (flipped
    // passes: reflections and shadow variants). A negative Vulkan viewport
    // height reproduces it.
    if draw.y_control & 1 != 0 {
        sy = -sy;
    }
    let [tx, ty, tz] = transform.translate;
    let (near, far) = if draw.depth_mode == 0 {
        (tz - sz, tz + sz)
    } else {
        (tz, tz + sz)
    };
    let (min_depth, max_depth) = (near.clamp(0.0, 1.0), far.clamp(0.0, 1.0));
    if sx == 0.0 || sy == 0.0 {
        // No transform set: the whole target.
        return vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth,
            max_depth,
        };
    }
    vk::Viewport {
        x: tx - sx,
        y: ty - sy,
        width: 2.0 * sx,
        height: 2.0 * sy,
        min_depth,
        max_depth,
    }
}

/// Whether two host viewports are the same state (ash's `Viewport` does not
/// compare).
fn same_viewport(a: &vk::Viewport, b: &vk::Viewport) -> bool {
    a.x == b.x
        && a.y == b.y
        && a.width == b.width
        && a.height == b.height
        && a.min_depth == b.min_depth
        && a.max_depth == b.max_depth
}

/// The guest scissor, or the whole target when it is off.
pub fn scissor(draw: &DrawCall, width: u32, height: u32) -> vk::Rect2D {
    let scissor = draw.scissor;
    if !scissor.enable {
        return vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
    }
    let (x1, x2) = (
        u32::from(scissor.x1).min(width),
        u32::from(scissor.x2).min(width),
    );
    let (y1, y2) = (
        u32::from(scissor.y1).min(height),
        u32::from(scissor.y2).min(height),
    );
    vk::Rect2D {
        offset: vk::Offset2D {
            x: x1 as i32,
            y: y1 as i32,
        },
        extent: vk::Extent2D {
            width: x2.saturating_sub(x1),
            height: y2.saturating_sub(y1),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quads_become_two_triangles_each() {
        let mut triangles = Vec::new();
        quads_to_triangles_into(&[0, 1, 2, 3, 4, 5, 6, 7], &mut triangles);
        assert_eq!(triangles, [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
        triangles.clear();
        quads_to_triangles_into(&[0, 1, 2], &mut triangles);
        assert!(triangles.is_empty());
    }

    #[test]
    fn large_mipmapped_textures_get_a_preview() {
        // 1024x1024 with 11 levels: the preview is levels 3 (128x128) on.
        assert_eq!(preview_level(1024, 1024, 1, 11, (0, 11)), Some(3));
        assert_eq!(preview_level(1024, 512, 1, 11, (0, 11)), Some(3));
        assert_eq!(preview_level(512, 512, 1, 10, (0, 10)), Some(2));
        // Small, single-level, 3D or partly viewed textures come whole.
        assert_eq!(preview_level(128, 128, 1, 8, (0, 8)), None);
        assert_eq!(preview_level(2048, 2048, 1, 1, (0, 1)), None);
        assert_eq!(preview_level(512, 512, 4, 10, (0, 10)), None);
        assert_eq!(preview_level(1024, 1024, 1, 11, (2, 9)), None);
    }

    #[test]
    fn array_layers_lie_a_block_linear_layer_apart() {
        // LM3's shadow pool: 64 Z16 layers, block-linear 16 GOBs high; the
        // game draws a 1024x1024 layer every 2 MiB and a 512x512 one every
        // 512 KiB.
        let header = |size: u32, descriptor: u32| TextureHeader {
            words: [
                0x2493a,
                0xadc3_0000,
                0x5 | (descriptor << 21),
                4 << 3,
                (size - 1) | (5 << 23),
                (size - 1) | (63 << 16),
                0,
                0,
            ],
        };
        assert_eq!(array_layer_stride(&header(1024, 3)), Some(2 << 20));
        assert_eq!(array_layer_stride(&header(512, 3)), Some(512 << 10));
        // Pitch-linear arrays are not drawn layer by layer.
        assert_eq!(array_layer_stride(&header(512, 2)), None);
    }

    #[test]
    fn restart_is_all_ones_of_the_index_size() {
        assert_eq!(all_ones(1), 0xFF);
        assert_eq!(all_ones(2), 0xFFFF);
        assert_eq!(all_ones(4), u32::MAX);
    }

    #[test]
    fn culling_and_winding_follow_the_guest() {
        // LM3: counter-clockwise with Y_CONTROL's flip for most draws.
        assert_eq!(front_face(0x901, 0x10), vk::FrontFace::COUNTER_CLOCKWISE);
        assert_eq!(front_face(0x901, 0), vk::FrontFace::CLOCKWISE);
        assert_eq!(front_face(0x900, 0x10), vk::FrontFace::CLOCKWISE);
        assert_eq!(front_face(0x900, 0), vk::FrontFace::COUNTER_CLOCKWISE);
        assert_eq!(cull_mode(true, 0x405), vk::CullModeFlags::BACK);
        assert_eq!(cull_mode(true, 0x404), vk::CullModeFlags::FRONT);
        assert_eq!(cull_mode(true, 0x408), vk::CullModeFlags::FRONT_AND_BACK);
        assert_eq!(cull_mode(false, 0x405), vk::CullModeFlags::NONE);
    }

    #[test]
    fn descriptor_batches_stay_within_the_pool() {
        assert_eq!(descriptor_batch_count(0), DESCRIPTOR_BATCH);
        assert_eq!(descriptor_batch_count(12_000), DESCRIPTOR_BATCH);
        assert_eq!(descriptor_batch_count(DESCRIPTOR_SETS - 100), 100);
        assert_eq!(descriptor_batch_count(DESCRIPTOR_SETS), 0);
        assert_eq!(descriptor_batch_count(DESCRIPTOR_SETS + 1), 0);
    }

    #[test]
    fn index_bounds_skip_restart_values() {
        let short = |values: &[u16]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        assert_eq!(index_bounds(&short(&[7, 3, 0xFFFF, 9]), 2, true), (3, 9));
        assert_eq!(index_bounds(&short(&[0xFFFF, 0xFFFF]), 2, true), (0, 0));
        assert_eq!(index_bounds(&short(&[0xFFFF, 0]), 2, true), (0, 0));
        assert_eq!(index_bounds(&[], 2, true), (0, 0));
        let long: Vec<u8> = [u32::MAX, 5, 0x1_0000]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(index_bounds(&long, 4, true), (5, 0x1_0000));
        // Without restarting, all ones is a vertex.
        assert_eq!(
            index_bounds(&short(&[7, 3, 0xFFFF, 9]), 2, false),
            (3, 0xFFFF)
        );
        assert_eq!(index_bounds(&[], 2, false), (0, 0));
    }

    #[test]
    fn texture_bindings_must_fit_the_layout() {
        assert!(texture_binding_fits(TEXTURE_BINDING_BASE));
        assert!(texture_binding_fits(
            TEXTURE_BINDING_BASE + TEXTURE_BINDINGS - 1
        ));
        assert!(!texture_binding_fits(
            TEXTURE_BINDING_BASE + TEXTURE_BINDINGS
        ));
        assert!(!texture_binding_fits(TEXTURE_BINDING_BASE - 1));
    }

    #[test]
    fn one_row_lookup_tables_are_supported() {
        assert!(texture_target_supported(TextureKind::D1, Some(Target::D1)));
        assert!(texture_target_supported(
            TextureKind::D1,
            Some(Target::D1Array)
        ));
        assert!(texture_target_supported(TextureKind::D2, Some(Target::D1)));
        assert!(texture_target_supported(
            TextureKind::D2,
            Some(Target::D1Array)
        ));
        assert!(texture_target_supported(
            TextureKind::D2Array,
            Some(Target::D1Array)
        ));
        assert!(!texture_target_supported(TextureKind::D3, Some(Target::D1)));
    }
}
