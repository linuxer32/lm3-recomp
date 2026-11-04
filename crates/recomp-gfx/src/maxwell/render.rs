//! What the command processor hands to a graphics backend.
//!
//! A [`DrawCall`] is a snapshot of the state that determines the draw. Guest
//! memory is not copied: shader code, constants, vertex data and textures stay
//! in guest memory, and the backend reads them through [`GpuMemory`] when it
//! records the draw. The command processor is synchronous, so memory is stable
//! for the duration of the callback.

use crate::maxwell::state::{
    BlendEquation, ColorTargetState, DepthTargetState, MaxwellState, PoolState, ScissorState,
    Size3D, StencilFace, VertexBuffer, ViewportExtents, ViewportTransform,
};

/// Constant buffer slots per shader stage.
pub const CONSTANT_BUFFER_SLOTS: usize = 18;

/// A constant buffer bound to a stage's slot: its GPU address and size.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConstantBufferBinding {
    pub address: u64,
    pub size: u32,
}

/// The constant buffers bound per stage (vertex, tessellation control,
/// tessellation evaluation, geometry, fragment) and slot.
pub type ConstantBufferBindings = [[Option<ConstantBufferBinding>; CONSTANT_BUFFER_SLOTS]; 5];
use crate::maxwell::compute::ComputeDispatch;
use crate::maxwell::twod::Blit;
use crate::GraphicsError;

/// GPU address space access. Addresses are GPU virtual addresses, translated
/// through the channel's mappings by the runtime.
pub trait GpuMemory {
    fn read_gpu(&self, address: u64, len: usize) -> Option<Vec<u8>>;

    /// Call `read` with the `len` bytes at `address`, in place where the
    /// memory allows (no copy); false when they cannot be read.
    fn read_gpu_with(&self, address: u64, len: usize, read: &mut dyn FnMut(&[u8])) -> bool {
        let Some(bytes) = self.read_gpu(address, len) else {
            return false;
        };
        read(&bytes);
        true
    }

    /// Watch writes to the `len` bytes at `address`, and report the parts
    /// written since check `since` to `written` (offset into the range and
    /// length, whole pages clipped to it). `since` `None` starts watching and
    /// reports nothing: read the bytes after it. Returns this check's number,
    /// the next call's `since`; `None` (the default) when writes there are
    /// not watched, and the bytes have to be read to see whether they
    /// changed.
    fn written_since(
        &self,
        _address: u64,
        _len: usize,
        _since: Option<u64>,
        _written: &mut dyn FnMut(usize, usize),
    ) -> Option<u64> {
        None
    }

    /// Write guest memory. The command processor uses this for semaphore
    /// releases and inline constant-buffer data. False when the address is not
    /// writable.
    fn write_gpu(&mut self, _address: u64, _bytes: &[u8]) -> bool {
        false
    }

    /// A canonical address for the backing memory of `address`: GPU aliases
    /// of the same memory resolve to one value. Render targets are keyed by
    /// it because the guest may clear a swapchain image through one mapping
    /// and draw it through another.
    fn resolve(&self, _address: u64) -> Option<u64> {
        None
    }

    /// The GPU's clock, in nanoseconds, for timestamped semaphore releases.
    fn gpu_time(&self) -> u64 {
        0
    }

    /// Advance a syncpoint by one: the command stream increments syncpoints
    /// itself (the puller's syncpoint method, the 3D class's
    /// `SYNCPOINT_ACTION`), and the title computes the fence values it waits
    /// for from those increments.
    fn increment_syncpoint(&mut self, _syncpoint: u32) {}

    fn read_u32(&self, address: u64) -> Option<u32> {
        let bytes = self.read_gpu(address, 4)?;
        Some(u32::from_le_bytes(bytes.as_slice().try_into().ok()?))
    }

    fn read_f32(&self, address: u64) -> Option<f32> {
        self.read_u32(address).map(f32::from_bits)
    }
}

/// A source of guest memory that holds nothing. Useful for tests and for
/// command streams whose draws do not need shader code.
pub struct EmptyMemory;

impl GpuMemory for EmptyMemory {
    fn read_gpu(&self, _address: u64, _len: usize) -> Option<Vec<u8>> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrimitiveTopology {
    #[default]
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
    Quads,
    QuadStrip,
    Polygon,
    LinesAdjacency,
    LineStripAdjacency,
    TrianglesAdjacency,
    TriangleStripAdjacency,
    Patches,
}

impl PrimitiveTopology {
    pub fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Points,
            1 => Self::Lines,
            2 => Self::LineLoop,
            3 => Self::LineStrip,
            4 => Self::Triangles,
            5 => Self::TriangleStrip,
            6 => Self::TriangleFan,
            7 => Self::Quads,
            8 => Self::QuadStrip,
            9 => Self::Polygon,
            10 => Self::LinesAdjacency,
            11 => Self::LineStripAdjacency,
            12 => Self::TrianglesAdjacency,
            13 => Self::TriangleStripAdjacency,
            14 => Self::Patches,
            _ => return None,
        })
    }

    /// The primitive type override register has its own numbering: 0 means no
    /// override, `1` is points, and the triangle fan is `0x1015`.
    pub fn from_override(value: u32) -> Self {
        match value {
            1 => Self::Points,
            2 => Self::Lines,
            3 => Self::LineStrip,
            4 => Self::Triangles,
            5 => Self::TriangleStrip,
            0x1015 => Self::TriangleFan,
            10 => Self::LinesAdjacency,
            11 => Self::LineStripAdjacency,
            12 => Self::TrianglesAdjacency,
            13 => Self::TriangleStripAdjacency,
            14 => Self::Patches,
            _ => Self::Triangles,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    U8,
    U16,
    U32,
}

impl IndexType {
    pub fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::U8,
            1 => Self::U16,
            2 => Self::U32,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    pub raw: u32,
}

impl VertexAttribute {
    pub fn buffer_index(self) -> usize {
        (self.raw & 0x1f) as usize
    }

    pub fn is_constant(self) -> bool {
        self.raw & 0x40 != 0
    }

    pub fn offset(self) -> u32 {
        (self.raw >> 7) & 0x3fff
    }

    pub fn format(self) -> u32 {
        (self.raw >> 21) & 0x3f
    }

    pub fn component_type(self) -> u32 {
        (self.raw >> 27) & 7
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexBuffer {
    pub address: u64,
    pub end_address: u64,
    pub index_type: IndexType,
    pub first: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawParameters {
    pub topology: PrimitiveTopology,
    pub count: u32,
    pub instance_count: u32,
    /// The instance index of a draw started by `DrawBegin` with the increment
    /// flag set; those draws have an instance count of one.
    pub instance_index: u32,
    pub first_index: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
    pub indexed: bool,
}

#[derive(Debug, Clone)]
pub struct DrawCall {
    pub parameters: DrawParameters,
    pub index_buffer: Option<IndexBuffer>,
    /// Index elements pushed inline as u32 values, when the draw did not read
    /// the bound index buffer.
    pub inline_indices: Vec<u32>,
    pub shader_addresses: [Option<u64>; 6],
    pub shader_controls: [u32; 6],
    pub vertex_attributes: [u32; 32],
    pub vertex_buffers: [VertexBuffer; 16],
    pub color_targets: [ColorTargetState; 8],
    pub color_target_count: usize,
    pub depth_target: DepthTargetState,
    pub viewport: ViewportTransform,
    pub viewport_extents: ViewportExtents,
    pub scissor: ScissorState,
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_test_func: u32,
    /// The stencil test (front, back faces), when enabled.
    pub stencil: Option<(StencilFace, StencilFace)>,
    pub blend_enables: [bool; 8],
    pub cull_enable: bool,
    pub front_face: u32,
    pub cull_face: u32,
    /// Polygon offset (depth bias) for filled triangles, when enabled:
    /// slope factor, constant units and clamp.
    pub depth_bias: Option<(f32, f32, f32)>,
    /// `Y_CONTROL`: bit 4 flips y, and so the front face's winding.
    pub y_control: u32,
    /// `VIEW_VOLUME_CLIP_CONTROL`: whether depth is clipped or clamped.
    pub view_volume_clip_control: u32,
    /// The index that restarts a strip or fan, when restarting is on.
    pub primitive_restart: Option<u32>,
    /// The attachment bound to each draw buffer, from the render target
    /// control register.
    pub color_target_indices: [usize; 8],
    /// Constant buffers bound per stage and slot (filled in by the
    /// processor, which tracks the bind methods).
    pub constant_buffers: ConstantBufferBindings,
    pub texture_pool: PoolState,
    pub sampler_pool: PoolState,
    /// The constant buffer slot that holds texture handles.
    pub texture_buffer_index: u32,
    pub samplers_from_texture_index: bool,
    pub blend_equations: [BlendEquation; 8],
    pub color_masks: [u32; 8],
    pub vertex_buffer_limits: [u64; 16],
    pub depth_mode: u32,
    /// The clip distances (bit per distance) that clip primitives.
    pub clip_distances: u8,
    pub depth_target_enable: bool,
    /// The depth-stencil target's width and height (and layers).
    pub depth_target_size: Size3D,
}

impl DrawCall {
    pub fn from_state(state: &MaxwellState, parameters: DrawParameters) -> Self {
        Self::from_state_with_constant_buffers(
            state,
            parameters,
            [[None; CONSTANT_BUFFER_SLOTS]; 5],
        )
    }

    /// [`from_state`](Self::from_state) with the processor's bound constant
    /// buffers, so the draw does not first fill an empty table only for the
    /// caller to overwrite it: the table is the largest part of a draw.
    pub fn from_state_with_constant_buffers(
        state: &MaxwellState,
        parameters: DrawParameters,
        constant_buffers: ConstantBufferBindings,
    ) -> Self {
        let mut shader_addresses = [None; 6];
        let mut shader_controls = [0; 6];
        for stage in 0..6 {
            shader_addresses[stage] = state.shader_address(stage);
            shader_controls[stage] = state.shader_control(stage);
        }

        let mut vertex_attributes = [0; 32];
        for (index, attribute) in vertex_attributes.iter_mut().enumerate() {
            *attribute = state.vertex_attribute(index);
        }

        let mut vertex_buffers = [VertexBuffer::default(); 16];
        for (index, buffer) in vertex_buffers.iter_mut().enumerate() {
            *buffer = state.vertex_buffer(index);
        }

        let mut color_targets = [ColorTargetState::default(); 8];
        for (index, target) in color_targets.iter_mut().enumerate() {
            *target = state.color_target(index);
        }

        let mut blend_enables = [false; 8];
        for (index, enable) in blend_enables.iter_mut().enumerate() {
            *enable = state.blend_enable(index);
        }

        let index_buffer = if parameters.indexed {
            let index = state.index_buffer();
            Some(IndexBuffer {
                address: index.address,
                end_address: index.end_address,
                index_type: IndexType::from_u32(index.index_type).unwrap_or(IndexType::U32),
                first: index.first,
            })
        } else {
            None
        };

        Self {
            parameters,
            index_buffer,
            inline_indices: Vec::new(),
            shader_addresses,
            shader_controls,
            vertex_attributes,
            vertex_buffers,
            color_targets,
            color_target_count: state.color_target_count(),
            depth_target: state.depth_target(),
            viewport: state.viewport_transform(0),
            viewport_extents: state.viewport_extents(0),
            scissor: state.scissor(0),
            depth_test_enable: state.depth_test_enable(),
            depth_write_enable: state.depth_write_enable(),
            depth_test_func: state.depth_test_func(),
            stencil: state.stencil(),
            blend_enables,
            cull_enable: state.cull_enable(),
            front_face: state.front_face(),
            cull_face: state.cull_face(),
            depth_bias: state.depth_bias(),
            y_control: state.y_control(),
            view_volume_clip_control: state.view_volume_clip_control(),
            primitive_restart: state.primitive_restart(),
            color_target_indices: color_target_permutation(state),
            constant_buffers,
            texture_pool: state.texture_pool(),
            sampler_pool: state.sampler_pool(),
            texture_buffer_index: state.texture_buffer_index(),
            samplers_from_texture_index: state.samplers_from_texture_index(),
            blend_equations: std::array::from_fn(|index| state.blend_equation(index)),
            color_masks: std::array::from_fn(|index| state.color_mask(index)),
            vertex_buffer_limits: std::array::from_fn(|index| state.vertex_buffer_limit(index)),
            depth_mode: state.depth_mode(),
            clip_distances: state.user_clip_enable(),
            depth_target_enable: state.depth_target_enable(),
            depth_target_size: state.depth_target_size(),
        }
    }

    /// The constant buffer a stage (as `shader_addresses` numbers them: 1
    /// vertex .. 5 fragment) has in `slot`.
    pub fn constant_buffer(&self, stage: usize, slot: usize) -> Option<ConstantBufferBinding> {
        let bind_stage = stage.checked_sub(1)?;
        *self.constant_buffers.get(bind_stage)?.get(slot)?
    }

    /// A draw whose index elements were pushed inline.
    pub fn with_inline_indices(mut self, indices: Vec<u32>) -> Self {
        self.inline_indices = indices;
        self.index_buffer = None;
        self
    }

    pub fn vertex_shader(&self) -> Option<u64> {
        self.shader_addresses[1]
    }

    pub fn fragment_shader(&self) -> Option<u64> {
        self.shader_addresses[5]
    }

    /// The byte offsets of the attributes in the bound vertex buffers,
    /// resolving each attribute's buffer index and offset.
    pub fn attribute_address(&self, attribute: usize) -> Option<u64> {
        let raw = self.vertex_attributes.get(attribute).copied()?;
        let attribute = VertexAttribute { raw };
        if attribute.is_constant() {
            return None;
        }
        let buffer = self.vertex_buffers.get(attribute.buffer_index())?;
        if !buffer.enabled() {
            return None;
        }
        Some(buffer.address.wrapping_add(u64::from(attribute.offset())))
    }

    /// The index of the first enabled color target attached to draw buffer
    /// zero, which is what a single-target frame goes to.
    pub fn first_color_target_address(&self) -> Option<u64> {
        for index in 0..self.color_target_count.min(8) {
            let target = self.color_targets[self.color_target_index(index)];
            if target.enabled() {
                return Some(target.address);
            }
        }
        None
    }

    /// The attachment index of draw buffer `index`.
    pub fn color_target_index(&self, index: usize) -> usize {
        self.color_target_indices
            .get(index)
            .copied()
            .unwrap_or(index.min(7))
    }
}

#[derive(Debug, Clone)]
pub struct ClearColorCall {
    pub index: usize,
    pub layer: u32,
    pub layer_count: u32,
    pub component_mask: u32,
    pub color: [f32; 4],
    pub target: ColorTargetState,
}

#[derive(Debug, Clone)]
pub struct ClearDepthStencilCall {
    pub clear_depth: bool,
    pub clear_stencil: bool,
    pub depth: f32,
    pub stencil: u32,
    pub target: DepthTargetState,
    pub width: u32,
    pub height: u32,
    pub layer_count: u32,
}

/// The graphics backend interface used by the command processor.
pub trait Renderer {
    fn draw(&mut self, _memory: &dyn GpuMemory, _draw: &DrawCall) -> Result<(), GraphicsError> {
        Ok(())
    }

    fn clear_color(
        &mut self,
        _memory: &dyn GpuMemory,
        _clear: &ClearColorCall,
    ) -> Result<(), GraphicsError> {
        Ok(())
    }

    fn clear_depth_stencil(
        &mut self,
        _memory: &dyn GpuMemory,
        _clear: &ClearDepthStencilCall,
    ) -> Result<(), GraphicsError> {
        Ok(())
    }

    /// Run a compute dispatch.
    fn dispatch(
        &mut self,
        _memory: &dyn GpuMemory,
        _dispatch: &ComputeDispatch,
    ) -> Result<(), GraphicsError> {
        Ok(())
    }

    /// A 2D engine copy between surfaces. True when the backend ran it (its
    /// source is an image the backend rendered); otherwise the processor
    /// copies in guest memory.
    fn blit(&mut self, _memory: &dyn GpuMemory, _blit: &Blit) -> Result<bool, GraphicsError> {
        Ok(false)
    }

    /// Called when a submission finishes.
    fn finish(&mut self) -> Result<(), GraphicsError> {
        Ok(())
    }

    /// Show the frame the title queued to its display, once the GPU work it
    /// waited for is done: the buffer is mapped at one of `addresses` (GPU
    /// addresses), and `crop` is the visible (width, height) from its
    /// top-left corner. Backends with a window present here.
    fn present(
        &mut self,
        _addresses: &[u64],
        _crop: Option<(u32, u32)>,
    ) -> Result<(), GraphicsError> {
        Ok(())
    }

    /// Called when the GPU has been idle for a while: a backend with a window
    /// shows its last frame again (the window may have been covered or
    /// minimized meanwhile).
    fn refresh(&mut self) -> Result<(), GraphicsError> {
        Ok(())
    }
}

/// Any boxed renderer is itself a renderer, so a channel can hold a backend
/// chosen at runtime.
impl<R: Renderer + ?Sized> Renderer for std::boxed::Box<R> {
    fn draw(&mut self, memory: &dyn GpuMemory, draw: &DrawCall) -> Result<(), GraphicsError> {
        (**self).draw(memory, draw)
    }

    fn dispatch(
        &mut self,
        memory: &dyn GpuMemory,
        dispatch: &ComputeDispatch,
    ) -> Result<(), GraphicsError> {
        (**self).dispatch(memory, dispatch)
    }

    fn blit(&mut self, memory: &dyn GpuMemory, blit: &Blit) -> Result<bool, GraphicsError> {
        (**self).blit(memory, blit)
    }

    fn clear_color(
        &mut self,
        memory: &dyn GpuMemory,
        clear: &ClearColorCall,
    ) -> Result<(), GraphicsError> {
        (**self).clear_color(memory, clear)
    }

    fn clear_depth_stencil(
        &mut self,
        memory: &dyn GpuMemory,
        clear: &ClearDepthStencilCall,
    ) -> Result<(), GraphicsError> {
        (**self).clear_depth_stencil(memory, clear)
    }

    fn finish(&mut self) -> Result<(), GraphicsError> {
        (**self).finish()
    }

    fn present(
        &mut self,
        addresses: &[u64],
        crop: Option<(u32, u32)>,
    ) -> Result<(), GraphicsError> {
        (**self).present(addresses, crop)
    }

    fn refresh(&mut self) -> Result<(), GraphicsError> {
        (**self).refresh()
    }
}

/// A renderer that counts and records draws. This is the default backend and
/// is useful for validating command processing without a display.
#[derive(Debug, Default)]
pub struct DrawRecorder {
    pub draws: Vec<DrawCall>,
    pub color_clears: Vec<ClearColorCall>,
    pub depth_clears: Vec<ClearDepthStencilCall>,
    pub finished: u64,
}

impl Renderer for DrawRecorder {
    fn draw(&mut self, _memory: &dyn GpuMemory, draw: &DrawCall) -> Result<(), GraphicsError> {
        self.draws.push(draw.clone());
        Ok(())
    }

    fn clear_color(
        &mut self,
        _memory: &dyn GpuMemory,
        clear: &ClearColorCall,
    ) -> Result<(), GraphicsError> {
        self.color_clears.push(clear.clone());
        Ok(())
    }

    fn clear_depth_stencil(
        &mut self,
        _memory: &dyn GpuMemory,
        clear: &ClearDepthStencilCall,
    ) -> Result<(), GraphicsError> {
        self.depth_clears.push(clear.clone());
        Ok(())
    }

    fn finish(&mut self) -> Result<(), GraphicsError> {
        self.finished += 1;
        Ok(())
    }
}

/// The draw-buffer-to-attachment permutation, for backends that need it.
pub fn color_target_permutation(state: &MaxwellState) -> [usize; 8] {
    let mut permutation = [0; 8];
    for (index, entry) in permutation.iter_mut().enumerate() {
        *entry = state.color_target_index(index);
    }
    permutation
}

/// The number of words in a shader program header (`SPH`).
pub const SHADER_HEADER_SIZE: usize = 0x50;

/// Reads a shader's program header and instruction stream from guest memory.
pub fn read_shader(memory: &dyn GpuMemory, address: u64, max_bytes: usize) -> Option<Vec<u8>> {
    memory.read_gpu(address, max_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxwell::methods as m;

    #[test]
    fn vertex_attribute_fields() {
        let attribute = VertexAttribute {
            raw: 3 | (1 << 6) | (0x40 << 7) | (0x1d << 21) | (7 << 27),
        };
        assert_eq!(attribute.buffer_index(), 3);
        assert!(attribute.is_constant());
        assert_eq!(attribute.offset(), 0x40);
        assert_eq!(attribute.format(), 0x1d);
        assert_eq!(attribute.component_type(), 7);
    }

    #[test]
    fn collects_draw_state() {
        let mut state = MaxwellState::default();
        state.write(m::SHADER_BASE_ADDRESS + 4, 0x1000);
        state.write(m::SHADER_STATE + 0x40 + 4, 0x80);
        state.write(m::SHADER_STATE + 5 * 0x40, 1);
        state.write(m::SHADER_STATE + 5 * 0x40 + 4, 0x180);
        state.write(m::RT_CONTROL, 1);
        state.write(m::RT_COLOR_STATE + 8, 1280);
        state.write(m::RT_COLOR_STATE + 0x0c, 720);
        state.write(m::RT_COLOR_STATE + 0x10, 1);

        let draw = DrawCall::from_state(
            &state,
            DrawParameters {
                topology: PrimitiveTopology::Triangles,
                count: 3,
                instance_count: 1,
                instance_index: 0,
                first_index: 0,
                first_vertex: 0,
                first_instance: 0,
                indexed: false,
            },
        );
        assert_eq!(draw.vertex_shader(), Some(0x1080));
        assert_eq!(draw.fragment_shader(), Some(0x1180));
        assert_eq!(draw.color_target_count, 1);
        assert_eq!(draw.first_color_target_address(), Some(0));
    }

    #[test]
    fn a_draw_carries_the_processors_constant_buffers() {
        let mut buffers = [[None; CONSTANT_BUFFER_SLOTS]; 5];
        buffers[4][3] = Some(ConstantBufferBinding {
            address: 0x1_0000_2000,
            size: 0x100,
        });
        let draw = DrawCall::from_state_with_constant_buffers(
            &MaxwellState::default(),
            DrawParameters {
                topology: PrimitiveTopology::Triangles,
                count: 3,
                instance_count: 1,
                instance_index: 0,
                first_index: 0,
                first_vertex: 0,
                first_instance: 0,
                indexed: false,
            },
            buffers,
        );
        assert_eq!(
            draw.constant_buffer(5, 3),
            Some(ConstantBufferBinding {
                address: 0x1_0000_2000,
                size: 0x100
            })
        );
    }
}
