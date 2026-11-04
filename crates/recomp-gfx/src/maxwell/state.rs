//! The Maxwell 3D register file.
//!
//! The GPU keeps the 3D engine state in a register file addressed by method
//! number. Command list methods below `0xE00` write one word of this state at
//! `method * 4`; the offsets are in [`super::methods`]. The layout follows the
//! hardware's class 0xB197 state block (the same layout as NVIDIA's Maxwell 3D
//! class documentation and the emulator implementations derived from it).

use super::methods as m;

pub const STATE_WORDS: usize = m::STATE_SIZE / 4;
pub const SHADER_STAGES: usize = 6;
pub const COLOR_TARGETS: usize = 8;
pub const VERTEX_ATTRIBUTES: usize = 32;
pub const VERTEX_BUFFERS: usize = 16;
pub const VIEWPORTS: usize = 16;

/// The GPU virtual address split into two 32-bit registers. The high word
/// comes first in the register file.
pub fn pack_va(high: u32, low: u32) -> u64 {
    (u64::from(high) << 32) | u64::from(low)
}

#[derive(Clone, Debug)]
pub struct MaxwellState {
    words: Vec<u32>,
}

impl Default for MaxwellState {
    fn default() -> Self {
        Self {
            words: vec![0; STATE_WORDS],
        }
    }
}

impl MaxwellState {
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// Read a register by byte offset. Reads outside the register file return
    /// zero, as the hardware does for unimplemented registers.
    pub fn read(&self, offset: usize) -> u32 {
        self.words.get(offset / 4).copied().unwrap_or(0)
    }

    pub fn read_i32(&self, offset: usize) -> i32 {
        self.read(offset) as i32
    }

    pub fn read_f32(&self, offset: usize) -> f32 {
        f32::from_bits(self.read(offset))
    }

    pub fn read_va(&self, offset: usize) -> u64 {
        pack_va(self.read(offset), self.read(offset + 4))
    }

    pub fn write(&mut self, offset: usize, value: u32) {
        if let Some(word) = self.words.get_mut(offset / 4) {
            *word = value;
        }
    }

    pub fn write_va(&mut self, offset: usize, address: u64) {
        self.write(offset, (address >> 32) as u32);
        self.write(offset + 4, address as u32);
    }

    /// The base address of the shader region. Shader offsets in the stage
    /// registers are relative to it.
    pub fn shader_base(&self) -> u64 {
        self.read_va(m::SHADER_BASE_ADDRESS)
    }

    /// The GPU address of the program header of `stage` (0 = vertex A,
    /// 1 = vertex B, ..., 5 = fragment), or `None` when the stage is
    /// disabled. Vertex B is always enabled, as the hardware has no enable
    /// bit for it.
    pub fn shader_address(&self, stage: usize) -> Option<u64> {
        if stage >= SHADER_STAGES {
            return None;
        }
        let base = m::SHADER_STATE + stage * 0x40;
        let control = self.read(base);
        if stage != 1 && control & 1 == 0 {
            return None;
        }
        Some(
            self.shader_base()
                .wrapping_add(u64::from(self.read(base + 4))),
        )
    }

    pub fn shader_control(&self, stage: usize) -> u32 {
        self.read(m::SHADER_STATE + stage * 0x40)
    }

    pub fn shader_type(&self, stage: usize) -> u32 {
        self.read(m::SHADER_STATE + stage * 0x40 + 0x10)
    }

    pub fn shader_max_registers(&self, stage: usize) -> u32 {
        self.read(m::SHADER_STATE + stage * 0x40 + 0x0c)
    }

    pub fn vertex_attribute(&self, index: usize) -> u32 {
        self.read(m::VERTEX_ATTRIB_STATE + index * 4)
    }

    pub fn vertex_buffer(&self, index: usize) -> VertexBuffer {
        let base = m::VERTEX_BUFFER_STATE + index * 0x10;
        VertexBuffer {
            control: self.read(base),
            address: self.read_va(base + 4),
            divisor: self.read(base + 0x0c),
        }
    }

    pub fn color_target(&self, index: usize) -> ColorTargetState {
        let base = m::RT_COLOR_STATE + index * 0x40;
        ColorTargetState {
            address: self.read_va(base),
            width_or_stride: self.read(base + 0x08),
            height: self.read(base + 0x0c),
            format: self.read(base + 0x10),
            memory_layout: self.read(base + 0x14),
            depth: self.read(base + 0x18),
            layer_size: self.read(base + 0x1c),
            base_layer: self.read(base + 0x20),
        }
    }

    pub fn depth_target(&self) -> DepthTargetState {
        let base = m::RT_DEPTH_STENCIL_STATE;
        DepthTargetState {
            address: self.read_va(base),
            format: self.read(base + 0x08),
            memory_layout: self.read(base + 0x0c),
            layer_size: self.read(base + 0x10),
        }
    }

    /// Number of active color targets, from the render target control
    /// register. The hardware supports up to eight.
    pub fn color_target_count(&self) -> usize {
        (self.read(m::RT_CONTROL) & 0xf) as usize
    }

    /// The attachment index bound to draw buffer `index`. Draw buffers are
    /// permuted independently of attachment numbers.
    pub fn color_target_index(&self, index: usize) -> usize {
        ((self.read(m::RT_CONTROL) >> (4 + index * 3)) & 7) as usize
    }

    pub fn viewport_transform(&self, index: usize) -> ViewportTransform {
        let base = m::VIEWPORT_TRANSFORM + index * 0x20;
        ViewportTransform {
            scale: [
                self.read_f32(base),
                self.read_f32(base + 4),
                self.read_f32(base + 8),
            ],
            translate: [
                self.read_f32(base + 0x0c),
                self.read_f32(base + 0x10),
                self.read_f32(base + 0x14),
            ],
            swizzle: self.read(base + 0x18),
        }
    }

    pub fn viewport_extents(&self, index: usize) -> ViewportExtents {
        let base = m::VIEWPORT_EXTENTS + index * 0x10;
        ViewportExtents {
            x: self.read(base) as u16,
            width: self.read(base + 4) as u16,
            y: self.read(base + 8) as u16,
            height: self.read(base + 0x0c) as u16,
            depth_near: self.read_f32(base + 0x10),
            depth_far: self.read_f32(base + 0x14),
        }
    }

    pub fn scissor(&self, index: usize) -> ScissorState {
        let base = m::SCISSOR_STATE + index * 0x10;
        let packed = self.read(base + 4);
        let packed2 = self.read(base + 8);
        // X1/X2 share a word, Y1/Y2 share the next. The registers are
        // written as 16-bit halves.
        ScissorState {
            enable: self.read(base) & 1 != 0,
            x1: packed as u16,
            x2: (packed >> 16) as u16,
            y1: packed2 as u16,
            y2: (packed2 >> 16) as u16,
        }
    }

    pub fn index_buffer(&self) -> IndexBufferState {
        let base = m::INDEX_BUFFER_STATE;
        IndexBufferState {
            address: self.read_va(base),
            end_address: self.read_va(base + 8),
            index_type: self.read(base + 0x10),
            first: self.read(base + 0x14),
        }
    }

    /// The vertex range of a non-indexed draw.
    pub fn vertex_buffer_draw_state(&self) -> VertexBufferDrawState {
        VertexBufferDrawState {
            first: self.read(m::VERTEX_BUFFER_DRAW_STATE),
            count: self.read(m::VERTEX_BUFFER_DRAW_STATE + 4),
        }
    }

    /// The size of the depth-stencil target. `depth` is the layer count.
    pub fn depth_target_size(&self) -> Size3D {
        let base = m::RT_DEPTH_STENCIL_SIZE;
        Size3D {
            width: self.read(base),
            height: self.read(base + 4),
            depth: self.read(base + 8),
        }
    }

    /// The size of the color target `index`. `depth` is the layer count.
    pub fn color_target_size(&self, index: usize) -> Size3D {
        let target = self.color_target(index);
        Size3D {
            width: target.width_or_stride,
            height: target.height,
            depth: target.depth,
        }
    }

    /// The constant buffer the next bind or inline update refers to: its
    /// size, then its address (high, low), then the offset inline updates
    /// write at. (LM3 writes 0x200, 5, 0xa7000, 0: 512 bytes at 0x5_000a7000.)
    pub fn uniform_buffer(&self) -> UniformBufferState {
        let base = m::UNIFORM_BUFFER_STATE;
        UniformBufferState {
            size: self.read(base),
            address: self.read_va(base + 4),
            offset: self.read(base + 0x0c),
        }
    }

    /// The write mask of color target `index`: one bit per component nibble
    /// (red bit 0, green bit 4, blue bit 8, alpha bit 12; LM3 writes 0x1111).
    pub fn color_mask(&self, index: usize) -> u32 {
        self.read(m::RT_COLOR_MASK + index * 4)
    }

    /// The color mask shared by all targets, when the shared-mask enable is
    /// set.
    pub fn shared_color_mask(&self) -> u32 {
        self.read(m::RT_COLOR_MASK_SHARED)
    }

    pub fn first_vertex(&self) -> u32 {
        self.read(m::FIRST_VERTEX)
    }

    pub fn first_instance(&self) -> u32 {
        self.read(m::FIRST_INSTANCE)
    }

    pub fn clear_color(&self) -> [f32; 4] {
        let base = m::CLEAR_COLORS;
        [
            self.read_f32(base),
            self.read_f32(base + 4),
            self.read_f32(base + 8),
            self.read_f32(base + 0x0c),
        ]
    }

    pub fn clear_depth(&self) -> f32 {
        self.read_f32(m::CLEAR_DEPTH_VALUE)
    }

    pub fn clear_stencil(&self) -> u32 {
        self.read(m::CLEAR_STENCIL_VALUE)
    }

    pub fn blend_enable(&self, index: usize) -> bool {
        self.read(m::BLEND_ENABLE + index * 4) & 1 != 0
    }

    pub fn depth_test_enable(&self) -> bool {
        self.read(m::DEPTH_TEST_ENABLE) & 1 != 0
    }

    pub fn depth_write_enable(&self) -> bool {
        self.read(m::DEPTH_WRITE_ENABLE) & 1 != 0
    }

    pub fn depth_test_func(&self) -> u32 {
        self.read(m::DEPTH_TEST_FUNC)
    }

    /// The stencil test, when enabled: front faces, and back faces (the
    /// front's state unless two-sided stencil is on).
    pub fn stencil(&self) -> Option<(StencilFace, StencilFace)> {
        if self.read(m::STENCIL_TEST_STATE) & 1 == 0 {
            return None;
        }
        // SET_STENCIL_OP_FAIL/ZFAIL/ZPASS, FUNC, FUNC_REF, FUNC_MASK, MASK
        // follow the enable; the back face's ops and function follow its
        // two-sided enable, its reference and masks are at 0xF54.
        let front = StencilFace {
            fail: self.read(m::STENCIL_TEST_STATE + 0x4),
            depth_fail: self.read(m::STENCIL_TEST_STATE + 0x8),
            pass: self.read(m::STENCIL_TEST_STATE + 0xC),
            func: self.read(m::STENCIL_TEST_STATE + 0x10),
            reference: self.read(m::STENCIL_TEST_STATE + 0x14) & 0xFF,
            compare_mask: self.read(m::STENCIL_TEST_STATE + 0x18) & 0xFF,
            write_mask: self.read(m::STENCIL_TEST_STATE + 0x1C) & 0xFF,
        };
        let back = if self.read(m::STENCIL_BACK_TEST_STATE) & 1 != 0 {
            StencilFace {
                fail: self.read(m::STENCIL_BACK_TEST_STATE + 0x4),
                depth_fail: self.read(m::STENCIL_BACK_TEST_STATE + 0x8),
                pass: self.read(m::STENCIL_BACK_TEST_STATE + 0xC),
                func: self.read(m::STENCIL_BACK_TEST_STATE + 0x10),
                reference: self.read(m::STENCIL_BACK_MASKS) & 0xFF,
                write_mask: self.read(m::STENCIL_BACK_MASKS + 0x4) & 0xFF,
                compare_mask: self.read(m::STENCIL_BACK_MASKS + 0x8) & 0xFF,
            }
        } else {
            front
        };
        Some((front, back))
    }

    pub fn cull_enable(&self) -> bool {
        self.read(m::FACE_STATE) & 1 != 0
    }

    pub fn front_face(&self) -> u32 {
        self.read(m::FACE_STATE + 4)
    }

    pub fn cull_face(&self) -> u32 {
        self.read(m::FACE_STATE + 8)
    }

    /// The window origin control: bit 0 a lower-left origin, bit 4 a
    /// flipped y (which turns the winding of front faces around).
    pub fn y_control(&self) -> u32 {
        self.read(m::Y_CONTROL)
    }

    /// The polygon offset (depth bias) for filled triangles, when enabled:
    /// the slope factor, the constant units and the clamp.
    pub fn depth_bias(&self) -> Option<(f32, f32, f32)> {
        (self.read(m::DEPTH_BIAS_STATE + 8) & 1 != 0).then(|| {
            (
                self.read_f32(m::DEPTH_BIAS_FACTOR),
                self.read_f32(m::DEPTH_BIAS_UNITS),
                self.read_f32(m::DEPTH_BIAS_CLAMP),
            )
        })
    }

    /// The index that restarts a strip or fan (`PRIMITIVE_RESTART_STATE`:
    /// enable, then the index), when restarting is on.
    pub fn primitive_restart(&self) -> Option<u32> {
        (self.read(m::PRIMITIVE_RESTART_STATE) & 1 != 0)
            .then(|| self.read(m::PRIMITIVE_RESTART_STATE + 4))
    }

    /// The view volume clip control (`VIEW_VOLUME_CLIP_CONTROL`); bits
    /// 11-13 say how geometry is clipped, and whether depth is clamped
    /// instead.
    pub fn view_volume_clip_control(&self) -> u32 {
        self.read(m::VIEW_VOLUME_CLIP_CONTROL)
    }

    /// The texture header pool (TIC): address and highest index.
    pub fn texture_pool(&self) -> PoolState {
        PoolState {
            address: self.read_va(m::TEXTURE_POOL_STATE),
            limit: self.read(m::TEXTURE_POOL_STATE + 8),
        }
    }

    /// The sampler pool (TSC): address and highest index.
    pub fn sampler_pool(&self) -> PoolState {
        PoolState {
            address: self.read_va(m::SAMPLER_POOL_STATE),
            limit: self.read(m::SAMPLER_POOL_STATE + 8),
        }
    }

    /// The constant buffer slot that holds texture handles.
    pub fn texture_buffer_index(&self) -> u32 {
        self.read(m::TEXTURE_BUFFER_INDEX)
    }

    /// Whether a handle's sampler is the one at its texture's index rather
    /// than the index in its upper bits.
    pub fn samplers_from_texture_index(&self) -> bool {
        self.read(m::SAMPLER_INDEX) & 1 != 0
    }

    /// Blending for color target `index`: its own state when blending is
    /// independent, otherwise the common state.
    pub fn blend_equation(&self, index: usize) -> BlendEquation {
        if self.read(m::BLEND_INDEPENDENT) & 1 != 0 {
            let base = m::BLEND_STATE + index * 0x20;
            BlendEquation {
                separate_alpha: self.read(base) & 1 != 0,
                color_op: self.read(base + 0x04),
                color_src: self.read(base + 0x08),
                color_dst: self.read(base + 0x0c),
                alpha_op: self.read(base + 0x10),
                alpha_src: self.read(base + 0x14),
                alpha_dst: self.read(base + 0x18),
            }
        } else {
            // The common state has an unused word before the alpha
            // destination factor.
            let base = m::BLEND_STATE_COMMON;
            BlendEquation {
                separate_alpha: self.read(base) & 1 != 0,
                color_op: self.read(base + 0x04),
                color_src: self.read(base + 0x08),
                color_dst: self.read(base + 0x0c),
                alpha_op: self.read(base + 0x10),
                alpha_src: self.read(base + 0x14),
                alpha_dst: self.read(base + 0x1c),
            }
        }
    }

    /// The last byte address vertex buffer `index` may be read at.
    pub fn vertex_buffer_limit(&self, index: usize) -> u64 {
        self.read_va(m::VERTEX_BUFFER_END_ADDRESS + index * 8)
    }

    /// 0: clip-space depth runs from -1 to 1 (OpenGL); 1: from 0 to 1.
    pub fn depth_mode(&self) -> u32 {
        self.read(m::DEPTH_MODE)
    }

    /// Which clip distances (bit per distance) clip primitives.
    pub fn user_clip_enable(&self) -> u8 {
        (self.read(m::USER_CLIP_ENABLE) & 0xFF) as u8
    }

    pub fn depth_target_enable(&self) -> bool {
        self.read(m::RT_DEPTH_STENCIL_ENABLE) & 1 != 0
    }

    pub fn alpha_test_enable(&self) -> bool {
        self.read(m::ALPHA_TEST_ENABLE) & 1 != 0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VertexBuffer {
    pub control: u32,
    pub address: u64,
    pub divisor: u32,
}

impl VertexBuffer {
    pub fn stride(&self) -> u32 {
        self.control & 0xfff
    }

    pub fn enabled(&self) -> bool {
        self.control & (1 << 12) != 0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColorTargetState {
    pub address: u64,
    pub width_or_stride: u32,
    pub height: u32,
    pub format: u32,
    pub memory_layout: u32,
    pub depth: u32,
    pub layer_size: u32,
    pub base_layer: u32,
}

impl ColorTargetState {
    pub fn enabled(&self) -> bool {
        self.format != 0 && self.width_or_stride != 0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DepthTargetState {
    pub address: u64,
    pub format: u32,
    pub memory_layout: u32,
    pub layer_size: u32,
}

/// One face's stencil test: ops on stencil fail, depth fail and pass (in
/// the guest's encoding: Direct3D-style 1-8 or OpenGL enums), the
/// comparison, and the reference and masks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct StencilFace {
    pub fail: u32,
    pub depth_fail: u32,
    pub pass: u32,
    pub func: u32,
    pub reference: u32,
    pub compare_mask: u32,
    pub write_mask: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ViewportTransform {
    pub scale: [f32; 3],
    pub translate: [f32; 3],
    pub swizzle: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ViewportExtents {
    pub x: u16,
    pub width: u16,
    pub y: u16,
    pub height: u16,
    pub depth_near: f32,
    pub depth_far: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScissorState {
    pub enable: bool,
    pub x1: u16,
    pub x2: u16,
    pub y1: u16,
    pub y2: u16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexBufferState {
    pub address: u64,
    pub end_address: u64,
    pub index_type: u32,
    pub first: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VertexBufferDrawState {
    pub first: u32,
    pub count: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Size3D {
    pub width: u32,
    pub height: u32,
    /// The layer count, for array and 3D targets.
    pub depth: u32,
}

/// A descriptor pool in GPU memory: entries are 32 bytes, `limit` is the
/// highest valid index.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolState {
    pub address: u64,
    pub limit: u32,
}

/// One color target's blend equation, in the hardware's enum values (both
/// the native and the OpenGL-style numbering occur).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct BlendEquation {
    pub separate_alpha: bool,
    pub color_op: u32,
    pub color_src: u32,
    pub color_dst: u32,
    pub alpha_op: u32,
    pub alpha_src: u32,
    pub alpha_dst: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UniformBufferState {
    pub address: u64,
    pub size: u32,
    /// Where inline updates write, relative to the address.
    pub offset: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_and_writes_methods() {
        let mut state = MaxwellState::default();
        state.write(m::SHADER_BASE_ADDRESS, 0x1234_0000);
        state.write(m::SHADER_BASE_ADDRESS + 4, 0x0000_5678);
        assert_eq!(state.shader_base(), 0x1234_0000_0000_5678);
    }

    #[test]
    fn vertex_b_always_has_a_shader() {
        let mut state = MaxwellState::default();
        state.write(m::SHADER_BASE_ADDRESS, 0);
        state.write(m::SHADER_BASE_ADDRESS + 4, 0x1000);
        state.write(m::SHADER_STATE + 0x40 + 4, 0x80);
        assert_eq!(state.shader_address(1), Some(0x1080));
        assert_eq!(state.shader_address(4), None);
        state.write(m::SHADER_STATE + 4 * 0x40, 1);
        state.write(m::SHADER_STATE + 4 * 0x40 + 4, 0x40);
        assert_eq!(state.shader_address(4), Some(0x1040));
    }

    #[test]
    fn color_target_permutation_is_unpacked() {
        let mut state = MaxwellState::default();
        state.write(m::RT_CONTROL, 2 | (3 << 4) | (5 << 7));
        assert_eq!(state.color_target_count(), 2);
        assert_eq!(state.color_target_index(0), 3);
        assert_eq!(state.color_target_index(1), 5);
    }

    #[test]
    fn depth_bias_reads_its_registers() {
        let mut state = MaxwellState::default();
        assert_eq!(state.depth_bias(), None);
        state.write(m::DEPTH_BIAS_STATE + 8, 1);
        state.write(m::DEPTH_BIAS_FACTOR, 1.5f32.to_bits());
        state.write(m::DEPTH_BIAS_UNITS, (-2.0f32).to_bits());
        state.write(m::DEPTH_BIAS_CLAMP, 0.25f32.to_bits());
        assert_eq!(state.depth_bias(), Some((1.5, -2.0, 0.25)));
    }

    #[test]
    fn vertex_buffer_details() {
        let mut state = MaxwellState::default();
        let base = m::VERTEX_BUFFER_STATE;
        state.write(base, 0x1234);
        state.write(base + 4, 0x0000_0001);
        state.write(base + 8, 0x2000_0000);
        state.write(base + 0x0c, 7);
        let buffer = state.vertex_buffer(0);
        assert_eq!(buffer.stride(), 0x234);
        assert!(buffer.enabled());
        assert_eq!(buffer.address, 0x0000_0001_2000_0000);
        assert_eq!(buffer.divisor, 7);
    }
}
