//! Guest (Maxwell) enums to Vulkan: render target formats, texture formats,
//! vertex attribute formats, blend equations and sampler modes. The values
//! follow the hardware encodings as Ryujinx's format tables list them.

use ash::vk;

use recomp_gfx::maxwell::render::PrimitiveTopology;
use recomp_gfx::maxwell::translate::AttributeType;

/// A render target format: `None` for formats the backend does not render
/// to yet.
pub fn color_target_format(format: u32) -> Option<vk::Format> {
    Some(match format {
        0xC0 => vk::Format::R32G32B32A32_SFLOAT,
        0xC1 => vk::Format::R32G32B32A32_SINT,
        0xC2 => vk::Format::R32G32B32A32_UINT,
        0xC8 => vk::Format::R16G16B16A16_SINT,
        0xC9 => vk::Format::R16G16B16A16_UINT,
        0xCC => vk::Format::R32G32_SINT,
        0xCD => vk::Format::R32G32_UINT,
        0xD2 => vk::Format::A2B10G10R10_UINT_PACK32,
        0xD8 => vk::Format::R8G8B8A8_SINT,
        0xD9 => vk::Format::R8G8B8A8_UINT,
        0xDC => vk::Format::R16G16_SINT,
        0xDD => vk::Format::R16G16_UINT,
        0xE3 => vk::Format::R32_SINT,
        0xE4 => vk::Format::R32_UINT,
        0xEC => vk::Format::R8G8_SINT,
        0xED => vk::Format::R8G8_UINT,
        0xF0 => vk::Format::R16_SINT,
        0xF1 => vk::Format::R16_UINT,
        0xF5 => vk::Format::R8_SINT,
        0xF6 => vk::Format::R8_UINT,
        0xC3 => vk::Format::R32G32B32A32_SFLOAT,
        0xC6 => vk::Format::R16G16B16A16_UNORM,
        0xC7 => vk::Format::R16G16B16A16_SNORM,
        0xCA | 0xCE => vk::Format::R16G16B16A16_SFLOAT,
        0xCB => vk::Format::R32G32_SFLOAT,
        0xCF | 0xE6 => vk::Format::B8G8R8A8_UNORM,
        0xD0 | 0xE7 => vk::Format::B8G8R8A8_SRGB,
        0xD1 => vk::Format::A2B10G10R10_UNORM_PACK32,
        0xD5 | 0xF9 => vk::Format::R8G8B8A8_UNORM,
        0xD6 | 0xFA => vk::Format::R8G8B8A8_SRGB,
        0xD7 => vk::Format::R8G8B8A8_SNORM,
        0xDA => vk::Format::R16G16_UNORM,
        0xDB => vk::Format::R16G16_SNORM,
        0xDE => vk::Format::R16G16_SFLOAT,
        0xE0 => vk::Format::B10G11R11_UFLOAT_PACK32,
        0xE5 => vk::Format::R32_SFLOAT,
        0xEA => vk::Format::R8G8_UNORM,
        0xEB => vk::Format::R8G8_SNORM,
        0xEE => vk::Format::R16_UNORM,
        0xEF => vk::Format::R16_SNORM,
        0xF2 => vk::Format::R16_SFLOAT,
        0xF3 => vk::Format::R8_UNORM,
        0xF4 => vk::Format::R8_SNORM,
        _ => return None,
    })
}

/// Bytes per texel of a color target format, for the pitch-linear targets
/// whose `width_or_stride` is a row stride (LM3's save screen and thumbnail).
pub fn color_target_bytes(format: u32) -> Option<u32> {
    Some(match format {
        0xC0..=0xC3 => 16,
        0xC6..=0xCE => 8,
        0xCF..=0xD2 | 0xD5..=0xDE | 0xE0 | 0xE3..=0xE7 | 0xF9 | 0xFA => 4,
        0xEA..=0xF2 => 2,
        0xF3..=0xF6 => 1,
        _ => return None,
    })
}

/// The width of a pitch-linear target whose first size word is a row stride.
pub fn pitch_linear_width(stride: u32, format: u32) -> Option<u32> {
    let bytes = color_target_bytes(format)?;
    (stride >= bytes).then_some(stride / bytes)
}

/// A depth-stencil target format (Maxwell's zeta formats). AMD's Vulkan
/// drivers have no 24-bit depth, so those become 32-bit float depth, with
/// 8-bit stencil where the guest's has one.
pub fn depth_format(format: u32) -> Option<vk::Format> {
    Some(match format {
        0x0A => vk::Format::D32_SFLOAT,
        0x13 => vk::Format::D16_UNORM,
        0x14 | 0x16 | 0x19 => vk::Format::D32_SFLOAT_S8_UINT,
        0x15 => vk::Format::D32_SFLOAT,
        _ => return None,
    })
}

/// Whether a depth format has a stencil aspect.
pub fn has_stencil(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D32_SFLOAT_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D16_UNORM_S8_UINT
    )
}

/// The aspects of an image of `format`.
pub fn aspects(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D32_SFLOAT | vk::Format::D16_UNORM => vk::ImageAspectFlags::DEPTH,
        format if has_stencil(format) => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}

/// A depth comparison, in either of the hardware's numberings (OpenGL's
/// 0x200-based, or 1-based).
pub fn compare_op(function: u32) -> vk::CompareOp {
    let index = if function >= 0x200 {
        function - 0x200
    } else {
        function.saturating_sub(1)
    };
    vk::CompareOp::from_raw(index.min(7) as i32)
}

/// A stencil operation, in either encoding the guest uses: Direct3D-style
/// 1-8 or the OpenGL enums.
pub fn stencil_op(op: u32) -> vk::StencilOp {
    match op {
        1 | 0x1E00 => vk::StencilOp::KEEP,
        2 | 0 => vk::StencilOp::ZERO,
        3 | 0x1E01 => vk::StencilOp::REPLACE,
        4 | 0x1E02 => vk::StencilOp::INCREMENT_AND_CLAMP,
        5 | 0x1E03 => vk::StencilOp::DECREMENT_AND_CLAMP,
        6 | 0x150A => vk::StencilOp::INVERT,
        7 | 0x8507 => vk::StencilOp::INCREMENT_AND_WRAP,
        8 | 0x8508 => vk::StencilOp::DECREMENT_AND_WRAP,
        _ => vk::StencilOp::KEEP,
    }
}

/// What a render target's components hold: integer targets take a fragment
/// program's register bits unchanged.
pub fn color_target_kind(format: u32) -> AttributeType {
    match format {
        0xC1 | 0xC8 | 0xCC | 0xD8 | 0xDC | 0xE3 | 0xEC | 0xF0 | 0xF5 => AttributeType::SignedInt,
        0xC2 | 0xC9 | 0xCD | 0xD2 | 0xD9 | 0xDD | 0xE4 | 0xED | 0xF1 | 0xF6 => {
            AttributeType::UnsignedInt
        }
        _ => AttributeType::Float,
    }
}

/// How a texture's texels are stored: plain texels, or 4x4 compressed blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TexelLayout {
    pub format: vk::Format,
    /// Bytes per texel, or per block for block-compressed formats.
    pub bytes: u32,
    /// Texels per block side (1 for plain formats, 4 for BCn).
    pub block: u32,
}

const fn plain(format: vk::Format, bytes: u32) -> TexelLayout {
    TexelLayout {
        format,
        bytes,
        block: 1,
    }
}

const fn compressed(format: vk::Format, bytes: u32) -> TexelLayout {
    TexelLayout {
        format,
        bytes,
        block: 4,
    }
}

/// Bytes per texel of a depth texture header format (bits 0:6): Z16, the
/// 24-bit depth and stencil ones, ZF32, and ZF32 with stencil.
pub fn depth_texel_bytes(format: u32) -> Option<u32> {
    Some(match format & 0x7F {
        0x3A => 2,
        0x29..=0x2B | 0x2F => 4,
        0x30 => 8,
        _ => return None,
    })
}

/// A texture format from a texture header's format word (bits 0:6 the
/// format, bits 7:9 the red component type) and its sRGB bit. `None` for
/// formats not decoded yet (ASTC, depth, packed 16-bit).
pub fn texture_format(format: u32, srgb: bool) -> Option<TexelLayout> {
    let code = format & 0x7F;
    // Component types: 1 snorm, 2 unorm, 3 sint, 4 uint, 7 float.
    let kind = (format >> 7) & 7;
    let pick = |unorm: vk::Format, srgb_format: vk::Format| if srgb { srgb_format } else { unorm };
    Some(match (code, kind) {
        (0x01, 3) => plain(vk::Format::R32G32B32A32_SINT, 16),
        (0x01, 7) => plain(vk::Format::R32G32B32A32_SFLOAT, 16),
        (0x01, 4) => plain(vk::Format::R32G32B32A32_UINT, 16),
        (0x03, 7) => plain(vk::Format::R16G16B16A16_SFLOAT, 8),
        (0x03, 1) => plain(vk::Format::R16G16B16A16_SNORM, 8),
        (0x03, 2) => plain(vk::Format::R16G16B16A16_UNORM, 8),
        (0x03, 3) => plain(vk::Format::R16G16B16A16_SINT, 8),
        (0x03, 4) => plain(vk::Format::R16G16B16A16_UINT, 8),
        (0x04, 7) => plain(vk::Format::R32G32_SFLOAT, 8),
        (0x04, 3) => plain(vk::Format::R32G32_SINT, 8),
        (0x04, 4) => plain(vk::Format::R32G32_UINT, 8),
        (0x07 | 0x08, 2) => plain(
            pick(vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB),
            4,
        ),
        (0x07 | 0x08, 1) => plain(vk::Format::R8G8B8A8_SNORM, 4),
        (0x07 | 0x08, 3) => plain(vk::Format::R8G8B8A8_SINT, 4),
        (0x07 | 0x08, 4) => plain(vk::Format::R8G8B8A8_UINT, 4),
        (0x09, 2) => plain(vk::Format::A2B10G10R10_UNORM_PACK32, 4),
        (0x09, 4) => plain(vk::Format::A2B10G10R10_UINT_PACK32, 4),
        (0x0C, 7) => plain(vk::Format::R16G16_SFLOAT, 4),
        (0x0C, 1) => plain(vk::Format::R16G16_SNORM, 4),
        (0x0C, 2) => plain(vk::Format::R16G16_UNORM, 4),
        (0x0C, 3) => plain(vk::Format::R16G16_SINT, 4),
        (0x0C, 4) => plain(vk::Format::R16G16_UINT, 4),
        (0x0F, 7) => plain(vk::Format::R32_SFLOAT, 4),
        (0x0F, 3) => plain(vk::Format::R32_SINT, 4),
        (0x0F, 4) => plain(vk::Format::R32_UINT, 4),
        (0x10, _) => compressed(vk::Format::BC6H_SFLOAT_BLOCK, 16),
        (0x11, _) => compressed(vk::Format::BC6H_UFLOAT_BLOCK, 16),
        (0x17, _) => compressed(
            pick(vk::Format::BC7_UNORM_BLOCK, vk::Format::BC7_SRGB_BLOCK),
            16,
        ),
        (0x18, 2) => plain(vk::Format::R8G8_UNORM, 2),
        (0x18, 1) => plain(vk::Format::R8G8_SNORM, 2),
        (0x18, 3) => plain(vk::Format::R8G8_SINT, 2),
        (0x18, 4) => plain(vk::Format::R8G8_UINT, 2),
        (0x1B, 7) => plain(vk::Format::R16_SFLOAT, 2),
        (0x1B, 1) => plain(vk::Format::R16_SNORM, 2),
        (0x1B, 2) => plain(vk::Format::R16_UNORM, 2),
        (0x1B, 3) => plain(vk::Format::R16_SINT, 2),
        (0x1B, 4) => plain(vk::Format::R16_UINT, 2),
        (0x1D, 2) => plain(vk::Format::R8_UNORM, 1),
        (0x1D, 1) => plain(vk::Format::R8_SNORM, 1),
        (0x1D, 3) => plain(vk::Format::R8_SINT, 1),
        (0x1D, 4) => plain(vk::Format::R8_UINT, 1),
        (0x20, _) => plain(vk::Format::E5B9G9R9_UFLOAT_PACK32, 4),
        (0x21, _) => plain(vk::Format::B10G11R11_UFLOAT_PACK32, 4),
        (0x24, _) => compressed(
            pick(
                vk::Format::BC1_RGBA_UNORM_BLOCK,
                vk::Format::BC1_RGBA_SRGB_BLOCK,
            ),
            8,
        ),
        (0x25, _) => compressed(
            pick(vk::Format::BC2_UNORM_BLOCK, vk::Format::BC2_SRGB_BLOCK),
            16,
        ),
        (0x26, _) => compressed(
            pick(vk::Format::BC3_UNORM_BLOCK, vk::Format::BC3_SRGB_BLOCK),
            16,
        ),
        (0x27, 1) => compressed(vk::Format::BC4_SNORM_BLOCK, 8),
        (0x27, _) => compressed(vk::Format::BC4_UNORM_BLOCK, 8),
        (0x28, 1) => compressed(vk::Format::BC5_SNORM_BLOCK, 16),
        (0x28, _) => compressed(vk::Format::BC5_UNORM_BLOCK, 16),
        _ => return None,
    })
}

/// The block footprint of an ASTC texture format (bits 0:6 of the format
/// word), whose blocks are 16 bytes.
pub fn astc_footprint(format: u32) -> Option<(u32, u32)> {
    Some(match format & 0x7F {
        0x40 => (4, 4),
        0x50 => (5, 4),
        0x41 => (5, 5),
        0x51 => (6, 5),
        0x42 => (6, 6),
        0x55 => (8, 5),
        0x52 => (8, 6),
        0x44 => (8, 8),
        0x56 => (10, 5),
        0x57 => (10, 6),
        0x53 => (10, 8),
        0x45 => (10, 10),
        0x54 => (12, 10),
        0x46 => (12, 12),
        _ => return None,
    })
}

/// A vertex attribute format from its size code (bits 21:26) and component
/// type (bits 27:29: 1 snorm, 2 unorm, 3 sint, 4 uint, 5 uscaled,
/// 6 sscaled, 7 float).
pub fn vertex_format(size: u32, kind: u32) -> Option<vk::Format> {
    use vk::Format as F;
    let by_kind = |formats: [F; 7]| formats.get(kind.checked_sub(1)? as usize).copied();
    match size {
        0x01 => by_kind([
            F::R32G32B32A32_SFLOAT,
            F::R32G32B32A32_SFLOAT,
            F::R32G32B32A32_SINT,
            F::R32G32B32A32_UINT,
            F::R32G32B32A32_SFLOAT,
            F::R32G32B32A32_SFLOAT,
            F::R32G32B32A32_SFLOAT,
        ]),
        0x02 => by_kind([
            F::R32G32B32_SFLOAT,
            F::R32G32B32_SFLOAT,
            F::R32G32B32_SINT,
            F::R32G32B32_UINT,
            F::R32G32B32_SFLOAT,
            F::R32G32B32_SFLOAT,
            F::R32G32B32_SFLOAT,
        ]),
        0x03 => by_kind([
            F::R16G16B16A16_SNORM,
            F::R16G16B16A16_UNORM,
            F::R16G16B16A16_SINT,
            F::R16G16B16A16_UINT,
            F::R16G16B16A16_USCALED,
            F::R16G16B16A16_SSCALED,
            F::R16G16B16A16_SFLOAT,
        ]),
        0x04 => by_kind([
            F::R32G32_SFLOAT,
            F::R32G32_SFLOAT,
            F::R32G32_SINT,
            F::R32G32_UINT,
            F::R32G32_SFLOAT,
            F::R32G32_SFLOAT,
            F::R32G32_SFLOAT,
        ]),
        0x05 => by_kind([
            F::R16G16B16_SNORM,
            F::R16G16B16_UNORM,
            F::R16G16B16_SINT,
            F::R16G16B16_UINT,
            F::R16G16B16_USCALED,
            F::R16G16B16_SSCALED,
            F::R16G16B16_SFLOAT,
        ]),
        0x0A => by_kind([
            F::R8G8B8A8_SNORM,
            F::R8G8B8A8_UNORM,
            F::R8G8B8A8_SINT,
            F::R8G8B8A8_UINT,
            F::R8G8B8A8_USCALED,
            F::R8G8B8A8_SSCALED,
            F::R8G8B8A8_UNORM,
        ]),
        0x0F => by_kind([
            F::R16G16_SNORM,
            F::R16G16_UNORM,
            F::R16G16_SINT,
            F::R16G16_UINT,
            F::R16G16_USCALED,
            F::R16G16_SSCALED,
            F::R16G16_SFLOAT,
        ]),
        0x12 => by_kind([
            F::R32_SFLOAT,
            F::R32_SFLOAT,
            F::R32_SINT,
            F::R32_UINT,
            F::R32_SFLOAT,
            F::R32_SFLOAT,
            F::R32_SFLOAT,
        ]),
        0x13 => by_kind([
            F::R8G8B8_SNORM,
            F::R8G8B8_UNORM,
            F::R8G8B8_SINT,
            F::R8G8B8_UINT,
            F::R8G8B8_USCALED,
            F::R8G8B8_SSCALED,
            F::R8G8B8_UNORM,
        ]),
        0x18 => by_kind([
            F::R8G8_SNORM,
            F::R8G8_UNORM,
            F::R8G8_SINT,
            F::R8G8_UINT,
            F::R8G8_USCALED,
            F::R8G8_SSCALED,
            F::R8G8_UNORM,
        ]),
        0x1B => by_kind([
            F::R16_SNORM,
            F::R16_UNORM,
            F::R16_SINT,
            F::R16_UINT,
            F::R16_USCALED,
            F::R16_SSCALED,
            F::R16_SFLOAT,
        ]),
        0x1D => by_kind([
            F::R8_SNORM,
            F::R8_UNORM,
            F::R8_SINT,
            F::R8_UINT,
            F::R8_USCALED,
            F::R8_SSCALED,
            F::R8_UNORM,
        ]),
        0x30 => Some(F::A2B10G10R10_UNORM_PACK32),
        0x31 => Some(F::B10G11R11_UFLOAT_PACK32),
        _ => None,
    }
}

/// The Vulkan topology a guest topology draws with. Quads are drawn as
/// triangle lists through a generated index buffer.
pub fn topology(topology: PrimitiveTopology) -> Option<vk::PrimitiveTopology> {
    Some(match topology {
        PrimitiveTopology::Points => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveTopology::Lines => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        PrimitiveTopology::Triangles | PrimitiveTopology::Quads => {
            vk::PrimitiveTopology::TRIANGLE_LIST
        }
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        PrimitiveTopology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
        // Tessellation is not translated yet. Draw the patch control mesh
        // as triangles: the coarse floor/object is visible instead of being
        // skipped entirely.
        PrimitiveTopology::Patches => vk::PrimitiveTopology::TRIANGLE_LIST,
        _ => return None,
    })
}

pub fn blend_op(op: u32) -> vk::BlendOp {
    match op {
        2 | 0x800A => vk::BlendOp::SUBTRACT,
        3 | 0x800B => vk::BlendOp::REVERSE_SUBTRACT,
        4 | 0x8007 => vk::BlendOp::MIN,
        5 | 0x8008 => vk::BlendOp::MAX,
        _ => vk::BlendOp::ADD,
    }
}

pub fn blend_factor(factor: u32) -> vk::BlendFactor {
    use vk::BlendFactor as B;
    match factor {
        1 | 0x4000 => B::ZERO,
        2 | 0x4001 => B::ONE,
        3 | 0x4300 => B::SRC_COLOR,
        4 | 0x4301 => B::ONE_MINUS_SRC_COLOR,
        5 | 0x4302 => B::SRC_ALPHA,
        6 | 0x4303 => B::ONE_MINUS_SRC_ALPHA,
        7 | 0x4304 => B::DST_ALPHA,
        8 | 0x4305 => B::ONE_MINUS_DST_ALPHA,
        9 | 0x4306 => B::DST_COLOR,
        0xA | 0x4307 => B::ONE_MINUS_DST_COLOR,
        0xB | 0x4308 => B::SRC_ALPHA_SATURATE,
        0x10 | 0xC900 => B::SRC1_COLOR,
        0x11 | 0xC901 => B::ONE_MINUS_SRC1_COLOR,
        0x12 | 0xC902 => B::SRC1_ALPHA,
        0x13 | 0xC903 => B::ONE_MINUS_SRC1_ALPHA,
        0x61 | 0xC001 => B::CONSTANT_COLOR,
        0x62 | 0xC002 => B::ONE_MINUS_CONSTANT_COLOR,
        0x63 | 0xC003 => B::CONSTANT_ALPHA,
        0x64 | 0xC004 => B::ONE_MINUS_CONSTANT_ALPHA,
        _ => B::ONE,
    }
}

pub fn address_mode(mode: u32) -> vk::SamplerAddressMode {
    match mode {
        0 => vk::SamplerAddressMode::REPEAT,
        1 | 5..=7 => vk::SamplerAddressMode::MIRRORED_REPEAT,
        3 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        _ => vk::SamplerAddressMode::CLAMP_TO_EDGE,
    }
}

/// A texture header swizzle source (0 zero, 2-5 r/g/b/a, 6-7 one).
pub fn swizzle(source: u32) -> vk::ComponentSwizzle {
    match source {
        2 => vk::ComponentSwizzle::R,
        3 => vk::ComponentSwizzle::G,
        4 => vk::ComponentSwizzle::B,
        5 => vk::ComponentSwizzle::A,
        6 | 7 => vk::ComponentSwizzle::ONE,
        _ => vk::ComponentSwizzle::ZERO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba8_textures_and_targets_map_to_rgba8() {
        // A8B8G8R8 with four unorm components, as Ryujinx's table keys it.
        let layout = texture_format(0x24908, false).expect("rgba8");
        assert_eq!(layout.format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!((layout.bytes, layout.block), (4, 1));
        assert_eq!(
            texture_format(0x24908, true).expect("srgb").format,
            vk::Format::R8G8B8A8_SRGB
        );
        assert_eq!(color_target_format(0xD5), Some(vk::Format::R8G8B8A8_UNORM));
        assert_eq!(
            color_target_format(0xCA),
            Some(vk::Format::R16G16B16A16_SFLOAT)
        );
        // Integer targets render, with integer fragment outputs.
        assert_eq!(color_target_format(0xD9), Some(vk::Format::R8G8B8A8_UINT));
        assert_eq!(color_target_format(0xE4), Some(vk::Format::R32_UINT));
        assert_eq!(color_target_kind(0xE4), AttributeType::UnsignedInt);
        assert_eq!(color_target_kind(0xD5), AttributeType::Float);
        // A pitch-linear target's size word is a row stride in bytes.
        assert_eq!(color_target_bytes(0xD6), Some(4));
        assert_eq!(color_target_bytes(0xC0), Some(16));
        assert_eq!(color_target_bytes(0xCC), Some(8));
        assert_eq!(color_target_bytes(0xEE), Some(2));
        assert_eq!(color_target_bytes(0xF3), Some(1));
        assert_eq!(color_target_bytes(0x1234), None);
        assert_eq!(pitch_linear_width(1600, 0xD6), Some(400));
        assert_eq!(pitch_linear_width(3, 0xD6), None);
    }

    #[test]
    fn depth_formats_and_comparisons() {
        assert_eq!(depth_format(0x16), Some(vk::Format::D32_SFLOAT_S8_UINT));
        assert!(has_stencil(vk::Format::D32_SFLOAT_S8_UINT));
        assert_eq!(aspects(vk::Format::D32_SFLOAT), vk::ImageAspectFlags::DEPTH);
        // OpenGL's GL_LEQUAL and the 1-based LessEqual are the same test.
        assert_eq!(compare_op(0x203), vk::CompareOp::LESS_OR_EQUAL);
        assert_eq!(compare_op(4), vk::CompareOp::LESS_OR_EQUAL);
    }

    #[test]
    fn bc_formats_are_four_by_four_blocks() {
        let bc1 = texture_format(0x24, false).expect("bc1");
        assert_eq!((bc1.bytes, bc1.block), (8, 4));
        let bc7 = texture_format(0x17, true).expect("bc7");
        assert_eq!(bc7.format, vk::Format::BC7_SRGB_BLOCK);
        assert_eq!(texture_format(0x40, false), None, "ASTC is not decoded yet");
    }

    #[test]
    fn texture_formats_cover_all_component_types() {
        let snorm16 = texture_format(0x03 | (1 << 7), false).expect("rgba16 snorm");
        assert_eq!(snorm16.format, vk::Format::R16G16B16A16_SNORM);
        let sint8 = texture_format(0x07 | (3 << 7), false).expect("rgba8 sint");
        assert_eq!(sint8.format, vk::Format::R8G8B8A8_SINT);
        let uint16 = texture_format(0x1B | (4 << 7), false).expect("r16 uint");
        assert_eq!(uint16.format, vk::Format::R16_UINT);
        let snorm8 = texture_format(0x1D | (1 << 7), false).expect("r8 snorm");
        assert_eq!(snorm8.format, vk::Format::R8_SNORM);
    }

    #[test]
    fn patches_fall_back_to_triangles() {
        assert_eq!(
            topology(recomp_gfx::maxwell::render::PrimitiveTopology::Patches),
            Some(vk::PrimitiveTopology::TRIANGLE_LIST)
        );
    }

    #[test]
    fn vertex_formats_follow_size_and_type() {
        assert_eq!(
            vertex_format(0x01, 7),
            Some(vk::Format::R32G32B32A32_SFLOAT)
        );
        assert_eq!(vertex_format(0x0A, 2), Some(vk::Format::R8G8B8A8_UNORM));
        assert_eq!(vertex_format(0x12, 4), Some(vk::Format::R32_UINT));
        assert_eq!(vertex_format(0x7F, 7), None);
    }

    #[test]
    fn blend_enums_accept_both_numberings() {
        assert_eq!(blend_factor(6), vk::BlendFactor::ONE_MINUS_SRC_ALPHA);
        assert_eq!(blend_factor(0x4303), vk::BlendFactor::ONE_MINUS_SRC_ALPHA);
        assert_eq!(blend_op(0x8006), vk::BlendOp::ADD);
        assert_eq!(blend_op(5), vk::BlendOp::MAX);
    }
}
