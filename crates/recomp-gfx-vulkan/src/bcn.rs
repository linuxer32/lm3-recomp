//! BCn blocks to plain texels, for devices that cannot sample BCn.
//!
//! Mobile GPUs leave `textureCompressionBC` off: the Adreno 650 lists no
//! BC format at all. An image made in a format the device does not
//! support is invalid, and on that phone the title's BC textures were
//! never shown (the loading screen's "Loading..." text) and then lost the
//! device at the title screen. Those textures are decoded here instead,
//! as ASTC is for desktop GPUs, to the plain format each BC format
//! decodes to: RGBA8 for BC1-3 and BC7, R8 and RG8 for BC4 and BC5 (signed
//! where they are), and RGBA16F for BC6H, which is HDR.
//!
//! BC7 blocks go through `texture2ddecoder`. The rest are decoded here:
//! that crate decodes BC2/BC3 colors in BC1's three-color mode, has no
//! signed BC4/BC5, and narrows BC6H to 8 bits.

use ash::vk;

/// A BCn block format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bcn {
    Bc1,
    Bc2,
    Bc3,
    Bc4 { signed: bool },
    Bc5 { signed: bool },
    Bc6h { signed: bool },
    Bc7,
}

impl Bcn {
    /// The BCn format `format` is and the plain format it decodes to
    /// (sRGB stays sRGB); `None` for a format that is not BCn.
    pub fn of(format: vk::Format) -> Option<(Self, vk::Format)> {
        use vk::Format as F;
        Some(match format {
            F::BC1_RGBA_UNORM_BLOCK => (Self::Bc1, F::R8G8B8A8_UNORM),
            F::BC1_RGBA_SRGB_BLOCK => (Self::Bc1, F::R8G8B8A8_SRGB),
            F::BC2_UNORM_BLOCK => (Self::Bc2, F::R8G8B8A8_UNORM),
            F::BC2_SRGB_BLOCK => (Self::Bc2, F::R8G8B8A8_SRGB),
            F::BC3_UNORM_BLOCK => (Self::Bc3, F::R8G8B8A8_UNORM),
            F::BC3_SRGB_BLOCK => (Self::Bc3, F::R8G8B8A8_SRGB),
            F::BC4_UNORM_BLOCK => (Self::Bc4 { signed: false }, F::R8_UNORM),
            F::BC4_SNORM_BLOCK => (Self::Bc4 { signed: true }, F::R8_SNORM),
            F::BC5_UNORM_BLOCK => (Self::Bc5 { signed: false }, F::R8G8_UNORM),
            F::BC5_SNORM_BLOCK => (Self::Bc5 { signed: true }, F::R8G8_SNORM),
            F::BC6H_UFLOAT_BLOCK => (Self::Bc6h { signed: false }, F::R16G16B16A16_SFLOAT),
            F::BC6H_SFLOAT_BLOCK => (Self::Bc6h { signed: true }, F::R16G16B16A16_SFLOAT),
            F::BC7_UNORM_BLOCK => (Self::Bc7, F::R8G8B8A8_UNORM),
            F::BC7_SRGB_BLOCK => (Self::Bc7, F::R8G8B8A8_SRGB),
            _ => return None,
        })
    }

    /// Bytes per 4x4 block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Bc1 | Self::Bc4 { .. } => 8,
            _ => 16,
        }
    }

    /// Bytes per decoded texel.
    pub fn texel_bytes(self) -> usize {
        match self {
            Self::Bc4 { .. } => 1,
            Self::Bc5 { .. } => 2,
            Self::Bc6h { .. } => 8,
            _ => 4,
        }
    }
}

/// `width` x `height` texels, tightly packed, from rows of 4x4 blocks
/// packed tightly. Blocks past the end of `blocks` decode as zero.
pub fn decode(kind: Bcn, blocks: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (width, height) = (width as usize, height as usize);
    let texel = kind.texel_bytes();
    let mut texels = vec![0u8; width * height * texel];
    let across = width.div_ceil(4);
    let count = across * height.div_ceil(4);
    let mut block_texels = [0u8; 16 * 8];
    for (index, block) in blocks
        .chunks_exact(kind.block_bytes())
        .take(count)
        .enumerate()
    {
        decode_block(kind, block, &mut block_texels);
        let (x, y) = (index % across * 4, index / across * 4);
        let columns = (width - x).min(4) * texel;
        for row in 0..(height - y).min(4) {
            let at = ((y + row) * width + x) * texel;
            texels[at..at + columns].copy_from_slice(&block_texels[row * 4 * texel..][..columns]);
        }
    }
    texels
}

/// One block's 16 texels, row by row, in the decoded format.
fn decode_block(kind: Bcn, block: &[u8], out: &mut [u8; 128]) {
    match kind {
        Bcn::Bc1 => colors(block, true, out),
        Bcn::Bc2 => {
            colors(&block[8..], false, out);
            for texel in 0..16 {
                let nibble = (block[texel / 2] >> (texel % 2 * 4)) & 0xF;
                out[texel * 4 + 3] = nibble * 0x11;
            }
        }
        Bcn::Bc3 => {
            colors(&block[8..], false, out);
            channel(block, false, out, 4, 3);
        }
        Bcn::Bc4 { signed } => channel(block, signed, out, 1, 0),
        Bcn::Bc5 { signed } => {
            channel(&block[..8], signed, out, 2, 0);
            channel(&block[8..], signed, out, 2, 1);
        }
        Bcn::Bc6h { signed } => bc6h(block, signed, out),
        Bcn::Bc7 => {
            let mut bgra = [0u32; 16];
            texture2ddecoder::decode_bc7_block(block, &mut bgra);
            for (texel, color) in out.as_chunks_mut::<4>().0.iter_mut().zip(bgra) {
                let [b, g, r, a] = color.to_le_bytes();
                texel.copy_from_slice(&[r, g, b, a]);
            }
        }
    }
}

/// A BC1 color block to RGBA8. BC2 and BC3 colors always use the four-color
/// mode; BC1's three-color mode makes index 3 transparent black.
fn colors(block: &[u8], bc1: bool, out: &mut [u8; 128]) {
    let expand = |color: u16| {
        let (r, g, b) = (color >> 11, (color >> 5) & 0x3F, color & 0x1F);
        [(r << 3 | r >> 2), (g << 2 | g >> 4), (b << 3 | b >> 2)]
    };
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let (e0, e1) = (expand(c0), expand(c1));
    let mix =
        |a: u16, b: u16, wa: u16, wb: u16| ((a * wa + b * wb + (wa + wb) / 2) / (wa + wb)) as u8;
    let mut palette = [[0u8; 4]; 4];
    for channel in 0..3 {
        let (a, b) = (e0[channel], e1[channel]);
        palette[0][channel] = a as u8;
        palette[1][channel] = b as u8;
        if c0 > c1 || !bc1 {
            palette[2][channel] = mix(a, b, 2, 1);
            palette[3][channel] = mix(a, b, 1, 2);
        } else {
            palette[2][channel] = mix(a, b, 1, 1);
        }
    }
    palette[0][3] = 255;
    palette[1][3] = 255;
    palette[2][3] = 255;
    palette[3][3] = if c0 > c1 || !bc1 { 255 } else { 0 };
    let indices = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for texel in 0..16 {
        let color = palette[(indices >> (texel * 2)) as usize & 3];
        out[texel * 4..texel * 4 + 4].copy_from_slice(&color);
    }
}

/// A BC4 block (BC3's alpha, BC5's halves) into byte `offset` of each
/// `stride`-byte texel: UNORM bytes, or SNORM bytes when `signed`.
fn channel(block: &[u8], signed: bool, out: &mut [u8; 128], stride: usize, offset: usize) {
    let endpoint = |byte: u8| {
        if signed {
            f32::from((byte as i8).max(-127))
        } else {
            f32::from(byte)
        }
    };
    let (a, b) = (endpoint(block[0]), endpoint(block[1]));
    let mut values = [a, b, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    if a > b {
        for step in 1..7 {
            values[step + 1] = (a * (7 - step) as f32 + b * step as f32) / 7.0;
        }
    } else {
        for step in 1..5 {
            values[step + 1] = (a * (5 - step) as f32 + b * step as f32) / 5.0;
        }
        values[6] = if signed { -127.0 } else { 0.0 };
        values[7] = if signed { 127.0 } else { 255.0 };
    }
    let mut indices = [0u8; 8];
    indices[..6].copy_from_slice(&block[2..8]);
    let indices = u64::from_le_bytes(indices);
    for texel in 0..16 {
        let value = values[(indices >> (texel * 3)) as usize & 7].round();
        out[texel * stride + offset] = if signed {
            value as i8 as u8
        } else {
            value as u8
        };
    }
}

/// How a BC6H mode stores its endpoints: transformed (deltas from the
/// first), two regions, endpoint bits and delta bits per channel.
struct Bc6hMode {
    transformed: bool,
    two_regions: bool,
    endpoint_bits: u32,
    delta_bits: [u32; 3],
    /// The header after the mode bits, as runs of (endpoint field, bits,
    /// shift): fields are r0..r3, g0..g3, b0..b3 (w, x, y, z).
    runs: &'static [(u8, u8, u8)],
}

/// BC6H's 14 modes, by their 2-bit (0, 1) or 5-bit mode number.
#[rustfmt::skip]
fn bc6h_mode(number: u32) -> Option<Bc6hMode> {
    let mode = |transformed: bool, two_regions: bool, endpoint_bits: u32, delta_bits: [u32; 3], runs: &'static [(u8, u8, u8)]| {
        Bc6hMode { transformed, two_regions, endpoint_bits, delta_bits, runs }
    };
    Some(match number {
        0x00 => mode(true, true, 10, [5, 5, 5], &[
            (6, 1, 4), (10, 1, 4), (11, 1, 4), (0, 10, 0), (4, 10, 0), (8, 10, 0), (1, 5, 0),
            (7, 1, 4), (6, 4, 0), (5, 5, 0), (11, 1, 0), (7, 4, 0), (9, 5, 0), (11, 1, 1),
            (10, 4, 0), (2, 5, 0), (11, 1, 2), (3, 5, 0), (11, 1, 3),
        ]),
        0x01 => mode(true, true, 7, [6, 6, 6], &[
            (6, 1, 5), (7, 1, 4), (7, 1, 5), (0, 7, 0), (11, 1, 0), (11, 1, 1), (10, 1, 4),
            (4, 7, 0), (10, 1, 5), (11, 1, 2), (6, 1, 4), (8, 7, 0), (11, 1, 3), (11, 1, 5),
            (11, 1, 4), (1, 6, 0), (6, 4, 0), (5, 6, 0), (7, 4, 0), (9, 6, 0), (10, 4, 0),
            (2, 6, 0), (3, 6, 0),
        ]),
        0x02 => mode(true, true, 11, [5, 4, 4], &[
            (0, 10, 0), (4, 10, 0), (8, 10, 0), (1, 5, 0), (0, 1, 10), (6, 4, 0), (5, 4, 0),
            (4, 1, 10), (11, 1, 0), (7, 4, 0), (9, 4, 0), (8, 1, 10), (11, 1, 1), (10, 4, 0),
            (2, 5, 0), (11, 1, 2), (3, 5, 0), (11, 1, 3),
        ]),
        0x03 => mode(false, false, 10, [10, 10, 10], &[
            (0, 10, 0), (4, 10, 0), (8, 10, 0), (1, 10, 0), (5, 10, 0), (9, 10, 0),
        ]),
        0x06 => mode(true, true, 11, [4, 5, 4], &[
            (0, 10, 0), (4, 10, 0), (8, 10, 0), (1, 4, 0), (0, 1, 10), (7, 1, 4), (6, 4, 0),
            (5, 5, 0), (4, 1, 10), (7, 4, 0), (9, 4, 0), (8, 1, 10), (11, 1, 1), (10, 4, 0),
            (2, 4, 0), (11, 1, 0), (11, 1, 2), (3, 4, 0), (6, 1, 4), (11, 1, 3),
        ]),
        0x07 => mode(true, false, 11, [9, 9, 9], &[
            (0, 10, 0), (4, 10, 0), (8, 10, 0), (1, 9, 0), (0, 1, 10), (5, 9, 0), (4, 1, 10),
            (9, 9, 0), (8, 1, 10),
        ]),
        0x0A => mode(true, true, 11, [4, 4, 5], &[
            (0, 10, 0), (4, 10, 0), (8, 10, 0), (1, 4, 0), (0, 1, 10), (10, 1, 4), (6, 4, 0),
            (5, 4, 0), (4, 1, 10), (11, 1, 0), (7, 4, 0), (9, 5, 0), (8, 1, 10), (10, 4, 0),
            (2, 4, 0), (11, 1, 1), (11, 1, 2), (3, 4, 0), (11, 1, 4), (11, 1, 3),
        ]),
        0x0B => mode(true, false, 12, [8, 8, 8], &[
            (0, 10, 0), (4, 10, 0), (8, 10, 0), (1, 8, 0), (0, 1, 11), (0, 1, 10), (5, 8, 0),
            (4, 1, 11), (4, 1, 10), (9, 8, 0), (8, 1, 11), (8, 1, 10),
        ]),
        0x0E => mode(true, true, 9, [5, 5, 5], &[
            (0, 9, 0), (10, 1, 4), (4, 9, 0), (6, 1, 4), (8, 9, 0), (11, 1, 4), (1, 5, 0),
            (7, 1, 4), (6, 4, 0), (5, 5, 0), (11, 1, 0), (7, 4, 0), (9, 5, 0), (11, 1, 1),
            (10, 4, 0), (2, 5, 0), (11, 1, 2), (3, 5, 0), (11, 1, 3),
        ]),
        0x0F => mode(true, false, 16, [4, 4, 4], &[
            (0, 10, 0), (4, 10, 0), (8, 10, 0), (1, 4, 0), (0, 1, 15), (0, 1, 14), (0, 1, 13),
            (0, 1, 12), (0, 1, 11), (0, 1, 10), (5, 4, 0), (4, 1, 15), (4, 1, 14), (4, 1, 13),
            (4, 1, 12), (4, 1, 11), (4, 1, 10), (9, 4, 0), (8, 1, 15), (8, 1, 14), (8, 1, 13),
            (8, 1, 12), (8, 1, 11), (8, 1, 10),
        ]),
        0x12 => mode(true, true, 8, [6, 5, 5], &[
            (0, 8, 0), (7, 1, 4), (10, 1, 4), (4, 8, 0), (11, 1, 2), (6, 1, 4), (8, 8, 0),
            (11, 1, 3), (11, 1, 4), (1, 6, 0), (6, 4, 0), (5, 5, 0), (11, 1, 0), (7, 4, 0),
            (9, 5, 0), (11, 1, 1), (10, 4, 0), (2, 6, 0), (3, 6, 0),
        ]),
        0x16 => mode(true, true, 8, [5, 6, 5], &[
            (0, 8, 0), (11, 1, 0), (10, 1, 4), (4, 8, 0), (6, 1, 5), (6, 1, 4), (8, 8, 0),
            (7, 1, 5), (11, 1, 4), (1, 5, 0), (7, 1, 4), (6, 4, 0), (5, 6, 0), (7, 4, 0),
            (9, 5, 0), (11, 1, 1), (10, 4, 0), (2, 5, 0), (11, 1, 2), (3, 5, 0), (11, 1, 3),
        ]),
        0x1A => mode(true, true, 8, [5, 5, 6], &[
            (0, 8, 0), (11, 1, 1), (10, 1, 4), (4, 8, 0), (10, 1, 5), (6, 1, 4), (8, 8, 0),
            (11, 1, 5), (11, 1, 4), (1, 5, 0), (7, 1, 4), (6, 4, 0), (5, 5, 0), (11, 1, 0),
            (7, 4, 0), (9, 6, 0), (10, 4, 0), (2, 5, 0), (11, 1, 2), (3, 5, 0), (11, 1, 3),
        ]),
        0x1E => mode(false, true, 6, [6, 6, 6], &[
            (0, 6, 0), (7, 1, 4), (11, 1, 0), (11, 1, 1), (10, 1, 4), (4, 6, 0), (6, 1, 5),
            (10, 1, 5), (11, 1, 2), (6, 1, 4), (8, 6, 0), (7, 1, 5), (11, 1, 3), (11, 1, 5),
            (11, 1, 4), (1, 6, 0), (6, 4, 0), (5, 6, 0), (7, 4, 0), (9, 6, 0), (10, 4, 0),
            (2, 6, 0), (3, 6, 0),
        ]),
        _ => return None,
    })
}

/// The 2-region partitions BC6H shares with BC7 (bit `i`: texel `i`'s
/// region) and each one's second anchor texel.
const PARTITIONS: [u16; 32] = [
    0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80, 0xC800, 0xFFEC, 0xFE80, 0xE800,
    0xFFE8, 0xFF00, 0xFFF0, 0xF000, 0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
    0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
];
const ANCHORS: [u8; 32] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2,
];
const WEIGHTS_3: [i32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHTS_4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// A BC6H block to RGBA16F (alpha 1). Reserved modes decode as black.
fn bc6h(block: &[u8], signed: bool, out: &mut [u8; 128]) {
    let bits = u128::from_le_bytes(block[..16].try_into().expect("16-byte block"));
    let mut at = 0;
    let mut read = |count: u32| {
        let value = ((bits >> at) & ((1u128 << count) - 1)) as i32;
        at += count;
        value
    };
    let mut number = read(2) as u32;
    if number & 2 != 0 {
        number |= (read(3) as u32) << 2;
    }
    const ONE: [u8; 2] = 0x3C00u16.to_le_bytes();
    let Some(mode) = bc6h_mode(number) else {
        for texel in out.as_chunks_mut::<8>().0 {
            texel.copy_from_slice(&[0, 0, 0, 0, 0, 0, ONE[0], ONE[1]]);
        }
        return;
    };
    let mut fields = [0i32; 12];
    for &(field, count, shift) in mode.runs {
        fields[usize::from(field)] |= read(u32::from(count)) << shift;
    }
    let extend = |value: i32, bits: u32| (value << (32 - bits)) >> (32 - bits);
    let endpoints = if mode.two_regions { 4 } else { 2 };
    let mut unquantized = [[0i32; 3]; 4];
    for channel in 0..3 {
        let field = &mut fields[channel * 4..channel * 4 + 4];
        if signed {
            field[0] = extend(field[0], mode.endpoint_bits);
        }
        let base = field[0];
        for value in field.iter_mut().take(endpoints).skip(1) {
            if signed || mode.transformed {
                *value = extend(*value, mode.delta_bits[channel]);
            }
            if mode.transformed {
                *value = (*value + base) & ((1 << mode.endpoint_bits) - 1);
                if signed {
                    *value = extend(*value, mode.endpoint_bits);
                }
            }
        }
        for (endpoint, value) in field.iter().take(endpoints).enumerate() {
            unquantized[endpoint][channel] = unquantize(*value, mode.endpoint_bits, signed);
        }
    }
    let partition = if mode.two_regions {
        read(5) as usize
    } else {
        0
    };
    let weights: &[i32] = if mode.two_regions {
        &WEIGHTS_3
    } else {
        &WEIGHTS_4
    };
    let index_bits = if mode.two_regions { 3 } else { 4 };
    for texel in 0..16 {
        let region = if mode.two_regions {
            usize::from(PARTITIONS[partition] >> texel & 1)
        } else {
            0
        };
        let anchor = texel == 0 || (mode.two_regions && texel == usize::from(ANCHORS[partition]));
        let weight = weights[read(index_bits - u32::from(anchor)) as usize];
        let (low, high) = (unquantized[region * 2], unquantized[region * 2 + 1]);
        let texel = &mut out[texel * 8..texel * 8 + 8];
        for channel in 0..3 {
            let value = (low[channel] * (64 - weight) + high[channel] * weight + 32) >> 6;
            texel[channel * 2..channel * 2 + 2].copy_from_slice(&half(value, signed).to_le_bytes());
        }
        texel[6..].copy_from_slice(&ONE);
    }
}

/// An endpoint widened to 16 bits (signed: 15 bits and the sign), as the
/// format's decoding defines.
fn unquantize(value: i32, bits: u32, signed: bool) -> i32 {
    if !signed {
        return if bits >= 15 {
            value
        } else if value == 0 {
            0
        } else if value == (1 << bits) - 1 {
            0xFFFF
        } else {
            ((value << 16) + 0x8000) >> bits
        };
    }
    if bits >= 16 {
        return value;
    }
    let magnitude = value.abs();
    let widened = if magnitude == 0 {
        0
    } else if magnitude >= (1 << (bits - 1)) - 1 {
        0x7FFF
    } else {
        ((magnitude << 15) + 0x4000) >> (bits - 1)
    };
    if value < 0 {
        -widened
    } else {
        widened
    }
}

/// An interpolated value scaled to the bits of a half float.
fn half(value: i32, signed: bool) -> u16 {
    if !signed {
        return ((value * 31) >> 6) as u16;
    }
    let magnitude = (value.abs() * 31) >> 5;
    if value < 0 {
        0x8000 | magnitude as u16
    } else {
        magnitude as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc1_three_color_blocks_have_a_transparent_index() {
        // color0 (black) <= color1 (white): index 2 is the mean, 3 clear.
        let block = [0x00, 0x00, 0xFF, 0xFF, 0b1110_0100, 0, 0, 0];
        let texels = decode(Bcn::Bc1, &block, 4, 1);
        assert_eq!(&texels[..4], &[0, 0, 0, 255]);
        assert_eq!(&texels[4..8], &[255, 255, 255, 255]);
        assert_eq!(&texels[8..12], &[128, 128, 128, 255]);
        assert_eq!(&texels[12..16], &[0, 0, 0, 0]);
        // BC3's colors never use that mode: index 3 is two thirds white.
        let mut bc3 = [0u8; 16];
        bc3[0] = 255;
        bc3[8..16].copy_from_slice(&block);
        let texels = decode(Bcn::Bc3, &bc3, 4, 1);
        assert_eq!(&texels[12..16], &[170, 170, 170, 255]);
    }

    #[test]
    fn signed_bc4_endpoints_clamp_to_minus_127() {
        // Endpoints -128 (read as -127) and 127; -127 is not above 127, so
        // this is the six-value mode, whose index 6 is -127 and 7 is 127.
        // Indices 0, 1, 6, 7.
        let indices: u64 = 1 << 3 | 6 << 6 | 7 << 9;
        let mut block = [0x80, 0x7F, 0, 0, 0, 0, 0, 0];
        block[2..].copy_from_slice(&indices.to_le_bytes()[..6]);
        let texels = decode(Bcn::Bc4 { signed: true }, &block, 4, 1);
        let texels: Vec<i8> = texels.iter().map(|&byte| byte as i8).collect();
        assert_eq!(texels, [-127, 127, -127, 127]);
    }

    #[test]
    fn bc6h_mode_3_reads_its_ten_bit_endpoints() {
        // Mode 0x03 (one region, plain 10-bit endpoints), every index 0:
        // each texel is the first endpoint, here (1023, 512, 0).
        let header: u128 = 0x03 | 1023 << 5 | 512 << 15;
        let texels = decode(Bcn::Bc6h { signed: false }, &header.to_le_bytes(), 1, 1);
        let channel = |at: usize| u16::from_le_bytes([texels[at], texels[at + 1]]);
        // 1023 is the largest 10-bit value: 0xFFFF, scaled to 0x7BFF (65504).
        assert_eq!(channel(0), 0x7BFF);
        assert_eq!(
            channel(2),
            (((((512 << 16) + 0x8000) >> 10) * 31) >> 6) as u16
        );
        assert_eq!(channel(4), 0);
        assert_eq!(channel(6), 0x3C00);
    }

    #[test]
    fn partial_blocks_keep_only_the_texels_inside() {
        let block = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0];
        let texels = decode(Bcn::Bc1, &[block, block].concat(), 6, 2);
        assert_eq!(texels.len(), 6 * 2 * 4);
        assert!(texels.chunks(4).all(|texel| texel == [255, 255, 255, 255]));
    }

    /// `BCN_CHECK_DIR=<dir>`: compare with another decoder's output for the
    /// blocks there (`<format>.blocks`, and `<format>.texels` as 4N x 4
    /// texels: RGBA8, L8 for BC4, RGB8 for BC5 and BC6H, whose halves are
    /// clamped to 0..1 and scaled to bytes), within 1 for rounding. BC7's
    /// reserved mode (a zero first byte) is left out: it decodes to
    /// transparent black, which Pillow, the decoder these were checked
    /// against, makes opaque.
    #[test]
    fn bcn_matches_reference_decodes() {
        let Some(dir) = std::env::var_os("BCN_CHECK_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        for (name, kind, channels) in [
            ("bc1", Bcn::Bc1, 4),
            ("bc2", Bcn::Bc2, 4),
            ("bc3", Bcn::Bc3, 4),
            ("bc4", Bcn::Bc4 { signed: false }, 1),
            ("bc5", Bcn::Bc5 { signed: false }, 3),
            ("bc6h", Bcn::Bc6h { signed: false }, 3),
            ("bc7", Bcn::Bc7, 4),
        ] {
            let blocks = std::fs::read(dir.join(format!("{name}.blocks"))).expect("blocks");
            let expected = std::fs::read(dir.join(format!("{name}.texels"))).expect("texels");
            let count = blocks.len() / kind.block_bytes();
            let ours = decode(kind, &blocks, 4 * count as u32, 4);
            let texel = kind.texel_bytes();
            let mut worst = 0;
            let mut differing = 0;
            for (index, wanted) in expected.chunks(channels).enumerate() {
                let block = index % (4 * count) / 4;
                if kind == Bcn::Bc7 && blocks[block * 16] == 0 {
                    continue;
                }
                let got = &ours[index * texel..][..texel];
                for (channel, &wanted) in wanted.iter().enumerate() {
                    let value = match kind {
                        Bcn::Bc6h { .. } => {
                            let half = u16::from_le_bytes([got[channel * 2], got[channel * 2 + 1]]);
                            (half_to_f32(half).clamp(0.0, 1.0) * 255.0) as u8
                        }
                        Bcn::Bc5 { .. } if channel == 2 => 0,
                        _ => got[channel],
                    };
                    let difference = value.abs_diff(wanted);
                    if difference > 1 {
                        differing += 1;
                        if differing <= 5 {
                            eprintln!("{name}: block {block} texel {index} channel {channel}: {value} vs {wanted}");
                        }
                    }
                    worst = worst.max(difference);
                }
            }
            eprintln!("{name}: {count} blocks, largest difference {worst}, {differing} channels off by more than 1");
            assert_eq!(differing, 0, "{name} differs");
        }
    }

    fn half_to_f32(half: u16) -> f32 {
        let exponent = (half >> 10) & 0x1F;
        let mantissa = f32::from(half & 0x3FF);
        let magnitude = match exponent {
            0 => mantissa * 2f32.powi(-24),
            31 => f32::INFINITY,
            _ => (1.0 + mantissa / 1024.0) * 2f32.powi(i32::from(exponent) - 15),
        };
        if half & 0x8000 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }
}
