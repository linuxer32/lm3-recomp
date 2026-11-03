//! Texture and sampler descriptors (TIC and TSC entries, 32 bytes each) and
//! the block-linear layout guest textures are stored in.
//!
//! Field positions follow Ryujinx's `TextureDescriptor` and
//! `SamplerDescriptor`.

/// A texture header (TIC entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHeader {
    pub words: [u32; 8],
}

/// How a texture's texels are laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Rows of `stride` bytes.
    Pitch { stride: u32 },
    /// Block-linear: blocks `gobs_in_y` GOBs high.
    BlockLinear { gobs_in_y: u32 },
}

/// The texture targets a header names (bits 23:26 of word 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    D1,
    D2,
    D3,
    Cube,
    D1Array,
    D2Array,
    Buffer,
    D2NoMipmap,
    CubeArray,
}

impl TextureHeader {
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let mut words = [0u32; 8];
        for (index, word) in words.iter_mut().enumerate() {
            *word = u32::from_le_bytes(bytes.get(index * 4..index * 4 + 4)?.try_into().ok()?);
        }
        Some(Self { words })
    }

    /// The format word: format in bits 0:6, component types in 7:18.
    pub fn format(&self) -> u32 {
        self.words[0] & 0x8007_FFFF
    }

    /// Swizzle sources for r, g, b and a.
    pub fn swizzle(&self) -> [u32; 4] {
        let word = self.words[0];
        [
            (word >> 19) & 7,
            (word >> 22) & 7,
            (word >> 25) & 7,
            (word >> 28) & 7,
        ]
    }

    pub fn address(&self) -> u64 {
        u64::from(self.words[1]) | (u64::from(self.words[2] & 0xFFFF) << 32)
    }

    pub fn layout(&self) -> Layout {
        // Descriptor types 3 and 4 are block-linear; 2 is pitch.
        match (self.words[2] >> 21) & 7 {
            2 => Layout::Pitch {
                stride: (self.words[3] & 0xFFFF) << 5,
            },
            _ => Layout::BlockLinear {
                gobs_in_y: 1 << ((self.words[3] >> 3) & 7),
            },
        }
    }

    /// Mip levels the texture has.
    pub fn levels(&self) -> u32 {
        (self.words[3] >> 28) + 1
    }

    /// The first and last mip level a view of the texture uses (word 7).
    pub fn view_levels(&self) -> (u32, u32) {
        (self.words[7] & 0xF, (self.words[7] >> 4) & 0xF)
    }

    /// Layers of an array texture (a cube's six faces count once).
    pub fn array_layers(&self) -> u32 {
        ((self.words[5] >> 16) & 0x3FFF) + 1
    }

    /// Slices of a 3D texture (the field array textures count layers in).
    pub fn depth(&self) -> u32 {
        self.array_layers()
    }

    /// GOBs per block in z (block-linear 3D textures).
    pub fn gobs_in_z(&self) -> u32 {
        1 << ((self.words[3] >> 6) & 7)
    }

    pub fn width(&self) -> u32 {
        (self.words[4] & 0xFFFF) + 1
    }

    pub fn srgb(&self) -> bool {
        self.words[4] & (1 << 22) != 0
    }

    pub fn target(&self) -> Option<Target> {
        Some(match (self.words[4] >> 23) & 0xF {
            0 => Target::D1,
            1 => Target::D2,
            2 => Target::D3,
            3 => Target::Cube,
            4 => Target::D1Array,
            5 => Target::D2Array,
            6 => Target::Buffer,
            7 => Target::D2NoMipmap,
            8 => Target::CubeArray,
            _ => return None,
        })
    }

    pub fn height(&self) -> u32 {
        (self.words[5] & 0xFFFF) + 1
    }
}

/// A sampler (TSC entry), in the fields the backend uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerHeader {
    pub words: [u32; 8],
}

impl SamplerHeader {
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        TextureHeader::parse(bytes).map(|header| Self {
            words: header.words,
        })
    }

    /// Address modes for u, v and p.
    pub fn address_modes(&self) -> [u32; 3] {
        let word = self.words[0];
        [word & 7, (word >> 3) & 7, (word >> 6) & 7]
    }

    pub fn depth_compare(&self) -> Option<u32> {
        (self.words[0] & (1 << 9) != 0).then_some((self.words[0] >> 10) & 7)
    }

    /// Magnification and minification filters are linear.
    pub fn linear(&self) -> (bool, bool) {
        let word = self.words[1];
        ((word & 3) == 2, ((word >> 4) & 3) == 2)
    }

    /// The anisotropic filtering ratio (1 to 16).
    pub fn max_anisotropy(&self) -> f32 {
        f32::from(1u8 << ((self.words[0] >> 20) & 7).min(4))
    }

    /// The mip filter: 1 none, 2 nearest, 3 linear.
    pub fn mip_filter(&self) -> u32 {
        (self.words[1] >> 6) & 3
    }

    /// Minimum and maximum level of detail.
    pub fn lod_range(&self) -> (f32, f32) {
        let word = self.words[2];
        (
            (word & 0xFFF) as f32 / 256.0,
            ((word >> 12) & 0xFFF) as f32 / 256.0,
        )
    }
}

/// Bytes in a GOB row and rows in a GOB.
const GOB_WIDTH: u64 = 64;
const GOB_HEIGHT: u64 = 8;
const GOB_SIZE: u64 = GOB_WIDTH * GOB_HEIGHT;

/// The GOBs per block in y a surface of `rows` rows really uses: the
/// hardware halves the header's value while the surface fits in half a
/// block (small textures and mip levels).
pub fn fitted_gobs_in_y(rows: u64, gobs_in_y: u64) -> u64 {
    let mut gobs = gobs_in_y.max(1);
    while gobs > 1 && rows <= (gobs / 2) * GOB_HEIGHT {
        gobs /= 2;
    }
    gobs
}

/// The GOBs per block in z a 3D surface of `depth` slices really uses, as
/// `fitted_gobs_in_y` does for rows.
pub fn fitted_gobs_in_z(depth: u64, gobs_in_z: u64) -> u64 {
    let mut gobs = gobs_in_z.max(1);
    while gobs > 1 && depth <= gobs / 2 {
        gobs /= 2;
    }
    gobs
}

/// Bytes from one array layer to the next: every mip level's block-linear
/// size, the total rounded up to a whole block (as Ryujinx's size
/// calculator lays layers out). `row_bytes_at(level)` and `rows_at(level)`
/// give a level's row length in bytes and its rows of blocks.
pub fn layer_stride(
    levels: u32,
    gobs_in_y: u64,
    rows_at: impl Fn(u32) -> u64,
    row_bytes_at: impl Fn(u32) -> u64,
) -> u64 {
    let (_, size) = mip_levels(levels, gobs_in_y, &rows_at, row_bytes_at);
    let block = fitted_gobs_in_y(rows_at(0), gobs_in_y) * GOB_SIZE;
    size.next_multiple_of(block)
}

/// Where each mip level of a block-linear layer starts and the GOBs per
/// block in y it uses (levels follow each other, each in blocks fitted to
/// its rows), and the end of the last level.
pub fn mip_levels(
    levels: u32,
    gobs_in_y: u64,
    rows_at: impl Fn(u32) -> u64,
    row_bytes_at: impl Fn(u32) -> u64,
) -> (Vec<(u64, u64)>, u64) {
    let mut size = 0;
    let mut gobs = gobs_in_y.max(1);
    let mut starts = Vec::new();
    for level in 0..levels {
        let rows = rows_at(level);
        gobs = fitted_gobs_in_y(rows, gobs);
        starts.push((size, gobs));
        size += block_linear_size(row_bytes_at(level), rows, gobs);
    }
    (starts, size)
}

/// The byte offset of byte `x` of row `y` in a block-linear surface whose
/// rows are `row_bytes` long and whose blocks are `gobs_in_y` GOBs high.
#[cfg(test)]
pub fn block_linear_offset(x: u64, y: u64, row_bytes: u64, gobs_in_y: u64) -> u64 {
    Volume::new(row_bytes, 1, gobs_in_y, 1).offset(x, y, 0)
}

/// The number of bytes a block-linear surface of `rows` rows covers.
pub fn block_linear_size(row_bytes: u64, rows: u64, gobs_in_y: u64) -> u64 {
    Volume::new(row_bytes, rows, gobs_in_y, 1).size(1)
}

/// The block-linear layout (shared with the command processor's 2D engine).
pub use recomp_gfx::maxwell::blocklinear::Volume;

/// Untile `rows` rows of `row_bytes` from a block-linear surface into
/// tightly packed rows. Bytes outside `source` read as zero.
#[cfg(test)]
pub fn untile(source: &[u8], row_bytes: u64, rows: u64, gobs_in_y: u64) -> Vec<u8> {
    Volume::new(row_bytes, rows, gobs_in_y, 1).untile(source, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gob_layout_matches_the_hardware_pattern() {
        // The first 16 bytes of rows 0 and 1, then the second 16 of row 0.
        assert_eq!(block_linear_offset(0, 0, 64, 1), 0);
        assert_eq!(block_linear_offset(0, 1, 64, 1), 16);
        assert_eq!(block_linear_offset(16, 0, 64, 1), 32);
        assert_eq!(block_linear_offset(0, 2, 64, 1), 64);
        assert_eq!(block_linear_offset(32, 0, 64, 1), 256);
        // The next GOB down in a two-GOB block, then the next block across.
        assert_eq!(block_linear_offset(0, 8, 128, 2), 512);
        assert_eq!(block_linear_offset(64, 0, 128, 2), 1024);
        assert_eq!(block_linear_size(128, 16, 2), 2048);
    }

    #[test]
    fn untiling_inverts_the_layout() {
        let (row_bytes, rows, gobs) = (128u64, 16u64, 2u64);
        let mut tiled = vec![0u8; block_linear_size(row_bytes, rows, gobs) as usize];
        for y in 0..rows {
            for x in 0..row_bytes {
                tiled[block_linear_offset(x, y, row_bytes, gobs) as usize] = (x + y * 3) as u8;
            }
        }
        let linear = untile(&tiled, row_bytes, rows, gobs);
        for y in 0..rows {
            for x in 0..row_bytes {
                assert_eq!(linear[(y * row_bytes + x) as usize], (x + y * 3) as u8);
            }
        }
    }

    #[test]
    fn volumes_stack_slices_inside_a_block() {
        // 128-byte rows, 16 rows, blocks 2 GOBs high and 2 slices deep.
        let volume = Volume::new(128, 16, 2, 2);
        // Slice 1 of the first block follows its slice 0's two GOBs.
        assert_eq!(volume.offset(0, 0, 1), 1024);
        // The next block across holds 4 GOBs.
        assert_eq!(volume.offset(64, 0, 0), 2048);
        // Slices 2 and 3 start after the first slice of blocks.
        assert_eq!(volume.offset(0, 0, 2), 4096);
        assert_eq!(volume.size(4), 8192);
        let mut tiled = vec![0u8; volume.size(3) as usize];
        for z in 0..3 {
            for y in 0..16 {
                for x in 0..128 {
                    tiled[volume.offset(x, y, z) as usize] = (x + y * 5 + z * 7) as u8;
                }
            }
        }
        let linear = volume.untile(&tiled, 3);
        for z in 0..3u64 {
            for y in 0..16u64 {
                for x in 0..128u64 {
                    let at = ((z * 16 + y) * 128 + x) as usize;
                    assert_eq!(linear[at], (x + y * 5 + z * 7) as u8);
                }
            }
        }
        assert_eq!(fitted_gobs_in_z(32, 1), 1);
        assert_eq!(fitted_gobs_in_z(3, 8), 4);
    }

    #[test]
    fn small_surfaces_use_shorter_blocks() {
        assert_eq!(fitted_gobs_in_y(720, 16), 16);
        // 32 rows fit in four GOBs: a 16-GOB header block is halved twice.
        assert_eq!(fitted_gobs_in_y(32, 16), 4);
        assert_eq!(fitted_gobs_in_y(1, 16), 1);
        // One level of 64 rows of 256 bytes, 8-GOB blocks: 4 x 1 blocks of
        // 4 KiB, already whole blocks.
        assert_eq!(layer_stride(1, 8, |_| 64, |_| 256), 4 * 4096);
    }

    #[test]
    fn mip_levels_follow_each_other_in_fitted_blocks() {
        // A 256x256 RGBA8 texture, 16-GOB blocks: level 0 is 1024-byte rows
        // of 256 rows (two blocks high), level 1 512 bytes x 128 rows in one
        // 16-GOB block, level 2 256 x 64 in 8-GOB blocks.
        let (levels, end) = mip_levels(3, 16, |level| 256 >> level, |level| 1024 >> level);
        assert_eq!(levels[0], (0, 16));
        assert_eq!(levels[1], (256 * 1024, 16));
        assert_eq!(levels[2], (256 * 1024 + 128 * 512, 8));
        assert_eq!(end, 256 * 1024 + 128 * 512 + 64 * 256);
    }

    #[test]
    fn texture_headers_unpack_their_fields() {
        let mut words = [0u32; 8];
        words[0] = 0x24908 | (2 << 19) | (3 << 22) | (4 << 25) | (5 << 28);
        words[1] = 0x1234_5000;
        words[2] = 0x5 | (3 << 21);
        words[3] = 4 << 3;
        words[4] = 1279 | (1 << 23);
        words[5] = 719;
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let header = TextureHeader::parse(&bytes).expect("parses");
        assert_eq!(header.format(), 0x24908);
        assert_eq!(header.swizzle(), [2, 3, 4, 5]);
        assert_eq!(header.address(), 0x5_1234_5000);
        assert_eq!(header.layout(), Layout::BlockLinear { gobs_in_y: 16 });
        assert_eq!((header.width(), header.height()), (1280, 720));
        assert_eq!(header.target(), Some(Target::D2));
    }
}
