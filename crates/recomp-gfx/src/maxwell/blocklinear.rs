//! The block-linear layout of Maxwell surfaces: 64-byte by 8-row GOBs,
//! grouped into blocks `gobs_in_y` GOBs high and `gobs_in_z` slices deep.
//! Textures, render targets and 2D engine copies use it.

/// Bytes in a GOB row and rows in a GOB.
pub const GOB_WIDTH: u64 = 64;
pub const GOB_HEIGHT: u64 = 8;
pub const GOB_SIZE: u64 = GOB_WIDTH * GOB_HEIGHT;

/// A block-linear surface's shape: rows of `row_bytes`, blocks `gobs_in_y`
/// GOBs high and `gobs_in_z` slices deep. A block is its z slices one after
/// another, each `gobs_in_y` GOBs; blocks run along a row, then down, then
/// through the slices of blocks.
#[derive(Debug, Clone, Copy)]
pub struct Volume {
    row_bytes: u64,
    rows: u64,
    gobs_in_y: u64,
    gobs_in_z: u64,
}

impl Volume {
    pub fn new(row_bytes: u64, rows: u64, gobs_in_y: u64, gobs_in_z: u64) -> Self {
        Self {
            row_bytes,
            rows,
            gobs_in_y: gobs_in_y.max(1),
            gobs_in_z: gobs_in_z.max(1),
        }
    }

    fn block_bytes(&self) -> u64 {
        GOB_SIZE * self.gobs_in_y * self.gobs_in_z
    }

    /// Bytes in one row of blocks.
    fn block_row_bytes(&self) -> u64 {
        self.row_bytes.div_ceil(GOB_WIDTH) * self.block_bytes()
    }

    /// Bytes in one slice of blocks (`gobs_in_z` slices of the surface).
    fn block_slice_bytes(&self) -> u64 {
        self.rows.div_ceil(GOB_HEIGHT * self.gobs_in_y) * self.block_row_bytes()
    }

    /// The bytes `depth` slices cover.
    pub fn size(&self, depth: u64) -> u64 {
        depth.div_ceil(self.gobs_in_z) * self.block_slice_bytes()
    }

    /// The offset of byte `x` of row `y` of slice `z`.
    pub fn offset(&self, x: u64, y: u64, z: u64) -> u64 {
        let block_height = GOB_HEIGHT * self.gobs_in_y;
        let (gx, gy) = (x % GOB_WIDTH, y % GOB_HEIGHT);
        // Within a GOB: two 32-byte halves of four 2-row, 16-byte sectors.
        let within =
            (gx / 32) * 256 + (gy / 2) * 64 + ((gx % 32) / 16) * 32 + (gy % 2) * 16 + gx % 16;
        (z / self.gobs_in_z) * self.block_slice_bytes()
            + (y / block_height) * self.block_row_bytes()
            + (x / GOB_WIDTH) * self.block_bytes()
            + (z % self.gobs_in_z) * self.gobs_in_y * GOB_SIZE
            + ((y % block_height) / GOB_HEIGHT) * GOB_SIZE
            + within
    }

    /// Untile `depth` slices into tightly packed rows, slice after slice.
    /// Bytes outside `source` read as zero.
    pub fn untile(&self, source: &[u8], depth: u64) -> Vec<u8> {
        let (row_bytes, rows) = (self.row_bytes, self.rows);
        let mut out = vec![0u8; (row_bytes * rows * depth) as usize];
        for z in 0..depth {
            for y in 0..rows {
                // Sixteen-byte runs stay contiguous in the GOB layout.
                let mut x = 0;
                while x < row_bytes {
                    let run = (16 - x % 16).min(row_bytes - x);
                    let from = self.offset(x, y, z) as usize;
                    let to = ((z * rows + y) * row_bytes + x) as usize;
                    if let Some(bytes) = source.get(from..from + run as usize) {
                        out[to..to + run as usize].copy_from_slice(bytes);
                    }
                    x += run;
                }
            }
        }
        out
    }
}
