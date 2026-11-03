//! The 2D engine (class 0x902D): scaled copies between surfaces.
//!
//! A copy names a destination and a source surface (address, format,
//! pitch-linear or block-linear, size), a destination rectangle, the source
//! origin and the source step per destination pixel (32.32 fixed point);
//! writing the source origin's integer y starts it. Register numbers follow
//! NVIDIA's class header (`cl902d.h`), as Ryujinx's `TwodClassState`.

use super::blocklinear::Volume;
use super::render::GpuMemory;

pub const CLASS_FERMI_TWOD: u32 = 0x902D;

/// Register (method) numbers.
pub mod method {
    pub const DST: u32 = 0x080;
    pub const SRC: u32 = 0x08C;
    pub const PIXELS_FROM_MEMORY: u32 = 0x22C;
    /// Writing the source origin's integer y starts the copy.
    pub const SRC_Y0_INT: u32 = 0x237;
    /// Sample mode: bit 4 filters linearly.
    pub const SAMPLE_MODE: u32 = 0x223;
}

const REGISTERS: usize = 0x300;

/// One side of a copy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Surface {
    pub address: u64,
    /// The color format, in the render target numbering.
    pub format: u32,
    pub linear: bool,
    pub pitch: u32,
    pub width: u32,
    pub height: u32,
    /// Block-linear blocks: GOBs high and slices deep.
    pub gobs_in_y: u32,
    pub gobs_in_z: u32,
    /// Slices of a 3D surface, and the one copied to or from.
    pub depth: u32,
    pub layer: u32,
}

impl Surface {
    /// Where pixel (`x`, `y`) of `bytes_per_pixel` bytes starts.
    fn offset(&self, x: u32, y: u32, bytes_per_pixel: u32) -> u64 {
        if self.linear {
            u64::from(y) * u64::from(self.pitch) + u64::from(x * bytes_per_pixel)
        } else {
            Volume::new(
                u64::from(self.width * bytes_per_pixel),
                u64::from(self.height),
                u64::from(self.gobs_in_y),
                u64::from(self.gobs_in_z),
            )
            .offset(
                u64::from(x * bytes_per_pixel),
                u64::from(y),
                u64::from(self.layer),
            )
        }
    }
}

/// Bytes per pixel of a color format (in the render target numbering).
pub fn bytes_per_pixel(format: u32) -> Option<u32> {
    Some(match format {
        0xC0..=0xC5 => 16,
        0xC6..=0xCE => 8,
        0xCF..=0xE7 | 0xF9 | 0xFA => 4,
        0xE8..=0xF2 | 0xF8 => 2,
        0xF3..=0xF7 => 1,
        _ => return None,
    })
}

/// Run a copy in guest memory: both surfaces as the CPU sees them, pixels
/// picked by nearest sampling. For copies whose source the GPU did not
/// render (uploads: LM3 writes its 3D color lookup table one slice at a
/// time from a linear buffer). False when the formats are unknown or differ
/// in size.
pub fn copy_in_memory(memory: &mut dyn GpuMemory, blit: &Blit) -> bool {
    let (Some(bytes), Some(source_bytes)) = (
        bytes_per_pixel(blit.dst.format),
        bytes_per_pixel(blit.src.format),
    ) else {
        return false;
    };
    if bytes != source_bytes {
        return false;
    }
    let [x0, y0, width, height] = blit.dst_rect;
    for row in 0..height {
        let source_y = (blit.src_origin[1] + f64::from(row) * blit.src_step[1]).max(0.0) as u32;
        if source_y >= blit.src.height || y0 + row >= blit.dst.height {
            continue;
        }
        for column in 0..width {
            let source_x =
                (blit.src_origin[0] + f64::from(column) * blit.src_step[0]).max(0.0) as u32;
            if source_x >= blit.src.width || x0 + column >= blit.dst.width {
                continue;
            }
            let from = blit.src.address + blit.src.offset(source_x, source_y, bytes);
            let to = blit.dst.address + blit.dst.offset(x0 + column, y0 + row, bytes);
            if let Some(pixel) = memory.read_gpu(from, bytes as usize) {
                memory.write_gpu(to, &pixel);
            }
        }
    }
    true
}

/// A copy, as a backend needs it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blit {
    pub dst: Surface,
    pub src: Surface,
    /// Destination rectangle: x, y, width, height.
    pub dst_rect: [u32; 4],
    /// Source origin and step per destination pixel.
    pub src_origin: [f64; 2],
    pub src_step: [f64; 2],
    pub linear_filter: bool,
}

#[derive(Debug, Clone)]
pub struct TwodState {
    registers: Vec<u32>,
}

impl Default for TwodState {
    fn default() -> Self {
        Self {
            registers: vec![0; REGISTERS],
        }
    }
}

impl TwodState {
    pub fn write(&mut self, method: u32, data: u32) {
        if let Some(register) = self.registers.get_mut(method as usize) {
            *register = data;
        }
    }

    fn read(&self, method: u32) -> u32 {
        self.registers.get(method as usize).copied().unwrap_or(0)
    }

    /// Whether this module uses a register.
    pub fn known(method: u32) -> bool {
        (method::DST..method::DST + 10).contains(&method)
            || (method::SRC..method::SRC + 10).contains(&method)
            || (method::PIXELS_FROM_MEMORY..=method::SRC_Y0_INT).contains(&method)
            || method == method::SAMPLE_MODE
    }

    /// FORMAT, LINEAR, BLOCK_SIZE, DEPTH, LAYER, PITCH, WIDTH, HEIGHT,
    /// ADDRESS_UPPER, ADDRESS_LOWER from `first`.
    fn surface(&self, first: u32) -> Surface {
        let block = self.read(first + 2);
        Surface {
            format: self.read(first),
            linear: self.read(first + 1) & 1 != 0,
            gobs_in_y: 1 << ((block >> 4) & 0xF),
            gobs_in_z: 1 << ((block >> 8) & 0xF),
            depth: self.read(first + 3),
            layer: self.read(first + 4),
            pitch: self.read(first + 5),
            width: self.read(first + 6),
            height: self.read(first + 7),
            address: (u64::from(self.read(first + 8) & 0xFF) << 32)
                | u64::from(self.read(first + 9)),
        }
    }

    /// The copy the registers describe.
    pub fn blit(&self) -> Blit {
        let base = method::PIXELS_FROM_MEMORY;
        let fixed = |fraction: u32, integer: u32| {
            f64::from(integer as i32) + f64::from(fraction) / 4_294_967_296.0
        };
        Blit {
            dst: self.surface(method::DST),
            src: self.surface(method::SRC),
            dst_rect: [
                self.read(base),
                self.read(base + 1),
                self.read(base + 2),
                self.read(base + 3),
            ],
            src_step: [
                fixed(self.read(base + 4), self.read(base + 5)),
                fixed(self.read(base + 6), self.read(base + 7)),
            ],
            src_origin: [
                fixed(self.read(base + 8), self.read(base + 9)),
                fixed(self.read(base + 10), self.read(base + 11)),
            ],
            linear_filter: self.read(method::SAMPLE_MODE) & 0x10 != 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guest memory as one buffer from address 0.
    struct Bytes(Vec<u8>);

    impl GpuMemory for Bytes {
        fn read_gpu(&self, address: u64, len: usize) -> Option<Vec<u8>> {
            let at = usize::try_from(address).ok()?;
            self.0.get(at..at + len).map(<[u8]>::to_vec)
        }

        fn write_gpu(&mut self, address: u64, bytes: &[u8]) -> bool {
            let at = address as usize;
            match self.0.get_mut(at..at + bytes.len()) {
                Some(target) => {
                    target.copy_from_slice(bytes);
                    true
                }
                None => false,
            }
        }
    }

    #[test]
    fn uploads_land_in_a_slice_of_a_block_linear_volume() {
        // A 4x2 RGBA8 (0xD5) linear source at 0x100, into slice 1 of a
        // 16x8x2 block-linear volume at 0x1000 (one-GOB blocks).
        let mut memory = Bytes(vec![0; 0x4000]);
        for index in 0..8u32 {
            memory.write_gpu(0x100 + u64::from(index) * 4, &(index + 1).to_le_bytes());
        }
        let surface = |address, linear, width, height, layer| Surface {
            address,
            format: 0xD5,
            linear,
            pitch: 16,
            width,
            height,
            gobs_in_y: 1,
            gobs_in_z: 1,
            depth: 2,
            layer,
        };
        let blit = Blit {
            dst: surface(0x1000, false, 16, 8, 1),
            src: surface(0x100, true, 4, 2, 0),
            dst_rect: [0, 0, 4, 2],
            src_origin: [0.0, 0.0],
            src_step: [1.0, 1.0],
            linear_filter: false,
        };
        assert!(copy_in_memory(&mut memory, &blit));
        let volume = Volume::new(64, 8, 1, 1);
        for y in 0..2u64 {
            for x in 0..4u64 {
                let at = 0x1000 + volume.offset(x * 4, y, 1);
                let expected = (y * 4 + x + 1) as u32;
                assert_eq!(memory.read_u32(at), Some(expected), "({x}, {y})");
            }
        }
        // Slice 0 untouched.
        assert_eq!(memory.read_u32(0x1000), Some(0));
    }

    #[test]
    fn copies_read_their_surfaces_and_rectangles() {
        let mut state = TwodState::default();
        for (method, value) in [
            (method::DST, 0xD5),
            (method::DST + 6, 640),
            (method::DST + 7, 360),
            (method::DST + 8, 0x5),
            (method::DST + 9, 0x1000_0000),
            (method::SRC, 0xD5),
            (method::SRC + 6, 1280),
            (method::SRC + 7, 720),
            (method::SRC + 9, 0x2000_0000),
            (method::PIXELS_FROM_MEMORY + 2, 640),
            (method::PIXELS_FROM_MEMORY + 3, 360),
            (method::PIXELS_FROM_MEMORY + 5, 2),
            (method::PIXELS_FROM_MEMORY + 7, 2),
            (method::PIXELS_FROM_MEMORY + 8, 0x8000_0000),
        ] {
            state.write(method, value);
        }
        let blit = state.blit();
        assert_eq!(blit.dst.address, 0x5_1000_0000);
        assert_eq!((blit.src.width, blit.src.height), (1280, 720));
        assert_eq!(blit.dst_rect, [0, 0, 640, 360]);
        assert_eq!(blit.src_step, [2.0, 2.0]);
        assert_eq!(blit.src_origin, [0.5, 0.0]);
    }
}
