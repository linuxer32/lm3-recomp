//! The compute engine (class 0xB1C0): its registers, and the launch
//! descriptor (QMD) a dispatch reads from GPU memory.
//!
//! A launch names a 256-byte queue meta data block (`LAUNCH_DESC_ADDRESS`)
//! holding the grid and block sizes, the program's offset in the code region
//! and the constant buffers bound to it. Field positions follow NVIDIA's
//! published QMD version 1.7 layout (Ryujinx's `ComputeQmd` reads the same).

use super::render::GpuMemory;
use super::state::PoolState;

pub const CLASS_KEPLER_COMPUTE: u32 = 0xB1C0;

/// Register (method) numbers of the compute class.
pub mod method {
    /// The launch descriptor's GPU address, shifted right by 8.
    pub const LAUNCH_DESC_ADDRESS: u32 = 0x0AD;
    pub const LAUNCH: u32 = 0x0AF;
    /// Texture header pool: address high, low, then the highest index.
    pub const TEX_HEADER_POOL: u32 = 0x557;
    /// Sampler pool: address high, low, then the highest index.
    pub const TEX_SAMPLER_POOL: u32 = 0x55D;
    /// The code region programs are offsets into: address high, low.
    pub const CODE_ADDRESS: u32 = 0x582;
    /// The constant buffer slot that holds texture handles.
    pub const BINDLESS_TEXTURE: u32 = 0x982;
    /// Report semaphore: address high, low, payload, then the operation
    /// (which releases it).
    pub const REPORT_SEMAPHORE: u32 = 0x6C0;
    pub const REPORT_SEMAPHORE_OPERATION: u32 = 0x6C3;

    /// The registers whose meaning this module uses.
    pub fn known(method: u32) -> bool {
        matches!(
            method,
            LAUNCH_DESC_ADDRESS
                | LAUNCH
                | 0x557..=0x559
                | 0x55D..=0x55F
                | 0x582..=0x583
                | BINDLESS_TEXTURE
                | 0x6C0..=0x6C3
        )
    }
}

const REGISTERS: usize = 0x1000;
const QMD_WORDS: usize = 64;

/// The compute class's registers, as the title last wrote them.
#[derive(Debug, Clone)]
pub struct ComputeState {
    registers: Vec<u32>,
}

impl Default for ComputeState {
    fn default() -> Self {
        Self {
            registers: vec![0; REGISTERS],
        }
    }
}

/// One dispatch, as a backend needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeDispatch {
    /// GPU address of the program's first instruction (compute programs have
    /// no header).
    pub program_address: u64,
    /// Thread blocks in x, y and z.
    pub grid: [u32; 3],
    /// Threads per block in x, y and z.
    pub block: [u32; 3],
    /// Bytes of shared memory per block.
    pub shared_memory_size: u32,
    /// Constant buffers 0-7: GPU address and size in bytes, when bound.
    pub constant_buffers: [Option<(u64, u32)>; 8],
    pub texture_pool: PoolState,
    pub sampler_pool: PoolState,
    /// The constant buffer slot that holds texture handles.
    pub texture_buffer_index: u32,
}

impl ComputeState {
    pub fn write(&mut self, method: u32, data: u32) {
        if let Some(register) = self.registers.get_mut(method as usize) {
            *register = data;
        }
    }

    fn read(&self, method: u32) -> u32 {
        self.registers.get(method as usize).copied().unwrap_or(0)
    }

    /// A 40-bit address split over a high and a low register.
    fn address(&self, high: u32) -> u64 {
        (u64::from(self.read(high) & 0xFF) << 32) | u64::from(self.read(high + 1))
    }

    fn pool(&self, first: u32) -> PoolState {
        PoolState {
            address: self.address(first),
            limit: self.read(first + 2),
        }
    }

    /// A report semaphore release's address, payload and whether it writes
    /// a 16-byte report (payload and timestamp) rather than the payload.
    pub fn report(&self, operation: u32) -> (u64, u32, bool) {
        (
            self.address(method::REPORT_SEMAPHORE),
            self.read(method::REPORT_SEMAPHORE + 2),
            operation & 3 == 1,
        )
    }

    /// The dispatch a launch starts, read from its descriptor; `None` when
    /// the descriptor is not readable.
    pub fn dispatch(&self, memory: &dyn GpuMemory) -> Option<ComputeDispatch> {
        let descriptor = u64::from(self.read(method::LAUNCH_DESC_ADDRESS)) << 8;
        let bytes = memory.read_gpu(descriptor, QMD_WORDS * 4)?;
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        Some(Self::parse(
            &words,
            self.address(method::CODE_ADDRESS),
            self.pool(method::TEX_HEADER_POOL),
            self.pool(method::TEX_SAMPLER_POOL),
            self.read(method::BINDLESS_TEXTURE),
        ))
    }

    fn parse(
        qmd: &[u32],
        code_address: u64,
        texture_pool: PoolState,
        sampler_pool: PoolState,
        texture_buffer_index: u32,
    ) -> ComputeDispatch {
        // 32-bit words: PROGRAM_OFFSET bit 256 (word 8), CTA_RASTER_* 384,
        // SHARED_MEMORY_SIZE 544 (17), CTA_THREAD_DIMENSION* 592 (18),
        // CONSTANT_BUFFER_VALID 640 (20), CONSTANT_BUFFER_ADDR_* 928 (29).
        let word = |index: usize| qmd.get(index).copied().unwrap_or(0);
        let constant_buffers = std::array::from_fn(|slot| {
            let valid = word(20) & (1 << slot) != 0;
            let (low, high) = (word(29 + slot * 2), word(30 + slot * 2));
            valid.then(|| ((u64::from(high & 0xFF) << 32) | u64::from(low), high >> 15))
        });
        ComputeDispatch {
            program_address: code_address + u64::from(word(8)),
            grid: [word(12) & 0x7FFF_FFFF, word(13) & 0xFFFF, word(13) >> 16],
            block: [word(18) >> 16, word(19) & 0xFFFF, word(19) >> 16],
            shared_memory_size: word(17) & 0x3_FFFF,
            constant_buffers,
            texture_pool,
            sampler_pool,
            texture_buffer_index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_descriptors_unpack_their_fields() {
        let mut qmd = [0u32; QMD_WORDS];
        qmd[8] = 0x1200;
        qmd[12] = 4;
        qmd[13] = 8 | (32 << 16);
        qmd[17] = 0x800;
        qmd[18] = 8 << 16;
        qmd[19] = 4 | (1 << 16);
        qmd[20] = 0b101;
        qmd[29] = 0x4000_0000;
        qmd[30] = 0x5 | (0x1000 << 15);
        qmd[33] = 0x5000_0100;
        qmd[34] = 0x6 | (0x100 << 15);
        let pool = PoolState {
            address: 0,
            limit: 0,
        };
        let dispatch = ComputeState::parse(&qmd, 0x5_0000_0000, pool, pool, 1);
        assert_eq!(dispatch.program_address, 0x5_0000_1200);
        assert_eq!(dispatch.grid, [4, 8, 32]);
        assert_eq!(dispatch.block, [8, 4, 1]);
        assert_eq!(dispatch.shared_memory_size, 0x800);
        assert_eq!(dispatch.constant_buffers[0], Some((0x5_4000_0000, 0x1000)));
        assert_eq!(dispatch.constant_buffers[1], None);
        assert_eq!(dispatch.constant_buffers[2], Some((0x6_5000_0100, 0x100)));
    }
}
