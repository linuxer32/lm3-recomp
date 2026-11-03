//! `nvdrv`: the GPU driver's device files. NVN reaches the GPU through
//! ioctls on `/dev/nvmap` (memory handles), `/dev/nvhost-as-gpu` (the GPU
//! address space), `/dev/nvhost-ctrl-gpu` (GPU properties),
//! `/dev/nvhost-ctrl` (syncpoint waits) and `/dev/nvhost-gpu` (channels).
//! Movies reach the multimedia engines through `/dev/nvhost-nvdec` and
//! `/dev/nvhost-vic`, whose submits `host1x` runs.
//!
//! Handles, mappings and channels are tracked. Submitted command lists go to
//! the GPU worker (`gpu.rs`), which decodes them and performs their semaphore
//! releases; the ioctl returns its fence before that happens, so a later
//! `EventWait` blocks until the worker reaches the syncpoint value, as it does
//! on hardware. Unknown ioctls are logged with their decoded number and
//! answered with their input unchanged.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::gpu;
use crate::host1x;
use crate::ipc::MemoryAccess;
use crate::server::{Call, Service};
use crate::services::{new_event, unimplemented};

/// NvResult values.
const NV_SUCCESS: u32 = 0;
const NV_BAD_PARAMETER: u32 = 4;
const NV_CONFIG_NOT_FOUND: u32 = 0x30006;

/// MapBufferEx / AllocSpace flag: the caller chose the GPU address.
const FIXED_OFFSET: u32 = 1;
/// Start of addresses handed out by the GPU address space.
const GPU_VA_START: u64 = 0x4_0000_0000;
/// Syncpoint each new GPU channel signals.
const FIRST_CHANNEL_SYNCPOINT: u32 = 0x10;

#[derive(Debug, Clone, Copy, Default)]
struct NvmapHandle {
    size: u32,
    align: u32,
    kind: u8,
    address: u64,
}

#[derive(Debug, Default)]
struct DriverState {
    handles: BTreeMap<u32, NvmapHandle>,
    next_handle: u32,
    next_va: u64,
    /// The syncpoint the next opened GPU channel will signal.
    next_syncpoint: u32,
    next_object: u64,
}

fn driver() -> std::sync::MutexGuard<'static, DriverState> {
    static STATE: Mutex<DriverState> = Mutex::new(DriverState {
        handles: BTreeMap::new(),
        next_handle: 1,
        next_va: GPU_VA_START,
        next_syncpoint: FIRST_CHANNEL_SYNCPOINT,
        next_object: 1,
    });
    STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A device file opened through nvdrv.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Device {
    Nvmap,
    AddressSpace,
    CtrlGpu,
    Ctrl,
    /// A GPU channel and the syncpoint it signals.
    Channel {
        syncpoint: u32,
    },
    /// A multimedia engine channel (NVDEC, VIC, NVJPG) and its syncpoint.
    Engine {
        syncpoint: u32,
    },
    Display,
    Other,
}

fn open_device(path: &str) -> Option<Device> {
    Some(match path {
        "/dev/nvmap" => Device::Nvmap,
        "/dev/nvhost-as-gpu" => Device::AddressSpace,
        "/dev/nvhost-ctrl-gpu" => Device::CtrlGpu,
        "/dev/nvhost-ctrl" => Device::Ctrl,
        "/dev/nvhost-gpu" | "/dev/nvhost-nvdec" | "/dev/nvhost-vic" | "/dev/nvhost-nvjpg" => {
            let mut state = driver();
            let syncpoint = state.next_syncpoint;
            state.next_syncpoint += 1;
            if path == "/dev/nvhost-gpu" {
                Device::Channel { syncpoint }
            } else {
                Device::Engine { syncpoint }
            }
        }
        "/dev/nvdisp-disp0" | "/dev/nvdisp-disp1" => Device::Display,
        "/dev/nvhost-dbg-gpu" | "/dev/nvhost-prof-gpu" => Device::Other,
        _ => return None,
    })
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    bytes
        .get(at..at + 4)
        .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    bytes
        .get(at..at + 8)
        .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
}

/// A count field as a usize.
fn count_at(bytes: &[u8], at: usize) -> usize {
    u32_at(bytes, at) as usize
}

fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    if let Some(target) = bytes.get_mut(at..at + 4) {
        target.copy_from_slice(&value.to_le_bytes());
    }
}

fn put_u64(bytes: &mut [u8], at: usize, value: u64) {
    if let Some(target) = bytes.get_mut(at..at + 8) {
        target.copy_from_slice(&value.to_le_bytes());
    }
}

/// GM20B (Tegra X1 GPU) properties, as nvhost-ctrl-gpu reports them.
fn gpu_characteristics() -> Vec<u8> {
    let mut c = vec![0u8; 0xA0];
    let words: [(usize, u32); 30] = [
        (0x00, 0x120),       // arch: GM200
        (0x04, 0xB),         // impl: GM20B
        (0x08, 0xA1),        // rev
        (0x0C, 1),           // num_gpc
        (0x10, 0x40000),     // l2_cache_size (u64)
        (0x20, 2),           // num_tpc_per_gpc
        (0x24, 0x20),        // bus_type: AXI
        (0x28, 0x20000),     // big_page_size
        (0x2C, 0x20000),     // compression_page_size
        (0x30, 27),          // pde_coverage_bit_count
        (0x34, 0x30000),     // available_big_page_sizes
        (0x38, 1),           // gpc_mask
        (0x3C, 0x503),       // sm_arch_sm_version
        (0x40, 0x503),       // sm_arch_spa_version
        (0x44, 0x80),        // sm_arch_warp_count
        (0x48, 0x28),        // gpu_va_bit_count
        (0x50, 0x55),        // flags (u64)
        (0x58, 0x902D),      // twod_class
        (0x5C, 0xB197),      // threed_class: MAXWELL_B
        (0x60, 0xB1C0),      // compute_class
        (0x64, 0xB06F),      // gpfifo_class
        (0x68, 0xA140),      // inline_to_memory_class
        (0x6C, 0xB0B5),      // dma_copy_class
        (0x70, 1),           // max_fbps_count
        (0x78, 2),           // max_ltc_per_fbp
        (0x7C, 1),           // max_lts_per_ltc
        (0x84, 1),           // max_gpc_count
        (0x88, 0x21D70),     // rop_l2_en_mask_0
        (0x90, 0x3032_6D67), // chipname "gm20b" (u64, little-endian)
        (0x94, 0x62),
    ];
    for (at, value) in words {
        put_u32(&mut c, at, value);
    }
    c
}

/// Every ioctl is logged with `RECOMP_TRACE_NVDRV=1`.
fn tracing() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("RECOMP_TRACE_NVDRV").is_some())
}

/// ctrl: EventWait / EventWaitAsync(syncpoint, threshold, timeout). Blocks until
/// the GPU worker has reached `threshold`; a negative timeout waits forever, a
/// zero timeout only polls. The value read at the end is reported back.
///
/// A wait that times out is still reported as NV_SUCCESS: the caller sees the
/// value it can use to decide whether to wait again, as on hardware.
fn event_wait(data: &mut [u8]) -> u32 {
    let syncpoint = u32_at(data, 0);
    let threshold = u32_at(data, 4);
    let timeout = data
        .get(8..16)
        .map_or(0, |raw| i64::from_le_bytes(raw.try_into().unwrap()));
    let value = gpu::wait_for_syncpoint(syncpoint, threshold, timeout);
    put_u32(data, 12, value);
    NV_SUCCESS
}

/// The memory an nvmap handle (or id: they are the same number here) was
/// allocated at.
pub fn nvmap_address(handle: u32) -> Option<u64> {
    driver().handles.get(&handle).map(|entry| entry.address)
}

/// Handle one ioctl on `device`: `data` holds the input struct and becomes
/// the output; `extra` is the inline buffer of Ioctl2/Ioctl3.
fn ioctl(mem: &dyn MemoryAccess, device: Device, cmd: u32, data: &mut [u8], extra: &[u8]) -> u32 {
    let (group, number) = ((cmd >> 8) & 0xFF, cmd & 0xFF);
    // A syncpoint wait blocks for the GPU worker, so it takes no driver lock:
    // another thread must be able to map buffers and submit while it waits.
    if device == Device::Ctrl && group == 0x00 && matches!(number, 0x1D | 0x1E) {
        return event_wait(data);
    }
    let mut state = driver();
    match (device, group, number) {
        // nvmap: Create(size) -> handle
        (Device::Nvmap, 0x01, 0x01) => {
            let handle = state.next_handle;
            state.next_handle += 1;
            let size = u32_at(data, 0);
            state.handles.insert(
                handle,
                NvmapHandle {
                    size,
                    ..NvmapHandle::default()
                },
            );
            put_u32(data, 4, handle);
        }
        // FromId(id) -> handle: ids are handles.
        (Device::Nvmap, 0x01, 0x03) => {
            let id = u32_at(data, 0);
            if !state.handles.contains_key(&id) {
                return NV_BAD_PARAMETER;
            }
            put_u32(data, 4, id);
        }
        // Alloc(handle, heap mask, flags, align, kind, address)
        (Device::Nvmap, 0x01, 0x04) => {
            let handle = u32_at(data, 0);
            let Some(entry) = state.handles.get_mut(&handle) else {
                return NV_BAD_PARAMETER;
            };
            entry.align = u32_at(data, 12);
            entry.kind = data.get(16).copied().unwrap_or(0);
            entry.address = u64_at(data, 24);
        }
        // Free(handle) -> (address, size, flags)
        (Device::Nvmap, 0x01, 0x05) => {
            let handle = u32_at(data, 0);
            let Some(entry) = state.handles.remove(&handle) else {
                return NV_BAD_PARAMETER;
            };
            put_u64(data, 8, entry.address);
            put_u32(data, 16, entry.size);
            put_u32(data, 20, 0);
        }
        // Param(handle, param) -> value
        (Device::Nvmap, 0x01, 0x09) => {
            let Some(entry) = state.handles.get(&u32_at(data, 0)).copied() else {
                return NV_BAD_PARAMETER;
            };
            let value = match u32_at(data, 4) {
                1 => entry.size,
                2 => entry.align,
                3 => entry.address as u32,
                4 => 0x4000_0000,
                5 => u32::from(entry.kind),
                _ => 0,
            };
            put_u32(data, 8, value);
        }
        // GetId(handle) -> id
        (Device::Nvmap, 0x01, 0x0E) => {
            let handle = u32_at(data, 4);
            put_u32(data, 0, handle);
        }
        // as-gpu: BindChannel(fd), FreeSpace, UnmapBuffer
        (Device::AddressSpace, 0x41, 0x01) => {}
        (Device::AddressSpace, 0x41, 0x03) => {}
        (Device::AddressSpace, 0x41, 0x05) => {
            gpu::mappings().remove(&u64_at(data, 0));
        }
        // AllocSpace(pages, page size, flags, offset) -> offset
        (Device::AddressSpace, 0x41, 0x02) => {
            let size = u64::from(u32_at(data, 0)) * u64::from(u32_at(data, 4));
            if u32_at(data, 8) & FIXED_OFFSET == 0 {
                let align = u64::from(u32_at(data, 4)).max(0x1000);
                let offset = state.next_va.next_multiple_of(align);
                state.next_va = offset + size;
                put_u64(data, 16, offset);
            }
        }
        // MapBufferEx(flags, kind, handle, page size, buffer offset,
        // mapping size, offset) -> offset
        (Device::AddressSpace, 0x41, 0x06) => {
            let flags = u32_at(data, 0);
            let Some(entry) = state.handles.get(&u32_at(data, 8)).copied() else {
                return NV_BAD_PARAMETER;
            };
            let buffer_offset = u64_at(data, 16);
            let size = match u64_at(data, 24) {
                0 => u64::from(entry.size),
                size => size,
            };
            let offset = if flags & FIXED_OFFSET != 0 {
                u64_at(data, 32)
            } else {
                let offset = state.next_va.next_multiple_of(0x20000);
                state.next_va = offset + size;
                offset
            };
            gpu::mappings().insert(offset, (entry.address + buffer_offset, size));
            put_u64(data, 32, offset);
        }
        // GetVARegions -> small and big page regions
        (Device::AddressSpace, 0x41, 0x08) => {
            put_u32(data, 8, 0x30);
            put_u64(data, 16, 0x0400_0000);
            put_u32(data, 24, 0x1000);
            put_u64(data, 32, 0x3F_BFFF);
            put_u64(data, 40, GPU_VA_START);
            put_u32(data, 48, 0x20000);
            put_u64(data, 56, 0x1_BFFF);
        }
        // AllocAsEx / InitializeEx, Remap
        (Device::AddressSpace, 0x41, 0x09) | (Device::AddressSpace, 0x41, 0x14) => {}
        // ctrl-gpu: ZCullGetCtxSize
        (Device::CtrlGpu, 0x47, 0x01) => put_u32(data, 0, 0x62000),
        // ZCullGetInfo
        (Device::CtrlGpu, 0x47, 0x02) => {
            for (index, value) in [0x20, 0x20, 0x400, 0x800, 0x20, 0x20, 0xC0, 0x20, 0x40, 0x10]
                .into_iter()
                .enumerate()
            {
                put_u32(data, index * 4, value);
            }
        }
        // ZBCSetTable, FlushL2
        (Device::CtrlGpu, 0x47, 0x03) | (Device::CtrlGpu, 0x47, 0x07) => {}
        // ZBCQueryTable
        (Device::CtrlGpu, 0x47, 0x04) => {}
        // GetCharacteristics -> (size, address, characteristics)
        (Device::CtrlGpu, 0x47, 0x05) => {
            put_u64(data, 0, 0xA0);
            let characteristics = gpu_characteristics();
            if let Some(target) = data.get_mut(16..16 + 0xA0) {
                target.copy_from_slice(&characteristics);
            }
        }
        // GetTPCMasks -> both TPCs
        (Device::CtrlGpu, 0x47, 0x06) => {
            if u32_at(data, 0) != 0 {
                put_u32(data, 16, 0x3);
            }
        }
        // GetActiveSlotMask
        (Device::CtrlGpu, 0x47, 0x14) => {
            put_u32(data, 0, 0x7);
            put_u32(data, 4, 0x1);
        }
        // GetGpuTime (ns)
        (Device::CtrlGpu, 0x47, 0x1C) => {
            put_u64(data, 0, crate::clock::now_ns());
        }
        // ctrl: GetConfig: no overrides set.
        (Device::Ctrl, 0x00, 0x1B) => return NV_CONFIG_NOT_FOUND,
        // ClearEventWait, EventRegister, EventUnregister(Batch)
        (Device::Ctrl, 0x00, 0x1C | 0x1F | 0x20 | 0x21) => {}
        // channel: SetNvmapFd, ChannelSetTimeout, ZCullBind,
        // SetErrorNotifier, SetChannelPriority, ChannelSetTimeslice
        (Device::Channel { .. }, 0x48, 0x01 | 0x03 | 0x0B | 0x0C | 0x0D | 0x1D) => {}
        // SetClientData, GetClientData
        (Device::Channel { .. }, 0x47, 0x14 | 0x15) => {}
        // AllocGPFIFOEx2(entries, flags, ...) -> fence: the value the next
        // submission on this channel's syncpoint will reach.
        (Device::Channel { syncpoint }, 0x48, 0x1A) => {
            put_u32(data, 24, syncpoint);
            put_u32(data, 28, gpu::syncpoint_pending(syncpoint));
        }
        // AllocObjCtx(class, flags) -> object id
        (Device::Channel { .. }, 0x48, 0x09) => {
            let object = state.next_object;
            state.next_object += 1;
            put_u64(data, 8, object);
        }
        // SubmitGPFIFO(address, entries, flags, fence) -> fence: the lists go
        // to the GPU worker, so the fence is only reached once they are decoded.
        (Device::Channel { syncpoint }, 0x48, 0x08 | 0x1B) => {
            // The GPFIFO entries follow the struct, or (Ioctl2) come in the
            // second input buffer.
            let count = u32_at(data, 8) as usize;
            let source = if extra.is_empty() {
                data.get(0x18..).unwrap_or(&[])
            } else {
                extra
            };
            let entries: Vec<u64> = source
                .as_chunks::<8>()
                .0
                .iter()
                .take(count)
                .map(|entry| u64::from_le_bytes(*entry))
                .collect();
            // The fence the submission completes is the one the caller passed,
            // or the channel's own syncpoint when it passed none.
            let fence = match u32_at(data, 16) {
                0 => syncpoint,
                id => id,
            };
            let value = gpu::allocate_fence(fence);
            gpu::submit(syncpoint, mem, entries, (fence, value));
            put_u32(data, 16, fence);
            put_u32(data, 20, value);
        }
        // Engine channels: SetNvmapFd, SetTimeout, SetSubmitTimeout.
        (Device::Engine { .. }, 0x48, 0x01 | 0x03) | (Device::Engine { .. }, 0x00, 0x07) => {}
        // GetSyncpoint(module) -> the channel's syncpoint.
        (Device::Engine { syncpoint }, 0x00, 0x02) => put_u32(data, 4, syncpoint),
        // GetWaitBase(module) -> 0.
        (Device::Engine { .. }, 0x00, 0x03) => put_u32(data, 4, 0),
        // MapCommandBuffer(count, ..., [handle, engine address]): give each
        // handle's memory an engine address. UnmapCommandBuffer keeps them.
        (Device::Engine { .. }, 0x00, 0x09) => {
            for entry in 0..u32_at(data, 0) as usize {
                let at = 12 + entry * 8;
                let handle = u32_at(data, at);
                let Some(memory) = state.handles.get(&handle).copied() else {
                    return NV_BAD_PARAMETER;
                };
                put_u32(
                    data,
                    at + 4,
                    host1x::map(handle, memory.address, memory.size),
                );
            }
        }
        (Device::Engine { .. }, 0x00, 0x0A) => {}
        // Submit(command buffers, relocations, syncpoint increments, fences)
        // -> the fence thresholds: the buffers run now.
        (Device::Engine { .. }, 0x00, 0x01) => {
            let count = |index: usize| u32_at(data, index * 4) as usize;
            let (buffers, relocations, increments, fences) =
                (count(0), count(1), count(2), count(3));
            let increments_at = 16 + buffers * 12 + relocations * 16 + relocations * 4;
            let fences_at = increments_at + increments * 20;
            // The values each increment's syncpoint reaches when done.
            let mut thresholds = Vec::new();
            for index in 0..increments {
                let at = increments_at + index * 20;
                let (syncpoint, times) = (u32_at(data, at), u32_at(data, at + 4));
                let mut threshold = gpu::syncpoint_pending(syncpoint);
                for _ in 0..times {
                    threshold = gpu::allocate_fence(syncpoint);
                }
                thresholds.push((syncpoint, threshold));
                if index < fences {
                    put_u32(data, fences_at + index * 4, threshold);
                }
            }
            let words: Vec<(u64, usize)> = (0..buffers)
                .filter_map(|index| {
                    let at = 16 + index * 12;
                    let memory = state.handles.get(&u32_at(data, at))?;
                    Some((
                        memory.address + u64::from(u32_at(data, at + 4)),
                        count_at(data, at + 8),
                    ))
                })
                .collect();
            // The engines take the driver lock's place: another thread may
            // map buffers while a frame decodes.
            drop(state);
            for (address, len) in words {
                if let Some(bytes) = mem.read_bytes(address, len * 4) {
                    let words: Vec<u32> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|word| u32::from_le_bytes(*word))
                        .collect();
                    host1x::submit(mem, &words);
                }
            }
            // Buffers that signal less than they promised still complete.
            for (syncpoint, threshold) in thresholds {
                while gpu::syncpoint_value(syncpoint) < threshold {
                    gpu::signal_syncpoint(syncpoint);
                }
            }
            return NV_SUCCESS;
        }
        _ => {
            println!(
                "[recomp-ipc] UNIMPLEMENTED nvdrv ioctl {device:?} group {group:#x} nr {number:#x} size {:#x} dir {} (echoed)",
                (cmd >> 16) & 0x3FFF,
                cmd >> 30
            );
        }
    }
    NV_SUCCESS
}

/// `nvdrv`, `nvdrv:a`, `nvdrv:s`, `nvdrv:t` (INvDrvServices)
pub struct NvDrvService {
    files: BTreeMap<u32, Device>,
    next_fd: u32,
}

impl NvDrvService {
    pub fn new() -> Self {
        NvDrvService {
            files: BTreeMap::new(),
            next_fd: 1,
        }
    }

    fn ioctl_call(
        &mut self,
        call: &mut Call<'_>,
        extra_in: bool,
        extra_out: bool,
    ) -> Result<(), u32> {
        let (fd, cmd) = (call.arg_u32(0), call.arg_u32(4));
        let Some(device) = self.files.get(&fd).copied() else {
            call.push_u32(NV_BAD_PARAMETER);
            return Ok(());
        };
        let size = ((cmd >> 16) & 0x3FFF) as usize;
        let mut data = call.in_buffer(0);
        data.resize(size.max(data.len()), 0);
        let extra = if extra_in {
            call.in_buffer(1)
        } else {
            Vec::new()
        };
        let result = ioctl(call.mem, device, cmd, &mut data, &extra);
        if tracing() {
            println!(
                "[recomp-ipc] nvdrv ioctl fd {fd} {device:?} group {:#x} nr {:#x} -> {result:#x}",
                (cmd >> 8) & 0xFF,
                cmd & 0xFF
            );
        }
        // Output-direction ioctls copy the struct back.
        if cmd >> 31 != 0 {
            call.write_out(0, &data);
        }
        // Ioctl3 returns the variable part in a second buffer.
        if extra_out {
            let extra = match (device, (cmd >> 8) & 0xFF, cmd & 0xFF) {
                (Device::CtrlGpu, 0x47, 0x05) => gpu_characteristics(),
                (Device::CtrlGpu, 0x47, 0x06) => 3u32.to_le_bytes().to_vec(),
                (Device::AddressSpace, 0x41, 0x08) => data.get(16..64).unwrap_or(&[]).to_vec(),
                _ => Vec::new(),
            };
            call.write_out(1, &extra);
        }
        call.push_u32(result);
        Ok(())
    }
}

impl Default for NvDrvService {
    fn default() -> Self {
        Self::new()
    }
}

impl Service for NvDrvService {
    fn name(&self) -> String {
        "nvdrv".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // Open(path) -> (fd, error)
            0 => {
                let raw = call.in_buffer(0);
                let end = raw.iter().position(|byte| *byte == 0).unwrap_or(raw.len());
                let path = String::from_utf8_lossy(&raw[..end]).into_owned();
                match open_device(&path) {
                    Some(device) => {
                        let fd = self.next_fd;
                        self.next_fd += 1;
                        self.files.insert(fd, device);
                        println!("[recomp-ipc] nvdrv open {path} -> fd {fd}");
                        call.push_u32(fd);
                        call.push_u32(NV_SUCCESS);
                    }
                    None => {
                        println!("[recomp-ipc] UNIMPLEMENTED nvdrv device {path}");
                        call.push_u32(0);
                        call.push_u32(NV_BAD_PARAMETER);
                    }
                }
            }
            1 => return self.ioctl_call(call, false, false),
            // Close(fd)
            2 => {
                self.files.remove(&call.arg_u32(0));
                call.push_u32(NV_SUCCESS);
            }
            // Initialize(transfer memory size, process, transfer memory),
            // SetAruid, SetGraphicsFirmwareMemoryMarginEnabled
            3 | 8 | 13 => call.push_u32(NV_SUCCESS),
            // QueryEvent(fd, event id) -> event
            4 => {
                call.push_copy_handle(new_event());
                call.push_u32(NV_SUCCESS);
            }
            11 => return self.ioctl_call(call, true, false),
            12 => return self.ioctl_call(call, false, true),
            _ => return unimplemented(call, "nvdrv"),
        }
        Ok(())
    }
}
