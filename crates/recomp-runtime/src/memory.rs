use recomp_services::ipc::{memory_state, MemoryBlock};
use std::cell::Cell;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MemoryStatus {
    Ok = 0,
    Unaligned = 1,
    OutOfBounds = 2,
    PermissionDenied = 3,
    Unmapped = 4,
    Uninitialized = 5,
    InvalidOutPtr = 6,
    SizeMismatch = 7,
    Internal = 8,
}

impl MemoryStatus {
    pub fn is_ok(self) -> bool {
        matches!(self, MemoryStatus::Ok)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPermissions {
    read: bool,
    write: bool,
    execute: bool,
}

impl MemoryPermissions {
    pub const fn new(read: bool, write: bool, execute: bool) -> Self {
        Self {
            read,
            write,
            execute,
        }
    }

    pub const fn read_only() -> Self {
        Self::new(true, false, false)
    }

    pub const fn read_write() -> Self {
        Self::new(true, true, false)
    }

    pub const fn read_execute() -> Self {
        Self::new(true, false, true)
    }

    pub fn allows_read(self) -> bool {
        self.read
    }

    pub fn allows_write(self) -> bool {
        self.write
    }

    pub fn allows_execute(self) -> bool {
        self.execute
    }
}

#[derive(Debug, Clone)]
pub struct MemoryRegionSpec {
    pub name: String,
    pub base: u64,
    pub size: u64,
    pub permissions: MemoryPermissions,
}

impl MemoryRegionSpec {
    pub fn new(
        name: impl Into<String>,
        base: u64,
        size: u64,
        permissions: MemoryPermissions,
    ) -> Self {
        Self {
            name: name.into(),
            base,
            size,
            permissions,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryLayout {
    pub regions: Vec<MemoryRegionSpec>,
}

impl MemoryLayout {
    pub fn new(regions: Vec<MemoryRegionSpec>) -> Self {
        Self { regions }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryLayoutError {
    #[error("memory region {name} has zero size")]
    ZeroSizedRegion { name: String },
    #[error("memory region {name} overflows address space")]
    RegionOverflow { name: String },
    #[error("memory regions {left} and {right} overlap")]
    RegionOverlap { left: String, right: String },
}

#[derive(Debug, Clone, Copy)]
enum AccessKind {
    Read,
    Write,
    #[allow(dead_code)]
    Execute,
    Init,
}

#[derive(Debug)]
struct MemoryRegion {
    spec: MemoryRegionSpec,
    data: Pages,
}

/// A region's bytes: its pages in the guest address space reservation
/// (`fastmem`), which generated code accesses directly, or a heap buffer
/// when there is no reservation (or the region cannot be placed in it).
#[derive(Debug)]
enum Pages {
    Mapped { data: *mut u8, len: usize },
    Heap(Vec<u8>),
}

// SAFETY: the pages are guest memory, shared by every guest thread like a
// machine's RAM; the runtime's access paths are what synchronize.
unsafe impl Send for Pages {}
unsafe impl Sync for Pages {}

impl Pages {
    /// `len` zeroed bytes for the region at guest address `base`.
    fn new(base: u64, len: usize) -> Self {
        match crate::fastmem::map_pages(base, len) {
            Some(data) => Pages::Mapped { data, len },
            None => Pages::Heap(vec![0; len]),
        }
    }

    /// Give the pages back (the region was unmapped).
    fn release(self) {
        match self {
            Pages::Mapped { data, len } => crate::fastmem::release_pages(data, len),
            Pages::Heap(buffer) => retire(std::iter::once(buffer)),
        }
    }
}

impl std::ops::Deref for Pages {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            // SAFETY: mapped for `len` bytes while the region exists.
            Pages::Mapped { data, len } => unsafe { std::slice::from_raw_parts(*data, *len) },
            Pages::Heap(buffer) => buffer,
        }
    }
}

impl std::ops::DerefMut for Pages {
    fn deref_mut(&mut self) -> &mut [u8] {
        match self {
            // SAFETY: as above.
            Pages::Mapped { data, len } => unsafe { std::slice::from_raw_parts_mut(*data, *len) },
            Pages::Heap(buffer) => buffer,
        }
    }
}

#[derive(Debug)]
struct RuntimeMemory {
    regions: Vec<MemoryRegion>,
}

impl RuntimeMemory {
    fn new(layout: &MemoryLayout) -> Result<Self, MemoryLayoutError> {
        // Tell fastmem how far the layout reaches before the first region is
        // mapped: a host whose user address space cannot hold all 512 GiB
        // (Android's 39-bit spaces) then reserves just this window instead
        // of leaving every guest access on the checked path.
        crate::fastmem::reserve_for(
            layout
                .regions
                .iter()
                .map(|spec| spec.base.saturating_add(spec.size))
                .max()
                .unwrap_or(0),
        );
        let mut regions = Vec::with_capacity(layout.regions.len());
        for spec in &layout.regions {
            if spec.size == 0 {
                return Err(MemoryLayoutError::ZeroSizedRegion {
                    name: spec.name.clone(),
                });
            }
            let end = spec.base.checked_add(spec.size).ok_or_else(|| {
                MemoryLayoutError::RegionOverflow {
                    name: spec.name.clone(),
                }
            })?;
            regions.push((spec.clone(), end));
        }

        regions.sort_by_key(|region| region.0.base);
        for pair in regions.windows(2) {
            let left = &pair[0].0;
            let left_end = pair[0].1;
            let right = &pair[1].0;
            if left_end > right.base {
                return Err(MemoryLayoutError::RegionOverlap {
                    left: left.name.clone(),
                    right: right.name.clone(),
                });
            }
        }

        let mapped = regions
            .into_iter()
            .map(|(spec, _)| MemoryRegion {
                data: Pages::new(spec.base, spec.size as usize),
                spec,
            })
            .collect();

        Ok(Self { regions: mapped })
    }

    /// The block containing `address`: a mapped region, or the free gap
    /// around it. Like a 39-bit Horizon process, free space ends at 2^39
    /// and everything above is one inaccessible block, whose end wraps to
    /// zero; that is where guest code walking memory (rtld) stops.
    fn query_block(&self, address: u64) -> MemoryBlock {
        let mut free_base = 0;
        for region in &self.regions {
            let end = region.spec.base + region.spec.size;
            if address < region.spec.base {
                return MemoryBlock {
                    base: free_base,
                    size: region.spec.base - free_base,
                    state: memory_state::FREE,
                    perm: 0,
                };
            }
            if address < end {
                let permissions = &region.spec.permissions;
                return MemoryBlock {
                    base: region.spec.base,
                    size: region.spec.size,
                    state: region_state(&region.spec),
                    perm: u32::from(permissions.read)
                        | u32::from(permissions.write) << 1
                        | u32::from(permissions.execute) << 2,
                };
            }
            free_base = end;
        }
        if address < ADDRESS_SPACE_END && free_base < ADDRESS_SPACE_END {
            return MemoryBlock {
                base: free_base,
                size: ADDRESS_SPACE_END - free_base,
                state: memory_state::FREE,
                perm: 0,
            };
        }
        let base = free_base.max(ADDRESS_SPACE_END);
        MemoryBlock {
            base,
            size: 0u64.wrapping_sub(base),
            state: memory_state::INACCESSIBLE,
            perm: 0,
        }
    }

    /// Map a region after boot, filled with `contents` (zeros beyond).
    fn map(&mut self, spec: MemoryRegionSpec, contents: &[u8]) -> Result<(), MemoryStatus> {
        let end = spec
            .base
            .checked_add(spec.size)
            .ok_or(MemoryStatus::OutOfBounds)?;
        let overlaps = self.regions.iter().any(|region| {
            region.spec.base < end && spec.base < region.spec.base + region.spec.size
        });
        if spec.size == 0 || overlaps {
            return Err(MemoryStatus::OutOfBounds);
        }
        let mut data = Pages::new(spec.base, spec.size as usize);
        let len = contents.len().min(data.len());
        data[..len].copy_from_slice(&contents[..len]);
        let index = self
            .regions
            .partition_point(|region| region.spec.base < spec.base);
        self.regions.insert(index, MemoryRegion { spec, data });
        Ok(())
    }

    /// Remove the region based at `base`, returning its bytes.
    fn unmap(&mut self, base: u64) -> Option<Pages> {
        let index = self
            .regions
            .iter()
            .position(|region| region.spec.base == base)?;
        Some(self.regions.remove(index).data)
    }

    /// Copy `len` bytes out of one region.
    fn read_block(&self, address: u64, len: usize) -> Result<Vec<u8>, MemoryStatus> {
        let region = self.resolve_region(address, len.max(1), AccessKind::Read, false)?;
        let offset = (address - region.spec.base) as usize;
        Ok(region.data[offset..offset + len].to_vec())
    }

    /// Copy `bytes` into one region.
    fn write_block(&mut self, address: u64, bytes: &[u8]) -> Result<(), MemoryStatus> {
        let region =
            self.resolve_region_mut(address, bytes.len().max(1), AccessKind::Write, false)?;
        let offset = (address - region.spec.base) as usize;
        region.data[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn load(&self, address: u64, size: usize) -> Result<u64, MemoryStatus> {
        // Horizon runs EL0 with alignment checking off: unaligned accesses
        // to normal memory are architecturally fine.
        let region = self.resolve_region(address, size, AccessKind::Read, false)?;
        let offset = (address - region.spec.base) as usize;
        let mut value = 0u64;
        for i in 0..size {
            value |= (region.data[offset + i] as u64) << (i * 8);
        }
        Ok(value)
    }

    fn store(&mut self, address: u64, size: usize, value: u64) -> Result<(), MemoryStatus> {
        let region = self.resolve_region_mut(address, size, AccessKind::Write, false)?;
        let offset = (address - region.spec.base) as usize;
        for i in 0..size {
            region.data[offset + i] = ((value >> (i * 8)) & 0xFF) as u8;
        }
        Ok(())
    }

    fn write_bytes_init(
        &mut self,
        address: u64,
        size: u64,
        bytes: &[u8],
    ) -> Result<(), MemoryStatus> {
        if bytes.len() as u64 != size {
            return Err(MemoryStatus::SizeMismatch);
        }
        let region = self.resolve_region_mut(address, size as usize, AccessKind::Init, false)?;
        let offset = (address - region.spec.base) as usize;
        let end = offset + bytes.len();
        region.data[offset..end].copy_from_slice(bytes);
        Ok(())
    }

    fn zero_fill_init(&mut self, address: u64, size: u64) -> Result<(), MemoryStatus> {
        let region = self.resolve_region_mut(address, size as usize, AccessKind::Init, false)?;
        let offset = (address - region.spec.base) as usize;
        let end = offset + size as usize;
        for value in &mut region.data[offset..end] {
            *value = 0;
        }
        Ok(())
    }

    fn resolve_region(
        &self,
        address: u64,
        size: usize,
        access: AccessKind,
        require_alignment: bool,
    ) -> Result<&MemoryRegion, MemoryStatus> {
        let index = self.resolve_region_inner(address, size, access, require_alignment)?;
        Ok(&self.regions[index])
    }

    fn resolve_region_mut(
        &mut self,
        address: u64,
        size: usize,
        access: AccessKind,
        require_alignment: bool,
    ) -> Result<&mut MemoryRegion, MemoryStatus> {
        let index = self.resolve_region_inner(address, size, access, require_alignment)?;
        Ok(&mut self.regions[index])
    }

    fn resolve_region_inner(
        &self,
        address: u64,
        size: usize,
        access: AccessKind,
        require_alignment: bool,
    ) -> Result<usize, MemoryStatus> {
        if size == 0 {
            return Err(MemoryStatus::Internal);
        }
        let size_u64 = size as u64;
        if require_alignment && !address.is_multiple_of(size_u64) {
            return Err(MemoryStatus::Unaligned);
        }
        let end = address
            .checked_add(size_u64)
            .ok_or(MemoryStatus::OutOfBounds)?;

        for (index, region) in self.regions.iter().enumerate() {
            let region_end = region
                .spec
                .base
                .checked_add(region.spec.size)
                .ok_or(MemoryStatus::OutOfBounds)?;
            if address < region.spec.base || address >= region_end {
                continue;
            }
            if end > region_end {
                return Err(MemoryStatus::OutOfBounds);
            }
            if !self.check_permissions(region, access) {
                return Err(MemoryStatus::PermissionDenied);
            }
            return Ok(index);
        }

        Err(MemoryStatus::Unmapped)
    }

    fn check_permissions(&self, region: &MemoryRegion, access: AccessKind) -> bool {
        match access {
            AccessKind::Read => region.spec.permissions.allows_read(),
            AccessKind::Write => region.spec.permissions.allows_write(),
            AccessKind::Execute => region.spec.permissions.allows_execute(),
            AccessKind::Init => true,
        }
    }
}

static MEMORY: OnceLock<Mutex<RuntimeMemory>> = OnceLock::new();

/// End of a 39-bit Horizon address space.
const ADDRESS_SPACE_END: u64 = 1 << 39;

pub fn init_memory(layout: MemoryLayout) -> Result<(), MemoryLayoutError> {
    let memory = RuntimeMemory::new(&layout)?;
    if let Some(cell) = MEMORY.get() {
        // Re-initialization replaces the memory contents in place; the runtime
        // normally initializes exactly once, but tests and multi-module loads
        // re-init with different layouts.
        let mut guard = cell.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let old = std::mem::replace(&mut *guard, memory);
        publish(&guard);
        // Pages only the old layout used become inaccessible; heap buffers
        // are retired.
        for region in old.regions {
            region.data.release();
        }
    } else {
        publish(&memory);
        let _ = MEMORY.set(Mutex::new(memory));
    }
    crate::fastmem::publish_base();
    Ok(())
}

/// A mapped region as loads and stores see it. Region buffers never move
/// or shrink while mapped, and unmapped ones are retired rather than freed,
/// so `data` stays valid for as long as anyone can hold this.
#[derive(Debug, Clone, Copy)]
struct FastRegion {
    base: u64,
    end: u64,
    data: *mut u8,
    read: bool,
    write: bool,
}

const NO_REGION: FastRegion = FastRegion {
    base: 0,
    end: 0,
    data: std::ptr::null_mut(),
    read: false,
    write: false,
};

/// The current mappings, sorted by address. Accesses read it without
/// locking; map, unmap and init publish a new table and bump the
/// generation, which invalidates every thread's cached region.
static FAST_TABLE: AtomicPtr<Vec<FastRegion>> = AtomicPtr::new(std::ptr::null_mut());
static GENERATION: AtomicU64 = AtomicU64::new(1);
static RETIRED: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

thread_local! {
    /// The region the last access hit, with the generation it is valid for.
    static LAST_REGION: Cell<(u64, FastRegion)> = const { Cell::new((0, NO_REGION)) };
}

fn retire(buffers: impl Iterator<Item = Vec<u8>>) {
    RETIRED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .extend(buffers);
}

/// Publish `memory`'s mappings to the lock-free access path. Old tables
/// are leaked: a thread may still be reading one, and maps are rare.
fn publish(memory: &RuntimeMemory) {
    let table: Vec<FastRegion> = memory
        .regions
        .iter()
        .map(|region| FastRegion {
            base: region.spec.base,
            end: region.spec.base + region.spec.size,
            data: region.data.as_ptr().cast_mut(),
            read: region.spec.permissions.read,
            write: region.spec.permissions.write,
        })
        .collect();
    FAST_TABLE.store(Box::into_raw(Box::new(table)), Ordering::Release);
    GENERATION.fetch_add(1, Ordering::Release);
}

/// The region holding all of `[address, address + size)`, if any.
#[inline]
fn fast_region(address: u64, size: u64) -> Option<FastRegion> {
    let end = address.checked_add(size)?;
    let generation = GENERATION.load(Ordering::Acquire);
    LAST_REGION.with(|last| {
        let (valid_for, region) = last.get();
        if valid_for == generation && address >= region.base && end <= region.end {
            return Some(region);
        }
        let table = FAST_TABLE.load(Ordering::Acquire);
        if table.is_null() {
            return None;
        }
        // SAFETY: published tables are never freed.
        let regions = unsafe { &*table };
        let index = regions.partition_point(|region| region.end <= address);
        let region = *regions.get(index)?;
        if address >= region.base && end <= region.end {
            last.set((generation, region));
            Some(region)
        } else {
            None
        }
    })
}

#[derive(Debug, Clone)]
pub struct MemoryInitSegment {
    pub name: String,
    pub base: u64,
    pub size: u64,
    pub bytes: Vec<u8>,
}

impl MemoryInitSegment {
    pub fn new(name: impl Into<String>, base: u64, size: u64, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            base,
            size,
            bytes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryZeroSegment {
    pub name: String,
    pub base: u64,
    pub size: u64,
}

impl MemoryZeroSegment {
    pub fn new(name: impl Into<String>, base: u64, size: u64) -> Self {
        Self {
            name: name.into(),
            base,
            size,
        }
    }
}

pub fn apply_memory_image(
    init_segments: &[MemoryInitSegment],
    zero_segments: &[MemoryZeroSegment],
) -> Result<(), MemoryStatus> {
    let status = with_memory_mut(|memory| {
        for segment in init_segments {
            memory.write_bytes_init(segment.base, segment.size, &segment.bytes)?;
        }
        for segment in zero_segments {
            memory.zero_fill_init(segment.base, segment.size)?;
        }
        Ok(())
    });
    if status.is_ok() {
        Ok(())
    } else {
        Err(status)
    }
}

fn with_memory_mut<F>(mut f: F) -> MemoryStatus
where
    F: FnMut(&mut RuntimeMemory) -> Result<(), MemoryStatus>,
{
    let memory = match MEMORY.get() {
        Some(memory) => memory,
        None => return MemoryStatus::Uninitialized,
    };
    let mut guard = match memory.lock() {
        Ok(guard) => guard,
        Err(_) => return MemoryStatus::Internal,
    };
    match f(&mut guard) {
        Ok(()) => MemoryStatus::Ok,
        Err(err) => err,
    }
}

fn with_memory<F, T>(mut f: F) -> Result<T, MemoryStatus>
where
    F: FnMut(&RuntimeMemory) -> Result<T, MemoryStatus>,
{
    let memory = match MEMORY.get() {
        Some(memory) => memory,
        None => return Err(MemoryStatus::Uninitialized),
    };
    let guard = match memory.lock() {
        Ok(guard) => guard,
        Err(_) => return Err(MemoryStatus::Internal),
    };
    f(&guard)
}

#[no_mangle]
pub extern "C" fn recomp_mem_load_u8(address: u64, out: *mut u64) -> MemoryStatus {
    mem_load_raw(address, 1, out)
}

#[no_mangle]
pub extern "C" fn recomp_mem_load_u16(address: u64, out: *mut u64) -> MemoryStatus {
    mem_load_raw(address, 2, out)
}

#[no_mangle]
pub extern "C" fn recomp_mem_load_u32(address: u64, out: *mut u64) -> MemoryStatus {
    mem_load_raw(address, 4, out)
}

#[no_mangle]
pub extern "C" fn recomp_mem_load_u64(address: u64, out: *mut u64) -> MemoryStatus {
    mem_load_raw(address, 8, out)
}

#[no_mangle]
pub extern "C" fn recomp_mem_store_u8(address: u64, value: u64) -> MemoryStatus {
    mem_store_raw(address, 1, value)
}

#[no_mangle]
pub extern "C" fn recomp_mem_store_u16(address: u64, value: u64) -> MemoryStatus {
    mem_store_raw(address, 2, value)
}

#[no_mangle]
pub extern "C" fn recomp_mem_store_u32(address: u64, value: u64) -> MemoryStatus {
    mem_store_raw(address, 4, value)
}

#[no_mangle]
pub extern "C" fn recomp_mem_store_u64(address: u64, value: u64) -> MemoryStatus {
    mem_store_raw(address, 8, value)
}

fn mem_load_raw(address: u64, size: usize, out: *mut u64) -> MemoryStatus {
    if out.is_null() {
        return MemoryStatus::InvalidOutPtr;
    }
    match with_memory(|memory| memory.load(address, size)) {
        Ok(value) => unsafe {
            *out = value;
            MemoryStatus::Ok
        },
        Err(err) => err,
    }
}

fn mem_store_raw(address: u64, size: usize, value: u64) -> MemoryStatus {
    with_memory_mut(|memory| memory.store(address, size, value))
}

pub(crate) fn mem_load_u8(address: u64) -> Result<u64, MemoryStatus> {
    mem_load(address, 1)
}

pub(crate) fn mem_load_u16(address: u64) -> Result<u64, MemoryStatus> {
    mem_load(address, 2)
}

pub(crate) fn mem_load_u32(address: u64) -> Result<u64, MemoryStatus> {
    mem_load(address, 4)
}

pub(crate) fn mem_load_u64(address: u64) -> Result<u64, MemoryStatus> {
    mem_load(address, 8)
}

pub(crate) fn mem_store_u8(address: u64, value: u64) -> Result<(), MemoryStatus> {
    mem_store(address, 1, value)
}

pub(crate) fn mem_store_u16(address: u64, value: u64) -> Result<(), MemoryStatus> {
    mem_store(address, 2, value)
}

pub(crate) fn mem_store_u32(address: u64, value: u64) -> Result<(), MemoryStatus> {
    mem_store(address, 4, value)
}

pub(crate) fn mem_store_u64(address: u64, value: u64) -> Result<(), MemoryStatus> {
    mem_store(address, 8, value)
}

/// Describe the guest memory region containing `address`.
pub(crate) fn query_block(address: u64) -> Option<MemoryBlock> {
    with_memory(|memory| Ok(memory.query_block(address))).ok()
}

/// The Horizon memory state of a region, from its role in the layout:
/// module segments are named `<module>-text|rodata|data|bss`.
fn region_state(spec: &MemoryRegionSpec) -> u32 {
    let name = spec.name.as_str();
    if spec.permissions.execute || name.ends_with("-rodata") {
        memory_state::CODE
    } else if name.ends_with("-data") || name.ends_with("-bss") {
        memory_state::CODE_DATA
    } else if name.starts_with("stack") {
        memory_state::STACK
    } else if name == "tls" || name == "ipc-buffer" {
        memory_state::THREAD_LOCAL
    } else if name.starts_with("shared") {
        memory_state::SHARED
    } else {
        memory_state::NORMAL
    }
}

/// Map a region after boot (shared memory, aliased stacks).
pub(crate) fn map_region(spec: MemoryRegionSpec, contents: &[u8]) -> Result<(), MemoryStatus> {
    let status = with_memory_mut(|memory| {
        memory.map(spec.clone(), contents)?;
        publish(memory);
        Ok(())
    });
    // A region that could not be placed in the reservation turns direct
    // access off.
    crate::fastmem::publish_base();
    if status.is_ok() {
        warn_above_window(&spec);
        Ok(())
    } else {
        Err(status)
    }
}

/// Say once per kind of region (its name without the address) that one was
/// mapped above the direct window a host with a 39-bit address space
/// reserves: every access to it takes the checked path, which is slow.
fn warn_above_window(spec: &MemoryRegionSpec) {
    let window = crate::fastmem::covered();
    if spec.base.saturating_add(spec.size) <= window {
        return;
    }
    static SEEN: std::sync::Mutex<std::collections::BTreeSet<String>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    let kind = spec
        .name
        .trim_end_matches(|c: char| c.is_ascii_hexdigit())
        .to_string();
    if SEEN.lock().is_ok_and(|mut seen| seen.insert(kind)) {
        eprintln!(
            "[recomp-runtime] {} at {:#x} ({:#x} bytes) is above the direct guest memory window ({window:#x}); accesses to it take the checked path",
            spec.name, spec.base, spec.size
        );
    }
}

/// Unmap the region based at `base`, returning a copy of its bytes (the
/// buffer itself is retired: another thread may still be using it).
pub(crate) fn unmap_region(base: u64) -> Option<Vec<u8>> {
    let mut bytes = None;
    with_memory_mut(|memory| {
        if let Some(data) = memory.unmap(base) {
            publish(memory);
            bytes = Some(data.to_vec());
            data.release();
        }
        Ok(())
    });
    bytes
}

/// Bulk copies within one region (IPC buffers, aliased memory).
pub(crate) fn read_block(address: u64, len: usize) -> Result<Vec<u8>, MemoryStatus> {
    match fast_region(address, len as u64) {
        Some(region) if region.read => {
            let offset = (address - region.base) as usize;
            // SAFETY: the range lies inside the live region buffer.
            Ok(unsafe { std::slice::from_raw_parts(region.data.add(offset), len) }.to_vec())
        }
        _ => with_memory(|memory| memory.read_block(address, len)),
    }
}

/// Call `read` with `len` bytes of guest memory in place, when they lie in
/// one readable region; false otherwise.
pub(crate) fn with_block(address: u64, len: usize, read: &mut dyn FnMut(&[u8])) -> bool {
    match fast_region(address, len as u64) {
        Some(region) if region.read => {
            let offset = (address - region.base) as usize;
            // SAFETY: the range lies inside the live region buffer.
            read(unsafe { std::slice::from_raw_parts(region.data.add(offset), len) });
            true
        }
        _ => false,
    }
}

/// The host address of the `len` bytes of guest memory at `address`, when
/// one region holds them.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn host_address(address: u64, len: usize) -> Option<u64> {
    let region = fast_region(address, len as u64)?;
    Some(region.data as u64 + (address - region.base))
}

/// Write `bytes` into mapped guest memory whatever its permissions: a mod
/// patches read-only data (a constant in .rodata).
pub(crate) fn patch_block(address: u64, bytes: &[u8]) -> Result<(), MemoryStatus> {
    let region = fast_region(address, bytes.len() as u64).ok_or(MemoryStatus::Unmapped)?;
    let offset = (address - region.base) as usize;
    // SAFETY: the range lies inside the live region buffer.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), region.data.add(offset), bytes.len()) };
    Ok(())
}

pub(crate) fn write_block(address: u64, bytes: &[u8]) -> Result<(), MemoryStatus> {
    match fast_region(address, bytes.len() as u64) {
        Some(region) if region.write => {
            let offset = (address - region.base) as usize;
            // SAFETY: the range lies inside the live region buffer.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), region.data.add(offset), bytes.len())
            };
            Ok(())
        }
        _ => {
            let status = with_memory_mut(|memory| memory.write_block(address, bytes));
            if status.is_ok() {
                Ok(())
            } else {
                Err(status)
            }
        }
    }
}

/// Guest memory is shared like a machine's RAM: guest threads, and the
/// kernel on their behalf, access it without host-level synchronization,
/// with the same unaligned little-endian loads and stores the hardware
/// performs.
#[inline]
fn mem_load(address: u64, size: usize) -> Result<u64, MemoryStatus> {
    checked_trace::note(address);
    if let Some(region) = fast_region(address, size as u64) {
        if region.read {
            let mut bytes = [0u8; 8];
            // SAFETY: the range lies inside the live region buffer.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    region.data.add((address - region.base) as usize),
                    bytes.as_mut_ptr(),
                    size,
                )
            };
            return Ok(u64::from_le_bytes(bytes));
        }
    }
    // Unmapped, out of bounds or not readable: the slow path names it.
    with_memory(|memory| memory.load(address, size))
}

#[inline]
fn mem_store(address: u64, size: usize, value: u64) -> Result<(), MemoryStatus> {
    checked_trace::note(address);
    if let Some(region) = fast_region(address, size as u64) {
        if region.write {
            // SAFETY: the range lies inside the live region buffer.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    value.to_le_bytes().as_ptr(),
                    region.data.add((address - region.base) as usize),
                    size,
                )
            };
            return Ok(());
        }
    }
    let status = with_memory_mut(|memory| memory.store(address, size, value));
    if status.is_ok() {
        Ok(())
    } else {
        Err(status)
    }
}

/// What a load-exclusive reserved for the calling thread: the address, the
/// access width, and the value(s) it observed. A store-exclusive only writes
/// when the memory still holds those values.
#[derive(Clone, Copy, Debug)]
struct Reservation {
    address: u64,
    size: usize,
    pair: bool,
    first: u64,
    second: u64,
}

thread_local! {
    /// The hardware's local exclusion is per thread, so the monitor is too.
    static MONITOR: Cell<Option<Reservation>> = const { Cell::new(None) };
}

/// The compare-and-swap of a store-exclusive has to be indivisible against
/// another thread's. Guest memory itself is shared without synchronization, as
/// a machine's RAM is; only the exclusive pair is guarded, by a lock chosen
/// from the address so unrelated spinlocks do not contend.
const EXCLUSIVE_SHARDS: usize = 64;
static EXCLUSIVE_LOCKS: [Mutex<()>; EXCLUSIVE_SHARDS] =
    [const { Mutex::new(()) }; EXCLUSIVE_SHARDS];

fn exclusive_shard(address: u64) -> &'static Mutex<()> {
    &EXCLUSIVE_LOCKS[((address >> 6) as usize) % EXCLUSIVE_SHARDS]
}

fn monitor_taken() -> Option<Reservation> {
    MONITOR.with(|monitor| {
        let reservation = monitor.get();
        monitor.set(None);
        reservation
    })
}

fn monitor_set(reservation: Reservation) {
    MONITOR.with(|monitor| monitor.set(Some(reservation)));
}

/// LDXR, LDAXR, LDXR pair: read the value(s) and reserve the address.
pub(crate) fn mem_load_exclusive(
    address: u64,
    size: usize,
    pair: bool,
) -> Result<(u64, u64), MemoryStatus> {
    let first = mem_load(address, size)?;
    let second = if pair {
        mem_load(address + size as u64, size)?
    } else {
        0
    };
    monitor_set(Reservation {
        address,
        size,
        pair,
        first,
        second,
    });
    // LDAXR's acquire (harmless for LDXR): later accesses, direct ones on an
    // AArch64 host included, stay after the load.
    std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
    Ok((first, second))
}

/// STXR, STLXR, STXR pair: write only if the address still holds what the
/// load observed. The result is the success flag the instruction returns: 0
/// when the store happened, 1 when it did not.
pub(crate) fn mem_store_exclusive(
    address: u64,
    size: usize,
    pair: bool,
    value: u64,
    second_value: u64,
) -> Result<u64, MemoryStatus> {
    let Some(reservation) = monitor_taken() else {
        return Ok(1);
    };
    if reservation.address != address
        || reservation.size != size
        || reservation.pair != pair
        || !is_aligned(address, size)
    {
        return Ok(1);
    }

    // The check and the write happen together under the address's lock, so no
    // other thread's exclusive store can slip between them.
    // STLXR's release (harmless for STXR): the guest's earlier accesses are
    // visible before the store. The shard lock only orders exclusives.
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    let guard = exclusive_shard(address)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let observed_first = mem_load(address, size)?;
    let observed_second = if pair {
        mem_load(address + size as u64, size)?
    } else {
        0
    };

    let unchanged =
        observed_first == reservation.first && (!pair || observed_second == reservation.second);
    if unchanged {
        mem_store(address, size, value)?;
        if pair {
            mem_store(address + size as u64, size, second_value)?;
        }
    }
    drop(guard);
    Ok(if unchanged { 0 } else { 1 })
}

/// Replace a word under the address's exclusive lock: a guest store-exclusive
/// sees the change (its compare fails) and cannot write between the read and
/// the write. The kernel uses this for mutex and arbitration words.
pub(crate) fn update_u32_exclusive(
    address: u64,
    update: &mut dyn FnMut(u32) -> u32,
) -> Result<u32, MemoryStatus> {
    let _guard = exclusive_shard(address)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let current = mem_load(address, 4)? as u32;
    mem_store(address, 4, u64::from(update(current)))?;
    Ok(current)
}

/// CLREX: drop the calling thread's reservation.
pub(crate) fn clear_exclusive() {
    monitor_taken();
}

/// A full memory barrier (DMB): guest threads run on host threads now, so
/// writes a producer makes must be visible before a consumer's read.
pub(crate) fn memory_barrier() {
    std::sync::atomic::fence(Ordering::SeqCst);
}

/// Whether `address` is aligned for an access of `size` bytes. Exclusive
/// accesses are aligned on hardware; an unaligned one fails the store.
fn is_aligned(address: u64, size: usize) -> bool {
    address.is_multiple_of(size as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The runtime memory is a process-wide singleton; unit tests that touch
    /// it must not overlap.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn test_layout() -> MemoryLayout {
        MemoryLayout::new(vec![
            MemoryRegionSpec::new("data", 0x1000, 0x40, MemoryPermissions::read_write()),
            MemoryRegionSpec::new("ro", 0x2000, 0x40, MemoryPermissions::read_only()),
        ])
    }

    #[test]
    fn load_store_round_trip() {
        let _guard = TEST_LOCK.lock().unwrap();
        init_memory(test_layout()).expect("init memory");
        mem_store_u32(0x1004, 0xDEADBEEF).expect("store");
        let value = mem_load_u32(0x1004).expect("load");
        assert_eq!(value, 0xDEADBEEF);
    }

    #[test]
    fn an_exclusive_store_writes_only_when_the_value_is_untouched() {
        let _guard = TEST_LOCK.lock().unwrap();
        init_memory(test_layout()).expect("init memory");
        mem_store_u32(0x1004, 7).expect("store");

        let (value, _) = mem_load_exclusive(0x1004, 4, false).expect("exclusive load");
        assert_eq!(value, 7);
        assert_eq!(
            mem_store_exclusive(0x1004, 4, false, 9, 0).expect("exclusive store"),
            0
        );
        assert_eq!(mem_load_u32(0x1004).expect("load"), 9);

        // A reservation is used once: a second store without a load fails.
        assert_eq!(
            mem_store_exclusive(0x1004, 4, false, 11, 0).expect("exclusive store"),
            1
        );
        assert_eq!(mem_load_u32(0x1004).expect("load"), 9);
    }

    #[test]
    fn a_store_between_the_pair_makes_the_store_exclusive_fail() {
        let _guard = TEST_LOCK.lock().unwrap();
        init_memory(test_layout()).expect("init memory");
        mem_store_u64(0x1008, 0x1000).expect("store");

        let (value, _) = mem_load_exclusive(0x1008, 8, false).expect("exclusive load");
        assert_eq!(value, 0x1000);
        mem_store_u64(0x1008, 0x2000).expect("plain store");
        assert_eq!(
            mem_store_exclusive(0x1008, 8, false, 0x3000, 0).expect("exclusive store"),
            1
        );
        assert_eq!(mem_load_u64(0x1008).expect("load"), 0x2000);
    }

    #[test]
    fn clearing_the_reservation_makes_the_store_fail() {
        let _guard = TEST_LOCK.lock().unwrap();
        init_memory(test_layout()).expect("init memory");
        mem_store_u32(0x100c, 4).expect("store");

        mem_load_exclusive(0x100c, 4, false).expect("exclusive load");
        clear_exclusive();
        assert_eq!(
            mem_store_exclusive(0x100c, 4, false, 5, 0).expect("exclusive store"),
            1
        );
    }

    #[test]
    fn a_pair_reserves_both_words() {
        let _guard = TEST_LOCK.lock().unwrap();
        init_memory(test_layout()).expect("init memory");
        mem_store_u32(0x1000, 1).expect("store");
        mem_store_u32(0x1004, 2).expect("store");

        let (first, second) = mem_load_exclusive(0x1000, 4, true).expect("pair load");
        assert_eq!((first, second), (1, 2));
        // The second word changed, so the pair store does not happen.
        mem_store_u32(0x1004, 3).expect("plain store");
        assert_eq!(
            mem_store_exclusive(0x1000, 4, true, 9, 8).expect("pair store"),
            1
        );
        assert_eq!(mem_load_u32(0x1000).expect("load"), 1);
    }

    #[test]
    fn threads_race_an_exclusive_lock() {
        let _guard = TEST_LOCK.lock().unwrap();
        init_memory(test_layout()).expect("init memory");
        mem_store_u32(0x1010, 0).expect("store");

        // The SDK's spinlock shape: reserve, check, claim. Every successful
        // claim increments the counter, so no thread's increment is lost.
        let threads: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    let mut claimed = 0u32;
                    for _ in 0..1000 {
                        loop {
                            let (value, _) = mem_load_exclusive(0x1010, 4, false).expect("load");
                            if mem_store_exclusive(0x1010, 4, false, value + 1, 0).expect("store")
                                == 0
                            {
                                claimed += 1;
                                break;
                            }
                        }
                    }
                    claimed
                })
            })
            .collect();

        let total: u32 = threads
            .into_iter()
            .map(|thread| thread.join().expect("thread"))
            .sum();
        assert_eq!(total, 4000);
        assert_eq!(mem_load_u32(0x1010).expect("load"), 4000);
    }

    #[test]
    fn kernel_updates_and_exclusive_stores_do_not_lose_each_other() {
        let _guard = TEST_LOCK.lock().unwrap();
        init_memory(test_layout()).expect("init memory");
        mem_store_u32(0x1010, 0).expect("store");

        // Guest threads increment with exclusive pairs while "kernel"
        // threads increment with update_u32_exclusive: every increment lands.
        let guests: Vec<_> = (0..2)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..1000 {
                        loop {
                            let (value, _) = mem_load_exclusive(0x1010, 4, false).expect("load");
                            if mem_store_exclusive(0x1010, 4, false, value + 1, 0).expect("store")
                                == 0
                            {
                                break;
                            }
                        }
                    }
                })
            })
            .collect();
        let kernels: Vec<_> = (0..2)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..1000 {
                        update_u32_exclusive(0x1010, &mut |value| value + 1).expect("update");
                    }
                })
            })
            .collect();
        for thread in guests.into_iter().chain(kernels) {
            thread.join().expect("thread");
        }
        assert_eq!(mem_load_u32(0x1010).expect("load"), 4000);
    }

    #[test]
    fn generated_code_accesses_guest_memory_directly() {
        let _guard = TEST_LOCK.lock().unwrap();
        init_memory(test_layout()).expect("init memory");
        assert_ne!(recomp_abi::memory_base(), 0, "the reservation is in use");
        mem_store_u32(0x1010, 0x1234_5678).expect("store");
        assert_eq!(recomp_abi::mem_load_u32(0x1010).ok(), Some(0x1234_5678));
        recomp_abi::mem_store_u16(0x1012, 0xBEEF).expect("direct store");
        assert_eq!(mem_load_u32(0x1010).expect("load"), 0xBEEF_5678);
        // Re-initializing maps fresh, zeroed pages at the same addresses.
        init_memory(test_layout()).expect("init memory");
        assert_eq!(recomp_abi::mem_load_u32(0x1010).ok(), Some(0));
    }

    #[test]
    fn regions_sharing_a_page_are_both_accessed_directly() {
        let _guard = TEST_LOCK.lock().unwrap();
        // A title's .bss starts where its .data ends, mid-page.
        let layout = MemoryLayout::new(vec![
            MemoryRegionSpec::new("data", 0x1000, 0x4b8, MemoryPermissions::read_write()),
            MemoryRegionSpec::new("bss", 0x14b8, 0x2000, MemoryPermissions::read_write()),
        ]);
        init_memory(layout).expect("init memory");
        assert_ne!(recomp_abi::memory_base(), 0, "the reservation is in use");
        recomp_abi::mem_store_u32(0x14b4, 0x1122_3344).expect("data store");
        recomp_abi::mem_store_u32(0x14b8, 0x5566_7788).expect("bss store");
        assert_eq!(mem_load_u32(0x14b4).expect("load"), 0x1122_3344);
        assert_eq!(mem_load_u32(0x14b8).expect("load"), 0x5566_7788);
        assert_eq!(recomp_abi::mem_load_u32(0x34b4).ok(), Some(0));
        init_memory(test_layout()).expect("init memory");
    }

    #[test]
    fn regions_in_the_same_host_page_keep_each_others_bytes() {
        let _guard = TEST_LOCK.lock().unwrap();
        // On a 16 KiB-page host both regions share one host page; the second
        // mapping must zero only its own guest pages, not the first region.
        let layout = MemoryLayout::new(vec![
            MemoryRegionSpec::new("data", 0x1000, 0x40, MemoryPermissions::read_write()),
            MemoryRegionSpec::new("bss", 0x2000, 0x40, MemoryPermissions::read_write()),
        ]);
        init_memory(layout).expect("init memory");
        assert_ne!(recomp_abi::memory_base(), 0, "the reservation is in use");
        recomp_abi::mem_store_u32(0x1008, 0x1111_2222).expect("first region store");
        recomp_abi::mem_store_u32(0x2008, 0x3333_4444).expect("second region store");
        assert_eq!(recomp_abi::mem_load_u32(0x1008).ok(), Some(0x1111_2222));
        assert_eq!(recomp_abi::mem_load_u32(0x2008).ok(), Some(0x3333_4444));

        // Releasing one region must not release a host page the other uses.
        let bytes = unmap_region(0x2000).expect("unmap second region");
        assert_eq!(bytes.len(), 0x40);
        assert_eq!(recomp_abi::mem_load_u32(0x1008).ok(), Some(0x1111_2222));
        assert!(mem_load_u32(0x2008).is_err(), "second region is gone");
        if crate::fastmem::host_page() > crate::fastmem::PAGE {
            // A 16 KiB host page cannot protect one 4 KiB guest page. The
            // released bytes are zeroed, so direct code cannot read the old
            // data even though the host page stays mapped for the neighbour.
            assert_eq!(recomp_abi::mem_load_u32(0x2008).ok(), Some(0));
        }
        init_memory(test_layout()).expect("restore layout");
    }

    #[test]
    fn unaligned_access_is_allowed() {
        let _guard = TEST_LOCK.lock().unwrap();
        init_memory(test_layout()).expect("init memory");
        mem_store_u32(0x1002, 0xDEADBEEF).expect("unaligned store");
        assert_eq!(mem_load_u32(0x1002).expect("unaligned load"), 0xDEADBEEF);
    }

    #[test]
    fn permission_denied_for_write() {
        let _guard = TEST_LOCK.lock().unwrap();
        init_memory(test_layout()).expect("init memory");
        let err = mem_store_u8(0x2000, 1).unwrap_err();
        assert_eq!(err, MemoryStatus::PermissionDenied);
    }

    #[test]
    fn apply_memory_image_writes_and_zeros() {
        let _guard = TEST_LOCK.lock().unwrap();
        init_memory(test_layout()).expect("init memory");
        let init = MemoryInitSegment::new("data", 0x1000, 4, vec![1, 2, 3, 4]);
        let zero = MemoryZeroSegment::new("bss", 0x1004, 4);
        apply_memory_image(&[init], &[zero]).expect("apply image");

        let value = mem_load_u32(0x1000).expect("load");
        assert_eq!(value, 0x04030201);
        let zeroed = mem_load_u32(0x1004).expect("load zero");
        assert_eq!(zeroed, 0);
    }
}

/// `RECOMP_TRACE_CHECKED=1`: which guest loads and stores reach the checked
/// path (the runtime call generated code makes when an address is not in
/// the direct window), printed every 10 s: the 256 MiB ranges they fall in
/// and, sampled, the guest blocks making them.
mod checked_trace {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    const RANGES: usize = 1 << 11;
    static COUNTS: [AtomicU64; RANGES] = [const { AtomicU64::new(0) }; RANGES];
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static BLOCKS: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);

    fn enabled() -> bool {
        static ENABLED: AtomicBool = AtomicBool::new(false);
        static STARTED: OnceLock<()> = OnceLock::new();
        STARTED.get_or_init(|| {
            if std::env::var("RECOMP_TRACE_CHECKED").is_ok_and(|value| value == "1") {
                ENABLED.store(true, Ordering::Release);
                let _ = std::thread::Builder::new()
                    .name("recomp-checked".to_string())
                    .spawn(report);
            }
        });
        ENABLED.load(Ordering::Relaxed)
    }

    #[inline]
    pub(super) fn note(address: u64) {
        if !enabled() {
            return;
        }
        COUNTS[(address >> 28) as usize % RANGES].fetch_add(1, Ordering::Relaxed);
        if CALLS.fetch_add(1, Ordering::Relaxed).is_multiple_of(1024) {
            let block = recomp_abi::current_block();
            if let Ok(mut blocks) = BLOCKS.lock() {
                *blocks
                    .get_or_insert_with(HashMap::new)
                    .entry(block)
                    .or_default() += 1;
            }
        }
    }

    fn report() {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            let calls = CALLS.swap(0, Ordering::Relaxed);
            let mut ranges: Vec<(u64, usize)> = COUNTS
                .iter()
                .enumerate()
                .map(|(range, count)| (count.swap(0, Ordering::Relaxed), range))
                .filter(|(count, _)| *count > 0)
                .collect();
            ranges.sort_unstable_by(|a, b| b.cmp(a));
            let mut text = format!("[recomp-checked] {calls} checked accesses in 10 s;");
            for (count, range) in ranges.iter().take(6) {
                text.push_str(&format!(" {:#x}: {count}", (*range as u64) << 28));
            }
            let mut blocks: Vec<(u64, u64)> = BLOCKS
                .lock()
                .ok()
                .and_then(|mut blocks| blocks.take())
                .unwrap_or_default()
                .into_iter()
                .map(|(block, count)| (count, block))
                .collect();
            blocks.sort_unstable_by(|a, b| b.cmp(a));
            text.push_str("; blocks (1 in 1024):");
            for (count, block) in blocks.iter().take(8) {
                text.push_str(&format!(" {block:#x} x{count}"));
            }
            println!("{text}");
        }
    }
}
