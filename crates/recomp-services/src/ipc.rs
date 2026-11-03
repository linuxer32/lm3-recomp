//! HIPC/CMIF IPC emulation for statically recompiled processes.
//!
//! Kernel SVCs `svcConnectToNamedPort`, `svcSendSyncRequest*`, and
//! `svcCloseHandle` are routed here. Wire layouts follow the Nintendo IPC
//! message framing the SDK actually builds (switchbrew.org/wiki/HIPC):
//! a 16-byte message header (type/options/pid), the HIPC header pair at
//! buffer+16, buffer descriptors from buffer+24, a receive list, and a CMIF
//! payload (SFCI magic, version, method id, inline params). Replies write
//! kernel-style pointer descriptors for handles ({offset 0, size handle})
//! and an SFCO CMIF block after the raw return data. `fs:GLOBAL` / `fs:USER`
//! are backed by a host RomFS image file when configured, otherwise by an
//! in-memory RomFS window in the guest address space.
//!
//! Deliberate simplifications vs. the kernel:
//! - Domain routing and per-session tokens are parsed but not enforced;
//!   named ports resolve directly to service objects.
//! - Requests are located by scanning for the SFCI magic (SDK builders vary
//!   in descriptor placement); replies follow libnx conventions.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Guest-memory access provided by the runtime.
pub trait MemoryAccess: Send + Sync + std::fmt::Debug {
    fn load_u8(&self, addr: u64) -> Option<u8>;
    fn load_u32(&self, addr: u64) -> Option<u32>;
    fn load_u64(&self, addr: u64) -> Option<u64>;
    fn store_u32(&self, addr: u64, value: u32) -> bool;
    fn store_u64(&self, addr: u64, value: u64) -> bool;
    fn read_bytes(&self, addr: u64, len: usize) -> Option<Vec<u8>>;
    fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool;

    /// Call `read` with the `len` bytes at `addr`, in place where the memory
    /// allows (no copy); false when they cannot be read. The renderer hashes
    /// tens of MB of vertices a frame this way.
    fn read_with(&self, addr: u64, len: usize, read: &mut dyn FnMut(&[u8])) -> bool {
        let Some(bytes) = self.read_bytes(addr, len) else {
            return false;
        };
        read(&bytes);
        true
    }

    /// Watch writes to the `len` bytes at `addr`, and report the parts
    /// written since check `since` to `written` (offset into the range and
    /// length, whole pages clipped to it). `since` `None` starts watching and
    /// reports nothing: read the bytes after it. Returns this check's number,
    /// the next call's `since`; `None` (the default) when writes there are
    /// not watched, and the bytes have to be read to see whether they
    /// changed.
    fn written_since(
        &self,
        _addr: u64,
        _len: usize,
        _since: Option<u64>,
        _written: &mut dyn FnMut(usize, usize),
    ) -> Option<u64> {
        None
    }

    /// Describe the block of address space containing `addr`, mapped or
    /// free, as svcQueryMemory reports it.
    fn query_block(&self, _addr: u64) -> Option<MemoryBlock> {
        None
    }

    /// Replace the word at `addr` with `update(current)` in one step that
    /// guest store-exclusives cannot split, and return the word it replaced.
    /// The kernel changes mutex and arbitration words this way: with guest
    /// threads running in parallel, a plain load and store would lose an
    /// owner's release made between them. The default suits one thread.
    fn update_u32(&self, addr: u64, update: &mut dyn FnMut(u32) -> u32) -> Option<u32> {
        let current = self.load_u32(addr)?;
        self.store_u32(addr, update(current)).then_some(current)
    }
}

/// svc::MemoryState values reported by svcQueryMemory.
pub mod memory_state {
    pub const FREE: u32 = 0x00;
    pub const CODE: u32 = 0x03;
    pub const CODE_DATA: u32 = 0x04;
    pub const NORMAL: u32 = 0x05;
    pub const SHARED: u32 = 0x06;
    pub const STACK: u32 = 0x0B;
    pub const THREAD_LOCAL: u32 = 0x0C;
    pub const INACCESSIBLE: u32 = 0x10;
}

/// One block of the guest address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBlock {
    pub base: u64,
    pub size: u64,
    /// A `memory_state` value.
    pub state: u32,
    /// svc::MemoryPermission bits: 1 read, 2 write, 4 execute.
    pub perm: u32,
}

pub const IPC_RESULT_OK: u64 = 0;
/// IPC message type: kernel response to a sync request.
pub const IPC_MESSAGE_TYPE_RESPONSE: u32 = 4;
pub const IPC_RESULT_NOT_FOUND: u64 = 0xC0A00009;
/// svc::ResultInvalidHandle (module 1, description 114).
pub const IPC_RESULT_INVALID_HANDLE: u64 = 0xE401;
pub const IPC_RESULT_BAD_MESSAGE: u64 = 0xE001C001;

/// HIPC Header0Tag magic (bits 0..15).
pub const HIPC_MAGIC: u32 = 0x4000;
/// CMIF request/response magic words.
pub const SFCI_MAGIC: u32 = 0x4943_4653; // 'SFCI'
pub const SFCO_MAGIC: u32 = 0x4F43_4653; // 'SFCO'
/// Legacy FS DirectoryEntry: path[256], u64 size, u32 type (0 file, 1 dir).
pub const DIRECTORY_ENTRY_SIZE: usize = 256 + 8 + 4;

#[derive(Debug, Clone)]
pub struct DirectoryEntry {
    pub path: String,
    pub size: u64,
    pub directory: bool,
}

impl DirectoryEntry {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![0u8; DIRECTORY_ENTRY_SIZE];
        let bytes = self.path.as_bytes();
        let n = bytes.len().min(255);
        out[..n].copy_from_slice(&bytes[..n]);
        out[256..264].copy_from_slice(&self.size.to_le_bytes());
        let kind = if self.directory { 1u32 } else { 0u32 };
        out[264..268].copy_from_slice(&kind.to_le_bytes());
        out
    }
}

/// Parsed HIPC request framing (Nintendo message layout).
#[derive(Debug, Clone)]
pub struct HipcRequest {
    pub msg_type: u32,
    pub options: u32,
    pub header0: u32,
    pub header1: u32,
    /// Inline pointer descriptors resolved to absolute guest addresses.
    pub pointer_buffers: Vec<(u64, u32)>,
    /// Absolute receive-list addresses and declared sizes.
    pub receive_list: Vec<(u64, u32)>,
    /// Byte offset of the receive-list descriptor region.
    pub receive_list_addr: u64,
    /// Byte offset of the CMIF payload (magic word).
    pub raw_addr: u64,
    /// CMIF method id.
    pub method: u32,
    /// Byte offset of inline CMIF parameters (raw_addr + 12).
    pub args_addr: u64,
}

/// Message header size: type/options u32 pair + reserved/pid u64.
const MESSAGE_HEADER_SIZE: u64 = 16;
/// HIPC header pair offset inside the message buffer.
const HIPC_HEADER_OFFSET: u64 = MESSAGE_HEADER_SIZE;

impl HipcRequest {
    pub fn parse(mem: &dyn MemoryAccess, buffer: u64) -> Option<Self> {
        let msg_type = mem.load_u32(buffer)?;
        let options = mem.load_u32(buffer + 4)?;
        let header0 = mem.load_u32(buffer + HIPC_HEADER_OFFSET)?;
        let header1 = mem.load_u32(buffer + HIPC_HEADER_OFFSET + 4)?;

        let pointer_count = (header0 >> 16) & 0xF;
        let send_count = (header0 >> 20) & 0xF;
        let receive_count = (header0 >> 24) & 0xF;
        let exchange_count = (header0 >> 28) & 0xF;
        let receive_list_count = (header1 >> 10) & 0xF;
        let receive_list_offset_words = (header1 >> 20) & 0x7FF;
        let special = (header1 >> 31) & 1;

        let mut pos = buffer + HIPC_HEADER_OFFSET + 8;
        if special != 0 {
            let special_header = mem.load_u32(pos)?;
            let copy_count = ((special_header >> 1) & 0xF) as u64;
            let move_count = ((special_header >> 5) & 0xF) as u64;
            // Special header word + one u32 handle descriptor per copy/move.
            pos += 4 + (copy_count + move_count) * 4;
        }

        let mut pointer_buffers = Vec::new();
        for _ in 0..pointer_count {
            let offset = mem.load_u32(pos)?;
            let size = mem.load_u32(pos + 4)?;
            // offset 0 with a nonzero size encodes an inline handle (replies).
            if offset != 0 {
                pointer_buffers.push((buffer + u64::from(offset), size));
            }
            pos += 8;
        }
        // Send/receive/exchange descriptors are {offset, size} pairs the
        // kernel translates into addresses; fsp-srv methods use inline data.
        pos += u64::from(send_count + receive_count + exchange_count) * 8;

        let rl_base = pos + u64::from(receive_list_offset_words) * 4;
        let mut receive_list = Vec::new();
        let mut cursor = rl_base;
        for _ in 0..receive_list_count {
            let offset = mem.load_u32(cursor).unwrap_or(0);
            let size = mem.load_u32(cursor + 4).unwrap_or(0);
            receive_list.push((buffer + u64::from(offset), size));
            cursor += 8;
        }

        // Locate the CMIF payload: SDK builders vary in descriptor layout,
        // so scan the message region for the SFCI magic word.
        let mut raw_addr = None;
        let mut scan = pos;
        while scan + 8 <= buffer + 0x100 {
            if mem.load_u32(scan) == Some(SFCI_MAGIC) {
                raw_addr = Some(scan);
                break;
            }
            scan += 4;
        }
        let raw_addr = raw_addr?;
        let version = mem.load_u32(raw_addr + 4)?;
        if version != 1 {
            return None;
        }
        let method = mem.load_u32(raw_addr + 8)?;

        Some(Self {
            msg_type,
            options,
            header0,
            header1,
            pointer_buffers,
            receive_list,
            receive_list_addr: pos + u64::from(receive_list_offset_words) * 4,
            raw_addr,
            method,
            args_addr: raw_addr + 12,
        })
    }
}

#[derive(Debug, Clone)]
enum Object {
    FsSrv,
    FileSystem {
        base: u64,
        len: u64,
    },
    Storage {
        base: u64,
        len: u64,
    },
    Directory {
        entries: Vec<DirectoryEntry>,
    },
    /// IStorage backed by a host RomFS image file. Reads are served from disk
    /// so multi-gigabyte images never enter the guest address space.
    HostStorage {
        image: PathBuf,
    },
    /// IFileSystem backed by a host RomFS image (metadata parsed on demand).
    HostFileSystem {
        image: PathBuf,
    },
    /// IFile backed by a region of the host RomFS image.
    HostFile {
        image: PathBuf,
        offset: u64,
        len: u64,
    },
    /// Placeholder service object: replies success to any request so SDK
    /// startup paths (ns:, am:, gl:, sm:...) can progress until real service
    /// emulation is implemented.
    Placeholder {
        name: String,
    },
}

#[derive(Default)]
struct HubState {
    next_handle: u64,
    objects: BTreeMap<u64, Object>,
    /// RomFS metadata parsed from the host image, cached after first use.
    host_meta: Option<HostRomFs>,
}

#[derive(Clone)]
pub struct IpcHub {
    inner: Arc<HubInner>,
}

impl std::fmt::Debug for IpcHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IpcHub")
            .field("romfs_base", &self.inner.romfs_base)
            .field("romfs_size", &self.inner.romfs_size)
            .field("romfs_image", &self.inner.romfs_image)
            .finish()
    }
}

struct HubInner {
    mem: Arc<dyn MemoryAccess>,
    romfs_base: u64,
    romfs_size: u64,
    romfs_image: Option<PathBuf>,
    state: Mutex<HubState>,
    /// CMIF sessions reached through `sm:`. Locked inside, per session, so a
    /// service that blocks does not hold up the others.
    server: crate::server::Server,
}

/// Save data lives next to the RomFS image, or where `RECOMP_SAVE_DIR` says
/// (tests then leave the player's saves alone). An empty variable keeps the
/// default, like the other runtime asset overrides.
fn save_root(romfs_image: Option<&Path>) -> Option<PathBuf> {
    std::env::var_os("RECOMP_SAVE_DIR")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            romfs_image
                .and_then(Path::parent)
                .map(|dir| dir.join("save"))
        })
}

impl IpcHub {
    pub fn new(
        mem: Arc<dyn MemoryAccess>,
        romfs_base: u64,
        romfs_size: u64,
        romfs_image: Option<PathBuf>,
    ) -> Self {
        let mut state = HubState {
            next_handle: 2,
            objects: BTreeMap::new(),
            host_meta: None,
        };
        // Pre-allocate the fs: session object libnx connects to at boot.
        state.objects.insert(1, Object::FsSrv);
        let save_root = save_root(romfs_image.as_deref());
        let server = crate::server::Server::new(crate::server::ServerContext {
            romfs_image: romfs_image.clone(),
            save_root,
        });
        Self {
            inner: Arc::new(HubInner {
                mem,
                romfs_base,
                romfs_size,
                romfs_image,
                state: Mutex::new(state),
                server,
            }),
        }
    }

    /// Write a u64 into guest memory (svcConnectToNamedPort out-handle).
    pub fn write_u64(&self, addr: u64, value: u64) -> bool {
        self.inner.mem.store_u64(addr, value)
    }

    /// svcConnectToNamedPort: returns (result code, handle).
    pub fn connect_to_named_port(&self, name_addr: u64) -> (u64, u64) {
        let name = self.read_cstring(name_addr);
        match name.as_str() {
            "sm:" => {
                let handle = self
                    .server()
                    .open(Box::new(crate::services::ServiceManager));
                (IPC_RESULT_OK, u64::from(handle))
            }
            "fs:GLOBAL" | "fs:USER" | "fs:SYSTEM" => (IPC_RESULT_OK, 1),
            other => {
                let mut state = self
                    .inner
                    .state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let handle = state.next_handle;
                state.next_handle += 1;
                state.objects.insert(
                    handle,
                    Object::Placeholder {
                        name: other.to_string(),
                    },
                );
                println!("[recomp-ipc] placeholder named port: {other} (handle {handle})");
                (IPC_RESULT_OK, handle)
            }
        }
    }

    fn read_cstring(&self, addr: u64) -> String {
        let mut bytes = Vec::new();
        let mut cur = addr;
        while bytes.len() < 256 {
            match self.inner.mem.load_u8(cur) {
                Some(0) | None => break,
                Some(byte) => {
                    bytes.push(byte);
                    cur += 1;
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The session registry. The server locks its own maps, so no hub-wide
    /// lock is held while a service call runs.
    fn server(&self) -> &crate::server::Server {
        &self.inner.server
    }

    /// svcCloseHandle
    pub fn close_handle(&self, handle: u64) -> u64 {
        if self.server().close(handle) {
            return IPC_RESULT_OK;
        }
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.objects.remove(&handle);
        IPC_RESULT_OK
    }

    /// svcSendSyncRequest: decode the HIPC request at `buffer_addr`, dispatch
    /// it, write the reply, and return the kernel result code.
    pub fn send_sync_request(&self, handle: u64, buffer_addr: u64) -> u64 {
        if self.server().owns(handle) {
            return self
                .server()
                .dispatch(self.inner.mem.as_ref(), handle, buffer_addr);
        }
        // The kernel validates the session handle before reading the message.
        let Some(object) = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .objects
            .get(&handle)
            .cloned()
        else {
            return IPC_RESULT_INVALID_HANDLE;
        };
        let Some(request) = HipcRequest::parse(self.inner.mem.as_ref(), buffer_addr) else {
            println!("[recomp-ipc] bad HIPC header at {buffer_addr:#x}");
            return IPC_RESULT_BAD_MESSAGE;
        };

        println!(
            "[recomp-ipc] request handle={handle} method={:#x} type={:#x} opts={:#x} ptr={} recv={} rl={}",
            request.method,
            request.msg_type,
            request.options,
            request.pointer_buffers.len(),
            request.receive_list.len(),
            request.receive_list.len(),
        );
        if std::env::var("RECOMP_IPC_DUMP").is_ok() {
            if let Some(bytes) = self.inner.mem.read_bytes(buffer_addr, 0x100) {
                println!("[recomp-ipc] request buffer dump: {}", hex_dump(&bytes));
            }
        }

        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        let reply = dispatch_fs(self.inner.as_ref(), &object, &request, &mut state);
        drop(state);

        // Kernel-style reply: message header, HIPC header at +16, pointer
        // descriptors for handles from +24, raw return data after the
        // descriptors, and the SFCO CMIF block aligned after the raw data.
        if !self
            .inner
            .mem
            .store_u32(buffer_addr, IPC_MESSAGE_TYPE_RESPONSE)
        {
            return IPC_RESULT_BAD_MESSAGE;
        }
        if !self.inner.mem.store_u32(buffer_addr + 4, 0) {
            return IPC_RESULT_BAD_MESSAGE;
        }
        if !self.inner.mem.store_u64(buffer_addr + 8, 0) {
            return IPC_RESULT_BAD_MESSAGE;
        }
        let header0 = HIPC_MAGIC | ((reply.handles.len() as u32) << 16);
        if !self
            .inner
            .mem
            .store_u32(buffer_addr + HIPC_HEADER_OFFSET, header0)
        {
            return IPC_RESULT_BAD_MESSAGE;
        }
        let data_words = reply.data.len().div_ceil(4) as u32;
        let header1 = data_words | ((reply.receive_written as u32) << 10);
        if !self
            .inner
            .mem
            .store_u32(buffer_addr + HIPC_HEADER_OFFSET + 4, header1)
        {
            return IPC_RESULT_BAD_MESSAGE;
        }

        let mut pos = buffer_addr + HIPC_HEADER_OFFSET + 8;
        for handle in &reply.handles {
            if !self.inner.mem.store_u32(pos, 0) {
                return IPC_RESULT_BAD_MESSAGE;
            }
            if !self.inner.mem.store_u32(pos + 4, *handle as u32) {
                return IPC_RESULT_BAD_MESSAGE;
            }
            pos += 8;
        }
        if !reply.data.is_empty() && !self.inner.mem.write_bytes(pos, &reply.data) {
            return IPC_RESULT_BAD_MESSAGE;
        }
        let cmif_block = (pos + u64::from(data_words) * 4 + 31) & !15;
        if !self.inner.mem.store_u32(cmif_block, SFCO_MAGIC) {
            return IPC_RESULT_BAD_MESSAGE;
        }
        if !self.inner.mem.store_u32(cmif_block + 4, 1) {
            return IPC_RESULT_BAD_MESSAGE;
        }
        if !self.inner.mem.store_u32(cmif_block + 8, reply.status) {
            return IPC_RESULT_BAD_MESSAGE;
        }
        if std::env::var("RECOMP_IPC_DUMP").is_ok() {
            if let Some(bytes) = self.inner.mem.read_bytes(buffer_addr, 0x100) {
                println!("[recomp-ipc] reply buffer dump: {}", hex_dump(&bytes));
            }
        }

        // Rewrite receive-list entry sizes to the bytes actually returned.
        for index in 0..reply.receive_written.min(request.receive_list.len()) {
            let entry_addr = request.receive_list_addr + (index as u64) * 8 + 4;
            let actual = reply.received_bytes[index].min(u64::from(request.receive_list[index].1));
            if !self
                .inner
                .mem
                .store_u32(entry_addr, (actual as u32) & 0xFFFF)
            {
                return IPC_RESULT_BAD_MESSAGE;
            }
        }
        IPC_RESULT_OK
    }
}

struct Reply {
    status: u32,
    data: Vec<u8>,
    /// Handles returned as kernel pointer descriptors ({offset 0, size handle}).
    handles: Vec<u64>,
    receive_written: usize,
    received_bytes: Vec<u64>,
}

impl Reply {
    fn ok(data: Vec<u8>) -> Self {
        Self {
            status: 0,
            data,
            handles: Vec::new(),
            receive_written: 0,
            received_bytes: Vec::new(),
        }
    }

    fn handle(handle: u64) -> Self {
        Self {
            status: 0,
            data: Vec::new(),
            handles: vec![handle],
            receive_written: 0,
            received_bytes: Vec::new(),
        }
    }

    fn with_receive(data: Vec<u8>, received: Vec<u64>) -> Self {
        Self {
            status: 0,
            data,
            handles: Vec::new(),
            receive_written: received.len(),
            received_bytes: received,
        }
    }

    fn err(status: u32) -> Self {
        Self {
            status,
            data: Vec::new(),
            handles: Vec::new(),
            receive_written: 0,
            received_bytes: Vec::new(),
        }
    }
}

fn read_inline_u64(mem: &dyn MemoryAccess, addr: u64) -> Option<u64> {
    // CMIF parameters are word-aligned, so a u64 may sit at addr % 8 == 4;
    // assemble from two u32 loads instead of requiring 8-byte alignment.
    let lo = u64::from(mem.load_u32(addr)?);
    let hi = u64::from(mem.load_u32(addr + 4)?);
    Some(lo | (hi << 32))
}

fn read_inline_string(mem: &dyn MemoryAccess, addr: u64) -> Option<String> {
    let len = mem.load_u32(addr)? as usize;
    let bytes = mem.read_bytes(addr + 4, len)?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// Raw path from a pointer descriptor buffer (NUL-trimmed), falling back to
/// the legacy length-prefixed inline string at the CMIF args offset.
fn request_path(hub: &HubInner, request: &HipcRequest) -> Option<String> {
    if let Some((addr, size)) = request.pointer_buffers.first() {
        let bytes = hub.mem.read_bytes(*addr, (*size as usize).min(256))?;
        let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
        return Some(String::from_utf8_lossy(&bytes[..end]).into_owned());
    }
    read_inline_string(hub.mem.as_ref(), request.args_addr)
}

fn hex_dump(bytes: &[u8]) -> String {
    bytes
        .chunks(16)
        .map(|chunk| {
            chunk
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Host-image backed RomFS metadata: header tables parsed once from disk.
struct HostRomFs {
    image_size: u64,
    data_offset: u64,
    dirs: Vec<RomDirEntry>,
    files: Vec<RomFileEntry>,
}

/// Nintendo RomFS header/table layout (the header size field marks the data
/// region start; tables live at the end of the image, after the data).
/// Directory entry: 24-byte header (parent index u32@0, name length u32@20),
/// inline name padded to 4. File entry: 32-byte header (directory table byte
/// offset u32@4, data offset u64@8 relative to the header size, size u64@16,
/// name length u32@28), inline name padded to 4.
fn parse_romfs_tables(
    header: &[u8],
    dir_bytes: &[u8],
    file_bytes: &[u8],
) -> Option<(Vec<RomDirEntry>, Vec<RomFileEntry>, u64)> {
    fn rd(bytes: &[u8], off: usize) -> Option<u64> {
        let raw: [u8; 8] = bytes.get(off..off + 8)?.try_into().ok()?;
        Some(u64::from_le_bytes(raw))
    }
    let data_base = rd(header, 0)?;

    let mut dirs = Vec::new();
    let mut dir_index_by_offset = BTreeMap::new();
    let mut cursor = 0usize;
    while cursor + 24 <= dir_bytes.len() {
        let parent = u32::from_le_bytes(dir_bytes[cursor..cursor + 4].try_into().unwrap());
        let name_len =
            u32::from_le_bytes(dir_bytes[cursor + 20..cursor + 24].try_into().unwrap()) as usize;
        if cursor + 24 + name_len > dir_bytes.len() {
            break;
        }
        let name =
            String::from_utf8_lossy(&dir_bytes[cursor + 24..cursor + 24 + name_len]).into_owned();
        dir_index_by_offset.insert(cursor, dirs.len() as u32);
        dirs.push(RomDirEntry { parent, name });
        cursor += 24 + name_len.next_multiple_of(4);
    }
    if dirs.is_empty() {
        return None;
    }

    let mut files = Vec::new();
    let mut cursor = 0usize;
    while cursor + 32 <= file_bytes.len() {
        let dir_offset = u32::from_le_bytes(file_bytes[cursor + 4..cursor + 8].try_into().unwrap());
        let offset = u64::from_le_bytes(file_bytes[cursor + 8..cursor + 16].try_into().unwrap());
        let size = u64::from_le_bytes(file_bytes[cursor + 16..cursor + 24].try_into().unwrap());
        let name_len =
            u32::from_le_bytes(file_bytes[cursor + 28..cursor + 32].try_into().unwrap()) as usize;
        if cursor + 32 + name_len > file_bytes.len() {
            break;
        }
        let name =
            String::from_utf8_lossy(&file_bytes[cursor + 32..cursor + 32 + name_len]).into_owned();
        let dir = dir_index_by_offset
            .get(&usize::try_from(dir_offset).ok()?)
            .copied()
            .unwrap_or(0);
        files.push(RomFileEntry {
            dir,
            name,
            offset,
            size,
        });
        cursor += 32 + name_len.next_multiple_of(4);
    }

    Some((dirs, files, data_base))
}

impl HostRomFs {
    fn parse(path: &std::path::Path) -> Option<Self> {
        let image_size = host_image_size(path)?;
        let header = host_read_at(path, 0, 0x48)?;
        let header_size = u64::from_le_bytes(header[0..8].try_into().ok()?);
        if header_size < 0x40 || image_size < header_size {
            return None;
        }
        let dir_table_off = u64::from_le_bytes(header[0x18..0x20].try_into().ok()?);
        let dir_table_size = u64::from_le_bytes(header[0x20..0x28].try_into().ok()?);
        let file_table_off = u64::from_le_bytes(header[0x38..0x40].try_into().ok()?);
        let file_table_size = u64::from_le_bytes(header[0x40..0x48].try_into().ok()?);
        if dir_table_off + dir_table_size > image_size
            || file_table_off + file_table_size > image_size
        {
            return None;
        }
        let dir_bytes = host_read_at(path, dir_table_off, dir_table_size as usize)?;
        let file_bytes = host_read_at(path, file_table_off, file_table_size as usize)?;
        let (dirs, files, data_base) = parse_romfs_tables(&header, &dir_bytes, &file_bytes)?;
        Some(Self {
            image_size,
            data_offset: data_base,
            dirs,
            files,
        })
    }

    fn lookup(&self, path: &str) -> Option<RomFileEntry> {
        let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();
        let mut dir_index: u32 = 0;
        for (i, seg) in segments.iter().enumerate() {
            if seg.is_empty() {
                continue;
            }
            if i + 1 == segments.len() {
                return self
                    .files
                    .iter()
                    .find(|f| f.dir == dir_index && f.name == *seg)
                    .cloned();
            }
            dir_index = self
                .dirs
                .iter()
                .enumerate()
                .find(|(_, d)| d.parent == dir_index && d.name == *seg)
                .map(|(index, _)| index as u32)?;
        }
        None
    }

    fn list(&self, path: &str) -> Vec<DirectoryEntry> {
        let mut dir_index: u32 = 0;
        for seg in path.trim_start_matches('/').split('/') {
            if seg.is_empty() {
                continue;
            }
            if let Some((index, _)) = self
                .dirs
                .iter()
                .enumerate()
                .find(|(_, d)| d.parent == dir_index && d.name == seg)
            {
                dir_index = index as u32;
            }
        }
        let mut out = Vec::new();
        for (index, dir) in self.dirs.iter().enumerate() {
            if dir.parent == dir_index && !(dir_index == 0 && index == 0) {
                out.push(DirectoryEntry {
                    path: dir.name.clone(),
                    size: 0,
                    directory: true,
                });
            }
        }
        for file in &self.files {
            if file.dir == dir_index {
                out.push(DirectoryEntry {
                    path: file.name.clone(),
                    size: file.size,
                    directory: false,
                });
            }
        }
        out
    }
}

fn host_image_size(path: &std::path::Path) -> Option<u64> {
    std::fs::File::open(path)
        .ok()
        .and_then(|file| file.metadata().ok())
        .map(|meta| meta.len())
}

fn host_read_at(path: &std::path::Path, offset: u64, len: usize) -> Option<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let size = file.metadata().ok()?.len();
    if offset >= size || len == 0 {
        return Some(Vec::new());
    }
    let want = len.min((size - offset) as usize);
    file.seek(SeekFrom::Start(offset)).ok()?;
    let mut buf = vec![0u8; want];
    if file.read_exact(&mut buf).is_err() {
        return None;
    }
    Some(buf)
}

fn ensure_host_meta<'a>(hub: &HubInner, state: &'a mut HubState) -> Option<&'a HostRomFs> {
    if state.host_meta.is_none() {
        let path = hub.romfs_image.as_ref()?;
        state.host_meta = HostRomFs::parse(path);
        match &state.host_meta {
            Some(meta) => println!(
                "[recomp-ipc] host RomFS: {} dirs, {} files, {} bytes",
                meta.dirs.len(),
                meta.files.len(),
                meta.image_size
            ),
            None => println!("[recomp-ipc] host RomFS image parse failed: {path:?}"),
        }
    }
    state.host_meta.as_ref()
}

fn romfs_window(hub: &HubInner) -> Option<(u64, u64)> {
    if hub.romfs_size == 0 {
        return None;
    }
    Some((hub.romfs_base, hub.romfs_size))
}

fn dispatch_fs(
    hub: &HubInner,
    object: &Object,
    request: &HipcRequest,
    state: &mut HubState,
) -> Reply {
    let method = request.method;
    let args_addr = request.args_addr;

    match object {
        Object::FsSrv => match method {
            // fsp-srv: SetCurrentProcess
            1 => Reply::ok(Vec::new()),
            // fsp-srv: OpenDataFileSystemByCurrentProcess -> IFileSystem
            2 => {
                state.next_handle += 1;
                let handle = state.next_handle;
                if let Some(image) = hub.romfs_image.clone() {
                    state
                        .objects
                        .insert(handle, Object::HostFileSystem { image });
                } else if let Some((base, len)) = romfs_window(hub) {
                    state
                        .objects
                        .insert(handle, Object::FileSystem { base, len });
                } else {
                    return Reply::err(0x260902); // RomFs not mounted
                }
                Reply::handle(handle)
            }
            // fsp-srv: OpenDataStorageByCurrentProcess -> romfs storage
            200 => {
                state.next_handle += 1;
                let handle = state.next_handle;
                if let Some(image) = hub.romfs_image.clone() {
                    state.objects.insert(handle, Object::HostStorage { image });
                } else if let Some((base, len)) = romfs_window(hub) {
                    state.objects.insert(handle, Object::Storage { base, len });
                } else {
                    return Reply::err(0x260902);
                }
                Reply::handle(handle)
            }
            // fsp-srv: OpenDataStorageByDataId -> same window (patch/dlc stub)
            202 => {
                state.next_handle += 1;
                let handle = state.next_handle;
                if let Some(image) = hub.romfs_image.clone() {
                    state.objects.insert(handle, Object::HostStorage { image });
                } else if let Some((base, len)) = romfs_window(hub) {
                    state.objects.insert(handle, Object::Storage { base, len });
                } else {
                    return Reply::err(0x260902);
                }
                Reply::handle(handle)
            }
            _ => Reply::err(0x0802A03C), // unsupported method
        },
        Object::Placeholder { name } => {
            println!("[recomp-ipc] placeholder reply for {name}");
            Reply::ok(Vec::new())
        }
        Object::FileSystem { base, len } => {
            let Some(image) = hub.mem.read_bytes(*base, *len as usize) else {
                return Reply::err(0x260902);
            };
            let Some(fs) = RomFs::parse(&image) else {
                return Reply::err(0x260902);
            };
            match method {
                // IFileSystem::OpenFile(path) -> storage over the file bytes
                1 => {
                    let Some(path) = request_path(hub, request) else {
                        return Reply::err(0x0802A01E);
                    };
                    let Some(entry) = fs.lookup(&path) else {
                        return Reply::err(0x02018C01); // path not found
                    };
                    state.next_handle += 1;
                    let handle = state.next_handle;
                    state.objects.insert(
                        handle,
                        Object::Storage {
                            base: *base + fs.data_base + entry.offset,
                            len: entry.size,
                        },
                    );
                    Reply::handle(handle)
                }
                // IFileSystem::OpenDirectory(path) -> directory object
                2 => {
                    let Some(path) = request_path(hub, request) else {
                        return Reply::err(0x0802A01E);
                    };
                    state.next_handle += 1;
                    let handle = state.next_handle;
                    state.objects.insert(
                        handle,
                        Object::Directory {
                            entries: fs.list(&path),
                        },
                    );
                    Reply::handle(handle)
                }
                _ => Reply::err(0x0802A03C),
            }
        }
        Object::Storage { base, len } => match method {
            // IStorage::GetSize
            1 => Reply::ok(len.to_le_bytes().to_vec()),
            // IStorage::Read(in offset u64; out receive buffer; out read size)
            0 => {
                let offset = read_inline_u64(hub.mem.as_ref(), args_addr).unwrap_or(0);
                let Some((address, size)) = request.receive_list.first() else {
                    return Reply::err(0xE001C001);
                };
                let want = u64::from(*size).min(len.saturating_sub(offset));
                let Some(bytes) = hub.mem.read_bytes(*base + offset, want as usize) else {
                    return Reply::err(0xE103C006);
                };
                if !hub.mem.write_bytes(*address, &bytes) {
                    return Reply::err(0xE103C006);
                }
                let mut data = (bytes.len() as u32).to_le_bytes().to_vec();
                data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                Reply::with_receive(data, vec![bytes.len() as u64])
            }
            _ => Reply::err(0x0802A03C),
        },
        Object::HostStorage { image } => match method {
            // IStorage::GetSize
            1 => match host_image_size(image) {
                Some(size) => Reply::ok(Vec::from(size.to_le_bytes())),
                None => Reply::err(0x260902),
            },
            // IStorage::Read(in offset u64; out receive buffer; out read size)
            0 => {
                let offset = read_inline_u64(hub.mem.as_ref(), args_addr).unwrap_or(0);
                let Some((address, size)) = request.receive_list.first() else {
                    return Reply::err(0xE001C001);
                };
                let Some(total) = host_image_size(image) else {
                    return Reply::err(0x260902);
                };
                let want = u64::from(*size).min(total.saturating_sub(offset));
                let Some(bytes) = host_read_at(image, offset, want as usize) else {
                    return Reply::err(0xE103C006);
                };
                if !hub.mem.write_bytes(*address, &bytes) {
                    return Reply::err(0xE103C006);
                }
                let mut data = (bytes.len() as u32).to_le_bytes().to_vec();
                data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                Reply::with_receive(data, vec![bytes.len() as u64])
            }
            // IStorage::OperateRange (stub success)
            4 => Reply::ok(Vec::new()),
            _ => Reply::err(0x0802A03C),
        },
        Object::HostFileSystem { image } => {
            // Resolve metadata and copy the needed values before touching
            // handle state, so the borrow of `state` ends early.
            enum HostFsAction {
                File(u64, u64),
                Dir(Vec<DirectoryEntry>),
            }
            let action = {
                let Some(meta) = ensure_host_meta(hub, state) else {
                    return Reply::err(0x260902);
                };
                match method {
                    // IFileSystem::OpenFile(path) -> IFile over the image region
                    1 => {
                        let Some(path) = request_path(hub, request) else {
                            return Reply::err(0x0802A01E);
                        };
                        match meta.lookup(&path) {
                            Some(entry) => Some(HostFsAction::File(
                                meta.data_offset + entry.offset,
                                entry.size,
                            )),
                            None => None,
                        }
                    }
                    // IFileSystem::OpenDirectory(path) -> directory object
                    2 => {
                        let Some(path) = request_path(hub, request) else {
                            return Reply::err(0x0802A01E);
                        };
                        Some(HostFsAction::Dir(meta.list(&path)))
                    }
                    _ => return Reply::err(0x0802A03C),
                }
            };
            match action {
                Some(HostFsAction::File(offset, len)) => {
                    state.next_handle += 1;
                    let handle = state.next_handle;
                    state.objects.insert(
                        handle,
                        Object::HostFile {
                            image: image.clone(),
                            offset,
                            len,
                        },
                    );
                    Reply::handle(handle)
                }
                Some(HostFsAction::Dir(entries)) => {
                    state.next_handle += 1;
                    let handle = state.next_handle;
                    state.objects.insert(handle, Object::Directory { entries });
                    Reply::handle(handle)
                }
                None => Reply::err(0x02018C01), // path not found
            }
        }
        Object::HostFile { image, offset, len } => match method {
            // IFile::GetSize
            1 => Reply::ok(Vec::from(len.to_le_bytes())),
            // IFile::Read(in offset u64; out receive buffer; out read size)
            0 => {
                let req_off = read_inline_u64(hub.mem.as_ref(), args_addr).unwrap_or(0);
                let Some((address, size)) = request.receive_list.first() else {
                    return Reply::err(0xE001C001);
                };
                let want = u64::from(*size).min(len.saturating_sub(req_off));
                let Some(bytes) = host_read_at(image, offset + req_off, want as usize) else {
                    return Reply::err(0xE103C006);
                };
                if !hub.mem.write_bytes(*address, &bytes) {
                    return Reply::err(0xE103C006);
                }
                let mut data = (bytes.len() as u32).to_le_bytes().to_vec();
                data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                Reply::with_receive(data, vec![bytes.len() as u64])
            }
            _ => Reply::err(0x0802A03C),
        },
        Object::Directory { entries } => match method {
            // IDirectory::GetEntryCount
            1 => Reply::ok((entries.len() as u32).to_le_bytes().to_vec()),
            // IDirectory::Read(entries into receive buffer; out count)
            0 => {
                let Some((address, size)) = request.receive_list.first() else {
                    return Reply::err(0xE001C001);
                };
                let fits = (u64::from(*size) as usize / DIRECTORY_ENTRY_SIZE).min(entries.len());
                let mut blob = Vec::with_capacity(fits * DIRECTORY_ENTRY_SIZE);
                for entry in &entries[..fits] {
                    blob.extend_from_slice(&entry.encode());
                }
                if !hub.mem.write_bytes(*address, &blob) {
                    return Reply::err(0xE103C006);
                }
                Reply::with_receive(
                    (fits as u32).to_le_bytes().to_vec(),
                    vec![blob.len() as u64],
                )
            }
            _ => Reply::err(0x0802A03C),
        },
    }
}

/// In-memory RomFS reader over a Nintendo RomFS image. Metadata is parsed
/// eagerly with `parse_romfs_tables`.
pub struct RomFs<'a> {
    _image: &'a [u8],
    dirs: Vec<RomDirEntry>,
    files: Vec<RomFileEntry>,
    /// Absolute offset where file data starts (the header size field).
    data_base: u64,
}

#[derive(Debug)]
pub struct RomDirEntry {
    pub parent: u32,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct RomFileEntry {
    pub dir: u32,
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

impl<'a> RomFs<'a> {
    pub fn parse(image: &'a [u8]) -> Option<Self> {
        let header_size =
            usize::try_from(u64::from_le_bytes(image.get(0..8)?.try_into().ok()?)).ok()?;
        if header_size < 0x40 || image.len() < header_size {
            return None;
        }
        let header = &image[..header_size];
        let rd = |off: usize| -> Option<usize> {
            usize::try_from(u64::from_le_bytes(
                header.get(off..off + 8)?.try_into().ok()?,
            ))
            .ok()
        };
        let dir_table_off = rd(0x18)?;
        let dir_table_size = rd(0x20)?;
        let file_table_off = rd(0x38)?;
        let file_table_size = rd(0x40)?;
        if dir_table_off + dir_table_size > image.len()
            || file_table_off + file_table_size > image.len()
        {
            return None;
        }
        let (dirs, files, data_base) = parse_romfs_tables(
            header,
            &image[dir_table_off..dir_table_off + dir_table_size],
            &image[file_table_off..file_table_off + file_table_size],
        )?;
        Some(Self {
            _image: image,
            dirs,
            files,
            data_base,
        })
    }

    /// Directory table entries in insertion order (root is index 0).
    pub fn directories(&self) -> &[RomDirEntry] {
        &self.dirs
    }

    pub fn files(&self) -> &[RomFileEntry] {
        &self.files
    }

    /// Resolve a `/`-separated path to a file entry.
    pub fn lookup(&self, path: &str) -> Option<RomFileEntry> {
        let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();
        let dirs = &self.dirs;
        let mut dir_index: u32 = 0; // root directory is index 0
        for (i, seg) in segments.iter().enumerate() {
            if seg.is_empty() {
                continue;
            }
            if i + 1 == segments.len() {
                return self
                    .files
                    .iter()
                    .find(|f| f.dir == dir_index && f.name == *seg)
                    .cloned();
            }
            dir_index = dirs
                .iter()
                .enumerate()
                .find(|(_, d)| d.parent == dir_index && d.name == *seg)
                .map(|(index, _)| index as u32)?;
        }
        None
    }

    /// Directory listing for a path (root "/" lists top-level entries).
    pub fn list(&self, path: &str) -> Vec<DirectoryEntry> {
        let dirs = &self.dirs;
        let mut dir_index: u32 = 0;
        for seg in path.trim_start_matches('/').split('/') {
            if seg.is_empty() {
                continue;
            }
            if let Some((index, _)) = dirs
                .iter()
                .enumerate()
                .find(|(_, d)| d.parent == dir_index && d.name == seg)
            {
                dir_index = index as u32;
            }
        }
        let mut out = Vec::new();
        for (index, dir) in dirs.iter().enumerate() {
            // Skip the root directory entry itself.
            if dir.parent == dir_index && !(dir_index == 0 && index == 0) {
                out.push(DirectoryEntry {
                    path: dir.name.clone(),
                    size: 0,
                    directory: true,
                });
            }
        }
        for file in &self.files {
            if file.dir == dir_index {
                out.push(DirectoryEntry {
                    path: file.name.clone(),
                    size: file.size,
                    directory: false,
                });
            }
        }
        out
    }
}

/// Build a RomFS image for tests/samples in the real Nintendo layout:
/// header (0x50 bytes), data region, directory hash table (empty), directory
/// table, file hash table (empty), file table. File data offsets are relative
/// to the header size.
pub fn build_romfs_image(files: &[(&str, &[u8])]) -> Vec<u8> {
    let header_size = 0x50usize;
    let mut dirs: Vec<(u32, String)> = vec![(0, String::new())];
    let mut data = Vec::new();
    let mut file_specs: Vec<(usize, u64, u64, String)> = Vec::new();

    for (path, bytes) in files {
        let segments: Vec<&str> = path.split('/').collect();
        let mut parent = 0u32;
        for seg in &segments[..segments.len() - 1] {
            match dirs
                .iter()
                .enumerate()
                .find(|(_, (p, name))| *p == parent && name == seg)
            {
                Some((index, _)) => parent = index as u32,
                None => {
                    dirs.push((parent, seg.to_string()));
                    parent = (dirs.len() - 1) as u32;
                }
            }
        }
        let name = segments.last().copied().unwrap_or("").to_string();
        let offset = data.len() as u64;
        data.extend_from_slice(bytes);
        file_specs.push((parent as usize, offset, bytes.len() as u64, name));
    }

    // Directory table: 24-byte headers, inline padded names. Track the byte
    // offset of each directory so file entries can reference it.
    let mut dir_table = Vec::new();
    let mut dir_offsets: Vec<usize> = Vec::new();
    for (parent, name) in &dirs {
        dir_offsets.push(dir_table.len());
        dir_table.extend_from_slice(&parent.to_le_bytes());
        dir_table.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes()); // next sibling
        dir_table.extend_from_slice(&0u32.to_le_bytes());
        dir_table.extend_from_slice(&0u32.to_le_bytes());
        dir_table.extend_from_slice(&0u32.to_le_bytes());
        dir_table.extend_from_slice(&(name.len() as u32).to_le_bytes());
        dir_table.extend_from_slice(name.as_bytes());
        while !dir_table.len().is_multiple_of(4) {
            dir_table.push(0);
        }
    }

    // File table: 32-byte headers, inline padded names.
    let mut file_table = Vec::new();
    for (dir_index, offset, size, name) in &file_specs {
        file_table.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes()); // next sibling
        file_table.extend_from_slice(&(dir_offsets[*dir_index] as u32).to_le_bytes());
        file_table.extend_from_slice(&offset.to_le_bytes());
        file_table.extend_from_slice(&size.to_le_bytes());
        file_table.extend_from_slice(&0u32.to_le_bytes());
        file_table.extend_from_slice(&(name.len() as u32).to_le_bytes());
        file_table.extend_from_slice(name.as_bytes());
        while !file_table.len().is_multiple_of(4) {
            file_table.push(0);
        }
    }

    // Layout: header | data | dir hash | dir table | file hash | file table.
    let dir_hash_off = header_size + data.len();
    let dir_table_off = dir_hash_off + 8;
    let file_hash_off = dir_table_off + dir_table.len();
    let file_table_off = file_hash_off + 8;

    let mut image = Vec::new();
    image.extend_from_slice(&(header_size as u64).to_le_bytes());
    image.extend_from_slice(&(dir_hash_off as u64).to_le_bytes());
    image.extend_from_slice(&8u64.to_le_bytes()); // empty hash buckets
    image.extend_from_slice(&(dir_table_off as u64).to_le_bytes());
    image.extend_from_slice(&(dir_table.len() as u64).to_le_bytes());
    image.extend_from_slice(&(file_hash_off as u64).to_le_bytes());
    image.extend_from_slice(&8u64.to_le_bytes()); // empty hash buckets
    image.extend_from_slice(&(file_table_off as u64).to_le_bytes());
    image.extend_from_slice(&(file_table.len() as u64).to_le_bytes());
    image.extend_from_slice(&0u64.to_le_bytes());
    image.extend_from_slice(&data);
    image.extend_from_slice(&[0u8; 8]); // empty dir hash table
    image.extend_from_slice(&dir_table);
    image.extend_from_slice(&[0u8; 8]); // empty file hash table
    image.extend_from_slice(&file_table);
    image
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default, Debug)]
    struct TestMemory {
        bytes: Mutex<Vec<u8>>,
    }

    impl MemoryAccess for TestMemory {
        fn load_u8(&self, addr: u64) -> Option<u8> {
            let guard = self.bytes.lock().unwrap();
            guard.get(addr as usize).copied()
        }
        fn load_u32(&self, addr: u64) -> Option<u32> {
            let guard = self.bytes.lock().unwrap();
            let raw: [u8; 4] = guard
                .get(addr as usize..addr as usize + 4)?
                .try_into()
                .ok()?;
            Some(u32::from_le_bytes(raw))
        }
        fn load_u64(&self, addr: u64) -> Option<u64> {
            let guard = self.bytes.lock().unwrap();
            let raw: [u8; 8] = guard
                .get(addr as usize..addr as usize + 8)?
                .try_into()
                .ok()?;
            Some(u64::from_le_bytes(raw))
        }
        fn store_u32(&self, addr: u64, value: u32) -> bool {
            let mut guard = self.bytes.lock().unwrap();
            if addr as usize + 4 > guard.len() {
                return false;
            }
            guard[addr as usize..addr as usize + 4].copy_from_slice(&value.to_le_bytes());
            true
        }
        fn store_u64(&self, addr: u64, value: u64) -> bool {
            let mut guard = self.bytes.lock().unwrap();
            if addr as usize + 8 > guard.len() {
                return false;
            }
            guard[addr as usize..addr as usize + 8].copy_from_slice(&value.to_le_bytes());
            true
        }
        fn read_bytes(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
            let guard = self.bytes.lock().unwrap();
            Some(guard.get(addr as usize..addr as usize + len)?.to_vec())
        }
        fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
            let mut guard = self.bytes.lock().unwrap();
            if addr as usize + bytes.len() > guard.len() {
                return false;
            }
            guard[addr as usize..addr as usize + bytes.len()].copy_from_slice(bytes);
            true
        }
    }

    impl TestMemory {
        fn store(&self, addr: u64, value: u32) {
            let mut guard = self.bytes.lock().unwrap();
            guard[addr as usize..addr as usize + 4].copy_from_slice(&value.to_le_bytes());
        }
        fn store64(&self, addr: u64, value: u64) {
            let mut guard = self.bytes.lock().unwrap();
            guard[addr as usize..addr as usize + 8].copy_from_slice(&value.to_le_bytes());
        }
        fn put(&self, addr: u64, bytes: &[u8]) {
            let mut guard = self.bytes.lock().unwrap();
            guard[addr as usize..addr as usize + bytes.len()].copy_from_slice(bytes);
        }
        fn load(&self, addr: u64) -> u32 {
            let guard = self.bytes.lock().unwrap();
            u32::from_le_bytes(guard[addr as usize..addr as usize + 4].try_into().unwrap())
        }
        fn load64(&self, addr: u64) -> u64 {
            let guard = self.bytes.lock().unwrap();
            u64::from_le_bytes(guard[addr as usize..addr as usize + 8].try_into().unwrap())
        }
        fn slice(&self, addr: u64, len: usize) -> Vec<u8> {
            let guard = self.bytes.lock().unwrap();
            guard[addr as usize..addr as usize + len].to_vec()
        }
    }

    const RL_ADDR: u64 = 80;

    /// Build an SDK-style request: 16-byte message header, HIPC header at
    /// +16, optional pointer descriptor at +24 (buffer-relative offset), CMIF
    /// payload after the descriptors, and a receive list at buffer + `RL_ADDR`.
    fn craft_request(
        mem: &Arc<TestMemory>,
        buf: u64,
        method: u32,
        pointer: Option<(u64, usize)>,
        receive: Option<(u64, u32)>,
    ) {
        mem.store(buf, 6); // request message type
        mem.store(buf + 4, 0xC); // options
        mem.store64(buf + 8, 0); // reserved/pid
        let ptr_count = pointer.map(|_| 1u32).unwrap_or(0);
        mem.store(buf + 16, HIPC_MAGIC | (ptr_count << 16));
        let rl_count = receive.map(|_| 1u32).unwrap_or(0);
        let raw = buf + 24 + u64::from(ptr_count) * 8;
        let rl_word = ((buf + RL_ADDR) - raw) / 4;
        mem.store(buf + 20, rl_count << 10 | ((rl_word as u32) << 20));
        if let Some((offset, size)) = pointer {
            mem.store(buf + 24, (offset - buf) as u32);
            mem.store(buf + 28, size as u32);
        }
        mem.store(raw, SFCI_MAGIC);
        mem.store(raw + 4, 1); // version
        mem.store(raw + 8, method);
        if let Some((addr, size)) = receive {
            mem.store(buf + RL_ADDR, (addr - buf) as u32);
            mem.store(buf + RL_ADDR + 4, size);
        }
    }

    fn reply_handle(mem: &Arc<TestMemory>, buf: u64) -> u64 {
        assert_eq!(
            mem.load(buf),
            IPC_MESSAGE_TYPE_RESPONSE,
            "reply message type"
        );
        assert_eq!(mem.load(buf + 24), 0, "inline handle descriptor offset");
        u64::from(mem.load(buf + 28))
    }

    fn reply_status(mem: &Arc<TestMemory>, buf: u64, handles: u64, data_words: u32) -> u32 {
        let pos = buf + 24 + handles * 8;
        let cmif = (pos + u64::from(data_words) * 4 + 31) & !15;
        assert_eq!(mem.load(cmif), SFCO_MAGIC, "reply CMIF magic");
        assert_eq!(mem.load(cmif + 4), 1, "reply CMIF version");
        mem.load(cmif + 8)
    }

    fn setup() -> (Arc<TestMemory>, IpcHub, u64, u64, Vec<u8>) {
        let romfs = build_romfs_image(&[
            ("dir1/file.txt", b"hello romfs"),
            ("data.bin", b"0123456789"),
        ]);
        let mem = Arc::new(TestMemory {
            bytes: Mutex::new(vec![0u8; 0x10000]),
        });
        let romfs_base = 0x2000u64;
        mem.put(romfs_base, &romfs);
        let buf = 0x1000u64;
        let port = 0x1800u64;
        mem.put(port, b"fs:GLOBAL\0");
        let hub = IpcHub::new(mem.clone(), romfs_base, romfs.len() as u64, None);
        (mem, hub, buf, port, romfs)
    }

    #[test]
    fn connect_fs_global_returns_handle() {
        let (_mem, hub, _buf, port, _romfs) = setup();
        let (code, handle) = hub.connect_to_named_port(port);
        assert_eq!(code, IPC_RESULT_OK);
        assert_eq!(handle, 1);
    }

    #[test]
    fn storage_get_size_and_read() {
        let (mem, hub, buf, _port, romfs) = setup();
        craft_request(&mem, buf, 200, None, None);
        assert_eq!(hub.send_sync_request(1, buf), IPC_RESULT_OK);
        assert_eq!(reply_status(&mem, buf, 1, 0), 0);
        let storage = reply_handle(&mem, buf);
        assert!(storage >= 2);

        // IStorage::GetSize -> u64 raw data (2 words)
        craft_request(&mem, buf, 1, None, None);
        assert_eq!(hub.send_sync_request(storage, buf), IPC_RESULT_OK);
        assert_eq!(reply_status(&mem, buf, 0, 2), 0);
        assert_eq!(mem.load64(buf + 24), romfs.len() as u64);

        // IStorage::Read: offset arg at raw+12, inline receive buffer.
        craft_request(&mem, buf, 0, None, Some((buf + 0x200, 5)));
        mem.store64(buf + 24 + 12, 6); // read offset (CMIF params)
        assert_eq!(hub.send_sync_request(storage, buf), IPC_RESULT_OK);
        assert_eq!(reply_status(&mem, buf, 0, 3), 0);
        assert_eq!(mem.slice(buf + 0x200, 5), &romfs[6..11]);
        assert_eq!(mem.load(buf + 24), 5); // read size
        assert_eq!(mem.load(buf + RL_ADDR + 4), 5); // receive-list entry rewritten
    }

    #[test]
    fn filesystem_open_file_reads_content() {
        let (mem, hub, buf, _port, _romfs) = setup();
        craft_request(&mem, buf, 2, None, None); // OpenDataFileSystemByCurrentProcess
        assert_eq!(hub.send_sync_request(1, buf), IPC_RESULT_OK);
        let fs_handle = reply_handle(&mem, buf);
        assert!(fs_handle >= 2);

        // IFileSystem::OpenFile("/data.bin") via pointer descriptor.
        craft_request(&mem, buf, 1, Some((buf + 0x100, 10)), None);
        mem.put(buf + 0x100, b"/data.bin\0");
        assert_eq!(hub.send_sync_request(fs_handle, buf), IPC_RESULT_OK);
        let file = reply_handle(&mem, buf);

        // IFile::GetSize
        craft_request(&mem, buf, 1, None, None);
        assert_eq!(hub.send_sync_request(file, buf), IPC_RESULT_OK);
        assert_eq!(mem.load64(buf + 24), 10);
    }

    #[test]
    fn filesystem_open_directory_lists_entries() {
        let (mem, hub, buf, _port, _romfs) = setup();
        craft_request(&mem, buf, 2, None, None);
        assert_eq!(hub.send_sync_request(1, buf), IPC_RESULT_OK);
        let fs_handle = reply_handle(&mem, buf);

        // IFileSystem::OpenDirectory("/") via pointer descriptor.
        craft_request(&mem, buf, 2, Some((buf + 0x100, 2)), None);
        mem.put(buf + 0x100, b"/\0");
        assert_eq!(hub.send_sync_request(fs_handle, buf), IPC_RESULT_OK);
        let dir = reply_handle(&mem, buf);

        // IDirectory::GetEntryCount
        craft_request(&mem, buf, 1, None, None);
        assert_eq!(hub.send_sync_request(dir, buf), IPC_RESULT_OK);
        assert_eq!(mem.load(buf + 24), 2);

        // IDirectory::Read into an inline receive buffer
        craft_request(
            &mem,
            buf,
            0,
            None,
            Some((buf + 0x200, (2 * DIRECTORY_ENTRY_SIZE) as u32)),
        );
        assert_eq!(hub.send_sync_request(dir, buf), IPC_RESULT_OK);
        assert_eq!(mem.load(buf + 24), 2);
        let first = mem.slice(buf + 0x200, DIRECTORY_ENTRY_SIZE);
        assert_eq!(&first[..8], b"dir1\0\0\0\0");
        assert_eq!(u32::from_le_bytes(first[264..268].try_into().unwrap()), 1);
        let second = mem.slice(
            buf + 0x200 + DIRECTORY_ENTRY_SIZE as u64,
            DIRECTORY_ENTRY_SIZE,
        );
        assert_eq!(&second[..8], b"data.bin");
        assert_eq!(u64::from_le_bytes(second[256..264].try_into().unwrap()), 10);
        assert_eq!(
            mem.load(buf + RL_ADDR + 4),
            (2 * DIRECTORY_ENTRY_SIZE) as u32
        );
    }

    #[test]
    fn host_image_storage_and_file() {
        let romfs = build_romfs_image(&[
            ("dir1/file.txt", b"hello romfs"),
            ("data.bin", b"0123456789"),
        ]);
        let image_path = std::env::temp_dir().join("recomp-ipc-host-romfs.bin");
        std::fs::write(&image_path, &romfs).expect("write host image");

        let mem = Arc::new(TestMemory {
            bytes: Mutex::new(vec![0u8; 0x10000]),
        });
        let buf = 0x1000u64;
        let port = 0x1800u64;
        mem.put(port, b"fs:GLOBAL\0");
        let hub = IpcHub::new(mem.clone(), 0, 0, Some(image_path.clone()));

        // OpenDataStorageByCurrentProcess -> HostStorage: GetSize = image size
        craft_request(&mem, buf, 200, None, None);
        assert_eq!(hub.connect_to_named_port(port).1, 1);
        assert_eq!(hub.send_sync_request(1, buf), IPC_RESULT_OK);
        let storage = reply_handle(&mem, buf);
        craft_request(&mem, buf, 1, None, None);
        assert_eq!(hub.send_sync_request(storage, buf), IPC_RESULT_OK);
        assert_eq!(mem.load64(buf + 24), romfs.len() as u64);

        // OpenDataFileSystemByCurrentProcess -> HostFileSystem -> HostFile
        craft_request(&mem, buf, 2, None, None);
        assert_eq!(hub.send_sync_request(1, buf), IPC_RESULT_OK);
        let fs_handle = reply_handle(&mem, buf);
        craft_request(&mem, buf, 1, Some((buf + 0x100, 10)), None);
        mem.put(buf + 0x100, b"/data.bin\0");
        assert_eq!(hub.send_sync_request(fs_handle, buf), IPC_RESULT_OK);
        let file = reply_handle(&mem, buf);

        craft_request(&mem, buf, 1, None, None);
        assert_eq!(hub.send_sync_request(file, buf), IPC_RESULT_OK);
        assert_eq!(mem.load64(buf + 24), 10);

        craft_request(&mem, buf, 0, None, Some((buf + 0x200, 10)));
        mem.store64(buf + 24 + 12, 0);
        assert_eq!(hub.send_sync_request(file, buf), IPC_RESULT_OK);
        assert_eq!(mem.slice(buf + 0x200, 10), b"0123456789");
        assert_eq!(mem.load(buf + RL_ADDR + 4), 10);

        // Directory listing through the host metadata
        craft_request(&mem, buf, 2, Some((buf + 0x100, 2)), None);
        mem.put(buf + 0x100, b"/\0");
        assert_eq!(hub.send_sync_request(fs_handle, buf), IPC_RESULT_OK);
        let dir = reply_handle(&mem, buf);
        craft_request(&mem, buf, 1, None, None);
        assert_eq!(hub.send_sync_request(dir, buf), IPC_RESULT_OK);
        assert_eq!(mem.load(buf + 24), 2);

        std::fs::remove_file(&image_path).ok();
    }

    #[test]
    fn romfs_lookup_and_list() {
        let romfs = build_romfs_image(&[("dir1/file.txt", b"hello"), ("data.bin", b"xyz")]);
        let fs = RomFs::parse(&romfs).expect("romfs parse");
        let entry = fs.lookup("/dir1/file.txt").expect("file");
        assert_eq!(entry.size, 5);
        assert_eq!(
            &romfs[fs.data_base as usize + entry.offset as usize..][..5],
            b"hello"
        );
        let root = fs.list("/");
        assert!(root.iter().any(|e| e.path == "dir1" && e.directory));
        assert!(root.iter().any(|e| e.path == "data.bin" && !e.directory));
        let inner = fs.list("/dir1");
        assert_eq!(inner.len(), 1);
        assert_eq!(inner[0].path, "file.txt");
    }

    #[test]
    fn close_handle_removes_object() {
        let (mem, hub, buf, _port, _romfs) = setup();
        craft_request(&mem, buf, 200, None, None);
        assert_eq!(hub.send_sync_request(1, buf), IPC_RESULT_OK);
        let storage = reply_handle(&mem, buf);
        assert_eq!(hub.close_handle(storage), IPC_RESULT_OK);
        craft_request(&mem, buf, 1, None, None);
        assert_eq!(
            hub.send_sync_request(storage, buf),
            IPC_RESULT_INVALID_HANDLE
        );
    }

    #[test]
    fn save_dir_env_overrides_the_romfs_sibling() {
        let override_dir = std::env::temp_dir().join("recomp-save-dir-env-test");
        std::env::set_var("RECOMP_SAVE_DIR", &override_dir);
        assert_eq!(
            save_root(Some(Path::new("/games/romfs.bin"))),
            Some(override_dir)
        );
        // An empty value is not a directory: the default stays.
        std::env::set_var("RECOMP_SAVE_DIR", "");
        assert_eq!(
            save_root(Some(Path::new("/games/romfs.bin"))),
            Some(PathBuf::from("/games/save"))
        );
        std::env::remove_var("RECOMP_SAVE_DIR");
        assert_eq!(save_root(None), None);
    }
}
