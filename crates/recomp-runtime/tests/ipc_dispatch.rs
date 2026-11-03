//! End-to-end IPC through the runtime SVC dispatch path: the guest writes a
//! HIPC/CMIF request into its memory image, calls svcConnectToNamedPort and
//! svcSendSyncRequest, and reads fs: replies back out of memory.

use recomp_runtime::{
    apply_memory_image, init_default_memory, init_ipc_services, init_kernel_services, mem_load_u32,
    mem_load_u64, mem_store_u32, mem_store_u64, mem_store_u8, svc_dispatch, KernelSvcDispatcher,
    MemoryInitSegment, MemoryLayout, MemoryPermissions, MemoryRegionSpec,
};
use recomp_services::ipc::build_romfs_image;

const HIPC_MAGIC: u32 = 0x4000;
const SFCI_MAGIC: u32 = 0x4943_4653;
const SFCO_MAGIC: u32 = 0x4F43_4653;
const BUFFER: u64 = 0x1000;
const ROMFS_BASE: u64 = 0x2000;

fn a(v: u64) -> i64 {
    v as i64
}

/// Write an SDK-style request: message header, HIPC header at +16, CMIF
/// payload at +24, optional inline receive-list entry at +64.
fn craft_request(method: u32, receive: Option<(u64, u32)>) {
    mem_store_u32(BUFFER, 6).expect("message type");
    mem_store_u32(BUFFER + 4, 0xC).expect("options");
    mem_store_u64(BUFFER + 8, 0).expect("reserved");
    mem_store_u32(BUFFER + 16, u64::from(HIPC_MAGIC)).expect("header0");
    let rl_count = receive.map(|_| 1u32).unwrap_or(0);
    let header1 = rl_count << 10 | (14 << 20); // receive list at BUFFER + 80
    mem_store_u32(BUFFER + 20, u64::from(header1)).expect("header1");
    mem_store_u32(BUFFER + 24, u64::from(SFCI_MAGIC)).expect("sfc magic");
    mem_store_u32(BUFFER + 28, 1).expect("version");
    mem_store_u32(BUFFER + 32, u64::from(method)).expect("command");
    if let Some((addr, size)) = receive {
        mem_store_u32(BUFFER + 80, u64::from((addr - BUFFER) as u32)).expect("rl offset");
        mem_store_u32(BUFFER + 84, u64::from(size)).expect("rl size");
    }
}

fn reply_handle() -> u64 {
    assert_eq!(mem_load_u32(BUFFER).expect("reply message type"), 4);
    assert_eq!(mem_load_u32(BUFFER + 24).expect("handle offset field"), 0);
    mem_load_u32(BUFFER + 28).expect("handle")
}

fn reply_status(handles: u64, data_words: u32) -> u64 {
    let pos = BUFFER + 24 + handles * 8;
    let cmif = (pos + u64::from(data_words) * 4 + 31) & !15;
    assert_eq!(
        mem_load_u32(cmif).expect("reply CMIF magic"),
        u64::from(SFCO_MAGIC)
    );
    assert_eq!(mem_load_u32(cmif + 4).expect("reply CMIF version"), 1);
    mem_load_u32(cmif + 8).expect("reply status")
}

#[test]
fn ipc_fs_round_trip_through_svc_dispatch() {
    let layout = MemoryLayout::new(vec![
        MemoryRegionSpec::new("ipc_buffer", BUFFER, 0x400, MemoryPermissions::read_write()),
        MemoryRegionSpec::new("romfs", ROMFS_BASE, 0x1000, MemoryPermissions::read_only()),
    ]);
    init_default_memory(layout).expect("memory init");

    let romfs = build_romfs_image(&[("data.bin", b"0123456789")]);
    let mut buffer = vec![0u8; 0x400];
    buffer[..10].copy_from_slice(b"fs:GLOBAL\0");
    apply_memory_image(
        &[
            MemoryInitSegment::new("ipc_buffer", BUFFER, buffer.len() as u64, buffer),
            MemoryInitSegment::new("romfs", ROMFS_BASE, romfs.len() as u64, romfs.clone()),
        ],
        &[],
    )
    .expect("image init");

    init_kernel_services(KernelSvcDispatcher::all_allowed(), 0x40000, 0x10000);
    // The IPC buffer is the thread-local region (TPIDRRO_EL0).
    init_ipc_services(ROMFS_BASE, romfs.len() as u64, None, Some(BUFFER)).expect("ipc init");

    // svcConnectToNamedPort(out_handle @ BUFFER+0x100, name @ BUFFER)
    // Kernel ABI: the handle lands in X1; the SDK wrapper stores it to *out.
    let connected =
        svc_dispatch("svcConnectToNamedPort", &[a(BUFFER + 0x100), a(BUFFER)]).expect("connect");
    let session = connected.x1.expect("session handle") as u64;
    mem_store_u64(BUFFER + 0x100, session).expect("store session handle");
    assert!(session >= 1);

    // OpenDataFileSystemByCurrentProcess (method 2)
    craft_request(2, None);
    svc_dispatch("svcSendSyncRequest", &[a(session)]).expect("open fs");
    assert_eq!(reply_status(1, 0), 0);
    let fs_handle = reply_handle();

    // IFileSystem::OpenFile("/data.bin") (inline string at CMIF args +12)
    craft_request(1, None);
    mem_store_u32(BUFFER + 36, 9).expect("path len");
    for (i, byte) in b"/data.bin".iter().enumerate() {
        mem_store_u8(BUFFER + 40 + i as u64, u64::from(*byte)).expect("path byte");
    }
    svc_dispatch("svcSendSyncRequest", &[a(fs_handle)]).expect("open file");
    let file_handle = reply_handle();

    // IFile::GetSize
    craft_request(1, None);
    svc_dispatch("svcSendSyncRequest", &[a(file_handle)]).expect("get size");
    assert_eq!(reply_status(0, 2), 0);
    assert_eq!(mem_load_u64(BUFFER + 24).expect("file size"), 10);

    // IFile::Read(offset=0) with a receive buffer inside the IPC region.
    craft_request(0, Some((BUFFER + 0x200, 10)));
    mem_store_u32(BUFFER + 36, 0).expect("read offset lo");
    mem_store_u32(BUFFER + 40, 0).expect("read offset hi");
    svc_dispatch("svcSendSyncRequest", &[a(file_handle)]).expect("read");
    assert_eq!(reply_status(0, 3), 0);
    assert_eq!(mem_load_u32(BUFFER + 24).expect("read size"), 10);
    assert_eq!(
        mem_load_u64(BUFFER + 0x200).expect("first 8 bytes"),
        0x3736353433323130
    );
    assert_eq!(mem_load_u32(BUFFER + 84).expect("entry size rewritten"), 10);

    // svcCloseHandle
    svc_dispatch("svcCloseHandle", &[a(file_handle)]).expect("close");
    // Kernel failures are reported to the guest in W0, as on hardware.
    let outcome = svc_dispatch("svcSendSyncRequest", &[a(file_handle)]).expect("dispatched");
    assert_eq!(
        outcome.x0 as u64,
        recomp_services::ipc::IPC_RESULT_INVALID_HANDLE
    );
}
