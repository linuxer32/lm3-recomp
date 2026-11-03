//! Guest threads through the SVC layer: create, start, exit, wait.

use recomp_runtime::{
    boot_main_thread, cpu_get, init_kernel_services, init_memory, mem_load_u64, mem_store_u32,
    mem_store_u64, register_dispatcher, svc_dispatch, tls_set, KernelSvcDispatcher, MemoryLayout,
    MemoryPermissions, MemoryRegionSpec,
};

const DATA: u64 = 0x1000_0000;
const TLS: u64 = 0x7000_0000;
const THREAD_ENTRY: u64 = 0x2000;

/// Stands in for lifted code: the thread entry stores its argument and its
/// handle, then exits.
fn guest(address: u64) -> recomp_abi::GuestResult {
    assert_eq!(address, THREAD_ENTRY);
    mem_store_u64(DATA, cpu_get(0) as u64)?;
    mem_store_u64(DATA + 8, cpu_get(1) as u64)?;
    svc_dispatch("svcExitThread", &[])?;
    unreachable!("svcExitThread returned");
}

#[test]
fn started_threads_run_and_signal_on_exit() {
    let rw = MemoryPermissions::new(true, true, false);
    init_memory(MemoryLayout::new(vec![
        MemoryRegionSpec::new("data", DATA, 0x1000, rw),
        MemoryRegionSpec::new("tls", TLS, 0x1000, rw),
    ]))
    .expect("memory");
    init_kernel_services(
        KernelSvcDispatcher::all_allowed(),
        0x1_0000_0000,
        0x1000_0000,
    );
    tls_set(TLS);
    boot_main_thread(0);
    register_dispatcher(guest);

    let created = svc_dispatch(
        "svcCreateThread",
        &[0, THREAD_ENTRY as i64, 0x1234, 0x7FFF_0000, 44, 0],
    )
    .expect("create");
    assert_eq!(created.x0, 0);
    let handle = created.x1.expect("thread handle");
    assert_eq!(
        svc_dispatch("svcStartThread", &[handle]).expect("start").x0,
        0
    );

    mem_store_u32(DATA + 0x100, handle as u64).expect("handle list");
    let waited =
        svc_dispatch("svcWaitSynchronization", &[0, (DATA + 0x100) as i64, 1, -1]).expect("wait");
    assert_eq!((waited.x0, waited.x1), (0, Some(0)));
    assert_eq!(mem_load_u64(DATA).expect("arg"), 0x1234);
    assert_eq!(mem_load_u64(DATA + 8).expect("handle"), handle as u64);
}
