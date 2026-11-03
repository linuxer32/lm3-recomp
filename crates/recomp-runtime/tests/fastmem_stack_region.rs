//! Guest memory a bounded fastmem window does not reach. The SDK aliases
//! every thread's stack into the stack region the kernel reports (svcGetInfo
//! 14 and 15), by default 2 GiB at 128 GiB. A host that can reserve only a
//! window (a phone's 39-bit address space) reports it right after the memory
//! layout instead, inside the window, so stack accesses stay direct; and a
//! region mapped above the window anyway stays on the checked path without
//! taking direct access away from the rest of guest memory.

use recomp_runtime::{
    init_kernel_services, init_memory, mem_load_u32, mem_store_u32, svc_dispatch_id,
    KernelSvcDispatcher, MemoryLayout, MemoryPermissions, MemoryRegionSpec,
};

const GIB: u64 = 1 << 30;

/// A layout that ends at 8 GiB (LM3's heap does) in a 10 GiB window, set up
/// once for the tests in this process.
fn setup() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::env::set_var("RECOMP_FASTMEM_GIB", "10");
        let layout = MemoryLayout::new(vec![
            MemoryRegionSpec::new("data", 0x1000, 0x2000, MemoryPermissions::read_write()),
            MemoryRegionSpec::new("heap", 4 * GIB, 4 * GIB, MemoryPermissions::read_write()),
        ]);
        init_memory(layout).expect("bounded window");
        init_kernel_services(KernelSvcDispatcher::all_allowed(), 4 * GIB, 4 * GIB);
    });
}

#[test]
fn a_bounded_window_holds_the_stack_region() {
    setup();
    assert_eq!(recomp_abi::memory_limit(), 10 * GIB);
    let info = |kind: i64| {
        svc_dispatch_id(0x29, &[0, kind, 0, 0])
            .expect("svcGetInfo")
            .x1
            .expect("value in x1") as u64
    };
    let (base, size) = (info(14), info(15));
    assert_eq!(base, 8 * GIB, "the stack region follows the layout");
    assert_eq!(size, 2 * GIB);
    assert!(
        base + size <= recomp_abi::memory_limit(),
        "the whole stack region is direct"
    );
}

#[test]
fn a_region_above_the_window_keeps_direct_access_on() {
    setup();
    let base = recomp_abi::memory_base();
    assert_ne!(base, 0);
    mem_store_u32(4 * GIB, 0x1234_5678).expect("heap store");

    // svcMapMemory(dst, src, size) to 128 GiB, far above the window: the
    // alias lives in a buffer of its own, on the checked path.
    let alias = 0x20_0000_0000_u64;
    let outcome =
        svc_dispatch_id(0x04, &[alias as i64, (4 * GIB) as i64, 0x1000]).expect("svcMapMemory");
    assert_eq!(outcome.x0, 0, "mapped");
    assert_eq!(mem_load_u32(alias).expect("alias load"), 0x1234_5678);
    // The rest of guest memory is still direct.
    assert_eq!(recomp_abi::memory_base(), base, "direct access stays on");
    assert_eq!(recomp_abi::memory().load_u32(0x1000).ok(), Some(0));
}
