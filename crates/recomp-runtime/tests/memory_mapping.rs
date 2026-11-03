//! svcMapMemory / svcMapSharedMemory create and remove regions at run time.

use recomp_runtime::{
    init_kernel_services, init_memory, mem_load_u32, mem_load_u8, mem_store_u32, svc_dispatch,
    KernelSvcDispatcher, MemoryLayout, MemoryPermissions, MemoryRegionSpec,
};
use recomp_services::kernel::{kernel, RESULT_INVALID_SIZE};

const HEAP: u64 = 0x1000_0000;
const STACK_ALIAS: u64 = 0x20_0000_0000;
const SHARED: u64 = 0x50_0000_0000;

#[derive(Debug)]
struct Guest;

impl recomp_services::MemoryAccess for Guest {
    fn load_u8(&self, addr: u64) -> Option<u8> {
        mem_load_u8(addr).ok().map(|value| value as u8)
    }
    fn load_u32(&self, addr: u64) -> Option<u32> {
        mem_load_u32(addr).ok().map(|value| value as u32)
    }
    fn load_u64(&self, _: u64) -> Option<u64> {
        None
    }
    fn store_u32(&self, addr: u64, value: u32) -> bool {
        mem_store_u32(addr, u64::from(value)).is_ok()
    }
    fn store_u64(&self, _: u64, _: u64) -> bool {
        false
    }
    fn read_bytes(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        (0..len as u64)
            .map(|offset| self.load_u8(addr + offset))
            .collect()
    }
    fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
        bytes.iter().enumerate().all(|(offset, byte)| {
            recomp_runtime::mem_store_u8(addr + offset as u64, u64::from(*byte)).is_ok()
        })
    }
}

fn state_at(address: u64) -> u32 {
    let info = HEAP + 0x8000;
    svc_dispatch("svcQueryMemory", &[info as i64, 0, address as i64]).expect("query");
    mem_load_u32(info + 16).expect("state") as u32
}

#[test]
fn stacks_alias_and_shared_memory_map_at_run_time() {
    let rw = MemoryPermissions::new(true, true, false);
    init_memory(MemoryLayout::new(vec![MemoryRegionSpec::new(
        "heap", HEAP, 0x10000, rw,
    )]))
    .expect("memory");
    init_kernel_services(KernelSvcDispatcher::all_allowed(), HEAP, 0x10000);

    // Free space ends with the 39-bit address space; above it is one
    // inaccessible block whose end wraps to zero.
    let info = HEAP + 0x8000;
    svc_dispatch("svcQueryMemory", &[info as i64, 0, STACK_ALIAS as i64]).expect("query");
    let (base, size) = (
        recomp_runtime::mem_load_u64(info).expect("base"),
        recomp_runtime::mem_load_u64(info + 8).expect("size"),
    );
    assert_eq!((base + size, state_at(STACK_ALIAS)), (1 << 39, 0));
    assert_eq!(state_at(1 << 40), 0x10);

    // MapMemory moves the bytes to the alias; UnmapMemory brings them back.
    mem_store_u32(HEAP + 0x10, 0x1234_5678).expect("seed");
    let map =
        svc_dispatch("svcMapMemory", &[STACK_ALIAS as i64, HEAP as i64, 0x1000]).expect("map");
    assert_eq!(map.x0, 0);
    assert_eq!(
        mem_load_u32(STACK_ALIAS + 0x10).expect("alias"),
        0x1234_5678
    );
    assert_eq!(
        state_at(STACK_ALIAS),
        0x0B,
        "aliased stacks report as Stack"
    );
    mem_store_u32(STACK_ALIAS + 0x20, 0xCAFE).expect("write through alias");
    let unmap =
        svc_dispatch("svcUnmapMemory", &[STACK_ALIAS as i64, HEAP as i64, 0x1000]).expect("unmap");
    assert_eq!(unmap.x0, 0);
    assert_eq!(mem_load_u32(HEAP + 0x20).expect("copied back"), 0xCAFE);
    assert!(mem_load_u32(STACK_ALIAS).is_err(), "alias is gone");

    // Shared memory starts with the service's bytes and follows its writes.
    let handle = kernel().create_shared_memory(vec![7; 0x1000]);
    let wrong = svc_dispatch(
        "svcMapSharedMemory",
        &[i64::from(handle), SHARED as i64, 0x2000, 3],
    )
    .expect("wrong size");
    assert_eq!(wrong.x0 as u64, RESULT_INVALID_SIZE);
    let shared = svc_dispatch(
        "svcMapSharedMemory",
        &[i64::from(handle), SHARED as i64, 0x1000, 3],
    )
    .expect("map shared");
    assert_eq!(shared.x0, 0);
    assert_eq!(mem_load_u8(SHARED + 0xFFF).expect("contents"), 7);
    assert_eq!(state_at(SHARED), 0x06);
    kernel().write_shared_memory(&Guest, handle, 4, &[9]);
    assert_eq!(mem_load_u8(SHARED + 4).expect("updated"), 9);
    svc_dispatch(
        "svcUnmapSharedMemory",
        &[i64::from(handle), SHARED as i64, 0x1000],
    )
    .expect("unmap shared");
    assert!(mem_load_u8(SHARED).is_err());
}
