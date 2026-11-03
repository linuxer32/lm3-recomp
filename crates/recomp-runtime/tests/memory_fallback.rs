//! The checked fallback used when the full 39-bit reservation cannot be
//! created (Android's 39-bit user address space is the common case, and
//! `RECOMP_FASTMEM=0` reproduces it deterministically here).

use recomp_runtime::{
    init_memory, mem_load_u32, mem_store_u32, MemoryLayout, MemoryPermissions, MemoryRegionSpec,
};

#[test]
fn guest_memory_is_correct_without_a_fastmem_reservation() {
    // A failed 512 GiB mmap and this switch both leave `reservation()` as
    // `None`; every region is then checked heap memory and no direct base is
    // published, so generated code cannot use a stale or rebased address.
    std::env::set_var("RECOMP_FASTMEM", "0");

    let layout = MemoryLayout::new(vec![
        MemoryRegionSpec::new("data", 0x1000, 0x40, MemoryPermissions::read_write()),
        MemoryRegionSpec::new("ro", 0x2000, 0x40, MemoryPermissions::read_only()),
    ]);
    init_memory(layout).expect("checked memory");
    assert_eq!(
        recomp_abi::memory_base(),
        0,
        "no reservation means no direct base"
    );

    mem_store_u32(0x1004, 0xDEAD_BEEF).expect("store");
    assert_eq!(mem_load_u32(0x1004).expect("load"), 0xDEAD_BEEF);
    // Horizon allows unaligned normal-memory accesses; the checked path keeps
    // them working without a page mapping.
    mem_store_u32(0x1002, 0x1122_3344).expect("unaligned store");
    assert_eq!(mem_load_u32(0x1002).expect("unaligned load"), 0x1122_3344);
    assert!(
        mem_store_u32(0x2000, 1).is_err(),
        "read-only regions stay read-only"
    );
    assert!(mem_load_u32(0x3000).is_err(), "unmapped stays unmapped");

    // Re-initialization still starts from zeroed heap regions.
    init_memory(MemoryLayout::new(vec![MemoryRegionSpec::new(
        "data",
        0x1000,
        0x40,
        MemoryPermissions::read_write(),
    )]))
    .expect("re-init");
    assert_eq!(mem_load_u32(0x1004).expect("fresh load"), 0);
}
