//! A bounded fastmem window: what a host whose user address space cannot
//! hold all 512 GiB (Android's 39-bit spaces, and `RECOMP_FASTMEM_GIB` here)
//! ends up with. Generated code keeps the direct path below the window and
//! falls back to the checked path at or above it, so an access past the
//! reservation can never touch host memory.

use recomp_runtime::{
    init_memory, mem_load_u32, MemoryLayout, MemoryPermissions, MemoryRegionSpec,
};

#[test]
fn a_bounded_window_keeps_direct_access_below_it() {
    // One GiB, the smallest window the test hook allows; the layout fits.
    std::env::set_var("RECOMP_FASTMEM_GIB", "1");
    let layout = MemoryLayout::new(vec![
        MemoryRegionSpec::new("data", 0x1000, 0x2000, MemoryPermissions::read_write()),
        // The top of the window: still inside the reservation.
        MemoryRegionSpec::new("high", 0x3ff0_0000, 0x1000, MemoryPermissions::read_write()),
    ]);
    init_memory(layout).expect("bounded window");

    assert_ne!(recomp_abi::memory_base(), 0, "the window is published");
    assert_eq!(
        recomp_abi::memory_limit(),
        1 << 30,
        "the window's end is published, not the whole 39-bit space"
    );

    // Generated code's view: a direct store and load inside the window.
    let memory = recomp_abi::memory();
    assert!(memory.store_u32(0x1004, 0xDEAD_BEEF).is_ok());
    assert_eq!(memory.load_u32(0x1004).unwrap(), 0xDEAD_BEEF);
    // The runtime's checked path sees the same bytes.
    assert_eq!(mem_load_u32(0x1004).unwrap(), 0xDEAD_BEEF);

    // At and above the window's end the view takes the checked path; the
    // address is unmapped, which is a guest fault, not a host fault.
    assert!(memory.load_u32(1 << 30).is_err());
    assert!(memory.store_u32((1 << 30) + 0x40, 1).is_err());
}
