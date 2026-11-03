//! The fallback memory access generated code uses when guest memory is not
//! one host reservation. On aarch64 and other non-x86 hosts, 16-byte vector
//! loads and stores go through `u128` (x86 keeps them in `__m128`); both must
//! be the same little-endian, unaligned bytes the guest sees.
//!
//! The direct path below covers the part that needs no runtime: addresses
//! below the reservation limit. The runtime usually defines these symbols;
//! the stubs make the slow, out-of-reservation branch linkable but panic if
//! this test ever reaches it.

use recomp_abi::{memory, set_memory_base, Fault, GuestResult};

#[no_mangle]
extern "Rust" fn __recomp_rt_mem_load_u64(address: u64) -> Result<u64, Fault> {
    panic!("unexpected slow load at {address:#x}")
}

#[no_mangle]
extern "Rust" fn __recomp_rt_mem_store_u64(address: u64, _value: u64) -> GuestResult {
    panic!("unexpected slow store at {address:#x}")
}

#[test]
fn vector_loads_and_stores_keep_the_guests_bytes_on_every_host() {
    let mut buffer = vec![0u8; 64];
    set_memory_base(buffer.as_mut_ptr() as usize);
    let memory = memory();

    let value = 0x0011_2233_4455_6677_8899_aabb_ccdd_eeffu128;
    // Every unaligned offset: a vector access is not required to be aligned.
    for offset in 0..16usize {
        memory
            .store_u128(offset as u64, value)
            .expect("direct store");
        assert_eq!(memory.load_u128(offset as u64).expect("direct load"), value);
        let bytes: [u8; 16] = buffer[offset..offset + 16].try_into().expect("16 bytes");
        assert_eq!(bytes, value.to_le_bytes(), "offset {offset}");
    }

    // A wider value must not disturb the neighboring bytes.
    buffer.fill(0xAA);
    memory.store_u128(16, value).expect("direct aligned store");
    assert_eq!(&buffer[..16], &[0xAA; 16]);
    assert_eq!(&buffer[16..32], &value.to_le_bytes());
    assert_eq!(&buffer[32..], &[0xAA; 32]);

    set_memory_base(0);
}
