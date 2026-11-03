//! Host implementations of guest functions (`[host_functions]` in a title's
//! `title.toml`). The generated function calls `__recomp_host_<name>` first
//! and runs its lifted code only when that returns false, so a host version
//! may leave rare cases (errors, unusual state) to the original.

use recomp_abi::Fault;

/// `NvOsGetTimeUS`: the host clock in microseconds. The guest version reads
/// the system counter and converts it through `TickManager`, and titles poll
/// it often enough that the conversion shows on their busiest thread.
#[no_mangle]
pub fn __recomp_host_nv_os_get_time_us() -> Result<bool, Fault> {
    let micros = recomp_services::clock::now_us();
    recomp_abi::cpu_set(0, micros as i64);
    Ok(true)
}

/// `memcpy`: copy guest bytes, or leave the lifted loop to do it when the
/// range is outside the fast-memory reservation.
#[no_mangle]
pub fn __recomp_host_memcpy() -> Result<bool, Fault> {
    let base = recomp_abi::memory_base();
    if base == 0 {
        return Ok(false);
    }
    let dst = recomp_abi::cpu_get(0) as u64;
    let src = recomp_abi::cpu_get(1) as u64;
    let len = recomp_abi::cpu_get(2) as u64;
    let memory = base as *mut u8;
    if !guest_range(len, dst) || !guest_range(len, src) {
        return Ok(false);
    }
    if len != 0 {
        // SAFETY: both ranges were checked against the reservation. A guest
        // memcpy promises non-overlap; `copy` also handles the overlap the
        // C contract leaves undefined, so no input can corrupt the copy.
        unsafe { copy_guest(memory, dst, src, len) };
    }
    recomp_abi::cpu_set(0, dst as i64);
    Ok(true)
}

/// SAFETY: `dst..dst+len` and `src..src+len` are inside `base`'s buffer.
unsafe fn copy_guest(base: *mut u8, dst: u64, src: u64, len: u64) {
    std::ptr::copy(base.add(src as usize), base.add(dst as usize), len as usize);
}

/// `memset`: fill guest bytes, or fall back to the lifted loop.
#[no_mangle]
pub fn __recomp_host_memset() -> Result<bool, Fault> {
    let base = recomp_abi::memory_base();
    if base == 0 {
        return Ok(false);
    }
    let dst = recomp_abi::cpu_get(0) as u64;
    let value = recomp_abi::cpu_get(1) as u8;
    let len = recomp_abi::cpu_get(2) as u64;
    if !guest_range(len, dst) {
        return Ok(false);
    }
    if len != 0 {
        // SAFETY: the range was checked against the reservation.
        unsafe {
            std::ptr::write_bytes((base as *mut u8).add(dst as usize), value, len as usize);
        }
    }
    recomp_abi::cpu_set(0, dst as i64);
    Ok(true)
}

fn guest_range(len: u64, address: u64) -> bool {
    let limit = guest_limit();
    len == 0 || address.checked_add(len).is_some_and(|end| end <= limit)
}

/// End (guest address) of the memory host functions may touch: the
/// reservation's covered window when a direct base is published (which can
/// be smaller than the whole space on a 39-bit host), else all of it.
fn guest_limit() -> u64 {
    match recomp_abi::memory_base() {
        0 => MEMORY_SPACE,
        _ => recomp_abi::memory_limit(),
    }
}

/// zlib's `inflate_mode` TYPE (zlib 1.2.9 and later number the modes from
/// 16180): the next block's header comes next.
const ZLIB_TYPE: u32 = 16191;

/// The guest address space the fast memory reservation covers.
const MEMORY_SPACE: u64 = 1 << 39;

/// zlib's `inflate_fast(strm, start)`, for the 64-bit zlib 1.2.11 layout
/// (`z_stream` and `inflate_state` field offsets below). Guest titles link
/// zlib statically; LM3 inflates every file it loads with it, which made
/// loading bound by this one recompiled loop.
#[no_mangle]
pub fn __recomp_host_zlib_inflate_fast() -> Result<bool, Fault> {
    let base = recomp_abi::memory_base();
    if base == 0 {
        return Ok(false);
    }
    let strm = recomp_abi::cpu_get(0) as u64;
    let start = recomp_abi::cpu_get(1) as u32;
    // SAFETY: guest memory is one host reservation from `base`;
    // `inflate_fast` checks every range it touches is inside it, and an
    // unmapped page faults into the runtime as a guest access would.
    Ok(unsafe { inflate_fast(base as *mut u8, strm, start) })
}

/// A decoding table entry (zlib's `code`): operation, bits, value.
#[derive(Clone, Copy)]
struct Code {
    op: u32,
    bits: u32,
    val: u32,
}

/// Decode until fewer than 6 input or 258 output bytes remain, or a block
/// ends, as zlib's `inflate_fast` does. Returns false without writing any
/// stream or state field on an invalid code or a distance back past the
/// window (the guest's version then runs from the same state and sets the
/// error; the output bytes written meanwhile it writes again).
unsafe fn inflate_fast(memory: *mut u8, strm: u64, start: u32) -> bool {
    let limit = guest_limit();
    let inside = |address: u64, len: u64| address.checked_add(len).is_some_and(|end| end <= limit);
    let at = |address: u64| memory.add(address as usize);
    let read_u32 = |address: u64| (at(address) as *const u32).read_unaligned();
    let read_u64 = |address: u64| (at(address) as *const u64).read_unaligned();
    if !inside(strm, 0x40) {
        return false;
    }
    // z_stream: next_in, avail_in, next_out, avail_out, state.
    let (next_in, avail_in) = (read_u64(strm), read_u32(strm + 0x8));
    let (next_out, avail_out) = (read_u64(strm + 0x18), read_u32(strm + 0x20));
    let state = read_u64(strm + 0x38);
    if avail_in < 6 || avail_out < 258 || start < avail_out || !inside(state, 0x80) {
        return false;
    }
    // inflate_state: the sliding window, the bit buffer and the tables.
    let (wsize, whave, wnext) = (
        read_u32(state + 0x3c) as usize,
        read_u32(state + 0x40) as usize,
        read_u32(state + 0x44) as usize,
    );
    let window = read_u64(state + 0x48);
    let mut hold = read_u64(state + 0x50);
    let mut bits = read_u32(state + 0x58);
    let (lcode, dcode) = (read_u64(state + 0x68), read_u64(state + 0x70));
    let (lenbits, distbits) = (read_u32(state + 0x78), read_u32(state + 0x7c));
    let before = (start - avail_out) as u64;
    let ranges_ok = inside(next_in, u64::from(avail_in) + 1)
        && next_out >= before
        && inside(next_out - before, before + u64::from(avail_out))
        && inside(window, wsize as u64)
        && lenbits < 16
        && distbits < 16
        && inside(lcode, 4 << 15)
        && inside(dcode, 4 << 15);
    if !ranges_ok {
        return false;
    }
    let code = |table: u64, index: u64| {
        let word = read_u32(table + index * 4);
        Code {
            op: word & 0xFF,
            bits: (word >> 8) & 0xFF,
            val: word >> 16,
        }
    };
    let input = at(next_in) as *const u8;
    let output = at(next_out);
    let window = at(window) as *const u8;
    let (lmask, dmask) = ((1u64 << lenbits) - 1, (1u64 << distbits) - 1);
    // Input read and output written so far; the loop runs while 6 input
    // and 258 output bytes remain.
    let (mut inp, last) = (0usize, avail_in as usize - 5);
    let (mut out, end) = (0usize, avail_out as usize - 257);
    let mut block_ended = false;

    macro_rules! pull {
        () => {{
            hold += u64::from(*input.add(inp)) << bits;
            inp += 1;
            bits += 8;
        }};
    }
    macro_rules! drop_bits {
        ($count:expr) => {{
            let count = $count;
            hold >>= count;
            bits -= count;
        }};
    }

    loop {
        if bits < 15 {
            pull!();
            pull!();
        }
        let mut here = code(lcode, hold & lmask);
        loop {
            drop_bits!(here.bits);
            let op = here.op;
            if op == 0 {
                // A literal.
                *output.add(out) = here.val as u8;
                out += 1;
                break;
            }
            if op & 16 != 0 {
                // A length, then its distance.
                let mut len = here.val as usize;
                let extra = op & 15;
                if extra != 0 {
                    if bits < extra {
                        pull!();
                    }
                    len += (hold & ((1u64 << extra) - 1)) as usize;
                    drop_bits!(extra);
                }
                if bits < 15 {
                    pull!();
                    pull!();
                }
                let mut here = code(dcode, hold & dmask);
                loop {
                    drop_bits!(here.bits);
                    let op = here.op;
                    if op & 16 != 0 {
                        let mut dist = here.val as usize;
                        let extra = op & 15;
                        if bits < extra {
                            pull!();
                            if bits < extra {
                                pull!();
                            }
                        }
                        dist += (hold & ((1u64 << extra) - 1)) as usize;
                        drop_bits!(extra);
                        let produced = before as usize + out;
                        let mut from: *const u8;
                        if dist > produced {
                            // From the window first.
                            let mut back = dist - produced;
                            if back > whave {
                                return false;
                            }
                            if wnext == 0 {
                                from = window.add(wsize - back);
                                if back < len {
                                    len -= back;
                                    copy(output, &mut out, &mut from, back);
                                    from = output.add(out).sub(dist);
                                }
                            } else if wnext < back {
                                // Wrapping around the window's end.
                                from = window.add(wsize + wnext - back);
                                back -= wnext;
                                if back < len {
                                    len -= back;
                                    copy(output, &mut out, &mut from, back);
                                    from = window;
                                    if wnext < len {
                                        len -= wnext;
                                        copy(output, &mut out, &mut from, wnext);
                                        from = output.add(out).sub(dist);
                                    }
                                }
                            } else {
                                from = window.add(wnext - back);
                                if back < len {
                                    len -= back;
                                    copy(output, &mut out, &mut from, back);
                                    from = output.add(out).sub(dist);
                                }
                            }
                        } else {
                            from = output.add(out).sub(dist);
                        }
                        copy(output, &mut out, &mut from, len);
                        break;
                    }
                    if op & 64 == 0 {
                        // A second-level distance code.
                        here = code(dcode, u64::from(here.val) + (hold & ((1u64 << op) - 1)));
                        continue;
                    }
                    return false;
                }
                break;
            }
            if op & 64 == 0 {
                // A second-level length code.
                here = code(lcode, u64::from(here.val) + (hold & ((1u64 << op) - 1)));
                continue;
            }
            if op & 32 != 0 {
                block_ended = true;
                break;
            }
            return false;
        }
        if block_ended || inp >= last || out >= end {
            break;
        }
    }

    // Return whole unused bytes to the input.
    let unused = (bits >> 3) as usize;
    inp -= unused;
    bits -= (unused as u32) << 3;
    hold &= (1u64 << bits) - 1;
    let write_u32 = |address: u64, value: u32| (at(address) as *mut u32).write_unaligned(value);
    let write_u64 = |address: u64, value: u64| (at(address) as *mut u64).write_unaligned(value);
    write_u64(strm, next_in + inp as u64);
    write_u32(strm + 0x8, avail_in - inp as u32);
    write_u64(strm + 0x18, next_out + out as u64);
    write_u32(strm + 0x20, avail_out - out as u32);
    write_u64(state + 0x50, hold);
    write_u32(state + 0x58, bits);
    if block_ended {
        write_u32(state + 0x8, ZLIB_TYPE);
    }
    true
}

/// Copy `count` bytes forward one at a time (a match may overlap the bytes
/// it produces), advancing both cursors.
unsafe fn copy(output: *mut u8, out: &mut usize, from: &mut *const u8, count: usize) {
    for _ in 0..count {
        *output.add(*out) = **from;
        *out += 1;
        *from = from.add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zlib stream's state, laid out in a byte buffer standing in for guest
    /// memory (address 0 is the buffer's start).
    struct Guest(Vec<u8>);

    impl Guest {
        fn put_u32(&mut self, address: u64, value: u32) {
            let at = address as usize;
            self.0[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        fn put_u64(&mut self, address: u64, value: u64) {
            let at = address as usize;
            self.0[at..at + 8].copy_from_slice(&value.to_le_bytes());
        }
        fn u32(&self, address: u64) -> u32 {
            let at = address as usize;
            u32::from_le_bytes(self.0[at..at + 4].try_into().unwrap())
        }
    }

    /// Fixed Huffman tables as zlib builds them (`fixedtables`): length
    /// codes with 9 index bits, distance codes with 5.
    fn fixed_tables(guest: &mut Guest, lcode: u64, dcode: u64) {
        // Literal/length code lengths: 0-143 eight bits, 144-255 nine,
        // 256-279 seven, 280-287 eight.
        let lengths: Vec<u32> = (0..288)
            .map(|symbol| match symbol {
                0..=143 => 8,
                144..=255 => 9,
                256..=279 => 7,
                _ => 8,
            })
            .collect();
        let base = [
            3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99,
            115, 131, 163, 195, 227, 258,
        ];
        let extra = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
        ];
        fill(guest, lcode, 9, &lengths, |symbol| match symbol {
            0..=255 => (0, symbol),
            256 => (96, 0),
            257..=285 => (
                16 + extra[symbol as usize - 257],
                base[symbol as usize - 257],
            ),
            _ => (64, 0),
        });
        let dbase = [
            1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025,
            1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
        ];
        let dlengths = vec![5; 30];
        fill(guest, dcode, 5, &dlengths, |symbol| {
            let extra = if symbol < 4 { 0 } else { symbol / 2 - 1 };
            (16 + extra, dbase[symbol as usize])
        });
    }

    /// A one-level table of canonical codes (every length <= `index_bits`):
    /// each code fills the entries whose low bits are its reversed code.
    fn fill(
        guest: &mut Guest,
        table: u64,
        index_bits: u32,
        lengths: &[u32],
        entry: impl Fn(u32) -> (u32, u32),
    ) {
        let mut next = [0u32; 16];
        let mut count = [0u32; 16];
        for &length in lengths {
            count[length as usize] += 1;
        }
        let mut code = 0;
        for bits in 1..16 {
            code = (code + count[bits - 1]) << 1;
            next[bits] = code;
        }
        for (symbol, &length) in lengths.iter().enumerate() {
            let code = next[length as usize];
            next[length as usize] += 1;
            let reversed = code.reverse_bits() >> (32 - length);
            let (op, val) = entry(symbol as u32);
            let word = op | (length << 8) | (val << 16);
            let mut index = reversed;
            while index < 1 << index_bits {
                guest.put_u32(table + u64::from(index) * 4, word);
                index += 1 << length;
            }
        }
    }

    #[test]
    fn inflates_a_fixed_huffman_block_with_back_references() {
        // Raw deflate of b"abcabcabcabc" at level 9: one fixed Huffman block
        // (three literals, then a length 9 distance 3 match).
        let compressed = [0x4b, 0x4c, 0x4a, 0x4e, 0x84, 0x21, 0x00];
        let mut guest = Guest(vec![0; 0x40000]);
        let (strm, state, input, output, window) = (0x100, 0x200, 0x1000, 0x2000, 0x3000);
        let (lcode, dcode) = (0x10000, 0x20000);
        fixed_tables(&mut guest, lcode, dcode);
        guest.0[input as usize..input as usize + compressed.len()].copy_from_slice(&compressed);
        // The block header (BFINAL 1, BTYPE 01) is consumed: 3 bits gone
        // from the first byte, its other 5 in the bit buffer.
        guest.put_u64(strm, input + 1);
        // Room past the data, so the loop reaches the end-of-block code
        // (it stops with fewer than 6 bytes left, as zlib's does).
        guest.put_u32(strm + 0x8, compressed.len() as u32 - 1 + 32);
        guest.put_u64(strm + 0x18, output);
        guest.put_u32(strm + 0x20, 0x1000);
        guest.put_u64(strm + 0x38, state);
        guest.put_u32(state + 0x3c, 0x8000);
        guest.put_u64(state + 0x48, window);
        guest.put_u64(state + 0x50, u64::from(compressed[0] >> 3));
        guest.put_u32(state + 0x58, 5);
        guest.put_u64(state + 0x68, lcode);
        guest.put_u64(state + 0x70, dcode);
        guest.put_u32(state + 0x78, 9);
        guest.put_u32(state + 0x7c, 5);
        let memory = guest.0.as_mut_ptr();
        assert!(unsafe { inflate_fast(memory, strm, 0x1000) });
        assert_eq!(
            &guest.0[output as usize..output as usize + 12],
            b"abcabcabcabc"
        );
        assert_eq!(guest.u32(strm + 0x20), 0x1000 - 12);
        assert_eq!(guest.u32(state + 0x8), ZLIB_TYPE);
    }

    #[test]
    fn guest_copies_handle_overlap_like_memmove() {
        let mut buffer = b"abcdef".to_vec();
        // SAFETY: the ranges are inside the buffer.
        unsafe { copy_guest(buffer.as_mut_ptr(), 2, 0, 4) };
        assert_eq!(&buffer, b"ababcd");
    }

    #[test]
    fn guest_ranges_stay_inside_the_reservation() {
        assert!(guest_range(16, MEMORY_SPACE - 16));
        assert!(!guest_range(17, MEMORY_SPACE - 16));
        assert!(!guest_range(1, u64::MAX));
        assert!(guest_range(0, u64::MAX));
    }

    #[test]
    fn the_time_host_function_returns_microseconds() {
        let before = recomp_services::clock::now_ns() / 1000;
        assert!(__recomp_host_nv_os_get_time_us().unwrap());
        let value = recomp_abi::cpu_get(0) as u64;
        let after = recomp_services::clock::now_ns() / 1000;
        assert!(value >= before && value <= after);
    }
}
