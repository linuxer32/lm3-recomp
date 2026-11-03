//! Guest memory the host's engines wrote behind the GPU's back (video
//! frames VIC converted): a renderer that caches textures read from guest
//! memory takes these ranges and reads the textures there again.

use std::sync::Mutex;

static WRITES: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Note that `len` bytes at GPU address `address` were written.
pub fn record(address: u64, len: u64) {
    let mut writes = WRITES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !writes.contains(&(address, len)) {
        writes.push((address, len));
    }
}

/// The ranges written since the last call, as (GPU address, length).
pub fn take() -> Vec<(u64, u64)> {
    std::mem::take(
        &mut *WRITES
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_are_taken_once() {
        record(0x1000, 0x10);
        record(0x1000, 0x10);
        record(0x2000, 0x20);
        let taken = take();
        assert!(taken.contains(&(0x1000, 0x10)) && taken.contains(&(0x2000, 0x20)));
        assert_eq!(taken.iter().filter(|write| write.0 == 0x1000).count(), 1);
        assert!(take().iter().all(|write| write.0 != 0x1000));
    }
}
