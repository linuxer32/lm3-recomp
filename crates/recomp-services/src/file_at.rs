//! Positioned file I/O: `read_at`/`write_all_at` are Unix-only extensions of
//! `File`, so fs and RomFS use these helpers and work on Windows too.

use std::fs::File;
use std::io;

/// Read into `buf` at `offset`, returning the bytes read (0 at EOF).
#[cfg(unix)]
pub(crate) fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

/// Write all of `buf` at `offset`.
#[cfg(unix)]
pub(crate) fn write_all_at(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

/// Read into `buf` at `offset`, returning the bytes read (0 at EOF).
#[cfg(windows)]
pub(crate) fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

/// Write at `offset`, returning the bytes written.
#[cfg(windows)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_write(buf, offset)
}

/// Write all of `buf` at `offset`.
#[cfg(windows)]
pub(crate) fn write_all_at(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    let mut done = 0;
    while done < buf.len() {
        match write_at(file, &buf[done..], offset + done as u64)? {
            0 => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write the whole buffer",
                ))
            }
            written => done += written,
        }
    }
    Ok(())
}
