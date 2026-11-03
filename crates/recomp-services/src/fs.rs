//! `fsp-srv`: the title's RomFS through IStorage, and save data kept in a
//! host directory.

use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::file_at::{read_at, write_all_at};
use crate::romfs::VirtualRomFs;

use crate::server::{Call, Service};
use crate::services::unimplemented;

/// fs::ResultPathNotFound (module 2, description 1).
pub const RESULT_PATH_NOT_FOUND: u32 = 0x202;
/// fs::ResultPathAlreadyExists (module 2, description 2).
pub const RESULT_PATH_ALREADY_EXISTS: u32 = 0x402;
/// fs::ResultTargetNotFound (module 2, description 1002).
pub const RESULT_TARGET_NOT_FOUND: u32 = 0x7_D402;
/// Size of one IDirectory entry: name[0x301], pad, type, pad, size.
const DIRECTORY_ENTRY_SIZE: usize = 0x310;
/// Space reported for save data file systems.
const SAVE_SPACE: u64 = 512 << 20;

fn io_result(error: std::io::Error) -> u32 {
    match error.kind() {
        ErrorKind::AlreadyExists => RESULT_PATH_ALREADY_EXISTS,
        _ => RESULT_PATH_NOT_FOUND,
    }
}

/// IFileSystemProxy
pub struct FileSystemProxy;

impl Service for FileSystemProxy {
    fn name(&self) -> String {
        "fsp-srv".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // SetCurrentProcess, CreateSaveDataFileSystem,
            // DisableAutoSaveDataCreation, SetGlobalAccessLogMode,
            // OutputAccessLogToSdCard
            1 | 22 | 1003 | 1004 | 1006 => {}
            // OpenSaveDataFileSystem(space, attribute),
            // OpenSaveDataFileSystemBySystemSaveDataId,
            // OpenReadOnlySaveDataFileSystem
            51..=53 => {
                let root = save_directory(call)?;
                std::fs::create_dir_all(&root).map_err(io_result)?;
                println!("[recomp-ipc] fsp-srv save data at {}", root.display());
                call.push_object(Box::new(HostFileSystem { root }));
            }
            // OpenDataStorageByCurrentProcess: the RomFS image, or one laid
            // out over a RomFS directory.
            200 => {
                let path = call
                    .context
                    .romfs_image
                    .as_ref()
                    .ok_or(RESULT_TARGET_NOT_FOUND)?;
                let storage = if path.is_dir() {
                    RomStorage::Directory(directory_romfs(path)?)
                } else {
                    RomStorage::Image(File::open(path).map_err(|_| RESULT_TARGET_NOT_FOUND)?)
                };
                call.push_object(Box::new(FileStorage(storage)));
            }
            // GetGlobalAccessLogMode
            1005 => call.push_u32(0),
            // GetProgramIndexForAccessLog -> (index, count)
            1011 => {
                call.push_u32(0);
                call.push_u32(1);
            }
            _ => return unimplemented(call, "fsp-srv"),
        }
        Ok(())
    }
}

/// The host directory for the save data a SaveDataAttribute names
/// (arguments: space id at 0, attribute at 8).
fn save_directory(call: &Call<'_>) -> Result<PathBuf, u32> {
    let root = call
        .context
        .save_root
        .as_ref()
        .ok_or(RESULT_TARGET_NOT_FOUND)?;
    let user = (call.arg_u64(0x10), call.arg_u64(0x18));
    let system_id = call.arg_u64(0x20);
    let kind = call.arg_bytes(0x28, 1)[0];
    let index = u16::from_le_bytes(call.arg_bytes(0x2A, 2).try_into().unwrap_or([0; 2]));
    Ok(root.join(format!(
        "{kind}-{:016x}{:016x}-{system_id:016x}-{index}",
        user.1, user.0
    )))
}

/// The virtual RomFS over `path`, built once and shared by every storage.
fn directory_romfs(path: &Path) -> Result<Arc<VirtualRomFs>, u32> {
    static BUILT: Mutex<Option<(PathBuf, Arc<VirtualRomFs>)>> = Mutex::new(None);
    let mut built = BUILT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((built_path, romfs)) = built.as_ref() {
        if built_path == path {
            return Ok(romfs.clone());
        }
    }
    let romfs = Arc::new(VirtualRomFs::build(path).map_err(|error| {
        println!(
            "[recomp-ipc] fsp-srv: cannot lay out RomFS over {}: {error}",
            path.display()
        );
        RESULT_TARGET_NOT_FOUND
    })?);
    println!(
        "[recomp-ipc] fsp-srv: RomFS laid out over {} ({} bytes)",
        path.display(),
        romfs.size()
    );
    *built = Some((path.to_path_buf(), romfs.clone()));
    Ok(romfs)
}

enum RomStorage {
    Image(File),
    Directory(Arc<VirtualRomFs>),
}

/// IStorage over the RomFS.
struct FileStorage(RomStorage);

impl Service for FileStorage {
    fn name(&self) -> String {
        "fsp-srv:IStorage".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // Read(offset, size) -> buffer
            0 => {
                let (offset, size) = (call.arg_u64(0), call.arg_u64(8));
                let capacity = call.out_buffer(0).map_or(0, |buffer| buffer.size);
                let mut bytes = vec![0u8; size.min(capacity) as usize];
                let read = match &self.0 {
                    RomStorage::Image(file) => read_fully(file, &mut bytes, offset),
                    RomStorage::Directory(romfs) => romfs.read(offset, &mut bytes),
                };
                bytes.truncate(read);
                call.write_out(0, &bytes);
                if tracing() {
                    let files = match &self.0 {
                        RomStorage::Directory(romfs) => romfs
                            .files_at(offset, size)
                            .iter()
                            .map(|(path, at)| format!(" {}+{at:#x}", path.display()))
                            .collect(),
                        RomStorage::Image(_) => String::new(),
                    };
                    println!("[recomp-fs] read {offset:#x} size {size:#x} -> {read:#x}{files}");
                }
            }
            // Flush
            2 => {}
            // GetSize
            4 => {
                let size = match &self.0 {
                    RomStorage::Image(file) => file.metadata().map(|meta| meta.len()).unwrap_or(0),
                    RomStorage::Directory(romfs) => romfs.size(),
                };
                call.push_u64(size);
            }
            _ => return unimplemented(call, "fsp-srv:IStorage"),
        }
        Ok(())
    }
}

/// RomFS reads are logged with `RECOMP_TRACE_FS=1`, naming the files read,
/// and so are writes to save data.
fn tracing() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("RECOMP_TRACE_FS").is_some())
}

/// Read until `bytes` is full or the file ends; returns the bytes read.
fn read_fully(file: &File, bytes: &mut [u8], offset: u64) -> usize {
    let mut done = 0;
    while done < bytes.len() {
        match read_at(file, &mut bytes[done..], offset + done as u64) {
            Ok(0) | Err(_) => break,
            Ok(n) => done += n,
        }
    }
    done
}

/// IFileSystem over a host directory.
struct HostFileSystem {
    root: PathBuf,
}

impl HostFileSystem {
    /// The host path for the guest path in input buffer `index`; `..`
    /// cannot leave the root.
    fn resolve(&self, call: &Call<'_>, index: usize) -> Result<PathBuf, u32> {
        let raw = call.in_buffer(index);
        let end = raw.iter().position(|byte| *byte == 0).unwrap_or(raw.len());
        let path = String::from_utf8_lossy(&raw[..end]).into_owned();
        let mut host = self.root.clone();
        for part in path.split('/') {
            match part {
                "" | "." => {}
                ".." => return Err(RESULT_PATH_NOT_FOUND),
                part => host.push(part),
            }
        }
        Ok(host)
    }
}

impl Service for HostFileSystem {
    fn name(&self) -> String {
        "fsp-srv:IFileSystem".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // CreateFile(path, size, option)
            0 => {
                let path = self.resolve(call, 0)?;
                let file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .map_err(io_result)?;
                file.set_len(call.arg_u64(0)).map_err(io_result)?;
            }
            // DeleteFile
            1 => std::fs::remove_file(self.resolve(call, 0)?).map_err(io_result)?,
            // CreateDirectory
            2 => std::fs::create_dir(self.resolve(call, 0)?).map_err(io_result)?,
            // DeleteDirectory
            3 => std::fs::remove_dir(self.resolve(call, 0)?).map_err(io_result)?,
            // DeleteDirectoryRecursively
            4 => std::fs::remove_dir_all(self.resolve(call, 0)?).map_err(io_result)?,
            // RenameFile, RenameDirectory
            5 | 6 => {
                let from = self.resolve(call, 0)?;
                let to = self.resolve(call, 1)?;
                if to.exists() {
                    return Err(RESULT_PATH_ALREADY_EXISTS);
                }
                std::fs::rename(from, to).map_err(io_result)?;
            }
            // GetEntryType: 0 directory, 1 file.
            7 => {
                let meta = std::fs::metadata(self.resolve(call, 0)?).map_err(io_result)?;
                call.push_u32(u32::from(!meta.is_dir()));
            }
            // OpenFile(path, mode): bit 0 read, bit 1 write, bit 2 append.
            8 => {
                let path = self.resolve(call, 0)?;
                let mode = call.arg_u32(0);
                if path.is_dir() {
                    return Err(RESULT_PATH_NOT_FOUND);
                }
                let file = OpenOptions::new()
                    .read(true)
                    .write(mode & 0b110 != 0)
                    .open(&path)
                    .map_err(io_result)?;
                call.push_object(Box::new(HostFile(file)));
            }
            // OpenDirectory(path, filter): bit 0 directories, bit 1 files.
            9 => {
                let path = self.resolve(call, 0)?;
                let filter = call.arg_u32(0);
                let mut entries = Vec::new();
                for entry in std::fs::read_dir(&path).map_err(io_result)?.flatten() {
                    let Ok(meta) = entry.metadata() else {
                        continue;
                    };
                    let wanted = if meta.is_dir() { 1 } else { 2 };
                    if filter & wanted != 0 {
                        let name = entry.file_name().to_string_lossy().into_owned();
                        let size = if meta.is_dir() { 0 } else { meta.len() };
                        entries.push((name, meta.is_dir(), size));
                    }
                }
                entries.sort();
                call.push_object(Box::new(HostDirectory { entries, next: 0 }));
            }
            // Commit
            10 => {}
            // GetFreeSpaceSize, GetTotalSpaceSize
            11 | 12 => call.push_u64(SAVE_SPACE),
            // CleanDirectoryRecursively
            13 => {
                let path = self.resolve(call, 0)?;
                for entry in std::fs::read_dir(&path).map_err(io_result)?.flatten() {
                    let target = entry.path();
                    if target.is_dir() {
                        std::fs::remove_dir_all(target).map_err(io_result)?;
                    } else {
                        std::fs::remove_file(target).map_err(io_result)?;
                    }
                }
            }
            // GetFileTimeStampRaw
            14 => {
                self.resolve(call, 0)?;
                call.push_bytes(&[0; 0x20]);
            }
            _ => return unimplemented(call, "fsp-srv:IFileSystem"),
        }
        Ok(())
    }
}

/// IFile over a host file.
struct HostFile(File);

impl Service for HostFile {
    fn name(&self) -> String {
        "fsp-srv:IFile".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // Read(option, offset, size) -> (buffer, bytes read)
            0 => {
                let (offset, size) = (call.arg_u64(8), call.arg_u64(16));
                let capacity = call.out_buffer(0).map_or(0, |buffer| buffer.size);
                let mut bytes = vec![0u8; size.min(capacity) as usize];
                let read = read_fully(&self.0, &mut bytes, offset);
                call.write_out(0, &bytes[..read]);
                call.push_u64(read as u64);
            }
            // Write(option, offset, size, buffer)
            1 => {
                let (offset, size) = (call.arg_u64(8), call.arg_u64(16));
                let bytes = call.in_buffer(0);
                let len = (size as usize).min(bytes.len());
                if tracing() {
                    println!("[recomp-fs] save data write: {len} bytes at {offset:#x}");
                }
                write_all_at(&self.0, &bytes[..len], offset).map_err(io_result)?;
            }
            // Flush
            2 => {}
            // SetSize
            3 => self.0.set_len(call.arg_u64(0)).map_err(io_result)?,
            // GetSize
            4 => {
                let size = self.0.metadata().map_err(io_result)?.len();
                call.push_u64(size);
            }
            _ => return unimplemented(call, "fsp-srv:IFile"),
        }
        Ok(())
    }
}

/// IDirectory over a listing taken when it was opened.
struct HostDirectory {
    entries: Vec<(String, bool, u64)>,
    next: usize,
}

impl Service for HostDirectory {
    fn name(&self) -> String {
        "fsp-srv:IDirectory".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // Read -> (entries, count)
            0 => {
                let capacity = call.out_buffer(0).map_or(0, |buffer| buffer.size) as usize;
                let count = (capacity / DIRECTORY_ENTRY_SIZE).min(self.entries.len() - self.next);
                let mut bytes = vec![0u8; count * DIRECTORY_ENTRY_SIZE];
                for (slot, (name, is_dir, size)) in bytes
                    .chunks_mut(DIRECTORY_ENTRY_SIZE)
                    .zip(&self.entries[self.next..self.next + count])
                {
                    let name = name.as_bytes();
                    let len = name.len().min(0x300);
                    slot[..len].copy_from_slice(&name[..len]);
                    slot[0x304] = u8::from(!is_dir);
                    slot[0x308..0x310].copy_from_slice(&size.to_le_bytes());
                }
                self.next += count;
                call.write_out(0, &bytes);
                call.push_u64(count as u64);
            }
            // GetEntryCount
            1 => call.push_u64(self.entries.len() as u64),
            _ => return unimplemented(call, "fsp-srv:IDirectory"),
        }
        Ok(())
    }
}
