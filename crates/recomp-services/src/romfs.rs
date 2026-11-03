//! A RomFS image laid out over a host directory, such as an emulator's
//! RomFS dump with the title's update applied. The header and the
//! directory and file tables are built in memory; file data is read from
//! the host files on demand, so nothing is copied.
//!
//! Layout, as Nintendo's tools produce it: the 0x50-byte header, file data
//! from 0x200 (each file 16-byte aligned), then the directory hash table,
//! directory entries, file hash table and file entries.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::file_at::read_at;

const HEADER_SIZE: u64 = 0x50;
const DATA_OFFSET: u64 = 0x200;
const NONE: u32 = u32::MAX;

/// The RomFS path hash: the parent entry's offset mixed with the name.
pub fn path_hash(parent: u32, name: &[u8]) -> u32 {
    name.iter().fold(parent ^ 123_456_789, |hash, byte| {
        hash.rotate_right(5) ^ u32::from(*byte)
    })
}

/// Hash table bucket count for `entries` entries.
pub fn bucket_count(entries: u32) -> u32 {
    match entries {
        0..=2 => 3,
        3..=18 => entries | 1,
        _ => {
            let mut count = entries;
            while [2, 3, 5, 7, 11, 13, 17]
                .iter()
                .any(|d| count.is_multiple_of(*d))
            {
                count += 1;
            }
            count
        }
    }
}

struct DirNode {
    name: Vec<u8>,
    parent: usize,
    children: Vec<usize>,
    files: Vec<usize>,
    offset: u32,
}

struct FileNode {
    name: Vec<u8>,
    parent: usize,
    path: PathBuf,
    size: u64,
    offset: u32,
    data: u64,
}

/// A file's bytes within the image.
struct Extent {
    start: u64,
    size: u64,
    path: PathBuf,
}

pub struct VirtualRomFs {
    header: Vec<u8>,
    tables: Vec<u8>,
    tables_offset: u64,
    extents: Vec<Extent>,
    size: u64,
    /// Open handles, by extent index. A guest load reads a file many times,
    /// and every `File::open` is a round trip through the storage daemon on
    /// Android's `/sdcard` (FUSE), so keep the handles instead of reopening
    /// them per read. Capped so a large image cannot exhaust the descriptor
    /// limit; readers hold their own `Arc`, so clearing is safe.
    open: Mutex<HashMap<usize, Arc<File>>>,
}

/// Handles kept open at once (`VirtualRomFs::open`).
const OPEN_FILES: usize = 256;

fn padded(name: &[u8]) -> usize {
    name.len().next_multiple_of(4)
}

impl VirtualRomFs {
    pub fn build(root: &Path) -> io::Result<Self> {
        let mut dirs = vec![DirNode {
            name: Vec::new(),
            parent: 0,
            children: Vec::new(),
            files: Vec::new(),
            offset: 0,
        }];
        let mut files: Vec<FileNode> = Vec::new();
        // Breadth-first, children sorted by name.
        let mut queue = vec![(0usize, root.to_path_buf())];
        let mut next = 0;
        while next < queue.len() {
            let (index, path) = queue[next].clone();
            next += 1;
            let mut entries: Vec<_> = std::fs::read_dir(&path)?.collect::<Result<_, _>>()?;
            entries.sort_by_key(|entry| entry.file_name());
            for entry in entries {
                let name = entry
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
                    .into_bytes();
                let meta = entry.metadata()?;
                if meta.is_dir() {
                    dirs.push(DirNode {
                        name,
                        parent: index,
                        children: Vec::new(),
                        files: Vec::new(),
                        offset: 0,
                    });
                    let child = dirs.len() - 1;
                    dirs[index].children.push(child);
                    queue.push((child, entry.path()));
                } else {
                    files.push(FileNode {
                        name,
                        parent: index,
                        path: entry.path(),
                        size: meta.len(),
                        offset: 0,
                        data: 0,
                    });
                    let file = files.len() - 1;
                    dirs[index].files.push(file);
                }
            }
        }

        // Entry offsets within their tables, and file data placement.
        let mut at = 0u32;
        for dir in &mut dirs {
            dir.offset = at;
            at += (0x18 + padded(&dir.name)) as u32;
        }
        let dir_meta_size = at;
        let mut at = 0u32;
        let mut data = 0u64;
        for file in &mut files {
            file.offset = at;
            at += (0x20 + padded(&file.name)) as u32;
            file.data = data;
            data = (data + file.size).next_multiple_of(16);
        }
        let file_meta_size = at;

        let dir_buckets = bucket_count(dirs.len() as u32) as usize;
        let file_buckets = bucket_count(files.len() as u32) as usize;
        let mut dir_hash = vec![NONE; dir_buckets];
        let mut file_hash = vec![NONE; file_buckets];
        let mut dir_next = vec![NONE; dirs.len()];
        let mut file_next = vec![NONE; files.len()];
        for (index, dir) in dirs.iter().enumerate() {
            let parent = dirs[dir.parent].offset;
            let bucket = path_hash(parent, &dir.name) as usize % dir_buckets;
            dir_next[index] = dir_hash[bucket];
            dir_hash[bucket] = dir.offset;
        }
        for (index, file) in files.iter().enumerate() {
            let parent = dirs[file.parent].offset;
            let bucket = path_hash(parent, &file.name) as usize % file_buckets;
            file_next[index] = file_hash[bucket];
            file_hash[bucket] = file.offset;
        }

        let mut dir_meta = Vec::with_capacity(dir_meta_size as usize);
        for (index, dir) in dirs.iter().enumerate() {
            let parent = &dirs[dir.parent];
            let sibling = parent
                .children
                .iter()
                .skip_while(|child| **child != index)
                .nth(1)
                .map_or(NONE, |next| dirs[*next].offset);
            let sibling = if index == 0 { NONE } else { sibling };
            let child = dir.children.first().map_or(NONE, |c| dirs[*c].offset);
            let file = dir.files.first().map_or(NONE, |f| files[*f].offset);
            for value in [
                parent.offset,
                sibling,
                child,
                file,
                dir_next[index],
                dir.name.len() as u32,
            ] {
                dir_meta.extend_from_slice(&value.to_le_bytes());
            }
            dir_meta.extend_from_slice(&dir.name);
            dir_meta.resize(dir_meta.len().next_multiple_of(4), 0);
        }
        let mut file_meta = Vec::with_capacity(file_meta_size as usize);
        for (index, file) in files.iter().enumerate() {
            let parent = &dirs[file.parent];
            let sibling = parent
                .files
                .iter()
                .skip_while(|f| **f != index)
                .nth(1)
                .map_or(NONE, |next| files[*next].offset);
            file_meta.extend_from_slice(&parent.offset.to_le_bytes());
            file_meta.extend_from_slice(&sibling.to_le_bytes());
            file_meta.extend_from_slice(&file.data.to_le_bytes());
            file_meta.extend_from_slice(&file.size.to_le_bytes());
            file_meta.extend_from_slice(&file_next[index].to_le_bytes());
            file_meta.extend_from_slice(&(file.name.len() as u32).to_le_bytes());
            file_meta.extend_from_slice(&file.name);
            file_meta.resize(file_meta.len().next_multiple_of(4), 0);
        }

        let tables_offset = (DATA_OFFSET + data).next_multiple_of(4);
        let mut tables = Vec::new();
        let mut sections = [0u64; 8];
        for (index, table) in [
            dir_hash
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            dir_meta,
            file_hash
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            file_meta,
        ]
        .into_iter()
        .enumerate()
        {
            sections[index * 2] = tables_offset + tables.len() as u64;
            sections[index * 2 + 1] = table.len() as u64;
            tables.extend_from_slice(&table);
        }
        let mut header = Vec::with_capacity(DATA_OFFSET as usize);
        header.extend_from_slice(&HEADER_SIZE.to_le_bytes());
        for value in sections {
            header.extend_from_slice(&value.to_le_bytes());
        }
        header.extend_from_slice(&DATA_OFFSET.to_le_bytes());
        header.resize(DATA_OFFSET as usize, 0);

        let extents = files
            .into_iter()
            .map(|file| Extent {
                start: DATA_OFFSET + file.data,
                size: file.size,
                path: file.path,
            })
            .collect();
        Ok(VirtualRomFs {
            size: tables_offset + tables.len() as u64,
            header,
            tables,
            tables_offset,
            extents,
            open: Mutex::new(HashMap::new()),
        })
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// The files whose data overlaps `len` bytes at `offset`, with the
    /// offset into each (for tracing what the guest reads).
    pub fn files_at(&self, offset: u64, len: u64) -> Vec<(&Path, u64)> {
        let end = offset.saturating_add(len);
        let first = self
            .extents
            .partition_point(|extent| extent.start + extent.size <= offset);
        self.extents[first..]
            .iter()
            .take_while(|extent| extent.start < end)
            .map(|extent| (extent.path.as_path(), offset.saturating_sub(extent.start)))
            .collect()
    }

    /// Read image bytes at `offset`; returns how many were read (short only
    /// at the end of the image). Padding between files reads as zeros.
    pub fn read(&self, offset: u64, buffer: &mut [u8]) -> usize {
        let len = (buffer.len() as u64).min(self.size.saturating_sub(offset)) as usize;
        let buffer = &mut buffer[..len];
        buffer.fill(0);
        let end = offset + len as u64;
        let copy = |buffer: &mut [u8], source: &[u8], source_start: u64| {
            let from = offset.max(source_start);
            let to = end.min(source_start + source.len() as u64);
            if from < to {
                buffer[(from - offset) as usize..(to - offset) as usize].copy_from_slice(
                    &source[(from - source_start) as usize..(to - source_start) as usize],
                );
            }
        };
        copy(buffer, &self.header, 0);
        copy(buffer, &self.tables, self.tables_offset);
        let first = self
            .extents
            .partition_point(|extent| extent.start + extent.size <= offset);
        for (index, extent) in self.extents[first..]
            .iter()
            .enumerate()
            .take_while(|(_, extent)| extent.start < end)
        {
            let from = offset.max(extent.start);
            let to = end.min(extent.start + extent.size);
            if from >= to {
                continue;
            }
            if let Some(file) = self.handle(first + index) {
                let target = &mut buffer[(from - offset) as usize..(to - offset) as usize];
                let mut done = 0;
                while done < target.len() {
                    match read_at(
                        &file,
                        &mut target[done..],
                        from - extent.start + done as u64,
                    ) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => done += n,
                    }
                }
            }
        }
        len
    }

    /// The open handle of extent `index`, opening it on first use.
    fn handle(&self, index: usize) -> Option<Arc<File>> {
        let mut open = self.open.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(file) = open.get(&index) {
            return Some(file.clone());
        }
        if open.len() >= OPEN_FILES {
            open.clear();
        }
        let file = Arc::new(File::open(&self.extents[index].path).ok()?);
        open.insert(index, file.clone());
        Some(file)
    }
}
