//! The virtual RomFS: files are found through the hash tables, as the
//! SDK's RomFS driver looks them up, and read back intact.

use recomp_services::romfs::{bucket_count, path_hash, VirtualRomFs};

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

#[test]
fn files_are_found_through_the_hash_tables() {
    let root = std::env::temp_dir().join(format!("recomp-romfs-{}", std::process::id()));
    std::fs::create_dir_all(root.join("ini/deep")).unwrap();
    let files: Vec<(&str, Vec<u8>)> = vec![
        ("flow.data", (0..7888u32).map(|i| i as u8).collect()),
        ("flow.patch", vec![7; 5668]),
        ("ini/e3rom.ini", b"[e3]\n".to_vec()),
        ("ini/deep/empty", Vec::new()),
    ];
    for (path, bytes) in &files {
        std::fs::write(root.join(path), bytes).unwrap();
    }
    let romfs = VirtualRomFs::build(&root).unwrap();
    let mut image = vec![0u8; romfs.size() as usize];
    assert_eq!(romfs.read(0, &mut image), image.len());
    let header: Vec<u64> = (0..10).map(|i| u64_at(&image, i * 8)).collect();
    assert_eq!((header[0], header[9]), (0x50, 0x200));
    let (dir_hash, dir_meta) = (header[1] as usize, header[3] as usize);
    let (file_hash, file_meta) = (header[5] as usize, header[7] as usize);
    let dir_buckets = header[2] as u32 / 4;
    let file_buckets = header[6] as u32 / 4;
    assert_eq!(file_buckets, bucket_count(files.len() as u32));

    // Look a name up in a hash table: walk the bucket chain for an entry
    // with this parent and name.
    let find = |table: usize, meta: usize, buckets: u32, parent: u32, name: &str, dir: bool| {
        let bucket = path_hash(parent, name.as_bytes()) % buckets;
        let mut entry = u32_at(&image, table + 4 * bucket as usize);
        while entry != u32::MAX {
            let at = meta + entry as usize;
            let (next_at, name_at) = if dir { (16, 0x18) } else { (24, 0x20) };
            let len = u32_at(&image, at + name_at - 4) as usize;
            if u32_at(&image, at) == parent
                && &image[at + name_at..at + name_at + len] == name.as_bytes()
            {
                return Some(entry);
            }
            entry = u32_at(&image, at + next_at);
        }
        None
    };
    for (path, bytes) in &files {
        let mut parent = 0;
        let (dirs, name) = path.rsplit_once('/').map_or(("", *path), |(d, n)| (d, n));
        for dir in dirs.split('/').filter(|d| !d.is_empty()) {
            parent = find(dir_hash, dir_meta, dir_buckets, parent, dir, true).expect(dir);
        }
        let entry = find(file_hash, file_meta, file_buckets, parent, name, false).expect(path);
        let at = file_meta + entry as usize;
        let (offset, size) = (u64_at(&image, at + 8), u64_at(&image, at + 16));
        let start = (0x200 + offset) as usize;
        assert_eq!(&image[start..start + size as usize], &bytes[..], "{path}");
    }
    std::fs::remove_dir_all(&root).ok();
}
