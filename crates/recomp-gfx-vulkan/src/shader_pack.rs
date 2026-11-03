//! Portable SPIR-V shader pack.
//!
//! The offline extractor translates the game's shipped Maxwell shader library
//! and writes this file. It is independent of the GPU and driver. At startup
//! the backend looks a shader up by its source hash and translation
//! environment; a hit skips Maxwell-to-SPIR-V translation and uses the
//! prebuilt module directly. The driver still builds a pipeline for the
//! target GPU, which is what `shader-precompile` warms separately.
//!
//! A pack holds several thousand modules (185 MB for LM3), so loading it does
//! not read the modules: the loader keeps the file open, reads the `LM3SPV02`
//! offset table, and indexes each entry by its 50-byte key (hash, stage and
//! type environment). A module's metadata and SPIR-V words are read and
//! parsed only when the game first asks for that key.

use std::fs::File;
use std::path::{Path, PathBuf};

use foldhash::HashMap;
use recomp_gfx::maxwell::sph::Stage;
use recomp_gfx::maxwell::translate::{TextureBinding, TextureKind, TranslatedShader};

const MAGIC_V1: &[u8; 8] = b"LM3SPV01";
const MAGIC_V2: &[u8; 8] = b"LM3SPV02";

/// Magic and entry count.
const HEADER_BYTES: u64 = 12;
/// Hash, stage and type environment: what a lookup matches.
const KEY_BYTES: usize = 8 + 1 + 41;

/// Where one entry starts and ends in the pack.
#[derive(Clone, Copy)]
struct Span {
    offset: u64,
    len: u64,
}

pub struct ShaderPack {
    file: File,
    path: PathBuf,
    entries: HashMap<(u64, [u8; 41]), Span>,
}

impl ShaderPack {
    pub fn load(path: &Path) -> Result<Self, String> {
        let started = std::time::Instant::now();
        let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let len = file
            .metadata()
            .map_err(|error| format!("{}: {error}", path.display()))?
            .len();
        let mut header = [0u8; HEADER_BYTES as usize];
        read_exact_at(&file, &mut header, 0)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let indexed = &header[..8] == MAGIC_V2;
        if !indexed && &header[..8] != MAGIC_V1 {
            return Err("bad shader pack magic".into());
        }
        let count = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;

        let mut entries = HashMap::default();
        if indexed {
            let table_bytes = count
                .checked_mul(8)
                .ok_or("shader pack entry count is too large")?;
            let table_end = HEADER_BYTES
                .checked_add(table_bytes as u64)
                .ok_or("shader pack entry count is too large")?;
            if table_end > len {
                return Err("shader pack truncated".into());
            }
            let mut table = vec![0u8; table_bytes];
            read_exact_at(&file, &mut table, HEADER_BYTES)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            let mut offsets = Vec::with_capacity(count);
            for chunk in table.chunks_exact(8) {
                offsets.push(u64::from_le_bytes(chunk.try_into().unwrap()));
            }
            for (index, &offset) in offsets.iter().enumerate() {
                let end = offsets.get(index + 1).copied().unwrap_or(len);
                if offset < table_end || end <= offset || end > len {
                    return Err(format!(
                        "shader pack entry {index} has bad bounds {offset:#x}..{end:#x}"
                    ));
                }
                let mut key = [0u8; KEY_BYTES];
                read_exact_at(&file, &mut key, offset)
                    .map_err(|error| format!("{}: {error}", path.display()))?;
                let mut cursor = Cursor { bytes: &key, at: 0 };
                let (hash, types, _) = cursor.key()?;
                entries.insert(
                    (hash, types),
                    Span {
                        offset,
                        len: end - offset,
                    },
                );
            }
        } else {
            // V1 has no offset table: walk it once to recover the spans. It
            // parses each entry to find the next one; a later lookup parses
            // the entry again from the file.
            let bytes =
                std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
            let mut cursor = Cursor {
                bytes: &bytes,
                at: HEADER_BYTES as usize,
            };
            for _ in 0..count {
                let offset = cursor.at as u64;
                let (hash, types, stage) = cursor.key()?;
                cursor.body(stage)?;
                entries.insert(
                    (hash, types),
                    Span {
                        offset,
                        len: cursor.at as u64 - offset,
                    },
                );
            }
        }

        println!(
            "[recomp-vulkan] indexed {} shader pack entries from {} in {:.1} ms",
            entries.len(),
            path.display(),
            started.elapsed().as_secs_f64() * 1000.0
        );
        Ok(Self {
            file,
            path: path.to_path_buf(),
            entries,
        })
    }

    /// The translated program for `hash` and `types`, read from the pack on
    /// first use. `Ok(None)` is a normal miss; `Err` means the pack could not
    /// be read (the caller falls back to live translation).
    pub fn lookup(&self, hash: u64, types: &[u8; 41]) -> Result<Option<TranslatedShader>, String> {
        let Some(span) = self.entries.get(&(hash, *types)).copied() else {
            return Ok(None);
        };
        let len = usize::try_from(span.len).map_err(|_| "shader pack entry is too large")?;
        let mut bytes = vec![0u8; len];
        read_exact_at(&self.file, &mut bytes, span.offset)
            .map_err(|error| format!("{}: {error}", self.path.display()))?;
        let mut cursor = Cursor {
            bytes: &bytes,
            at: 0,
        };
        let (entry_hash, entry_types, stage) = cursor.key()?;
        if entry_hash != hash || entry_types != *types {
            return Err(format!(
                "shader pack entry at {:#x} does not match its index",
                span.offset
            ));
        }
        Ok(Some(cursor.body(stage)?))
    }
}

/// Read `buf` from `offset`. `read_at` is Unix-only on `File`, so Windows
/// needs its own loop.
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        let mut done = 0;
        while done < buf.len() {
            match read_at(file, &mut buf[done..], offset + done as u64)? {
                0 => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "shader pack truncated",
                    ))
                }
                read => done += read,
            }
        }
        Ok(())
    }
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(len).ok_or("shader pack overflow")?;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or("shader pack truncated")?;
        self.at = end;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// Hash, translation environment and stage, the fields a lookup matches.
    fn key(&mut self) -> Result<(u64, [u8; 41], Stage), String> {
        let hash = self.u64()?;
        let stage = match self.byte()? {
            1 => Stage::Vertex,
            5 => Stage::Fragment,
            other => return Err(format!("bad shader stage {other}")),
        };
        let mut types = [0u8; 41];
        types.copy_from_slice(self.take(41)?);
        Ok((hash, types, stage))
    }

    /// Everything after the key.
    fn body(&mut self, stage: Stage) -> Result<TranslatedShader, String> {
        let descriptor_set = self.u32()?;
        let words = self.u32_list()?;
        let uniform_buffers = self.u32_list()?;
        let uniform_extents = self.u32_list()?;
        let texture_count = self.u32()? as usize;
        if texture_count > self.bytes.len().saturating_sub(self.at) / 10 {
            return Err("shader pack truncated".into());
        }
        let mut textures = Vec::with_capacity(texture_count);
        for _ in 0..texture_count {
            let binding = self.u32()?;
            let handle_offset = self.u32()?;
            let kind = match self.byte()? {
                1 => TextureKind::D1,
                2 => TextureKind::D2,
                3 => TextureKind::D2Array,
                4 => TextureKind::D3,
                5 => TextureKind::Cube,
                other => return Err(format!("bad texture kind {other}")),
            };
            let shadow = self.byte()? != 0;
            textures.push(TextureBinding {
                binding,
                handle_offset,
                kind,
                shadow,
            });
        }
        let inputs = self.u32_list()?;
        let outputs = self.u32_list()?;
        let color_outputs = self.u32_list()?;
        Ok(TranslatedShader {
            stage,
            words,
            descriptor_set,
            uniform_buffers,
            uniform_extents,
            textures,
            inputs,
            outputs,
            color_outputs,
        })
    }

    fn u32_list(&mut self) -> Result<Vec<u32>, String> {
        let count = self.u32()? as usize;
        if count > self.bytes.len().saturating_sub(self.at) / 4 {
            return Err("shader pack truncated".into());
        }
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.u32()?);
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Times a load of the external portable pack (not part of the normal
    /// test run): `RECOMP_SHADER_PACK=<file> cargo test ... -- --ignored`.
    #[test]
    #[ignore = "reads the portable shader pack named by RECOMP_SHADER_PACK"]
    fn benchmark_pack_load() {
        let path = std::env::var_os("RECOMP_SHADER_PACK").expect("RECOMP_SHADER_PACK");
        let started = std::time::Instant::now();
        let pack = ShaderPack::load(Path::new(&path)).expect("load shader pack");
        println!("shader pack load: {:?}", started.elapsed());
        let &(hash, types) = pack.entries.keys().next().expect("indexed entries");
        let started = std::time::Instant::now();
        let shader = pack
            .lookup(hash, &types)
            .expect("lookup")
            .expect("indexed entry");
        println!(
            "first shader lookup: {:?} ({} words)",
            started.elapsed(),
            shader.words.len()
        );
        std::hint::black_box(&pack);
        std::thread::sleep(std::time::Duration::from_secs(2));
    }

    /// Reads every indexed entry of the external pack once, to check that a
    /// lazy lookup parses what the offline writer wrote.
    #[test]
    #[ignore = "reads the portable shader pack named by RECOMP_SHADER_PACK"]
    fn verify_pack_entries() {
        let path = std::env::var_os("RECOMP_SHADER_PACK").expect("RECOMP_SHADER_PACK");
        let pack = ShaderPack::load(Path::new(&path)).expect("load shader pack");
        let mut words = 0usize;
        for &(hash, types) in pack.entries.keys() {
            let shader = pack
                .lookup(hash, &types)
                .expect("lookup")
                .expect("indexed entry");
            words += shader.words.len();
        }
        println!(
            "parsed {} shader pack entries, {words} SPIR-V words",
            pack.entries.len()
        );
    }

    fn put_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn put_list(out: &mut Vec<u8>, values: &[u32]) {
        put_u32(out, values.len() as u32);
        for value in values {
            put_u32(out, *value);
        }
    }

    fn entry(hash: u64, stage: u8, types: [u8; 41], words: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&hash.to_le_bytes());
        out.push(stage);
        out.extend_from_slice(&types);
        put_u32(&mut out, 7);
        put_list(&mut out, words);
        put_list(&mut out, &[0]);
        put_list(&mut out, &[16]);
        put_u32(&mut out, 1);
        put_u32(&mut out, 33);
        put_u32(&mut out, 44);
        out.push(2);
        out.push(1);
        put_list(&mut out, &[1]);
        put_list(&mut out, &[2]);
        put_list(&mut out, &[3]);
        out
    }

    fn pack(magic: &[u8; 8], entries: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(magic);
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        if magic == MAGIC_V2 {
            let mut offset = HEADER_BYTES as usize + entries.len() * 8;
            for entry in entries {
                out.extend_from_slice(&(offset as u64).to_le_bytes());
                offset += entry.len();
            }
        }
        for entry in entries {
            out.extend_from_slice(entry);
        }
        out
    }

    fn load_pack(name: &str, bytes: &[u8]) -> (PathBuf, ShaderPack) {
        let path = std::env::temp_dir().join(format!(
            "recomp-shader-pack-{}-{name}.bin",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        let pack = ShaderPack::load(&path).unwrap();
        (path, pack)
    }

    fn check_entries(pack: &ShaderPack) {
        let mut types = [0u8; 41];
        types[40] = 2;
        let shader = pack.lookup(0x1234, &types).unwrap().unwrap();
        assert_eq!(shader.stage, Stage::Fragment);
        assert_eq!(shader.words, [0x0723_0203, 0x0001_000b]);
        assert_eq!(shader.descriptor_set, 7);
        assert_eq!(shader.uniform_buffers, [0]);
        assert_eq!(shader.uniform_extents, [16]);
        assert_eq!(shader.textures.len(), 1);
        assert_eq!(shader.textures[0].binding, 33);
        assert_eq!(shader.textures[0].handle_offset, 44);
        assert_eq!(shader.textures[0].kind, TextureKind::D2);
        assert!(shader.textures[0].shadow);
        assert_eq!(shader.inputs, [1]);
        assert_eq!(shader.outputs, [2]);
        assert_eq!(shader.color_outputs, [3]);
        assert!(pack.lookup(0x9999, &types).unwrap().is_none());
    }

    #[test]
    fn duplicate_keys_keep_the_last_entry() {
        let types = [0u8; 41];
        let bytes = pack(
            MAGIC_V2,
            &[entry(0x1234, 5, types, &[1]), entry(0x1234, 1, types, &[2])],
        );
        let (path, pack) = load_pack("duplicate", &bytes);
        assert_eq!(pack.entries.len(), 1);
        let shader = pack.lookup(0x1234, &types).unwrap().unwrap();
        assert_eq!(shader.stage, Stage::Vertex);
        assert_eq!(shader.words, [2]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn v2_loads_only_an_index_and_parses_on_lookup() {
        let mut types = [0u8; 41];
        types[40] = 2;
        let bytes = pack(
            MAGIC_V2,
            &[entry(0x1234, 5, types, &[0x0723_0203, 0x0001_000b])],
        );
        let (path, pack) = load_pack("v2", &bytes);
        check_entries(&pack);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn v1_still_loads() {
        let mut types = [0u8; 41];
        types[40] = 2;
        let bytes = pack(
            MAGIC_V1,
            &[entry(0x1234, 5, types, &[0x0723_0203, 0x0001_000b])],
        );
        let (path, pack) = load_pack("v1", &bytes);
        check_entries(&pack);
        std::fs::remove_file(path).unwrap();
    }
}
