//! Full listings of captured shader programs (`RECOMP_CAPTURE_SHADERS`):
//! every instruction reachable from the entry, following branches.
//!
//! `cargo run -p recomp-gfx --example shader_listing -- <capture dir>`

use std::collections::{BTreeMap, BTreeSet};

use recomp_gfx::maxwell::sph::{self, ShaderProgramHeader, HEADER_SIZE, INSTRUCTION_SIZE};

fn word_at(bytes: &[u8], offset: u64) -> Option<u64> {
    let at = offset as usize;
    Some(u64::from_le_bytes(
        bytes.get(at..at + INSTRUCTION_SIZE)?.try_into().ok()?,
    ))
}

fn next(offset: u64) -> u64 {
    let mut next = offset + INSTRUCTION_SIZE as u64;
    if sph::is_schedule_word(next) {
        next += INSTRUCTION_SIZE as u64;
    }
    next
}

fn main() {
    let dir = std::env::args().nth(1).expect("capture directory");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("read capture directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "bin"))
        .collect();
    files.sort();
    let mut histogram: BTreeMap<&'static str, usize> = BTreeMap::new();
    for path in files {
        let mut bytes = std::fs::read(&path).expect("read program");
        // Compute programs have no header: list them behind a zeroed one.
        if path
            .file_stem()
            .is_some_and(|stem| stem.to_string_lossy().ends_with("-compute"))
        {
            let mut padded = vec![0u8; HEADER_SIZE];
            padded[0] = 1;
            padded.extend_from_slice(&bytes);
            bytes = padded;
        }
        let Some(header) = ShaderProgramHeader::parse(&bytes) else {
            println!("{}: no header", path.display());
            continue;
        };
        println!("== {} {}", path.display(), header.describe());
        let mut reachable = BTreeSet::new();
        let mut pending = vec![next(HEADER_SIZE as u64 - INSTRUCTION_SIZE as u64)];
        while let Some(offset) = pending.pop() {
            if !reachable.insert(offset) {
                continue;
            }
            let Some(word) = word_at(&bytes, offset) else {
                continue;
            };
            let instruction = sph::decode_at(offset, word);
            let always = instruction.pred == sph::PREDICATE_TRUE && !instruction.pred_inv;
            let target = (offset as i64 + 8 + i64::from(sph::branch_offset(word))) as u64;
            match instruction.name {
                "EXIT" | "KIL" if always => {}
                "BRA" | "SYNC" if always && instruction.name == "BRA" => pending.push(target),
                "SYNC" => {}
                "BRA" => {
                    pending.push(target);
                    pending.push(next(offset));
                }
                "SSY" => {
                    pending.push(target);
                    pending.push(next(offset));
                }
                "unknown" => {}
                _ => pending.push(next(offset)),
            }
        }
        for offset in &reachable {
            let Some(word) = word_at(&bytes, *offset) else {
                continue;
            };
            let instruction = sph::decode_at(*offset, word);
            *histogram.entry(instruction.name).or_default() += 1;
            println!("  {:016x}  {}", word, instruction.format());
        }
    }
    println!("== instruction counts");
    for (name, count) in histogram {
        println!("  {name:8} {count}");
    }
}
