//! Offline translation of the shipped Maxwell shader library.
//!
//! The library (`init` asset `0063.dat`) is a sequence of 256-byte-aligned
//! blocks. Each block starts with the 0x12345678 marker, has 0x30 bytes of
//! block metadata, then a standard 0x50-byte shader program header and its
//! instruction stream at +0x80.
//!
//! `cargo run -p recomp-gfx --example translate_library -- <library> <out-dir>`
//!
//! Writes each translated module as `<index>-<stage>.spv`, plus
//! `manifest.tsv` with the block offset, stage, code size and translation
//! environment. Failures are recorded too, so the missing cases are visible.

use std::io::Write;

use recomp_gfx::maxwell::sph::{self, Stage, HEADER_SIZE};
use recomp_gfx::maxwell::translate::{
    self, AttributeType, Environment, TextureKind, TranslatedShader,
};

const BLOCK_MAGIC: u32 = 0x1234_5678;
const BLOCK_HEADER: usize = 0x30;

fn main() {
    let mut args = std::env::args().skip(1);
    let library = args.next().expect("shader library path");
    let out_dir = args.next().expect("output directory");
    let binary_path = args.next();
    std::fs::create_dir_all(&out_dir).expect("create output directory");

    let bytes = std::fs::read(&library).expect("read shader library");
    let mut blocks = Vec::new();
    let mut at = 0usize;
    while at + 4 <= bytes.len() {
        if u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) == BLOCK_MAGIC {
            blocks.push(at);
        }
        at += 4;
    }
    println!("{} shader blocks", blocks.len());

    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.tsv")).unwrap();
    writeln!(
        manifest,
        "index\toffset\tstage\tcode_bytes\tstatus\tshader_bytes\tspirv_words\tcolors\ttextures\tuniforms\tfile"
    )
    .unwrap();

    let (mut ok, mut failed) = (0, 0);
    let mut pack_entries: Vec<Vec<u8>> = Vec::new();
    for (index, &start) in blocks.iter().enumerate() {
        let end = blocks
            .get(index + 1)
            .copied()
            .unwrap_or(bytes.len())
            .min(bytes.len());
        if start + BLOCK_HEADER + HEADER_SIZE > end {
            continue;
        }
        let program = &bytes[start + BLOCK_HEADER..end];
        let Some(header) = sph::ShaderProgramHeader::parse(program) else {
            writeln!(
                manifest,
                "{index}\t{start:#x}\t?\t{}\tfailed\t0\t0\t0\t0\t0\t",
                end - start
            )
            .unwrap();
            failed += 1;
            continue;
        };
        let stage = header.stage;
        let mut environment = Environment::default();
        // The library stores the shaders, not the draws that use them. Start
        // with the common float/float environment; the manifest records it so
        // other variants can be added. `SHADER_COLOR_TYPE=uint|int` builds a
        // variant for integer render targets, which changes the SPIR-V.
        environment.clip_distances = 0;
        if let Ok(kind) = std::env::var("SHADER_COLOR_TYPE") {
            let kind = match kind.as_str() {
                "uint" => AttributeType::UnsignedInt,
                "int" => AttributeType::SignedInt,
                _ => AttributeType::Float,
            };
            environment.color_targets = [kind; 8];
        }
        match translate::translate(program, &environment) {
            Ok(shader) => {
                ok += 1;
                let file = format!("{index:04}-{stage:?}.spv");
                write_spirv(&format!("{out_dir}/{file}"), &shader);
                writeln!(
                    manifest,
                    "{index}\t{start:#x}\t{stage:?}\t{}\tok\t{}\t{}\t{:?}\t{}\t{}\t{}",
                    end - start,
                    shader.words.len(),
                    shader.words.len(),
                    shader.color_outputs,
                    shader.textures.len(),
                    shader.uniform_buffers.len(),
                    file
                )
                .unwrap();
                pack_entries.push(pack_entry(
                    &program[..program.len().min(0x200)],
                    stage,
                    &environment,
                    &shader,
                ));
                if index < 3 || index % 500 == 0 {
                    println!("{index} {stage:?}: {} words", shader.words.len());
                }
            }
            Err(error) => {
                failed += 1;
                writeln!(
                    manifest,
                    "{index}\t{start:#x}\t{stage:?}\t{}\tfailed: {error}\t0\t0\t0\t0\t0\t",
                    end - start
                )
                .unwrap();
                if failed <= 20 {
                    println!("{index} {stage:?}: {error}");
                }
            }
        }
    }
    if let Some(path) = binary_path {
        // V2 starts with an offset table: the runtime reads only this table
        // and each entry's 50-byte key at startup, then parses an entry the
        // first time the game uses it. Keep the offsets increasing.
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"LM3SPV02").unwrap();
        file.write_all(&(pack_entries.len() as u32).to_le_bytes())
            .unwrap();
        let mut offset = 8 + 4 + pack_entries.len() * 8;
        for entry in &pack_entries {
            file.write_all(&(offset as u64).to_le_bytes()).unwrap();
            offset += entry.len();
        }
        for entry in &pack_entries {
            file.write_all(entry).unwrap();
        }
        println!("{ok} shader pack entries written to {path}");
    }
    println!("{ok} translated, {failed} failed");
}

fn code_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

fn environment_types(environment: &Environment) -> [u8; 41] {
    let mut types = [0u8; 41];
    let code = |kind: AttributeType| match kind {
        AttributeType::Float => 0,
        AttributeType::SignedInt => 1,
        AttributeType::UnsignedInt => 2,
    };
    for (index, kind) in environment.vertex_attributes.iter().enumerate() {
        types[index] = code(*kind);
    }
    for (index, kind) in environment.color_targets.iter().enumerate() {
        types[32 + index] = code(*kind);
    }
    types[40] = environment.clip_distances;
    types
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_list(out: &mut Vec<u8>, values: &[u32]) {
    put_u32(out, values.len() as u32);
    for value in values {
        put_u32(out, *value);
    }
}

fn pack_entry(
    program: &[u8],
    stage: Stage,
    environment: &Environment,
    shader: &TranslatedShader,
) -> Vec<u8> {
    let mut out = Vec::new();
    put_u64(&mut out, code_hash(program));
    let stage = match stage {
        Stage::Vertex => 1,
        Stage::Fragment => 5,
        _ => 0,
    };
    out.push(stage);
    out.extend_from_slice(&environment_types(environment));
    put_u32(&mut out, shader.descriptor_set);
    put_list(&mut out, &shader.words);
    put_list(&mut out, &shader.uniform_buffers);
    put_list(&mut out, &shader.uniform_extents);
    put_u32(&mut out, shader.textures.len() as u32);
    for texture in &shader.textures {
        put_u32(&mut out, texture.binding);
        put_u32(&mut out, texture.handle_offset);
        out.push(match texture.kind {
            TextureKind::D1 => 1,
            TextureKind::D2 => 2,
            TextureKind::D2Array => 3,
            TextureKind::D3 => 4,
            TextureKind::Cube => 5,
        });
        out.push(u8::from(texture.shadow));
    }
    put_list(&mut out, &shader.inputs);
    put_list(&mut out, &shader.outputs);
    put_list(&mut out, &shader.color_outputs);
    out
}

fn write_spirv(path: &str, shader: &TranslatedShader) {
    let mut file = std::fs::File::create(path).unwrap();
    for word in &shader.words {
        file.write_all(&word.to_le_bytes()).unwrap();
    }
}
