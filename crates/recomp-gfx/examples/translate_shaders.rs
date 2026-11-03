//! Translate captured shader programs (`RECOMP_CAPTURE_SHADERS`) to SPIR-V
//! and check each module with `spirv-val` (Vulkan 1.1 rules) when it is
//! installed. Modules are written next to the captures as `<name>.spv`.
//!
//! `cargo run -p recomp-gfx --example translate_shaders -- <capture dir>`
//!
//! Compute programs (`<address>-compute.bin`, no header) are translated for
//! an 8 x 8 block with 16 KiB of shared memory. A program's constant buffer
//! 1, when captured (`<name>.cb1`), is what its `BRX` jump tables are read
//! from.

use recomp_gfx::maxwell::translate::{translate, translate_compute, Environment};

fn main() {
    let dir = std::env::args().nth(1).expect("capture directory");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("read capture directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "bin"))
        .collect();
    files.sort();
    let (mut translated, mut valid) = (0, 0);
    for path in &files {
        let bytes = std::fs::read(path).expect("read program");
        let name = path.file_stem().unwrap_or_default().to_string_lossy();
        let constant_buffer_1 = std::fs::read(path.with_extension("cb1")).unwrap_or_default();
        let result = if name.ends_with("-compute") {
            let environment = Environment {
                local_size: [8, 8, 1],
                shared_memory_bytes: 0x4000,
                constant_buffer_1,
                ..Environment::default()
            };
            translate_compute(&bytes, &environment)
        } else {
            let environment = Environment {
                constant_buffer_1,
                ..Environment::default()
            };
            translate(&bytes, &environment)
        };
        let shader = match result {
            Ok(shader) => shader,
            Err(error) => {
                println!("{name}: {error}");
                continue;
            }
        };
        translated += 1;
        let spv = path.with_extension("spv");
        let bytes: Vec<u8> = shader
            .words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        std::fs::write(&spv, bytes).expect("write module");
        let check = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.1"])
            .arg(&spv)
            .output();
        let verdict = match check {
            Ok(output) if output.status.success() => {
                valid += 1;
                "valid".to_string()
            }
            Ok(output) => String::from_utf8_lossy(&output.stderr)
                .lines()
                .take(3)
                .collect::<Vec<_>>()
                .join(" | "),
            Err(_) => "spirv-val not run".to_string(),
        };
        println!(
            "{name}: {:?} {} words, cbufs {:?}, textures {}, inputs {:?}, outputs {:?}, colors {:?}: {verdict}",
            shader.stage,
            shader.words.len(),
            shader.uniform_buffers,
            shader.textures.len(),
            shader.inputs,
            shader.outputs,
            shader.color_outputs,
        );
    }
    println!("{translated}/{} translated, {valid} valid", files.len());
}
