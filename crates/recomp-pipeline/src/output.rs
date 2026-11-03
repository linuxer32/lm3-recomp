use crate::memory::{MemoryImageDescriptor, MemoryLayoutDescriptor};
use crate::pipeline::{
    ensure_dir, FunctionBody, MemoryImageSource, RustFunction, RustProgram, RustTerminator,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

const BUILD_MANIFEST_SELF_PATH: &str = "manifest.json";
const BUILD_MANIFEST_SELF_SHA_PLACEHOLDER: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Serialize, Clone)]
pub struct BuildManifest {
    pub title: String,
    pub abi_version: String,
    pub module_sha256: String,
    pub config_sha256: String,
    pub provenance_sha256: String,
    pub inputs: Vec<InputSummary>,
    pub memory_layout: MemoryLayoutDescriptor,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_image: Option<MemoryImageDescriptor>,
    pub manifest_self_hash_basis: String,
    pub generated_files: Vec<GeneratedFile>,
}

#[derive(Debug, Serialize, Clone)]
pub struct InputSummary {
    pub path: PathBuf,
    pub format: String,
    pub sha256: String,
    pub size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct GeneratedFile {
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

pub const CHUNK_SIZE: usize = 2000;
/// Upper bound on generated source per chunk crate. rustc's peak memory
/// grows with crate size, so retail-sized programs are split by bytes rather
/// than by function count.
pub const CHUNK_BYTES: usize = 4 << 20;
/// Chunk boundaries are content-defined so that a change to one function
/// rebuilds only its chunk instead of shifting every later function into a
/// different crate: once a chunk holds CHUNK_MIN_BYTES, it ends before the
/// next function whose name hashes to a cut point (one in
/// CHUNK_CUT_ODDS), or before a function that would push it past
/// CHUNK_BYTES.
pub const CHUNK_MIN_BYTES: usize = 2 << 20;
const CHUNK_CUT_ODDS: u64 = 64;

/// FNV-1a: a stable hash of a function's name for chunk cut points.
fn is_chunk_cut_point(name: &str) -> bool {
    let hash = name.bytes().fold(0xCBF2_9CE4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
    });
    hash % CHUNK_CUT_ODDS == 0
}

/// Whether a function of `code_len` bytes named `name` starts a new chunk
/// after a chunk of `chunk_len` bytes.
fn starts_new_chunk(chunk_len: usize, code_len: usize, name: &str) -> bool {
    let full = chunk_len + code_len > CHUNK_BYTES;
    let cut = chunk_len >= CHUNK_MIN_BYTES && is_chunk_cut_point(name);
    chunk_len > CHUNK_HEADER.len() && (full || cut)
}

const CHUNK_HEADER: &str = "#![allow(warnings)]\n\n";

/// The generated program is a Windows GUI executable when built with the
/// `windows-subsystem` feature (the release game build: no console window
/// beside the game). Debug builds keep stdout/stderr, and the attribute does
/// nothing off Windows.
const WINDOWS_SUBSYSTEM_ATTRIBUTE: &str = "#![cfg_attr(all(windows, feature = \"windows-subsystem\"), windows_subsystem = \"windows\")]\n";

/// The opt-in feature behind [`WINDOWS_SUBSYSTEM_ATTRIBUTE`]; the generated
/// package declares it in `Cargo.toml`.
const WINDOWS_SUBSYSTEM_FEATURE: &str = "[features]\ndefault = []\nwindows-subsystem = []\n\n";

/// Per-title code generation switches that do not change guest semantics.
#[derive(Debug, Clone, Copy)]
struct CodegenOptions {
    /// Record each block's address in TLS (`recomp_abi::enter_block`).
    block_tracking: bool,
    /// Report loop backedges to the runtime (`recomp_abi::backedge`).
    loop_backedges: bool,
    /// Access guest registers through `(*ctx)` fields instead of function
    /// locals (requires the guest-context ABI).
    direct_registers: bool,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            block_tracking: true,
            loop_backedges: true,
            direct_registers: false,
        }
    }
}

impl CodegenOptions {
    fn for_program(program: &RustProgram) -> Self {
        Self {
            block_tracking: program.block_tracking,
            // Back edges let serialized guest threads switch at loops that
            // make no calls. In parallel mode `loop_point` only counts
            // progress for the 5-second hang report, so a title that gives
            // up per-block tracking may give that up too.
            loop_backedges: program.block_tracking || !program.parallel,
            direct_registers: program.guest_context && program.direct_registers,
        }
    }
}

pub fn emit_project(
    out_dir: &Path,
    runtime_rel: &Path,
    program: &RustProgram,
    manifest: &BuildManifest,
    memory_sources: &[MemoryImageSource],
) -> Result<(Vec<PathBuf>, BuildManifest), String> {
    ensure_dir(out_dir).map_err(|err| err.to_string())?;

    let mut written = Vec::new();
    let mut generated_files = Vec::new();
    let clobbers = ClobberAnalysis::from_program(program);

    if program.functions.len() > CHUNK_SIZE {
        let chunks_dir = out_dir.join("chunks");
        ensure_dir(&chunks_dir).map_err(|err| err.to_string())?;
        let codegen = CodegenOptions::for_program(program);

        // Stream functions into chunks so neither this process nor rustc
        // ever holds more than one chunk's source at a time.
        let mut chunk_of = Vec::with_capacity(program.functions.len());
        let mut num_chunks = 0;
        let mut large_chunks = Vec::new();
        let mut lib_rs = String::from(CHUNK_HEADER);
        let functions = function_names(program);
        let mut calls = ChunkCalls::default();
        for (index, function) in program.functions.iter().enumerate() {
            // Its callees go with the chunk it lands in, decided next.
            let mut called = std::collections::BTreeSet::new();
            let code = direct_calls(
                &emit_function(function, true, &clobbers, program.guest_context, codegen),
                &functions,
                &mut called,
                program.guest_context,
            );
            if starts_new_chunk(lib_rs.len(), code.len(), &function.name) {
                if lib_rs.len() > LARGE_CHUNK_BYTES {
                    large_chunks.push(num_chunks);
                }
                write_chunk(
                    &chunks_dir,
                    num_chunks,
                    runtime_rel,
                    &calls.with_externs(&lib_rs, program.guest_context),
                    &mut written,
                    &mut generated_files,
                )?;
                num_chunks += 1;
                lib_rs = String::from(CHUNK_HEADER);
                calls = ChunkCalls::default();
            }
            lib_rs.push_str(&code);
            lib_rs.push('\n');
            calls.defined.insert(function.name.clone());
            calls.called.append(&mut called);
            chunk_of.push(num_chunks);
            if index + 1 == program.functions.len() {
                if lib_rs.len() > LARGE_CHUNK_BYTES {
                    large_chunks.push(num_chunks);
                }
                write_chunk(
                    &chunks_dir,
                    num_chunks,
                    runtime_rel,
                    &calls.with_externs(&lib_rs, program.guest_context),
                    &mut written,
                    &mut generated_files,
                )?;
                num_chunks += 1;
            }
        }
        remove_stale_chunks(&chunks_dir, num_chunks)?;

        let cargo_toml = emit_workspace_cargo_toml(program, runtime_rel, num_chunks, &large_chunks);
        let cargo_path = out_dir.join("Cargo.toml");
        write_if_changed(&cargo_path, &cargo_toml).map_err(|err| err.to_string())?;
        written.push(cargo_path);
        generated_files.push(GeneratedFile {
            path: "Cargo.toml".to_string(),
            sha256: sha256_bytes(cargo_toml.as_bytes()),
            size: cargo_toml.len() as u64,
        });

        write_cargo_config(out_dir, program, &mut written, &mut generated_files)?;

        let src_dir = out_dir.join("src");
        ensure_dir(&src_dir).map_err(|err| err.to_string())?;
        let main_rs = emit_main_rs_chunked(program, &chunk_of);
        let main_path = src_dir.join("main.rs");
        write_if_changed(&main_path, &main_rs).map_err(|err| err.to_string())?;
        written.push(main_path);
        generated_files.push(GeneratedFile {
            path: "src/main.rs".to_string(),
            sha256: sha256_bytes(main_rs.as_bytes()),
            size: main_rs.len() as u64,
        });
    } else {
        let cargo_toml = emit_cargo_toml(program, runtime_rel);
        let cargo_path = out_dir.join("Cargo.toml");
        write_if_changed(&cargo_path, &cargo_toml).map_err(|err| err.to_string())?;
        written.push(cargo_path);
        generated_files.push(GeneratedFile {
            path: "Cargo.toml".to_string(),
            sha256: sha256_bytes(cargo_toml.as_bytes()),
            size: cargo_toml.len() as u64,
        });
        write_cargo_config(out_dir, program, &mut written, &mut generated_files)?;

        let src_dir = out_dir.join("src");
        ensure_dir(&src_dir).map_err(|err| err.to_string())?;
        let main_rs = emit_main_rs(program, &clobbers);
        let main_path = src_dir.join("main.rs");
        write_if_changed(&main_path, &main_rs).map_err(|err| err.to_string())?;
        written.push(main_path);
        generated_files.push(GeneratedFile {
            path: "src/main.rs".to_string(),
            sha256: sha256_bytes(main_rs.as_bytes()),
            size: main_rs.len() as u64,
        });
    }

    if !memory_sources.is_empty() {
        for source in memory_sources {
            let dest_path = out_dir.join(&source.dest_path);
            if let Some(parent) = dest_path.parent() {
                ensure_dir(parent).map_err(|err| err.to_string())?;
            }
            let bytes = fs::read(&source.source_path).map_err(|err| err.to_string())?;
            fs::write(&dest_path, &bytes).map_err(|err| err.to_string())?;
            written.push(dest_path.clone());
            generated_files.push(GeneratedFile {
                path: source.dest_path.to_string_lossy().to_string(),
                sha256: sha256_bytes(&bytes),
                size: bytes.len() as u64,
            });
        }
    }

    let manifest_path = out_dir.join("manifest.json");
    let mut updated_manifest = manifest.clone();
    updated_manifest.generated_files = generated_files;
    let (updated_manifest, manifest_json) = build_manifest_json(updated_manifest)?;
    fs::write(&manifest_path, manifest_json).map_err(|err| err.to_string())?;
    written.push(manifest_path);

    Ok((written, updated_manifest))
}

fn build_manifest_json(mut manifest: BuildManifest) -> Result<(BuildManifest, String), String> {
    if manifest
        .generated_files
        .iter()
        .any(|file| file.path == BUILD_MANIFEST_SELF_PATH)
    {
        return Err("build manifest already present in generated files".to_string());
    }

    manifest.manifest_self_hash_basis = "generated_files_self_placeholder".to_string();
    manifest.generated_files.push(GeneratedFile {
        path: BUILD_MANIFEST_SELF_PATH.to_string(),
        sha256: BUILD_MANIFEST_SELF_SHA_PLACEHOLDER.to_string(),
        size: 0,
    });
    manifest.generated_files.sort_by(|a, b| a.path.cmp(&b.path));

    let size = stabilize_manifest_size(&mut manifest)?;
    let self_hash = manifest_self_hash(&manifest)?;
    let self_entry = find_manifest_self_entry_mut(&mut manifest)?;
    self_entry.size = size;
    self_entry.sha256 = self_hash;

    let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|err| err.to_string())?;
    let final_size = manifest_json.len() as u64;
    if final_size != size {
        return Err(format!(
            "build manifest size mismatch: expected {size}, got {final_size}"
        ));
    }

    Ok((manifest, manifest_json))
}

fn manifest_self_hash(manifest: &BuildManifest) -> Result<String, String> {
    let mut normalized = manifest.clone();
    let self_entry = find_manifest_self_entry_mut(&mut normalized)?;
    self_entry.sha256 = BUILD_MANIFEST_SELF_SHA_PLACEHOLDER.to_string();
    let manifest_json = serde_json::to_string_pretty(&normalized).map_err(|err| err.to_string())?;
    Ok(sha256_bytes(manifest_json.as_bytes()))
}

fn stabilize_manifest_size(manifest: &mut BuildManifest) -> Result<u64, String> {
    let mut size = 0_u64;
    for _ in 0..5 {
        let self_entry = find_manifest_self_entry_mut(manifest)?;
        self_entry.sha256 = BUILD_MANIFEST_SELF_SHA_PLACEHOLDER.to_string();
        self_entry.size = size;
        let manifest_json =
            serde_json::to_string_pretty(manifest).map_err(|err| err.to_string())?;
        let new_size = manifest_json.len() as u64;
        if new_size == size {
            return Ok(size);
        }
        size = new_size;
    }
    Err("build manifest size did not stabilize".to_string())
}

fn find_manifest_self_entry_mut(
    manifest: &mut BuildManifest,
) -> Result<&mut GeneratedFile, String> {
    manifest
        .generated_files
        .iter_mut()
        .find(|entry| entry.path == BUILD_MANIFEST_SELF_PATH)
        .ok_or_else(|| "build manifest self entry not found".to_string())
}

fn emit_cargo_toml(program: &RustProgram, runtime_rel: &Path) -> String {
    format!(
        "[package]\nname = \"recomp-{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nrecomp-runtime = {{ path = {runtime} }}\nrecomp-abi = {{ path = {abi} }}\n{graphics}\n# Lifted titles can exceed a gigabyte of generated Rust. Incremental\n# compilation and full debuginfo make rustc's memory use unmanageable\n# (SIGSEGV on large crates), so emitted projects build without them.\n{features}{DEV_PROFILE}{SIMD_PROFILE}",
        name = sanitize_name(&program.title),
        runtime = toml_path(runtime_rel),
        abi = toml_path(&abi_path(runtime_rel)),
        graphics = graphics_dependency(program, runtime_rel),
        features = WINDOWS_SUBSYSTEM_FEATURE
    )
}

/// A path as a TOML basic string in a Cargo manifest.
///
/// Cargo accepts `/` separators on every host, so a Windows path is emitted
/// with forward slashes: a backslash in a TOML string would otherwise start
/// an escape (`\S`, `\c`, ...) and Cargo would reject the manifest. Quotes
/// and control characters are escaped as TOML requires.
fn toml_path(path: &Path) -> String {
    toml_string(&forward_slash(&path.to_string_lossy()))
}

/// A value as a TOML basic string.
fn toml_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04X}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn forward_slash(value: &str) -> String {
    value.replace('\\', "/")
}

/// The dependency line for the title's graphics backend (empty for the
/// recording backend). Backend crates sit next to `recomp-runtime`.
fn graphics_dependency(program: &RustProgram, runtime_rel: &Path) -> String {
    match program.graphics {
        crate::config::GraphicsBackend::Recording => String::new(),
        crate::config::GraphicsBackend::Vulkan => format!(
            "recomp-gfx-vulkan = {{ path = {} }}\n",
            toml_path(
                &runtime_rel
                    .parent()
                    .unwrap_or_else(|| Path::new(""))
                    .join("recomp-gfx-vulkan")
            )
        ),
    }
}

/// The title's runtime defaults, first thing in `main`: parallel guest
/// threads when the title asks for them (`RECOMP_PARALLEL=0` still
/// serializes them).
fn emit_runtime_defaults(out: &mut String, program: &RustProgram) {
    // The title's script, so the program runs with it however it is
    // started (`RECOMP_SCRIPT` still picks another; an empty one none). A
    // copy beside the executable wins over the path recorded at build time.
    if let Some(script) = &program.script {
        let relative = match &script.relative {
            Some(relative) => format!("Some({})", crate::pipeline::rust_string_literal(relative)),
            None => "None".to_string(),
        };
        out.push_str("    if std::env::var_os(\"RECOMP_SCRIPT\").is_none() {\n");
        out.push_str(&format!(
            "        if let Some(path) = recomp_runtime::resolve_asset(\"RECOMP_SCRIPT\", {relative}, Some({})) {{\n",
            crate::pipeline::rust_string_literal(&script.fallback)
        ));
        out.push_str("            if path.exists() {\n");
        out.push_str("                std::env::set_var(\"RECOMP_SCRIPT\", path);\n");
        out.push_str("            }\n");
        out.push_str("        }\n");
        out.push_str("    }\n");
    }
    if !program.parallel {
        return;
    }
    out.push_str("    match std::env::var(\"RECOMP_PARALLEL\") {\n");
    out.push_str(
        "        Ok(value) if value == \"0\" => std::env::remove_var(\"RECOMP_PARALLEL\"),\n",
    );
    out.push_str("        Ok(_) => {}\n");
    out.push_str("        Err(_) => std::env::set_var(\"RECOMP_PARALLEL\", \"1\"),\n");
    out.push_str("    }\n");
}

/// Installs the title's graphics backend before the guest starts: the
/// Vulkan window opens unless the title runs with `RECOMP_VULKAN=0` (the
/// recording backend stays in use then, or if the window cannot open).
fn emit_graphics_init(out: &mut String, program: &RustProgram) {
    if program.graphics != crate::config::GraphicsBackend::Vulkan {
        return;
    }
    // The window opens unless RECOMP_VULKAN=0.
    out.push_str(
        "    if std::env::var(\"RECOMP_VULKAN\").map_or(true, |value| value != \"0\") {\n",
    );
    out.push_str("        match recomp_gfx_vulkan::install() {\n");
    out.push_str("            Ok(()) => println!(\"[recomp-vulkan] window backend installed\"),\n");
    out.push_str("            Err(error) => println!(\"[recomp-vulkan] install failed: {error}; the recording backend stays in use\"),\n");
    out.push_str("        }\n");
    out.push_str("    }\n");
}

/// Attach IPC/fs with the title's RomFS image. `RECOMP_ROMFS` picks another
/// image (an empty one disables it); otherwise a copy beside the executable
/// wins over the path recorded at build time. A project without
/// `[runtime.ipc]` attaches the hub only when an image is supplied, so its
/// behavior is unchanged without the variable.
fn emit_ipc_init(out: &mut String, program: &RustProgram) {
    let asset = program
        .ipc
        .as_ref()
        .and_then(|ipc| ipc.romfs_image.as_ref());
    let relative = asset.and_then(|asset| asset.relative.as_ref()).map_or_else(
        || "None".to_string(),
        |relative| format!("Some({})", crate::pipeline::rust_string_literal(relative)),
    );
    let fallback = asset.map_or_else(
        || "None".to_string(),
        |asset| {
            format!(
                "Some({})",
                crate::pipeline::rust_string_literal(&asset.fallback)
            )
        },
    );
    out.push_str(&format!(
        "    let romfs_image = recomp_runtime::resolve_asset(\"RECOMP_ROMFS\", {relative}, {fallback});\n"
    ));
    match &program.ipc {
        Some(ipc) => {
            let ipc_buffer = match ipc.ipc_buffer {
                Some(addr) => format!("Some({addr:#x})"),
                None => "None".to_string(),
            };
            out.push_str(&format!(
                "    recomp_runtime::init_ipc_services({:#x}, {:#x}, romfs_image.as_deref(), {ipc_buffer})?;\n",
                ipc.romfs_base, ipc.romfs_size
            ));
        }
        None => {
            out.push_str("    if romfs_image.is_some() {\n");
            out.push_str(
                "        recomp_runtime::init_ipc_services(0, 0, romfs_image.as_deref(), None)?;\n",
            );
            out.push_str("    }\n");
        }
    }
}

/// `recomp-abi` sits next to `recomp-runtime` in the toolchain checkout.
fn abi_path(runtime_rel: &Path) -> PathBuf {
    runtime_rel
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("recomp-abi")
}

fn emit_workspace_cargo_toml(
    program: &RustProgram,
    runtime_rel: &Path,
    num_chunks: usize,
    large_chunks: &[usize],
) -> String {
    let mut out = format!(
        "[package]\nname = \"recomp-{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nrecomp-runtime = {{ path = {runtime} }}\nrecomp-abi = {{ path = {abi} }}\n",
        name = sanitize_name(&program.title),
        runtime = toml_path(runtime_rel),
        abi = toml_path(&abi_path(runtime_rel))
    );
    out.push_str(&graphics_dependency(program, runtime_rel));
    for i in 0..num_chunks {
        out.push_str(&format!("chunk_{i} = {{ path = \"chunks/chunk_{i}\" }}\n"));
    }
    out.push_str("\n[features]\ndefault = []\nwindows-subsystem = []\n");
    out.push_str("\n[workspace]\nmembers = [\n    \".\",\n");
    for i in 0..num_chunks {
        out.push_str(&format!("    \"chunks/chunk_{i}\",\n"));
    }
    out.push_str("]\n\n");
    out.push_str(DEV_PROFILE);
    out.push_str(SIMD_PROFILE);
    for chunk in large_chunks {
        out.push_str(&format!(
            "\n[profile.dev.package.chunk_{chunk}]\nopt-level = 1\n"
        ));
    }
    out
}

/// First line of a `.cargo/config.toml` the pipeline wrote (and may remove).
const CARGO_CONFIG_HEADER: &str =
    "# Generated by recomp-pipeline from the title's [runtime] target_cpu.";

/// The CPU rustc should compile the generated program for. An explicit CPU is
/// kept; `native` only means the build host when no cross target was set, or
/// when the target keeps the host's x86-64 instruction set. Every other
/// cross target gets `generic`, which on aarch64 still includes Advanced SIMD
/// (`neon` in rustc's target features), so lifted SIMD code stays exact and
/// does not assume features the target device may lack.
fn resolved_target_cpu(target: Option<&str>, cpu: &str) -> String {
    if cpu != "native" {
        return cpu.to_string();
    }
    let Some(target) = target else {
        return "native".to_string();
    };
    let arch = target.split('-').next().unwrap_or("");
    match arch {
        "x86_64" | "amd64" if cfg!(target_arch = "x86_64") => "native".to_string(),
        _ => "generic".to_string(),
    }
}

/// `.cargo/config.toml` selecting the generated program's target and the CPU
/// it is compiled for, or none. Generated code keeps Arm's results either
/// way: Rust never fuses a multiply and an add on its own, so FMA only speeds
/// up the `mul_add`s lifted code asks for (a libm call without it).
fn write_cargo_config(
    out_dir: &Path,
    program: &RustProgram,
    written: &mut Vec<PathBuf>,
    generated_files: &mut Vec<GeneratedFile>,
) -> Result<(), String> {
    let path = out_dir.join(".cargo").join("config.toml");
    let target = program.target.as_deref();
    let cpu = program
        .target_cpu
        .as_deref()
        .map(|cpu| resolved_target_cpu(target, cpu));
    if target.is_none() && cpu.is_none() {
        // Drop a config this pipeline wrote earlier, never a hand-made one.
        if std::fs::read_to_string(&path).is_ok_and(|text| text.starts_with(CARGO_CONFIG_HEADER)) {
            std::fs::remove_file(&path).map_err(|err| err.to_string())?;
        }
        return Ok(());
    }
    let mut text = format!("{CARGO_CONFIG_HEADER}\n");
    if let Some(target) = target {
        text.push_str(&format!("\n[build]\ntarget = \"{target}\"\n"));
    }
    if let Some(cpu) = cpu {
        if let Some(target) = target {
            // Flags under [build] would also apply to build-host artifacts
            // and, more importantly, would use the host CPU's features when
            // cross-compiling. Target-specific flags stay with the target.
            text.push_str(&format!(
                "\n[target.\"{target}\"]\nrustflags = [\"-C\", \"target-cpu={cpu}\"]\n"
            ));
        } else {
            text.push_str(&format!(
                "\n[build]\nrustflags = [\"-C\", \"target-cpu={cpu}\"]\n"
            ));
        }
    }
    ensure_dir(&out_dir.join(".cargo")).map_err(|err| err.to_string())?;
    write_if_changed(&path, &text).map_err(|err| err.to_string())?;
    written.push(path);
    generated_files.push(GeneratedFile {
        path: ".cargo/config.toml".to_string(),
        sha256: sha256_bytes(text.as_bytes()),
        size: text.len() as u64,
    });
    Ok(())
}

/// Without debuginfo or incremental state (both blow up on gigabytes of
/// generated code). At -O0 a large title's machine code passes the 2 GiB
/// that x86-64 PC-relative relocations reach and the link fails; -O1 is
/// about 2.6 times smaller and runs far faster, for about twice the
/// compile time, and -O2 (which threads more of the jumps between a
/// function's blocks) costs about 20% more than -O1 for a typical chunk.
/// Lifted code wraps its arithmetic explicitly, so overflow checks and
/// debug assertions only cost time. Lifted code reports faults through
/// `Result`, so unwinding tables would only add size.
const DEV_PROFILE: &str = "[profile.dev]\ndebug = false\nincremental = false\nopt-level = 2\noverflow-checks = false\ndebug-assertions = false\npanic = \"abort\"\n";

/// Chunks holding a single huge function compile for many minutes at -O1
/// (LM3 has a 30 MB one, the critical path of a full build); they stay at
/// -O1 rather than risk the slower -O2 passes.
const LARGE_CHUNK_BYTES: usize = 2 * CHUNK_BYTES;

/// Every lifted floating-point and SIMD instruction runs through the
/// `recomp-simd` evaluator, so it is optimized even in dev builds.
const SIMD_PROFILE: &str = "\n[profile.dev.package.recomp-simd]\nopt-level = 3\n";

/// The program's functions by guest address.
fn function_names(program: &RustProgram) -> std::collections::HashMap<u64, String> {
    program
        .functions
        .iter()
        .filter_map(|function| Some((function_address(&function.name)?, function.name.clone())))
        .collect()
}

/// Calls to a guest address the program has a function for go straight to
/// that function (`recomp_abi::call`, which still goes through `dispatch`
/// when the runtime needs to see calls) instead of looking the address up:
/// the lookup's table is several MB for a large title, and the lookups were
/// a twentieth of LM3's busiest threads. `called` collects the functions
/// named.
fn direct_calls(
    code: &str,
    functions: &std::collections::HashMap<u64, String>,
    called: &mut std::collections::BTreeSet<String>,
    guest_context: bool,
) -> String {
    const DISPATCH: &str = "recomp_abi::dispatch(0x";
    let mut out = String::with_capacity(code.len());
    let mut rest = code;
    while let Some(at) = rest.find(DISPATCH) {
        out.push_str(&rest[..at]);
        let after = &rest[at + DISPATCH.len()..];
        let digits = after
            .find(|c: char| !c.is_ascii_hexdigit())
            .unwrap_or(after.len());
        let function = u64::from_str_radix(&after[..digits], 16)
            .ok()
            .filter(|_| after[digits..].starts_with(')'))
            .and_then(|address| Some((address, functions.get(&address)?)));
        match function {
            Some((address, name)) => {
                let arguments = if guest_context { "(ctx)" } else { "()" };
                out.push_str(&format!(
                    "recomp_abi::call({address:#x}, || unsafe {{ {name}{arguments} }})"
                ));
                called.insert(name.clone());
                rest = &after[digits + 1..];
            }
            None => {
                out.push_str(DISPATCH);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// The functions a chunk defines and calls directly.
#[derive(Default)]
struct ChunkCalls {
    defined: std::collections::HashSet<String>,
    called: std::collections::BTreeSet<String>,
}

impl ChunkCalls {
    /// The chunk's source with declarations of the functions it calls in
    /// other chunks, which the linker resolves by their unmangled names
    /// (chunks cannot depend on each other: they call in cycles).
    fn with_externs(&self, lib_rs: &str, guest_context: bool) -> String {
        let external: Vec<&String> = self
            .called
            .iter()
            .filter(|name| !self.defined.contains(*name))
            .collect();
        if external.is_empty() {
            return lib_rs.to_string();
        }
        let mut out = String::from(CHUNK_HEADER);
        out.push_str("extern \"Rust\" {\n");
        for name in external {
            if guest_context {
                out.push_str(&format!(
                    "    fn {name}(ctx: *mut recomp_abi::GuestContext) -> recomp_abi::GuestResult;\n"
                ));
            } else {
                out.push_str(&format!("    fn {name}() -> recomp_abi::GuestResult;\n"));
            }
        }
        out.push_str("}\n\n");
        out.push_str(&lib_rs[CHUNK_HEADER.len()..]);
        out
    }
}

/// Chunk crates depend only on `recomp-abi`, so runtime changes relink the
/// program without recompiling generated code.
fn emit_chunk_cargo_toml(chunk_name: &str, runtime_rel: &Path) -> String {
    let abi = abi_path(runtime_rel);
    let chunk_abi = if abi.is_absolute() {
        toml_path(&abi)
    } else {
        toml_string(&format!("../../{}", forward_slash(&abi.to_string_lossy())))
    };
    format!(
        "[package]\nname = \"{chunk_name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nrecomp-abi = {{ path = {chunk_abi} }}\n"
    )
}

fn write_chunk(
    chunks_dir: &Path,
    index: usize,
    runtime_rel: &Path,
    lib_rs: &str,
    written: &mut Vec<PathBuf>,
    generated_files: &mut Vec<GeneratedFile>,
) -> Result<(), String> {
    let chunk_name = format!("chunk_{index}");
    let chunk_dir = chunks_dir.join(&chunk_name);
    let chunk_src_dir = chunk_dir.join("src");
    ensure_dir(&chunk_src_dir).map_err(|err| err.to_string())?;

    let chunk_cargo_toml = emit_chunk_cargo_toml(&chunk_name, runtime_rel);
    let chunk_cargo_path = chunk_dir.join("Cargo.toml");
    write_if_changed(&chunk_cargo_path, &chunk_cargo_toml).map_err(|err| err.to_string())?;
    written.push(chunk_cargo_path);
    generated_files.push(GeneratedFile {
        path: format!("chunks/{chunk_name}/Cargo.toml"),
        sha256: sha256_bytes(chunk_cargo_toml.as_bytes()),
        size: chunk_cargo_toml.len() as u64,
    });

    let lib_path = chunk_src_dir.join("lib.rs");
    write_if_changed(&lib_path, lib_rs).map_err(|err| err.to_string())?;
    written.push(lib_path);
    generated_files.push(GeneratedFile {
        path: format!("chunks/{chunk_name}/src/lib.rs"),
        sha256: sha256_bytes(lib_rs.as_bytes()),
        size: lib_rs.len() as u64,
    });
    Ok(())
}

/// Write `contents` unless the file already holds exactly that. Keeping the
/// old mtime lets cargo skip recompiling unchanged chunk crates.
fn write_if_changed(path: &Path, contents: impl AsRef<[u8]>) -> std::io::Result<()> {
    let contents = contents.as_ref();
    if fs::read(path).is_ok_and(|existing| existing == contents) {
        return Ok(());
    }
    fs::write(path, contents)
}

/// Delete `chunk_N` crates left by an earlier run that produced more chunks.
fn remove_stale_chunks(chunks_dir: &Path, num_chunks: usize) -> Result<(), String> {
    for entry in fs::read_dir(chunks_dir).map_err(|err| err.to_string())? {
        let entry = entry.map_err(|err| err.to_string())?;
        let name = entry.file_name();
        let Some(index) = name
            .to_str()
            .and_then(|name| name.strip_prefix("chunk_"))
            .and_then(|index| index.parse::<usize>().ok())
        else {
            continue;
        };
        if index >= num_chunks {
            fs::remove_dir_all(entry.path()).map_err(|err| err.to_string())?;
        }
    }
    Ok(())
}

fn emit_main_rs(program: &RustProgram, clobbers: &ClobberAnalysis) -> String {
    let mut out = String::new();
    // Register variables are written per-block; unused mut/assignment warnings
    // are expected in partially lifted functions.
    out.push_str("#![allow(warnings)]\n");
    out.push_str(WINDOWS_SUBSYSTEM_ATTRIBUTE);
    out.push('\n');
    out.push_str("\nuse recomp_runtime;\n\n");
    out.push_str("fn main() -> Result<(), recomp_abi::Fault> {\n");
    emit_runtime_defaults(&mut out, program);
    out.push_str(&format!(
        "    println!(\"recomp target: {}\");\n",
        program.title
    ));
    out.push_str(&format!(
        "    println!(\"abi version: {}\");\n",
        program.abi_version
    ));
    out.push_str("    let runtime_config = recomp_runtime::RuntimeConfig::new(");
    out.push_str(match program.performance_mode {
        crate::config::PerformanceMode::Handheld => "recomp_runtime::PerformanceMode::Handheld",
        crate::config::PerformanceMode::Docked => "recomp_runtime::PerformanceMode::Docked",
    });
    out.push_str(");\n");
    out.push_str("    recomp_runtime::init(&runtime_config);\n");
    emit_memory_layout_init(&mut out, &program.memory_layout);
    emit_memory_image_init(&mut out, program.memory_image.as_ref());
    let heap = program
        .memory_layout
        .regions
        .iter()
        .find(|region| region.name == "heap");
    let heap_base = heap.map(|region| region.base).unwrap_or(0x20000000);
    let heap_size = heap.map(|region| region.size).unwrap_or(0x40000);
    out.push_str(&format!(
        "    recomp_runtime::init_kernel_services(recomp_runtime::KernelSvcDispatcher::all_allowed(), {heap_base:#x}, {heap_size:#x});\n"
    ));
    emit_ipc_init(&mut out, program);
    emit_graphics_init(&mut out, program);
    let initial_sp = program.initial_sp.unwrap_or(0);
    out.push_str(&format!(
        "    recomp_runtime::boot_main_thread({initial_sp:#x});\n"
    ));
    out.push_str("    recomp_runtime::register_dispatcher(__recomp_dispatch);\n");
    out.push_str("    recomp_runtime::register_targets(__RECOMP_TARGETS);\n");
    if program.guest_context {
        out.push_str(&format!(
            "    unsafe {{ {}(recomp_abi::guest_context()) }}?;\n",
            program.entry
        ));
    } else {
        out.push_str(&format!("    {}()?;\n", program.entry));
    }
    out.push_str("    Ok(())\n}\n\n");

    let functions = function_names(program);
    let codegen = CodegenOptions::for_program(program);
    let mut called = std::collections::BTreeSet::new();
    for function in &program.functions {
        out.push_str(&direct_calls(
            &emit_function(function, false, clobbers, program.guest_context, codegen),
            &functions,
            &mut called,
            program.guest_context,
        ));
        out.push('\n');
    }

    emit_dispatch(&mut out, program);

    out
}

fn emit_main_rs_chunked(program: &RustProgram, chunk_of: &[usize]) -> String {
    let mut out = String::new();
    out.push_str("#![allow(warnings)]\n");
    out.push_str(WINDOWS_SUBSYSTEM_ATTRIBUTE);
    out.push('\n');
    out.push_str("\nuse recomp_runtime;\n\n");
    out.push_str("fn main() -> Result<(), recomp_abi::Fault> {\n");
    emit_runtime_defaults(&mut out, program);
    out.push_str(&format!(
        "    println!(\"recomp target: {}\");\n",
        program.title
    ));
    out.push_str(&format!(
        "    println!(\"abi version: {}\");\n",
        program.abi_version
    ));
    out.push_str("    let runtime_config = recomp_runtime::RuntimeConfig::new(");
    out.push_str(match program.performance_mode {
        crate::config::PerformanceMode::Handheld => "recomp_runtime::PerformanceMode::Handheld",
        crate::config::PerformanceMode::Docked => "recomp_runtime::PerformanceMode::Docked",
    });
    out.push_str(");\n");
    out.push_str("    recomp_runtime::init(&runtime_config);\n");
    emit_memory_layout_init(&mut out, &program.memory_layout);
    emit_memory_image_init(&mut out, program.memory_image.as_ref());
    let heap = program
        .memory_layout
        .regions
        .iter()
        .find(|region| region.name == "heap");
    let heap_base = heap.map(|region| region.base).unwrap_or(0x20000000);
    let heap_size = heap.map(|region| region.size).unwrap_or(0x40000);
    out.push_str(&format!(
        "    recomp_runtime::init_kernel_services(recomp_runtime::KernelSvcDispatcher::all_allowed(), {heap_base:#x}, {heap_size:#x});\n"
    ));
    emit_ipc_init(&mut out, program);
    emit_graphics_init(&mut out, program);
    let initial_sp = program.initial_sp.unwrap_or(0);
    out.push_str(&format!(
        "    recomp_runtime::boot_main_thread({initial_sp:#x});\n"
    ));
    out.push_str("    recomp_runtime::register_dispatcher(__recomp_dispatch);\n");
    out.push_str("    recomp_runtime::register_targets(__RECOMP_TARGETS);\n");

    if let Some(entry_idx) = program
        .functions
        .iter()
        .position(|f| f.name == program.entry)
    {
        let chunk_idx = chunk_of[entry_idx];
        if program.guest_context {
            out.push_str(&format!(
                "    unsafe {{ chunk_{chunk_idx}::{}(recomp_abi::guest_context()) }}?;\n",
                program.entry
            ));
        } else {
            out.push_str(&format!("    chunk_{chunk_idx}::{}()?;\n", program.entry));
        }
    } else if let Some(address) = program.entry_address {
        out.push_str(&format!("    recomp_runtime::dispatch({address:#x})?;\n"));
    } else if program.guest_context {
        out.push_str(&format!(
            "    unsafe {{ {}(recomp_abi::guest_context()) }}?;\n",
            program.entry
        ));
    } else {
        out.push_str(&format!("    {}()?;\n", program.entry));
    }

    out.push_str("    Ok(())\n}\n\n");

    emit_dispatch_chunked(&mut out, program, chunk_of);

    out
}

fn emit_dispatch(out: &mut String, program: &RustProgram) {
    let mut targets = std::collections::BTreeMap::new();
    for function in &program.functions {
        if let Some(address) = function_address(&function.name) {
            targets.insert(address, function.name.clone());
        }
    }
    if let Some(address) = program.entry_address {
        if program.functions.iter().any(|f| f.name == program.entry) {
            targets.insert(address, program.entry.clone());
        }
    }

    out.push_str("type __RecompFn = fn() -> recomp_abi::GuestResult;\n");
    // The runtime's call table takes `fn()`; a guest-context function gets a
    // wrapper that hands it this thread's context.
    if program.guest_context {
        for (address, name) in &targets {
            out.push_str(&format!(
                "fn __recomp_entry_{address:016x}() -> recomp_abi::GuestResult {{\n    unsafe {{ {name}(recomp_abi::guest_context()) }}\n}}\n"
            ));
        }
    }
    out.push_str("static __RECOMP_TARGETS: &[(u64, __RecompFn)] = &[\n");
    for (address, name) in &targets {
        if program.guest_context {
            out.push_str(&format!(
                "    ({address:#018x}, __recomp_entry_{address:016x}),\n"
            ));
        } else {
            out.push_str(&format!("    ({address:#018x}, {name}),\n"));
        }
    }
    out.push_str("];\n\n");
    out.push_str("fn __recomp_dispatch(target: u64) -> recomp_abi::GuestResult {\n");
    out.push_str(
        "    match __RECOMP_TARGETS.binary_search_by_key(&target, |(address, _)| *address) {\n",
    );
    out.push_str("        Ok(index) => (__RECOMP_TARGETS[index].1)(),\n");
    out.push_str(
        "        Err(_) => Err(recomp_runtime::RuntimeError::InvalidBranch { address: target }.into()),\n",
    );
    out.push_str("    }\n}\n");
}

fn emit_dispatch_chunked(out: &mut String, program: &RustProgram, chunk_of: &[usize]) {
    let mut targets = std::collections::BTreeMap::new();
    for (idx, function) in program.functions.iter().enumerate() {
        let chunk_idx = chunk_of[idx];
        if let Some(address) = function_address(&function.name) {
            targets.insert(address, format!("chunk_{chunk_idx}::{}", function.name));
        }
    }
    if let Some(address) = program.entry_address {
        if let Some(entry_idx) = program
            .functions
            .iter()
            .position(|f| f.name == program.entry)
        {
            let chunk_idx = chunk_of[entry_idx];
            targets.insert(address, format!("chunk_{chunk_idx}::{}", program.entry));
        }
    }

    out.push_str("type __RecompFn = fn() -> recomp_abi::GuestResult;\n");
    if program.guest_context {
        for (address, path) in &targets {
            out.push_str(&format!(
                "fn __recomp_entry_{address:016x}() -> recomp_abi::GuestResult {{\n    unsafe {{ {path}(recomp_abi::guest_context()) }}\n}}\n"
            ));
        }
    }
    out.push_str("static __RECOMP_TARGETS: &[(u64, __RecompFn)] = &[\n");
    for (address, path) in &targets {
        if program.guest_context {
            out.push_str(&format!(
                "    ({address:#018x}, __recomp_entry_{address:016x}),\n"
            ));
        } else {
            out.push_str(&format!("    ({address:#018x}, {path}),\n"));
        }
    }
    out.push_str("];\n\n");
    out.push_str("fn __recomp_dispatch(target: u64) -> recomp_abi::GuestResult {\n");
    out.push_str(
        "    match __RECOMP_TARGETS.binary_search_by_key(&target, |(address, _)| *address) {\n",
    );
    out.push_str("        Ok(index) => (__RECOMP_TARGETS[index].1)(),\n");
    out.push_str(
        "        Err(_) => Err(recomp_runtime::RuntimeError::InvalidBranch { address: target }.into()),\n",
    );
    out.push_str("    }\n}\n");
}

pub(crate) fn function_address(name: &str) -> Option<u64> {
    let hex = name.strip_prefix("fn_")?;
    if hex.len() != 16 {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

fn emit_memory_layout_init(out: &mut String, layout: &MemoryLayoutDescriptor) {
    out.push_str("    let memory_layout = recomp_runtime::MemoryLayout::new(vec![\n");
    for region in &layout.regions {
        out.push_str(&format!(
            "        recomp_runtime::MemoryRegionSpec::new(\"{}\", {:#x}, {:#x}, recomp_runtime::MemoryPermissions::new({}, {}, {})),\n",
            region.name,
            region.base,
            region.size,
            region.permissions.read,
            region.permissions.write,
            region.permissions.execute
        ));
    }
    out.push_str("    ]);\n");
    out.push_str("    recomp_runtime::init_memory(memory_layout)?;\n");
}

fn emit_memory_image_init(out: &mut String, image: Option<&MemoryImageDescriptor>) {
    let Some(image) = image else {
        return;
    };

    out.push_str("    let mut init_segments = Vec::new();\n");
    for segment in &image.init_segments {
        out.push_str(&format!(
            "    let segment_path = recomp_runtime::resolve_relative_asset({0}, Some(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/\", {0})));\n",
            crate::pipeline::rust_string_literal(&segment.init_path)
        ));
        out.push_str(
            "    let bytes = std::fs::read(&segment_path).map_err(|err| recomp_runtime::RuntimeError::Io { message: format!(\"read {}: {err}\", segment_path.display()) })?;\n",
        );
        out.push_str(&format!(
            "    init_segments.push(recomp_runtime::MemoryInitSegment::new(\"{}\", {:#x}, {:#x}, bytes));\n",
            segment.name, segment.base, segment.size
        ));
    }

    out.push_str("    let zero_segments = vec![\n");
    for segment in &image.zero_segments {
        out.push_str(&format!(
            "        recomp_runtime::MemoryZeroSegment::new(\"{}\", {:#x}, {:#x}),\n",
            segment.name, segment.base, segment.size
        ));
    }
    out.push_str("    ];\n");
    out.push_str("    recomp_runtime::apply_memory_image(&init_segments, &zero_segments)?;\n");
}

/// Functions with more blocks than this are emitted as several Rust
/// functions ("parts") joined by a trampoline. rustc's cost grows faster
/// than linearly with function size; lifted LM3 initializers reach 2800
/// blocks, which took rustc over half an hour as a single function.
const MAX_PART_BLOCKS: usize = 64;

/// The call to a function's host implementation (`host_functions`), first
/// in its body: when it returns true the lifted code below does not run.
fn emit_host_call(out: &mut String, function: &RustFunction) {
    if let Some(host) = &function.host {
        out.push_str(&format!(
            "    extern \"Rust\" {{\n        fn __recomp_host_{host}() -> Result<bool, recomp_abi::Fault>;\n    }}\n"
        ));
        out.push_str(&format!(
            "    // SAFETY: the runtime defines this symbol with this signature.\n    if unsafe {{ __recomp_host_{host}() }}? {{\n        return Ok(());\n    }}\n"
        ));
    }
}

/// A set of guest registers (x0-x31 and v0-v31) plus the NZCV flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GuestMask {
    x: u32,
    v: u32,
    flags: bool,
}

impl GuestMask {
    const ALL: GuestMask = GuestMask {
        x: u32::MAX,
        v: u32::MAX,
        flags: true,
    };

    fn insert(&mut self, name: &str) -> bool {
        if let Some(index) = crate::pipeline::reg_index(name) {
            let bit = 1u32 << index;
            let changed = self.x & bit == 0;
            self.x |= bit;
            changed
        } else if let Some(index) = crate::pipeline::vreg_index(name) {
            let bit = 1u32 << index;
            let changed = self.v & bit == 0;
            self.v |= bit;
            changed
        } else {
            false
        }
    }

    fn contains(&self, name: &str) -> bool {
        if let Some(index) = crate::pipeline::reg_index(name) {
            self.x & (1u32 << index) != 0
        } else if let Some(index) = crate::pipeline::vreg_index(name) {
            self.v & (1u32 << index) != 0
        } else {
            false
        }
    }

    /// Adds `other` to this mask; true when a bit was added.
    fn union(&mut self, other: GuestMask) -> bool {
        let changed = (self.x | other.x) != self.x
            || (self.v | other.v) != self.v
            || (other.flags && !self.flags);
        self.x |= other.x;
        self.v |= other.v;
        self.flags |= other.flags;
        changed
    }

    fn intersection(self, other: GuestMask) -> GuestMask {
        GuestMask {
            x: self.x & other.x,
            v: self.v & other.v,
            flags: self.flags && other.flags,
        }
    }

    /// The registers of `regs` this mask names.
    fn select(&self, regs: &[String]) -> Vec<String> {
        regs.iter()
            .filter(|reg| self.contains(reg))
            .cloned()
            .collect()
    }
}

/// Whether the identifier ending at `pos` is the left side of an assignment
/// (`=`, not `==` or `=>`, or a compound assignment).
fn is_assignment_after(bytes: &[u8], mut pos: usize) -> bool {
    while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }
    if bytes.get(pos) == Some(&b'=') {
        return !matches!(bytes.get(pos + 1), Some(b'=') | Some(b'>'));
    }
    const COMPOUND: [&[u8]; 10] = [
        b"+=", b"-=", b"*=", b"/=", b"%=", b"&=", b"|=", b"^=", b"<<=", b">>=",
    ];
    COMPOUND
        .iter()
        .any(|operator| bytes[pos..].starts_with(operator))
}

/// A `(*ctx).x[N]`, `(*ctx).v[N]` or `(*ctx).nzcv` access starting at
/// `pos`, and the position just past it. Direct register mode writes guest
/// registers as context fields; the clobber analysis must see those accesses
/// just like local register names.
fn context_field_at(line: &str, pos: usize) -> Option<(GuestMask, usize)> {
    let rest = line.get(pos..)?.strip_prefix("(*ctx).")?;
    if let Some(rest) = rest.strip_prefix("x[").or_else(|| rest.strip_prefix("v[")) {
        let vector = line.as_bytes().get(pos + "(*ctx).".len()) == Some(&b'v');
        let close = rest.find(']')?;
        let index: usize = rest[..close].parse().ok()?;
        if index >= 32 {
            return None;
        }
        let bit = 1u32 << index;
        let mask = if vector {
            GuestMask {
                v: bit,
                ..GuestMask::default()
            }
        } else {
            GuestMask {
                x: bit,
                ..GuestMask::default()
            }
        };
        return Some((mask, pos + "(*ctx).".len() + 2 + close + 1));
    }
    if rest.starts_with("nzcv") {
        return Some((
            GuestMask {
                flags: true,
                ..GuestMask::default()
            },
            pos + "(*ctx).nzcv".len(),
        ));
    }
    None
}

/// The guest registers a generated line reads and writes, and the writes
/// that happen unconditionally. A line with a branch (`if let`, `match`,
/// `if ... else`) assigns nothing for certain, so its writes are excluded
/// from the definite set; reads are counted either way, which only makes
/// the result more conservative.
fn scan_line(line: &str) -> (GuestMask, GuestMask, GuestMask) {
    let bytes = line.as_bytes();
    let mut reads = GuestMask::default();
    let mut writes = GuestMask::default();
    let mut pos = 0;
    while pos < bytes.len() {
        let start = pos;
        let ch = bytes[pos];
        if ch == b'(' && line[pos..].starts_with("(*ctx).") {
            if let Some((field, end)) = context_field_at(line, pos) {
                if is_assignment_after(bytes, end) {
                    writes.union(field);
                } else {
                    reads.union(field);
                }
                pos = end;
                continue;
            }
        }
        if ch == b'_' || ch.is_ascii_alphabetic() {
            while pos < bytes.len() && (bytes[pos] == b'_' || bytes[pos].is_ascii_alphanumeric()) {
                pos += 1;
            }
            // Literal suffixes (`8u64`) are not identifiers.
            if start > 0 && bytes[start - 1].is_ascii_digit() {
                continue;
            }
            // `out.x1` is a struct field, not guest register x1.
            let mut before = start;
            while before > 0 && matches!(bytes[before - 1], b' ' | b'\t') {
                before -= 1;
            }
            if before > 0 && bytes[before - 1] == b'.' {
                continue;
            }
            let name = &line[start..pos];
            if crate::pipeline::reg_index(name).is_none()
                && crate::pipeline::vreg_index(name).is_none()
            {
                continue;
            }
            if is_assignment_after(bytes, pos) {
                writes.insert(name);
            } else {
                reads.insert(name);
            }
        } else if ch == b'"' {
            // Messages mention registers ("x0 is null") but do not access
            // them.
            pos += 1;
            while pos < bytes.len() {
                match bytes[pos] {
                    b'\\' => pos += 2,
                    b'"' => {
                        pos += 1;
                        break;
                    }
                    _ => pos += 1,
                }
            }
        } else {
            pos += 1;
        }
    }
    let conditional = line.contains("if ")
        || line.contains("if(")
        || line.contains("match ")
        || line.contains(" else ");
    let definite = if conditional {
        GuestMask::default()
    } else {
        writes
    };
    (reads, writes, definite)
}

/// The registers a block terminator reads (a branch or call through one).
fn terminator_reads(terminator: &RustTerminator) -> GuestMask {
    let mut reads = GuestMask::default();
    match terminator {
        RustTerminator::BrCond {
            cond_expr: Some(expr),
            ..
        } => reads = scan_line(expr).0,
        RustTerminator::Call { return_address, .. } => {
            // The return-address check reads x30 after the call. Lifted BLs
            // write it just before the terminator, so it is normally
            // defined; a hand-written module may not.
            if return_address.is_some() {
                reads.insert("x30");
            }
        }
        RustTerminator::CallIndirect {
            reg,
            return_address,
            ..
        } => {
            reads.insert(reg);
            if return_address.is_some() {
                reads.insert("x30");
            }
        }
        RustTerminator::BrIndirect { reg } | RustTerminator::Switch { reg, .. } => {
            reads.insert(reg);
        }
        _ => {}
    }
    reads
}

#[derive(Clone, Copy, Debug, Default)]
struct BlockFacts {
    /// Registers read before this block writes them, ignoring values carried
    /// in from predecessors.
    use_before_def: GuestMask,
    /// Registers any line in the block writes (including conditionally).
    kills: GuestMask,
}

fn block_facts(block: &crate::pipeline::RustBlock) -> BlockFacts {
    let mut facts = BlockFacts::default();
    let mut defined = GuestMask::default();
    for line in &block.lines {
        let (reads, writes, definite) = scan_line(line);
        add_undefined(reads, defined, &mut facts.use_before_def);
        defined.union(definite);
        facts.kills.union(writes);
    }
    add_undefined(
        terminator_reads(&block.terminator),
        defined,
        &mut facts.use_before_def,
    );
    facts
}

fn add_undefined(reads: GuestMask, defined: GuestMask, target: &mut GuestMask) {
    let missing = GuestMask {
        x: reads.x & !defined.x,
        v: reads.v & !defined.v,
        flags: reads.flags && !defined.flags,
    };
    target.union(missing);
}

/// The registers a function reads before writing them on any path: what its
/// caller must have in the thread's register file when it enters.
///
/// Definite assignment is a must-analysis over the block graph: a register
/// counts as defined at a block's end when every predecessor defined it.
/// Loops make the block-entry sets start at "everything defined" and shrink,
/// so a back edge never invents reads (the greatest fixed point).
fn entry_reads(blocks: &[crate::pipeline::RustBlock], facts: &[BlockFacts]) -> GuestMask {
    let count = blocks.len();
    let mut required = GuestMask::default();
    if count == 0 {
        return required;
    }
    let index_of: HashMap<&str, usize> = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.label.as_str(), index))
        .collect();
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (index, block) in blocks.iter().enumerate() {
        let mut edges: Vec<&str> = Vec::new();
        match &block.terminator {
            RustTerminator::Br { target } => edges.push(target),
            RustTerminator::BrCond {
                then_label,
                else_label,
                ..
            } => {
                edges.push(then_label);
                edges.push(else_label);
            }
            RustTerminator::Call { next, .. } | RustTerminator::CallIndirect { next, .. } => {
                edges.push(next);
            }
            RustTerminator::Switch { cases, .. } => {
                edges.extend(cases.iter().map(|(_, label)| label.as_str()));
            }
            RustTerminator::BrIndirect { .. }
            | RustTerminator::Trap { .. }
            | RustTerminator::Ret => {}
        }
        for label in edges {
            if let Some(&target) = index_of.get(label) {
                predecessors[target].push(index);
            }
        }
    }

    let mut in_defined = vec![GuestMask::ALL; count];
    in_defined[0] = GuestMask::default();
    let mut out_defined: Vec<GuestMask> = (0..count)
        .map(|index| {
            let mut out = in_defined[index];
            out.union(facts[index].kills);
            out
        })
        .collect();
    loop {
        let mut changed = false;
        for index in 0..count {
            let incoming = if index == 0 || predecessors[index].is_empty() {
                GuestMask::default()
            } else {
                let mut incoming = GuestMask::ALL;
                for &predecessor in &predecessors[index] {
                    incoming = incoming.intersection(out_defined[predecessor]);
                }
                incoming
            };
            if incoming != in_defined[index] {
                in_defined[index] = incoming;
                changed = true;
            }
            let mut out = in_defined[index];
            out.union(facts[index].kills);
            if out != out_defined[index] {
                out_defined[index] = out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for index in 0..count {
        let missing = GuestMask {
            x: facts[index].use_before_def.x & !in_defined[index].x,
            v: facts[index].use_before_def.v & !in_defined[index].v,
            flags: facts[index].use_before_def.flags && !in_defined[index].flags,
        };
        required.union(missing);
    }
    required
}

struct LocalFacts {
    required: GuestMask,
    clobber: GuestMask,
    flags: bool,
    callees: Vec<usize>,
    unknown: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct ClobberFacts {
    required: GuestMask,
    clobber: GuestMask,
    unknown: bool,
}

/// Whether a linear function calls code whose effects are not known here.
fn line_enters_unknown_code(line: &str) -> bool {
    line.contains("recomp_abi::dispatch(")
        || line.contains("recomp_abi::call(")
        || (line.trim_end().ends_with("()?;") && !line.contains("recomp_abi::"))
}

/// One function's own reads and writes, and its calls resolved to indices.
fn local_facts(function: &RustFunction, address_of: &HashMap<u64, usize>) -> LocalFacts {
    let mut local = LocalFacts {
        required: GuestMask::default(),
        clobber: GuestMask::default(),
        flags: function.needs_flags,
        callees: Vec::new(),
        // Host implementations use the runtime's register accessors and can
        // write any register, so its lifted callees cannot be analyzed.
        unknown: function.host.is_some(),
    };
    match &function.body {
        FunctionBody::Linear(lines) => {
            let mut defined = GuestMask::default();
            for line in lines {
                let (reads, writes, definite) = scan_line(line);
                add_undefined(reads, defined, &mut local.required);
                defined.union(definite);
                local.clobber.union(writes);
                local.unknown |= line_enters_unknown_code(line);
            }
        }
        FunctionBody::Blocks(blocks) => {
            let facts: Vec<BlockFacts> = blocks.iter().map(block_facts).collect();
            local.required = entry_reads(blocks, &facts);
            for fact in &facts {
                local.clobber.union(fact.kills);
            }
            for block in blocks {
                match &block.terminator {
                    RustTerminator::Call { call_line, .. } => {
                        match dispatch_call_address(call_line)
                            .and_then(|address| address_of.get(&address).copied())
                        {
                            Some(callee) => local.callees.push(callee),
                            None => local.unknown = true,
                        }
                    }
                    // A tail dispatch may run any function; its register
                    // effects are not known at generation time.
                    RustTerminator::CallIndirect { .. }
                    | RustTerminator::BrIndirect { .. }
                    | RustTerminator::Switch { .. } => local.unknown = true,
                    _ => {}
                }
            }
        }
    }
    local
}

/// Propagates reads and writes through the call graph. Starting from every
/// function's own facts and unioning callees into callers reaches the least
/// fixed point; recursive cycles (functions calling each other) settle after
/// each register is added at most once per caller.
fn propagate(locals: Vec<LocalFacts>) -> Vec<ClobberFacts> {
    let count = locals.len();
    let mut facts: Vec<ClobberFacts> = locals
        .iter()
        .map(|local| {
            let mut required = local.required;
            let mut clobber = local.clobber;
            if local.flags {
                required.flags = true;
                clobber.flags = true;
            }
            if local.unknown {
                required = GuestMask::ALL;
                clobber = GuestMask::ALL;
            }
            ClobberFacts {
                required,
                clobber,
                unknown: local.unknown,
            }
        })
        .collect();
    let mut callers: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (caller, local) in locals.iter().enumerate() {
        for &callee in &local.callees {
            callers[callee].push(caller);
        }
    }
    let mut queued = vec![true; count];
    let mut queue: VecDeque<usize> = (0..count).collect();
    while let Some(callee) = queue.pop_front() {
        queued[callee] = false;
        let callee_required = facts[callee].required;
        let callee_clobber = facts[callee].clobber;
        let callee_unknown = facts[callee].unknown;
        for &caller in &callers[callee] {
            let mut changed = false;
            if facts[caller].required.union(callee_required) {
                changed = true;
            }
            if facts[caller].clobber.union(callee_clobber) {
                changed = true;
            }
            if callee_unknown && !facts[caller].unknown {
                facts[caller].unknown = true;
                changed = true;
            }
            if changed && !queued[caller] {
                queued[caller] = true;
                queue.push_back(caller);
            }
        }
    }
    facts
}

/// A direct callee known well enough to spill selectively at its call.
#[derive(Debug)]
struct CallTarget {
    name: String,
    /// Registers it, or a function it calls, reads before writing.
    required: GuestMask,
    /// Registers it, or a function it calls, may write.
    clobber: GuestMask,
    /// An indirect call, host implementation, or unresolved target makes the
    /// call sites fall back to the full spill.
    unknown: bool,
}

/// Per-function clobber masks, looked up by guest address at direct calls.
#[derive(Debug, Default)]
struct ClobberAnalysis {
    by_address: HashMap<u64, CallTarget>,
}

impl ClobberAnalysis {
    fn from_program(program: &RustProgram) -> Self {
        Self::from_functions(&program.functions)
    }

    fn from_functions(functions: &[RustFunction]) -> Self {
        let mut address_of = HashMap::new();
        for (index, function) in functions.iter().enumerate() {
            if let Some(address) = function_address(&function.name) {
                address_of.insert(address, index);
            }
        }
        let locals: Vec<LocalFacts> = functions
            .iter()
            .map(|function| local_facts(function, &address_of))
            .collect();
        let facts = propagate(locals);
        let mut by_address = HashMap::new();
        for (index, function) in functions.iter().enumerate() {
            let Some(address) = function_address(&function.name) else {
                continue;
            };
            by_address.insert(
                address,
                CallTarget {
                    name: function.name.clone(),
                    required: facts[index].required,
                    clobber: facts[index].clobber,
                    unknown: facts[index].unknown,
                },
            );
        }
        Self { by_address }
    }

    /// The target of a call that `direct_calls` will rewrite to a direct
    /// call, when its effects are fully known.
    fn call_target(&self, call_line: &str) -> Option<&CallTarget> {
        let address = dispatch_call_address(call_line)?;
        self.by_address
            .get(&address)
            .filter(|target| !target.unknown)
    }
}

/// The guest address of a `recomp_abi::dispatch(0x...)` call line, exactly
/// where [`direct_calls`] turns it into a direct call.
fn dispatch_call_address(call_line: &str) -> Option<u64> {
    const DISPATCH: &str = "recomp_abi::dispatch(0x";
    let after = call_line.trim_start().strip_prefix(DISPATCH)?;
    let digits = after
        .find(|ch: char| !ch.is_ascii_hexdigit())
        .unwrap_or(after.len());
    if !after[digits..].starts_with(')') {
        return None;
    }
    u64::from_str_radix(&after[..digits], 16).ok()
}

fn emit_function(
    function: &RustFunction,
    is_pub: bool,
    clobbers: &ClobberAnalysis,
    guest_context: bool,
    codegen: CodegenOptions,
) -> String {
    if let FunctionBody::Blocks(blocks) = &function.body {
        if blocks.len() > MAX_PART_BLOCKS {
            return emit_split_function(function, blocks, is_pub, clobbers, guest_context, codegen);
        }
    }
    let mut out = String::new();
    // Public functions are a chunk's: other chunks call them by this name.
    let vis = if is_pub { "#[no_mangle]\npub " } else { "" };
    let parameters = if guest_context {
        "ctx: *mut recomp_abi::GuestContext"
    } else {
        ""
    };
    // Dereferencing the caller's context pointer is the function's
    // precondition, so guest-context functions are `unsafe fn`.
    let unsafety = if guest_context { "unsafe " } else { "" };
    out.push_str(&format!(
        "{vis}{unsafety}fn {}({parameters}) -> recomp_abi::GuestResult {{\n",
        function.name
    ));
    emit_host_call(&mut out, function);
    let (block_temps, local_temps) = match &function.body {
        FunctionBody::Blocks(blocks) => block_local_temps(function, blocks),
        FunctionBody::Linear(_) => (Vec::new(), std::collections::HashSet::new()),
    };
    emit_prologue(
        &mut out,
        function,
        &local_temps,
        false,
        guest_context,
        codegen.direct_registers,
    );
    match &function.body {
        FunctionBody::Linear(lines) => {
            for line in lines {
                out.push_str("    ");
                out.push_str(line);
                out.push('\n');
            }
            if lines
                .last()
                .map(|line| !line.trim_start().starts_with("return"))
                .unwrap_or(true)
            {
                for spill in crate::pipeline::spill_lines(
                    &function.regs,
                    guest_context,
                    codegen.direct_registers,
                ) {
                    out.push_str("    ");
                    out.push_str(&spill);
                    out.push('\n');
                }
                out.push_str("    Ok(())\n");
            }
        }
        FunctionBody::Blocks(blocks) => {
            // Blocks are numbered in order; branching by integer keeps both
            // the generated code and each dispatch small.
            let index_of = block_indices(blocks);
            let exits = Exits {
                index_of: &index_of,
                regs: &function.regs,
                split: None,
                needs_flags: function.needs_flags,
                guest_context,
                direct: codegen.direct_registers,
                current: std::cell::Cell::new(None),
                clobbers,
                loop_backedges: codegen.loop_backedges,
            };
            out.push_str("    let mut block: u32 = 0;\n");
            emit_block_loop(&mut out, blocks, &block_temps, &exits, codegen, |_| true);
        }
    }
    out.push_str("}\n");
    out
}

/// A split function: a trampoline over parts of at most `MAX_PART_BLOCKS`
/// blocks. Each part loads the guest registers on entry and spills them on
/// exit; NZCV travels with the (part, block) the trampoline jumps to.
fn emit_split_function(
    function: &RustFunction,
    blocks: &[crate::pipeline::RustBlock],
    is_pub: bool,
    clobbers: &ClobberAnalysis,
    guest_context: bool,
    codegen: CodegenOptions,
) -> String {
    let name = &function.name;
    // Public functions are a chunk's: other chunks call them by this name.
    let vis = if is_pub { "#[no_mangle]\npub " } else { "" };
    let part_of: Vec<usize> = (0..blocks.len()).map(|i| i / MAX_PART_BLOCKS).collect();
    let parts = blocks.len().div_ceil(MAX_PART_BLOCKS);
    let index_of = block_indices(blocks);
    let (block_temps, local_temps) = block_local_temps(function, blocks);
    let parameters = if guest_context {
        "ctx: *mut recomp_abi::GuestContext"
    } else {
        ""
    };
    let unsafety = if guest_context { "unsafe " } else { "" };
    let direct = codegen.direct_registers;

    let mut out = format!("{vis}{unsafety}fn {name}({parameters}) -> recomp_abi::GuestResult {{\n");
    emit_host_call(&mut out, function);
    if direct {
        // The parts share the caller's context, so flags stay in its `nzcv`
        // field and only (part, block) travels through the trampoline.
        out.push_str("    let mut next: Option<(u32, u32)> = Some((0, 0));\n");
        out.push_str("    while let Some((part, block)) = next {\n");
        out.push_str("        next = match part {\n");
        for part in 0..parts {
            out.push_str(&format!(
                "            {part} => unsafe {{ {name}__p{part}(ctx, block)? }},\n"
            ));
        }
        out.push_str("            _ => unreachable!(),\n");
        out.push_str("        };\n");
        out.push_str("    }\n");
        out.push_str("    Ok(())\n");
        out.push_str("}\n");

        for part in 0..parts {
            let part_parameters = format!("{parameters}, entry: u32");
            out.push_str(&format!(
                "\n#[inline(never)]\n{unsafety}fn {name}__p{part}({part_parameters}) -> Result<Option<(u32, u32)>, recomp_abi::Fault> {{\n"
            ));
            emit_prologue(
                &mut out,
                function,
                &local_temps,
                true,
                guest_context,
                direct,
            );
            out.push_str("    let mut block: u32 = entry;\n");
            let exits = Exits {
                index_of: &index_of,
                regs: &function.regs,
                split: Some((&part_of, part)),
                needs_flags: function.needs_flags,
                guest_context,
                direct,
                current: std::cell::Cell::new(None),
                clobbers,
                loop_backedges: codegen.loop_backedges,
            };
            emit_block_loop(&mut out, blocks, &block_temps, &exits, codegen, |index| {
                part_of[index] == part
            });
            out.push_str("}\n");
        }
        return out;
    }
    let nzcv = if !function.needs_flags {
        "0"
    } else if guest_context {
        "unsafe { (*ctx).nzcv }"
    } else {
        "recomp_abi::cpu_nzcv()"
    };
    out.push_str(&format!(
        "    let mut next: Option<(u32, u32, u8)> = Some((0, 0, {nzcv}));\n"
    ));
    out.push_str("    while let Some((part, block, nzcv)) = next {\n");
    out.push_str("        next = match part {\n");
    for part in 0..parts {
        let arguments = if guest_context {
            "ctx, block, nzcv"
        } else {
            "block, nzcv"
        };
        if guest_context {
            out.push_str(&format!(
                "            {part} => unsafe {{ {name}__p{part}({arguments})? }},\n"
            ));
        } else {
            out.push_str(&format!(
                "            {part} => {name}__p{part}({arguments})?,\n"
            ));
        }
    }
    out.push_str("            _ => unreachable!(),\n");
    out.push_str("        };\n");
    out.push_str("    }\n");
    out.push_str("    Ok(())\n");
    out.push_str("}\n");

    for part in 0..parts {
        let part_parameters = if guest_context {
            format!("{parameters}, entry: u32, nzcv: u8")
        } else {
            "entry: u32, nzcv: u8".to_string()
        };
        out.push_str(&format!(
            "\n#[inline(never)]\n{unsafety}fn {name}__p{part}({part_parameters}) -> Result<Option<(u32, u32, u8)>, recomp_abi::Fault> {{\n"
        ));
        emit_prologue(
            &mut out,
            function,
            &local_temps,
            true,
            guest_context,
            direct,
        );
        out.push_str("    let mut block: u32 = entry;\n");
        let exits = Exits {
            index_of: &index_of,
            regs: &function.regs,
            split: Some((&part_of, part)),
            needs_flags: function.needs_flags,
            guest_context,
            direct,
            current: std::cell::Cell::new(None),
            clobbers,
            loop_backedges: codegen.loop_backedges,
        };
        emit_block_loop(&mut out, blocks, &block_temps, &exits, codegen, |index| {
            part_of[index] == part
        });
        out.push_str("}\n");
    }
    out
}

fn block_indices(blocks: &[crate::pipeline::RustBlock]) -> std::collections::HashMap<&str, usize> {
    blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.label.as_str(), index))
        .collect()
}

/// Register, temp and flag declarations at the top of a function or part.
/// Parts take the incoming NZCV from their `nzcv` parameter, functions from
/// the thread's flags (a caller's flags stay visible, as in hardware).
/// Whether the function loads or stores guest memory (through `__mem`).
fn uses_memory(function: &RustFunction) -> bool {
    let uses = |line: &String| line.contains("__mem.");
    match &function.body {
        FunctionBody::Linear(lines) => lines.iter().any(uses),
        FunctionBody::Blocks(blocks) => blocks.iter().any(|block| block.lines.iter().any(uses)),
    }
}

fn emit_prologue(
    out: &mut String,
    function: &RustFunction,
    local_temps: &std::collections::HashSet<&str>,
    flags_from_param: bool,
    guest_context: bool,
    direct: bool,
) {
    if !guest_context {
        if function
            .regs
            .iter()
            .any(|reg| crate::pipeline::reg_index(reg).is_some())
        {
            out.push_str("    let __regs = recomp_abi::cpu_regs();\n");
        }
        if function
            .regs
            .iter()
            .any(|reg| crate::pipeline::vreg_index(reg).is_some())
        {
            out.push_str("    let __vregs = recomp_abi::cpu_vregs();\n");
        }
    }
    if uses_memory(function) {
        out.push_str("    let __mem = recomp_abi::memory();\n");
    }
    for reg in &function.regs {
        if local_temps.contains(reg.as_str()) {
            continue;
        }
        // Direct mode reads and writes guest registers through `ctx`; only
        // instruction temps are locals.
        if direct
            && (crate::pipeline::reg_index(reg).is_some()
                || crate::pipeline::vreg_index(reg).is_some())
        {
            continue;
        }
        if let Some(index) = crate::pipeline::vreg_index(reg) {
            if guest_context {
                out.push_str(&format!(
                    "    let mut {reg}: u128 = unsafe {{ (*ctx).v[{index}] }};\n"
                ));
            } else {
                out.push_str(&format!("    let mut {reg}: u128 = __vregs[{index}];\n"));
            }
        } else if let Some(index) = crate::pipeline::reg_index(reg) {
            if guest_context {
                out.push_str(&format!(
                    "    let mut {reg}: i64 = unsafe {{ (*ctx).x[{index}] }};\n"
                ));
            } else {
                out.push_str(&format!("    let mut {reg}: i64 = __regs[{index}];\n"));
            }
        } else {
            out.push_str(&format!("    {}\n", temp_declaration(reg)));
        }
    }
    if function.needs_flags && !direct {
        if !flags_from_param {
            if guest_context {
                out.push_str("    let nzcv = unsafe { (*ctx).nzcv };\n");
            } else {
                out.push_str("    let nzcv = recomp_abi::cpu_nzcv();\n");
            }
        }
        out.push_str("    let mut flag_n = nzcv & 8 != 0;\n");
        out.push_str("    let mut flag_z = nzcv & 4 != 0;\n");
        out.push_str("    let mut flag_c = nzcv & 2 != 0;\n");
        out.push_str("    let mut flag_v = nzcv & 1 != 0;\n");
    }
    if !function.regs.is_empty() || (function.needs_flags && !direct) {
        out.push('\n');
    }
}

/// The `loop { match block { ... } }` dispatch over the blocks `include`
/// selects.
fn emit_block_loop(
    out: &mut String,
    blocks: &[crate::pipeline::RustBlock],
    block_temps: &[Vec<String>],
    exits: &Exits<'_>,
    codegen: CodegenOptions,
    include: impl Fn(usize) -> bool,
) {
    if !codegen.block_tracking {
        // Without per-block tracking the cell keeps the function's entry
        // address, so a fault or a watched store still names the function.
        let entry = blocks
            .iter()
            .enumerate()
            .find(|(index, _)| include(*index))
            .and_then(|(_, block)| block_address(&block.label));
        if let Some(address) = entry {
            out.push_str(&format!("    recomp_abi::enter_block({address:#x});\n"));
        }
    }
    out.push_str("    loop {\n");
    out.push_str("        match block {\n");
    for (index, block) in blocks.iter().enumerate() {
        if !include(index) {
            continue;
        }
        out.push_str(&format!("            {index} => {{\n"));
        // Blocks lifted from a binary are named after their address.
        if codegen.block_tracking {
            if let Some(address) = block_address(&block.label) {
                out.push_str(&format!(
                    "                recomp_abi::enter_block({address:#x});\n"
                ));
            }
        }
        for temp in &block_temps[index] {
            out.push_str(&format!("                {}\n", temp_declaration(temp)));
        }
        for line in &block.lines {
            out.push_str("                ");
            out.push_str(line);
            out.push('\n');
        }
        exits.current.set(block_address(&block.label));
        emit_block_terminator(out, &block.terminator, exits);
        out.push_str("            },\n");
    }
    out.push_str("            _ => unreachable!(),\n");
    out.push_str("        }\n");
    out.push_str("    }\n");
}

/// The guest address of a lifted block (named `bb_<hex address>`).
fn block_address(label: &str) -> Option<u64> {
    label
        .strip_prefix("bb_")
        .and_then(|hex| u64::from_str_radix(hex, 16).ok())
}

/// The flag locals packed as NZCV.
const NZCV: &str =
    "((flag_n as u8) << 3) | ((flag_z as u8) << 2) | ((flag_c as u8) << 1) | (flag_v as u8)";

/// How control leaves a block: within one Rust function, or out of one part
/// of a split function.
struct Exits<'a> {
    index_of: &'a std::collections::HashMap<&'a str, usize>,
    regs: &'a [String],
    /// For split functions: the part of every block, and the current part.
    split: Option<(&'a [usize], usize)>,
    needs_flags: bool,
    /// Spill through the function's `*mut GuestContext` instead of the
    /// snapshot thread-local.
    guest_context: bool,
    /// Guest registers are context fields: no spill, reload or flag locals
    /// are needed at exits and calls.
    direct: bool,
    /// The guest address of the block whose exits are being emitted: a
    /// jump to it or an earlier block closes a loop.
    current: std::cell::Cell<Option<u64>>,
    clobbers: &'a ClobberAnalysis,
    /// Report loop backedges (`backedge`), for serialized preemption or the
    /// hang monitor.
    loop_backedges: bool,
}

impl Exits<'_> {
    fn spill(&self) -> String {
        crate::pipeline::spill_lines(self.regs, self.guest_context, self.direct).join(" ")
    }

    /// A guest register read expression.
    fn read_reg(&self, reg: &str) -> String {
        if self.direct {
            if let Some(index) = crate::pipeline::reg_index(reg) {
                return format!("(unsafe {{ (*ctx).x[{index}] }})");
            }
            if let Some(index) = crate::pipeline::vreg_index(reg) {
                return format!("(unsafe {{ (*ctx).v[{index}] }})");
            }
        }
        reg.to_string()
    }

    /// The lines of a call that must spill every register: indirect calls,
    /// unknown callees, and the fallback while the runtime needs to see
    /// calls (serialized execution, script hooks).
    fn full_call_lines(&self, call_line: &str) -> Vec<String> {
        if self.direct {
            // The context is already the whole register file; the call and
            // any hooks or serialized execution read it directly.
            return vec![call_line.to_string()];
        }
        let mut lines = crate::pipeline::spill_lines(self.regs, self.guest_context, self.direct);
        if let Some(store) = self.flags_store_line() {
            lines.push(store);
        }
        lines.push(call_line.to_string());
        lines.extend(crate::pipeline::reload_lines(
            self.regs,
            self.guest_context,
            self.direct,
        ));
        if let Some(reload) = self.flags_in() {
            lines.push(reload);
        }
        lines
    }

    fn flags_store_line(&self) -> Option<String> {
        if !self.needs_flags || self.direct {
            return None;
        }
        if self.guest_context {
            Some(format!("unsafe {{ (*ctx).nzcv = {NZCV}; }}"))
        } else {
            Some(format!("recomp_abi::cpu_set_nzcv({NZCV});"))
        }
    }

    /// A direct call's lines. When the callee's reads and writes are known,
    /// the fast path stores only its inputs and reloads only its outputs;
    /// the runtime's call hooks and serialized execution still get the full
    /// register file through the slow path. Direct register mode keeps the
    /// whole register file in the context, so every call is just a call.
    fn call_lines(&self, call_line: &str) -> Vec<String> {
        if self.direct {
            return vec![call_line.to_string()];
        }
        let Some(target) = self.clobbers.call_target(call_line) else {
            return self.full_call_lines(call_line);
        };
        // A callee's prologue loads every register it tracks and its exits
        // store them again, including on paths that never wrote one. Every
        // register it may write is therefore also stored here, so such a
        // path cannot put a stale value back into the caller's register
        // file; the reload then only has to read the callee's outputs.
        let mut store_mask = target.required;
        store_mask.union(target.clobber);
        let store = store_mask.select(self.regs);
        let reload = target.clobber.select(self.regs);
        let tracked = self
            .regs
            .iter()
            .filter(|reg| {
                crate::pipeline::reg_index(reg).is_some()
                    || crate::pipeline::vreg_index(reg).is_some()
            })
            .count();
        if store.len() >= tracked && reload.len() >= tracked {
            return self.full_call_lines(call_line);
        }
        let store_flags = self.needs_flags && target.required.flags;
        let reload_flags = self.needs_flags && target.clobber.flags;
        let name = &target.name;

        let mut lines = Vec::new();
        lines.push("if recomp_abi::direct_calls_enabled() {".to_string());
        lines.extend(crate::pipeline::spill_lines(
            &store,
            self.guest_context,
            self.direct,
        ));
        if store_flags {
            if let Some(store) = self.flags_store_line() {
                lines.push(store);
            }
        }
        let arguments = if self.guest_context { "(ctx)" } else { "()" };
        lines.push(format!(
            "recomp_abi::call_direct(|| unsafe {{ {name}{arguments} }})?;"
        ));
        lines.extend(crate::pipeline::reload_lines(
            &reload,
            self.guest_context,
            self.direct,
        ));
        if reload_flags {
            if let Some(reload) = self.flags_in() {
                lines.push(reload);
            }
        }
        lines.push("} else {".to_string());
        lines.extend(self.full_call_lines(call_line));
        lines.push("}".to_string());
        lines
    }

    fn goto(&self, label: &str) -> String {
        let Some(&index) = self.index_of.get(label) else {
            return format!("panic!(\"unknown block label: {label}\");");
        };
        let loops = matches!(
            (self.current.get(), block_address(label)),
            (Some(current), Some(target)) if target <= current
        );
        let backedge = if loops && self.loop_backedges {
            "recomp_abi::backedge(); "
        } else {
            ""
        };
        match self.split {
            Some((part_of, part)) if part_of[index] != part => {
                if self.direct {
                    return format!("{backedge}return Ok(Some(({}, {index})));", part_of[index]);
                }
                let nzcv = if self.needs_flags { NZCV } else { "0" };
                format!(
                    "{backedge}{} return Ok(Some(({}, {index}, {nzcv})));",
                    self.spill(),
                    part_of[index]
                )
            }
            _ => format!("{backedge}block = {index}; continue;"),
        }
    }

    fn ret(&self) -> String {
        let value = if self.split.is_some() {
            "Ok(None)"
        } else {
            "Ok(())"
        };
        format!("{}{} return {value};", self.spill(), self.flags_out())
    }

    /// Hands this function's flags to whatever runs next (a callee, the
    /// caller, a tail-called function).
    fn flags_out(&self) -> String {
        match self.flags_store_line() {
            Some(store) => format!(" {store}"),
            None => String::new(),
        }
    }

    /// Takes the flags back after a call.
    fn flags_in(&self) -> Option<String> {
        if !self.needs_flags || self.direct {
            return None;
        }
        let load = if self.guest_context {
            "let nzcv = unsafe { (*ctx).nzcv };"
        } else {
            "let nzcv = recomp_abi::cpu_nzcv();"
        };
        Some(format!(
            "{{ {load} flag_n = nzcv & 8 != 0; flag_z = nzcv & 4 != 0; flag_c = nzcv & 2 != 0; flag_v = nzcv & 1 != 0; }}"
        ))
    }

    /// After a call: `ret` jumps to X30, so a callee that returned with X30
    /// pointing elsewhere (rtld's `mov x30, x0; ret`, longjmp) unwinds this
    /// frame too, until a caller's return address matches.
    fn unless_returned_to(&self, address: u64) -> String {
        format!(
            "if {} as u64 != {address:#x} {{ {} }}",
            self.read_reg("x30"),
            self.ret()
        )
    }

    fn tail_dispatch(&self, reg: &str) -> String {
        let reg = self.read_reg(reg);
        if self.split.is_some() {
            format!(
                "{}{} recomp_abi::dispatch({reg} as u64)?; return Ok(None);",
                self.spill(),
                self.flags_out()
            )
        } else {
            format!(
                "{}{} return recomp_abi::dispatch({reg} as u64);",
                self.spill(),
                self.flags_out()
            )
        }
    }
}

fn temp_declaration(name: &str) -> String {
    let ty = if name.starts_with("vtmp") {
        "u128"
    } else {
        "i64"
    };
    format!("let mut {name}: {ty} = 0;")
}

/// Temps (names that are not guest registers) used by exactly one block,
/// per block, plus the set of all such block-local temps. Lifted temps never
/// outlive the instruction that creates them, so declaring them inside their
/// block keeps live ranges local; function-wide declarations made rustc's
/// cost grow quadratically on very large lifted functions.
fn block_local_temps<'a>(
    function: &'a RustFunction,
    blocks: &[crate::pipeline::RustBlock],
) -> (Vec<Vec<String>>, std::collections::HashSet<&'a str>) {
    let temps: std::collections::HashSet<&str> = function
        .regs
        .iter()
        .map(String::as_str)
        .filter(|name| {
            crate::pipeline::reg_index(name).is_none()
                && crate::pipeline::vreg_index(name).is_none()
        })
        .collect();
    let mut users: std::collections::HashMap<&str, Vec<usize>> = std::collections::HashMap::new();
    for (index, block) in blocks.iter().enumerate() {
        let mut seen = std::collections::HashSet::new();
        let terminator_reg = match &block.terminator {
            RustTerminator::CallIndirect { reg, .. }
            | RustTerminator::BrIndirect { reg }
            | RustTerminator::Switch { reg, .. } => Some(reg.as_str()),
            _ => None,
        };
        for ident in block
            .lines
            .iter()
            .flat_map(|line| identifiers(line))
            .chain(terminator_reg)
        {
            if let Some(temp) = temps.get(ident) {
                if seen.insert(*temp) {
                    users.entry(temp).or_default().push(index);
                }
            }
        }
    }
    let mut per_block = vec![Vec::new(); blocks.len()];
    let mut local = std::collections::HashSet::new();
    for reg in &function.regs {
        if let Some([block]) = users.get(reg.as_str()).map(Vec::as_slice) {
            per_block[*block].push(reg.clone());
            local.insert(reg.as_str());
        }
    }
    (per_block, local)
}

/// Rust identifiers in a generated line (literal suffixes such as `8u64`
/// are skipped).
fn identifiers(line: &str) -> impl Iterator<Item = &str> {
    let bytes = line.as_bytes();
    let mut pos = 0;
    std::iter::from_fn(move || {
        while pos < bytes.len() {
            let start = pos;
            let ch = bytes[pos];
            if ch == b'_' || ch.is_ascii_alphabetic() {
                while pos < bytes.len()
                    && (bytes[pos] == b'_' || bytes[pos].is_ascii_alphanumeric())
                {
                    pos += 1;
                }
                let after_digit = start > 0 && bytes[start - 1].is_ascii_digit();
                if !after_digit {
                    return Some(&line[start..pos]);
                }
            } else {
                pos += 1;
            }
        }
        None
    })
}

fn emit_block_terminator(out: &mut String, terminator: &RustTerminator, exits: &Exits<'_>) {
    let mut line = |text: String| {
        out.push_str("                ");
        out.push_str(text.trim_start());
        out.push('\n');
    };
    match terminator {
        RustTerminator::Br { target } => line(exits.goto(target)),
        RustTerminator::BrCond {
            cond_expr,
            cond,
            then_label,
            else_label,
        } => match cond_expr {
            Some(expr) => line(format!(
                "if {expr} {{ {} }} else {{ {} }}",
                exits.goto(then_label),
                exits.goto(else_label)
            )),
            None => line(format!("panic!(\"unsupported condition: {cond}\");")),
        },
        RustTerminator::Call {
            call_line,
            next,
            return_address,
        } => {
            for text in exits.call_lines(call_line) {
                line(text);
            }
            if let Some(address) = return_address {
                line(exits.unless_returned_to(*address));
            }
            line(exits.goto(next));
        }
        RustTerminator::CallIndirect {
            reg,
            next,
            return_address,
        } => {
            for spill in crate::pipeline::spill_lines(exits.regs, exits.guest_context, exits.direct)
            {
                line(spill);
            }
            let flags_out = exits.flags_out();
            if !flags_out.is_empty() {
                line(flags_out);
            }
            line(format!(
                "recomp_abi::dispatch({} as u64)?;",
                exits.read_reg(reg)
            ));
            for reload in
                crate::pipeline::reload_lines(exits.regs, exits.guest_context, exits.direct)
            {
                line(reload);
            }
            if let Some(flags_in) = exits.flags_in() {
                line(flags_in);
            }
            if let Some(address) = return_address {
                line(exits.unless_returned_to(*address));
            }
            line(exits.goto(next));
        }
        RustTerminator::BrIndirect { reg } => line(exits.tail_dispatch(reg)),
        RustTerminator::Switch { reg, cases } => {
            line(format!("match {} as u64 {{", exits.read_reg(reg)));
            for (address, label) in cases {
                line(format!("{address:#x} => {{ {} }}", exits.goto(label)));
            }
            line(format!("_ => {{ {} }}", exits.tail_dispatch(reg)));
            line("}".to_string());
        }
        RustTerminator::Trap { pc, word } => line(format!(
            "return Err(recomp_abi::unsupported({pc:#x}, {word:#010x}));"
        )),
        RustTerminator::Ret => line(exits.ret()),
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            'a'..='z' | '0'..='9' => ch,
            'A'..='Z' => ch.to_ascii_lowercase(),
            _ => '-',
        })
        .collect()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    format!("{:x}", digest)
}

#[cfg(test)]
mod chunk_tests {
    use super::*;

    /// For each function, the index of the first function in its chunk.
    fn chunk_starts(functions: &[(String, usize)]) -> Vec<usize> {
        let mut starts = Vec::with_capacity(functions.len());
        let mut chunk_len = CHUNK_HEADER.len();
        let mut start = 0;
        for (index, (name, len)) in functions.iter().enumerate() {
            if starts_new_chunk(chunk_len, *len, name) {
                chunk_len = CHUNK_HEADER.len();
                start = index;
            }
            chunk_len += len + 1;
            starts.push(start);
        }
        starts
    }

    #[test]
    fn a_grown_function_reshuffles_only_nearby_chunks() {
        let mut functions: Vec<(String, usize)> = (0..40_000u64)
            .map(|i| {
                let size = 1_000 + (i as usize * 7_919) % 40_000;
                (format!("fn_{:016x}", 0x0800_0000 + i * 0x40), size)
            })
            .collect();
        let before = chunk_starts(&functions);
        functions[20_000].1 += 200_000;
        let after = chunk_starts(&functions);
        let moved = before.iter().zip(&after).filter(|(a, b)| a != b).count();
        let chunks = before
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let per_chunk = functions.len() / chunks;
        assert!(chunks > 100, "{chunks} chunks");
        assert!(
            moved <= 3 * per_chunk,
            "{moved} functions changed chunk (about {per_chunk} per chunk)"
        );
    }

    #[test]
    fn cargo_paths_use_forward_slashes() {
        assert_eq!(
            toml_path(Path::new(r"C:\SwitchRecomp\crates\recomp-runtime")),
            "\"C:/SwitchRecomp/crates/recomp-runtime\""
        );
        let program = context_program("entry", "entry");
        let cargo = emit_cargo_toml(&program, Path::new("/toolchain/crates/recomp-runtime"));
        assert!(
            cargo.contains("recomp-runtime = { path = \"/toolchain/crates/recomp-runtime\" }"),
            "{cargo}"
        );
        assert!(
            cargo.contains("recomp-abi = { path = \"/toolchain/crates/recomp-abi\" }"),
            "{cargo}"
        );
        assert!(!cargo.contains('\\'), "{cargo}");
    }

    #[test]
    fn cargo_paths_escape_toml_strings() {
        let path = Path::new("C:\\SwitchRecomp\\a \"title\"\\runtime");
        assert_eq!(
            toml_path(path),
            "\"C:/SwitchRecomp/a \\\"title\\\"/runtime\""
        );
        assert_eq!(toml_string("a\\b\tc"), "\"a\\\\b\\tc\"");
        assert_eq!(toml_string("line\nnext"), "\"line\\nnext\"");
    }

    #[test]
    fn an_absolute_runtime_path_is_a_safe_cargo_value() {
        // Pathdiff returns None across Windows drives; the absolute fallback
        // must still be a valid TOML value with `/` separators.
        let program = context_program("entry", "entry");
        let cargo = emit_cargo_toml(&program, Path::new(r"D:\toolchain\crates\recomp-runtime"));
        assert!(
            cargo.contains("recomp-runtime = { path = \"D:/toolchain/crates/recomp-runtime\" }"),
            "{cargo}"
        );
        assert!(!cargo.contains(r"\t"), "{cargo}");
    }

    #[test]
    fn chunk_manifests_use_forward_slash_abi_paths() {
        let cargo = emit_chunk_cargo_toml("chunk_0", Path::new("toolchain/crates/recomp-runtime"));
        assert!(
            cargo.contains("recomp-abi = { path = \"../../toolchain/crates/recomp-abi\" }"),
            "{cargo}"
        );
        assert!(!cargo.contains('\\'), "{cargo}");
    }

    #[test]
    fn script_default_resolves_through_the_environment_first() {
        let mut program = context_program("entry", "entry");
        program.script = Some(crate::config::RuntimeAsset {
            relative: Some("scripts/game.rhai".to_string()),
            fallback: "/build/config/scripts/game.rhai".to_string(),
        });
        let mut main = String::new();
        emit_runtime_defaults(&mut main, &program);
        assert!(
            main.contains(
                "recomp_runtime::resolve_asset(\"RECOMP_SCRIPT\", Some(\"scripts/game.rhai\"), Some(\"/build/config/scripts/game.rhai\"))"
            ),
            "{main}"
        );
    }

    #[test]
    fn romfs_default_resolves_through_the_environment_first() {
        let mut program = context_program("entry", "entry");
        program.ipc = Some(crate::config::IpcConfig {
            romfs_base: 0x4800,
            romfs_size: 0x101,
            romfs_image: Some(crate::config::RuntimeAsset {
                relative: Some("romfs.bin".to_string()),
                fallback: "/build/assets/romfs.bin".to_string(),
            }),
            ipc_buffer: Some(0x2000),
        });
        let mut main = String::new();
        emit_ipc_init(&mut main, &program);
        assert!(
            main.contains(
                "recomp_runtime::resolve_asset(\"RECOMP_ROMFS\", Some(\"romfs.bin\"), Some(\"/build/assets/romfs.bin\"))"
            ),
            "{main}"
        );
        assert!(
            main.contains("init_ipc_services(0x4800, 0x101, romfs_image.as_deref(), Some(0x2000))"),
            "{main}"
        );
    }

    #[test]
    fn a_project_without_ipc_uses_romfs_only_when_the_environment_sets_one() {
        let program = context_program("entry", "entry");
        let mut main = String::new();
        emit_ipc_init(&mut main, &program);
        assert!(
            main.contains(
                "let romfs_image = recomp_runtime::resolve_asset(\"RECOMP_ROMFS\", None, None);"
            ),
            "{main}"
        );
        assert!(
            main.contains("if romfs_image.is_some() {\n        recomp_runtime::init_ipc_services(0, 0, romfs_image.as_deref(), None)?;"),
            "{main}"
        );
    }

    #[test]
    fn segments_resolve_beside_the_executable_with_the_build_path_as_fallback() {
        let image = MemoryImageDescriptor {
            init_segments: vec![crate::memory::MemoryInitSegmentDescriptor {
                name: "data".to_string(),
                base: 0x1000,
                size: 4,
                init_path: "segments/data-0.bin".to_string(),
            }],
            zero_segments: Vec::new(),
        };
        let mut main = String::new();
        emit_memory_image_init(&mut main, Some(&image));
        assert!(
            main.contains(
                "recomp_runtime::resolve_relative_asset(\"segments/data-0.bin\", Some(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/\", \"segments/data-0.bin\")))"
            ),
            "{main}"
        );
        assert!(!main.contains("std::fs::read(concat!"), "{main}");
    }

    #[test]
    fn host_functions_are_tried_before_the_lifted_code() {
        let function = RustFunction {
            name: "fn_0000000000001000".to_string(),
            host: Some("zlib_inflate_fast".to_string()),
            regs: vec!["x0".to_string()],
            needs_flags: false,
            body: FunctionBody::Linear(vec!["x0 = 1;".to_string()]),
        };
        let code = emit_function(
            &function,
            true,
            &ClobberAnalysis::default(),
            false,
            CodegenOptions::default(),
        );
        let host = code
            .find("__recomp_host_zlib_inflate_fast() }?")
            .expect("host call");
        let lifted = code.find("x0 = 1;").expect("lifted body");
        assert!(host < lifted, "{code}");
        assert!(code.contains("return Ok(());"));
    }

    #[test]
    fn flags_a_callee_sets_reach_the_caller() {
        // bl helper (cmp; ret); b.hi overflow: the branch reads the
        // callee's flags.
        let block = |label: &str, terminator| crate::pipeline::RustBlock {
            label: label.to_string(),
            lines: Vec::new(),
            terminator,
        };
        let function = RustFunction {
            name: "fn_0000000000001000".to_string(),
            host: None,
            regs: vec!["x30".to_string()],
            needs_flags: true,
            body: FunctionBody::Blocks(vec![
                block(
                    "bb_1000",
                    RustTerminator::Call {
                        call_line: "recomp_abi::dispatch(0x2000)?;".to_string(),
                        next: "bb_1004".to_string(),
                        return_address: Some(0x1004),
                    },
                ),
                block(
                    "bb_1004",
                    RustTerminator::BrCond {
                        cond_expr: Some("flag_c && !flag_z".to_string()),
                        cond: "hi".to_string(),
                        then_label: "bb_1008".to_string(),
                        else_label: "bb_1008".to_string(),
                    },
                ),
                block("bb_1008", RustTerminator::Ret),
            ]),
        };
        let code = emit_function(
            &function,
            true,
            &ClobberAnalysis::default(),
            false,
            CodegenOptions::default(),
        );
        let entry = code
            .find("let nzcv = recomp_abi::cpu_nzcv();")
            .expect("entry");
        let store = code.find("recomp_abi::cpu_set_nzcv(").expect("store");
        let call = code.find("dispatch(0x2000)").expect("call");
        let reload = code
            .find("{ let nzcv = recomp_abi::cpu_nzcv();")
            .expect("reload");
        let branch = code.find("if flag_c && !flag_z").expect("branch");
        assert!(
            entry < store && store < call && call < reload && reload < branch,
            "{code}"
        );
        let ret = code.rfind("return Ok(());").expect("return");
        assert!(
            code[branch..ret].contains("recomp_abi::cpu_set_nzcv("),
            "{code}"
        );
    }

    #[test]
    fn calls_to_known_functions_go_straight_to_them() {
        let functions = std::collections::HashMap::from([
            (0x2000, "fn_0000000000002000".to_string()),
            (0x3000, "fn_0000000000003000".to_string()),
        ]);
        let mut called = std::collections::BTreeSet::new();
        let code = "recomp_abi::dispatch(0x2000)?;\nrecomp_abi::dispatch(0x2400)?;\nrecomp_abi::dispatch(x2 as u64)?;\nrecomp_abi::dispatch(0x3000)?;";
        let rewritten = direct_calls(code, &functions, &mut called, false);
        assert_eq!(
            rewritten,
            "recomp_abi::call(0x2000, || unsafe { fn_0000000000002000() })?;\n\
             recomp_abi::dispatch(0x2400)?;\n\
             recomp_abi::dispatch(x2 as u64)?;\n\
             recomp_abi::call(0x3000, || unsafe { fn_0000000000003000() })?;"
        );
        assert_eq!(
            called.into_iter().collect::<Vec<_>>(),
            ["fn_0000000000002000", "fn_0000000000003000"]
        );
    }

    #[test]
    fn a_chunk_declares_the_functions_it_calls_elsewhere() {
        let mut calls = ChunkCalls::default();
        calls.defined.insert("fn_0000000000002000".to_string());
        calls.called.insert("fn_0000000000002000".to_string());
        calls.called.insert("fn_0000000000003000".to_string());
        let lib_rs = format!("{CHUNK_HEADER}#[no_mangle]\npub fn fn_0000000000002000() {{}}\n");
        let with_externs = calls.with_externs(&lib_rs, false);
        assert_eq!(
            with_externs,
            format!(
                "{CHUNK_HEADER}extern \"Rust\" {{\n    fn fn_0000000000003000() -> recomp_abi::GuestResult;\n}}\n\n#[no_mangle]\npub fn fn_0000000000002000() {{}}\n"
            )
        );
        assert_eq!(ChunkCalls::default().with_externs(&lib_rs, false), lib_rs);
        let with_context = calls.with_externs(&lib_rs, true);
        assert_eq!(
            with_context,
            format!(
                "{CHUNK_HEADER}extern \"Rust\" {{\n    fn fn_0000000000003000(ctx: *mut recomp_abi::GuestContext) -> recomp_abi::GuestResult;\n}}\n\n#[no_mangle]\npub fn fn_0000000000002000() {{}}\n"
            )
        );
    }

    #[test]
    fn native_target_cpu_becomes_generic_for_cross_architectures() {
        assert_eq!(resolved_target_cpu(None, "native"), "native");
        assert_eq!(
            resolved_target_cpu(Some("aarch64-linux-android"), "native"),
            "generic"
        );
        assert_eq!(
            resolved_target_cpu(Some("aarch64-unknown-linux-gnu"), "native"),
            "generic"
        );
        assert_eq!(
            resolved_target_cpu(Some("aarch64-linux-android"), "cortex-a76"),
            "cortex-a76"
        );
        let x86 = resolved_target_cpu(Some("x86_64-unknown-linux-gnu"), "native");
        if cfg!(target_arch = "x86_64") {
            assert_eq!(x86, "native");
        } else {
            assert_eq!(x86, "generic");
        }
    }

    fn context_program(entry: &str, function: &str) -> RustProgram {
        RustProgram {
            title: "Context".to_string(),
            abi_version: "0.1.0".to_string(),
            entry: entry.to_string(),
            functions: vec![RustFunction {
                name: function.to_string(),
                host: None,
                regs: Vec::new(),
                needs_flags: false,
                body: FunctionBody::Linear(vec!["return Ok(());".to_string()]),
            }],
            performance_mode: crate::config::PerformanceMode::Handheld,
            graphics: crate::config::GraphicsBackend::Recording,
            parallel: false,
            block_tracking: true,
            target_cpu: None,
            target: None,
            guest_context: true,
            direct_registers: false,
            script: None,
            memory_layout: MemoryLayoutDescriptor::minimal_default(),
            memory_image: None,
            ipc: None,
            initial_sp: None,
            entry_address: None,
        }
    }

    #[test]
    fn guest_context_chunked_targets_wrap_functions() {
        let program = context_program("entry", "fn_0000000000001000");
        let mut out = String::new();
        emit_dispatch_chunked(&mut out, &program, &[0]);
        assert!(
            out.contains(
                "fn __recomp_entry_0000000000001000() -> recomp_abi::GuestResult {\n    unsafe { chunk_0::fn_0000000000001000(recomp_abi::guest_context()) }\n}"
            ),
            "{out}"
        );
        assert!(
            out.contains("(0x0000000000001000, __recomp_entry_0000000000001000),"),
            "{out}"
        );
    }

    #[test]
    fn guest_context_chunked_entry_takes_the_context() {
        let program = context_program("entry", "entry");
        let main = emit_main_rs_chunked(&program, &[0]);
        assert!(
            main.contains("unsafe { chunk_0::entry(recomp_abi::guest_context()) }?;"),
            "{main}"
        );
    }

    #[test]
    fn windows_release_builds_can_hide_the_console() {
        let mut program = context_program("entry", "entry");
        program.graphics = crate::config::GraphicsBackend::Vulkan;
        program.script = Some(crate::config::RuntimeAsset {
            relative: Some("scripts/game.rhai".to_string()),
            fallback: "/build/scripts/game.rhai".to_string(),
        });
        program.ipc = Some(crate::config::IpcConfig {
            romfs_base: 0,
            romfs_size: 0,
            romfs_image: Some(crate::config::RuntimeAsset {
                relative: Some("romfs".to_string()),
                fallback: "/build/romfs".to_string(),
            }),
            ipc_buffer: None,
        });
        let main = emit_main_rs_chunked(&program, &[0]);
        assert!(
            main.starts_with(
                "#![allow(warnings)]\n#![cfg_attr(all(windows, feature = \"windows-subsystem\"), windows_subsystem = \"windows\")]\n"
            ),
            "{main}"
        );
        assert!(
            main.contains(
                "recomp_runtime::resolve_asset(\"RECOMP_SCRIPT\", Some(\"scripts/game.rhai\"), Some(\"/build/scripts/game.rhai\"))"
            ),
            "{main}"
        );
        assert!(
            main.contains(
                "recomp_runtime::resolve_asset(\"RECOMP_ROMFS\", Some(\"romfs\"), Some(\"/build/romfs\"))"
            ),
            "{main}"
        );
        let workspace = emit_workspace_cargo_toml(&program, Path::new("../recomp-runtime"), 1, &[]);
        assert!(
            workspace.contains("[features]\ndefault = []\nwindows-subsystem = []\n"),
            "{workspace}"
        );
        let single = emit_cargo_toml(&program, Path::new("../recomp-runtime"));
        assert!(
            single.contains("[features]\ndefault = []\nwindows-subsystem = []\n"),
            "{single}"
        );
    }

    #[test]
    fn guest_context_split_parts_share_the_context() {
        let blocks = (0..MAX_PART_BLOCKS + 1)
            .map(|index| crate::pipeline::RustBlock {
                label: format!("bb_{:x}", 0x1000 + index * 4),
                lines: Vec::new(),
                terminator: RustTerminator::Ret,
            })
            .collect::<Vec<_>>();
        let function = RustFunction {
            name: "fn_0000000000001000".to_string(),
            host: None,
            regs: vec!["x0".to_string(), "x30".to_string()],
            needs_flags: true,
            body: FunctionBody::Blocks(blocks),
        };
        let code = emit_function(
            &function,
            true,
            &ClobberAnalysis::default(),
            true,
            CodegenOptions::default(),
        );
        assert!(
            code.contains(
                "unsafe fn fn_0000000000001000(ctx: *mut recomp_abi::GuestContext) -> recomp_abi::GuestResult"
            ),
            "{code}"
        );
        assert!(
            code.contains(
                "unsafe fn fn_0000000000001000__p0(ctx: *mut recomp_abi::GuestContext, entry: u32, nzcv: u8)"
            ),
            "{code}"
        );
        assert!(
            code.contains("Some((0, 0, unsafe { (*ctx).nzcv }))"),
            "{code}"
        );
        assert!(
            code.contains("unsafe { fn_0000000000001000__p0(ctx, block, nzcv)? }"),
            "{code}"
        );
        assert!(code.contains("unsafe { (*ctx).x[30] = x30; }"), "{code}");
        // A part takes the incoming flags from its parameter; only the
        // trampoline reads them from the context.
        let part = code
            .find("__p0(ctx: *mut")
            .map(|at| &code[at..])
            .expect("part function");
        assert!(!part.contains("let nzcv = unsafe"), "{code}");
    }
}

#[cfg(test)]
mod clobber_tests {
    use super::*;

    fn function(name: &str, regs: &[&str], body: FunctionBody) -> RustFunction {
        RustFunction {
            name: name.to_string(),
            host: None,
            regs: regs.iter().map(|reg| (*reg).to_string()).collect(),
            needs_flags: false,
            body,
        }
    }

    fn block(
        label: &str,
        lines: &[&str],
        terminator: RustTerminator,
    ) -> crate::pipeline::RustBlock {
        crate::pipeline::RustBlock {
            label: label.to_string(),
            lines: lines.iter().map(|line| (*line).to_string()).collect(),
            terminator,
        }
    }

    fn call(target: u64, next: &str, return_address: Option<u64>) -> RustTerminator {
        RustTerminator::Call {
            call_line: format!("recomp_abi::dispatch({target:#x})?;"),
            next: next.to_string(),
            return_address,
        }
    }

    fn ret() -> RustTerminator {
        RustTerminator::Ret
    }

    fn between<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
        let at = text
            .find(start)
            .unwrap_or_else(|| panic!("missing {start:?}: {text}"));
        let rest = &text[at + start.len()..];
        let to = rest
            .find(end)
            .unwrap_or_else(|| panic!("missing {end:?}: {text}"));
        &rest[..to]
    }

    fn fast_path(code: &str) -> &str {
        between(code, "if recomp_abi::direct_calls_enabled() {", "} else {")
    }

    fn slow_path(code: &str) -> &str {
        let at = code.find("} else {").expect("slow path");
        &code[at + "} else {".len()..]
    }

    fn leaf_callee() -> RustFunction {
        function(
            "fn_0000000000002000",
            &["x0", "x1"],
            FunctionBody::Linear(vec!["x0 = x1;".to_string(), "return Ok(());".to_string()]),
        )
    }

    fn calling_caller(regs: &[&str]) -> RustFunction {
        let mut regs = regs.to_vec();
        if !regs.contains(&"x30") {
            regs.push("x30");
        }
        function(
            "fn_0000000000001000",
            &regs,
            FunctionBody::Blocks(vec![
                block("bb_1000", &[], call(0x2000, "bb_1004", Some(0x1004))),
                block("bb_1004", &[], ret()),
            ]),
        )
    }

    #[test]
    fn a_leaf_call_spills_only_its_inputs_and_reloads_only_its_outputs() {
        let functions = [calling_caller(&["x0", "x1", "x9"]), leaf_callee()];
        let analysis = ClobberAnalysis::from_functions(&functions);
        let target = analysis
            .call_target("recomp_abi::dispatch(0x2000)?;")
            .expect("known callee");
        assert!(target.required.contains("x1"), "{:?}", target.required);
        assert!(!target.required.contains("x0"), "{:?}", target.required);
        assert!(target.clobber.contains("x0"), "{:?}", target.clobber);
        assert!(!target.clobber.contains("x1"), "{:?}", target.clobber);

        let code = emit_function(
            &functions[0],
            true,
            &analysis,
            false,
            CodegenOptions::default(),
        );
        let fast = fast_path(&code);
        assert!(
            fast.contains("recomp_abi::cpu_store(&[(0, x0), (1, x1)]);"),
            "{code}"
        );
        assert!(!fast.contains("x9"), "{code}");
        assert!(
            fast.contains("recomp_abi::call_direct(|| unsafe { fn_0000000000002000() })?;"),
            "{code}"
        );
        assert!(fast.contains("x0 = __r[0];"), "{code}");
        assert!(!fast.contains("x9 = __r[9];"), "{code}");

        // The fallback still writes the whole register file for hooks and
        // serialized calls.
        let slow = slow_path(&code);
        assert!(slow.contains("(9, x9)"), "{code}");
        assert!(slow.contains("x9 = __r[9];"), "{code}");
        // `direct_calls` rewrites this to `recomp_abi::call` when it emits
        // the function; the slow path is the one that keeps the dispatch.
        assert!(slow.contains("recomp_abi::dispatch(0x2000)?;"), "{code}");
    }

    #[test]
    fn guest_context_calls_keep_the_selective_masks() {
        let functions = [calling_caller(&["x0", "x1", "x9"]), leaf_callee()];
        let analysis = ClobberAnalysis::from_functions(&functions);
        let code = emit_function(
            &functions[0],
            true,
            &analysis,
            true,
            CodegenOptions::default(),
        );

        let fast = fast_path(&code);
        assert!(fast.contains("unsafe { (*ctx).x[0] = x0; }"), "{code}");
        assert!(fast.contains("unsafe { (*ctx).x[1] = x1; }"), "{code}");
        assert!(!fast.contains("x9"), "{code}");
        assert!(
            fast.contains("recomp_abi::call_direct(|| unsafe { fn_0000000000002000(ctx) })?;"),
            "{code}"
        );
        assert!(fast.contains("x0 = unsafe { (*ctx).x[0] };"), "{code}");
        assert!(!fast.contains("x9 = unsafe"), "{code}");

        // The fallback still writes the whole register file for hooks and
        // serialized calls; only its storage is the context now.
        let slow = slow_path(&code);
        assert!(slow.contains("unsafe { (*ctx).x[9] = x9; }"), "{code}");
        assert!(slow.contains("recomp_abi::dispatch(0x2000)?;"), "{code}");
        assert!(slow.contains("x9 = unsafe { (*ctx).x[9] };"), "{code}");
        assert!(!code.contains("recomp_abi::cpu_regs()"), "{code}");
        assert!(!code.contains("recomp_abi::cpu_store"), "{code}");
    }

    #[test]
    fn guest_context_calls_handle_flags_and_vectors() {
        let caller = RustFunction {
            name: "fn_0000000000001000".to_string(),
            host: None,
            regs: ["x0", "v0", "v1", "x30"]
                .iter()
                .map(|reg| (*reg).to_string())
                .collect(),
            needs_flags: true,
            body: FunctionBody::Blocks(vec![
                block("bb_1000", &[], call(0x2000, "bb_1004", Some(0x1004))),
                block("bb_1004", &[], ret()),
            ]),
        };
        let mut callee = function(
            "fn_0000000000002000",
            &["x0", "v0"],
            FunctionBody::Linear(vec![
                "v0 = 1;".to_string(),
                "x0 = 2;".to_string(),
                "return Ok(());".to_string(),
            ]),
        );
        callee.needs_flags = true;
        let functions = [caller, callee];
        let analysis = ClobberAnalysis::from_functions(&functions);
        let code = emit_function(
            &functions[0],
            true,
            &analysis,
            true,
            CodegenOptions::default(),
        );
        let fast = fast_path(&code);
        assert!(fast.contains("unsafe { (*ctx).v[0] = v0; }"), "{code}");
        assert!(fast.contains("v0 = unsafe { (*ctx).v[0] };"), "{code}");
        assert!(!fast.contains("v[1]"), "{code}");
        assert!(fast.contains("unsafe { (*ctx).nzcv ="), "{code}");
        assert!(
            fast.contains("let nzcv = unsafe { (*ctx).nzcv };"),
            "{code}"
        );
    }

    #[test]
    fn a_callees_write_reaches_the_caller() {
        let callee = function(
            "fn_0000000000002000",
            &["x0", "x9"],
            FunctionBody::Linear(vec![
                "x0 = 1;".to_string(),
                "x9 = x0;".to_string(),
                "return Ok(());".to_string(),
            ]),
        );
        let functions = [calling_caller(&["x0", "x9"]), callee];
        let analysis = ClobberAnalysis::from_functions(&functions);
        let target = analysis
            .call_target("recomp_abi::dispatch(0x2000)?;")
            .expect("known callee");
        assert!(target.clobber.contains("x9"), "{:?}", target.clobber);

        let code = emit_function(
            &functions[0],
            true,
            &analysis,
            false,
            CodegenOptions::default(),
        );
        let fast = fast_path(&code);
        let call_at = fast.find("call_direct").expect("direct call");
        let reload_at = fast.find("x9 = __r[9];").expect("x9 reload");
        assert!(call_at < reload_at, "{code}");
    }

    #[test]
    fn reads_before_writes_are_the_only_registers_a_callee_needs() {
        // x5 is written in the first block, so the second block's read needs
        // nothing from the caller; x6 was never written.
        let callee = function(
            "fn_0000000000002000",
            &["x0", "x5", "x6"],
            FunctionBody::Blocks(vec![
                block(
                    "bb_2000",
                    &["x5 = 11;"],
                    RustTerminator::Br {
                        target: "bb_2004".to_string(),
                    },
                ),
                block("bb_2004", &["x0 = x5.wrapping_add(x6);"], ret()),
            ]),
        );
        let functions = [calling_caller(&["x0"]), callee];
        let analysis = ClobberAnalysis::from_functions(&functions);
        let target = analysis
            .call_target("recomp_abi::dispatch(0x2000)?;")
            .expect("known callee");
        assert!(target.required.contains("x6"), "{:?}", target.required);
        assert!(!target.required.contains("x5"), "{:?}", target.required);
        assert!(target.clobber.contains("x0"), "{:?}", target.clobber);
        assert!(target.clobber.contains("x5"), "{:?}", target.clobber);
    }

    #[test]
    fn recursive_callees_reach_a_fixed_point() {
        let first = function(
            "fn_0000000000002000",
            &["x0"],
            FunctionBody::Blocks(vec![
                block("bb_2000", &["x0 = 5;"], call(0x3000, "bb_2004", None)),
                block("bb_2004", &[], ret()),
            ]),
        );
        let second = function(
            "fn_0000000000003000",
            &["x3", "x7"],
            FunctionBody::Blocks(vec![
                block("bb_3000", &["x3 = 7;"], call(0x2000, "bb_3004", None)),
                block("bb_3004", &[], ret()),
            ]),
        );
        let analysis = ClobberAnalysis::from_functions(&[first, second]);
        let first_target = analysis
            .call_target("recomp_abi::dispatch(0x3000)?;")
            .expect("known callee");
        let second_target = analysis
            .call_target("recomp_abi::dispatch(0x2000)?;")
            .expect("known callee");
        // The cycle carries x3 (written by the second function) back into
        // the first function's clobbers, and x0 the other way.
        assert!(
            first_target.clobber.contains("x3"),
            "{:?}",
            first_target.clobber
        );
        assert!(
            second_target.clobber.contains("x0"),
            "{:?}",
            second_target.clobber
        );
        assert!(
            !first_target.clobber.contains("x7"),
            "{:?}",
            first_target.clobber
        );
    }

    #[test]
    fn a_callee_that_may_write_a_register_stores_the_callers_value_first() {
        // The callee writes x9 on one path and returns the value it loaded
        // on the other, so the caller has to put its current x9 in the
        // register file even though the callee never reads x9.
        let callee = function(
            "fn_0000000000002000",
            &["x0", "x9"],
            FunctionBody::Blocks(vec![
                block(
                    "bb_2000",
                    &[],
                    RustTerminator::BrCond {
                        cond_expr: Some("x0 != 0".to_string()),
                        cond: "ne".to_string(),
                        then_label: "bb_2004".to_string(),
                        else_label: "bb_2008".to_string(),
                    },
                ),
                block("bb_2004", &["x9 = 5;"], ret()),
                block("bb_2008", &[], ret()),
            ]),
        );
        let functions = [calling_caller(&["x0", "x9"]), callee];
        let analysis = ClobberAnalysis::from_functions(&functions);
        let target = analysis
            .call_target("recomp_abi::dispatch(0x2000)?;")
            .expect("known callee");
        assert!(target.required.contains("x0"), "{:?}", target.required);
        assert!(!target.required.contains("x9"), "{:?}", target.required);
        assert!(target.clobber.contains("x9"), "{:?}", target.clobber);

        let code = emit_function(
            &functions[0],
            true,
            &analysis,
            false,
            CodegenOptions::default(),
        );
        let fast = fast_path(&code);
        assert!(
            fast.contains("recomp_abi::cpu_store(&[(0, x0), (9, x9)]);"),
            "{code}"
        );
        assert!(fast.contains("x9 = __r[9];"), "{code}");
    }

    #[test]
    fn flags_travel_across_the_fast_path_when_the_callee_uses_them() {
        let mut callee = leaf_callee();
        callee.needs_flags = true;
        let mut caller = calling_caller(&["x0"]);
        caller.needs_flags = true;
        let functions = [caller, callee];
        let analysis = ClobberAnalysis::from_functions(&functions);
        let target = analysis
            .call_target("recomp_abi::dispatch(0x2000)?;")
            .expect("known callee");
        assert!(target.required.flags);
        assert!(target.clobber.flags);

        let code = emit_function(
            &functions[0],
            true,
            &analysis,
            false,
            CodegenOptions::default(),
        );
        let fast = fast_path(&code);
        let store = fast.find("recomp_abi::cpu_set_nzcv(").expect("flag store");
        let call = fast.find("call_direct").expect("direct call");
        let reload = fast
            .find("let nzcv = recomp_abi::cpu_nzcv()")
            .expect("flag reload");
        assert!(store < call && call < reload, "{code}");
    }

    #[test]
    fn vector_register_reads_and_writes_use_the_same_masks() {
        let callee = function(
            "fn_0000000000002000",
            &["v1", "v2"],
            FunctionBody::Linear(vec!["v1 = v2;".to_string(), "return Ok(());".to_string()]),
        );
        let functions = [calling_caller(&["v0", "v1", "v2", "v3"]), callee];
        let analysis = ClobberAnalysis::from_functions(&functions);
        let target = analysis
            .call_target("recomp_abi::dispatch(0x2000)?;")
            .expect("known callee");
        assert!(target.required.contains("v2"), "{:?}", target.required);
        assert!(target.clobber.contains("v1"), "{:?}", target.clobber);

        let code = emit_function(
            &functions[0],
            true,
            &analysis,
            false,
            CodegenOptions::default(),
        );
        let fast = fast_path(&code);
        assert!(
            fast.contains("recomp_abi::cpu_vstore(&[(1, v1), (2, v2)]);"),
            "{code}"
        );
        assert!(fast.contains("v1 = __v[1];"), "{code}");
        assert!(!fast.contains("v0") && !fast.contains("v3"), "{code}");
    }

    #[test]
    fn an_unknown_callee_keeps_the_full_spill() {
        let indirect = function(
            "fn_0000000000002000",
            &["x0", "x9"],
            FunctionBody::Blocks(vec![
                block(
                    "bb_2000",
                    &[],
                    RustTerminator::CallIndirect {
                        reg: "x2".to_string(),
                        next: "bb_2004".to_string(),
                        return_address: None,
                    },
                ),
                block("bb_2004", &[], ret()),
            ]),
        );
        let functions = [calling_caller(&["x0", "x9"]), indirect];
        let analysis = ClobberAnalysis::from_functions(&functions);
        assert!(analysis
            .call_target("recomp_abi::dispatch(0x2000)?;")
            .is_none());
        let code = emit_function(
            &functions[0],
            true,
            &analysis,
            false,
            CodegenOptions::default(),
        );
        assert!(!code.contains("direct_calls_enabled"), "{code}");
        assert!(!code.contains("call_direct"), "{code}");
        assert!(code.contains("(0, x0), (9, x9), (30, x30)"), "{code}");
        assert!(code.contains("x9 = __r[9];"), "{code}");
    }

    #[test]
    fn a_callee_that_ignores_flags_does_not_force_a_save_on_the_fast_path() {
        let mut caller = calling_caller(&["x0"]);
        caller.needs_flags = true;
        let functions = [caller, leaf_callee()];
        let analysis = ClobberAnalysis::from_functions(&functions);
        let code = emit_function(
            &functions[0],
            true,
            &analysis,
            false,
            CodegenOptions::default(),
        );
        let fast = fast_path(&code);
        assert!(!fast.contains("cpu_set_nzcv"), "{code}");
        assert!(
            !fast.contains("let nzcv = recomp_abi::cpu_nzcv()"),
            "{code}"
        );
        // The slow path still saves the caller's flags as before.
        assert!(slow_path(&code).contains("cpu_set_nzcv"), "{code}");
    }
}

#[cfg(test)]
mod direct_register_tests {
    use super::*;

    #[test]
    fn context_field_accesses_count_as_guest_register_reads_and_writes() {
        let (reads, writes, definite) = scan_line(
            "unsafe { (*ctx).x[2] = (unsafe { (*ctx).x[0] }).wrapping_add((unsafe { (*ctx).x[1] })); }",
        );
        assert!(reads.contains("x0") && reads.contains("x1"), "{reads:?}");
        assert!(
            writes.contains("x2") && definite.contains("x2"),
            "{writes:?}"
        );

        let (reads, writes, definite) =
            scan_line("unsafe { (*ctx).v[3] = (unsafe { (*ctx).v[1] }); }");
        assert!(reads.contains("v1"), "{reads:?}");
        assert!(
            writes.contains("v3") && definite.contains("v3"),
            "{writes:?}"
        );

        let (reads, writes, _) = scan_line("unsafe { (*ctx).nzcv = __f; }");
        assert!(!reads.flags, "{reads:?}");
        assert!(writes.flags, "{writes:?}");

        let (reads, writes, _) = scan_line("if ((unsafe { (*ctx).nzcv } & 4) != 0) { }");
        assert!(reads.flags, "{reads:?}");
        assert!(!writes.flags, "{writes:?}");
    }
}
