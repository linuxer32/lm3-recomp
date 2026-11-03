use recomp_pipeline::config::TitleConfig;
use recomp_pipeline::input::Module;
use recomp_pipeline::memory::{
    MemoryLayoutDescriptor, MemoryPermissionsDescriptor, MemoryRegionDescriptor,
};
use recomp_pipeline::{run_pipeline, PipelineOptions};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

const MODULE_JSON: &str = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "ops": [
        { "op": "const_i64", "dst": "x0", "imm": 7 },
        { "op": "const_i64", "dst": "x1", "imm": 35 },
        { "op": "add_i64", "dst": "x2", "lhs": "x0", "rhs": "x1" },
        { "op": "syscall", "name": "svc_log", "args": ["x2"] },
        { "op": "syscall", "name": "svcGetSystemTick", "args": [] },
        { "op": "ret" }
      ]
    }
  ]
}"#;

const CONFIG_TOML: &str = r#"
title = "Minimal Sample"
entry = "entry"
abi_version = "0.1.0"

[runtime]
performance_mode = "handheld"

[runtime.memory_layout]
[[runtime.memory_layout.regions]]
name = "code"
base = 0x1000_0000
size = 0x0001_0000
permissions = { read = true, write = false, execute = true }

[[runtime.memory_layout.regions]]
name = "data"
base = 0x2000_0000
size = 0x0004_0000
permissions = { read = true, write = true, execute = false }

[stubs]
svc_log = "log"
"#;

const CONFIG_TOML_DEFAULT_LAYOUT: &str = r#"
title = "Minimal Sample"
entry = "entry"
abi_version = "0.1.0"

[stubs]
svc_log = "log"
"#;

const CONFIG_TOML_OVERLAP_LAYOUT: &str = r#"
title = "Overlap Sample"
entry = "entry"
abi_version = "0.1.0"

[runtime]
performance_mode = "handheld"

[runtime.memory_layout]
[[runtime.memory_layout.regions]]
name = "first"
base = 0x1000
size = 0x200
permissions = { read = true, write = false, execute = true }

[[runtime.memory_layout.regions]]
name = "second"
base = 0x1100
size = 0x200
permissions = { read = true, write = false, execute = false }
"#;

const CONFIG_TOML_ZERO_LAYOUT: &str = r#"
title = "Zero Sample"
entry = "entry"
abi_version = "0.1.0"

[runtime.memory_layout]
[[runtime.memory_layout.regions]]
name = "zero"
base = 0x1000
size = 0x0
permissions = { read = true, write = false, execute = false }
"#;

#[test]
fn parse_title_config() {
    let config = TitleConfig::parse(CONFIG_TOML).expect("config parses");
    assert_eq!(config.title, "Minimal Sample");
    assert_eq!(config.entry, "entry");
    assert_eq!(config.abi_version, "0.1.0");
    assert!(config.stubs.contains_key("svc_log"));
    assert_eq!(config.memory_layout.regions.len(), 2);
}

#[test]
fn parse_title_config_block_tracking() {
    // On by default: fault reports and RECOMP_WATCH_STORE see the block.
    let config = TitleConfig::parse(CONFIG_TOML_DEFAULT_LAYOUT).expect("config parses");
    assert!(config.runtime.block_tracking);
    let config = TitleConfig::parse(&format!(
        "{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\nblock_tracking = true\n"
    ))
    .expect("config parses");
    assert!(config.runtime.block_tracking);
    let config = TitleConfig::parse(&format!(
        "{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\nblock_tracking = false\n"
    ))
    .expect("config parses");
    assert!(!config.runtime.block_tracking);
}

#[test]
fn parse_title_config_host_functions() {
    let config = TitleConfig::parse(&format!(
        "{CONFIG_TOML_DEFAULT_LAYOUT}\n[host_functions]\nfn_00000000090415a0 = \"zlib_inflate_fast\"\n"
    ))
    .expect("config parses");
    assert_eq!(
        config.host_functions["fn_00000000090415a0"],
        "zlib_inflate_fast"
    );
    let err = TitleConfig::parse(&format!(
        "{CONFIG_TOML_DEFAULT_LAYOUT}\n[host_functions]\nfn_1 = \"Not An Identifier\"\n"
    ))
    .unwrap_err();
    assert!(err.contains("not a lowercase identifier"), "{err}");
}

#[test]
fn parse_title_config_defaults_layout() {
    let config = TitleConfig::parse(CONFIG_TOML_DEFAULT_LAYOUT).expect("config parses");
    assert_eq!(config.memory_layout.regions.len(), 5);
}

#[test]
fn parse_title_config_rejects_overlapping_layout() {
    let err = TitleConfig::parse(CONFIG_TOML_OVERLAP_LAYOUT).unwrap_err();
    assert!(err.contains("overlap"));
}

#[test]
fn parse_title_config_rejects_zero_size_layout() {
    let err = TitleConfig::parse(CONFIG_TOML_ZERO_LAYOUT).unwrap_err();
    assert!(err.contains("zero size"));
}

#[test]
fn memory_layout_rejects_overflow_range() {
    let layout = MemoryLayoutDescriptor {
        regions: vec![MemoryRegionDescriptor::new(
            "overflow",
            u64::MAX - 0x10,
            0x20,
            MemoryPermissionsDescriptor::new(true, false, false),
        )],
    };
    let err = layout.validate().unwrap_err();
    assert!(err.contains("overflows address space"));
}

#[test]
fn module_validation_rejects_unknown_arch() {
    let module: Module =
        serde_json::from_str(r#"{"arch":"mips","functions":[]}"#).expect("module parses");
    let err = module.validate_arch().unwrap_err();
    assert!(err.contains("unsupported arch"));
}

#[test]
fn pipeline_emits_project() {
    let temp = tempfile::tempdir().expect("tempdir");
    let module_path = temp.path().join("module.json");
    let config_path = temp.path().join("title.toml");
    let provenance_path = temp.path().join("provenance.toml");
    let nso_path = temp.path().join("main.nso");
    let out_dir = temp.path().join("out");
    let runtime_path = PathBuf::from("../crates/recomp-runtime");

    fs::write(&module_path, MODULE_JSON).expect("write module");
    fs::write(&config_path, CONFIG_TOML_DEFAULT_LAYOUT).expect("write config");
    fs::write(&nso_path, b"NSO0").expect("write nso");

    let module_hash = sha256_hex(MODULE_JSON.as_bytes());
    let nso_hash = sha256_hex(b"NSO0");
    let provenance = format!(
        "schema_version = \"1\"\n\n[title]\nname = \"Minimal Sample\"\ntitle_id = \"0100000000000000\"\nversion = \"1.0.0\"\nregion = \"US\"\n\n[collection]\ndevice = \"demo\"\ncollected_at = \"2026-01-30\"\n\n[collection.tool]\nname = \"manual\"\nversion = \"1.0\"\n\n[[inputs]]\npath = \"module.json\"\nformat = \"lifted_json\"\nsha256 = \"{module_hash}\"\nsize = {module_size}\nrole = \"lifted_module\"\n\n[[inputs]]\npath = \"main.nso\"\nformat = \"nso0\"\nsha256 = \"{nso_hash}\"\nsize = 4\nrole = \"main_executable\"\n",
        module_hash = module_hash,
        module_size = MODULE_JSON.len()
    );
    fs::write(&provenance_path, provenance).expect("write provenance");

    let report = run_pipeline(PipelineOptions {
        module_path,
        config_path,
        provenance_path,
        out_dir: out_dir.clone(),
        runtime_path,
    })
    .expect("pipeline runs");

    let cargo_toml = out_dir.join("Cargo.toml");
    let main_rs = out_dir.join("src/main.rs");
    let manifest = out_dir.join("manifest.json");
    assert!(cargo_toml.exists(), "Cargo.toml emitted");
    assert!(main_rs.exists(), "main.rs emitted");
    assert!(manifest.exists(), "manifest.json emitted");

    let main_src = fs::read_to_string(main_rs).expect("read main.rs");
    assert!(main_src.contains("svc_log"));
    // Known SVCs are resolved to their kernel id at generation time.
    assert!(
        main_src.contains("recomp_abi::svc_dispatch_id(0x1eu64, &[])"),
        "numeric SVC dispatch emitted: {main_src}"
    );
    assert!(!main_src.contains("svc_dispatch(\"svcGetSystemTick\""));
    let manifest_src = fs::read_to_string(manifest).expect("read manifest.json");
    let manifest_json: serde_json::Value =
        serde_json::from_str(&manifest_src).expect("parse manifest.json");
    assert!(manifest_json.get("module_sha256").is_some());
    assert!(manifest_json.get("memory_layout").is_some());
    let regions = manifest_json
        .get("memory_layout")
        .and_then(|value| value.get("regions"))
        .and_then(|value| value.as_array())
        .expect("memory_layout.regions array");
    assert_eq!(regions.len(), 5);
    assert_eq!(
        manifest_json
            .get("manifest_self_hash_basis")
            .and_then(|value| value.as_str()),
        Some("generated_files_self_placeholder")
    );
    let generated_files = manifest_json
        .get("generated_files")
        .and_then(|value| value.as_array())
        .expect("generated_files array");
    assert!(generated_files.iter().any(|entry| {
        entry.get("path").and_then(|value| value.as_str()) == Some("manifest.json")
    }));
    assert_eq!(report.files_written.len(), 3);
    assert_eq!(report.detected_inputs.len(), 2);
}

#[test]
fn pipeline_emits_custom_memory_layout() {
    let temp = tempfile::tempdir().expect("tempdir");
    let module_path = temp.path().join("module.json");
    let config_path = temp.path().join("title.toml");
    let provenance_path = temp.path().join("provenance.toml");
    let nso_path = temp.path().join("main.nso");
    let out_dir = temp.path().join("out");
    let runtime_path = PathBuf::from("../crates/recomp-runtime");

    fs::write(&module_path, MODULE_JSON).expect("write module");
    fs::write(&config_path, CONFIG_TOML).expect("write config");
    fs::write(&nso_path, b"NSO0").expect("write nso");

    let module_hash = sha256_hex(MODULE_JSON.as_bytes());
    let nso_hash = sha256_hex(b"NSO0");
    let provenance = format!(
        "schema_version = \"1\"\n\n[title]\nname = \"Minimal Sample\"\ntitle_id = \"0100000000000000\"\nversion = \"1.0.0\"\nregion = \"US\"\n\n[collection]\ndevice = \"demo\"\ncollected_at = \"2026-01-30\"\n\n[collection.tool]\nname = \"manual\"\nversion = \"1.0\"\n\n[[inputs]]\npath = \"module.json\"\nformat = \"lifted_json\"\nsha256 = \"{module_hash}\"\nsize = {module_size}\nrole = \"lifted_module\"\n\n[[inputs]]\npath = \"main.nso\"\nformat = \"nso0\"\nsha256 = \"{nso_hash}\"\nsize = 4\nrole = \"main_executable\"\n",
        module_hash = module_hash,
        module_size = MODULE_JSON.len()
    );
    fs::write(&provenance_path, provenance).expect("write provenance");

    run_pipeline(PipelineOptions {
        module_path,
        config_path,
        provenance_path,
        out_dir: out_dir.clone(),
        runtime_path,
    })
    .expect("pipeline runs");

    let manifest = out_dir.join("manifest.json");
    let manifest_src = fs::read_to_string(manifest).expect("read manifest.json");
    let manifest_json: serde_json::Value =
        serde_json::from_str(&manifest_src).expect("parse manifest.json");
    let regions = manifest_json
        .get("memory_layout")
        .and_then(|value| value.get("regions"))
        .and_then(|value| value.as_array())
        .expect("memory_layout.regions array");
    assert_eq!(regions.len(), 2);
}

/// Run the pipeline on a lifted module; returns `src/main.rs` and
/// `Cargo.toml` of the emitted project.
fn emit_project(module_json: &str) -> (String, String) {
    emit_project_with(module_json, CONFIG_TOML_DEFAULT_LAYOUT)
}

fn emit_project_with(module_json: &str, config_toml: &str) -> (String, String) {
    let (main_src, cargo_toml, _) = emit_project_files(module_json, config_toml);
    (main_src, cargo_toml)
}

/// main.rs, Cargo.toml and `.cargo/config.toml` (if written).
fn emit_project_files(module_json: &str, config_toml: &str) -> (String, String, Option<String>) {
    let temp = tempfile::tempdir().expect("tempdir");
    let module_path = temp.path().join("module.json");
    let config_path = temp.path().join("title.toml");
    let provenance_path = temp.path().join("provenance.toml");
    let nso_path = temp.path().join("main.nso");
    let out_dir = temp.path().join("out");

    fs::write(&module_path, module_json).expect("write module");
    fs::write(&config_path, config_toml).expect("write config");
    fs::write(&nso_path, b"NSO0").expect("write nso");
    let provenance = format!(
        "schema_version = \"1\"\n\n[title]\nname = \"Minimal Sample\"\ntitle_id = \"0100000000000000\"\nversion = \"1.0.0\"\nregion = \"US\"\n\n[collection]\ndevice = \"demo\"\ncollected_at = \"2026-01-30\"\n\n[collection.tool]\nname = \"manual\"\nversion = \"1.0\"\n\n[[inputs]]\npath = \"module.json\"\nformat = \"lifted_json\"\nsha256 = \"{module_hash}\"\nsize = {module_size}\nrole = \"lifted_module\"\n\n[[inputs]]\npath = \"main.nso\"\nformat = \"nso0\"\nsha256 = \"{nso_hash}\"\nsize = 4\nrole = \"main_executable\"\n",
        module_hash = sha256_hex(module_json.as_bytes()),
        module_size = module_json.len(),
        nso_hash = sha256_hex(b"NSO0"),
    );
    fs::write(&provenance_path, provenance).expect("write provenance");

    run_pipeline(PipelineOptions {
        module_path,
        config_path,
        provenance_path,
        out_dir: out_dir.clone(),
        runtime_path: PathBuf::from("../crates/recomp-runtime"),
    })
    .expect("pipeline runs");
    (
        fs::read_to_string(out_dir.join("src/main.rs")).expect("read main.rs"),
        fs::read_to_string(out_dir.join("Cargo.toml")).expect("read Cargo.toml"),
        fs::read_to_string(out_dir.join(".cargo/config.toml")).ok(),
    )
}

#[test]
fn target_cpu_compiles_the_program_for_that_cpu() {
    let module_json = r#"{ "arch": "aarch64", "functions": [ { "name": "entry", "ops": [ { "op": "ret" } ] } ] }"#;
    let (_, _, config) = emit_project_files(
        module_json,
        &format!("{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\ntarget_cpu = \"native\"\n"),
    );
    let config = config.expect(".cargo/config.toml written");
    assert!(
        config.contains("rustflags = [\"-C\", \"target-cpu=native\"]"),
        "{config}"
    );
    let (_, _, config) = emit_project_files(module_json, CONFIG_TOML_DEFAULT_LAYOUT);
    assert!(config.is_none(), "{config:?}");
}

#[test]
fn a_cross_target_gets_a_portable_cpu_and_target_specific_flags() {
    let module_json = r#"{ "arch": "aarch64", "functions": [ { "name": "entry", "ops": [ { "op": "ret" } ] } ] }"#;
    let (_, _, config) = emit_project_files(
        module_json,
        &format!(
            "{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\ntarget = \"aarch64-linux-android\"\ntarget_cpu = \"native\"\n"
        ),
    );
    let config = config.expect(".cargo/config.toml written");
    assert!(
        config.contains("[build]\ntarget = \"aarch64-linux-android\""),
        "{config}"
    );
    assert!(
        config.contains(
            "[target.\"aarch64-linux-android\"]\nrustflags = [\"-C\", \"target-cpu=generic\"]"
        ),
        "{config}"
    );
    assert!(!config.contains("target-cpu=native"), "{config}");
    // An explicit CPU is kept, and a target without target_cpu still selects
    // the cross target (but adds no CPU flags).
    let (_, _, config) = emit_project_files(
        module_json,
        &format!(
            "{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\ntarget = \"aarch64-linux-android\"\ntarget_cpu = \"cortex-a76\"\n"
        ),
    );
    let config = config.expect(".cargo/config.toml written");
    assert!(config.contains("target-cpu=cortex-a76"), "{config}");
    let (_, _, config) = emit_project_files(
        module_json,
        &format!("{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\ntarget = \"aarch64-linux-android\"\n"),
    );
    let config = config.expect(".cargo/config.toml written");
    assert!(
        config.contains("target = \"aarch64-linux-android\""),
        "{config}"
    );
    assert!(!config.contains("rustflags"), "{config}");
}

#[test]
fn an_invalid_cross_target_is_rejected() {
    let config =
        format!("{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\ntarget = \"aarch64 linux android\"\n");
    let error = TitleConfig::parse(&config).expect_err("invalid target must fail");
    assert!(error.contains("not a Cargo target triple"), "{error}");
}

#[test]
fn generated_code_uses_only_portable_abi_symbols() {
    let module_json = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "ops": [
        { "op": "simd", "word": 505487360, "vector_reads": ["v0", "v1"], "writes_flags": true },
        { "op": "simd", "word": 505482242, "vector_reads": ["v0", "v1"], "dst": "v2", "reads_flags": true },
        { "op": "simd", "word": 506986528, "vector_reads": ["v1"], "dst": "x0" },
        { "op": "ret" }
      ]
    }
  ]
}"#;
    let (main_src, _, config) = emit_project_files(
        module_json,
        &format!(
            "{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\ntarget = \"aarch64-unknown-linux-gnu\"\ntarget_cpu = \"native\"\n"
        ),
    );
    // Generated chunks call the stable recomp-abi surface, never a host
    // intrinsic or an x86-only helper; the fallback lives behind that API.
    for forbidden in [
        "core::arch",
        "std::arch",
        "is_x86_feature_detected",
        "_mm_",
        "__m128",
        "#[target_feature",
    ] {
        assert!(!main_src.contains(forbidden), "{forbidden} in {main_src}");
    }
    assert!(main_src.contains("recomp_abi::simd::fast::"), "{main_src}");
    let config = config.expect(".cargo/config.toml written");
    assert!(config.contains("target-cpu=generic"), "{config}");
}

#[test]
fn the_title_script_is_the_default_one() {
    let module_json = r#"{ "arch": "aarch64", "functions": [ { "name": "entry", "ops": [ { "op": "ret" } ] } ] }"#;
    let (main, _, _) = emit_project_files(
        module_json,
        &format!("{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\nscript = \"scripts/game.rhai\"\n"),
    );
    // Resolved against the config file's directory (a temporary one here)
    // and looked up beside the executable first.
    assert!(
        main.contains(
            "recomp_runtime::resolve_asset(\"RECOMP_SCRIPT\", Some(\"scripts/game.rhai\"), Some("
        ),
        "{main}"
    );
    assert!(main.contains("/scripts/game.rhai\")"), "{main}");
    assert!(
        main.contains("std::env::var_os(\"RECOMP_SCRIPT\").is_none()"),
        "{main}"
    );
    assert!(
        main.contains("std::env::set_var(\"RECOMP_SCRIPT\", path);"),
        "{main}"
    );
    let (main, _, _) = emit_project_files(module_json, CONFIG_TOML_DEFAULT_LAYOUT);
    assert!(!main.contains("RECOMP_SCRIPT"), "{main}");
}

#[test]
fn the_romfs_image_is_config_relative_and_overridable() {
    let module_json = r#"{ "arch": "aarch64", "functions": [ { "name": "entry", "ops": [ { "op": "ret" } ] } ] }"#;
    let (main, _, _) = emit_project_files(
        module_json,
        &format!(
            "{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime.ipc]\nromfs_base = 0x4800\nromfs_size = 0x101\nromfs_image = \"romfs.bin\"\n"
        ),
    );
    assert!(
        main.contains("recomp_runtime::resolve_asset(\"RECOMP_ROMFS\", Some(\"romfs.bin\"), Some("),
        "{main}"
    );
    assert!(main.contains("/romfs.bin\")"), "{main}");
    assert!(
        main.contains("init_ipc_services(0x4800, 0x101, romfs_image.as_deref(), None)"),
        "{main}"
    );
}

#[test]
fn an_ipc_less_project_still_accepts_an_environment_romfs() {
    let module_json = r#"{ "arch": "aarch64", "functions": [ { "name": "entry", "ops": [ { "op": "ret" } ] } ] }"#;
    let (main, _, _) = emit_project_files(module_json, CONFIG_TOML_DEFAULT_LAYOUT);
    assert!(
        main.contains("recomp_runtime::resolve_asset(\"RECOMP_ROMFS\", None, None)"),
        "{main}"
    );
    assert!(main.contains("if romfs_image.is_some()"), "{main}");
}

#[test]
fn asset_paths_are_overridden_by_the_environment() {
    let temp = tempfile::tempdir().expect("tempdir");
    let env_var = "RECOMP_TEST_ASSET_OVERRIDE";
    let override_path = temp.path().join("romfs.bin");

    std::env::set_var(env_var, &override_path);
    assert_eq!(
        recomp_services::assets::resolve_asset(
            env_var,
            Some("romfs.bin"),
            Some("/build/romfs.bin")
        ),
        Some(override_path.clone())
    );

    // An empty value disables the asset instead of falling back.
    std::env::set_var(env_var, "");
    assert_eq!(
        recomp_services::assets::resolve_asset(
            env_var,
            Some("romfs.bin"),
            Some("/build/romfs.bin")
        ),
        None
    );

    // Unset: the build-time path keeps an unpackaged build working.
    std::env::remove_var(env_var);
    assert_eq!(
        recomp_services::assets::resolve_asset(
            env_var,
            Some("romfs.bin"),
            Some("/build/romfs.bin")
        ),
        Some(PathBuf::from("/build/romfs.bin"))
    );
}

#[test]
fn pipeline_lowers_simd_ops() {
    let module_json = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "ops": [
        { "op": "simd", "word": 505487360, "vector_reads": ["v0", "v1"], "writes_flags": true },
        { "op": "simd", "word": 505482242, "vector_reads": ["v0", "v1"], "dst": "v2", "reads_flags": true },
        { "op": "simd", "word": 506986528, "vector_reads": ["v1"], "dst": "x0" },
        { "op": "simd", "word": 505610496, "vector_reads": [], "general_read": "x8", "dst": "v0" },
        { "op": "simd", "word": 241264672, "vector_reads": ["v1"], "dst": "v0" },
        { "op": "ret" }
      ]
    }
  ]
}"#;
    let (main_src, cargo_toml) = emit_project(module_json);
    // The common instructions are decoded now into direct calls: fcmp s0, s1
    // sets the flags...
    assert!(main_src.contains(
        "{ let __f = recomp_abi::simd::fast::scalar_compare(false, v0, v1); flag_n = __f & 8 != 0;"
    ));
    // ...fcsel s2, s0, s1, eq passes them in...
    assert!(main_src
        .contains("v2 = recomp_abi::simd::fast::scalar_select(false, 0u32, ((flag_n as u8) << 3)"));
    // ...fcvtzs w0, s1 writes a general register; ucvtf s0, w8 reads one.
    assert!(main_src.contains(
        "x0 = recomp_abi::simd::fast::float_to_int(false, 24u32, false, v1) as u64 as i64;"
    ));
    assert!(main_src
        .contains("v0 = recomp_abi::simd::fast::int_to_float(false, true, false, x8 as u64);"));
    // Others still go through the evaluator: fcvtn v0.4h, v1.4s.
    assert!(main_src.contains("recomp_abi::simd::exec(0xe616820, &[(1, v1)], 0, 0)?;"));
    assert!(main_src.contains("v0 = __s.value;"));
    assert!(cargo_toml.contains("[profile.dev.package.recomp-simd]\nopt-level = 3"));
    assert!(cargo_toml.contains("opt-level = 2\noverflow-checks = false"));
    assert!(cargo_toml.contains("panic = \"abort\""));
}

#[test]
fn pipeline_lowers_register_tests_without_touching_flags() {
    // subs-style flags, then a cbnz-like test, then a branch on the flags.
    let module_json = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "blocks": [
        { "label": "bb_0", "start": 0, "ops": [ { "op": "cmp_i64", "lhs": "x1", "rhs": "x2" } ],
          "terminator": { "op": "br_test", "reg": "x10", "mask": 4294967295, "zero": false, "then": "bb_0", "else": "bb_1" } },
        { "label": "bb_1", "start": 4,
          "terminator": { "op": "br_test", "reg": "x3", "mask": 1099511627776, "zero": true, "then": "bb_2", "else": "bb_3" } },
        { "label": "bb_2", "start": 8,
          "terminator": { "op": "br_cond", "cond": "ne", "then": "bb_3", "else": "bb_3" } },
        { "label": "bb_3", "start": 12, "terminator": { "op": "ret" } }
      ]
    }
  ]
}"#;
    let (main_src, _) = emit_project(module_json);
    assert!(main_src.contains("if ((x10 as u64) & 0xffffffff) != 0 {"));
    assert!(main_src.contains("if ((x3 as u64) & 0x10000000000) == 0 {"));
    // Only the compare writes the flags.
    assert_eq!(main_src.matches("flag_z = __").count(), 1);
}

#[test]
fn a_direct_call_spills_only_the_callee_registers() {
    let module_json = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "fn_0000000000001000",
      "blocks": [
        { "label": "bb_1000", "start": 4096,
          "ops": [ { "op": "const_i64", "dst": "x9", "imm": 1 } ],
          "terminator": { "op": "call", "target": "fn_0000000000002000", "next": "bb_1004", "return_address": 4100 } },
        { "label": "bb_1004", "start": 4100,
          "ops": [ { "op": "add_i64", "dst": "x0", "lhs": "x0", "rhs": "x9" } ],
          "terminator": { "op": "ret" } }
      ]
    },
    {
      "name": "fn_0000000000002000",
      "blocks": [
        { "label": "bb_2000", "start": 8192,
          "ops": [ { "op": "const_i64", "dst": "x0", "imm": 7 } ],
          "terminator": { "op": "ret" } }
      ]
    }
  ]
}"#;
    let (main, _) = emit_project(module_json);
    let fast_start = main
        .find("if recomp_abi::direct_calls_enabled() {")
        .expect("selective call path");
    let fast_end = fast_start
        + main[fast_start..]
            .find("} else {")
            .expect("full-spill fallback");
    let fast = &main[fast_start..fast_end];
    // The callee only touches x0, so the call saves just that register
    // (its writes can spill back on paths that leave it alone) and reloads
    // its result; x9 and the return address stay in their locals.
    assert!(
        fast.contains("recomp_abi::call_direct(|| unsafe { fn_0000000000002000() })?;"),
        "{main}"
    );
    assert!(
        fast.contains("recomp_abi::cpu_store(&[(0, x0)]);"),
        "{main}"
    );
    assert!(!fast.contains("x9"), "{main}");
    assert!(!fast.contains("x30"), "{main}");
    assert!(fast.contains("x0 = __r[0];"), "{main}");
    // Hooks and serialized calls still see the whole register file.
    let slow = &main[fast_end..];
    assert!(
        slow.contains("recomp_abi::call(0x2000, || unsafe { fn_0000000000002000() })?;"),
        "{main}"
    );
    assert!(slow.contains("(9, x9)"), "{main}");
}

#[test]
fn pipeline_lets_other_threads_in_on_loop_backedges() {
    // bb_10 loops on itself; bb_20 is reached forwards.
    let module_json = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "blocks": [
        { "label": "bb_10", "start": 16,
          "terminator": { "op": "br_test", "reg": "x0", "mask": 1, "zero": true, "then": "bb_10", "else": "bb_20" } },
        { "label": "bb_20", "start": 32, "terminator": { "op": "ret" } }
      ]
    }
  ]
}"#;
    let (main_src, _) = emit_project(module_json);
    assert!(main_src.contains(
        "if ((x0 as u64) & 0x1) == 0 { recomp_abi::backedge(); block = 0; continue; } else { block = 1; continue; }"
    ));
    assert_eq!(main_src.matches("recomp_abi::backedge()").count(), 1);
}

/// Two blocks, the first looping back to itself: two `enter_block` calls and
/// one `backedge` by default.
const LOOP_MODULE_JSON: &str = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "blocks": [
        { "label": "bb_1000", "start": 4096,
          "terminator": { "op": "br_test", "reg": "x0", "mask": 1, "zero": true, "then": "bb_1000", "else": "bb_1004" } },
        { "label": "bb_1004", "start": 4100, "terminator": { "op": "ret" } }
      ]
    }
  ]
}"#;

#[test]
fn default_block_tracking_is_unchanged() {
    let (omitted, _) = emit_project(LOOP_MODULE_JSON);
    let (explicit, _) = emit_project_with(
        LOOP_MODULE_JSON,
        &format!("{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\nblock_tracking = true\n"),
    );
    assert_eq!(omitted, explicit);
    assert_eq!(omitted.matches("recomp_abi::enter_block(").count(), 2);
    assert_eq!(omitted.matches("recomp_abi::backedge()").count(), 1);
}

#[test]
fn block_tracking_off_keeps_one_entry_and_parallel_drops_backedges() {
    let (main_src, _) = emit_project_with(
        LOOP_MODULE_JSON,
        &format!(
            "{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\nblock_tracking = false\nparallel = true\n"
        ),
    );
    // One call, at the function's first block, before the dispatch loop.
    assert_eq!(
        main_src.matches("recomp_abi::enter_block(").count(),
        1,
        "{main_src}"
    );
    assert!(
        main_src.contains("recomp_abi::enter_block(0x1000);"),
        "{main_src}"
    );
    assert!(
        !main_src.contains("recomp_abi::enter_block(0x1004);"),
        "{main_src}"
    );
    let entry = main_src.find("recomp_abi::enter_block(0x1000);").unwrap();
    let loop_start = main_src.find("loop {").unwrap();
    assert!(entry < loop_start, "{main_src}");
    // Parallel threads are placed by the host, so the back edge only fed the
    // hang monitor.
    assert!(!main_src.contains("recomp_abi::backedge()"), "{main_src}");
}

#[test]
fn block_tracking_off_keeps_backedges_for_serialized_threads() {
    let (main_src, _) = emit_project_with(
        LOOP_MODULE_JSON,
        &format!("{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\nblock_tracking = false\n"),
    );
    assert_eq!(main_src.matches("recomp_abi::enter_block(").count(), 1);
    assert_eq!(main_src.matches("recomp_abi::backedge()").count(), 1);
    assert!(
        main_src.contains(
            "if ((x0 as u64) & 0x1) == 0 { recomp_abi::backedge(); block = 0; continue; } else { block = 1; continue; }"
        ),
        "{main_src}"
    );
}

#[test]
fn block_tracking_off_keeps_one_entry_per_split_part() {
    let mut blocks = String::new();
    for index in 0..65 {
        if index > 0 {
            blocks.push(',');
        }
        let address = 0x1000 + index * 4;
        let terminator = if index + 1 < 65 {
            format!("{{ \"op\": \"br\", \"target\": \"bb_{:x}\" }}", address + 4)
        } else {
            "{ \"op\": \"ret\" }".to_string()
        };
        blocks.push_str(&format!(
            "{{ \"label\": \"bb_{address:x}\", \"start\": {address}, \"terminator\": {terminator} }}"
        ));
    }
    let module_json = format!(
        "{{ \"arch\": \"aarch64\", \"functions\": [ {{ \"name\": \"entry\", \"blocks\": [{blocks}] }} ] }}"
    );
    let (main_src, _) = emit_project_with(
        &module_json,
        &format!(
            "{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\nblock_tracking = false\nparallel = true\n"
        ),
    );
    // More blocks than one part holds (64): each part starts with one call.
    assert_eq!(
        main_src.matches("recomp_abi::enter_block(").count(),
        2,
        "{main_src}"
    );
    assert!(
        main_src.contains("recomp_abi::enter_block(0x1000);"),
        "{main_src}"
    );
    assert!(
        main_src.contains("recomp_abi::enter_block(0x1100);"),
        "{main_src}"
    );
    assert!(!main_src.contains("recomp_abi::backedge()"), "{main_src}");
}

#[test]
fn pipeline_links_the_vulkan_backend_when_configured() {
    let module_json = r#"{
  "arch": "aarch64",
  "functions": [ { "name": "entry", "ops": [ { "op": "ret" } ] } ]
}"#;
    let config = format!("{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\ngraphics = \"vulkan\"\n");
    let (main_src, cargo_toml) = emit_project_with(module_json, &config);
    // The backend crate sits next to the runtime in the toolchain checkout.
    let path_of = |name: &str| {
        let line = cargo_toml
            .lines()
            .find(|line| line.starts_with(&format!("{name} = ")))
            .unwrap_or_else(|| panic!("{name} dependency in {cargo_toml}"));
        line.split('"').nth(1).expect("path").to_string()
    };
    let runtime = path_of("recomp-runtime");
    assert_eq!(
        path_of("recomp-gfx-vulkan"),
        runtime.replace("recomp-runtime", "recomp-gfx-vulkan")
    );
    // The window opens unless RECOMP_VULKAN=0.
    assert!(main_src
        .contains("if std::env::var(\"RECOMP_VULKAN\").map_or(true, |value| value != \"0\") {"));
    assert!(main_src.contains("recomp_gfx_vulkan::install()"));
    // The window opens before the guest starts.
    assert!(main_src.find("recomp_gfx_vulkan::install()") < main_src.find("boot_main_thread"));

    // Without the setting, neither appears.
    let (main_src, cargo_toml) = emit_project(module_json);
    assert!(!cargo_toml.contains("recomp-gfx-vulkan"));
    assert!(!main_src.contains("recomp_gfx_vulkan"));
    assert!(!main_src.contains("RECOMP_PARALLEL"));
}

#[test]
fn pipeline_makes_parallel_threads_the_default_when_configured() {
    let module_json = r#"{
  "arch": "aarch64",
  "functions": [ { "name": "entry", "ops": [ { "op": "ret" } ] } ]
}"#;
    let config = format!("{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\nparallel = true\n");
    let (main_src, _) = emit_project_with(module_json, &config);
    let main_start = main_src.find("fn main()").expect("main");
    let default = main_src
        .find("Err(_) => std::env::set_var(\"RECOMP_PARALLEL\", \"1\"),")
        .expect("parallel default");
    // First thing in main, before the runtime starts anything.
    assert!(default > main_start);
    assert!(default < main_src.find("init_kernel_services").expect("kernel init"));
}

#[test]
fn config_rejects_an_unknown_graphics_backend() {
    let config = format!("{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\ngraphics = \"opengl\"\n");
    let error = TitleConfig::parse(&config).expect_err("unknown backend");
    assert!(
        error.contains("unknown graphics backend: opengl"),
        "{error}"
    );
}

#[test]
fn pipeline_lowers_add_with_carry() {
    let module_json = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "ops": [
        { "op": "add_carry", "dst": "x0", "lhs": "x1", "rhs": "x2", "subtract": false, "set_flags": true, "is_32": false },
        { "op": "add_carry", "dst": "x3", "lhs": "x4", "rhs": "x5", "subtract": true, "set_flags": false, "is_32": true },
        { "op": "ret" }
      ]
    }
  ]
}"#;
    let (main_src, _) = emit_project(module_json);
    assert!(main_src.contains("let __c = flag_c as u64;"));
    assert!(main_src.contains("flag_c = (__u >> 64) != 0;"));
    assert!(main_src.contains("(!(x5 as u64)) & 0xffff_ffffu64"));
    assert!(main_src.contains("x3 = __res as i64;"));
}

#[test]
fn pipeline_lowers_load_store_ops() {
    let temp = tempfile::tempdir().expect("tempdir");
    let module_path = temp.path().join("module.json");
    let config_path = temp.path().join("title.toml");
    let provenance_path = temp.path().join("provenance.toml");
    let nso_path = temp.path().join("main.nso");
    let out_dir = temp.path().join("out");
    let runtime_path = PathBuf::from("../crates/recomp-runtime");

    let module_json = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "ops": [
        { "op": "const_i64", "dst": "x0", "imm": 4096 },
        { "op": "const_i64", "dst": "x1", "imm": 123 },
        { "op": "store_i32", "src": "x1", "addr": "x0", "offset": 0 },
        { "op": "load_i32", "dst": "x2", "addr": "x0", "offset": 0 },
        { "op": "ret" }
      ]
    }
  ]
}"#;

    fs::write(&module_path, module_json).expect("write module");
    fs::write(&config_path, CONFIG_TOML_DEFAULT_LAYOUT).expect("write config");
    fs::write(&nso_path, b"NSO0").expect("write nso");

    let module_hash = sha256_hex(module_json.as_bytes());
    let nso_hash = sha256_hex(b"NSO0");
    let provenance = format!(
        "schema_version = \"1\"\n\n[title]\nname = \"Minimal Sample\"\ntitle_id = \"0100000000000000\"\nversion = \"1.0.0\"\nregion = \"US\"\n\n[collection]\ndevice = \"demo\"\ncollected_at = \"2026-01-30\"\n\n[collection.tool]\nname = \"manual\"\nversion = \"1.0\"\n\n[[inputs]]\npath = \"module.json\"\nformat = \"lifted_json\"\nsha256 = \"{module_hash}\"\nsize = {module_size}\nrole = \"lifted_module\"\n\n[[inputs]]\npath = \"main.nso\"\nformat = \"nso0\"\nsha256 = \"{nso_hash}\"\nsize = 4\nrole = \"main_executable\"\n",
        module_hash = module_hash,
        module_size = module_json.len()
    );
    fs::write(&provenance_path, provenance).expect("write provenance");

    run_pipeline(PipelineOptions {
        module_path,
        config_path,
        provenance_path,
        out_dir: out_dir.clone(),
        runtime_path,
    })
    .expect("pipeline runs");

    let main_rs = out_dir.join("src/main.rs");
    let main_src = fs::read_to_string(main_rs).expect("read main.rs");
    assert!(main_src.contains("let __mem = recomp_abi::memory();"));
    assert!(main_src.contains("__mem.store_u32("));
    assert!(main_src.contains("__mem.load_u32("));
}

#[test]
fn pipeline_rejects_overlapping_segments() {
    let temp = tempfile::tempdir().expect("tempdir");
    let module_path = temp.path().join("module.json");
    let config_path = temp.path().join("title.toml");
    let provenance_path = temp.path().join("provenance.toml");
    let nso_path = temp.path().join("main.nso");
    let out_dir = temp.path().join("out");
    let runtime_path = PathBuf::from("../crates/recomp-runtime");

    let module_json = r#"{
  "arch": "aarch64",
  "segments": [
    {
      "name": "seg_a",
      "base": 4096,
      "size": 4096,
      "permissions": { "read": true, "write": true, "execute": false },
      "zero_fill": true
    },
    {
      "name": "seg_b",
      "base": 6144,
      "size": 4096,
      "permissions": { "read": true, "write": true, "execute": false },
      "zero_fill": true
    }
  ],
  "functions": [
    { "name": "entry", "ops": [ { "op": "ret" } ] }
  ]
}"#;

    fs::write(&module_path, module_json).expect("write module");
    fs::write(&config_path, CONFIG_TOML_DEFAULT_LAYOUT).expect("write config");
    fs::write(&nso_path, b"NSO0").expect("write nso");

    let module_hash = sha256_hex(module_json.as_bytes());
    let nso_hash = sha256_hex(b"NSO0");
    let provenance = format!(
        "schema_version = \"1\"\n\n[title]\nname = \"Minimal Sample\"\ntitle_id = \"0100000000000000\"\nversion = \"1.0.0\"\nregion = \"US\"\n\n[collection]\ndevice = \"demo\"\ncollected_at = \"2026-01-30\"\n\n[collection.tool]\nname = \"manual\"\nversion = \"1.0\"\n\n[[inputs]]\npath = \"module.json\"\nformat = \"lifted_json\"\nsha256 = \"{module_hash}\"\nsize = {module_size}\nrole = \"lifted_module\"\n\n[[inputs]]\npath = \"main.nso\"\nformat = \"nso0\"\nsha256 = \"{nso_hash}\"\nsize = 4\nrole = \"main_executable\"\n",
        module_hash = module_hash,
        module_size = module_json.len()
    );
    fs::write(&provenance_path, provenance).expect("write provenance");

    let err = run_pipeline(PipelineOptions {
        module_path,
        config_path,
        provenance_path,
        out_dir,
        runtime_path,
    })
    .expect_err("pipeline rejects overlapping segments");
    assert!(err.to_string().contains("overlap"));
}

#[test]
fn pipeline_rejects_overflowing_segments() {
    let temp = tempfile::tempdir().expect("tempdir");
    let module_path = temp.path().join("module.json");
    let config_path = temp.path().join("title.toml");
    let provenance_path = temp.path().join("provenance.toml");
    let nso_path = temp.path().join("main.nso");
    let out_dir = temp.path().join("out");
    let runtime_path = PathBuf::from("../crates/recomp-runtime");

    let module_json = r#"{
  "arch": "aarch64",
  "segments": [
    {
      "name": "seg_overflow",
      "base": 18446744073709547520,
      "size": 8192,
      "permissions": { "read": true, "write": true, "execute": false },
      "zero_fill": true
    }
  ],
  "functions": [
    { "name": "entry", "ops": [ { "op": "ret" } ] }
  ]
}"#;

    fs::write(&module_path, module_json).expect("write module");
    fs::write(&config_path, CONFIG_TOML_DEFAULT_LAYOUT).expect("write config");
    fs::write(&nso_path, b"NSO0").expect("write nso");

    let module_hash = sha256_hex(module_json.as_bytes());
    let nso_hash = sha256_hex(b"NSO0");
    let provenance = format!(
        "schema_version = \"1\"\n\n[title]\nname = \"Minimal Sample\"\ntitle_id = \"0100000000000000\"\nversion = \"1.0.0\"\nregion = \"US\"\n\n[collection]\ndevice = \"demo\"\ncollected_at = \"2026-01-30\"\n\n[collection.tool]\nname = \"manual\"\nversion = \"1.0\"\n\n[[inputs]]\npath = \"module.json\"\nformat = \"lifted_json\"\nsha256 = \"{module_hash}\"\nsize = {module_size}\nrole = \"lifted_module\"\n\n[[inputs]]\npath = \"main.nso\"\nformat = \"nso0\"\nsha256 = \"{nso_hash}\"\nsize = 4\nrole = \"main_executable\"\n",
        module_hash = module_hash,
        module_size = module_json.len()
    );
    fs::write(&provenance_path, provenance).expect("write provenance");

    let err = run_pipeline(PipelineOptions {
        module_path,
        config_path,
        provenance_path,
        out_dir,
        runtime_path,
    })
    .expect_err("pipeline rejects overflowing segments");
    assert!(err.to_string().contains("overflows address space"));
}

#[test]
fn pipeline_emits_memory_image() {
    let temp = tempfile::tempdir().expect("tempdir");
    let module_path = temp.path().join("module.json");
    let config_path = temp.path().join("title.toml");
    let provenance_path = temp.path().join("provenance.toml");
    let nso_path = temp.path().join("main.nso");
    let data_path = temp.path().join("data.bin");
    let out_dir = temp.path().join("out");
    let runtime_path = PathBuf::from("../crates/recomp-runtime");

    let module_json = r#"{
  "arch": "aarch64",
  "segments": [
    {
      "name": "data",
      "base": 4096,
      "size": 8,
      "permissions": { "read": true, "write": true, "execute": false },
      "init_path": "data.bin",
      "init_size": 4,
      "zero_fill": true
    }
  ],
  "functions": [
    { "name": "entry", "ops": [ { "op": "ret" } ] }
  ]
}"#;

    fs::write(&module_path, module_json).expect("write module");
    fs::write(&config_path, CONFIG_TOML_DEFAULT_LAYOUT).expect("write config");
    fs::write(&nso_path, b"NSO0").expect("write nso");
    fs::write(&data_path, [1u8, 2, 3, 4]).expect("write data");

    let module_hash = sha256_hex(module_json.as_bytes());
    let nso_hash = sha256_hex(b"NSO0");
    let provenance = format!(
        "schema_version = \"1\"\n\n[title]\nname = \"Minimal Sample\"\ntitle_id = \"0100000000000000\"\nversion = \"1.0.0\"\nregion = \"US\"\n\n[collection]\ndevice = \"demo\"\ncollected_at = \"2026-01-30\"\n\n[collection.tool]\nname = \"manual\"\nversion = \"1.0\"\n\n[[inputs]]\npath = \"module.json\"\nformat = \"lifted_json\"\nsha256 = \"{module_hash}\"\nsize = {module_size}\nrole = \"lifted_module\"\n\n[[inputs]]\npath = \"main.nso\"\nformat = \"nso0\"\nsha256 = \"{nso_hash}\"\nsize = 4\nrole = \"main_executable\"\n",
        module_hash = module_hash,
        module_size = module_json.len()
    );
    fs::write(&provenance_path, provenance).expect("write provenance");

    run_pipeline(PipelineOptions {
        module_path,
        config_path,
        provenance_path,
        out_dir: out_dir.clone(),
        runtime_path,
    })
    .expect("pipeline runs");

    let segment_path = out_dir.join("segments/data-0.bin");
    assert!(segment_path.exists(), "segment file emitted");
    let segment_bytes = fs::read(&segment_path).expect("read segment file");
    assert_eq!(segment_bytes, [1u8, 2, 3, 4]);

    let manifest = out_dir.join("manifest.json");
    let manifest_src = fs::read_to_string(manifest).expect("read manifest.json");
    let manifest_json: serde_json::Value =
        serde_json::from_str(&manifest_src).expect("parse manifest.json");
    let memory_image = manifest_json
        .get("memory_image")
        .and_then(|value| value.as_object())
        .expect("memory_image object");
    let init_segments = memory_image
        .get("init_segments")
        .and_then(|value| value.as_array())
        .expect("init_segments array");
    let zero_segments = memory_image
        .get("zero_segments")
        .and_then(|value| value.as_array())
        .expect("zero_segments array");
    assert_eq!(init_segments.len(), 1);
    assert_eq!(zero_segments.len(), 1);

    let main_rs = out_dir.join("src/main.rs");
    let main_src = fs::read_to_string(main_rs).expect("read main.rs");
    assert!(main_src.contains("apply_memory_image"));
}

#[test]
fn pipeline_rejects_homebrew_module_json() {
    let temp = tempfile::tempdir().expect("tempdir");
    let module_path = temp.path().join("module.json");
    let config_path = temp.path().join("title.toml");
    let provenance_path = temp.path().join("provenance.toml");
    let out_dir = temp.path().join("out");
    let runtime_path = PathBuf::from("../crates/recomp-runtime");

    let homebrew_module = r#"{
  "schema_version": "1",
  "module_type": "homebrew",
  "modules": [
    {
      "name": "sample",
      "format": "nro",
      "input_path": "module.nro",
      "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "input_size": 4,
      "build_id": "deadbeef",
      "segments": [],
      "bss": { "size": 0, "memory_offset": 0 }
    }
  ]
}"#;

    fs::write(&module_path, homebrew_module).expect("write module");
    let err = run_pipeline(PipelineOptions {
        module_path,
        config_path,
        provenance_path,
        out_dir,
        runtime_path,
    })
    .expect_err("pipeline rejects homebrew module.json");

    let message = err.to_string();
    assert!(message.contains("homebrew module.json detected"));
}

#[test]
fn pipeline_emits_exclusive_accesses_and_barriers() {
    let temp = tempfile::tempdir().expect("tempdir");
    let module_path = temp.path().join("module.json");
    let config_path = temp.path().join("title.toml");
    let provenance_path = temp.path().join("provenance.toml");
    let nso_path = temp.path().join("main.nso");
    let out_dir = temp.path().join("out");
    let runtime_path = PathBuf::from("../crates/recomp-runtime");

    // A spinlock shape: reserve, order, claim, drop the reservation.
    let module = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "ops": [
        { "op": "const_i64", "dst": "x1", "imm": 32 },
        { "op": "const_i64", "dst": "x2", "imm": 1 },
        { "op": "exclusive_load", "dst": "x0", "addr": "x1", "size": 8, "pair": false },
        { "op": "memory_fence" },
        { "op": "acquire_fence" },
        { "op": "release_fence" },
        { "op": "exclusive_store", "src": "x2", "status": "x3", "addr": "x1", "size": 8, "pair": false },
        { "op": "clear_exclusive" },
        { "op": "ret" }
      ]
    }
  ]
}"#;

    fs::write(&module_path, module).expect("write module");
    fs::write(&config_path, CONFIG_TOML_DEFAULT_LAYOUT).expect("write config");
    fs::write(&nso_path, b"NSO0").expect("write nso");

    let module_hash = sha256_hex(module.as_bytes());
    let provenance = format!(
        "schema_version = \"1\"\n\n[title]\nname = \"Minimal Sample\"\ntitle_id = \"0100000000000000\"\nversion = \"1.0.0\"\nregion = \"US\"\n\n[collection]\ndevice = \"demo\"\ncollected_at = \"2026-01-30\"\n\n[collection.tool]\nname = \"manual\"\nversion = \"1.0\"\n\n[[inputs]]\npath = \"module.json\"\nformat = \"lifted_json\"\nsha256 = \"{module_hash}\"\nsize = {module_size}\nrole = \"lifted_module\"\n\n[[inputs]]\npath = \"main.nso\"\nformat = \"nso0\"\nsha256 = \"{nso_hash}\"\nsize = 4\nrole = \"main_executable\"\n",
        module_hash = module_hash,
        module_size = module.len(),
        nso_hash = sha256_hex(b"NSO0")
    );
    fs::write(&provenance_path, provenance).expect("write provenance");

    run_pipeline(PipelineOptions {
        module_path,
        config_path,
        provenance_path,
        out_dir: out_dir.clone(),
        runtime_path,
    })
    .expect("pipeline runs");

    let src = fs::read_to_string(out_dir.join("src/main.rs")).expect("read main.rs");
    assert!(
        src.contains("recomp_abi::mem_load_exclusive(x1 as u64, 8, false)?"),
        "exclusive load reaches the ABI"
    );
    assert!(
        src.contains("recomp_abi::mem_store_exclusive(x1 as u64, 8, false, x2 as u64, 0)?"),
        "exclusive store reaches the ABI"
    );
    assert!(src.contains("recomp_abi::clear_exclusive()"));
    assert!(src.contains("recomp_abi::memory_fence()"));
    assert!(src.contains("recomp_abi::acquire_fence()"));
    assert!(src.contains("recomp_abi::release_fence()"));
}

/// A small caller/callee shape with flags, a vector register, a nested
/// direct call, and the two indirect fallbacks (call and tail branch).
const CONTEXT_MODULE_JSON: &str = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "blocks": [
        { "label": "bb_1000", "start": 4096, "ops": [
            { "op": "const_i64", "dst": "x0", "imm": 5 },
            { "op": "const_i64", "dst": "x1", "imm": 7 },
            { "op": "const_i64", "dst": "x9", "imm": 9 },
            { "op": "v_const", "dst": "v0", "lo": 3, "hi": 4 },
            { "op": "const_i64", "dst": "x30", "imm": 4100 }
          ],
          "terminator": { "op": "call", "target": "fn_0000000000002000", "next": "bb_1004", "return_address": 4100 } },
        { "label": "bb_1004", "start": 4100, "ops": [
            { "op": "cmp_i64", "lhs": "x0", "rhs": "x1" }
          ],
          "terminator": { "op": "ret" } }
      ]
    },
    {
      "name": "fn_0000000000002000",
      "blocks": [
        { "label": "bb_2000", "start": 8192, "ops": [
            { "op": "add_i64", "dst": "x0", "lhs": "x0", "rhs": "x1" },
            { "op": "v_const", "dst": "v1", "lo": 5, "hi": 6 }
          ],
          "terminator": { "op": "ret" } }
      ]
    },
    {
      "name": "fn_0000000000003000",
      "blocks": [
        { "label": "bb_3000", "start": 12288, "ops": [
            { "op": "const_i64", "dst": "x30", "imm": 12292 }
          ],
          "terminator": { "op": "call_indirect", "reg": "x2", "next": "bb_3004", "return_address": 12292 } },
        { "label": "bb_3004", "start": 12292, "terminator": { "op": "ret" } }
      ]
    },
    {
      "name": "fn_0000000000004000",
      "blocks": [
        { "label": "bb_4000", "start": 16384,
          "terminator": { "op": "br_indirect", "reg": "x3" } }
      ]
    }
  ]
}"#;

const CONFIG_TOML_GUEST_CONTEXT: &str = r#"
title = "Minimal Sample"
entry = "entry"
abi_version = "0.1.0"

[runtime]
guest_context = true

[stubs]
svc_log = "log"
"#;

#[test]
fn guest_context_abi_is_off_by_default() {
    let (main, _) = emit_project(CONTEXT_MODULE_JSON);
    assert!(
        main.contains("fn entry() -> recomp_abi::GuestResult {"),
        "{main}"
    );
    assert!(main.contains("let __regs = recomp_abi::cpu_regs();"));
    let fast_start = main
        .find("if recomp_abi::direct_calls_enabled() {")
        .expect("selective call path");
    let fast = &main[fast_start..];
    assert!(
        fast.contains("recomp_abi::call_direct(|| unsafe { fn_0000000000002000() })?;"),
        "{main}"
    );
    assert!(!main.contains("ctx: *mut recomp_abi::GuestContext"));
}

#[test]
fn guest_context_abi_accesses_the_shared_register_file() {
    let (main, _) = emit_project_with(CONTEXT_MODULE_JSON, CONFIG_TOML_GUEST_CONTEXT);
    // The signature, and the register loads go through `ctx`; there is no
    // whole-register-file snapshot.
    assert!(main.contains(
        "unsafe fn entry(ctx: *mut recomp_abi::GuestContext) -> recomp_abi::GuestResult {"
    ));
    assert!(main.contains("let mut x0: i64 = unsafe { (*ctx).x[0] };"));
    assert!(main.contains("let mut v0: u128 = unsafe { (*ctx).v[0] };"));
    assert!(main.contains("let nzcv = unsafe { (*ctx).nzcv };"));
    assert!(!main.contains("recomp_abi::cpu_regs()"), "{main}");
    assert!(!main.contains("recomp_abi::cpu_vregs()"), "{main}");

    // The clobber fast path still applies: the callee only needs and
    // clobbers x0/x1, so the selective spill and reload are those fields
    // only, and the call hands it the context directly.
    let fast_start = main
        .find("if recomp_abi::direct_calls_enabled() {")
        .expect("selective call path");
    let fast_end = fast_start
        + main[fast_start..]
            .find("} else {")
            .expect("full-spill fallback");
    let fast = &main[fast_start..fast_end];
    assert!(fast.contains("unsafe { (*ctx).x[0] = x0; }"), "{main}");
    assert!(fast.contains("unsafe { (*ctx).x[1] = x1; }"), "{main}");
    assert!(
        fast.contains("recomp_abi::call_direct(|| unsafe { fn_0000000000002000(ctx) })?;"),
        "{main}"
    );
    assert!(fast.contains("x0 = unsafe { (*ctx).x[0] };"), "{main}");
    assert!(!fast.contains("x9"), "{main}");
    assert!(!fast.contains("x30"), "{main}");
    assert!(!fast.contains("v0"), "{main}");
    assert!(!fast.contains("(*ctx).nzcv"), "{main}");

    // Hooks and serialized calls still get the full live set and flags, and
    // the fallback goes through `dispatch` (rewritten to `call`).
    let slow = &main[fast_end..];
    assert!(slow.contains("unsafe { (*ctx).x[9] = x9; }"), "{main}");
    assert!(slow.contains("unsafe { (*ctx).v[0] = v0; }"), "{main}");
    assert!(slow.contains("unsafe { (*ctx).x[30] = x30; }"), "{main}");
    assert!(
        slow.contains("unsafe { (*ctx).nzcv = ((flag_n as u8) << 3)"),
        "{main}"
    );
    assert!(
        slow.contains("recomp_abi::call(0x2000, || unsafe { fn_0000000000002000(ctx) })?;"),
        "{main}"
    );
    assert!(!main.contains("recomp_abi::cpu_store"), "{main}");
    assert!(!main.contains("recomp_abi::cpu_vstore"), "{main}");

    // The return-address check is unchanged.
    assert!(main.contains("if x30 as u64 != 0x1004 {"), "{main}");
    // The runtime's `fn()` call table gets wrappers that supply the thread's
    // context, so hooks and indirect calls keep the same entry point.
    assert!(main.contains(
        "fn __recomp_entry_0000000000002000() -> recomp_abi::GuestResult {\n    unsafe { fn_0000000000002000(recomp_abi::guest_context()) }\n}"
    ));
    assert!(main.contains("unsafe { entry(recomp_abi::guest_context()) }?;"));
    // Indirect calls and tail branches still go through `dispatch`, which
    // reads the same storage.
    assert!(main.contains("recomp_abi::dispatch(x2 as u64)?;"));
    assert!(main.contains("return recomp_abi::dispatch(x3 as u64);"));
}

#[test]
fn guest_context_block_tracking_off_keeps_the_context_abi() {
    let config = CONFIG_TOML_GUEST_CONTEXT.replace(
        "guest_context = true",
        "guest_context = true\nparallel = true\nblock_tracking = false",
    );
    let (main, _) = emit_project_with(CONTEXT_MODULE_JSON, &config);

    // The context ABI and the clobber fast path are unchanged.
    assert!(
        main.contains(
            "unsafe fn entry(ctx: *mut recomp_abi::GuestContext) -> recomp_abi::GuestResult {"
        ),
        "{main}"
    );
    let fast_start = main
        .find("if recomp_abi::direct_calls_enabled() {")
        .expect("selective call path");
    let fast_end = fast_start
        + main[fast_start..]
            .find("} else {")
            .expect("full-spill fallback");
    let fast = &main[fast_start..fast_end];
    assert!(fast.contains("unsafe { (*ctx).x[0] = x0; }"), "{main}");
    assert!(
        fast.contains("recomp_abi::call_direct(|| unsafe { fn_0000000000002000(ctx) })?;"),
        "{main}"
    );

    // One entry record per function, at its first block, and no back edges
    // in parallel mode.
    assert_eq!(
        main.matches("recomp_abi::enter_block(").count(),
        4,
        "{main}"
    );
    assert!(main.contains("recomp_abi::enter_block(0x1000);"), "{main}");
    assert!(!main.contains("recomp_abi::enter_block(0x1004);"), "{main}");
    assert!(!main.contains("recomp_abi::backedge()"), "{main}");
}

#[test]
fn guest_context_abi_spills_linear_functions() {
    let module = r#"{
  "arch": "aarch64",
  "functions": [ { "name": "entry", "ops": [
    { "op": "const_i64", "dst": "x0", "imm": 7 },
    { "op": "const_i64", "dst": "x1", "imm": 35 },
    { "op": "add_i64", "dst": "x2", "lhs": "x0", "rhs": "x1" },
    { "op": "ret" }
  ] } ]
}"#;
    let (main, _) = emit_project_with(module, CONFIG_TOML_GUEST_CONTEXT);
    assert!(main.contains(
        "unsafe fn entry(ctx: *mut recomp_abi::GuestContext) -> recomp_abi::GuestResult {"
    ));
    assert!(main.contains("let mut x2: i64 = unsafe { (*ctx).x[2] };"));
    // The return spill goes to the context, immediately before the return.
    assert!(main.contains(
        "unsafe { (*ctx).x[0] = x0; }\n    unsafe { (*ctx).x[1] = x1; }\n    unsafe { (*ctx).x[2] = x2; }\n    return Ok(());"
    ));
    assert!(!main.contains("recomp_abi::cpu_store"), "{main}");
}

#[test]
fn guest_context_is_a_title_setting() {
    let config = TitleConfig::parse(CONFIG_TOML_GUEST_CONTEXT).expect("config parses");
    assert!(config.runtime.guest_context);
    assert!(!config.runtime.direct_registers);
    let config = TitleConfig::parse(CONFIG_TOML_DEFAULT_LAYOUT).expect("config parses");
    assert!(!config.runtime.guest_context);
    assert!(!config.runtime.direct_registers);
}

#[test]
fn direct_registers_requires_the_guest_context() {
    let config = CONFIG_TOML_GUEST_CONTEXT.replace(
        "guest_context = true",
        "guest_context = true\ndirect_registers = true",
    );
    let config = TitleConfig::parse(&config).expect("config parses");
    assert!(config.runtime.direct_registers);

    let error = TitleConfig::parse(&format!(
        "{CONFIG_TOML_DEFAULT_LAYOUT}\n[runtime]\ndirect_registers = true\n"
    ))
    .expect_err("direct registers without the context must be rejected");
    assert!(error.contains("requires guest_context = true"), "{error}");
}

/// Direct register lowering: no prologue register loads, no call spills, and
/// guest accesses are context fields. The module covers integer ALU,
/// loads/stores, flags, a vector op, a nested call and the X30 check.
const DIRECT_MODULE_JSON: &str = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "blocks": [
        { "label": "bb_1000", "start": 4096, "ops": [
            { "op": "const_i64", "dst": "x0", "imm": 5 },
            { "op": "const_i64", "dst": "x1", "imm": 7 },
            { "op": "const_i64", "dst": "x9", "imm": 8192 },
            { "op": "v_const", "dst": "v0", "lo": 3, "hi": 4 },
            { "op": "add_i64", "dst": "x2", "lhs": "x0", "rhs": "x1" },
            { "op": "store_i64", "src": "x2", "addr": "x9", "offset": 0 },
            { "op": "load_i64", "dst": "x3", "addr": "x9", "offset": 0 },
            { "op": "v_bitwise", "dst": "v1", "lhs": "v0", "rhs": "v0", "kind": "eor", "q": true },
            { "op": "const_i64", "dst": "x30", "imm": 4100 }
          ],
          "terminator": { "op": "call", "target": "fn_0000000000002000", "next": "bb_1004", "return_address": 4100 } },
        { "label": "bb_1004", "start": 4100, "ops": [
            { "op": "cmp_i64", "lhs": "x0", "rhs": "x1" }
          ],
          "terminator": { "op": "br_cond", "cond": "ne", "then": "bb_1008", "else": "bb_100c" } },
        { "label": "bb_1008", "start": 4104, "ops": [
            { "op": "mov_i64", "dst": "x4", "src": "x2" }
          ], "terminator": { "op": "ret" } },
        { "label": "bb_100c", "start": 4108, "ops": [
            { "op": "sub_i64", "dst": "x4", "lhs": "x2", "rhs": "x1" }
          ], "terminator": { "op": "ret" } }
      ]
    },
    {
      "name": "fn_0000000000002000",
      "blocks": [
        { "label": "bb_2000", "start": 8192, "ops": [
            { "op": "add_i64", "dst": "x0", "lhs": "x0", "rhs": "x1" },
            { "op": "v_const", "dst": "v1", "lo": 5, "hi": 6 }
          ], "terminator": { "op": "ret" } }
      ]
    }
  ]
}"#;

const CONFIG_TOML_DIRECT: &str = r#"
title = "Direct Sample"
entry = "entry"
abi_version = "0.1.0"

[runtime]
guest_context = true
direct_registers = true

[stubs]
svc_log = "log"
"#;

#[test]
fn direct_registers_lower_guest_accesses_to_context_fields() {
    let (main, _) = emit_project_with(DIRECT_MODULE_JSON, CONFIG_TOML_DIRECT);

    // The context ABI is unchanged, but there are no register locals: only
    // instruction temps are locals.
    assert!(main.contains(
        "unsafe fn entry(ctx: *mut recomp_abi::GuestContext) -> recomp_abi::GuestResult {"
    ));
    assert!(!main.contains("let mut x0: i64"), "{main}");
    assert!(!main.contains("let mut v0: u128"), "{main}");
    assert!(!main.contains("let mut flag_"), "{main}");
    assert!(!main.contains("let nzcv ="), "{main}");
    assert!(!main.contains("recomp_abi::cpu_regs()"), "{main}");

    // Integer ALU, constants, loads and stores address the context fields.
    assert!(main.contains("unsafe { (*ctx).x[0] = 5; }"), "{main}");
    assert!(
        main.contains("unsafe { (*ctx).x[2] = (unsafe { (*ctx).x[0] }).wrapping_add((unsafe { (*ctx).x[1] })); }"),
        "{main}"
    );
    assert!(
        main.contains(
            "__mem.store_u64((unsafe { (*ctx).x[9] }) as u64, (unsafe { (*ctx).x[2] }) as u64)?;"
        ),
        "{main}"
    );
    assert!(
        main.contains(
            "unsafe { (*ctx).x[3] = __mem.load_u64((unsafe { (*ctx).x[9] }) as u64)? as i64; }"
        ),
        "{main}"
    );

    // The vector op writes the context vector file.
    assert!(
        main.contains(
            "unsafe { (*ctx).v[1] = ((unsafe { (*ctx).v[0] }) ^ (unsafe { (*ctx).v[0] })); }"
        ),
        "{main}"
    );

    // Flags are one packed context byte: CMP writes it, the branch reads it.
    assert!(
        main.contains("unsafe { (*ctx).nzcv = ((((__recomp_cmp_res < 0) as u8) << 3)"),
        "{main}"
    );
    assert!(
        main.contains("if !((unsafe { (*ctx).nzcv } & 4) != 0) {"),
        "{main}"
    );
    assert!(!main.contains("__recomp_nzcv & 8"), "{main}");

    // A direct call is just `call` with the context: no spill or reload.
    let call = "recomp_abi::call(0x2000, || unsafe { fn_0000000000002000(ctx) })?;";
    assert!(main.contains(call), "{main}");
    assert!(!main.contains("direct_calls_enabled"), "{main}");
    assert!(!main.contains("unsafe { (*ctx).x[0] = x0; }"), "{main}");
    assert!(!main.contains("x0 = unsafe { (*ctx).x[0] };"), "{main}");

    // The X30 return check reads the context.
    assert!(
        main.contains("if (unsafe { (*ctx).x[30] }) as u64 != 0x1004 {"),
        "{main}"
    );
}

#[test]
fn direct_registers_keep_the_context_off_path_unchanged() {
    // The same module with guest_context but without direct registers still
    // loads the register set into locals in the prologue.
    let (main, _) = emit_project_with(DIRECT_MODULE_JSON, CONFIG_TOML_GUEST_CONTEXT);
    assert!(
        main.contains("let mut x0: i64 = unsafe { (*ctx).x[0] };"),
        "{main}"
    );
    assert!(
        main.contains("if recomp_abi::direct_calls_enabled() {"),
        "{main}"
    );
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    format!("{:x}", digest)
}

/// FMLA 4S (three-same), FMUL 4S and FMLA by element, in direct-register
/// mode: the fast paths write the destination register in place.
const SIMD_DIRECT_MODULE_JSON: &str = r#"{
  "arch": "aarch64",
  "functions": [
    {
      "name": "entry",
      "blocks": [
        { "label": "bb_1000", "start": 4096, "ops": [
            { "op": "v_const", "dst": "v0", "lo": 3, "hi": 4 },
            { "op": "v_const", "dst": "v1", "lo": 5, "hi": 6 },
            { "op": "v_const", "dst": "v2", "lo": 7, "hi": 8 },
            { "op": "simd", "word": 1310903331, "vector_reads": ["v3", "v1", "v2"], "dst": "v3" },
            { "op": "simd", "word": 1336021024, "vector_reads": ["v0", "v1", "v2"], "dst": "v0" }
        ], "terminator": { "op": "ret" } }
      ]
    }
  ]
}"#;

#[test]
fn direct_registers_write_vector_fast_path_results_in_place() {
    let (main, _) = emit_project_with(SIMD_DIRECT_MODULE_JSON, CONFIG_TOML_DIRECT);
    // FMLA v3.4s: the whole register goes through the destination pointer,
    // so the host path can store it with one 16-byte access. The arguments
    // still read the old destination as the addend.
    let fmla = "recomp_abi::simd::fast::vector3_into(core::ptr::addr_of_mut!((*ctx).v[3]), 25u32, false, true, (unsafe { (*ctx).v[3] }), (unsafe { (*ctx).v[1] }), (unsafe { (*ctx).v[2] }))";
    assert!(main.contains(fmla), "{main}");
    // FMLA v0.4s, v1.4s, v2.s[1] (by element).
    let element = "recomp_abi::simd::fast::vector_element_into(core::ptr::addr_of_mut!((*ctx).v[0]), 0u32, false, true, false, 1u32, (unsafe { (*ctx).v[0] }), (unsafe { (*ctx).v[1] }), (unsafe { (*ctx).v[2] }))";
    assert!(main.contains(element), "{main}");
    // No assignment form for these two calls: the store lives in the host
    // path.
    assert!(
        !main.contains("(*ctx).v[3] = recomp_abi::simd::fast::vector3("),
        "{main}"
    );
}

#[test]
fn locals_mode_keeps_the_value_shaped_fast_paths() {
    let (main, _) = emit_project_with(SIMD_DIRECT_MODULE_JSON, CONFIG_TOML_GUEST_CONTEXT);
    assert!(
        main.contains("v3 = recomp_abi::simd::fast::vector3(25u32, false, true, v3, v1, v2);"),
        "{main}"
    );
    assert!(
        main.contains("v0 = recomp_abi::simd::fast::vector_element(0u32, false, true, false, 1u32, v0, v1, v2);"),
        "{main}"
    );
    assert!(!main.contains("vector3_into"), "{main}");
}
