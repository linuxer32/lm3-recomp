use recomp_pipeline::homebrew::{lift_homebrew, LiftMode, LiftOptions};
use serde_json::Value;
use std::fs;

#[test]
fn homebrew_lift_emits_stub_module() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.path();
    let segment_dir = base_dir.join("segments/sample");
    fs::create_dir_all(&segment_dir).expect("segment dir");
    let segment_path = segment_dir.join("text.bin");
    fs::write(&segment_path, [0_u8; 16]).expect("segment data");

    let module_json = format!(
        r#"{{
  "schema_version": "1",
  "module_type": "homebrew",
  "modules": [
    {{
      "name": "sample",
      "format": "nro",
      "input_path": "module.nro",
      "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "input_size": 16,
      "build_id": "deadbeef",
      "segments": [
        {{
          "name": "text",
          "file_offset": 0,
          "file_size": 16,
          "memory_offset": 0,
          "memory_size": 16,
          "permissions": "r-x",
          "output_path": "{}"
        }}
      ],
      "bss": {{ "size": 0, "memory_offset": 0 }}
    }}
  ]
}}"#,
        segment_path.strip_prefix(base_dir).unwrap().display()
    );

    let module_path = base_dir.join("module.json");
    fs::write(&module_path, module_json).expect("write module.json");

    let out_dir = base_dir.join("lifted");
    let report = lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: None,
        module_filter: None,
        extra_entries: Vec::new(),
        entry_name: "entry".to_string(),
        mode: LiftMode::Stub,
    })
    .expect("lift homebrew");

    assert_eq!(report.functions_emitted, 1);
    assert!(!report.warnings.is_empty());

    let lifted_path = out_dir.join("module.json");
    let lifted_src = fs::read_to_string(lifted_path).expect("read lifted module");
    let lifted_json: Value = serde_json::from_str(&lifted_src).expect("parse lifted module");
    assert_eq!(
        lifted_json.get("arch").and_then(|v| v.as_str()),
        Some("aarch64")
    );
    let functions = lifted_json
        .get("functions")
        .and_then(|v| v.as_array())
        .expect("functions");
    assert_eq!(functions.len(), 1);
    assert_eq!(
        functions[0].get("name").and_then(|v| v.as_str()),
        Some("entry")
    );
    let ops = functions[0]
        .get("ops")
        .and_then(|v| v.as_array())
        .expect("ops");
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].get("op").and_then(|v| v.as_str()), Some("ret"));
}

#[test]
fn homebrew_lift_decodes_minimal_block() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.path();
    let segment_dir = base_dir.join("segments/sample");
    fs::create_dir_all(&segment_dir).expect("segment dir");
    let segment_path = segment_dir.join("text.bin");

    let words = [movz_x(0, 7), movz_x(1, 35), add_reg_x(2, 0, 1), ret_x(30)];
    let mut bytes = Vec::new();
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    fs::write(&segment_path, &bytes).expect("segment data");

    let module_json = format!(
        r#"{{
  "schema_version": "1",
  "module_type": "homebrew",
  "modules": [
    {{
      "name": "sample",
      "format": "nro",
      "input_path": "module.nro",
      "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "input_size": {input_size},
      "build_id": "deadbeef",
      "segments": [
        {{
          "name": "text",
          "file_offset": 0,
          "file_size": {file_size},
          "memory_offset": 0,
          "memory_size": {file_size},
          "permissions": "r-x",
          "output_path": "{path}"
        }}
      ],
      "bss": {{ "size": 0, "memory_offset": 0 }}
    }}
  ]
}}"#,
        input_size = bytes.len(),
        file_size = bytes.len(),
        path = segment_path.strip_prefix(base_dir).unwrap().display()
    );

    let module_path = base_dir.join("module.json");
    fs::write(&module_path, module_json).expect("write module.json");

    let out_dir = base_dir.join("lifted");
    let report = lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: None,
        module_filter: None,
        extra_entries: Vec::new(),
        entry_name: "entry".to_string(),
        mode: LiftMode::Decode,
    })
    .expect("lift homebrew decode");

    assert_eq!(report.functions_emitted, 1);

    let lifted_path = out_dir.join("module.json");
    let lifted_src = fs::read_to_string(lifted_path).expect("read lifted module");
    let lifted_json: Value = serde_json::from_str(&lifted_src).expect("parse lifted module");
    let functions = lifted_json
        .get("functions")
        .and_then(|v| v.as_array())
        .expect("functions");
    let blocks = functions[0]
        .get("blocks")
        .and_then(|v| v.as_array())
        .expect("blocks");
    assert_eq!(blocks.len(), 1);
    let ops = blocks[0]
        .get("ops")
        .and_then(|v| v.as_array())
        .expect("block ops");
    assert!(ops.len() >= 3);
    assert_eq!(ops[0].get("op").and_then(|v| v.as_str()), Some("const_i64"));
    assert_eq!(ops[1].get("op").and_then(|v| v.as_str()), Some("const_i64"));
    assert_eq!(ops[2].get("op").and_then(|v| v.as_str()), Some("add_i64"));
    let terminator = blocks[0]
        .get("terminator")
        .and_then(|v| v.get("op"))
        .and_then(|v| v.as_str());
    assert_eq!(terminator, Some("ret"));
}

fn movz_x(dst: u8, imm: u16) -> u32 {
    let hw = 0_u32;
    let sf = 1_u32;
    let opc = 0b10_u32;
    let fixed = 0b100101_u32;
    (sf << 31) | (opc << 29) | (fixed << 23) | (hw << 21) | ((imm as u32) << 5) | (dst as u32)
}

fn add_reg_x(dst: u8, lhs: u8, rhs: u8) -> u32 {
    let sf = 1_u32;
    let op = 0_u32;
    let s = 0_u32;
    let opcode = 0b01011_u32;
    let shift = 0_u32;
    let imm6 = 0_u32;
    (sf << 31)
        | (op << 30)
        | (s << 29)
        | (opcode << 24)
        | (shift << 22)
        | (imm6 << 10)
        | ((rhs as u32) << 16)
        | ((lhs as u32) << 5)
        | (dst as u32)
}

fn ret_x(reg: u8) -> u32 {
    0xD65F0000 | ((reg as u32) << 5)
}

#[test]
fn homebrew_lift_discovers_call_targets() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.path();
    let segment_dir = base_dir.join("segments/sample");
    fs::create_dir_all(&segment_dir).expect("segment dir");
    let segment_path = segment_dir.join("text.bin");

    let mut bytes = vec![0_u8; 0x24];
    let bl = bl_to(0x0, 0x20);
    bytes[0..4].copy_from_slice(&bl.to_le_bytes());
    let ret0 = ret_x(30);
    bytes[4..8].copy_from_slice(&ret0.to_le_bytes());
    let ret1 = ret_x(30);
    bytes[0x20..0x24].copy_from_slice(&ret1.to_le_bytes());
    fs::write(&segment_path, &bytes).expect("segment data");

    let module_json = format!(
        r#"{{
  "schema_version": "1",
  "module_type": "homebrew",
  "modules": [
    {{
      "name": "sample",
      "format": "nro",
      "input_path": "module.nro",
      "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "input_size": {input_size},
      "build_id": "deadbeef",
      "segments": [
        {{
          "name": "text",
          "file_offset": 0,
          "file_size": {file_size},
          "memory_offset": 0,
          "memory_size": {file_size},
          "permissions": "r-x",
          "output_path": "{path}"
        }}
      ],
      "bss": {{ "size": 0, "memory_offset": 0 }}
    }}
  ]
}}"#,
        input_size = bytes.len(),
        file_size = bytes.len(),
        path = segment_path.strip_prefix(base_dir).unwrap().display()
    );

    let module_path = base_dir.join("module.json");
    fs::write(&module_path, module_json).expect("write module.json");

    let out_dir = base_dir.join("lifted");
    let report = lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: None,
        module_filter: None,
        extra_entries: Vec::new(),
        entry_name: "entry".to_string(),
        mode: LiftMode::Decode,
    })
    .expect("lift homebrew decode");

    assert_eq!(report.functions_emitted, 2);

    let lifted_path = out_dir.join("module.json");
    let lifted_src = fs::read_to_string(lifted_path).expect("read lifted module");
    let lifted_json: Value = serde_json::from_str(&lifted_src).expect("parse lifted module");
    let functions = lifted_json
        .get("functions")
        .and_then(|v| v.as_array())
        .expect("functions");
    assert_eq!(functions.len(), 2);
    let names: Vec<_> = functions
        .iter()
        .filter_map(|func| func.get("name").and_then(|v| v.as_str()))
        .collect();
    assert!(names.contains(&"entry"));
    assert!(names.contains(&"fn_0000000000000020"));
}

#[test]
fn homebrew_lift_builds_conditional_blocks() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.path();
    let segment_dir = base_dir.join("segments/sample");
    fs::create_dir_all(&segment_dir).expect("segment dir");
    let segment_path = segment_dir.join("text.bin");

    let cbz = cbz_to(0x0, 0x8, 0);
    let ret = ret_x(30);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&cbz.to_le_bytes());
    bytes.extend_from_slice(&ret.to_le_bytes());
    bytes.extend_from_slice(&ret.to_le_bytes());
    fs::write(&segment_path, &bytes).expect("segment data");

    let module_json = format!(
        r#"{{
  "schema_version": "1",
  "module_type": "homebrew",
  "modules": [
    {{
      "name": "sample",
      "format": "nro",
      "input_path": "module.nro",
      "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "input_size": {input_size},
      "build_id": "deadbeef",
      "segments": [
        {{
          "name": "text",
          "file_offset": 0,
          "file_size": {file_size},
          "memory_offset": 0,
          "memory_size": {file_size},
          "permissions": "r-x",
          "output_path": "{path}"
        }}
      ],
      "bss": {{ "size": 0, "memory_offset": 0 }}
    }}
  ]
}}"#,
        input_size = bytes.len(),
        file_size = bytes.len(),
        path = segment_path.strip_prefix(base_dir).unwrap().display()
    );

    let module_path = base_dir.join("module.json");
    fs::write(&module_path, module_json).expect("write module.json");

    let out_dir = base_dir.join("lifted");
    let report = lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: None,
        module_filter: None,
        extra_entries: Vec::new(),
        entry_name: "entry".to_string(),
        mode: LiftMode::Decode,
    })
    .expect("lift homebrew decode");

    assert_eq!(report.functions_emitted, 1);

    let lifted_path = out_dir.join("module.json");
    let lifted_src = fs::read_to_string(lifted_path).expect("read lifted module");
    let lifted_json: Value = serde_json::from_str(&lifted_src).expect("parse lifted module");
    let functions = lifted_json
        .get("functions")
        .and_then(|v| v.as_array())
        .expect("functions");
    let blocks = functions[0]
        .get("blocks")
        .and_then(|v| v.as_array())
        .expect("blocks");
    assert!(blocks.len() >= 2);
    let first_term = blocks[0]
        .get("terminator")
        .and_then(|v| v.get("op"))
        .and_then(|v| v.as_str());
    assert_eq!(first_term, Some("br_test"));
}

#[test]
fn homebrew_lift_handles_indirect_branch() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.path();
    let segment_dir = base_dir.join("segments/sample");
    fs::create_dir_all(&segment_dir).expect("segment dir");
    let segment_path = segment_dir.join("text.bin");

    let br = br_x(1);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&br.to_le_bytes());
    fs::write(&segment_path, &bytes).expect("segment data");

    let module_json = format!(
        r#"{{
  "schema_version": "1",
  "module_type": "homebrew",
  "modules": [
    {{
      "name": "sample",
      "format": "nro",
      "input_path": "module.nro",
      "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "input_size": {input_size},
      "build_id": "deadbeef",
      "segments": [
        {{
          "name": "text",
          "file_offset": 0,
          "file_size": {file_size},
          "memory_offset": 0,
          "memory_size": {file_size},
          "permissions": "r-x",
          "output_path": "{path}"
        }}
      ],
      "bss": {{ "size": 0, "memory_offset": 0 }}
    }}
  ]
}}"#,
        input_size = bytes.len(),
        file_size = bytes.len(),
        path = segment_path.strip_prefix(base_dir).unwrap().display()
    );

    let module_path = base_dir.join("module.json");
    fs::write(&module_path, module_json).expect("write module.json");

    let out_dir = base_dir.join("lifted");
    lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: None,
        module_filter: None,
        extra_entries: Vec::new(),
        entry_name: "entry".to_string(),
        mode: LiftMode::Decode,
    })
    .expect("lift homebrew decode");

    let lifted_path = out_dir.join("module.json");
    let lifted_src = fs::read_to_string(lifted_path).expect("read lifted module");
    let lifted_json: Value = serde_json::from_str(&lifted_src).expect("parse lifted module");
    let functions = lifted_json
        .get("functions")
        .and_then(|v| v.as_array())
        .expect("functions");
    let blocks = functions[0]
        .get("blocks")
        .and_then(|v| v.as_array())
        .expect("blocks");
    let term = blocks[0]
        .get("terminator")
        .and_then(|v| v.get("op"))
        .and_then(|v| v.as_str());
    assert_eq!(term, Some("br_indirect"));
}

#[test]
fn homebrew_lift_decodes_load_store_ops() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.path();
    let segment_dir = base_dir.join("segments/sample");
    fs::create_dir_all(&segment_dir).expect("segment dir");
    let segment_path = segment_dir.join("text.bin");

    let ldr = ldr_x_imm(0, 1, 16);
    let str = str_x_imm(0, 1, 16);
    let ret = ret_x(30);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&ldr.to_le_bytes());
    bytes.extend_from_slice(&str.to_le_bytes());
    bytes.extend_from_slice(&ret.to_le_bytes());
    fs::write(&segment_path, &bytes).expect("segment data");

    let module_json = format!(
        r#"{{
  "schema_version": "1",
  "module_type": "homebrew",
  "modules": [
    {{
      "name": "sample",
      "format": "nro",
      "input_path": "module.nro",
      "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "input_size": {input_size},
      "build_id": "deadbeef",
      "segments": [
        {{
          "name": "text",
          "file_offset": 0,
          "file_size": {file_size},
          "memory_offset": 0,
          "memory_size": {file_size},
          "permissions": "r-x",
          "output_path": "{path}"
        }}
      ],
      "bss": {{ "size": 0, "memory_offset": 0 }}
    }}
  ]
}}"#,
        input_size = bytes.len(),
        file_size = bytes.len(),
        path = segment_path.strip_prefix(base_dir).unwrap().display()
    );

    let module_path = base_dir.join("module.json");
    fs::write(&module_path, module_json).expect("write module.json");

    let out_dir = base_dir.join("lifted");
    lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: None,
        module_filter: None,
        extra_entries: Vec::new(),
        entry_name: "entry".to_string(),
        mode: LiftMode::Decode,
    })
    .expect("lift homebrew decode");

    let lifted_path = out_dir.join("module.json");
    let lifted_src = fs::read_to_string(lifted_path).expect("read lifted module");
    let lifted_json: Value = serde_json::from_str(&lifted_src).expect("parse lifted module");
    let functions = lifted_json
        .get("functions")
        .and_then(|v| v.as_array())
        .expect("functions");
    let ops = functions[0]
        .get("blocks")
        .and_then(|v| v.as_array())
        .expect("blocks")[0]
        .get("ops")
        .and_then(|v| v.as_array())
        .expect("ops");
    assert_eq!(ops[0].get("op").and_then(|v| v.as_str()), Some("load_i64"));
    assert_eq!(ops[1].get("op").and_then(|v| v.as_str()), Some("store_i64"));
}

#[test]
fn homebrew_lift_splits_oversized_block() {
    // 10,001 NOPs then RET: longer than one block may be.
    let mut words = vec![0xD503_201F_u32; 10_001];
    words.push(ret_x(30));
    let temp = tempfile::tempdir().expect("tempdir");
    let block = lift_single_block_in(&temp, &words);
    let lifted: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("lifted/module.json")).expect("read lifted module"),
    )
    .expect("parse lifted module");
    let blocks = lifted["functions"][0]["blocks"].as_array().expect("blocks");
    assert_eq!(blocks.len(), 2);
    assert_eq!(block["terminator"]["op"], "br");
    assert_eq!(block["terminator"]["target"], blocks[1]["label"]);
    assert_eq!(blocks[1]["terminator"]["op"], "ret");
}

fn bl_to(from: u64, target: u64) -> u32 {
    let delta = target as i64 - from as i64;
    let imm26 = (delta >> 2) & 0x03FF_FFFF;
    0x9400_0000 | (imm26 as u32)
}

fn cbz_to(from: u64, target: u64, reg: u8) -> u32 {
    let delta = target as i64 - from as i64;
    let imm19 = (delta >> 2) & 0x7FFFF;
    0xB400_0000 | ((imm19 as u32) << 5) | (reg as u32)
}

fn br_x(reg: u8) -> u32 {
    0xD61F_0000 | ((reg as u32) << 5)
}

fn ldr_x_imm(rt: u8, rn: u8, offset: u64) -> u32 {
    let size = 3_u32;
    let imm12 = (offset >> size) as u32;
    0x3900_0000 | (size << 30) | (1 << 22) | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32)
}

fn str_x_imm(rt: u8, rn: u8, offset: u64) -> u32 {
    let size = 3_u32;
    let imm12 = (offset >> size) as u32;
    0x3900_0000 | (size << 30) | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32)
}

#[test]
fn homebrew_lift_decodes_svc_instructions_to_kernel_syscalls() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.path();
    let segment_dir = base_dir.join("segments/sample");
    fs::create_dir_all(&segment_dir).expect("segment dir");
    let segment_path = segment_dir.join("text.bin");
    // movz x0,#0x1000 ; svc SetHeapSize ; movz x0,#0 ; svc ExitProcess ; ret
    let bytes: Vec<u8> = [
        0xD2840000u32,
        0xD4000021,
        0xD2800000,
        0xD40000E1,
        0xD65F03C0,
    ]
    .iter()
    .flat_map(|word| word.to_le_bytes())
    .collect();
    fs::write(&segment_path, &bytes).expect("segment data");

    let module_json = format!(
        r#"{{
  "schema_version": "1",
  "module_type": "homebrew",
  "modules": [
    {{
      "name": "sample",
      "format": "nro",
      "input_path": "module.nro",
      "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "input_size": {input_size},
      "build_id": "deadbeef",
      "segments": [
        {{
          "name": "text",
          "file_offset": 0,
          "file_size": {file_size},
          "memory_offset": 0,
          "memory_size": {file_size},
          "permissions": "r-x",
          "output_path": "{path}"
        }}
      ],
      "bss": {{ "size": 0, "memory_offset": 0 }}
    }}
  ]
}}"#,
        input_size = bytes.len(),
        file_size = bytes.len(),
        path = segment_path.strip_prefix(base_dir).unwrap().display()
    );

    let module_path = base_dir.join("module.json");
    fs::write(&module_path, module_json).expect("write module.json");

    let out_dir = base_dir.join("lifted");
    let report = lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: None,
        module_filter: None,
        extra_entries: Vec::new(),
        entry_name: "entry".to_string(),
        mode: LiftMode::Decode,
    })
    .expect("lift homebrew svc decode");

    assert_eq!(report.functions_emitted, 1);
    assert!(report.provenance_path.exists(), "lift emits provenance");

    let lifted_src = fs::read_to_string(out_dir.join("module.json")).expect("read lifted module");
    let lifted_json: Value = serde_json::from_str(&lifted_src).expect("parse lifted module");
    let ops = lifted_json
        .get("functions")
        .and_then(|v| v.as_array())
        .and_then(|funcs| funcs[0].get("blocks"))
        .and_then(|v| v.as_array())
        .and_then(|blocks| blocks[0].get("ops"))
        .and_then(|v| v.as_array())
        .expect("ops");
    let syscall_names: Vec<&str> = ops
        .iter()
        .filter(|op| op.get("op").and_then(|v| v.as_str()) == Some("syscall"))
        .filter_map(|op| op.get("name").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(syscall_names, vec!["svcSetHeapSize", "svcExitProcess"]);
}

#[test]
fn homebrew_lift_decodes_system_registers() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.path();
    let segment_dir = base_dir.join("segments/sample");
    fs::create_dir_all(&segment_dir).expect("segment dir");
    let segment_path = segment_dir.join("text.bin");

    // mrs x0, tpidr_el0; msr tpidr_el0, x1; mrs x2, nzcv;
    // msr nzcv, x3; mrs x4, cntpct_el0; ret
    let words: [u32; 6] = [
        0xD53B_D040,
        0xD51B_D041,
        0xD53B_4202,
        0xD51B_4203,
        0xD53B_E024,
        0xD65F_03C0,
    ];
    let mut bytes = Vec::new();
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    fs::write(&segment_path, &bytes).expect("segment data");

    let module_json = format!(
        r#"{{
  "schema_version": "1",
  "module_type": "homebrew",
  "modules": [
    {{
      "name": "sample",
      "format": "nro",
      "input_path": "module.nro",
      "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "input_size": {input_size},
      "build_id": "deadbeef",
      "segments": [
        {{
          "name": "text",
          "file_offset": 0,
          "file_size": {file_size},
          "memory_offset": 0,
          "memory_size": {file_size},
          "permissions": "r-x",
          "output_path": "{path}"
        }}
      ],
      "bss": {{ "size": 0, "memory_offset": 0 }}
    }}
  ]
}}"#,
        input_size = bytes.len(),
        file_size = bytes.len(),
        path = segment_path.strip_prefix(base_dir).unwrap().display()
    );

    let module_path = base_dir.join("module.json");
    fs::write(&module_path, module_json).expect("write module.json");

    let out_dir = base_dir.join("lifted");
    let report = lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: None,
        module_filter: None,
        extra_entries: Vec::new(),
        entry_name: "entry".to_string(),
        mode: LiftMode::Decode,
    })
    .expect("lift homebrew decode");
    assert_eq!(report.functions_emitted, 1);
    assert!(
        !report
            .warnings
            .iter()
            .any(|warning| warning.contains("unsupported system register")),
        "known system registers must decode: {:?}",
        report.warnings
    );

    let lifted_src = fs::read_to_string(out_dir.join("module.json")).expect("read lifted module");
    let lifted_json: Value = serde_json::from_str(&lifted_src).expect("parse lifted module");
    let ops = lifted_json["functions"][0]["blocks"][0]["ops"]
        .as_array()
        .expect("block ops");
    let names = ops
        .iter()
        .filter_map(|op| op.get("op").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "read_sys_reg",
            "write_sys_reg",
            "read_nzcv",
            "write_nzcv",
            "read_sys_reg"
        ]
    );
    assert_eq!(
        ops[0].get("reg").and_then(|value| value.as_str()),
        Some("tpidr_el0")
    );
    assert_eq!(
        ops[4].get("reg").and_then(|value| value.as_str()),
        Some("cntpct_el0")
    );
}

#[test]
fn homebrew_lift_provenance_lists_lifted_module() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.path();
    let segment_dir = base_dir.join("segments/sample");
    fs::create_dir_all(&segment_dir).expect("segment dir");
    fs::write(
        segment_dir.join("text.bin"),
        [0xD65F03C0u32.to_le_bytes()[0], 0x03, 0x5f, 0xd6],
    )
    .expect("segment data");

    let module_json = r#"{
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
      "segments": [
        {
          "name": "text",
          "file_offset": 0,
          "file_size": 4,
          "memory_offset": 0,
          "memory_size": 4,
          "permissions": "r-x",
          "output_path": "segments/sample/text.bin"
        }
      ],
      "bss": { "size": 0, "memory_offset": 0 }
    }
  ]
}"#;
    let module_path = base_dir.join("module.json");
    fs::write(&module_path, module_json).expect("write module.json");

    let out_dir = base_dir.join("lifted");
    let report = lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: None,
        module_filter: None,
        extra_entries: Vec::new(),
        entry_name: "entry".to_string(),
        mode: LiftMode::Decode,
    })
    .expect("lift homebrew");

    let provenance_src = fs::read_to_string(&report.provenance_path).expect("read provenance");
    assert!(provenance_src.contains("lifted_json"));
    assert!(provenance_src.contains("homebrew-lift"));
    let provenance = recomp_pipeline::provenance::ProvenanceManifest::parse(&provenance_src)
        .expect("parse provenance");
    let validation = provenance
        .validate(&report.provenance_path, &provenance_src)
        .expect("validate provenance");
    assert!(validation
        .inputs
        .iter()
        .any(|input| input.path == out_dir.join("module.json")));
}

#[test]
fn homebrew_lift_applies_unscaled_writeback() {
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.path();
    let segment_dir = base_dir.join("segments/sample");
    fs::create_dir_all(&segment_dir).expect("segment dir");
    let segment_path = segment_dir.join("text.bin");

    // ldr x8, [x19], #8 ; str x0, [sp, #-16]! ; ldur x1, [x2, #-8] ; ret
    let words = [0xF840_8668_u32, 0xF81F_0FE0, 0xF85F_8041, ret_x(30)];
    let mut bytes = Vec::new();
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    fs::write(&segment_path, &bytes).expect("segment data");

    let module_json = format!(
        r#"{{
  "schema_version": "1",
  "module_type": "homebrew",
  "modules": [
    {{
      "name": "sample",
      "format": "nro",
      "input_path": "module.nro",
      "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "input_size": {input_size},
      "build_id": "deadbeef",
      "segments": [
        {{
          "name": "text",
          "file_offset": 0,
          "file_size": {file_size},
          "memory_offset": 0,
          "memory_size": {file_size},
          "permissions": "r-x",
          "output_path": "{path}"
        }}
      ],
      "bss": {{ "size": 0, "memory_offset": 0 }}
    }}
  ]
}}"#,
        input_size = bytes.len(),
        file_size = bytes.len(),
        path = segment_path.strip_prefix(base_dir).unwrap().display()
    );

    let module_path = base_dir.join("module.json");
    fs::write(&module_path, module_json).expect("write module.json");

    let out_dir = base_dir.join("lifted");
    lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: None,
        module_filter: None,
        extra_entries: Vec::new(),
        entry_name: "entry".to_string(),
        mode: LiftMode::Decode,
    })
    .expect("lift homebrew decode");

    let lifted_src = fs::read_to_string(out_dir.join("module.json")).expect("read lifted module");
    let lifted_json: Value = serde_json::from_str(&lifted_src).expect("parse lifted module");
    let ops = lifted_json["functions"][0]["blocks"][0]["ops"]
        .as_array()
        .expect("ops");
    let summary: Vec<String> = ops
        .iter()
        .map(|op| {
            let field = |name: &str| op.get(name).map(|v| v.to_string()).unwrap_or_default();
            format!(
                "{} {}{} {}{}",
                op["op"].as_str().unwrap_or(""),
                field("dst"),
                field("src"),
                field("addr"),
                field("offset")
            )
        })
        .collect();

    // Post-index: access at the old base, then advance it.
    assert_eq!(summary[0], r#"load_i64 "x8" "x19"0"#);
    assert_eq!(ops[1]["imm"], 8);
    assert_eq!(summary[2], r#"add_i64 "x19" "#);
    // Pre-index: advance SP first, then access the updated address.
    assert_eq!(ops[3]["imm"], -16);
    assert_eq!(summary[4], r#"add_i64 "x31" "#);
    assert_eq!(summary[5], r#"store_i64 "x0" "x31"0"#);
    // Unscaled: plain signed offset, no writeback.
    assert_eq!(summary[6], r#"load_i64 "x1" "x2"-8"#);
    assert_eq!(ops.len(), 7);
}

/// Lift `words` as one function at address 0 and return its first block.
fn lift_single_block(words: &[u32]) -> Value {
    let temp = tempfile::tempdir().expect("tempdir");
    lift_single_block_in(&temp, words)
}

fn lift_single_block_in(temp: &tempfile::TempDir, words: &[u32]) -> Value {
    let base_dir = temp.path();
    let segment_dir = base_dir.join("segments/sample");
    fs::create_dir_all(&segment_dir).expect("segment dir");
    let segment_path = segment_dir.join("text.bin");
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    fs::write(&segment_path, &bytes).expect("segment data");
    let module_json = serde_json::json!({
        "schema_version": "1",
        "module_type": "homebrew",
        "modules": [{
            "name": "sample",
            "format": "nro",
            "input_path": "module.nro",
            "input_sha256": "a".repeat(64),
            "input_size": bytes.len(),
            "build_id": "deadbeef",
            "segments": [{
                "name": "text",
                "file_offset": 0,
                "file_size": bytes.len(),
                "memory_offset": 0,
                "memory_size": bytes.len(),
                "permissions": "r-x",
                "output_path": "segments/sample/text.bin"
            }],
            "bss": { "size": 0, "memory_offset": 0 }
        }]
    });
    let module_path = base_dir.join("module.json");
    fs::write(&module_path, module_json.to_string()).expect("write module.json");
    let out_dir = base_dir.join("lifted");
    lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: None,
        module_filter: None,
        extra_entries: Vec::new(),
        entry_name: "entry".to_string(),
        mode: LiftMode::Decode,
    })
    .expect("lift homebrew decode");
    let lifted_src = fs::read_to_string(out_dir.join("module.json")).expect("read lifted module");
    let lifted: Value = serde_json::from_str(&lifted_src).expect("parse lifted module");
    lifted["functions"][0]["blocks"][0].clone()
}

#[test]
fn homebrew_lift_routes_fp_and_simd_to_the_evaluator() {
    // ucvtf s0, w8 ; fcmp s0, s1 ; fcsel s2, s0, s1, eq ; fcvtzs wzr, s2 ;
    // scvtf d3, xzr ; fadd v4.4s, v5.4s, v6.4s ; udf #0
    let block = lift_single_block(&[
        0x1E23_0100,
        0x1E21_2000,
        0x1E21_0C02,
        0x1E38_005F,
        0x9E62_03E3,
        0x4E26_D4A4,
        0x0000_0000,
    ]);
    let ops = block["ops"].as_array().expect("ops");
    let simd: Vec<&Value> = ops.iter().filter(|op| op["op"] == "simd").collect();
    assert_eq!(simd.len(), 6, "{ops:#?}");
    assert_eq!(simd[0]["word"], 0x1E23_0100);
    assert_eq!(simd[0]["general_read"], "x8");
    assert_eq!(simd[0]["dst"], "v0");
    assert_eq!(simd[1]["vector_reads"], serde_json::json!(["v0", "v1"]));
    assert_eq!(simd[1]["dst"], Value::Null);
    assert_eq!(simd[1]["writes_flags"], true);
    assert_eq!(simd[2]["reads_flags"], true);
    assert_eq!(simd[2]["dst"], "v2");
    // The zero register is neither written nor read.
    assert_eq!(simd[3]["dst"], Value::Null);
    assert_eq!(simd[4]["general_read"], Value::Null);
    assert_eq!(simd[4]["dst"], "v3");
    assert_eq!(simd[5]["vector_reads"], serde_json::json!(["v5", "v6"]));
}

#[test]
fn homebrew_lift_decodes_literal_loads_carries_and_cache_ops() {
    // ldr w20, #8 ; ldr q1, #-4 ; adcs x0, x1, x2 ; sbc w3, w4, wzr ;
    // dc civac, x10 ; ic ivau, x9 ; udf #0
    let block = lift_single_block(&[
        0x1800_0054,
        0x9CFF_FFE1,
        0xBA02_0020,
        0x5A1F_0083,
        0xD50B_7E2A,
        0xD50B_7529,
        0x0000_0000,
    ]);
    let ops = block["ops"].as_array().expect("ops");
    let named = |name: &str| -> Vec<&Value> { ops.iter().filter(|op| op["op"] == name).collect() };
    let literals = named("pc_rel");
    assert_eq!(
        (literals[0]["pc"].as_i64(), literals[0]["offset"].as_i64()),
        (Some(0), Some(8))
    );
    assert_eq!(
        (literals[1]["pc"].as_i64(), literals[1]["offset"].as_i64()),
        (Some(4), Some(-4))
    );
    let load = named("load_i32")[0];
    assert_eq!(load["dst"], "x20");
    assert_eq!(load["addr"], literals[0]["dst"]);
    let vload = named("v_load")[0];
    assert_eq!(
        (vload["dst"].as_str(), vload["bytes"].as_u64()),
        (Some("v1"), Some(16))
    );
    let carries = named("add_carry");
    assert_eq!(carries[0]["dst"], "x0");
    assert_eq!(carries[0]["lhs"], "x1");
    assert_eq!(carries[0]["rhs"], "x2");
    assert_eq!(carries[0]["set_flags"], true);
    assert_eq!(carries[0]["is_32"], false);
    assert_eq!(carries[1]["dst"], "x3");
    assert_eq!(carries[1]["subtract"], true);
    assert_eq!(carries[1]["is_32"], true);
    assert_ne!(carries[1]["rhs"], "x31", "wzr reads zero, not sp");
    // The cache maintenance instructions decode; only udf traps.
    assert_eq!(block["terminator"]["op"], "trap");
    assert_eq!(block["terminator"]["pc"], 24);
}

#[test]
fn homebrew_lift_expands_simd_structure_loads_and_stores() {
    let lift = |word: u32| {
        lift_single_block(&[word, 0x0000_0000])["ops"]
            .as_array()
            .expect("ops")
            .clone()
    };
    let names = |ops: &[Value]| {
        ops.iter()
            .map(|op| op["op"].as_str().unwrap_or("").to_string())
            .collect::<Vec<_>>()
    };

    // ld1 {v0.4s, v1.4s}, [x0], #32
    let ops = lift(0x4CDF_A800);
    assert_eq!(names(&ops), ["v_load", "v_load", "const_i64", "add_i64"]);
    assert_eq!(
        (ops[1]["dst"].as_str(), ops[1]["offset"].as_i64()),
        (Some("v1"), Some(16))
    );
    assert_eq!(ops[2]["imm"], 32);
    assert_eq!(ops[3]["dst"], "x0");

    // ld4 {v0.4s-v3.4s}, [x1]: registers cleared, then lanes interleaved.
    let ops = lift(0x4C40_0820);
    assert_eq!(ops.iter().filter(|op| op["op"] == "v_const").count(), 4);
    let inserts: Vec<&Value> = ops.iter().filter(|op| op["op"] == "v_ins_gpr").collect();
    assert_eq!(inserts.len(), 16);
    assert_eq!(
        (inserts[1]["dst"].as_str(), inserts[1]["index"].as_u64()),
        (Some("v1"), Some(0))
    );
    assert_eq!(
        (inserts[4]["dst"].as_str(), inserts[4]["index"].as_u64()),
        (Some("v0"), Some(1))
    );
    let loads: Vec<&Value> = ops.iter().filter(|op| op["op"] == "load_i32").collect();
    assert_eq!(loads[5]["offset"], 20);

    // ld1r {v5.4s}, [sp]
    let ops = lift(0x4D40_CBE5);
    assert_eq!(names(&ops), ["load_i32", "v_dup"]);
    assert_eq!(ops[0]["addr"], "x31");
    assert_eq!(
        (
            ops[1]["dst"].as_str(),
            ops[1]["esize"].as_u64(),
            ops[1]["q"].as_bool()
        ),
        (Some("v5"), Some(4), Some(true))
    );

    // st1 {v6.s}[1], [x3]
    let ops = lift(0x0D00_9066);
    assert_eq!(names(&ops), ["v_extract", "store_i32"]);
    assert_eq!(ops[0]["index"], 1);

    // ld1 {v7.b}[9], [x4], x5
    let ops = lift(0x4DC5_0487);
    assert_eq!(names(&ops), ["load_i8", "v_ins_gpr", "add_i64"]);
    assert_eq!(ops[1]["index"], 9);
    assert_eq!(
        (ops[2]["dst"].as_str(), ops[2]["rhs"].as_str()),
        (Some("x4"), Some("x5"))
    );

    // st2 {v30.2d, v31.2d}, [x6]: v30[0], v31[0], v30[1], v31[1].
    let ops = lift(0x4C00_8CDE);
    let extracts: Vec<(Option<&str>, Option<u64>)> = ops
        .iter()
        .filter(|op| op["op"] == "v_extract")
        .map(|op| (op["src"].as_str(), op["index"].as_u64()))
        .collect();
    assert_eq!(
        extracts,
        [
            (Some("v30"), Some(0)),
            (Some("v31"), Some(0)),
            (Some("v30"), Some(1)),
            (Some("v31"), Some(1))
        ]
    );
    let stores: Vec<i64> = ops
        .iter()
        .filter(|op| op["op"] == "store_i64")
        .map(|op| op["offset"].as_i64().unwrap_or(-1))
        .collect();
    assert_eq!(stores, [0, 8, 16, 24]);

    // ld3r {v8.8b, v9.8b, v10.8b}, [x7], #3
    let ops = lift(0x0DDF_E0E8);
    assert_eq!(
        ops.iter()
            .filter(|op| op["op"] == "v_dup" && op["q"] == false)
            .count(),
        3
    );
    assert_eq!(ops.iter().rev().nth(1).expect("imm")["imm"], 3);
}

#[test]
fn homebrew_lift_decodes_immediates_and_signed_loads() {
    // movz w0, #1 ; ldrsw x1, [x2] ; orr w3, wzr, #0x3f800000 ; udf #0
    let block = lift_single_block(&[0x5280_0020, 0xB980_0041, 0x3209_1BE3, 0x0000_0000]);
    let ops = block["ops"].as_array().expect("ops");
    let has = |op: &str, field: &str, value: Value| {
        ops.iter()
            .any(|item| item["op"] == op && item[field] == value)
    };

    // The MOVZ immediate is bits 20:5.
    assert_eq!(ops[0]["op"], "const_i64");
    assert_eq!(ops[0]["dst"], "x0");
    assert_eq!(ops[0]["imm"], 1);
    // LDRSW (opc=10) is a load that sign-extends, never a store.
    assert!(has("load_i32", "dst", "x1".into()));
    assert!(!ops.iter().any(|item| item["op"] == "store_i32"));
    assert!(has("asr_i64", "dst", "x1".into()));
    // ORR (immediate) decodes its bitmask.
    assert!(has("const_i64", "imm", 0x3F80_0000.into()));
    assert!(has("or_i64", "dst", "x3".into()));
    // An unsupported encoding ends the block in a trap, not a return.
    assert_eq!(block["terminator"]["op"], "trap");
    assert_eq!(block["terminator"]["pc"], 12);
    assert_eq!(block["terminator"]["word"], 0);
}

#[test]
fn homebrew_lifts_exclusive_accesses_and_barriers() {
    // Encodings assembled with llvm-mc: ldxr x0,[x1] ; stxr w4,x1,[x2] ;
    // ldxp x0,x1,[x2] ; stxp w4,x1,x2,[x5] ; ldar x5,[x7] ; stlr x6,[x7] ;
    // clrex ; dmb ish ; isb
    let block = lift_single_block(&[
        0xC85F_7C20,
        0xC804_7C41,
        0xC87F_0440,
        0xC824_08A1,
        0xC8DF_7CE5,
        0xC89F_FCE6,
        0xD503_3F5F,
        0xD503_3BBF,
        0xD503_3FDF,
        ret_x(30),
    ]);
    let ops = block["ops"].as_array().expect("ops");

    // LDXR reserves the address; STXR's status register gets the result.
    let load = ops
        .iter()
        .find(|item| item["op"] == "exclusive_load" && item["pair"] == Value::Bool(false))
        .expect("ldxr becomes an exclusive load");
    assert_eq!(load["dst"], "x0");
    assert_eq!(load["size"], 8);
    assert_eq!(load["dst2"], Value::Null);

    let store = ops
        .iter()
        .find(|item| item["op"] == "exclusive_store" && item["pair"] == Value::Bool(false))
        .expect("stxr becomes an exclusive store");
    assert_eq!(store["status"], "x4");
    assert_eq!(store["src"], "x1");
    assert_eq!(store["src2"], Value::Null);

    // The pair forms carry a second register and reserve both words.
    let pair_load = ops
        .iter()
        .find(|item| item["op"] == "exclusive_load" && item["pair"] == Value::Bool(true))
        .expect("ldxp becomes a pair exclusive load");
    assert_eq!(pair_load["dst"], "x0");
    assert_eq!(pair_load["dst2"], "x1");

    let pair_store = ops
        .iter()
        .find(|item| item["op"] == "exclusive_store" && item["pair"] == Value::Bool(true))
        .expect("stxp becomes a pair exclusive store");
    assert_eq!(pair_store["status"], "x4");
    assert_eq!(pair_store["src"], "x1");
    assert_eq!(pair_store["src2"], "x2");

    // LDAR is not exclusive: a plain load, then an acquire fence (an AArch64
    // host reorders plain accesses); STLR a release fence, then the store.
    let ldar = ops
        .iter()
        .position(|item| item["op"] == "load_i64" && item["dst"] == "x5")
        .expect("ldar becomes a plain load");
    assert_eq!(ops[ldar + 1]["op"], "acquire_fence");
    assert!(!ops
        .iter()
        .any(|item| item["op"] == "exclusive_load" && item["dst"] == "x5"));
    let stlr = ops
        .iter()
        .position(|item| item["op"] == "release_fence")
        .expect("stlr orders what came before");
    assert!(ops[stlr + 1..].iter().take(3).any(|item| item["op"]
        .as_str()
        .is_some_and(|op| op.starts_with("store"))));

    // CLREX drops the reservation; DMB and ISB become host fences.
    assert_eq!(
        ops.iter()
            .filter(|item| item["op"] == "clear_exclusive")
            .count(),
        1
    );
    assert_eq!(
        ops.iter()
            .filter(|item| item["op"] == "memory_fence")
            .count(),
        2
    );
    // No op traps: every one of these encodings is handled.
    assert_eq!(block["terminator"]["op"], "ret");
}

#[test]
fn homebrew_lift_calls_set_the_link_register() {
    // blr x30 at 0x0: the target must be read before LR is overwritten.
    let block = lift_single_block(&[0xD63F_03C0, ret_x(30)]);
    let ops = block["ops"].as_array().expect("ops");
    assert_eq!(ops[0]["op"], "mov_i64");
    assert_eq!(ops[0]["src"], "x30");
    let target = ops[0]["dst"].clone();
    assert_eq!(ops[1]["op"], "const_i64");
    assert_eq!(ops[1]["dst"], "x30");
    assert_eq!(ops[1]["imm"], 4);
    assert_eq!(block["terminator"]["op"], "call_indirect");
    assert_eq!(block["terminator"]["reg"], target);

    // bl +8 at 0x0 sets x30 = 4 (rtld reads it as data).
    let block = lift_single_block(&[bl_to(0, 8), 0, ret_x(30)]);
    let ops = block["ops"].as_array().expect("ops");
    assert_eq!(ops.last().expect("op")["dst"], "x30");
    assert_eq!(ops.last().expect("op")["imm"], 4);
    assert_eq!(block["terminator"]["op"], "call");
}

#[test]
fn homebrew_lift_recovers_jump_tables() {
    // cmp x0, #2 ; b.hi 0x30 ; adrp x1, 0 ; add x1, x1, #0x40
    // ldrsw x2, [x1, x0, lsl #2] ; add x2, x2, x1 ; br x2
    // 0x1c/0x20/0x24: the three cases (ret) ; 0x30: default (ret)
    // 0x40: int32 offsets of the cases relative to the table
    let mut words = vec![
        0xF100_081F,
        0x5400_0168,
        0x9000_0001,
        0x9101_0021,
        0xB8A0_7822,
        0x8B01_0042,
        0xD61F_0040,
        ret_x(30),
        ret_x(30),
        ret_x(30),
        0xD503_201F,
        0xD503_201F,
        ret_x(30),
        0xD503_201F,
        0xD503_201F,
        0xD503_201F,
    ];
    for target in [0x1c_i32, 0x20, 0x24] {
        words.push((target - 0x40) as u32);
    }
    let temp = tempfile::tempdir().expect("tempdir");
    lift_single_block_in(&temp, &words);
    let lifted: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("lifted/module.json")).expect("read lifted module"),
    )
    .expect("parse lifted module");
    let blocks = lifted["functions"][0]["blocks"].as_array().expect("blocks");
    let switch = blocks
        .iter()
        .find(|block| block["terminator"]["op"] == "switch")
        .expect("br through the table becomes a switch");
    let cases: Vec<u64> = switch["terminator"]["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| case[0].as_u64().expect("case address"))
        .collect();
    assert_eq!(cases, vec![0x1c, 0x20, 0x24]);
    // Every case is decoded as a block of the same function.
    for case in [0x1c_u64, 0x20, 0x24] {
        assert!(blocks.iter().any(|block| block["start"] == case));
    }
}

#[test]
fn homebrew_lift_follows_moves_to_the_jump_table_bound() {
    // cmp w3, #2 ; b.hi 0x34 ; mov w0, w3 ; adrp x1, 0 ; add x1, x1, #0x40
    // ldrsw x2, [x1, x0, lsl #2] ; add x2, x2, x1 ; br x2
    let mut words = vec![
        0x7100_087F,
        0x5400_0188,
        0x2A03_03E0,
        0x9000_0001,
        0x9101_0021,
        0xB8A0_7822,
        0x8B01_0042,
        0xD61F_0040,
        ret_x(30),
        ret_x(30),
        ret_x(30),
        0xD503_201F,
        0xD503_201F,
        ret_x(30),
        0xD503_201F,
        0xD503_201F,
    ];
    for target in [0x20_i32, 0x24, 0x28] {
        words.push((target - 0x40) as u32);
    }
    let temp = tempfile::tempdir().expect("tempdir");
    lift_single_block_in(&temp, &words);
    let lifted: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("lifted/module.json")).expect("read lifted module"),
    )
    .expect("parse lifted module");
    let switch = lifted["functions"][0]["blocks"]
        .as_array()
        .expect("blocks")
        .iter()
        .find(|block| block["terminator"]["op"] == "switch")
        .expect("switch recovered");
    assert_eq!(
        switch["terminator"]["cases"]
            .as_array()
            .expect("cases")
            .len(),
        3
    );
}

#[test]
fn homebrew_lift_finds_the_bound_past_a_clobbered_index_register() {
    // LM3 0x805abfc: cmp w1, #6 ; b.hi 0x5c ; adrp x8 (x8 reused) ;
    // mov v9, v2 ; ldr s2, [x8] ; mov w8, w1 ; adrp x9 ; add x9, #0x80 ;
    // ldrsw x8, [x9, x8, lsl #2] ; fmul ; add x8, x8, x9 ; mov v8, v3 ;
    // fminnm ; br x8. Cases at 0x38-0x50, default at 0x5c, table at 0x80.
    let mut words = vec![
        0x7100_183F,
        0x5400_02C8,
        0x9000_0008,
        0x4EA2_1C49,
        0xBD42_D902,
        0x2A01_03E8,
        0x9000_0009,
        0x9102_0129,
        0xB8A8_7928,
        0x1E22_0802,
        0x8B09_0108,
        0x4EA3_1C68,
        0x1E21_7841,
        0xD61F_0100,
    ];
    words.extend([ret_x(30); 7]);
    words.extend([0xD503_201F; 2]);
    words.push(ret_x(30));
    words.extend([0xD503_201F; 8]);
    for target in (0x38..=0x50).step_by(4) {
        words.push((target - 0x80_i32) as u32);
    }
    let temp = tempfile::tempdir().expect("tempdir");
    lift_single_block_in(&temp, &words);
    let lifted: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("lifted/module.json")).expect("read lifted module"),
    )
    .expect("parse lifted module");
    let switch = lifted["functions"][0]["blocks"]
        .as_array()
        .expect("blocks")
        .iter()
        .find(|block| block["terminator"]["op"] == "switch")
        .expect("switch recovered");
    assert_eq!(
        switch["terminator"]["cases"]
            .as_array()
            .expect("cases")
            .len(),
        7
    );
}

#[test]
fn homebrew_lift_finds_the_guard_of_a_separate_table_block() {
    // 0x00 cmp x8, #5 ; b.ls 0x10 ; ret ; b 0x08
    // 0x10 adrp x9, 0 ; add x9, x9, #0x40 ; ldrsw x8, [x9, x8, lsl #2]
    //      add x8, x8, x9 ; br x8
    // 0x24-0x38: six cases (ret) ; 0x40: the table, then data that is not
    // a valid entry (an off-by-one count would fail the whole table).
    let mut words = vec![
        0xF100_151F,
        0x5400_0069,
        ret_x(30),
        0x17FF_FFFF,
        0x9000_0009,
        0x9101_0129,
        0xB8A8_7928,
        0x8B09_0108,
        0xD61F_0100,
    ];
    words.extend([ret_x(30); 6]);
    words.push(0xD503_201F);
    for target in (0x24..=0x38).step_by(4) {
        words.push((target - 0x40_i32) as u32);
    }
    words.push(0x7FFF_FFF0);
    let temp = tempfile::tempdir().expect("tempdir");
    lift_single_block_in(&temp, &words);
    let lifted: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("lifted/module.json")).expect("read lifted module"),
    )
    .expect("parse lifted module");
    let switch = lifted["functions"][0]["blocks"]
        .as_array()
        .expect("blocks")
        .iter()
        .find(|block| block["terminator"]["op"] == "switch")
        .expect("switch recovered");
    let cases: Vec<u64> = switch["terminator"]["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| case[0].as_u64().expect("case address"))
        .collect();
    assert_eq!(cases, vec![0x24, 0x28, 0x2c, 0x30, 0x34, 0x38]);
}

#[test]
fn homebrew_lift_bounds_a_table_through_a_masked_copy() {
    // and w10, w27, #0xff ; and x9, x27, #0xff ; cmp w10, #2 ; b.hi 0x34
    // adrp x23, 0 ; add x23, x23, #0x40 ; ldrsw x9, [x23, x9, lsl #2]
    // add x9, x9, x23 ; br x9 ; 0x24-0x2c: three cases ; 0x34: default
    let mut words = vec![
        0x1200_1F6A,
        0x9240_1F69,
        0x7100_095F,
        0x5400_0148,
        0x9000_0017,
        0x9101_02F7,
        0xB8A9_7AE9,
        0x8B17_0129,
        0xD61F_0120,
        ret_x(30),
        ret_x(30),
        ret_x(30),
        0xD503_201F,
        ret_x(30),
        0xD503_201F,
        0xD503_201F,
    ];
    for target in [0x24_i32, 0x28, 0x2c] {
        words.push((target - 0x40) as u32);
    }
    words.push(0x7FFF_FFF0);
    let temp = tempfile::tempdir().expect("tempdir");
    lift_single_block_in(&temp, &words);
    let lifted: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("lifted/module.json")).expect("read lifted module"),
    )
    .expect("parse lifted module");
    let switch = lifted["functions"][0]["blocks"]
        .as_array()
        .expect("blocks")
        .iter()
        .find(|block| block["terminator"]["op"] == "switch")
        .expect("switch recovered");
    assert_eq!(
        switch["terminator"]["cases"]
            .as_array()
            .expect("cases")
            .len(),
        3
    );
}

#[test]
fn homebrew_lift_bounds_a_table_compared_after_the_move() {
    // mov w8, w1 ; cmp w1, #2 ; b.hi 0x30 ; adrp x9, 0 ; add x9, x9, #0x40
    // ldrsw x10, [x9, x8, lsl #2] ; add x9, x10, x9 ; br x9
    // 0x20-0x28: three cases ; 0x30: default
    let mut words = vec![
        0x2A01_03E8,
        0x7100_083F,
        0x5400_0148,
        0x9000_0009,
        0x9101_0129,
        0xB8A8_792A,
        0x8B09_0149,
        0xD61F_0120,
        ret_x(30),
        ret_x(30),
        ret_x(30),
        0xD503_201F,
        ret_x(30),
        0xD503_201F,
        0xD503_201F,
        0xD503_201F,
    ];
    for target in [0x20_i32, 0x24, 0x28] {
        words.push((target - 0x40) as u32);
    }
    words.push(0x7FFF_FFF0);
    let temp = tempfile::tempdir().expect("tempdir");
    lift_single_block_in(&temp, &words);
    let lifted: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("lifted/module.json")).expect("read lifted module"),
    )
    .expect("parse lifted module");
    let switch = lifted["functions"][0]["blocks"]
        .as_array()
        .expect("blocks")
        .iter()
        .find(|block| block["terminator"]["op"] == "switch")
        .expect("switch recovered");
    assert_eq!(
        switch["terminator"]["cases"]
            .as_array()
            .expect("cases")
            .len(),
        3
    );
}

/// The `switch` terminator lifted from `words` (a table at 0x40 with three
/// case entries and the default's), if the table was recovered. A bounds
/// check yields three cases; reading to the table's end, four.
fn lift_table_switch(words: &[u32]) -> Option<Value> {
    let mut words = words.to_vec();
    words.resize(16, 0xD503_201F);
    for target in [0x28_i32, 0x2c, 0x30, 0x34] {
        words.push((target - 0x40) as u32);
    }
    words.push(0x7FFF_FFF0);
    let temp = tempfile::tempdir().expect("tempdir");
    lift_single_block_in(&temp, &words);
    let lifted: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("lifted/module.json")).expect("read lifted module"),
    )
    .expect("parse lifted module");
    lifted["functions"][0]["blocks"]
        .as_array()
        .expect("blocks")
        .iter()
        .find(|block| block["terminator"]["op"] == "switch")
        .map(|block| block["terminator"].clone())
}

#[test]
fn homebrew_lift_bounds_a_table_through_an_earlier_mask_of_the_index() {
    // and w9, w27, #0xff ; cmp w9, #2 ; b.hi 0x34 ; nop
    // and x9, x27, #0xff ; adrp x23, 0 ; add x23, x23, #0x40
    // ldrsw x9, [x23, x9, lsl #2] ; add x9, x9, x23 ; br x9
    // 0x28-0x30: three cases ; 0x34: default
    let table = [
        0x9240_1F69,
        0x9000_0017,
        0x9101_02F7,
        0xB8A9_7AE9,
        0x8B17_0129,
        0xD61F_0120,
        ret_x(30),
        ret_x(30),
        ret_x(30),
        ret_x(30),
    ];
    let mut words = vec![0x1200_1F69, 0x7100_093F, 0x5400_0168, 0xD503_201F];
    words.extend(table);
    let switch = lift_table_switch(&words).expect("switch recovered");
    assert_eq!(switch["cases"].as_array().expect("cases").len(), 3);

    // add x27, x27, #1 between the ANDs: the compare tests another value,
    // so the table is read without its bound.
    words[3] = 0x9100_077B;
    let switch = lift_table_switch(&words).expect("switch recovered");
    assert_eq!(switch["cases"].as_array().expect("cases").len(), 4);
}

#[test]
fn homebrew_lift_finds_a_table_address_hoisted_into_the_prologue() {
    // adrp x28, 0 ; add x28, x28, #0xd0 ; 40 nops
    // 0xa8: cmp w8, #2 ; b.hi 0xc8 ; ldrsw x8, [x28, x8, lsl #2]
    // add x8, x8, x28 ; br x8 ; 0xbc-0xc4: three cases ; 0xc8: default
    let mut words = vec![0x9000_001C, 0x9103_439C];
    words.extend([0xD503_201F; 40]);
    words.extend([
        0x7100_091F,
        0x5400_00E8,
        0xB8A8_7B88,
        0x8B1C_0108,
        0xD61F_0100,
        ret_x(30),
        ret_x(30),
        ret_x(30),
        ret_x(30),
        0xD503_201F,
    ]);
    assert_eq!(words.len() * 4, 0xD0);
    for target in [0xBC_i32, 0xC0, 0xC4] {
        words.push((target - 0xD0) as u32);
    }
    let temp = tempfile::tempdir().expect("tempdir");
    lift_single_block_in(&temp, &words);
    let lifted: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("lifted/module.json")).expect("read lifted module"),
    )
    .expect("parse lifted module");
    let switch = lifted["functions"][0]["blocks"]
        .as_array()
        .expect("blocks")
        .iter()
        .find(|block| block["terminator"]["op"] == "switch")
        .expect("switch recovered");
    let cases: Vec<u64> = switch["terminator"]["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| case[0].as_u64().unwrap_or(0))
        .collect();
    assert_eq!(cases, [0xBC, 0xC0, 0xC4]);
}

#[test]
fn homebrew_lift_follows_a_table_index_through_its_stack_slot() {
    // mov w8, w21 ; str x8, [sp, #0x98] ; 40 nops
    // 0xa8: cmp w21, #2 ; b.hi 0xd4 ; ldr x8, [sp, #0x98] ; adrp x9, 0
    // add x9, x9, #0xe0 ; ldrsw x8, [x9, x8, lsl #2] ; add x8, x8, x9
    // br x8 ; 0xc8-0xd0: three cases ; 0xd4: default ; 0xe0: table
    let lift_switch = |words: &[u32]| {
        let temp = tempfile::tempdir().expect("tempdir");
        lift_single_block_in(&temp, words);
        let lifted: Value = serde_json::from_str(
            &fs::read_to_string(temp.path().join("lifted/module.json"))
                .expect("read lifted module"),
        )
        .expect("parse lifted module");
        lifted["functions"][0]["blocks"]
            .as_array()
            .expect("blocks")
            .iter()
            .find(|block| block["terminator"]["op"] == "switch")
            .map(|block| {
                block["terminator"]["cases"]
                    .as_array()
                    .expect("cases")
                    .len()
            })
    };
    let mut words = vec![0x2A15_03E8, 0xF900_4FE8];
    words.extend([0xD503_201F; 40]);
    words.extend([
        0x7100_0ABF,
        0x5400_0148,
        0xF940_4FE8,
        0x9000_0009,
        0x9103_8129,
        0xB8A8_7928,
        0x8B09_0108,
        0xD61F_0100,
        ret_x(30),
        ret_x(30),
        ret_x(30),
        ret_x(30),
        0xD503_201F,
        0xD503_201F,
    ]);
    assert_eq!(words.len() * 4, 0xE0);
    for target in [0xC8_i32, 0xCC, 0xD0, 0xD4] {
        words.push((target - 0xE0) as u32);
    }
    assert_eq!(lift_switch(&words), Some(3));

    // add w21, w21, #1 after the spill: the compare tests another value,
    // so the table is read without its bound.
    words[10] = 0x1100_06B5;
    assert_eq!(lift_switch(&words), Some(4));
}

#[test]
fn homebrew_lift_follows_a_copy_of_the_table_address() {
    // cmp w21, #2 ; b.hi 0x34 ; adrp x8, 0 ; add x8, x8, #0x40 ; mov x9, x8
    // ldrsw x8, [x8, x21, lsl #2] ; add x8, x8, x9 ; br x8
    // 0x28-0x30: three cases ; 0x34: default
    let mut words = vec![
        0x7100_0ABF,
        0x5400_0188,
        0x9000_0008,
        0x9101_0108,
        0xAA08_03E9,
        0xB8B5_7908,
        0x8B09_0108,
        0xD61F_0100,
    ];
    words.extend([ret_x(30); 6]);
    let switch = lift_table_switch(&words).expect("switch recovered");
    assert_eq!(switch["cases"].as_array().expect("cases").len(), 3);
}

#[test]
fn homebrew_lift_reads_an_unguarded_table_up_to_the_next_function() {
    // tbnz w10, #2, 0x30 (no compare bounds the index) ; adrp x11, 0
    // add x11, x11, #0x40 ; ldrsw x10, [x11, x10, lsl #2] ; add x10, x10, x11
    // br x10 ; 0x18-0x20: three cases ; 0x2c: bl 0x34 ; 0x30: default
    // 0x34: another function ; 0x40: the table, then entries past its end
    let mut words = vec![
        0x3710_018A,
        0x9000_000B,
        0x9101_016B,
        0xB8AA_796A,
        0x8B0B_014A,
        0xD61F_0140,
        ret_x(30),
        ret_x(30),
        ret_x(30),
        0xD503_201F,
        0xD503_201F,
        0x9400_0002,
        ret_x(30),
        ret_x(30),
        0xD503_201F,
        0xD503_201F,
    ];
    for target in [0x18_i32, 0x1c, 0x20, 0x34, 0x18] {
        words.push((target - 0x40) as u32);
    }
    let temp = tempfile::tempdir().expect("tempdir");
    lift_single_block_in(&temp, &words);
    let lifted: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("lifted/module.json")).expect("read lifted module"),
    )
    .expect("parse lifted module");
    let entry = lifted["functions"]
        .as_array()
        .expect("functions")
        .iter()
        .find(|function| function["name"] == "entry")
        .expect("entry function");
    let switch = entry["blocks"]
        .as_array()
        .expect("blocks")
        .iter()
        .find(|block| block["terminator"]["op"] == "switch")
        .expect("switch recovered");
    let cases: Vec<u64> = switch["terminator"]["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| case[0].as_u64().unwrap_or(0))
        .collect();
    assert_eq!(cases, [0x18, 0x1C, 0x20]);
}

#[test]
fn homebrew_lift_finds_a_table_base_past_an_epilogue() {
    // adrp x27, 0 ; add x27, x27, #0x40 ; b 0x14
    // ldp x28, x27, [sp, #0x40] ; ret (an exit path)
    // 0x14: cmp w8, #2 ; b.hi 0x34 ; ldrsw x8, [x27, x8, lsl #2]
    // add x8, x8, x27 ; br x8 ; 0x28-0x30: three cases ; 0x34: default
    let mut words = vec![
        0x9000_001B,
        0x9101_037B,
        0x1400_0003,
        0xA944_6FFC,
        ret_x(30),
        0x7100_091F,
        0x5400_00E8,
        0xB8A8_7B68,
        0x8B1B_0108,
        0xD61F_0100,
    ];
    words.extend([ret_x(30); 4]);
    let switch = lift_table_switch(&words).expect("switch recovered");
    assert_eq!(switch["cases"].as_array().expect("cases").len(), 3);
}

#[test]
fn homebrew_lift_tries_the_table_addresses_a_function_forms() {
    // adrp x25, 0 ; add x25, x25, #0x40 ; b 0x14 ; mov x25, x16 (another
    // path, the latest write before the table code) ; nop
    // 0x14: cmp w8, #2 ; b.hi 0x34 ; ldrsw x8, [x25, x8, lsl #2]
    // add x8, x8, x25 ; br x8 ; 0x28-0x30: three cases ; 0x34: default
    let mut words = vec![
        0x9000_0019,
        0x9101_0339,
        0x1400_0003,
        0xAA10_03F9,
        0xD503_201F,
        0x7100_091F,
        0x5400_00E8,
        0xB8A8_7B28,
        0x8B19_0108,
        0xD61F_0100,
    ];
    words.extend([ret_x(30); 4]);
    let switch = lift_table_switch(&words).expect("switch recovered");
    assert_eq!(switch["cases"].as_array().expect("cases").len(), 3);
}

#[test]
fn homebrew_lift_reads_xzr_and_the_low_word_in_cbz() {
    // cbz xzr, #8: always taken (register 31 is XZR, not SP).
    let block = lift_single_block(&[0xB400_005F, ret_x(30), ret_x(30)]);
    assert_eq!(block["terminator"]["op"], "br");
    assert_eq!(block["terminator"]["target"], "bb_0000000000000008");

    // cbz w8, #8: only the low 32 bits of x8 are tested, and the flags are
    // left alone (`subs; stlxr; cbnz retry; b.ne` relies on it).
    let block = lift_single_block(&[0x3400_0048, ret_x(30), ret_x(30)]);
    let term = &block["terminator"];
    assert_eq!(term["op"], "br_test");
    assert_eq!(term["reg"], "x8");
    assert_eq!(term["mask"], 0xFFFF_FFFFu64);
    assert_eq!(term["zero"], true);
    assert_eq!(term["then"], "bb_0000000000000008");
    assert!(!block["ops"]
        .as_array()
        .expect("ops")
        .iter()
        .any(|op| op["op"] == "cmp_i64"));

    // tbnz x3, #40, #8: one bit, branching when it is set.
    let block = lift_single_block(&[0xB740_0043, ret_x(30), ret_x(30)]);
    let term = &block["terminator"];
    assert_eq!(term["op"], "br_test");
    assert_eq!(term["mask"], 1u64 << 40);
    assert_eq!(term["zero"], false);
}

#[test]
fn homebrew_lift_seeds_address_taken_functions() {
    // adr x0, #8 ; ret ; (0x8) ret: the function at 0x8 is only referenced
    // by its address.
    let temp = tempfile::tempdir().expect("tempdir");
    lift_single_block_in(&temp, &[0x1000_0040, ret_x(30), ret_x(30)]);
    let lifted: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("lifted/module.json")).expect("read lifted module"),
    )
    .expect("parse lifted module");
    let names: Vec<&str> = lifted["functions"]
        .as_array()
        .expect("functions")
        .iter()
        .map(|function| function["name"].as_str().expect("name"))
        .collect();
    assert!(names.contains(&"fn_0000000000000008"), "{names:?}");
}

#[test]
fn homebrew_lift_puts_the_entry_block_first() {
    // 0x0: ret (shared code at a lower address) ; 0x4 (entry): b 0x0.
    let temp = tempfile::tempdir().expect("tempdir");
    let base_dir = temp.path();
    lift_single_block_in(&temp, &[ret_x(30), 0x17FF_FFFF]);
    let module_path = base_dir.join("module.json");
    let out_dir = base_dir.join("lifted_entry");
    lift_homebrew(LiftOptions {
        module_json_path: module_path,
        out_dir: out_dir.clone(),
        entry_address: Some(4),
        module_filter: None,
        entry_name: "entry".to_string(),
        mode: LiftMode::Decode,
        extra_entries: Vec::new(),
    })
    .expect("lift homebrew decode");
    let lifted: Value =
        serde_json::from_str(&fs::read_to_string(out_dir.join("module.json")).expect("read"))
            .expect("parse");
    let entry = lifted["functions"]
        .as_array()
        .expect("functions")
        .iter()
        .find(|function| function["name"] == "entry")
        .expect("entry function");
    assert_eq!(entry["blocks"][0]["start"], 4);
    assert_eq!(entry["blocks"][1]["start"], 0);
}
