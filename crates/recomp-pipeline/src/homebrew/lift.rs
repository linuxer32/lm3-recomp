use crate::homebrew::module::{ModuleJson, ModuleSegment, MODULE_SCHEMA_VERSION};
use crate::input::{
    Block, Function, Module, ModuleSegment as RuntimeSegment, ModuleSegmentPermissions, Op,
    Terminator,
};
use crate::provenance::{
    CollectionInfo, InputFormatHint, InputRecord, ProvenanceManifest, TitleInfo, ToolInfo,
    SCHEMA_VERSION,
};
use serde_json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const MAX_BLOCK_INSTRUCTIONS: usize = 10_000;
const MAX_FUNCTION_INSTRUCTIONS: usize = 200_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiftMode {
    Stub,
    Decode,
}

#[derive(Debug)]
pub struct LiftOptions {
    pub module_json_path: PathBuf,
    pub out_dir: PathBuf,
    pub entry_name: String,
    /// Override the entry address (memory-relative); defaults to the start of
    /// .text. NSO images put the MOD0 header at .text start, so static boots
    /// typically need an address past the header.
    pub entry_address: Option<u64>,
    /// Lift only segments belonging to this module name (e.g. the NSO boot
    /// module) when the intake module.json contains placeholder modules.
    pub module_filter: Option<String>,
    /// Additional function entry points (absolute addresses) for code
    /// reached only through computed pointers, such as callbacks rtld forms
    /// with PC-relative arithmetic.
    pub extra_entries: Vec<u64>,
    pub mode: LiftMode,
}

#[derive(Debug)]
pub struct LiftReport {
    pub module_json_path: PathBuf,
    pub provenance_path: PathBuf,
    pub functions_emitted: usize,
    pub warnings: Vec<String>,
}

pub fn lift_homebrew(options: LiftOptions) -> Result<LiftReport, String> {
    if options.entry_name.trim().is_empty() {
        return Err("entry name must be non-empty".to_string());
    }

    let module_json_path = absolute_path(&options.module_json_path)?;
    let out_dir = absolute_path(&options.out_dir)?;

    let module_src = fs::read_to_string(&module_json_path).map_err(|err| err.to_string())?;
    let module_json: ModuleJson =
        serde_json::from_str(&module_src).map_err(|err| err.to_string())?;

    validate_homebrew_module(&module_json)?;

    let base_dir = module_json_path
        .parent()
        .ok_or_else(|| "homebrew module.json has no parent directory".to_string())?;

    match options.mode {
        LiftMode::Stub => lift_stub(&options, &module_json, base_dir, &out_dir),
        LiftMode::Decode => lift_decode(&options, &module_json, base_dir, &out_dir),
    }
}

fn validate_homebrew_module(module_json: &ModuleJson) -> Result<(), String> {
    if module_json.schema_version != MODULE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported homebrew module schema version: {}",
            module_json.schema_version
        ));
    }
    if module_json.module_type != "homebrew" {
        return Err(format!(
            "unsupported module type for homebrew lifter: {}",
            module_json.module_type
        ));
    }
    if module_json.modules.is_empty() {
        return Err("homebrew module list is empty".to_string());
    }
    Ok(())
}

fn lift_stub(
    options: &LiftOptions,
    module_json: &ModuleJson,
    base_dir: &Path,
    out_dir: &Path,
) -> Result<LiftReport, String> {
    let mut warnings = Vec::new();

    if module_json.modules.len() > 1 {
        warnings.push(format!(
            "homebrew lifter emitted a stub entry for {} modules without decoding instructions",
            module_json.modules.len()
        ));
    } else {
        warnings
            .push("homebrew lifter emitted a stub entry without decoding instructions".to_string());
    }

    let segments = collect_segments(module_json, base_dir, options.module_filter.as_deref())?;
    if segments.is_empty() {
        warnings.push("homebrew module contains no segments".to_string());
    }

    ensure_dir(out_dir)?;

    let lifted = Module {
        arch: "aarch64".to_string(),
        segments: Vec::new(),
        entry_address: options.entry_address,
        functions: vec![Function {
            name: options.entry_name.clone(),
            ops: vec![Op::Ret],
            blocks: Vec::new(),
        }],
    };

    let output_path = out_dir.join("module.json");
    let output_json = serde_json::to_string_pretty(&lifted).map_err(|err| err.to_string())?;
    fs::write(&output_path, output_json).map_err(|err| err.to_string())?;

    let provenance_path =
        emit_lift_provenance(out_dir, &output_path, &options.entry_name, module_json)?;

    Ok(LiftReport {
        module_json_path: output_path,
        provenance_path,
        functions_emitted: 1,
        warnings,
    })
}

fn lift_decode(
    options: &LiftOptions,
    module_json: &ModuleJson,
    base_dir: &Path,
    out_dir: &Path,
) -> Result<LiftReport, String> {
    let mut warnings = Vec::new();
    if module_json.modules.len() > 1 {
        warnings.push(format!(
            "homebrew lifter linking {} modules into one address space",
            module_json.modules.len()
        ));
    }

    let segments = collect_segments(module_json, base_dir, options.module_filter.as_deref())?;
    let texts = load_text_segments(&segments, &mut warnings)?;
    let image = ImageBytes::load(&segments);
    let entry_addr = options
        .entry_address
        .or_else(|| texts.first().map(|text| text.base_addr))
        .ok_or_else(|| "no executable text segment found in homebrew module".to_string())?;

    let mut functions = Vec::new();
    let mut pending = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut name_table = FunctionNames::new();
    name_table.seed_entry(entry_addr, options.entry_name.clone());

    pending.insert(entry_addr);
    pending.extend(options.extra_entries.iter().copied());

    // Seed every `bl` target so the call graph is lifted even when individual
    // functions truncate on unsupported encodings, and every text address
    // taken PC-relatively.
    let mut bl_targets = 0usize;
    let mut address_taken = 0usize;
    for text in &texts {
        for offset in (0..text.bytes.len()).step_by(4) {
            let word =
                u32::from_le_bytes(text.bytes[offset..offset + 4].try_into().unwrap_or([0; 4]));
            let pc = text.base_addr + offset as u64;
            if let Some(target) = pc_relative_target(text, pc, word) {
                if target.is_multiple_of(4)
                    && texts.iter().any(|text| text.contains(target))
                    && pending.insert(target)
                {
                    address_taken += 1;
                }
            }
            if word & 0xFC000000 == 0x94000000 {
                let imm26 = (word & 0x03FFFFFF) as i64;
                let target = (text.base_addr + offset as u64) as i64 + sign_extend(imm26 << 2, 28);
                let target = target as u64;
                if texts.iter().any(|text| text.contains(target)) && !seen.contains(&target) {
                    pending.insert(target);
                    bl_targets += 1;
                }
            }
        }
    }
    if bl_targets > 0 {
        warnings.push(format!(
            "homebrew lifter seeded {bl_targets} call targets from .text scan"
        ));
    }
    if address_taken > 0 {
        warnings.push(format!(
            "homebrew lifter seeded {address_taken} address-taken functions from adr/adrp"
        ));
    }

    // Retail Switch modules keep function pointers (vtables, jump tables) as
    // relocated addresses in read-only/data segments. Intake records the ones
    // that resolve into executable memory; seed exactly those.
    let mut pointer_targets = 0usize;
    let has_code_pointers = module_json
        .modules
        .iter()
        .any(|module| !module.code_pointers.is_empty());
    if has_code_pointers {
        for module in &module_json.modules {
            for address in &module.code_pointers {
                if texts.iter().any(|text| text.contains(*address))
                    && !seen.contains(address)
                    && pending.insert(*address)
                {
                    pointer_targets += 1;
                }
            }
        }
    } else {
        // Legacy module.json inputs without relocation metadata: fall back to
        // scanning non-executable segments for in-text values.
        for segment in &segments {
            if segment.permissions.contains('x') {
                continue;
            }
            let Ok(bytes) = fs::read(&segment.path) else {
                continue;
            };
            for chunk in bytes.as_chunks::<8>().0 {
                let value = u64::from_le_bytes(*chunk);
                if value >= 0x1000
                    && value % 4 == 0
                    && texts.iter().any(|text| text.contains(value))
                    && !seen.contains(&value)
                    && pending.insert(value)
                {
                    pointer_targets += 1;
                }
            }
        }
    }
    if pointer_targets > 0 {
        warnings.push(format!(
            "homebrew lifter seeded {pointer_targets} code pointers from relocation metadata"
        ));
    }

    // Known function starts: each function is assumed to end at the next
    // one (bounds the targets of jump tables whose size is not found).
    let function_starts: std::sync::Arc<BTreeSet<u64>> =
        std::sync::Arc::new(pending.iter().chain(seen.iter()).copied().collect());
    let mut decode_failures = 0usize;
    while let Some(func_addr) = pop_first(&mut pending) {
        let Some(text) = texts.iter().find(|text| text.contains(func_addr)) else {
            warnings.push(format!(
                "function entry 0x{func_addr:x} is outside all text segments"
            ));
            continue;
        };
        if seen.contains(&func_addr) {
            continue;
        }
        seen.insert(func_addr);

        let func_name = name_table.name_for(func_addr);
        let limit = function_starts
            .range(func_addr + 1..)
            .next()
            .copied()
            .unwrap_or(u64::MAX);
        let mut state = DecodeState::new(func_addr, limit, function_starts.clone());
        let (blocks, call_targets) = match decode_function(
            text,
            &image,
            func_addr,
            &mut state,
            &mut warnings,
            &mut name_table,
        ) {
            Ok(result) => result,
            Err(err) => {
                decode_failures += 1;
                if decode_failures <= 16 {
                    warnings.push(format!("failed to decode 0x{func_addr:x}: {err}"));
                }
                continue;
            }
        };

        for target in call_targets {
            if !texts.iter().any(|text| text.contains(target)) {
                warnings.push(format!("call target 0x{target:x} is outside text segments"));
                continue;
            }
            pending.insert(target);
        }

        functions.push(Function {
            name: func_name,
            ops: Vec::new(),
            blocks,
        });
    }

    if decode_failures > 16 {
        warnings.push(format!(
            "{decode_failures} functions failed to decode (first 16 reported)"
        ));
    }
    if functions.is_empty() {
        warnings.push("decoded zero functions from text segment".to_string());
    }

    ensure_dir(out_dir)?;

    let lifted_segments = emit_segment_copies(out_dir, &segments)?;
    let lifted = Module {
        arch: "aarch64".to_string(),
        segments: lifted_segments,
        entry_address: Some(entry_addr),
        functions,
    };

    let output_path = out_dir.join("module.json");
    let output_json = serde_json::to_string_pretty(&lifted).map_err(|err| err.to_string())?;
    fs::write(&output_path, output_json).map_err(|err| err.to_string())?;

    let provenance_path =
        emit_lift_provenance(out_dir, &output_path, &options.entry_name, module_json)?;

    Ok(LiftReport {
        module_json_path: output_path,
        provenance_path,
        functions_emitted: lifted.functions.len(),
        warnings,
    })
}

fn load_text_segments(
    segments: &[SegmentInfo],
    warnings: &mut Vec<String>,
) -> Result<Vec<TextSegment>, String> {
    let candidates = segments
        .iter()
        .filter(|segment| segment.name == "text" || segment.permissions.contains('x'));
    let mut texts = Vec::new();
    for segment in candidates {
        let bytes = fs::read(&segment.path).map_err(|err| err.to_string())?;
        if bytes.len() as u64 != segment.file_size {
            warnings.push(format!(
                "text segment size mismatch: manifest {} bytes, file {} bytes",
                segment.file_size,
                bytes.len()
            ));
        }
        if bytes.len() % 4 != 0 {
            return Err(format!(
                "text segment size is not 4-byte aligned: {} bytes",
                bytes.len()
            ));
        }
        texts.push(TextSegment {
            bytes,
            base_addr: segment.module_base.wrapping_add(segment.memory_offset),
        });
    }
    if texts.is_empty() {
        return Err("no executable text segment found in homebrew module".to_string());
    }
    Ok(texts)
}

fn emit_lift_provenance(
    out_dir: &Path,
    module_json_path: &Path,
    entry_name: &str,
    module_json: &ModuleJson,
) -> Result<PathBuf, String> {
    let source_note = format!(
        "lifted from {} ({} modules)",
        module_json.module_type,
        module_json.modules.len()
    );
    let lifted_src = fs::read_to_string(module_json_path)
        .map_err(|err| format!("read lifted module.json: {err}"))?;
    let manifest = ProvenanceManifest {
        schema_version: SCHEMA_VERSION.to_string(),
        title: TitleInfo {
            name: entry_name.to_string(),
            title_id: "0000000000000000".to_string(),
            version: "0.0.0".to_string(),
            region: "DEMO".to_string(),
        },
        collection: CollectionInfo {
            device: "homebrew".to_string(),
            tool: ToolInfo {
                name: "homebrew-lift".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            decryption_tool: None,
            notes: Some(source_note),
            collected_at: "generated".to_string(),
        },
        inputs: vec![InputRecord {
            path: PathBuf::from("module.json"),
            sha256: sha256_hex(lifted_src.as_bytes()),
            size: Some(lifted_src.len() as u64),
            format: Some(InputFormatHint::LiftedJson),
            role: Some("lifted_module".to_string()),
        }],
    };
    let toml_src = toml::to_string_pretty(&manifest)
        .map_err(|err| format!("serialize lift provenance: {err}"))?;
    let provenance_path = out_dir.join("provenance.toml");
    fs::write(&provenance_path, toml_src).map_err(|err| format!("write lift provenance: {err}"))?;
    Ok(provenance_path)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone)]
struct SegmentInfo {
    module_name: String,
    name: String,
    permissions: String,
    module_base: u64,
    memory_offset: u64,
    file_size: u64,
    path: PathBuf,
}

#[derive(Debug, Clone)]
struct TextSegment {
    bytes: Vec<u8>,
    base_addr: u64,
}

impl TextSegment {
    fn contains(&self, addr: u64) -> bool {
        let end = self.base_addr + self.bytes.len() as u64;
        addr >= self.base_addr && addr < end
    }

    fn read_word(&self, addr: u64) -> Result<u32, String> {
        if !self.contains(addr) {
            return Err(format!("address 0x{addr:x} outside text segment"));
        }
        let offset = addr - self.base_addr;
        if !offset.is_multiple_of(4) {
            return Err(format!("unaligned instruction address 0x{addr:x}"));
        }
        let idx = offset as usize;
        if idx + 4 > self.bytes.len() {
            return Err(format!("instruction read overrun at 0x{addr:x}"));
        }
        Ok(u32::from_le_bytes([
            self.bytes[idx],
            self.bytes[idx + 1],
            self.bytes[idx + 2],
            self.bytes[idx + 3],
        ]))
    }
}

/// Copy extracted segment images into the lifted module directory so the
/// pipeline can emit runtime memory initialization (relocated .data bytes,
/// embedded RomFS blobs, IPC buffers).
fn emit_segment_copies(
    out_dir: &Path,
    segments: &[SegmentInfo],
) -> Result<Vec<RuntimeSegment>, String> {
    let segments_dir = out_dir.join("segments");
    ensure_dir(&segments_dir)?;
    let mut out = Vec::new();
    for segment in segments {
        let file_name = format!("{}-{}.bin", segment.module_name, segment.name);
        let dest = segments_dir.join(&file_name);
        fs::copy(&segment.path, &dest)
            .map_err(|err| format!("copy segment {}: {err}", segment.path.display()))?;
        let permissions = ModuleSegmentPermissions {
            read: segment.permissions.contains('r'),
            write: segment.permissions.contains('w'),
            execute: segment.permissions.contains('x'),
        };
        out.push(RuntimeSegment {
            name: format!("{}-{}", segment.module_name, segment.name),
            base: segment.module_base.wrapping_add(segment.memory_offset),
            size: segment.file_size,
            permissions,
            init_path: Some(format!("segments/{file_name}")),
            init_size: Some(segment.file_size),
            zero_fill: false,
        });
    }
    Ok(out)
}

fn collect_segments(
    module_json: &ModuleJson,
    base_dir: &Path,
    filter: Option<&str>,
) -> Result<Vec<SegmentInfo>, String> {
    let mut segments = Vec::new();
    for module in &module_json.modules {
        if let Some(filter) = filter {
            if module.name != filter {
                continue;
            }
        }
        for segment in &module.segments {
            let segment_path = resolve_segment_path(base_dir, segment);
            if !segment_path.exists() {
                return Err(format!(
                    "segment file not found: {}",
                    segment_path.display()
                ));
            }
            segments.push(SegmentInfo {
                module_name: module.name.clone(),
                name: segment.name.clone(),
                permissions: segment.permissions.clone(),
                module_base: module.base,
                memory_offset: segment.memory_offset,
                file_size: segment.file_size,
                path: segment_path,
            });
        }
    }
    Ok(segments)
}

#[derive(Debug, Default)]
struct DecodeState {
    temp_counter: u32,
    /// The entry of the function being decoded.
    entry: u64,
    /// The next known function start after it.
    limit: u64,
    /// Every known function start (from the scan before decoding).
    starts: std::sync::Arc<BTreeSet<u64>>,
}

impl DecodeState {
    fn new(entry: u64, limit: u64, starts: std::sync::Arc<BTreeSet<u64>>) -> Self {
        Self {
            temp_counter: 0,
            entry,
            limit,
            starts,
        }
    }

    /// The code a jump table at `pc` may target: this function, and the
    /// span between the known function starts around `pc` (the function's
    /// code may run on past the next start).
    fn table_bounds(&self, pc: u64) -> [std::ops::Range<u64>; 2] {
        let below = self.starts.range(..=pc).next_back().copied().unwrap_or(0);
        let above = self
            .starts
            .range(pc + 1..)
            .next()
            .copied()
            .unwrap_or(u64::MAX);
        [self.entry..self.limit, below..above]
    }

    fn next_temp(&mut self, prefix: &str) -> String {
        let name = format!("{prefix}{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }
}

struct FunctionNames {
    names: BTreeMap<u64, String>,
}

impl FunctionNames {
    fn new() -> Self {
        Self {
            names: BTreeMap::new(),
        }
    }

    fn name_for(&mut self, addr: u64) -> String {
        if let Some(name) = self.names.get(&addr) {
            return name.clone();
        }
        let name = format!("fn_{addr:016x}");
        self.names.insert(addr, name.clone());
        name
    }

    fn seed_entry(&mut self, addr: u64, name: String) {
        self.names.insert(addr, name);
    }
}

/// Every segment of the module image, for reading data the lifter needs
/// (jump tables in .rodata).
#[derive(Debug, Default)]
struct ImageBytes {
    segments: Vec<(u64, Vec<u8>)>,
}

impl ImageBytes {
    fn load(segments: &[SegmentInfo]) -> Self {
        let segments = segments
            .iter()
            .filter_map(|segment| {
                let bytes = fs::read(&segment.path).ok()?;
                Some((
                    segment.module_base.wrapping_add(segment.memory_offset),
                    bytes,
                ))
            })
            .collect();
        Self { segments }
    }

    /// Little-endian read of `size` (1, 2, 4 or 8) bytes.
    fn read(&self, addr: u64, size: usize) -> Option<u64> {
        self.segments.iter().find_map(|(base, bytes)| {
            let offset = usize::try_from(addr.checked_sub(*base)?).ok()?;
            let slice = bytes.get(offset..offset.checked_add(size)?)?;
            let mut buf = [0u8; 8];
            buf[..size].copy_from_slice(slice);
            Some(u64::from_le_bytes(buf))
        })
    }
}

/// Instructions searched backwards from a `br` for its jump table.
const JUMP_TABLE_WINDOW: u64 = 32;
/// How far before its `br` a function's hoisted table address is looked for.
const MAX_HOISTED_TABLE_DISTANCE: u64 = 0x1_0000;
const MAX_JUMP_TABLE_ENTRIES: u64 = 4096;
/// Entries read from a table whose bounds check is not recognized.
const MAX_UNBOUNDED_TABLE_ENTRIES: u64 = 256;

/// General registers an instruction writes (approximate: enough to find the
/// latest definition of a register in straight-line code).
fn written_regs(word: u32) -> [Option<u8>; 2] {
    let rd = (word & 0x1F) as u8;
    let rn = ((word >> 5) & 0x1F) as u8;
    let rt2 = ((word >> 10) & 0x1F) as u8;
    // Branches and system instructions (bits 28:26 = 101).
    if (word >> 26) & 0x7 == 0b101 {
        if word & 0xFC00_0000 == 0x9400_0000 || word & 0xFFFF_FC1F == 0xD63F_0000 {
            return [Some(30), None];
        }
        if word & 0xFFF0_0000 == 0xD530_0000 {
            return [Some(rd), None];
        }
        return [None, None];
    }
    // Loads and stores (bits 27, 25 = 1, 0).
    if (word >> 25) & 0b101 == 0b100 {
        let is_pair = (word >> 27) & 0x7 == 0b101;
        let vector = (word >> 26) & 1 == 1;
        // Pairs carry L in bit 22; single-register integer forms load for
        // any opc but 00 (LDRSW is opc 10).
        let is_load = if is_pair || vector {
            (word >> 22) & 1 == 1
        } else {
            (word >> 22) & 0x3 != 0
        };
        let writeback = if is_pair {
            matches!((word >> 23) & 0x3, 1 | 3)
        } else {
            (word >> 24) & 0x3 == 0 && (word >> 21) & 1 == 0 && (word >> 10) & 1 == 1
        };
        let base = writeback.then_some(rn);
        if !is_load || vector {
            return [base, None];
        }
        return [Some(rd), if is_pair { Some(rt2) } else { base }];
    }
    // SIMD and floating-point data processing (bits 27:25 = 111) writes a
    // vector register, except conversions to integers (FCVT*S/U), FMOV to a
    // general register, and SMOV/UMOV. Taking `mov v8.16b, v3.16b` for a
    // write of x8 hid a jump table's `add x8` from the `br x8` after it.
    if (word >> 25) & 0x7 == 0b111 {
        let to_general = (word & 0x7F20_FC00 == 0x1E20_0000
            && matches!((word >> 16) & 0x7, 0 | 1 | 4 | 5 | 6))
            || word & 0xBFE0_FC00 == 0x0E00_2C00
            || word & 0xBFE0_FC00 == 0x0E00_3C00;
        return if to_general {
            [Some(rd), None]
        } else {
            [None, None]
        };
    }
    [Some(rd), None]
}

/// The statically known value a register holds at `pc` from ADRP/ADR, ADD
/// (immediate) and MOV (register) chains within the window, looking past
/// stack reloads of callee-saved registers (epilogues on other paths).
fn static_address(text: &TextSegment, reg: u8, pc: u64, floor: u64, depth: u32) -> Option<u64> {
    let (def_pc, word) = find_definition(text, reg, pc, floor)?;
    if depth > 4 {
        return None;
    }
    if word & 0x9F00_0000 == 0x9000_0000 {
        // ADRP
        let imm = ((((word >> 5) & 0x7FFFF) << 2) | ((word >> 29) & 0x3)) as i64;
        return Some((def_pc & !0xFFF).wrapping_add((sign_extend(imm, 21) << 12) as u64));
    }
    if word & 0x9F00_0000 == 0x1000_0000 {
        // ADR
        let imm = ((((word >> 5) & 0x7FFFF) << 2) | ((word >> 29) & 0x3)) as i64;
        return Some(def_pc.wrapping_add(sign_extend(imm, 21) as u64));
    }
    if word & 0xFF80_0000 == 0x9100_0000 {
        // ADD Xd, Xn, #imm{, lsl #12}
        let shift = if (word >> 22) & 1 == 1 { 12 } else { 0 };
        let imm = u64::from((word >> 10) & 0xFFF) << shift;
        let src = ((word >> 5) & 0x1F) as u8;
        return static_address(text, src, def_pc, floor, depth + 1).map(|base| base + imm);
    }
    if word & 0xFFE0_FFE0 == 0xAA00_03E0 {
        // MOV Xd, Xm (ORR Xd, XZR, Xm): a copy of the table address.
        let src = ((word >> 16) & 0x1F) as u8;
        return static_address(text, src, def_pc, floor, depth + 1);
    }
    // A callee-saved register reloaded from the stack is an epilogue on
    // another path (`ldp x28, x27, [sp, #0x40]` before a `ret`): look past it.
    let is_pair_load = matches!(word & 0xFFC0_0000, 0xA940_0000 | 0xA8C0_0000);
    let is_load = word & 0xFFC0_0000 == 0xF940_0000;
    if (is_pair_load || is_load) && (word >> 5) & 0x1F == 31 && reg >= 19 {
        return static_address(text, reg, def_pc, floor, depth + 1);
    }
    None
}

/// The latest instruction before `pc` (and after `floor`) that writes `reg`.
fn find_definition(text: &TextSegment, reg: u8, pc: u64, floor: u64) -> Option<(u64, u32)> {
    let mut at = pc;
    while at > floor {
        at -= 4;
        let word = text.read_word(at).ok()?;
        if written_regs(word).contains(&Some(reg)) {
            return Some((at, word));
        }
    }
    None
}

/// Recover the targets of a compiler jump table ending in `br x<reg>` at
/// `br_pc`:
///
/// ```text
/// cmp   x<idx>, #N                     ; N + 1 entries
/// adrp  x<tab>, table ; add x<tab>, x<tab>, :lo12:table
/// [adr  x<anchor>, anchor]
/// ldr   w/x<e>, [x<tab>, x<idx>, lsl #s]   ; ldrsw/ldr/ldrh/ldrb
/// add   x<reg>, x<anchor|tab>, x<e>{, lsl|sxtw #k}
/// br    x<reg>
/// ```
///
/// Returns `None` unless every piece is found and every target is an
/// aligned address inside `text`.
///
/// The table address may be formed far before the `br`: compilers hoist it
/// into a callee-saved register in the prologue of a function that
/// dispatches in a loop. So the table and base registers are also looked
/// for back to the function's entry (and further, in case that entry is a
/// label inside it). When no bounds check on the index is found, or the
/// one found does not fit the table, entries are read while their targets
/// stay inside the function (`function`: its entry up to the next known
/// function, and the span of known starts around the `br`). A wrong guess
/// cannot misdirect control: the generated switch matches the computed
/// target at run time and dispatches anything else.
fn resolve_jump_table(
    text: &TextSegment,
    image: &ImageBytes,
    br_pc: u64,
    reg: u8,
    function: [std::ops::Range<u64>; 2],
) -> Option<Vec<u64>> {
    let entry = function[0].start;
    let floor = br_pc
        .saturating_sub(4 * JUMP_TABLE_WINDOW)
        .max(text.base_addr);
    let function_floor = if entry < floor && br_pc - entry <= MAX_HOISTED_TABLE_DISTANCE {
        entry.max(text.base_addr)
    } else {
        floor
    };
    // A hoisted table address is also looked for before the entry seen for
    // this function, which may be a label inside it.
    let hoist_floor = br_pc
        .saturating_sub(MAX_HOISTED_TABLE_DISTANCE)
        .max(text.base_addr);
    let base_address = |reg: u8, pc: u64| {
        static_address(text, reg, pc, floor, 0)
            .or_else(|| static_address(text, reg, pc, function_floor, 0))
            .or_else(|| static_address(text, reg, pc, hoist_floor, 0))
    };
    let (add_pc, add) = find_definition(text, reg, br_pc, floor)?;
    // ADD (shifted register) 64-bit: LSL only; or ADD (extended register).
    let (lhs, rhs, shift, extend) = if add & 0xFF20_0000 == 0x8B00_0000 {
        if (add >> 22) & 0x3 != 0 {
            return None;
        }
        (
            ((add >> 5) & 0x1F) as u8,
            ((add >> 16) & 0x1F) as u8,
            (add >> 10) & 0x3F,
            None,
        )
    } else if add & 0xFFE0_0000 == 0x8B20_0000 {
        (
            ((add >> 5) & 0x1F) as u8,
            ((add >> 16) & 0x1F) as u8,
            (add >> 10) & 0x7,
            Some((add >> 13) & 0x7),
        )
    } else {
        return None;
    };
    if shift > 4 {
        return None;
    }
    // One operand is the loaded entry, the other a static address. The
    // shifted/extended operand (rhs) is usually the entry; rtld adds the
    // table base as rhs instead.
    for (entry_reg, base_reg, entry_shift) in [(rhs, lhs, shift), (lhs, rhs, 0)] {
        if extend.is_some() && entry_reg != rhs {
            continue;
        }
        let Some((load_pc, load)) = find_definition(text, entry_reg, add_pc, floor) else {
            continue;
        };
        let Some(mem) = decode_register_offset_load_store(load) else {
            continue;
        };
        if mem.kind == MemKind::Store || mem.kind == MemKind::Prefetch || mem.rt != entry_reg {
            continue;
        }
        let entry_bytes = 1u64 << mem.size;
        if mem.size > 0 && !mem.scaled {
            continue;
        }
        // The targets of the first `count` entries of the table at `table`
        // relative to `base`. `Bounded`: every entry must be a code
        // address; `Inside`: also inside the function; `Unbounded`: reading
        // stops at the first target outside the function.
        let read_targets = |table: u64, base: u64, count: u64, mode: TableRead| {
            let mut targets = Vec::new();
            for index in 0..count {
                let Some(raw) = image.read(table + index * entry_bytes, entry_bytes as usize)
                else {
                    break;
                };
                let bits = (entry_bytes * 8) as u32;
                let loaded = if mem.kind == MemKind::Load {
                    raw
                } else {
                    sign_extend(raw as i64, bits as u8) as u64
                };
                let value = match extend {
                    // SXTW / SXTX / UXTW / UXTX of the entry register.
                    Some(0b110) => sign_extend((loaded & 0xFFFF_FFFF) as i64, 32) as u64,
                    Some(0b010) => loaded & 0xFFFF_FFFF,
                    _ => loaded,
                };
                let target = base.wrapping_add(value << entry_shift);
                let valid = target.is_multiple_of(4) && text.contains(target);
                let inside = valid && function.iter().any(|range| range.contains(&target));
                match mode {
                    TableRead::Bounded if !valid => return None,
                    TableRead::Inside if !inside => return None,
                    // Past the end of a table whose size is unknown.
                    TableRead::Unbounded if !inside => break,
                    _ => {}
                }
                if !targets.contains(&target) {
                    targets.push(target);
                }
            }
            (mode != TableRead::Unbounded || targets.len() >= 2).then_some(targets)
        };
        let count = jump_table_count(text, mem.rm, load_pc, floor, function_floor)
            .filter(|count| *count <= MAX_JUMP_TABLE_ENTRIES);
        if let (Some(table), Some(base)) = (
            base_address(mem.base, load_pc),
            base_address(base_reg, add_pc),
        ) {
            if let Some(targets) =
                count.and_then(|count| read_targets(table, base, count, TableRead::Bounded))
            {
                return Some(targets);
            }
            // No bounds check was recognized, or the one found does not
            // fit this table.
            return read_targets(
                table,
                base,
                MAX_UNBOUNDED_TABLE_ENTRIES,
                TableRead::Unbounded,
            );
        }
        // The table address reaches the br along several paths (a state
        // machine passing it between registers): take every table address
        // the function forms in the register whose entries all land in the
        // function, and switch over all their targets.
        if mem.base != base_reg {
            return None;
        }
        let mut targets: Vec<u64> = Vec::new();
        for table in formed_addresses(text, mem.base, &function) {
            let found = match count {
                Some(count) => read_targets(table, table, count, TableRead::Inside),
                None => read_targets(
                    table,
                    table,
                    MAX_UNBOUNDED_TABLE_ENTRIES,
                    TableRead::Unbounded,
                ),
            };
            for target in found.into_iter().flatten() {
                if !targets.contains(&target) {
                    targets.push(target);
                }
            }
        }
        return (!targets.is_empty()).then_some(targets);
    }
    None
}

/// How the entries of a jump table are checked as they are read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TableRead {
    Bounded,
    Inside,
    Unbounded,
}

/// The addresses `adrp x<reg>; add x<reg>, x<reg>, #lo12` pairs form in
/// the code of `ranges` (at most 256 KiB of each is scanned).
fn formed_addresses(text: &TextSegment, reg: u8, ranges: &[std::ops::Range<u64>]) -> Vec<u64> {
    let mut addresses = Vec::new();
    for range in ranges {
        let mut pc = range.start.max(text.base_addr);
        let end = range.end.min(range.start.saturating_add(0x4_0000));
        while pc < end {
            let Ok(word) = text.read_word(pc) else {
                break;
            };
            if word & 0x9F00_001F == 0x9000_0000 | u32::from(reg) {
                if let Some(address) = pc_relative_target(text, pc, word) {
                    if !addresses.contains(&address) {
                        addresses.push(address);
                    }
                }
            }
            pc += 4;
        }
    }
    addresses
}

/// Whether `br x<reg>` at `br_pc` jumps through a table: its target is an
/// ADD of an entry loaded with a register offset. Reported when the table
/// cannot be recovered, since such a branch then fails at run time.
fn looks_like_jump_table(text: &TextSegment, br_pc: u64, reg: u8) -> bool {
    let floor = br_pc
        .saturating_sub(4 * JUMP_TABLE_WINDOW)
        .max(text.base_addr);
    let Some((add_pc, add)) = find_definition(text, reg, br_pc, floor) else {
        return false;
    };
    if add & 0xFF20_0000 != 0x8B00_0000 && add & 0xFFE0_0000 != 0x8B20_0000 {
        return false;
    }
    [(add >> 5) & 0x1F, (add >> 16) & 0x1F]
        .iter()
        .any(|operand| {
            find_definition(text, *operand as u8, add_pc, floor)
                .and_then(|(_, load)| decode_register_offset_load_store(load))
                .is_some_and(|mem| mem.kind != MemKind::Store && mem.kind != MemKind::Prefetch)
        })
}

/// How far a branch-to-table guard looks for the branch into the table
/// code when that code is a block of its own.
const JUMP_TABLE_GUARD_RANGE: u64 = 0x4000;

/// `N` if `word` is `cmp <reg>, #N` (SUBS immediate to XZR/WZR, no shift).
fn compare_immediate(word: u32, reg: u8) -> Option<u64> {
    (word & 0x7FC0_001F == 0x7100_001F && ((word >> 5) & 0x1F) as u8 == reg)
        .then(|| u64::from((word >> 10) & 0xFFF))
}

/// The condition and target of a `b.cond` at `pc`.
fn branch_condition(word: u32, pc: u64) -> Option<(u32, u64)> {
    (word & 0xFF00_0010 == 0x5400_0000).then(|| {
        let offset = sign_extend(i64::from((word >> 5) & 0x7FFFF) << 2, 21);
        (word & 0xF, pc.wrapping_add(offset as u64))
    })
}

/// Entries `cmp <idx>, #n` admits when `cond` is the in-range condition.
fn in_range_count(n: u64, cond: u32) -> Option<u64> {
    match cond {
        // LS, LE: idx <= n
        0b1001 | 0b1101 => Some(n + 1),
        // LO, LT: idx < n
        0b0011 | 0b1011 => Some(n),
        _ => None,
    }
}

/// B, BR and RET never fall through.
fn ends_flow(word: u32) -> bool {
    word & 0xFC00_0000 == 0x1400_0000
        || word & 0xFFFF_FC1F == 0xD61F_0000
        || word & 0xFFFF_FC1F == 0xD65F_0000
}

/// Entries admitted by the `cmp` at `at` whose following branch leaves
/// the range (`cmp; b.hi default`).
fn fallthrough_guard_count(text: &TextSegment, at: u64, n: u64) -> Option<u64> {
    let next = text.read_word(at + 4).ok()?;
    Some(match branch_condition(next, at + 4) {
        Some((cond, _)) => in_range_count(n, cond ^ 1).unwrap_or(n + 1),
        None => n + 1,
    })
}

/// A compare that may test a copy of a jump table's index: its register,
/// the entry count it admits, and the AND defining the register once found.
type GuardCandidate = (u8, u64, Option<(u8, u64)>);

/// How many entries a jump table indexed by `index_reg` has, from the
/// `cmp <idx>, #N` guarding it and the branch after that compare. Register
/// moves are followed (`cmp w20, #3; ...; mov w8, w20`), and so are masked
/// copies: the compare may test another AND of the same source with the
/// same mask (`and w10, w27, #0xff; and x9, x27, #0xff; cmp w10, #N`),
/// including an earlier value of the index register itself (`and w9, w22,
/// #0xff; cmp w9, #5; b.hi default; and x9, x22, #0xff`), as long as the
/// source is not written between the two ANDs.
/// Usually the guard falls through into the table code (`cmp; b.hi
/// default`); when that code is a block of its own, the guard is the
/// compare before the conditional branch into it (`cmp x8, #5; b.ls table`).
/// An index reloaded from a stack slot is followed to the store that
/// filled it, back to the function's entry (`entry_floor`).
fn jump_table_count(
    text: &TextSegment,
    index_reg: u8,
    pc: u64,
    floor: u64,
    entry_floor: u64,
) -> Option<u64> {
    let mut index_reg = index_reg;
    // Compares on other registers, with their counts, and the AND that
    // defines each of them once found.
    let mut candidates: Vec<GuardCandidate> = Vec::new();
    let mut index_mask: Option<(u8, u64)> = None;
    // The stack slot the index was reloaded from, while the store that
    // filled it is looked for (back to the function's entry).
    let mut slot: Option<u64> = None;
    let mut floor = floor;
    let mut at = pc;
    while at > floor {
        at -= 4;
        let word = text.read_word(at).ok()?;
        if index_mask.is_none() && slot.is_none() {
            if let Some(n) = compare_immediate(word, index_reg) {
                return fallthrough_guard_count(text, at, n);
            }
        }
        // A masked copy made before the source changed holds another value.
        if let Some((src, _)) = index_mask {
            if written_regs(word).contains(&Some(src)) {
                return None;
            }
        }
        // Any other `cmp <reg>, #N` might test a masked copy of the index
        // (once the index's AND is found, so might one on its register).
        if word & 0x7FC0_001F == 0x7100_001F {
            let reg = ((word >> 5) & 0x1F) as u8;
            if (reg != index_reg || index_mask.is_some() || slot.is_some())
                && !candidates.iter().any(|(r, _, _)| *r == reg)
            {
                let n = u64::from((word >> 10) & 0xFFF);
                if let Some(count) = fallthrough_guard_count(text, at, n) {
                    candidates.push((reg, count, None));
                }
            }
        }
        if let Some(offset) = slot {
            // The spill the index was reloaded from: carry on with the
            // register it stored, as for a move.
            if let Some(src) = stack_store_source(word, offset) {
                if let Some((_, count, _)) = candidates.iter().find(|(reg, _, _)| *reg == src) {
                    return Some(*count);
                }
                index_reg = src;
                slot = None;
                floor = at.saturating_sub(4 * JUMP_TABLE_WINDOW).max(entry_floor);
            } else {
                let written = written_regs(word);
                candidates.retain(|(reg, _, _)| !written.contains(&Some(*reg)));
            }
            continue;
        }
        if let Some(logic) = decode_logical_immediate(word).filter(|logic| logic.opc == 0) {
            let def = Some((logic.src, logic.imm));
            if logic.dst == index_reg && index_mask.is_none() {
                index_mask = def;
            } else if let Some(candidate) = candidates
                .iter_mut()
                .find(|(reg, _, found)| *reg == logic.dst && found.is_none())
            {
                candidate.2 = def;
            }
            if let Some((_, count, _)) = candidates
                .iter()
                .find(|(_, _, found)| index_mask.is_some() && *found == index_mask)
            {
                return Some(*count);
            }
            continue;
        }
        if index_mask.is_some() {
            // A compare only tests the AND found further back if nothing
            // else writes its register in between.
            let written = written_regs(word);
            candidates.retain(|(reg, _, found)| found.is_some() || !written.contains(&Some(*reg)));
            continue;
        }
        // The index reloaded from the stack (`ldr x8, [sp, #0x98]`, the
        // compare having tested the register that was spilled there).
        if let Some(offset) = stack_load_offset(word, index_reg) {
            slot = Some(offset);
            floor = floor.min(entry_floor);
            continue;
        }
        // MOV (register): ORR Rd, ZR, Rm.
        if word & 0x7FE0_FFE0 == 0x2A00_03E0 && (word & 0x1F) as u8 == index_reg {
            index_reg = ((word >> 16) & 0x1F) as u8;
            // A later compare on the source tests the index too
            // (`mov w8, w1; cmp w1, #7`).
            if let Some((_, count, _)) = candidates.iter().find(|(reg, _, _)| *reg == index_reg) {
                return Some(*count);
            }
        } else if ends_flow(word) {
            return guarded_branch_count(text, at + 4, index_reg);
        } else {
            // A compare is only an alias of the index if its register is
            // not overwritten before it.
            let written = written_regs(word);
            candidates.retain(|(reg, _, _)| !written.contains(&Some(*reg)));
        }
    }
    None
}

/// The `sp` offset `word` loads `reg` from (`ldr x/w<reg>, [sp, #off]`).
fn stack_load_offset(word: u32, reg: u8) -> Option<u64> {
    let scale = match word & 0xFFC0_0000 {
        0xF940_0000 => 3,
        0xB940_0000 => 2,
        _ => return None,
    };
    ((word >> 5) & 0x1F == 31 && (word & 0x1F) as u8 == reg)
        .then(|| u64::from((word >> 10) & 0xFFF) << scale)
}

/// The register `word` stores at `[sp, #offset]`: STR (X or W, unsigned
/// offset) or either register of an STP (signed offset).
fn stack_store_source(word: u32, offset: u64) -> Option<u8> {
    if (word >> 5) & 0x1F != 31 {
        return None;
    }
    let rt = (word & 0x1F) as u8;
    let scaled = |scale: u32| u64::from((word >> 10) & 0xFFF) << scale;
    match word & 0xFFC0_0000 {
        0xF900_0000 => (scaled(3) == offset).then_some(rt),
        0xB900_0000 => (scaled(2) == offset).then_some(rt),
        0xA900_0000 | 0x2900_0000 => {
            let scale = if word >> 31 == 1 { 3 } else { 2 };
            let first = sign_extend(i64::from((word >> 15) & 0x7F), 7) << scale;
            if first as u64 == offset {
                Some(rt)
            } else if (first + (1 << scale)) as u64 == offset {
                Some(((word >> 10) & 0x1F) as u8)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The count from a `cmp <idx>, #N; b.cond block` near `block` that
/// branches into it when the index is in range.
fn guarded_branch_count(text: &TextSegment, block: u64, index_reg: u8) -> Option<u64> {
    let start = block
        .saturating_sub(JUMP_TABLE_GUARD_RANGE)
        .max(text.base_addr + 4);
    let end = block + JUMP_TABLE_GUARD_RANGE;
    let mut pc = start;
    while pc < end {
        let Ok(word) = text.read_word(pc) else {
            break;
        };
        if let Some((cond, target)) = branch_condition(word, pc) {
            if target == block {
                let guard = text.read_word(pc - 4).ok()?;
                if let Some(n) = compare_immediate(guard, index_reg) {
                    if let Some(count) = in_range_count(n, cond) {
                        return Some(count);
                    }
                }
            }
        }
        pc += 4;
    }
    None
}

/// A text address formed PC-relatively at `pc`: `adr`, or `adrp` followed
/// within a few instructions by `add` (immediate) on the same register.
/// Position-independent code takes function addresses this way (callbacks,
/// thread entries), with no relocation to discover them by.
fn pc_relative_target(text: &TextSegment, pc: u64, word: u32) -> Option<u64> {
    let imm = || {
        sign_extend(
            ((((word >> 5) & 0x7FFFF) << 2) | ((word >> 29) & 0x3)) as i64,
            21,
        )
    };
    if word & 0x9F00_0000 == 0x1000_0000 {
        return Some(pc.wrapping_add(imm() as u64));
    }
    if word & 0x9F00_0000 != 0x9000_0000 {
        return None;
    }
    let page = (pc & !0xFFF).wrapping_add((imm() << 12) as u64);
    let rd = word & 0x1F;
    for step in 1..=4 {
        let next = text.read_word(pc + 4 * step).ok()?;
        if next & 0xFFC0_0000 == 0x9100_0000 && (next >> 5) & 0x1F == rd {
            return Some(page + u64::from((next >> 10) & 0xFFF));
        }
        if next & 0x1F == rd {
            return None;
        }
    }
    None
}

fn decode_function(
    text: &TextSegment,
    image: &ImageBytes,
    entry: u64,
    state: &mut DecodeState,
    warnings: &mut Vec<String>,
    names: &mut FunctionNames,
) -> Result<(Vec<Block>, Vec<u64>), String> {
    let mut blocks = BTreeMap::new();
    let mut pending = BTreeSet::new();
    let mut call_targets = BTreeSet::new();
    let mut decoded_count = 0_usize;

    pending.insert(entry);

    while let Some(addr) = pop_first(&mut pending) {
        if blocks.contains_key(&addr) {
            continue;
        }

        let (block, block_targets, block_calls, block_insts) =
            decode_block(text, image, addr, state, warnings, names)?;
        decoded_count += block_insts;
        if decoded_count > MAX_FUNCTION_INSTRUCTIONS {
            return Err(format!(
                "function decode limit exceeded ({} instructions)",
                MAX_FUNCTION_INSTRUCTIONS
            ));
        }

        for target in block_targets {
            pending.insert(target);
        }
        for target in block_calls {
            call_targets.insert(target);
        }
        blocks.insert(addr, block);
    }

    // Execution starts at the first block, which must be the entry even when
    // the function also branches to lower addresses (a backward tail branch
    // into shared code, cold blocks placed before the entry).
    let mut ordered = Vec::with_capacity(blocks.len());
    ordered.extend(blocks.remove(&entry));
    ordered.extend(blocks.into_values());
    Ok((ordered, call_targets.into_iter().collect()))
}

fn decode_block(
    text: &TextSegment,
    image: &ImageBytes,
    start: u64,
    state: &mut DecodeState,
    warnings: &mut Vec<String>,
    names: &mut FunctionNames,
) -> Result<(Block, Vec<u64>, Vec<u64>, usize), String> {
    if !start.is_multiple_of(4) {
        return Err(format!("unaligned block start 0x{start:x}"));
    }
    if !text.contains(start) {
        return Err(format!("block start 0x{start:x} outside text segment"));
    }

    let mut ops = Vec::new();
    let mut pc = start;
    let mut inst_count = 0_usize;

    loop {
        if inst_count >= MAX_BLOCK_INSTRUCTIONS {
            // Split very long straight-line code (large static initializers)
            // into chained blocks rather than dropping the function.
            let (terminator, block_targets, call_targets) =
                lower_terminator(TermInfo::Br { target: pc }, names);
            let block = Block {
                label: block_label(start),
                start,
                ops,
                terminator,
            };
            return Ok((block, block_targets, call_targets, inst_count));
        }
        let word = text.read_word(pc)?;
        let decoded = decode_one(word, pc, state, &mut ops, warnings)?;
        inst_count += 1;

        match decoded {
            DecodedOutcome::Continue => {
                pc = pc.wrapping_add(4);
                continue;
            }
            DecodedOutcome::Terminate(term) => {
                let term = match term {
                    TermInfo::BrIndirect { reg } => {
                        match resolve_jump_table(text, image, pc, reg, state.table_bounds(pc)) {
                            Some(targets) => TermInfo::Switch { reg, targets },
                            None => {
                                if looks_like_jump_table(text, pc, reg) {
                                    warnings.push(format!(
                                        "unresolved jump table: br x{reg} at 0x{pc:x}"
                                    ));
                                }
                                TermInfo::BrIndirect { reg }
                            }
                        }
                    }
                    other => other,
                };
                let (terminator, block_targets, call_targets) = lower_terminator(term, names);
                let block = Block {
                    label: block_label(start),
                    start,
                    ops,
                    terminator,
                };
                return Ok((block, block_targets, call_targets, inst_count));
            }
        }
    }
}

fn lower_terminator(term: TermInfo, names: &mut FunctionNames) -> (Terminator, Vec<u64>, Vec<u64>) {
    match term {
        TermInfo::Br { target } => (
            Terminator::Br {
                target: block_label(target),
            },
            vec![target],
            Vec::new(),
        ),
        TermInfo::BrCond {
            cond,
            then_target,
            else_target,
        } => (
            Terminator::BrCond {
                cond,
                then: block_label(then_target),
                else_target: block_label(else_target),
            },
            vec![then_target, else_target],
            Vec::new(),
        ),
        TermInfo::BrTest {
            reg,
            mask,
            zero,
            then_target,
            else_target,
        } => (
            Terminator::BrTest {
                reg: reg_name(reg),
                mask,
                zero,
                then: block_label(then_target),
                else_target: block_label(else_target),
            },
            vec![then_target, else_target],
            Vec::new(),
        ),
        TermInfo::Call { target, next } => {
            let name = names.name_for(target);
            (
                Terminator::Call {
                    target: name,
                    next: block_label(next),
                    return_address: Some(next),
                },
                vec![next],
                vec![target],
            )
        }
        TermInfo::BrIndirect { reg } => (
            Terminator::BrIndirect { reg: reg_name(reg) },
            Vec::new(),
            Vec::new(),
        ),
        TermInfo::Switch { reg, targets } => (
            Terminator::Switch {
                reg: reg_name(reg),
                cases: targets
                    .iter()
                    .map(|target| (*target, block_label(*target)))
                    .collect(),
            },
            targets,
            Vec::new(),
        ),
        TermInfo::CallIndirect { reg, next } => (
            Terminator::CallIndirect {
                reg,
                next: block_label(next),
                return_address: Some(next),
            },
            vec![next],
            Vec::new(),
        ),
        TermInfo::Trap { pc, word } => (Terminator::Trap { pc, word }, Vec::new(), Vec::new()),
        TermInfo::Ret => (Terminator::Ret, Vec::new(), Vec::new()),
    }
}

#[derive(Debug)]
enum DecodedOutcome {
    Continue,
    Terminate(TermInfo),
}

#[derive(Debug)]
enum TermInfo {
    Br {
        target: u64,
    },
    BrCond {
        cond: String,
        then_target: u64,
        else_target: u64,
    },
    /// To `then_target` when `reg & mask` is zero (`zero`) or non-zero.
    BrTest {
        reg: u8,
        mask: u64,
        zero: bool,
        then_target: u64,
        else_target: u64,
    },
    Call {
        target: u64,
        next: u64,
    },
    BrIndirect {
        reg: u8,
    },
    /// A recovered jump table: `br reg` to one of `targets`.
    Switch {
        reg: u8,
        targets: Vec<u64>,
    },
    CallIndirect {
        /// Register or temp holding the target.
        reg: String,
        next: u64,
    },
    Trap {
        pc: u64,
        word: u32,
    },
    Ret,
}

fn decode_one(
    word: u32,
    pc: u64,
    state: &mut DecodeState,
    ops: &mut Vec<Op>,
    warnings: &mut Vec<String>,
) -> Result<DecodedOutcome, String> {
    if let Some(barrier) = decode_barrier(word) {
        ops.push(barrier);
        return Ok(DecodedOutcome::Continue);
    }
    if is_hint_or_barrier(word) || is_cache_maintenance(word) {
        return Ok(DecodedOutcome::Continue);
    }
    if let Some(imm) = svc_imm(word) {
        let name = recomp_services::kernel_svc_name(imm).unwrap_or("svcBreak");
        ops.push(Op::Syscall {
            name: name.to_string(),
            args: vec![
                "x0".to_string(),
                "x1".to_string(),
                "x2".to_string(),
                "x3".to_string(),
                "x4".to_string(),
                "x5".to_string(),
                "x6".to_string(),
                "x7".to_string(),
            ],
        });
        return Ok(DecodedOutcome::Continue);
    }
    if is_ret(word) {
        return Ok(DecodedOutcome::Terminate(TermInfo::Ret));
    }
    if let Some(reg) = decode_br_register(word) {
        return Ok(DecodedOutcome::Terminate(TermInfo::BrIndirect { reg }));
    }
    if let Some(target) = decode_branch_imm(word, pc) {
        return Ok(DecodedOutcome::Terminate(TermInfo::Br { target }));
    }
    if let Some(target) = decode_branch_link(word, pc) {
        set_link_register(pc, ops);
        return Ok(DecodedOutcome::Terminate(TermInfo::Call {
            target,
            next: pc.wrapping_add(4),
        }));
    }
    if let Some((cond, target)) = decode_branch_cond(word, pc)? {
        return Ok(DecodedOutcome::Terminate(TermInfo::BrCond {
            cond,
            then_target: target,
            else_target: pc.wrapping_add(4),
        }));
    }

    // CBZ/CBNZ and TBZ/TBNZ test a register without touching NZCV (code
    // branches on flags set before them: `subs; stlxr; cbnz retry; b.ne`).
    // Register 31 is XZR here (`cbz xzr` is an unconditional branch).
    let test = decode_cbz(word, pc)?
        .map(|(is_nz, reg, is_32, target)| {
            let mask = if is_32 { 0xFFFF_FFFF } else { u64::MAX };
            (is_nz, reg, mask, target)
        })
        .or(decode_tbz(word, pc)?
            .map(|(is_nz, reg, bit, target)| (is_nz, reg, 1u64 << bit, target)));
    if let Some((is_nz, reg, mask, target)) = test {
        let fallthrough = pc.wrapping_add(4);
        if reg == 31 {
            let target = if is_nz { fallthrough } else { target };
            return Ok(DecodedOutcome::Terminate(TermInfo::Br { target }));
        }
        return Ok(DecodedOutcome::Terminate(TermInfo::BrTest {
            reg,
            mask,
            zero: !is_nz,
            then_target: target,
            else_target: fallthrough,
        }));
    }

    if let Some(decoded) = decode_sysreg(word) {
        emit_sysreg(decoded, pc, state, ops, warnings);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_mov_wide(word)? {
        emit_mov_wide(decoded, pc, state, ops)?;
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_adr(word, pc)? {
        ops.push(decoded);
        return Ok(DecodedOutcome::Continue);
    }

    if emit_load_literal(word, pc, state, ops) {
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_add_carry(word) {
        emit_add_carry(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_add_sub(word) {
        emit_add_sub(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_logical_immediate(word) {
        emit_logical_immediate(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_logical_register(word) {
        emit_logical_register(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_load_store(word) {
        emit_load_store(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_unscaled_load_store(word) {
        emit_unscaled_load_store(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_register_offset_load_store(word) {
        emit_register_offset_load_store(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_pair_load_store(word) {
        emit_pair_load_store(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_simd_load_store(word) {
        emit_simd_load_store(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_simd_pair(word) {
        emit_simd_pair(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_simd_structure(word) {
        emit_simd_structure(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_simd_move(word) {
        emit_simd_move(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(operands) = recomp_simd::operands(word) {
        ops.push(simd_op(word, &operands));
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_multiply(word) {
        emit_multiply(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_extract(word) {
        emit_extract(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_cond_select(word) {
        emit_cond_select(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_cond_compare(word) {
        emit_cond_compare(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_two_source(word) {
        emit_two_source(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_one_source(word) {
        emit_one_source(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_exclusive(word) {
        emit_exclusive(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if let Some(decoded) = decode_bitfield(word) {
        emit_bitfield(decoded, state, ops);
        return Ok(DecodedOutcome::Continue);
    }

    if is_blr(word) {
        // Read the target before LR is overwritten (`blr x30`).
        let target = state.next_temp("target");
        ops.push(Op::MovI64 {
            dst: target.clone(),
            src: reg_name(((word >> 5) & 0x1F) as u8),
        });
        set_link_register(pc, ops);
        return Ok(DecodedOutcome::Terminate(TermInfo::CallIndirect {
            reg: target,
            next: pc.wrapping_add(4),
        }));
    }

    warnings.push(format!(
        "unsupported instruction 0x{word:08x} at 0x{pc:016x}; block ends in a trap"
    ));
    Ok(DecodedOutcome::Terminate(TermInfo::Trap { pc, word }))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SysReg {
    TpidrEl0,
    TpidrroEl0,
    CntpctEl0,
    CntfrqEl0,
    CtrEl0,
    Fpcr,
    Fpsr,
    Nzcv,
    Unknown,
}

impl SysReg {
    fn name(self) -> &'static str {
        match self {
            SysReg::TpidrEl0 => "tpidr_el0",
            SysReg::TpidrroEl0 => "tpidrro_el0",
            SysReg::CntpctEl0 => "cntpct_el0",
            SysReg::CntfrqEl0 => "cntfrq_el0",
            SysReg::CtrEl0 => "ctr_el0",
            SysReg::Fpcr => "fpcr",
            SysReg::Fpsr => "fpsr",
            SysReg::Nzcv => "nzcv",
            SysReg::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SysRegOp {
    is_read: bool,
    reg: SysReg,
    rt: u8,
    /// op0=op1=CRn=CRm=op2 encoded for diagnostics.
    encoding: u32,
}

fn decode_sysreg(word: u32) -> Option<SysRegOp> {
    let top = word & 0xFFF0_0000;
    let is_read = match top {
        0xD530_0000 => true,
        0xD510_0000 => false,
        _ => return None,
    };
    let op0 = (word >> 19) & 0x3;
    let op1 = (word >> 16) & 0x7;
    let crn = (word >> 12) & 0xF;
    let crm = (word >> 8) & 0xF;
    let op2 = (word >> 5) & 0x7;
    let rt = (word & 0x1F) as u8;
    let reg = match (op0, op1, crn, crm, op2) {
        (3, 3, 13, 0, 2) => SysReg::TpidrEl0,
        (3, 3, 13, 0, 3) => SysReg::TpidrroEl0,
        (3, 3, 14, 0, 1) => SysReg::CntpctEl0,
        (3, 3, 14, 0, 0) => SysReg::CntfrqEl0,
        (3, 3, 0, 0, 1) => SysReg::CtrEl0,
        (3, 3, 4, 4, 0) => SysReg::Fpcr,
        (3, 3, 4, 4, 1) => SysReg::Fpsr,
        (3, 3, 4, 2, 0) => SysReg::Nzcv,
        _ => SysReg::Unknown,
    };
    Some(SysRegOp {
        is_read,
        reg,
        rt,
        encoding: (op0 << 11) | (op1 << 8) | (crn << 4) | (crm << 1) | op2,
    })
}

fn emit_sysreg(
    decoded: SysRegOp,
    pc: u64,
    state: &mut DecodeState,
    ops: &mut Vec<Op>,
    warnings: &mut Vec<String>,
) {
    // Rt=31 is XZR for MRS/MSR, not SP.
    let rt = if decoded.is_read {
        write_zr(decoded.rt, state)
    } else {
        read_zr(decoded.rt, state, ops)
    };
    if decoded.reg == SysReg::Unknown {
        warnings.push(format!(
            "unsupported system register access 0x{:04x} at 0x{pc:016x}; {}",
            decoded.encoding,
            if decoded.is_read {
                "read as zero"
            } else {
                "write ignored"
            }
        ));
    }
    if decoded.is_read {
        match decoded.reg {
            SysReg::Nzcv => ops.push(Op::ReadNzcv { dst: rt }),
            SysReg::Unknown => ops.push(Op::ConstI64 { dst: rt, imm: 0 }),
            _ => ops.push(Op::ReadSysReg {
                dst: rt,
                reg: decoded.reg.name().to_string(),
            }),
        }
    } else {
        match decoded.reg {
            SysReg::Nzcv => ops.push(Op::WriteNzcv { src: rt }),
            SysReg::Unknown => {}
            _ => ops.push(Op::WriteSysReg {
                reg: decoded.reg.name().to_string(),
                src: rt,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MovWide {
    dst: u8,
    kind: MovWideKind,
    imm: u16,
    shift: u8,
    is_32: bool,
}

#[derive(Debug, Clone, Copy)]
enum MovWideKind {
    MovZ,
    MovN,
    MovK,
}

fn decode_mov_wide(word: u32) -> Result<Option<MovWide>, String> {
    let sf = (word >> 31) & 0x1;
    let opc = (word >> 29) & 0x3;
    let fixed = (word >> 23) & 0x3F;
    if fixed != 0b100101 {
        return Ok(None);
    }
    let hw = ((word >> 21) & 0x3) as u8;
    let imm16 = ((word >> 5) & 0xFFFF) as u16;
    let rd = (word & 0x1F) as u8;
    let kind = match opc {
        0b00 => MovWideKind::MovN,
        0b10 => MovWideKind::MovZ,
        0b11 => MovWideKind::MovK,
        _ => return Ok(None),
    };
    Ok(Some(MovWide {
        dst: rd,
        kind,
        imm: imm16,
        shift: hw,
        is_32: sf == 0,
    }))
}

fn emit_mov_wide(
    decoded: MovWide,
    _pc: u64,
    state: &mut DecodeState,
    ops: &mut Vec<Op>,
) -> Result<(), String> {
    let shift_bits = (decoded.shift as u32) * 16;
    let imm_value = (decoded.imm as u64) << shift_bits;
    // Rd=31 is XZR here: the result is discarded.
    let dst = write_zr(decoded.dst, state);
    match decoded.kind {
        MovWideKind::MovZ | MovWideKind::MovN => {
            let value = if matches!(decoded.kind, MovWideKind::MovZ) {
                imm_value
            } else {
                !imm_value
            };
            ops.push(Op::ConstI64 {
                dst: dst.clone(),
                imm: value as i64,
            });
        }
        MovWideKind::MovK => {
            // Keep the other halfwords of the runtime value.
            let mask = state.next_temp("imm");
            ops.push(Op::ConstI64 {
                dst: mask.clone(),
                imm: !(0xFFFF_u64 << shift_bits) as i64,
            });
            ops.push(Op::AndI64 {
                dst: dst.clone(),
                lhs: dst.clone(),
                rhs: mask,
            });
            let imm = state.next_temp("imm");
            ops.push(Op::ConstI64 {
                dst: imm.clone(),
                imm: imm_value as i64,
            });
            ops.push(Op::OrI64 {
                dst: dst.clone(),
                lhs: dst.clone(),
                rhs: imm,
            });
        }
    }
    if decoded.is_32 {
        zero_extend_32(&dst, state, ops);
    }
    Ok(())
}

fn decode_adr(word: u32, pc: u64) -> Result<Option<Op>, String> {
    let op = (word >> 31) & 0x1;
    let fixed = (word >> 24) & 0x1F;
    if fixed != 0b10000 {
        return Ok(None);
    }
    let immlo = (word >> 29) & 0x3;
    let immhi = (word >> 5) & 0x7FFFF;
    let rd = (word & 0x1F) as u8;
    let imm = ((immhi << 2) | immlo) as i64;
    let signed = sign_extend(imm, 21);
    if op == 0 {
        return Ok(Some(Op::PcRel {
            dst: reg_name(rd),
            pc: pc as i64,
            offset: signed,
        }));
    }
    let page = (pc as i64) & !0xFFF;
    let offset = signed << 12;
    Ok(Some(Op::PcRel {
        dst: reg_name(rd),
        pc: page,
        offset,
    }))
}

/// The second operand of an add/sub.
#[derive(Debug, Clone, Copy)]
enum AddSubOperand {
    Imm(u64),
    /// Rm shifted by `amount` (0 LSL, 1 LSR, 2 ASR).
    Shifted {
        rm: u8,
        shift: u8,
        amount: u8,
    },
    /// Rm extended per `option` (0-3 UXTB..UXTX, 4-7 SXTB..SXTX), then
    /// shifted left by `amount`.
    Extended {
        rm: u8,
        option: u8,
        amount: u8,
    },
}

#[derive(Debug, Clone, Copy)]
struct AddSub {
    dst: u8,
    src: u8,
    operand: AddSubOperand,
    is_sub: bool,
    set_flags: bool,
    is_32: bool,
}

/// ADD/ADDS/SUB/SUBS in the immediate, shifted-register and
/// extended-register forms (CMP, CMN, NEG and MOV to/from SP are aliases).
fn decode_add_sub(word: u32) -> Option<AddSub> {
    let is_32 = (word >> 31) & 1 == 0;
    let rm = ((word >> 16) & 0x1F) as u8;
    let operand = match (word >> 24) & 0x1F {
        0b10001 => {
            let shift = (word >> 22) & 0x3;
            if shift > 1 {
                return None;
            }
            AddSubOperand::Imm(u64::from((word >> 10) & 0xFFF) << (shift * 12))
        }
        0b01011 if (word >> 21) & 1 == 0 => {
            let shift = ((word >> 22) & 0x3) as u8;
            let amount = ((word >> 10) & 0x3F) as u8;
            if shift == 3 || (is_32 && amount >= 32) {
                return None;
            }
            AddSubOperand::Shifted { rm, shift, amount }
        }
        0b01011 => {
            let amount = ((word >> 10) & 0x7) as u8;
            if (word >> 22) & 0x3 != 0 || amount > 4 {
                return None;
            }
            AddSubOperand::Extended {
                rm,
                option: ((word >> 13) & 0x7) as u8,
                amount,
            }
        }
        _ => return None,
    };
    Some(AddSub {
        dst: (word & 0x1F) as u8,
        src: ((word >> 5) & 0x1F) as u8,
        operand,
        is_sub: (word >> 30) & 1 == 1,
        set_flags: (word >> 29) & 1 == 1,
        is_32,
    })
}

fn emit_add_sub(decoded: AddSub, state: &mut DecodeState, ops: &mut Vec<Op>) {
    // Register 31 is SP for Rn (and for Rd unless flags are set) in the
    // immediate and extended forms, and XZR throughout the shifted form.
    let sp_form = !matches!(decoded.operand, AddSubOperand::Shifted { .. });
    let lhs = if sp_form {
        reg_name(decoded.src)
    } else {
        read_zr(decoded.src, state, ops)
    };
    let rhs = match decoded.operand {
        AddSubOperand::Imm(imm) => const_temp(imm as i64, state, ops),
        AddSubOperand::Shifted { rm, shift, amount } => {
            emit_shifted_operand(rm, shift, amount, decoded.is_32, state, ops)
        }
        AddSubOperand::Extended { rm, option, amount } => {
            emit_extended_operand(rm, option, amount, state, ops)
        }
    };
    // Flags read the original operands, so set them before Rd is written.
    if decoded.set_flags {
        set_add_sub_flags(&lhs, &rhs, decoded.is_sub, decoded.is_32, state, ops);
    }
    let dst = if sp_form && !decoded.set_flags {
        reg_name(decoded.dst)
    } else {
        write_zr(decoded.dst, state)
    };
    if decoded.is_sub {
        ops.push(Op::SubI64 {
            dst: dst.clone(),
            lhs,
            rhs,
        });
    } else {
        ops.push(Op::AddI64 {
            dst: dst.clone(),
            lhs,
            rhs,
        });
    }
    if decoded.is_32 {
        zero_extend_32(&dst, state, ops);
    }
}

/// Set NZCV for `lhs - rhs` (or `lhs + rhs`). W forms compare the operands
/// shifted into the top half, which yields the 32-bit N, Z, C and V.
fn set_add_sub_flags(
    lhs: &str,
    rhs: &str,
    is_sub: bool,
    is_32: bool,
    state: &mut DecodeState,
    ops: &mut Vec<Op>,
) {
    let (lhs, rhs) = if is_32 {
        let shift = const_temp(32, state, ops);
        let wide_lhs = state.next_temp("flags");
        ops.push(Op::LslI64 {
            dst: wide_lhs.clone(),
            lhs: lhs.to_string(),
            rhs: shift.clone(),
        });
        let wide_rhs = state.next_temp("flags");
        ops.push(Op::LslI64 {
            dst: wide_rhs.clone(),
            lhs: rhs.to_string(),
            rhs: shift,
        });
        (wide_lhs, wide_rhs)
    } else {
        (lhs.to_string(), rhs.to_string())
    };
    if is_sub {
        ops.push(Op::CmpI64 { lhs, rhs });
    } else {
        ops.push(Op::CmnI64 { lhs, rhs });
    }
}

fn const_temp(value: i64, state: &mut DecodeState, ops: &mut Vec<Op>) -> String {
    let temp = state.next_temp("imm");
    ops.push(Op::ConstI64 {
        dst: temp.clone(),
        imm: value,
    });
    temp
}

/// Copy `src` into a fresh temp so it can be modified in place.
fn copy_temp(src: String, prefix: &str, state: &mut DecodeState, ops: &mut Vec<Op>) -> String {
    let temp = state.next_temp(prefix);
    ops.push(Op::MovI64 {
        dst: temp.clone(),
        src,
    });
    temp
}

/// Zero-extend the low `bits` of `reg` in place.
fn zero_extend_bits(reg: &str, bits: u32, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let mask = const_temp(((1u64 << bits) - 1) as i64, state, ops);
    ops.push(Op::AndI64 {
        dst: reg.to_string(),
        lhs: reg.to_string(),
        rhs: mask,
    });
}

/// Rm shifted by `amount` at the operand width (0 LSL, 1 LSR, 2 ASR, 3 ROR).
fn emit_shifted_operand(
    rm: u8,
    shift: u8,
    amount: u8,
    is_32: bool,
    state: &mut DecodeState,
    ops: &mut Vec<Op>,
) -> String {
    let src = read_zr(rm, state, ops);
    if amount == 0 {
        return src;
    }
    let value = copy_temp(src, "op2", state, ops);
    if is_32 {
        // Right shifts and rotates of a W register must not see bits 63:32.
        match shift {
            1 | 3 => zero_extend_32(&value, state, ops),
            2 => sign_extend_reg(&value, 32, state, ops),
            _ => {}
        }
    }
    let by = const_temp(i64::from(amount), state, ops);
    match shift {
        0 => ops.push(Op::LslI64 {
            dst: value.clone(),
            lhs: value.clone(),
            rhs: by,
        }),
        1 => ops.push(Op::LsrI64 {
            dst: value.clone(),
            lhs: value.clone(),
            rhs: by,
        }),
        2 => ops.push(Op::AsrI64 {
            dst: value.clone(),
            lhs: value.clone(),
            rhs: by,
        }),
        _ => {
            let width = if is_32 { 32 } else { 64 };
            emit_rotate_right(&value, &by, width, state, ops);
        }
    }
    value
}

/// Rotate the low `width` bits of `value` (already zero-extended for W)
/// right by the amount in `by` (0..width), in place.
fn emit_rotate_right(
    value: &str,
    by: &str,
    width: i64,
    state: &mut DecodeState,
    ops: &mut Vec<Op>,
) {
    let width_reg = const_temp(width, state, ops);
    let back = state.next_temp("rot");
    ops.push(Op::SubI64 {
        dst: back.clone(),
        lhs: width_reg,
        rhs: by.to_string(),
    });
    // A rotate by 0 shifts left by `width`; codegen masks shift amounts to
    // 63, and the W result is zero-extended below.
    let high = state.next_temp("rot");
    ops.push(Op::LslI64 {
        dst: high.clone(),
        lhs: value.to_string(),
        rhs: back,
    });
    ops.push(Op::LsrI64 {
        dst: value.to_string(),
        lhs: value.to_string(),
        rhs: by.to_string(),
    });
    ops.push(Op::OrI64 {
        dst: value.to_string(),
        lhs: value.to_string(),
        rhs: high,
    });
    if width == 32 {
        zero_extend_32(value, state, ops);
    }
}

/// Rm extended per `option`, then shifted left by `amount`.
fn emit_extended_operand(
    rm: u8,
    option: u8,
    amount: u8,
    state: &mut DecodeState,
    ops: &mut Vec<Op>,
) -> String {
    let src = read_zr(rm, state, ops);
    let value = copy_temp(src, "op2", state, ops);
    let bits = 8u32 << (option & 0x3);
    if bits < 64 {
        if option & 0x4 != 0 {
            sign_extend_reg(&value, bits, state, ops);
        } else {
            zero_extend_bits(&value, bits, state, ops);
        }
    }
    if amount > 0 {
        let by = const_temp(i64::from(amount), state, ops);
        ops.push(Op::LslI64 {
            dst: value.clone(),
            lhs: value.clone(),
            rhs: by,
        });
    }
    value
}

#[derive(Debug, Clone, Copy)]
struct LogicalReg {
    dst: u8,
    lhs: u8,
    rhs: u8,
    opc: u8,
    /// N bit: BIC/ORN/EON/BICS use the inverted operand.
    invert: bool,
    shift: u8,
    amount: u8,
    is_32: bool,
}

/// AND/BIC/ORR/ORN/EOR/EON/ANDS/BICS (shifted register); MOV, MVN and TST
/// are aliases.
fn decode_logical_register(word: u32) -> Option<LogicalReg> {
    if (word >> 24) & 0x1F != 0b01010 {
        return None;
    }
    let is_32 = (word >> 31) & 1 == 0;
    let amount = ((word >> 10) & 0x3F) as u8;
    if is_32 && amount >= 32 {
        return None;
    }
    Some(LogicalReg {
        dst: (word & 0x1F) as u8,
        lhs: ((word >> 5) & 0x1F) as u8,
        rhs: ((word >> 16) & 0x1F) as u8,
        opc: ((word >> 29) & 0x3) as u8,
        invert: (word >> 21) & 1 == 1,
        shift: ((word >> 22) & 0x3) as u8,
        amount,
        is_32,
    })
}

fn emit_logical_register(decoded: LogicalReg, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let lhs = read_zr(decoded.lhs, state, ops);
    let mut rhs = emit_shifted_operand(
        decoded.rhs,
        decoded.shift,
        decoded.amount,
        decoded.is_32,
        state,
        ops,
    );
    if decoded.invert {
        let ones = const_temp(-1, state, ops);
        let inverted = state.next_temp("op2");
        ops.push(Op::XorI64 {
            dst: inverted.clone(),
            lhs: rhs,
            rhs: ones,
        });
        rhs = inverted;
    }
    let dst = write_zr(decoded.dst, state);
    ops.push(match decoded.opc {
        0b01 => Op::OrI64 {
            dst: dst.clone(),
            lhs,
            rhs,
        },
        0b10 => Op::XorI64 {
            dst: dst.clone(),
            lhs,
            rhs,
        },
        _ => Op::AndI64 {
            dst: dst.clone(),
            lhs,
            rhs,
        },
    });
    if decoded.is_32 {
        zero_extend_32(&dst, state, ops);
    }
    if decoded.opc == 0b11 {
        set_logical_flags(&dst, decoded.is_32, state, ops);
    }
}

/// ANDS/TST: N and Z from the operand width, C and V cleared. `result` must
/// already be zero-extended for 32-bit forms.
fn set_logical_flags(result: &str, is_32: bool, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let value = if is_32 {
        // Move bit 31 to the sign bit so N reflects the W result.
        let shift = state.next_temp("imm");
        ops.push(Op::ConstI64 {
            dst: shift.clone(),
            imm: 32,
        });
        let wide = state.next_temp("flags");
        ops.push(Op::LslI64 {
            dst: wide.clone(),
            lhs: result.to_string(),
            rhs: shift,
        });
        wide
    } else {
        result.to_string()
    };
    ops.push(Op::TestI64 {
        lhs: value.clone(),
        rhs: value,
    });
}

#[derive(Debug, Clone, Copy)]
struct LogicalImm {
    dst: u8,
    src: u8,
    opc: u8,
    imm: u64,
    is_32: bool,
}

/// AND/ORR/EOR/ANDS (immediate): bits 28:23 = 100100.
fn decode_logical_immediate(word: u32) -> Option<LogicalImm> {
    if (word >> 23) & 0x3F != 0b100100 {
        return None;
    }
    let is_32 = (word >> 31) & 1 == 0;
    let imm = decode_bit_masks(
        (word >> 22) & 1,
        (word >> 10) & 0x3F,
        (word >> 16) & 0x3F,
        is_32,
    )?;
    Some(LogicalImm {
        dst: (word & 0x1F) as u8,
        src: ((word >> 5) & 0x1F) as u8,
        opc: ((word >> 29) & 0x3) as u8,
        imm,
        is_32,
    })
}

/// The architectural DecodeBitMasks for logical immediates: a run of
/// `imms + 1` ones, rotated right by `immr` within an element of 2..64 bits,
/// replicated across the register. Returns None for reserved encodings.
fn decode_bit_masks(n: u32, imms: u32, immr: u32, is_32: bool) -> Option<u64> {
    let combined = (n << 6) | (!imms & 0x3F);
    if combined == 0 || (is_32 && n == 1) {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    if len < 1 {
        return None;
    }
    let levels = (1u32 << len) - 1;
    let s = imms & levels;
    let r = immr & levels;
    if s == levels {
        return None;
    }
    let esize = 1u32 << len;
    let emask = if esize == 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    let welem = (1u64 << (s + 1)) - 1;
    let elem = if r == 0 {
        welem
    } else {
        ((welem >> r) | (welem << (esize - r))) & emask
    };
    let mut value = 0u64;
    let mut pos = 0;
    while pos < 64 {
        value |= elem << pos;
        pos += esize;
    }
    Some(if is_32 { value & 0xFFFF_FFFF } else { value })
}

fn emit_logical_immediate(decoded: LogicalImm, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let src = read_zr(decoded.src, state, ops);
    let imm = state.next_temp("imm");
    ops.push(Op::ConstI64 {
        dst: imm.clone(),
        imm: decoded.imm as i64,
    });
    // Rd=31 is SP for AND/ORR/EOR and XZR for ANDS.
    let dst = if decoded.opc == 0b11 {
        write_zr(decoded.dst, state)
    } else {
        reg_name(decoded.dst)
    };
    let op = match decoded.opc {
        0b01 => Op::OrI64 {
            dst: dst.clone(),
            lhs: src,
            rhs: imm,
        },
        0b10 => Op::XorI64 {
            dst: dst.clone(),
            lhs: src,
            rhs: imm,
        },
        _ => Op::AndI64 {
            dst: dst.clone(),
            lhs: src,
            rhs: imm,
        },
    };
    ops.push(op);
    if decoded.is_32 {
        zero_extend_32(&dst, state, ops);
    }
    if decoded.opc == 0b11 {
        set_logical_flags(&dst, decoded.is_32, state, ops);
    }
}

/// What a single-register integer load/store does, from its size (bits
/// 31:30) and opc (bits 23:22) fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemKind {
    Store,
    /// LDR/LDRB/LDRH: zero-extending load.
    Load,
    /// LDRSB/LDRSH/LDRSW into an X register.
    LoadSigned64,
    /// LDRSB/LDRSH into a W register.
    LoadSigned32,
    /// PRFM: a cache hint with no architectural effect.
    Prefetch,
}

/// Classify the fields shared by every single-register integer load/store
/// form. FP/SIMD forms (V, bit 26) and unallocated combinations return None
/// so the instruction traps instead of being miscompiled.
fn classify_mem(word: u32) -> Option<(MemKind, u8)> {
    if (word >> 26) & 1 == 1 {
        return None;
    }
    let size = ((word >> 30) & 0x3) as u8;
    let kind = match ((word >> 22) & 0x3, size) {
        (0b00, _) => MemKind::Store,
        (0b01, _) => MemKind::Load,
        (0b10, 3) => MemKind::Prefetch,
        (0b10, _) => MemKind::LoadSigned64,
        (0b11, 0 | 1) => MemKind::LoadSigned32,
        _ => return None,
    };
    Some((kind, size))
}

/// Emit one `1 << size`-byte access of Rt at `addr + offset`.
fn emit_mem_access(
    kind: MemKind,
    size: u8,
    rt: u8,
    addr: String,
    offset: i64,
    state: &mut DecodeState,
    ops: &mut Vec<Op>,
) {
    match kind {
        MemKind::Prefetch => {}
        MemKind::Store => {
            // Rt=31 is XZR: stores write zero.
            let src = read_zr(rt, state, ops);
            ops.push(match size {
                0 => Op::StoreI8 { src, addr, offset },
                1 => Op::StoreI16 { src, addr, offset },
                2 => Op::StoreI32 { src, addr, offset },
                _ => Op::StoreI64 { src, addr, offset },
            });
        }
        MemKind::Load | MemKind::LoadSigned64 | MemKind::LoadSigned32 => {
            // Rt=31 is XZR: the loaded value is discarded.
            let dst = write_zr(rt, state);
            ops.push(match size {
                0 => Op::LoadI8 {
                    dst: dst.clone(),
                    addr,
                    offset,
                },
                1 => Op::LoadI16 {
                    dst: dst.clone(),
                    addr,
                    offset,
                },
                2 => Op::LoadI32 {
                    dst: dst.clone(),
                    addr,
                    offset,
                },
                _ => Op::LoadI64 {
                    dst: dst.clone(),
                    addr,
                    offset,
                },
            });
            // Narrow loads already zero-extend; signed forms widen here.
            if kind != MemKind::Load {
                sign_extend_reg(&dst, 8 << size, state, ops);
                if kind == MemKind::LoadSigned32 {
                    zero_extend_32(&dst, state, ops);
                }
            }
        }
    }
}

/// Sign-extend the low `bits` of `reg` to 64 bits.
fn sign_extend_reg(reg: &str, bits: u32, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let shift = state.next_temp("imm");
    ops.push(Op::ConstI64 {
        dst: shift.clone(),
        imm: i64::from(64 - bits),
    });
    ops.push(Op::LslI64 {
        dst: reg.to_string(),
        lhs: reg.to_string(),
        rhs: shift.clone(),
    });
    ops.push(Op::AsrI64 {
        dst: reg.to_string(),
        lhs: reg.to_string(),
        rhs: shift,
    });
}

#[derive(Debug, Clone, Copy)]
struct LoadStore {
    kind: MemKind,
    size: u8,
    base: u8,
    rt: u8,
    offset: u64,
}

/// Unsigned scaled 12-bit offset: bits 29:27 = 111, bits 25:24 = 01.
fn decode_load_store(word: u32) -> Option<LoadStore> {
    if (word & 0x3B00_0000) != 0x3900_0000 {
        return None;
    }
    let (kind, size) = classify_mem(word)?;
    let imm12 = (word >> 10) & 0xFFF;
    Some(LoadStore {
        kind,
        size,
        base: ((word >> 5) & 0x1F) as u8,
        rt: (word & 0x1F) as u8,
        offset: u64::from(imm12) << size,
    })
}

fn emit_load_store(decoded: LoadStore, state: &mut DecodeState, ops: &mut Vec<Op>) {
    emit_mem_access(
        decoded.kind,
        decoded.size,
        decoded.rt,
        reg_name(decoded.base),
        decoded.offset as i64,
        state,
        ops,
    );
}

#[derive(Debug, Clone, Copy)]
struct UnscaledMem {
    kind: MemKind,
    size: u8,
    base: u8,
    rt: u8,
    offset: i64,
    mode: u8, // 0 = unscaled, 1 = post-index, 2 = unprivileged, 3 = pre-index
}

/// Signed 9-bit offset forms (LDUR/STUR, LDTR/STTR, pre- and post-index):
/// bits 29:27 = 111, bits 25:24 = 00, bit 21 clear.
fn decode_unscaled_load_store(word: u32) -> Option<UnscaledMem> {
    if (word & 0x3B20_0000) != 0x3800_0000 {
        return None;
    }
    let (kind, size) = classify_mem(word)?;
    let mode = ((word >> 10) & 0x3) as u8;
    // There is no PRFM with writeback.
    if kind == MemKind::Prefetch && mode & 1 == 1 {
        return None;
    }
    Some(UnscaledMem {
        kind,
        size,
        base: ((word >> 5) & 0x1F) as u8,
        rt: (word & 0x1F) as u8,
        offset: sign_extend(((word >> 12) & 0x1FF) as i64, 9),
        mode,
    })
}

fn emit_unscaled_load_store(decoded: UnscaledMem, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let base = reg_name(decoded.base);
    let advance_base = |state: &mut DecodeState, ops: &mut Vec<Op>| {
        if decoded.offset == 0 {
            return;
        }
        let imm = state.next_temp("imm");
        ops.push(Op::ConstI64 {
            dst: imm.clone(),
            imm: decoded.offset,
        });
        ops.push(Op::AddI64 {
            dst: base.clone(),
            lhs: base.clone(),
            rhs: imm,
        });
    };
    // Pre-index updates the base first and accesses the updated address;
    // post-index accesses the base and advances it afterwards.
    let offset = match decoded.mode {
        1 => 0,
        3 => {
            advance_base(state, ops);
            0
        }
        _ => decoded.offset,
    };
    emit_mem_access(
        decoded.kind,
        decoded.size,
        decoded.rt,
        base.clone(),
        offset,
        state,
        ops,
    );
    if decoded.mode == 1 {
        advance_base(state, ops);
    }
}

#[derive(Debug, Clone, Copy)]
struct RegOffsetMem {
    kind: MemKind,
    size: u8,
    base: u8,
    rm: u8,
    rt: u8,
    /// Index extend: 0b010 UXTW, 0b011 LSL (UXTX), 0b110 SXTW, 0b111 SXTX.
    option: u8,
    /// S bit: scale the index by the access size.
    scaled: bool,
}

/// Register offset: bits 29:27 = 111, bits 25:24 = 00, bit 21 set and
/// bits 11:10 = 10 (bit 21 with 11:10 = 00 is the atomic-memory group).
fn decode_register_offset_load_store(word: u32) -> Option<RegOffsetMem> {
    if (word & 0x3B20_0C00) != 0x3820_0800 {
        return None;
    }
    let (kind, size) = classify_mem(word)?;
    let option = ((word >> 13) & 0x7) as u8;
    if option & 0b010 == 0 {
        return None;
    }
    Some(RegOffsetMem {
        kind,
        size,
        base: ((word >> 5) & 0x1F) as u8,
        rm: ((word >> 16) & 0x1F) as u8,
        rt: (word & 0x1F) as u8,
        option,
        scaled: (word >> 12) & 1 == 1,
    })
}

fn emit_register_offset_load_store(
    decoded: RegOffsetMem,
    state: &mut DecodeState,
    ops: &mut Vec<Op>,
) {
    let shift = if decoded.scaled {
        u32::from(decoded.size)
    } else {
        0
    };
    let addr = register_offset_address(decoded.base, decoded.rm, decoded.option, shift, state, ops);
    emit_mem_access(decoded.kind, decoded.size, decoded.rt, addr, 0, state, ops);
}

fn vreg(reg: u8) -> String {
    format!("v{reg}")
}

/// How a load/store forms its address.
#[derive(Debug, Clone, Copy)]
enum MemAddressing {
    Offset(i64),
    PostIndex(i64),
    PreIndex(i64),
    /// Base plus Rm, extended per `option` and scaled by the access size
    /// when `scaled`.
    Register {
        rm: u8,
        option: u8,
        scaled: bool,
    },
}

/// `base + extend(Rm) << shift` into a fresh temp (register-offset forms).
fn register_offset_address(
    base: u8,
    rm: u8,
    option: u8,
    shift: u32,
    state: &mut DecodeState,
    ops: &mut Vec<Op>,
) -> String {
    // Rm=31 is XZR; Rn=31 is SP.
    let index = read_zr(rm, state, ops);
    let addr = copy_temp(index, "addr", state, ops);
    match option {
        0b010 => zero_extend_32(&addr, state, ops),
        0b110 => sign_extend_reg(&addr, 32, state, ops),
        _ => {}
    }
    if shift > 0 {
        let by = const_temp(i64::from(shift), state, ops);
        ops.push(Op::LslI64 {
            dst: addr.clone(),
            lhs: addr.clone(),
            rhs: by,
        });
    }
    ops.push(Op::AddI64 {
        dst: addr.clone(),
        lhs: reg_name(base),
        rhs: addr.clone(),
    });
    addr
}

/// Add `offset` to a base register in place (load/store writeback).
fn advance_register(reg: &str, offset: i64, state: &mut DecodeState, ops: &mut Vec<Op>) {
    if offset == 0 {
        return;
    }
    let imm = const_temp(offset, state, ops);
    ops.push(Op::AddI64 {
        dst: reg.to_string(),
        lhs: reg.to_string(),
        rhs: imm,
    });
}

#[derive(Debug, Clone, Copy)]
struct SimdMem {
    bytes: u8,
    is_load: bool,
    rt: u8,
    base: u8,
    addressing: MemAddressing,
}

/// SIMD&FP single-register LDR/STR/LDUR/STUR of B, H, S, D and Q registers
/// (unsigned offset, unscaled, pre/post-index and register offset).
fn decode_simd_load_store(word: u32) -> Option<SimdMem> {
    let size = (word >> 30) & 0x3;
    let opc = (word >> 22) & 0x3;
    let bytes: u8 = if opc & 0b10 != 0 {
        if size != 0 {
            return None;
        }
        16
    } else {
        1 << size
    };
    let addressing = if word & 0x3F00_0000 == 0x3D00_0000 {
        MemAddressing::Offset(i64::from((word >> 10) & 0xFFF) * i64::from(bytes))
    } else if word & 0x3F20_0000 == 0x3C00_0000 {
        let imm9 = sign_extend(((word >> 12) & 0x1FF) as i64, 9);
        match (word >> 10) & 0x3 {
            0 => MemAddressing::Offset(imm9),
            1 => MemAddressing::PostIndex(imm9),
            3 => MemAddressing::PreIndex(imm9),
            _ => return None,
        }
    } else if word & 0x3F20_0C00 == 0x3C20_0800 {
        let option = ((word >> 13) & 0x7) as u8;
        if option & 0b010 == 0 {
            return None;
        }
        MemAddressing::Register {
            rm: ((word >> 16) & 0x1F) as u8,
            option,
            scaled: (word >> 12) & 1 == 1,
        }
    } else {
        return None;
    };
    Some(SimdMem {
        bytes,
        is_load: opc & 1 == 1,
        rt: (word & 0x1F) as u8,
        base: ((word >> 5) & 0x1F) as u8,
        addressing,
    })
}

fn emit_simd_access(
    is_load: bool,
    rt: String,
    addr: String,
    offset: i64,
    bytes: u8,
    ops: &mut Vec<Op>,
) {
    ops.push(if is_load {
        Op::VLoad {
            dst: rt,
            addr,
            offset,
            bytes,
        }
    } else {
        Op::VStore {
            src: rt,
            addr,
            offset,
            bytes,
        }
    });
}

/// LDR (literal): W, X, LDRSW, PRFM and the S/D/Q forms.
fn emit_load_literal(word: u32, pc: u64, state: &mut DecodeState, ops: &mut Vec<Op>) -> bool {
    if word & 0x3B00_0000 != 0x1800_0000 {
        return false;
    }
    let opc = (word >> 30) & 0x3;
    let vector = (word >> 26) & 1 == 1;
    let rt = (word & 0x1F) as u8;
    if vector && opc == 3 {
        return false;
    }
    let addr = state.next_temp("literal");
    ops.push(Op::PcRel {
        dst: addr.clone(),
        pc: pc as i64,
        offset: sign_extend(i64::from((word >> 5) & 0x7FFFF) << 2, 21),
    });
    if vector {
        emit_simd_access(true, format!("v{rt}"), addr, 0, 4 << opc, ops);
    } else {
        let (kind, size) = match opc {
            0 => (MemKind::Load, 2),
            1 => (MemKind::Load, 3),
            2 => (MemKind::LoadSigned64, 2),
            _ => (MemKind::Prefetch, 3),
        };
        emit_mem_access(kind, size, rt, addr, 0, state, ops);
    }
    true
}

/// LD1-LD4 and ST1-ST4: multiple structures, one lane, or (LDnR) one
/// element replicated, with optional post-index writeback.
#[derive(Debug, Clone, Copy)]
struct SimdStructure {
    load: bool,
    rt: u8,
    rn: u8,
    /// Post-index register; 31 adds the transfer size instead.
    post: Option<u8>,
    kind: StructureKind,
}

#[derive(Debug, Clone, Copy)]
enum StructureKind {
    /// `rpt` groups of `selem` registers, elements interleaved.
    Multiple {
        selem: u8,
        rpt: u8,
        esize: u8,
        q: bool,
    },
    /// Element `index` of `selem` registers.
    Lane { selem: u8, esize: u8, index: u8 },
    /// One element per register, replicated to every lane.
    Replicate { selem: u8, esize: u8, q: bool },
}

fn decode_simd_structure(word: u32) -> Option<SimdStructure> {
    let q = (word >> 30) & 1;
    let load = (word >> 22) & 1 == 1;
    let size = (word >> 10) & 0x3;
    let post = ((word >> 23) & 1 == 1).then_some(((word >> 16) & 0x1F) as u8);
    let kind = if word & 0xBFBF_0000 == 0x0C00_0000 || word & 0xBFA0_0000 == 0x0C80_0000 {
        let (selem, rpt) = match (word >> 12) & 0xF {
            0b0000 => (4, 1),
            0b0010 => (1, 4),
            0b0100 => (3, 1),
            0b0110 => (1, 3),
            0b0111 => (1, 1),
            0b1000 => (2, 1),
            0b1010 => (1, 2),
            _ => return None,
        };
        if size == 3 && q == 0 && selem != 1 {
            return None;
        }
        StructureKind::Multiple {
            selem,
            rpt,
            esize: 1 << size,
            q: q == 1,
        }
    } else if word & 0xBF9F_0000 == 0x0D00_0000 || word & 0xBF80_0000 == 0x0D80_0000 {
        let opcode = (word >> 13) & 0x7;
        let s = (word >> 12) & 1;
        let selem = (((opcode & 1) << 1) | ((word >> 21) & 1)) as u8 + 1;
        let lane = |esize: u8, index: u32| StructureKind::Lane {
            selem,
            esize,
            index: index as u8,
        };
        match opcode >> 1 {
            0 => lane(1, (q << 3) | (s << 2) | size),
            1 if size & 1 == 0 => lane(2, (q << 2) | (s << 1) | (size >> 1)),
            2 if size == 0 => lane(4, (q << 1) | s),
            2 if size == 1 && s == 0 => lane(8, q),
            3 if load && s == 0 => StructureKind::Replicate {
                selem,
                esize: 1 << size,
                q: q == 1,
            },
            _ => return None,
        }
    } else {
        return None;
    };
    Some(SimdStructure {
        load,
        rt: (word & 0x1F) as u8,
        rn: ((word >> 5) & 0x1F) as u8,
        post,
        kind,
    })
}

fn emit_simd_structure(decoded: SimdStructure, state: &mut DecodeState, ops: &mut Vec<Op>) {
    // Rn = 31 is SP.
    let base = reg_name(decoded.rn);
    let vreg = |i: u8| format!("v{}", (decoded.rt + i) % 32);
    let mut offset = 0i64;
    match decoded.kind {
        StructureKind::Multiple {
            selem: 1, rpt, q, ..
        } => {
            let bytes = if q { 16 } else { 8 };
            for r in 0..rpt {
                emit_simd_access(decoded.load, vreg(r), base.clone(), offset, bytes, ops);
                offset += i64::from(bytes);
            }
        }
        StructureKind::Multiple {
            selem, esize, q, ..
        } => {
            let lanes = if q { 16 } else { 8 } / esize;
            if decoded.load {
                // Every lane is written; a 64-bit form clears the top half.
                for s in 0..selem {
                    ops.push(Op::VConst {
                        dst: vreg(s),
                        lo: 0,
                        hi: 0,
                    });
                }
            }
            for index in 0..lanes {
                for s in 0..selem {
                    emit_simd_element(
                        decoded.load,
                        vreg(s),
                        esize,
                        index,
                        base.clone(),
                        offset,
                        state,
                        ops,
                    );
                    offset += i64::from(esize);
                }
            }
        }
        StructureKind::Lane {
            selem,
            esize,
            index,
        } => {
            for s in 0..selem {
                emit_simd_element(
                    decoded.load,
                    vreg(s),
                    esize,
                    index,
                    base.clone(),
                    offset,
                    state,
                    ops,
                );
                offset += i64::from(esize);
            }
        }
        StructureKind::Replicate { selem, esize, q } => {
            for s in 0..selem {
                let value = state.next_temp("lane");
                ops.push(load_op(esize, value.clone(), base.clone(), offset));
                ops.push(Op::VDup {
                    dst: vreg(s),
                    src: value,
                    esize,
                    q,
                });
                offset += i64::from(esize);
            }
        }
    }
    if let Some(rm) = decoded.post {
        let amount = if rm == 31 {
            const_temp(offset, state, ops)
        } else {
            reg_name(rm)
        };
        ops.push(Op::AddI64 {
            dst: base.clone(),
            lhs: base,
            rhs: amount,
        });
    }
}

/// Load or store lane `index` (of `esize` bytes) of `vreg` at
/// `base + offset`.
#[allow(clippy::too_many_arguments)]
fn emit_simd_element(
    load: bool,
    vreg: String,
    esize: u8,
    index: u8,
    base: String,
    offset: i64,
    state: &mut DecodeState,
    ops: &mut Vec<Op>,
) {
    let value = state.next_temp("lane");
    if load {
        ops.push(load_op(esize, value.clone(), base, offset));
        ops.push(Op::VInsGpr {
            dst: vreg,
            src: value,
            esize,
            index,
        });
    } else {
        ops.push(Op::VExtract {
            dst: value.clone(),
            src: vreg,
            esize,
            index,
            signed: false,
        });
        ops.push(match esize {
            1 => Op::StoreI8 {
                src: value,
                addr: base,
                offset,
            },
            2 => Op::StoreI16 {
                src: value,
                addr: base,
                offset,
            },
            4 => Op::StoreI32 {
                src: value,
                addr: base,
                offset,
            },
            _ => Op::StoreI64 {
                src: value,
                addr: base,
                offset,
            },
        });
    }
}

/// A zero-extending load of `esize` bytes.
fn load_op(esize: u8, dst: String, addr: String, offset: i64) -> Op {
    match esize {
        1 => Op::LoadI8 { dst, addr, offset },
        2 => Op::LoadI16 { dst, addr, offset },
        4 => Op::LoadI32 { dst, addr, offset },
        _ => Op::LoadI64 { dst, addr, offset },
    }
}

/// ADC, ADCS, SBC, SBCS (and NGC/NGCS, which read XZR).
#[derive(Debug, Clone, Copy)]
struct AddCarry {
    rd: u8,
    rn: u8,
    rm: u8,
    subtract: bool,
    set_flags: bool,
    is_32: bool,
}

fn decode_add_carry(word: u32) -> Option<AddCarry> {
    if word & 0x1FE0_FC00 != 0x1A00_0000 {
        return None;
    }
    Some(AddCarry {
        rd: (word & 0x1F) as u8,
        rn: ((word >> 5) & 0x1F) as u8,
        rm: ((word >> 16) & 0x1F) as u8,
        subtract: (word >> 30) & 1 == 1,
        set_flags: (word >> 29) & 1 == 1,
        is_32: (word >> 31) & 1 == 0,
    })
}

fn emit_add_carry(decoded: AddCarry, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let lhs = read_zr(decoded.rn, state, ops);
    let rhs = read_zr(decoded.rm, state, ops);
    ops.push(Op::AddCarry {
        dst: write_zr(decoded.rd, state),
        lhs,
        rhs,
        subtract: decoded.subtract,
        set_flags: decoded.set_flags,
        is_32: decoded.is_32,
    });
}

fn emit_simd_load_store(decoded: SimdMem, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let base = reg_name(decoded.base);
    let rt = vreg(decoded.rt);
    match decoded.addressing {
        MemAddressing::Offset(offset) => {
            emit_simd_access(decoded.is_load, rt, base, offset, decoded.bytes, ops)
        }
        MemAddressing::PreIndex(offset) => {
            advance_register(&base, offset, state, ops);
            emit_simd_access(decoded.is_load, rt, base, 0, decoded.bytes, ops);
        }
        MemAddressing::PostIndex(offset) => {
            emit_simd_access(decoded.is_load, rt, base.clone(), 0, decoded.bytes, ops);
            advance_register(&base, offset, state, ops);
        }
        MemAddressing::Register { rm, option, scaled } => {
            let shift = if scaled {
                decoded.bytes.trailing_zeros()
            } else {
                0
            };
            let addr = register_offset_address(decoded.base, rm, option, shift, state, ops);
            emit_simd_access(decoded.is_load, rt, addr, 0, decoded.bytes, ops);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SimdPair {
    bytes: u8,
    is_load: bool,
    rt: u8,
    rt2: u8,
    base: u8,
    offset: i64,
    mode: u8, // 0 = non-temporal offset, 1 = post-index, 2 = offset, 3 = pre-index
}

/// SIMD&FP STP/LDP/STNP/LDNP of S, D and Q registers: bits 29:27 = 101,
/// V (bit 26) set, bit 25 clear.
fn decode_simd_pair(word: u32) -> Option<SimdPair> {
    if (word >> 25) & 0x1F != 0b10110 {
        return None;
    }
    let bytes = match word >> 30 {
        0 => 4,
        1 => 8,
        2 => 16,
        _ => return None,
    };
    Some(SimdPair {
        bytes,
        is_load: (word >> 22) & 1 == 1,
        rt: (word & 0x1F) as u8,
        rt2: ((word >> 10) & 0x1F) as u8,
        base: ((word >> 5) & 0x1F) as u8,
        offset: sign_extend(((word >> 15) & 0x7F) as i64, 7) * i64::from(bytes),
        mode: ((word >> 23) & 0x3) as u8,
    })
}

fn emit_simd_pair(decoded: SimdPair, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let base = reg_name(decoded.base);
    let second = i64::from(decoded.bytes);
    let access_offset = match decoded.mode {
        3 => {
            advance_register(&base, decoded.offset, state, ops);
            0
        }
        1 => 0,
        _ => decoded.offset,
    };
    for (reg, extra) in [(decoded.rt, 0), (decoded.rt2, second)] {
        emit_simd_access(
            decoded.is_load,
            vreg(reg),
            base.clone(),
            access_offset + extra,
            decoded.bytes,
            ops,
        );
    }
    if decoded.mode == 1 {
        advance_register(&base, decoded.offset, state, ops);
    }
}

/// AdvSIMD and FP data movement (no arithmetic).
#[derive(Debug, Clone, Copy)]
enum SimdMove {
    /// DUP Vd.T, Rn.
    DupGeneral { rd: u8, rn: u8, esize: u8, q: bool },
    /// DUP Vd.T, Vn.Ts[i].
    DupElement {
        rd: u8,
        rn: u8,
        esize: u8,
        index: u8,
        q: bool,
    },
    /// UMOV/SMOV Rd, Vn.T[i] and FMOV Rd, Vn.
    ToGeneral {
        rd: u8,
        rn: u8,
        esize: u8,
        index: u8,
        signed: bool,
        is_32: bool,
    },
    /// INS Vd.T[i], Rn and FMOV Vd.D[1], Xn.
    InsGeneral {
        rd: u8,
        rn: u8,
        esize: u8,
        index: u8,
    },
    /// FMOV Hd/Sd/Dd, Rn: lane 0 written, the rest zeroed.
    FromGeneral { rd: u8, rn: u8, esize: u8 },
    /// INS Vd.T[i], Vn.T[j].
    InsElement {
        rd: u8,
        rn: u8,
        esize: u8,
        dst_index: u8,
        src_index: u8,
    },
    /// MOVI/MVNI and FMOV (immediate).
    Const { rd: u8, value: u128 },
    /// ORR/BIC (vector, immediate).
    BitwiseImm {
        rd: u8,
        value: u128,
        kind: &'static str,
        q: bool,
    },
    /// FMOV (register): copy the low `bytes`, zero the rest.
    MovScalar { rd: u8, rn: u8, bytes: u8 },
    /// AND/BIC/ORR/ORN/EOR (vector).
    Bitwise {
        rd: u8,
        rn: u8,
        rm: u8,
        kind: &'static str,
        q: bool,
    },
    /// BSL/BIT/BIF (vector).
    BitSelect {
        rd: u8,
        rn: u8,
        rm: u8,
        kind: u8,
        q: bool,
    },
}

/// Element size (log2 bytes) encoded by the lowest set bit of imm5.
fn imm5_size(imm5: u32) -> Option<u32> {
    let size = imm5.trailing_zeros();
    (size <= 3).then_some(size)
}

/// VFPExpandImm: the 8-bit floating-point immediate of FMOV.
fn vfp_expand_imm(imm8: u32, bits: u32) -> u64 {
    let exponent_bits = match bits {
        16 => 5,
        32 => 8,
        _ => 11,
    };
    let fraction_bits = bits - exponent_bits - 1;
    let sign = u64::from((imm8 >> 7) & 1);
    let b6 = u64::from((imm8 >> 6) & 1);
    let replicated = if b6 == 1 {
        (1u64 << (exponent_bits - 3)) - 1
    } else {
        0
    };
    let exponent =
        ((1 - b6) << (exponent_bits - 1)) | (replicated << 2) | u64::from((imm8 >> 4) & 0x3);
    let fraction = u64::from(imm8 & 0xF) << (fraction_bits - 4);
    (sign << (bits - 1)) | (exponent << fraction_bits) | fraction
}

/// AdvSIMDExpandImm: the 64-bit pattern of a modified immediate.
fn advsimd_expand_imm(op: u32, cmode: u32, imm8: u32) -> u64 {
    let imm = u64::from(imm8);
    let rep32 = |value: u64| value | (value << 32);
    let rep16 = |value: u64| value | (value << 16) | (value << 32) | (value << 48);
    match cmode >> 1 {
        0b000 => rep32(imm),
        0b001 => rep32(imm << 8),
        0b010 => rep32(imm << 16),
        0b011 => rep32(imm << 24),
        0b100 => rep16(imm),
        0b101 => rep16(imm << 8),
        0b110 if cmode & 1 == 0 => rep32((imm << 8) | 0xFF),
        0b110 => rep32((imm << 16) | 0xFFFF),
        _ => match (cmode & 1, op) {
            (0, 0) => imm * 0x0101_0101_0101_0101,
            (0, _) => (0..8).fold(0u64, |acc, bit| {
                if (imm8 >> bit) & 1 == 1 {
                    acc | (0xFF << (bit * 8))
                } else {
                    acc
                }
            }),
            (_, 0) => rep32(vfp_expand_imm(imm8, 32)),
            _ => vfp_expand_imm(imm8, 64),
        },
    }
}

fn decode_simd_move(word: u32) -> Option<SimdMove> {
    let rd = (word & 0x1F) as u8;
    let rn = ((word >> 5) & 0x1F) as u8;
    let rm = ((word >> 16) & 0x1F) as u8;
    let q = (word >> 30) & 1 == 1;
    let imm5 = (word >> 16) & 0x1F;

    if word & 0xBFE0_FC00 == 0x0E00_0400 {
        let size = imm5_size(imm5)?;
        if size == 3 && !q {
            return None;
        }
        return Some(SimdMove::DupElement {
            rd,
            rn,
            esize: 1 << size,
            index: (imm5 >> (size + 1)) as u8,
            q,
        });
    }
    if word & 0xBFE0_FC00 == 0x0E00_0C00 {
        let size = imm5_size(imm5)?;
        if size == 3 && !q {
            return None;
        }
        return Some(SimdMove::DupGeneral {
            rd,
            rn,
            esize: 1 << size,
            q,
        });
    }
    if word & 0xBFE0_FC00 == 0x0E00_3C00 || word & 0xBFE0_FC00 == 0x0E00_2C00 {
        let signed = word & 0xBFE0_FC00 == 0x0E00_2C00;
        let size = imm5_size(imm5)?;
        // UMOV: W for B/H/S, X for D. SMOV: W for B/H, X for B/H/S.
        let valid = if signed {
            size < if q { 3 } else { 2 }
        } else {
            (size == 3) == q
        };
        if !valid {
            return None;
        }
        return Some(SimdMove::ToGeneral {
            rd,
            rn,
            esize: 1 << size,
            index: (imm5 >> (size + 1)) as u8,
            signed,
            is_32: !q,
        });
    }
    if word & 0xFFE0_FC00 == 0x4E00_1C00 {
        let size = imm5_size(imm5)?;
        return Some(SimdMove::InsGeneral {
            rd,
            rn,
            esize: 1 << size,
            index: (imm5 >> (size + 1)) as u8,
        });
    }
    if word & 0xFFE0_8400 == 0x6E00_0400 {
        let size = imm5_size(imm5)?;
        return Some(SimdMove::InsElement {
            rd,
            rn,
            esize: 1 << size,
            dst_index: (imm5 >> (size + 1)) as u8,
            src_index: (((word >> 11) & 0xF) >> size) as u8,
        });
    }
    if word & 0x9FF8_0400 == 0x0F00_0400 {
        // AdvSIMD modified immediate; o2 (bit 11) selects FP16 forms.
        if (word >> 11) & 1 == 1 {
            return None;
        }
        let op = (word >> 29) & 1;
        let cmode = (word >> 12) & 0xF;
        let imm8 = (((word >> 16) & 0x7) << 5) | ((word >> 5) & 0x1F);
        if cmode == 0b1111 && op == 1 && !q {
            return None;
        }
        let pattern = advsimd_expand_imm(op, cmode, imm8);
        let width = |value: u64| {
            if q {
                u128::from(value) | (u128::from(value) << 64)
            } else {
                u128::from(value)
            }
        };
        let is_orr_bic = cmode < 0b1100 && cmode & 1 == 1;
        if is_orr_bic {
            return Some(SimdMove::BitwiseImm {
                rd,
                value: width(pattern),
                kind: if op == 1 { "bic" } else { "orr" },
                q,
            });
        }
        // MVNI: op=1 with cmode 0xx0, 10x0 or 110x.
        let inverted = op == 1 && cmode < 0b1110;
        let value = if inverted { !pattern } else { pattern };
        return Some(SimdMove::Const {
            rd,
            value: width(value),
        });
    }
    if word & 0x7F20_FC00 == 0x1E20_0000 {
        // FMOV (general): sf, type (23:22), rmode (20:19), opcode (18:16).
        let sf = word >> 31;
        let ftype = (word >> 22) & 0x3;
        let rmode = (word >> 19) & 0x3;
        let opcode = (word >> 16) & 0x7;
        let to_general = match opcode {
            6 => true,
            7 => false,
            _ => return None,
        };
        let esize: u8 = match (sf, ftype, rmode) {
            (0, 0, 0) => 4,
            (1, 1, 0) | (1, 2, 1) => 8,
            (_, 3, 0) => 2,
            _ => return None,
        };
        let upper = rmode == 1;
        return Some(match (to_general, upper) {
            (true, _) => SimdMove::ToGeneral {
                rd,
                rn,
                esize,
                index: u8::from(upper),
                signed: false,
                is_32: sf == 0,
            },
            (false, true) => SimdMove::InsGeneral {
                rd,
                rn,
                esize,
                index: 1,
            },
            (false, false) => SimdMove::FromGeneral { rd, rn, esize },
        });
    }
    let scalar_bytes = |ftype: u32| match ftype {
        0 => Some(4),
        1 => Some(8),
        3 => Some(2),
        _ => None,
    };
    if word & 0xFF3F_FC00 == 0x1E20_4000 {
        return Some(SimdMove::MovScalar {
            rd,
            rn,
            bytes: scalar_bytes((word >> 22) & 0x3)?,
        });
    }
    if word & 0xFF20_1FE0 == 0x1E20_1000 {
        let bytes: u8 = scalar_bytes((word >> 22) & 0x3)?;
        let value = vfp_expand_imm((word >> 13) & 0xFF, u32::from(bytes) * 8);
        return Some(SimdMove::Const {
            rd,
            value: u128::from(value),
        });
    }
    if word & 0x9F20_FC00 == 0x0E20_1C00 {
        let size = (word >> 22) & 0x3;
        if (word >> 29) & 1 == 0 {
            let kind = ["and", "bic", "orr", "orn"][size as usize];
            return Some(SimdMove::Bitwise {
                rd,
                rn,
                rm,
                kind,
                q,
            });
        }
        if size == 0 {
            return Some(SimdMove::Bitwise {
                rd,
                rn,
                rm,
                kind: "eor",
                q,
            });
        }
        return Some(SimdMove::BitSelect {
            rd,
            rn,
            rm,
            kind: size as u8,
            q,
        });
    }
    None
}

fn emit_simd_move(decoded: SimdMove, state: &mut DecodeState, ops: &mut Vec<Op>) {
    match decoded {
        SimdMove::DupGeneral { rd, rn, esize, q } => {
            let src = read_zr(rn, state, ops);
            ops.push(Op::VDup {
                dst: vreg(rd),
                src,
                esize,
                q,
            });
        }
        SimdMove::DupElement {
            rd,
            rn,
            esize,
            index,
            q,
        } => {
            let element = state.next_temp("lane");
            ops.push(Op::VExtract {
                dst: element.clone(),
                src: vreg(rn),
                esize,
                index,
                signed: false,
            });
            ops.push(Op::VDup {
                dst: vreg(rd),
                src: element,
                esize,
                q,
            });
        }
        SimdMove::ToGeneral {
            rd,
            rn,
            esize,
            index,
            signed,
            is_32,
        } => {
            let dst = write_zr(rd, state);
            ops.push(Op::VExtract {
                dst: dst.clone(),
                src: vreg(rn),
                esize,
                index,
                signed,
            });
            if signed && is_32 {
                zero_extend_32(&dst, state, ops);
            }
        }
        SimdMove::InsGeneral {
            rd,
            rn,
            esize,
            index,
        } => {
            let src = read_zr(rn, state, ops);
            ops.push(Op::VInsGpr {
                dst: vreg(rd),
                src,
                esize,
                index,
            });
        }
        SimdMove::FromGeneral { rd, rn, esize } => {
            let src = read_zr(rn, state, ops);
            ops.push(Op::VConst {
                dst: vreg(rd),
                lo: 0,
                hi: 0,
            });
            ops.push(Op::VInsGpr {
                dst: vreg(rd),
                src,
                esize,
                index: 0,
            });
        }
        SimdMove::InsElement {
            rd,
            rn,
            esize,
            dst_index,
            src_index,
        } => ops.push(Op::VInsLane {
            dst: vreg(rd),
            src: vreg(rn),
            esize,
            dst_index,
            src_index,
        }),
        SimdMove::Const { rd, value } => ops.push(Op::VConst {
            dst: vreg(rd),
            lo: value as u64 as i64,
            hi: (value >> 64) as u64 as i64,
        }),
        SimdMove::BitwiseImm { rd, value, kind, q } => {
            let constant = state.next_temp("vtmp");
            ops.push(Op::VConst {
                dst: constant.clone(),
                lo: value as u64 as i64,
                hi: (value >> 64) as u64 as i64,
            });
            ops.push(Op::VBitwise {
                dst: vreg(rd),
                lhs: vreg(rd),
                rhs: constant,
                kind: kind.to_string(),
                q,
            });
        }
        SimdMove::MovScalar { rd, rn, bytes } => ops.push(Op::VMov {
            dst: vreg(rd),
            src: vreg(rn),
            bytes,
        }),
        SimdMove::Bitwise {
            rd,
            rn,
            rm,
            kind,
            q,
        } => ops.push(Op::VBitwise {
            dst: vreg(rd),
            lhs: vreg(rn),
            rhs: vreg(rm),
            kind: kind.to_string(),
            q,
        }),
        SimdMove::BitSelect {
            rd,
            rn,
            rm,
            kind,
            q,
        } => {
            // BSL: Vd selects Vn/Vm. BIT: Vm selects Vn over Vd. BIF: Vm
            // selects Vd over Vn.
            let (mask, set, clear) = match kind {
                1 => (vreg(rd), vreg(rn), vreg(rm)),
                2 => (vreg(rm), vreg(rn), vreg(rd)),
                _ => (vreg(rm), vreg(rd), vreg(rn)),
            };
            ops.push(Op::VBitSelect {
                dst: vreg(rd),
                mask,
                set,
                clear,
                q,
            });
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PairMem {
    is_load: bool,
    /// LDPSW: sign-extend both 32-bit loads.
    signed: bool,
    scale: u64,
    rt: u8,
    rt2: u8,
    base: u8,
    offset: i64,
    mode: u8, // 0 = non-temporal offset, 1 = post-index, 2 = offset, 3 = pre-index
}

fn decode_pair_load_store(word: u32) -> Option<PairMem> {
    let byte = word >> 24;
    let is_load = (word >> 22) & 1 == 1;
    // Integer STP/LDP/STNP/LDNP, and LDPSW (opc=01 with L set); FP/SIMD
    // pairs (V, bit 26) are decoded separately.
    let signed = [0x68, 0x69].contains(&byte) && is_load;
    if ![0x28, 0x29, 0xa8, 0xa9].contains(&byte) && !signed {
        return None;
    }
    let scale = if (word >> 31) & 1 == 1 { 8 } else { 4 };
    let mode = (word >> 23) & 3;
    let imm7 = sign_extend(((word >> 15) & 0x7F) as i64, 7);
    Some(PairMem {
        is_load,
        signed,
        scale,
        rt: (word & 0x1F) as u8,
        rt2: ((word >> 10) & 0x1F) as u8,
        base: ((word >> 5) & 0x1F) as u8,
        offset: imm7 * scale as i64,
        mode: mode as u8,
    })
}

fn emit_pair_load_store(decoded: PairMem, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let base = reg_name(decoded.base);
    let (rt, rt2) = if decoded.is_load {
        (write_zr(decoded.rt, state), write_zr(decoded.rt2, state))
    } else {
        (
            read_zr(decoded.rt, state, ops),
            read_zr(decoded.rt2, state, ops),
        )
    };
    let sign_extend_targets = decoded.signed.then(|| (rt.clone(), rt2.clone()));
    let second = decoded.scale as i64;
    let load = |dst: String, addr: String, offset: i64| {
        if decoded.scale == 8 {
            Op::LoadI64 { dst, addr, offset }
        } else {
            Op::LoadI32 { dst, addr, offset }
        }
    };
    let store = |src: String, addr: String, offset: i64| {
        if decoded.scale == 8 {
            Op::StoreI64 { src, addr, offset }
        } else {
            Op::StoreI32 { src, addr, offset }
        }
    };
    match decoded.mode {
        0 | 2 => {
            // Offset: base unchanged.
            if decoded.is_load {
                // Both loads use the original base even when the first
                // destination is the base register (`ldp x0, x1, [x0]`).
                let base = if decoded.rt == decoded.base {
                    copy_temp(base, "addr", state, ops)
                } else {
                    base
                };
                ops.push(load(rt.clone(), base.clone(), decoded.offset));
                ops.push(load(rt2, base, decoded.offset + second));
            } else {
                ops.push(store(rt.clone(), base.clone(), decoded.offset));
                ops.push(store(rt2, base, decoded.offset + second));
            }
        }
        3 => {
            // Pre-index: update base first, access at the updated address.
            if decoded.offset != 0 {
                let imm = state.next_temp("imm");
                ops.push(Op::ConstI64 {
                    dst: imm.clone(),
                    imm: decoded.offset,
                });
                ops.push(Op::AddI64 {
                    dst: base.clone(),
                    lhs: base.clone(),
                    rhs: imm,
                });
            }
            if decoded.is_load {
                ops.push(load(rt.clone(), base.clone(), 0));
                ops.push(load(rt2, base.clone(), second));
            } else {
                ops.push(store(rt.clone(), base.clone(), 0));
                ops.push(store(rt2, base.clone(), second));
            }
        }
        _ => {
            // Post-index: access at base, then advance it.
            if decoded.is_load {
                ops.push(load(rt.clone(), base.clone(), 0));
                ops.push(load(rt2, base.clone(), second));
            } else {
                ops.push(store(rt.clone(), base.clone(), 0));
                ops.push(store(rt2, base.clone(), second));
            }
            if decoded.offset != 0 {
                let imm = state.next_temp("imm");
                ops.push(Op::ConstI64 {
                    dst: imm.clone(),
                    imm: decoded.offset,
                });
                ops.push(Op::AddI64 {
                    dst: base.clone(),
                    lhs: base.clone(),
                    rhs: imm,
                });
            }
        }
    }
    if let Some((first, second)) = sign_extend_targets {
        sign_extend_reg(&first, 32, state, ops);
        sign_extend_reg(&second, 32, state, ops);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BitfieldKind {
    /// SBFM: SBFX, SBFIZ, ASR, SXTB/SXTH/SXTW.
    Signed,
    /// BFM: BFI, BFXIL.
    Insert,
    /// UBFM: UBFX, UBFIZ, LSL, LSR, UXTB/UXTH.
    Unsigned,
}

#[derive(Debug, Clone, Copy)]
struct Bitfield {
    kind: BitfieldKind,
    dst: u8,
    src: u8,
    immr: u32,
    imms: u32,
    is_32: bool,
}

/// SBFM/BFM/UBFM: bits 28:23 = 100110 with N == sf.
fn decode_bitfield(word: u32) -> Option<Bitfield> {
    if (word >> 23) & 0x3F != 0b100110 {
        return None;
    }
    let sf = (word >> 31) & 1;
    if (word >> 22) & 1 != sf {
        return None;
    }
    let immr = (word >> 16) & 0x3F;
    let imms = (word >> 10) & 0x3F;
    if sf == 0 && (immr >= 32 || imms >= 32) {
        return None;
    }
    let kind = match (word >> 29) & 0x3 {
        0b00 => BitfieldKind::Signed,
        0b01 => BitfieldKind::Insert,
        0b10 => BitfieldKind::Unsigned,
        _ => return None,
    };
    Some(Bitfield {
        kind,
        dst: (word & 0x1F) as u8,
        src: ((word >> 5) & 0x1F) as u8,
        immr,
        imms,
        is_32: sf == 0,
    })
}

fn low_mask(len: u32) -> u64 {
    if len >= 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    }
}

fn emit_bitfield(decoded: Bitfield, state: &mut DecodeState, ops: &mut Vec<Op>) {
    // BFM into XZR has no effect.
    if decoded.kind == BitfieldKind::Insert && decoded.dst == 31 {
        return;
    }
    let width = if decoded.is_32 { 32 } else { 64 };
    let (r, s) = (decoded.immr, decoded.imms);
    // imms >= immr extracts src<s:r> to bit 0; otherwise src<s:0> is
    // deposited at bit `width - immr`.
    let (low, len, pos) = if s >= r {
        (r, s - r + 1, 0)
    } else {
        (0, s + 1, width - r)
    };
    let src = read_zr(decoded.src, state, ops);
    let field = state.next_temp("bf");
    if decoded.kind == BitfieldKind::Signed {
        // Make bit `s` the sign bit, then shift back down arithmetically.
        let up = const_temp(i64::from(63 - s), state, ops);
        ops.push(Op::LslI64 {
            dst: field.clone(),
            lhs: src,
            rhs: up,
        });
        let down = const_temp(i64::from(63 - s + low), state, ops);
        ops.push(Op::AsrI64 {
            dst: field.clone(),
            lhs: field.clone(),
            rhs: down,
        });
    } else {
        let down = const_temp(i64::from(low), state, ops);
        ops.push(Op::LsrI64 {
            dst: field.clone(),
            lhs: src,
            rhs: down,
        });
        zero_extend_bits_wide(&field, len, state, ops);
    }
    if pos > 0 {
        let up = const_temp(i64::from(pos), state, ops);
        ops.push(Op::LslI64 {
            dst: field.clone(),
            lhs: field.clone(),
            rhs: up,
        });
    }
    let dst = if decoded.kind == BitfieldKind::Insert {
        // BFM keeps the destination bits outside the field.
        let dst = reg_name(decoded.dst);
        let keep = const_temp(!(low_mask(len) << pos) as i64, state, ops);
        ops.push(Op::AndI64 {
            dst: dst.clone(),
            lhs: dst.clone(),
            rhs: keep,
        });
        ops.push(Op::OrI64 {
            dst: dst.clone(),
            lhs: dst.clone(),
            rhs: field,
        });
        dst
    } else {
        let dst = write_zr(decoded.dst, state);
        ops.push(Op::MovI64 {
            dst: dst.clone(),
            src: field,
        });
        dst
    };
    if decoded.is_32 {
        zero_extend_32(&dst, state, ops);
    }
}

/// Like `zero_extend_bits` but accepts a full 64-bit width (a no-op).
fn zero_extend_bits_wide(reg: &str, bits: u32, state: &mut DecodeState, ops: &mut Vec<Op>) {
    if bits < 64 {
        zero_extend_bits(reg, bits, state, ops);
    }
}

#[derive(Debug, Clone, Copy)]
struct Extract {
    dst: u8,
    high: u8,
    low: u8,
    lsb: u32,
    is_32: bool,
}

/// EXTR (ROR immediate when Rn == Rm): bits 30:23 = 00100111, N == sf,
/// bit 21 clear.
fn decode_extract(word: u32) -> Option<Extract> {
    if (word >> 23) & 0xFF != 0b0010_0111 || (word >> 21) & 1 != 0 {
        return None;
    }
    let sf = (word >> 31) & 1;
    let lsb = (word >> 10) & 0x3F;
    if (word >> 22) & 1 != sf || (sf == 0 && lsb >= 32) {
        return None;
    }
    Some(Extract {
        dst: (word & 0x1F) as u8,
        high: ((word >> 5) & 0x1F) as u8,
        low: ((word >> 16) & 0x1F) as u8,
        lsb,
        is_32: sf == 0,
    })
}

fn emit_extract(decoded: Extract, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let width = if decoded.is_32 { 32 } else { 64 };
    let low_src = read_zr(decoded.low, state, ops);
    let low = copy_temp(low_src, "ext", state, ops);
    if decoded.is_32 {
        zero_extend_32(&low, state, ops);
    }
    if decoded.lsb > 0 {
        let high_src = read_zr(decoded.high, state, ops);
        let high = state.next_temp("ext");
        let up = const_temp(i64::from(width - decoded.lsb), state, ops);
        ops.push(Op::LslI64 {
            dst: high.clone(),
            lhs: high_src,
            rhs: up,
        });
        let down = const_temp(i64::from(decoded.lsb), state, ops);
        ops.push(Op::LsrI64 {
            dst: low.clone(),
            lhs: low.clone(),
            rhs: down,
        });
        ops.push(Op::OrI64 {
            dst: low.clone(),
            lhs: low.clone(),
            rhs: high,
        });
    }
    let dst = write_zr(decoded.dst, state);
    ops.push(Op::MovI64 {
        dst: dst.clone(),
        src: low,
    });
    if decoded.is_32 {
        zero_extend_32(&dst, state, ops);
    }
}

#[derive(Debug, Clone, Copy)]
struct CondSelect {
    dst: u8,
    lhs: u8,
    rhs: u8,
    cond: u8,
    /// 0 CSEL, 1 CSINC, 2 CSINV, 3 CSNEG.
    op: u8,
    is_32: bool,
}

/// CSEL/CSINC/CSINV/CSNEG: bits 29:21 = 011010100, bit 11 clear. CSET,
/// CSETM, CINC, CINV and CNEG are aliases.
fn decode_cond_select(word: u32) -> Option<CondSelect> {
    if (word >> 21) & 0x1FF != 0b0_1101_0100 || (word >> 11) & 1 != 0 {
        return None;
    }
    Some(CondSelect {
        dst: (word & 0x1F) as u8,
        lhs: ((word >> 5) & 0x1F) as u8,
        rhs: ((word >> 16) & 0x1F) as u8,
        cond: ((word >> 12) & 0xF) as u8,
        op: ((((word >> 30) & 1) << 1) | ((word >> 10) & 1)) as u8,
        is_32: (word >> 31) & 1 == 0,
    })
}

fn emit_cond_select(decoded: CondSelect, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let lhs = read_zr(decoded.lhs, state, ops);
    let rm = read_zr(decoded.rhs, state, ops);
    let rhs = match decoded.op {
        0 => rm,
        1 => {
            let one = const_temp(1, state, ops);
            let value = state.next_temp("csel");
            ops.push(Op::AddI64 {
                dst: value.clone(),
                lhs: rm,
                rhs: one,
            });
            value
        }
        2 => {
            let ones = const_temp(-1, state, ops);
            let value = state.next_temp("csel");
            ops.push(Op::XorI64 {
                dst: value.clone(),
                lhs: rm,
                rhs: ones,
            });
            value
        }
        _ => {
            let zero = const_temp(0, state, ops);
            let value = state.next_temp("csel");
            ops.push(Op::SubI64 {
                dst: value.clone(),
                lhs: zero,
                rhs: rm,
            });
            value
        }
    };
    let dst = write_zr(decoded.dst, state);
    ops.push(Op::SelectI64 {
        dst: dst.clone(),
        cond: cond_code(decoded.cond).unwrap_or("al").to_string(),
        lhs,
        rhs,
    });
    if decoded.is_32 {
        zero_extend_32(&dst, state, ops);
    }
}

#[derive(Debug, Clone, Copy)]
enum CondCompareOperand {
    Imm(u8),
    Reg(u8),
}

#[derive(Debug, Clone, Copy)]
struct CondCompare {
    lhs: u8,
    rhs: CondCompareOperand,
    cond: u8,
    nzcv: u8,
    is_sub: bool,
    is_32: bool,
}

/// CCMN/CCMP (register and immediate): bits 29:21 = 111010010, bits 10 and
/// 4 clear.
fn decode_cond_compare(word: u32) -> Option<CondCompare> {
    if (word >> 21) & 0x1FF != 0b1_1101_0010 || (word >> 10) & 1 != 0 || (word >> 4) & 1 != 0 {
        return None;
    }
    let field = ((word >> 16) & 0x1F) as u8;
    Some(CondCompare {
        lhs: ((word >> 5) & 0x1F) as u8,
        rhs: if (word >> 11) & 1 == 1 {
            CondCompareOperand::Imm(field)
        } else {
            CondCompareOperand::Reg(field)
        },
        cond: ((word >> 12) & 0xF) as u8,
        nzcv: (word & 0xF) as u8,
        is_sub: (word >> 30) & 1 == 1,
        is_32: (word >> 31) & 1 == 0,
    })
}

fn emit_cond_compare(decoded: CondCompare, state: &mut DecodeState, ops: &mut Vec<Op>) {
    // Evaluate the condition on the incoming flags before the compare
    // replaces them.
    let one = const_temp(1, state, ops);
    let zero = const_temp(0, state, ops);
    let taken = state.next_temp("ccmp");
    ops.push(Op::SelectI64 {
        dst: taken.clone(),
        cond: cond_code(decoded.cond).unwrap_or("al").to_string(),
        lhs: one,
        rhs: zero.clone(),
    });
    let lhs = read_zr(decoded.lhs, state, ops);
    let rhs = match decoded.rhs {
        CondCompareOperand::Imm(imm) => const_temp(i64::from(imm), state, ops),
        CondCompareOperand::Reg(reg) => read_zr(reg, state, ops),
    };
    set_add_sub_flags(&lhs, &rhs, decoded.is_sub, decoded.is_32, state, ops);
    // NZCV = taken ? compare result : #nzcv, selected arithmetically because
    // the compare has already replaced the flags a SelectI64 would read.
    let computed = state.next_temp("ccmp");
    ops.push(Op::ReadNzcv {
        dst: computed.clone(),
    });
    let mask = state.next_temp("ccmp");
    ops.push(Op::SubI64 {
        dst: mask.clone(),
        lhs: zero,
        rhs: taken,
    });
    ops.push(Op::AndI64 {
        dst: computed.clone(),
        lhs: computed.clone(),
        rhs: mask.clone(),
    });
    let ones = const_temp(-1, state, ops);
    ops.push(Op::XorI64 {
        dst: mask.clone(),
        lhs: mask.clone(),
        rhs: ones,
    });
    let fallback = const_temp(i64::from(decoded.nzcv) << 28, state, ops);
    ops.push(Op::AndI64 {
        dst: fallback.clone(),
        lhs: fallback.clone(),
        rhs: mask,
    });
    ops.push(Op::OrI64 {
        dst: computed.clone(),
        lhs: computed.clone(),
        rhs: fallback,
    });
    ops.push(Op::WriteNzcv { src: computed });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MulKind {
    /// MADD/MSUB (MUL, MNEG).
    Full,
    /// SMADDL/SMSUBL (SMULL).
    SignedLong,
    /// UMADDL/UMSUBL (UMULL).
    UnsignedLong,
    SignedHigh,
    UnsignedHigh,
}

#[derive(Debug, Clone, Copy)]
struct MulOp {
    kind: MulKind,
    dst: u8,
    lhs: u8,
    rhs: u8,
    addend: u8,
    is_sub: bool,
    is_32: bool,
}

/// Data-processing (3 source): bits 30:24 = 0011011.
fn decode_multiply(word: u32) -> Option<MulOp> {
    if (word >> 24) & 0x7F != 0b001_1011 {
        return None;
    }
    let sf = (word >> 31) & 1;
    let is_sub = (word >> 15) & 1 == 1;
    let kind = match (sf, (word >> 21) & 0x7, is_sub) {
        (_, 0b000, _) => MulKind::Full,
        (1, 0b001, _) => MulKind::SignedLong,
        (1, 0b101, _) => MulKind::UnsignedLong,
        (1, 0b010, false) => MulKind::SignedHigh,
        (1, 0b110, false) => MulKind::UnsignedHigh,
        _ => return None,
    };
    Some(MulOp {
        kind,
        dst: (word & 0x1F) as u8,
        lhs: ((word >> 5) & 0x1F) as u8,
        rhs: ((word >> 16) & 0x1F) as u8,
        addend: ((word >> 10) & 0x1F) as u8,
        is_sub,
        is_32: sf == 0,
    })
}

/// A copy of the low 32 bits of `src`, sign- or zero-extended.
fn extended_32(src: String, signed: bool, state: &mut DecodeState, ops: &mut Vec<Op>) -> String {
    let value = copy_temp(src, "ext", state, ops);
    if signed {
        sign_extend_reg(&value, 32, state, ops);
    } else {
        zero_extend_32(&value, state, ops);
    }
    value
}

fn emit_multiply(decoded: MulOp, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let mut lhs = read_zr(decoded.lhs, state, ops);
    let mut rhs = read_zr(decoded.rhs, state, ops);
    match decoded.kind {
        MulKind::SignedHigh | MulKind::UnsignedHigh => {
            let dst = write_zr(decoded.dst, state);
            ops.push(if decoded.kind == MulKind::SignedHigh {
                Op::MulHiS64 { dst, lhs, rhs }
            } else {
                Op::MulHiU64 { dst, lhs, rhs }
            });
            return;
        }
        MulKind::SignedLong | MulKind::UnsignedLong => {
            let signed = decoded.kind == MulKind::SignedLong;
            lhs = extended_32(lhs, signed, state, ops);
            rhs = extended_32(rhs, signed, state, ops);
        }
        MulKind::Full => {}
    }
    let product = state.next_temp("mul");
    ops.push(Op::MulI64 {
        dst: product.clone(),
        lhs,
        rhs,
    });
    let addend = read_zr(decoded.addend, state, ops);
    let dst = write_zr(decoded.dst, state);
    if decoded.is_sub {
        ops.push(Op::SubI64 {
            dst: dst.clone(),
            lhs: addend,
            rhs: product,
        });
    } else {
        ops.push(Op::AddI64 {
            dst: dst.clone(),
            lhs: addend,
            rhs: product,
        });
    }
    if decoded.is_32 {
        zero_extend_32(&dst, state, ops);
    }
}

#[derive(Debug, Clone, Copy)]
struct TwoSource {
    /// 2 UDIV, 3 SDIV, 8 LSLV, 9 LSRV, 10 ASRV, 11 RORV.
    opcode: u8,
    dst: u8,
    lhs: u8,
    rhs: u8,
    is_32: bool,
}

/// Data-processing (2 source): bits 30:21 = 0011010110.
fn decode_two_source(word: u32) -> Option<TwoSource> {
    if (word >> 21) & 0x3FF != 0b00_1101_0110 {
        return None;
    }
    let opcode = ((word >> 10) & 0x3F) as u8;
    if !matches!(opcode, 2 | 3 | 8..=11) {
        return None;
    }
    Some(TwoSource {
        opcode,
        dst: (word & 0x1F) as u8,
        lhs: ((word >> 5) & 0x1F) as u8,
        rhs: ((word >> 16) & 0x1F) as u8,
        is_32: (word >> 31) & 1 == 0,
    })
}

fn emit_two_source(decoded: TwoSource, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let lhs = read_zr(decoded.lhs, state, ops);
    let rhs = read_zr(decoded.rhs, state, ops);
    let result = if decoded.opcode <= 3 {
        let signed = decoded.opcode == 3;
        let (lhs, rhs) = if decoded.is_32 {
            (
                extended_32(lhs, signed, state, ops),
                extended_32(rhs, signed, state, ops),
            )
        } else {
            (lhs, rhs)
        };
        let quotient = state.next_temp("div");
        ops.push(if signed {
            Op::SdivI64 {
                dst: quotient.clone(),
                lhs,
                rhs,
            }
        } else {
            Op::UdivI64 {
                dst: quotient.clone(),
                lhs,
                rhs,
            }
        });
        quotient
    } else {
        let width: i64 = if decoded.is_32 { 32 } else { 64 };
        // Variable shifts use Rm modulo the operand width.
        let modulus = const_temp(width - 1, state, ops);
        let by = state.next_temp("shift");
        ops.push(Op::AndI64 {
            dst: by.clone(),
            lhs: rhs,
            rhs: modulus,
        });
        let value = copy_temp(lhs, "shift", state, ops);
        if decoded.is_32 {
            match decoded.opcode {
                9 | 11 => zero_extend_32(&value, state, ops),
                10 => sign_extend_reg(&value, 32, state, ops),
                _ => {}
            }
        }
        match decoded.opcode {
            8 => ops.push(Op::LslI64 {
                dst: value.clone(),
                lhs: value.clone(),
                rhs: by,
            }),
            9 => ops.push(Op::LsrI64 {
                dst: value.clone(),
                lhs: value.clone(),
                rhs: by,
            }),
            10 => ops.push(Op::AsrI64 {
                dst: value.clone(),
                lhs: value.clone(),
                rhs: by,
            }),
            _ => emit_rotate_right(&value, &by, width, state, ops),
        }
        value
    };
    let dst = write_zr(decoded.dst, state);
    ops.push(Op::MovI64 {
        dst: dst.clone(),
        src: result,
    });
    if decoded.is_32 {
        zero_extend_32(&dst, state, ops);
    }
}

#[derive(Debug, Clone, Copy)]
struct OneSource {
    /// 0 RBIT, 1 REV16, 2 REV32 (X) / REV (W), 3 REV (X), 4 CLZ, 5 CLS.
    opcode: u8,
    dst: u8,
    src: u8,
    is_32: bool,
}

/// Data-processing (1 source): bits 30:21 = 1011010110, bits 20:16 clear.
fn decode_one_source(word: u32) -> Option<OneSource> {
    if (word >> 21) & 0x3FF != 0b10_1101_0110 || (word >> 16) & 0x1F != 0 {
        return None;
    }
    let opcode = ((word >> 10) & 0x3F) as u8;
    let is_32 = (word >> 31) & 1 == 0;
    if opcode > 5 || (is_32 && opcode == 3) {
        return None;
    }
    Some(OneSource {
        opcode,
        dst: (word & 0x1F) as u8,
        src: ((word >> 5) & 0x1F) as u8,
        is_32,
    })
}

fn emit_one_source(decoded: OneSource, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let src = read_zr(decoded.src, state, ops);
    let value = copy_temp(src, "un", state, ops);
    if decoded.is_32 {
        if decoded.opcode == 5 {
            sign_extend_reg(&value, 32, state, ops);
        } else {
            zero_extend_32(&value, state, ops);
        }
    }
    let shift_right = |by: i64, state: &mut DecodeState, ops: &mut Vec<Op>| {
        let by = const_temp(by, state, ops);
        ops.push(Op::LsrI64 {
            dst: value.clone(),
            lhs: value.clone(),
            rhs: by,
        });
    };
    let subtract = |amount: i64, state: &mut DecodeState, ops: &mut Vec<Op>| {
        let amount = const_temp(amount, state, ops);
        ops.push(Op::SubI64 {
            dst: value.clone(),
            lhs: value.clone(),
            rhs: amount,
        });
    };
    match (decoded.opcode, decoded.is_32) {
        (0, _) => {
            ops.push(Op::RbitI64 {
                dst: value.clone(),
                src: value.clone(),
            });
            if decoded.is_32 {
                shift_right(32, state, ops);
            }
        }
        (1, _) => {
            // Swap the bytes of every halfword.
            let high = state.next_temp("un");
            let eight = const_temp(8, state, ops);
            ops.push(Op::LsrI64 {
                dst: high.clone(),
                lhs: value.clone(),
                rhs: eight.clone(),
            });
            let mask = const_temp(0x00FF_00FF_00FF_00FF, state, ops);
            ops.push(Op::AndI64 {
                dst: high.clone(),
                lhs: high.clone(),
                rhs: mask.clone(),
            });
            ops.push(Op::AndI64 {
                dst: value.clone(),
                lhs: value.clone(),
                rhs: mask,
            });
            ops.push(Op::LslI64 {
                dst: value.clone(),
                lhs: value.clone(),
                rhs: eight,
            });
            ops.push(Op::OrI64 {
                dst: value.clone(),
                lhs: value.clone(),
                rhs: high,
            });
        }
        (2, true) => {
            ops.push(Op::RevI64 {
                dst: value.clone(),
                src: value.clone(),
            });
            shift_right(32, state, ops);
        }
        (2, false) => {
            // REV32: reverse the bytes of each word, i.e. a full byte
            // reverse followed by swapping the two words.
            ops.push(Op::RevI64 {
                dst: value.clone(),
                src: value.clone(),
            });
            let thirty_two = const_temp(32, state, ops);
            emit_rotate_right(&value, &thirty_two, 64, state, ops);
        }
        (3, _) => ops.push(Op::RevI64 {
            dst: value.clone(),
            src: value.clone(),
        }),
        (4, _) => {
            ops.push(Op::ClzI64 {
                dst: value.clone(),
                src: value.clone(),
            });
            if decoded.is_32 {
                subtract(32, state, ops);
            }
        }
        _ => {
            // CLS(x) = CLZ((x ^ (x << 1)) >> 1) - 1 over 64 bits; a W source
            // was sign-extended above and has 32 extra sign bits.
            let mixed = state.next_temp("un");
            let one = const_temp(1, state, ops);
            ops.push(Op::LslI64 {
                dst: mixed.clone(),
                lhs: value.clone(),
                rhs: one.clone(),
            });
            ops.push(Op::XorI64 {
                dst: mixed.clone(),
                lhs: mixed.clone(),
                rhs: value.clone(),
            });
            ops.push(Op::LsrI64 {
                dst: mixed.clone(),
                lhs: mixed.clone(),
                rhs: one,
            });
            ops.push(Op::ClzI64 {
                dst: value.clone(),
                src: mixed,
            });
            subtract(if decoded.is_32 { 33 } else { 1 }, state, ops);
        }
    }
    let dst = write_zr(decoded.dst, state);
    ops.push(Op::MovI64 {
        dst: dst.clone(),
        src: value,
    });
    if decoded.is_32 {
        zero_extend_32(&dst, state, ops);
    }
}

#[derive(Debug, Clone, Copy)]
struct Exclusive {
    size: u8,
    is_load: bool,
    pair: bool,
    /// False for LDAR/STLR, which order an access rather than reserving it.
    exclusive: bool,
    /// Rs, the success register of a store-exclusive.
    status: Option<u8>,
    rt: u8,
    rt2: u8,
    base: u8,
}

/// Load/store exclusive (LDXR/LDAXR/STXR/STLXR and pairs) and
/// load-acquire/store-release (LDAR/STLR): bits 29:24 = 001000. Bit 23 is set
/// for the acquire/release forms, which do not reserve; bit 21 marks the pair
/// forms. The exclusive pair becomes a reservation and a compare-and-swap, so
/// guest threads that run at the same time contend for the same lock as they do
/// on hardware. LDAR/STLR are plain accesses with an acquire fence after the
/// load or a release fence before the store: free on x86-64, whose total store
/// order already orders them, and real barriers on an AArch64 host.
fn decode_exclusive(word: u32) -> Option<Exclusive> {
    if (word >> 24) & 0x3F != 0b00_1000 {
        return None;
    }
    let size = ((word >> 30) & 0x3) as u8;
    let is_load = (word >> 22) & 1 == 1;
    let rs = ((word >> 16) & 0x1F) as u8;
    let (pair, exclusive) = match ((word >> 23) & 1, (word >> 21) & 1) {
        (0, 0) => (false, true),
        (0, 1) if size >= 2 => (true, true),
        (1, 0) => (false, false),
        _ => return None,
    };
    Some(Exclusive {
        size,
        is_load,
        pair,
        exclusive,
        status: (exclusive && !is_load).then_some(rs),
        rt: (word & 0x1F) as u8,
        rt2: ((word >> 10) & 0x1F) as u8,
        base: ((word >> 5) & 0x1F) as u8,
    })
}

fn emit_exclusive(decoded: Exclusive, state: &mut DecodeState, ops: &mut Vec<Op>) {
    // Copy the base so a load into Rt cannot disturb the second access.
    let addr = copy_temp(reg_name(decoded.base), "addr", state, ops);

    // LDAR and STLR reserve nothing: they are a plain access with ordering.
    // An AArch64 host reorders plain accesses as the guest's hardware would
    // for plain ones, so the ordering is a fence of its own (LM3's job queue
    // read a null job without it once its accesses went direct on a phone).
    if !decoded.exclusive {
        if decoded.is_load {
            emit_mem_access(MemKind::Load, decoded.size, decoded.rt, addr, 0, state, ops);
            ops.push(Op::AcquireFence);
        } else {
            ops.push(Op::ReleaseFence);
            emit_mem_access(
                MemKind::Store,
                decoded.size,
                decoded.rt,
                addr,
                0,
                state,
                ops,
            );
        }
        return;
    }

    let size = 1u64 << decoded.size;

    if decoded.is_load {
        ops.push(Op::ExclusiveLoad {
            dst: write_zr(decoded.rt, state),
            dst2: decoded.pair.then(|| write_zr(decoded.rt2, state)),
            addr,
            size,
            pair: decoded.pair,
            offset: 0,
        });
        return;
    }

    // Rt is the value to store, Rs the success flag. Both are read before the
    // flag is written, as the hardware does.
    let src = read_zr(decoded.rt, state, ops);
    let src2 = decoded.pair.then(|| read_zr(decoded.rt2, state, ops));
    ops.push(Op::ExclusiveStore {
        src,
        src2,
        status: write_zr(decoded.status.unwrap_or(31), state),
        addr,
        size,
        pair: decoded.pair,
        offset: 0,
    });
}

/// The system class with CRn = 3 holds the barriers and CLREX; bits 7:5 say
/// which: 2 is CLREX, 5 DMB, 6 DSB, 7 ISB. Guest threads run on host threads
/// now, so a data barrier is a real fence and CLREX drops the reservation.
/// A hint (CRn = 0) and the other system instructions still do nothing.
fn decode_barrier(word: u32) -> Option<Op> {
    if word & 0xFFFF_F000 != 0xD503_3000 {
        return None;
    }
    Some(match (word >> 5) & 7 {
        2 => Op::ClearExclusive,
        5..=7 => Op::MemoryFence,
        _ => return None,
    })
}

/// HINT (NOP, YIELD, WFE, ...) has no effect on lifted code. The barriers and
/// CLREX in the same class are handled by [`decode_barrier`].
fn is_hint_or_barrier(word: u32) -> bool {
    let masked = word & 0xFFFF_F01F;
    masked == 0xD503_201F || masked == 0xD503_301F
}

/// DC and IC cache maintenance (SYS with CRn = 7) has no effect on lifted
/// code, except DC ZVA, which zeroes memory.
fn is_cache_maintenance(word: u32) -> bool {
    let is_dc_zva = word & 0xFFFF_FFE0 == 0xD50B_7420;
    word & 0xFFF8_F000 == 0xD508_7000 && !is_dc_zva
}

fn is_blr(word: u32) -> bool {
    (word & 0xFFFFFC1F) == 0xD63F0000
}

fn decode_branch_imm(word: u32, pc: u64) -> Option<u64> {
    if (word & 0xFC000000) != 0x14000000 {
        return None;
    }
    let imm26 = (word & 0x03FFFFFF) as i64;
    let offset = sign_extend(imm26 << 2, 28);
    Some((pc as i64 + offset) as u64)
}

fn decode_branch_link(word: u32, pc: u64) -> Option<u64> {
    if (word & 0xFC000000) != 0x94000000 {
        return None;
    }
    let imm26 = (word & 0x03FFFFFF) as i64;
    let offset = sign_extend(imm26 << 2, 28);
    Some((pc as i64 + offset) as u64)
}

fn decode_branch_cond(word: u32, pc: u64) -> Result<Option<(String, u64)>, String> {
    if (word & 0xFF000010) != 0x54000000 {
        return Ok(None);
    }
    let cond = (word & 0xF) as u8;
    let cond_str = cond_code(cond).ok_or_else(|| format!("unsupported condition {cond}"))?;
    let imm19 = ((word >> 5) & 0x7FFFF) as i64;
    let offset = sign_extend(imm19 << 2, 21);
    Ok(Some((cond_str.to_string(), (pc as i64 + offset) as u64)))
}

/// CBZ/CBNZ: (is CBNZ, Rt, is the W form, target).
fn decode_cbz(word: u32, pc: u64) -> Result<Option<(bool, u8, bool, u64)>, String> {
    let top = word & 0x7F000000;
    let is_cbz = top == 0x34000000 || top == 0xB4000000;
    let is_cbnz = top == 0x35000000 || top == 0xB5000000;
    if !is_cbz && !is_cbnz {
        return Ok(None);
    }
    let imm19 = ((word >> 5) & 0x7FFFF) as i64;
    let offset = sign_extend(imm19 << 2, 21);
    let rt = (word & 0x1F) as u8;
    let is_32 = word >> 31 == 0;
    Ok(Some((is_cbnz, rt, is_32, (pc as i64 + offset) as u64)))
}

fn decode_tbz(word: u32, pc: u64) -> Result<Option<(bool, u8, u8, u64)>, String> {
    let top = word & 0x7F000000;
    let is_tbz = top == 0x36000000 || top == 0xB6000000;
    let is_tbnz = top == 0x37000000 || top == 0xB7000000;
    if !is_tbz && !is_tbnz {
        return Ok(None);
    }
    let b5 = ((word >> 31) & 0x1) as u8;
    let b40 = ((word >> 19) & 0x1F) as u8;
    let bit = (b5 << 5) | b40;
    let imm14 = ((word >> 5) & 0x3FFF) as i64;
    let offset = sign_extend(imm14 << 2, 16);
    let rt = (word & 0x1F) as u8;
    Ok(Some((is_tbnz, rt, bit, (pc as i64 + offset) as u64)))
}

fn decode_br_register(word: u32) -> Option<u8> {
    if (word & 0xFFFFFC1F) != 0xD61F0000 {
        return None;
    }
    Some(((word >> 5) & 0x1F) as u8)
}

fn is_ret(word: u32) -> bool {
    (word & 0xFFFFFC1F) == 0xD65F0000
}

/// BL/BLR write the return address to X30. Lifted calls return through the
/// host stack, but guest code also reads LR as data (rtld's PC-relative
/// `bl next; .word offset; ldr w0, [x30]` idiom, return-address queries).
fn set_link_register(pc: u64, ops: &mut Vec<Op>) {
    ops.push(Op::ConstI64 {
        dst: reg_name(30),
        imm: pc.wrapping_add(4) as i64,
    });
}

/// SVC #imm16: 11010100 000 imm16 000 01.
fn svc_imm(word: u32) -> Option<u64> {
    if word & 0xFFE0_001F == 0xD400_0001 {
        Some((word >> 5) as u64 & 0xFFFF)
    } else {
        None
    }
}

fn cond_code(code: u8) -> Option<&'static str> {
    match code {
        0x0 => Some("eq"),
        0x1 => Some("ne"),
        0x2 => Some("cs"),
        0x3 => Some("cc"),
        0x4 => Some("mi"),
        0x5 => Some("pl"),
        0x6 => Some("vs"),
        0x7 => Some("vc"),
        0x8 => Some("hi"),
        0x9 => Some("ls"),
        0xA => Some("ge"),
        0xB => Some("lt"),
        0xC => Some("gt"),
        0xD => Some("le"),
        0xE => Some("al"),
        _ => None,
    }
}

fn sign_extend(value: i64, bits: u8) -> i64 {
    let shift = 64 - bits as i64;
    (value << shift) >> shift
}

fn zero_extend_32(dst: &str, state: &mut DecodeState, ops: &mut Vec<Op>) {
    let mask = state.next_temp("imm");
    ops.push(Op::ConstI64 {
        dst: mask.clone(),
        imm: 0xFFFF_FFFF,
    });
    ops.push(Op::AndI64 {
        dst: dst.to_string(),
        lhs: dst.to_string(),
        rhs: mask,
    });
}

fn reg_name(reg: u8) -> String {
    format!("x{reg}")
}

/// A floating-point or SIMD instruction left to the run-time evaluator.
/// General register 31 is the zero register here: read as 0, written to
/// nowhere.
fn simd_op(word: u32, operands: &recomp_simd::Operands) -> Op {
    let general = |reg: Option<u8>| reg.filter(|reg| *reg != 31).map(reg_name);
    Op::Simd {
        word,
        vector_reads: operands
            .vector_reads
            .iter()
            .map(|reg| format!("v{reg}"))
            .collect(),
        general_read: general(operands.general_read),
        dst: operands
            .vector_write
            .map(|reg| format!("v{reg}"))
            .or_else(|| general(operands.general_write)),
        reads_flags: operands.reads_flags,
        writes_flags: operands.writes_flags,
    }
}

/// Read a register operand where encoding 31 means XZR/WZR (data-processing
/// register forms). Loads/stores use `reg_name` for base registers because
/// encoding 31 there means SP.
fn read_zr(reg: u8, state: &mut DecodeState, ops: &mut Vec<Op>) -> String {
    if reg == 31 {
        let zero = state.next_temp("zr");
        ops.push(Op::ConstI64 {
            dst: zero.clone(),
            imm: 0,
        });
        zero
    } else {
        reg_name(reg)
    }
}

/// Destination for data-processing register forms: encoding 31 discards the
/// result (XZR), never SP.
fn write_zr(reg: u8, state: &mut DecodeState) -> String {
    if reg == 31 {
        state.next_temp("zr_dst")
    } else {
        reg_name(reg)
    }
}

fn block_label(addr: u64) -> String {
    format!("bb_{addr:016x}")
}

fn pop_first(set: &mut BTreeSet<u64>) -> Option<u64> {
    let first = set.iter().next().copied();
    if let Some(value) = first {
        set.remove(&value);
    }
    first
}

fn resolve_segment_path(base_dir: &Path, segment: &ModuleSegment) -> PathBuf {
    let path = Path::new(&segment.output_path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn ensure_dir(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path).map_err(|err| err.to_string())
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map_err(|err| err.to_string())
            .map(|cwd| cwd.join(path))
    }
}
