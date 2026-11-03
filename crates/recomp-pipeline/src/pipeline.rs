use crate::config::{PerformanceMode, StubBehavior, TitleConfig};
use crate::homebrew::ModuleJson;
use crate::input::{Block, Function, Module, Op, Terminator};
use crate::memory::{
    MemoryImageDescriptor, MemoryInitSegmentDescriptor, MemoryLayoutDescriptor,
    MemoryZeroSegmentDescriptor,
};
use crate::output::{emit_project, BuildManifest, GeneratedFile, InputSummary};
use crate::provenance::{ProvenanceManifest, ValidatedInput};
use pathdiff::diff_paths;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct PipelineOptions {
    pub module_path: PathBuf,
    pub config_path: PathBuf,
    pub provenance_path: PathBuf,
    pub out_dir: PathBuf,
    pub runtime_path: PathBuf,
}

#[derive(Debug)]
pub struct PipelineReport {
    pub out_dir: PathBuf,
    pub files_written: Vec<PathBuf>,
    pub detected_inputs: Vec<ValidatedInput>,
}

#[derive(Debug)]
pub struct MemoryImageSource {
    pub source_path: PathBuf,
    pub dest_path: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("config error: {0}")]
    Config(String),
    #[error("module error: {0}")]
    Module(String),
    #[error("provenance error: {0}")]
    Provenance(String),
    #[error("emit error: {0}")]
    Emit(String),
}

pub fn run_pipeline(options: PipelineOptions) -> Result<PipelineReport, PipelineError> {
    let module_path = absolute_path(&options.module_path)?;
    let config_path = absolute_path(&options.config_path)?;
    let provenance_path = absolute_path(&options.provenance_path)?;
    let out_dir = absolute_path(&options.out_dir)?;
    let runtime_path = absolute_path(&options.runtime_path)?;

    let module_src = fs::read_to_string(&module_path)?;
    let module_hash = sha256_hex(&module_src);
    let parsed = parse_module_source(&module_src)?;
    // Lifted retail modules are gigabytes of JSON; free the text early.
    drop(module_src);
    let mut module = match parsed {
        ModuleSource::Lifted(module) => {
            module.validate_arch().map_err(PipelineError::Module)?;
            module
        }
        ModuleSource::Homebrew(module_json) => {
            return Err(PipelineError::Module(format!(
                "homebrew module.json detected (schema_version={}, module_type={}). Run the lifter to produce a lifted module.json before translation.",
                module_json.schema_version, module_json.module_type
            )));
        }
    };

    let config_src = fs::read_to_string(&config_path)?;
    let mut config = TitleConfig::parse(&config_src).map_err(PipelineError::Config)?;
    // Defaults for user-supplied files are resolved against the config file,
    // wherever the program is started from; the path as configured stays
    // available for a copy placed beside the executable.
    let config_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    if let Some(script) = config.runtime.script.take() {
        config.runtime.script = Some(script.resolve(config_dir));
    }
    if let Some(ipc) = config.runtime.ipc.as_mut() {
        if let Some(image) = ipc.romfs_image.take() {
            ipc.romfs_image = Some(image.resolve(config_dir));
        }
    }

    let provenance_src = fs::read_to_string(&provenance_path)?;
    let provenance =
        ProvenanceManifest::parse(&provenance_src).map_err(PipelineError::Provenance)?;
    let provenance_validation = provenance
        .validate(&provenance_path, &provenance_src)
        .map_err(PipelineError::Provenance)?;
    if !provenance_validation
        .inputs
        .iter()
        .any(|input| input.path == module_path)
    {
        return Err(PipelineError::Provenance(format!(
            "module input is not listed in provenance metadata: {}",
            module_path.display()
        )));
    }

    let (memory_image, memory_sources) = build_memory_image(&module, &module_path)?;
    let program = translate_module(&mut module, &config, memory_image.clone())?;

    let config_hash = sha256_hex(&config_src);
    let provenance_hash = provenance_validation.manifest_sha256.clone();

    let runtime_rel = runtime_relative_path(&runtime_path, &out_dir);
    let inputs = provenance_validation
        .inputs
        .iter()
        .map(|input| InputSummary {
            path: input.path.clone(),
            format: input.format.as_str().to_string(),
            sha256: input.sha256.clone(),
            size: input.size,
            role: input.role.clone(),
        })
        .collect::<Vec<_>>();
    let manifest = BuildManifest {
        title: program.title.clone(),
        abi_version: program.abi_version.clone(),
        module_sha256: module_hash,
        config_sha256: config_hash,
        provenance_sha256: provenance_hash,
        inputs,
        memory_layout: program.memory_layout.clone(),
        memory_image: program.memory_image.clone(),
        manifest_self_hash_basis: String::new(),
        generated_files: Vec::<GeneratedFile>::new(),
    };

    let (files_written, _manifest) =
        emit_project(&out_dir, &runtime_rel, &program, &manifest, &memory_sources)
            .map_err(PipelineError::Emit)?;

    Ok(PipelineReport {
        out_dir,
        files_written,
        detected_inputs: provenance_validation.inputs,
    })
}

#[derive(Debug)]
enum ModuleSource {
    Lifted(Module),
    Homebrew(ModuleJson),
}

fn parse_module_source(module_src: &str) -> Result<ModuleSource, PipelineError> {
    // Probe the top-level keys without building a JSON tree: a generic
    // `Value` of a multi-gigabyte lifted module needs several times its size.
    #[derive(serde::Deserialize)]
    struct Probe {
        schema_version: Option<serde::de::IgnoredAny>,
        module_type: Option<serde::de::IgnoredAny>,
        modules: Option<serde::de::IgnoredAny>,
    }
    let probe: Probe = serde_json::from_str(module_src)
        .map_err(|err| PipelineError::Module(format!("invalid module JSON: {err}")))?;
    if probe.schema_version.is_some() && probe.module_type.is_some() && probe.modules.is_some() {
        let module_json: ModuleJson = serde_json::from_str(module_src)
            .map_err(|err| PipelineError::Module(format!("invalid homebrew module JSON: {err}")))?;
        return Ok(ModuleSource::Homebrew(module_json));
    }
    let module: Module = serde_json::from_str(module_src)
        .map_err(|err| PipelineError::Module(format!("invalid module JSON: {err}")))?;
    Ok(ModuleSource::Lifted(module))
}

fn build_memory_image(
    module: &Module,
    module_path: &Path,
) -> Result<(Option<MemoryImageDescriptor>, Vec<MemoryImageSource>), PipelineError> {
    if module.segments.is_empty() {
        return Ok((None, Vec::new()));
    }

    validate_module_segments(&module.segments)?;

    let base_dir = module_path.parent().unwrap_or_else(|| Path::new("."));
    let mut descriptor = MemoryImageDescriptor::empty();
    let mut sources = Vec::new();

    for (index, segment) in module.segments.iter().enumerate() {
        let mut wrote_init = false;
        if let Some(init_path) = &segment.init_path {
            let source_path = base_dir.join(init_path);
            let metadata = fs::metadata(&source_path)?;
            let file_size = metadata.len();
            let init_size = segment.init_size.unwrap_or(file_size);
            if init_size != file_size {
                return Err(PipelineError::Module(format!(
                    "segment {} init_size {} does not match file size {}",
                    segment.name, init_size, file_size
                )));
            }
            if init_size > segment.size {
                return Err(PipelineError::Module(format!(
                    "segment {} init_size {} exceeds segment size {}",
                    segment.name, init_size, segment.size
                )));
            }
            let name = sanitize_segment_name(&segment.name, index);
            let dest_path = PathBuf::from(format!("segments/{name}.bin"));
            descriptor.init_segments.push(MemoryInitSegmentDescriptor {
                name: segment.name.clone(),
                base: segment.base,
                size: init_size,
                init_path: dest_path.to_string_lossy().to_string(),
            });
            sources.push(MemoryImageSource {
                source_path,
                dest_path,
            });
            wrote_init = true;

            if segment.zero_fill && segment.size > init_size {
                descriptor.zero_segments.push(MemoryZeroSegmentDescriptor {
                    name: format!("{}_bss", segment.name),
                    base: segment.base + init_size,
                    size: segment.size - init_size,
                });
            }
        }

        if segment.zero_fill && !wrote_init {
            descriptor.zero_segments.push(MemoryZeroSegmentDescriptor {
                name: segment.name.clone(),
                base: segment.base,
                size: segment.size,
            });
        }
    }

    if descriptor.is_empty() {
        return Ok((None, Vec::new()));
    }

    Ok((Some(descriptor), sources))
}

fn validate_module_segments(segments: &[crate::input::ModuleSegment]) -> Result<(), PipelineError> {
    let mut ranges = Vec::with_capacity(segments.len());
    for segment in segments {
        let end = segment.base.checked_add(segment.size).ok_or_else(|| {
            PipelineError::Module(format!("segment {} overflows address space", segment.name))
        })?;
        ranges.push((segment.name.as_str(), segment.base, end));
    }

    ranges.sort_by_key(|range| range.1);
    for window in ranges.windows(2) {
        let left = window[0];
        let right = window[1];
        if left.2 > right.1 {
            return Err(PipelineError::Module(format!(
                "segments {} and {} overlap",
                left.0, right.0
            )));
        }
    }

    Ok(())
}

fn sanitize_segment_name(name: &str, index: usize) -> String {
    let mut sanitized: String = name
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => ch,
            _ => '-',
        })
        .collect();
    if sanitized.trim_matches('-').is_empty() {
        sanitized = "segment".to_string();
    }
    format!("{sanitized}-{index}")
}

fn translate_module(
    module: &mut Module,
    config: &TitleConfig,
    memory_image: Option<MemoryImageDescriptor>,
) -> Result<RustProgram, PipelineError> {
    let entry = config.entry.clone();
    let mut functions = Vec::new();
    // Consume the parsed functions so each is freed once translated.
    for function in std::mem::take(&mut module.functions) {
        functions.push(translate_function(&function, config)?);
    }

    Ok(RustProgram {
        title: config.title.clone(),
        abi_version: config.abi_version.clone(),
        entry,
        functions,
        performance_mode: config.runtime.performance_mode,
        graphics: config.runtime.graphics,
        parallel: config.runtime.parallel,
        block_tracking: config.runtime.block_tracking,
        target_cpu: config.runtime.target_cpu.clone(),
        target: config.runtime.target.clone(),
        guest_context: config.runtime.guest_context,
        direct_registers: config.runtime.direct_registers,
        script: config.runtime.script.clone(),
        memory_layout: config.memory_layout.clone(),
        memory_image,
        ipc: config.runtime.ipc.clone(),
        initial_sp: config.runtime.initial_sp,
        entry_address: module.entry_address,
    })
}

fn translate_function(
    function: &Function,
    config: &TitleConfig,
) -> Result<RustFunction, PipelineError> {
    if !function.blocks.is_empty() {
        translate_block_function(function, config)
    } else if !function.ops.is_empty() {
        translate_linear_function(function, config)
    } else {
        Err(PipelineError::Module(format!(
            "function {} has no ops or blocks",
            function.name
        )))
    }
}

/// Temps assigned exactly once, by a constant, mapped to the Rust literal
/// that replaces them. Lifted code materializes every immediate into a temp;
/// inlining them removes an assignment and a declaration per immediate.
fn constant_temps(function: &Function) -> std::collections::HashMap<&str, String> {
    let mut defs: std::collections::HashMap<&str, (usize, Option<i64>)> =
        std::collections::HashMap::new();
    for op in function.blocks.iter().flat_map(|block| &block.ops) {
        if let Some(dst) = op_dst(op) {
            let entry = defs.entry(dst).or_insert((0, None));
            entry.0 += 1;
            if let Op::ConstI64 { imm, .. } = op {
                entry.1 = Some(*imm);
            }
        }
    }
    defs.into_iter()
        .filter_map(|(name, (count, imm))| match imm {
            Some(imm) if count == 1 && reg_index(name).is_none() => {
                let literal = if imm == i64::MIN {
                    "i64::MIN".to_string()
                } else {
                    format!("({imm}i64)")
                };
                Some((name, literal))
            }
            _ => None,
        })
        .collect()
}

/// The register an op writes, if any.
fn op_dst(op: &Op) -> Option<&str> {
    match op {
        Op::ConstI64 { dst, .. }
        | Op::AddI64 { dst, .. }
        | Op::MovI64 { dst, .. }
        | Op::SubI64 { dst, .. }
        | Op::AndI64 { dst, .. }
        | Op::OrI64 { dst, .. }
        | Op::XorI64 { dst, .. }
        | Op::LslI64 { dst, .. }
        | Op::LsrI64 { dst, .. }
        | Op::AsrI64 { dst, .. }
        | Op::MulI64 { dst, .. }
        | Op::MulHiU64 { dst, .. }
        | Op::MulHiS64 { dst, .. }
        | Op::UdivI64 { dst, .. }
        | Op::SdivI64 { dst, .. }
        | Op::SelectI64 { dst, .. }
        | Op::ClzI64 { dst, .. }
        | Op::RbitI64 { dst, .. }
        | Op::RevI64 { dst, .. }
        | Op::PcRel { dst, .. }
        | Op::LoadI8 { dst, .. }
        | Op::LoadI16 { dst, .. }
        | Op::LoadI32 { dst, .. }
        | Op::LoadI64 { dst, .. }
        | Op::ExclusiveLoad { dst, .. }
        | Op::ExclusiveStore { status: dst, .. }
        | Op::ReadSysReg { dst, .. }
        | Op::VConst { dst, .. }
        | Op::VLoad { dst, .. }
        | Op::VMov { dst, .. }
        | Op::VDup { dst, .. }
        | Op::VInsGpr { dst, .. }
        | Op::VExtract { dst, .. }
        | Op::VInsLane { dst, .. }
        | Op::VBitwise { dst, .. }
        | Op::VBitSelect { dst, .. }
        | Op::ReadNzcv { dst } => Some(dst),
        Op::Simd { dst, .. } => dst.as_deref(),
        Op::AddCarry { dst, .. } => Some(dst),
        Op::CmpI64 { .. }
        | Op::CmnI64 { .. }
        | Op::TestI64 { .. }
        | Op::StoreI8 { .. }
        | Op::StoreI16 { .. }
        | Op::StoreI32 { .. }
        | Op::StoreI64 { .. }
        | Op::VStore { .. }
        | Op::Br { .. }
        | Op::BrCond { .. }
        | Op::Call { .. }
        | Op::Syscall { .. }
        | Op::WriteSysReg { .. }
        | Op::WriteNzcv { .. }
        | Op::ClearExclusive
        | Op::MemoryFence
        | Op::AcquireFence
        | Op::ReleaseFence
        | Op::Ret => None,
    }
}

/// Replace whole identifiers in `line` that appear in `map`.
fn substitute_idents(line: &str, map: &std::collections::HashMap<&str, String>) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(start) = rest.find(|ch: char| ch == '_' || ch.is_ascii_alphabetic()) {
        out.push_str(&rest[..start]);
        let token = &rest[start..];
        let len = token
            .find(|ch: char| !(ch == '_' || ch.is_ascii_alphanumeric()))
            .unwrap_or(token.len());
        let ident = &token[..len];
        // Identifier characters glued to a preceding digit belong to a
        // literal suffix such as `8u64`, not to a separate identifier.
        let after_digit = out.chars().last().is_some_and(|ch| ch.is_ascii_digit());
        match map.get(ident) {
            Some(literal) if !after_digit => out.push_str(literal),
            _ => out.push_str(ident),
        }
        rest = &token[len..];
    }
    out.push_str(rest);
    out
}

fn track_reg(regs: &mut Vec<String>, name: &str) {
    if !regs.iter().any(|item| item == name) {
        regs.push(name.to_string());
    }
}

/// The packed NZCV byte of the `flag_n`..`flag_v` locals.
const LOCAL_FLAGS_BYTE: &str =
    "((flag_n as u8) << 3) | ((flag_z as u8) << 2) | ((flag_c as u8) << 1) | (flag_v as u8)";

/// How op lowering names guest registers.
///
/// The snapshot ABI keeps every live register in a function local and copies
/// the set through the context around calls. Direct mode makes each guest
/// register read and write address the function's `(*ctx)` fields instead, so
/// the host compiler allocates them and there is no prologue snapshot and no
/// call spill or reload. Instruction temporaries and constant-folded values
/// stay locals in both modes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RegisterAccess {
    direct: bool,
}

impl RegisterAccess {
    pub(crate) fn locals() -> Self {
        Self { direct: false }
    }

    pub(crate) fn direct() -> Self {
        Self { direct: true }
    }

    pub(crate) fn is_direct(self) -> bool {
        self.direct
    }

    /// A guest register read expression (a local name, or the context field).
    fn read(self, reg: &str) -> String {
        if self.direct {
            if let Some(index) = reg_index(reg) {
                return format!("(unsafe {{ (*ctx).x[{index}] }})");
            }
            if let Some(index) = vreg_index(reg) {
                return format!("(unsafe {{ (*ctx).v[{index}] }})");
            }
        }
        reg.to_string()
    }

    /// A generated statement that assigns `expr` to guest register `dst`.
    fn assign(self, dst: &str, expr: &str) -> String {
        if self.direct {
            if let Some(index) = reg_index(dst) {
                return format!("unsafe {{ (*ctx).x[{index}] = {expr}; }}");
            }
            if let Some(index) = vreg_index(dst) {
                return format!("unsafe {{ (*ctx).v[{index}] = {expr}; }}");
            }
        }
        format!("{dst} = {expr};")
    }

    /// A generated statement that replaces NZCV with a packed byte expression
    /// (`n << 3 | z << 2 | c << 1 | v`).
    fn assign_nzcv(self, expr: &str) -> String {
        if self.direct {
            format!("unsafe {{ (*ctx).nzcv = {expr}; }}")
        } else {
            format!(
                "let __recomp_nzcv = {expr}; flag_n = __recomp_nzcv & 8 != 0; flag_z = __recomp_nzcv & 4 != 0; flag_c = __recomp_nzcv & 2 != 0; flag_v = __recomp_nzcv & 1 != 0;"
            )
        }
    }

    /// A condition operand for one of the four flags.
    fn flag(self, name: char) -> String {
        if self.direct {
            let bit = match name {
                'n' => 8,
                'z' => 4,
                'c' => 2,
                'v' => 1,
                other => unreachable!("unknown flag {other}"),
            };
            format!("((unsafe {{ (*ctx).nzcv }} & {bit}) != 0)")
        } else {
            format!("flag_{name}")
        }
    }

    /// The packed NZCV byte expression.
    fn flags_byte(self) -> String {
        if self.direct {
            "(unsafe { (*ctx).nzcv })".to_string()
        } else {
            LOCAL_FLAGS_BYTE.to_string()
        }
    }
}

fn render_syscall(
    name: &str,
    behavior: StubBehavior,
    args: &[String],
    access: RegisterAccess,
) -> String {
    let args_list = args
        .iter()
        .map(|arg| access.read(arg))
        .collect::<Vec<_>>()
        .join(", ");
    if let Some(id) = recomp_services::kernel_svc_id(name) {
        // Resolve the name here; only the numeric id reaches the runtime. The
        // runtime reports which output registers each SVC writes, so the
        // generated code does not bake in the kernel's register table.
        return format!(
            "{{ let out = recomp_abi::svc_dispatch_id(0x{id:02x}u64, &[{args_list}])?; {} if let Some(value) = out.x1 {{ {} }} if let Some(value) = out.x2 {{ {} }} }}",
            access.assign("x0", "out.x0"),
            access.assign("x1", "value"),
            access.assign("x2", "value")
        );
    }
    match behavior {
        StubBehavior::Log => {
            format!("recomp_abi::syscall_stub(\"log\", \"{name}\", &[{args_list}])?;")
        }
        StubBehavior::Noop => {
            format!("recomp_abi::syscall_stub(\"noop\", \"{name}\", &[{args_list}])?;")
        }
        StubBehavior::Panic => {
            format!("recomp_abi::syscall_stub(\"panic\", \"{name}\", &[{args_list}])?;")
        }
    }
}

/// Direct register access is only generated together with the context ABI.
fn register_access(config: &TitleConfig) -> RegisterAccess {
    if config.runtime.guest_context && config.runtime.direct_registers {
        RegisterAccess::direct()
    } else {
        RegisterAccess::locals()
    }
}

fn translate_linear_function(
    function: &Function,
    config: &TitleConfig,
) -> Result<RustFunction, PipelineError> {
    let mut regs = Vec::new();
    let mut lines = Vec::new();
    let mut needs_flags = false;
    let access = register_access(config);

    for op in &function.ops {
        translate_op(op, config, access, &mut regs, &mut lines, &mut needs_flags)?;
    }

    let spills = spill_lines(&regs, config.runtime.guest_context, access.is_direct());
    let mut spilled = Vec::with_capacity(lines.len());
    for line in lines {
        if line == "return Ok(());" {
            spilled.extend(spills.iter().cloned());
            spilled.push("return Ok(());".to_string());
        } else {
            spilled.push(line);
        }
    }
    let lines = spilled;

    Ok(RustFunction {
        name: function.name.clone(),
        host: config.host_functions.get(&function.name).cloned(),
        regs,
        needs_flags,
        body: FunctionBody::Linear(lines),
    })
}

fn translate_block_function(
    function: &Function,
    config: &TitleConfig,
) -> Result<RustFunction, PipelineError> {
    let mut regs = Vec::new();
    let mut blocks = Vec::new();
    let mut needs_flags = false;
    let constants = constant_temps(function);
    let access = register_access(config);

    for block in &function.blocks {
        let mut lines = Vec::new();
        for op in &block.ops {
            if let Op::ConstI64 { dst, .. } = op {
                if constants.contains_key(dst.as_str()) {
                    continue;
                }
            }
            translate_op(op, config, access, &mut regs, &mut lines, &mut needs_flags)?;
        }
        if !constants.is_empty() {
            for line in &mut lines {
                *line = substitute_idents(line, &constants);
            }
        }
        let terminator = translate_terminator(block, access, &mut regs, &mut needs_flags)?;
        blocks.push(RustBlock {
            label: block.label.clone(),
            lines,
            terminator,
        });
    }
    regs.retain(|reg| !constants.contains_key(reg.as_str()));

    Ok(RustFunction {
        name: function.name.clone(),
        host: config.host_functions.get(&function.name).cloned(),
        regs,
        needs_flags,
        body: FunctionBody::Blocks(blocks),
    })
}

fn translate_op(
    op: &Op,
    config: &TitleConfig,
    access: RegisterAccess,
    regs: &mut Vec<String>,
    lines: &mut Vec<String>,
    needs_flags: &mut bool,
) -> Result<(), PipelineError> {
    match op {
        Op::ConstI64 { dst, imm } => {
            track_reg(regs, dst);
            lines.push(access.assign(dst, &format!("{imm}")));
        }
        Op::AddI64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(
                dst,
                &format!("{}.wrapping_add({})", access.read(lhs), access.read(rhs)),
            ));
        }
        Op::MovI64 { dst, src } => {
            track_reg(regs, dst);
            track_reg(regs, src);
            lines.push(access.assign(dst, &access.read(src)));
        }
        Op::SubI64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(
                dst,
                &format!("{}.wrapping_sub({})", access.read(lhs), access.read(rhs)),
            ));
        }
        Op::AndI64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(dst, &format!("{} & {}", access.read(lhs), access.read(rhs))));
        }
        Op::OrI64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(dst, &format!("{} | {}", access.read(lhs), access.read(rhs))));
        }
        Op::XorI64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(dst, &format!("{} ^ {}", access.read(lhs), access.read(rhs))));
        }
        Op::CmpI64 { lhs, rhs } => {
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            *needs_flags = true;
            let lhs = access.read(lhs);
            let rhs = access.read(rhs);
            lines.push(format!(
                "let (__recomp_cmp_res, __recomp_cmp_overflow) = {lhs}.overflowing_sub({rhs});"
            ));
            lines.push(format!(
                "let (_, __recomp_cmp_borrow) = ({lhs} as u64).overflowing_sub({rhs} as u64);"
            ));
            if access.is_direct() {
                lines.push(access.assign_nzcv(
                    "((((__recomp_cmp_res < 0) as u8) << 3) | (((__recomp_cmp_res == 0) as u8) << 2) | (((!__recomp_cmp_borrow) as u8) << 1) | (__recomp_cmp_overflow as u8))",
                ));
            } else {
                lines.push("flag_n = __recomp_cmp_res < 0;".to_string());
                lines.push("flag_z = __recomp_cmp_res == 0;".to_string());
                lines.push("flag_c = !__recomp_cmp_borrow;".to_string());
                lines.push("flag_v = __recomp_cmp_overflow;".to_string());
            }
        }
        Op::CmnI64 { lhs, rhs } => {
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            *needs_flags = true;
            let lhs = access.read(lhs);
            let rhs = access.read(rhs);
            lines.push(format!(
                "let (__recomp_cmn_res, __recomp_cmn_overflow) = {lhs}.overflowing_add({rhs});"
            ));
            lines.push(format!(
                "let (_, __recomp_cmn_carry) = ({lhs} as u64).overflowing_add({rhs} as u64);"
            ));
            if access.is_direct() {
                lines.push(access.assign_nzcv(
                    "((((__recomp_cmn_res < 0) as u8) << 3) | (((__recomp_cmn_res == 0) as u8) << 2) | ((__recomp_cmn_carry as u8) << 1) | (__recomp_cmn_overflow as u8))",
                ));
            } else {
                lines.push("flag_n = __recomp_cmn_res < 0;".to_string());
                lines.push("flag_z = __recomp_cmn_res == 0;".to_string());
                lines.push("flag_c = __recomp_cmn_carry;".to_string());
                lines.push("flag_v = __recomp_cmn_overflow;".to_string());
            }
        }
        Op::TestI64 { lhs, rhs } => {
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            *needs_flags = true;
            let lhs = access.read(lhs);
            let rhs = access.read(rhs);
            lines.push(format!("let __recomp_tst_res = {lhs} & {rhs};"));
            if access.is_direct() {
                lines.push(access.assign_nzcv(
                    "((((__recomp_tst_res < 0) as u8) << 3) | ((__recomp_tst_res == 0) as u8) << 2)",
                ));
            } else {
                lines.push("flag_n = __recomp_tst_res < 0;".to_string());
                lines.push("flag_z = __recomp_tst_res == 0;".to_string());
                lines.push("flag_c = false;".to_string());
                lines.push("flag_v = false;".to_string());
            }
        }
        Op::LslI64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(
                dst,
                &format!(
                    "(({} as u64) << (({} as u64) & 63)) as i64",
                    access.read(lhs),
                    access.read(rhs)
                ),
            ));
        }
        Op::LsrI64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(
                dst,
                &format!(
                    "(({} as u64) >> (({} as u64) & 63)) as i64",
                    access.read(lhs),
                    access.read(rhs)
                ),
            ));
        }
        Op::AsrI64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(
                dst,
                &format!(
                    "{} >> (({} as u64) & 63)",
                    access.read(lhs),
                    access.read(rhs)
                ),
            ));
        }
        Op::MulI64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(
                dst,
                &format!("{}.wrapping_mul({})", access.read(lhs), access.read(rhs)),
            ));
        }
        Op::MulHiU64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(
                dst,
                &format!(
                    "((({} as u64 as u128) * ({} as u64 as u128)) >> 64) as i64",
                    access.read(lhs),
                    access.read(rhs)
                ),
            ));
        }
        Op::MulHiS64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(
                dst,
                &format!(
                    "((({} as i128) * ({} as i128)) >> 64) as i64",
                    access.read(lhs),
                    access.read(rhs)
                ),
            ));
        }
        Op::UdivI64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            lines.push(access.assign(
                dst,
                &format!(
                    "({} as u64).checked_div({} as u64).unwrap_or(0) as i64",
                    access.read(lhs),
                    access.read(rhs)
                ),
            ));
        }
        Op::SdivI64 { dst, lhs, rhs } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            let lhs = access.read(lhs);
            let rhs = access.read(rhs);
            lines.push(access.assign(
                dst,
                &format!("if {rhs} == 0 {{ 0 }} else {{ {lhs}.wrapping_div({rhs}) }}"),
            ));
        }
        Op::SelectI64 {
            dst,
            cond,
            lhs,
            rhs,
        } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            *needs_flags = true;
            let expr = render_cond_expr(cond, access).ok_or_else(|| {
                PipelineError::Emit(format!("unsupported select condition {cond}"))
            })?;
            lines.push(access.assign(
                dst,
                &format!(
                    "if {expr} {{ {} }} else {{ {} }}",
                    access.read(lhs),
                    access.read(rhs)
                ),
            ));
        }
        Op::VConst { dst, lo, hi } => {
            track_reg(regs, dst);
            let value = ((*hi as u64 as u128) << 64) | (*lo as u64 as u128);
            lines.push(access.assign(dst, &format!("{value:#x}u128")));
        }
        Op::VLoad {
            dst,
            addr,
            offset,
            bytes,
        } => {
            track_reg(regs, dst);
            track_reg(regs, addr);
            let address = address_expr(addr, *offset, access);
            if *bytes == 16 {
                lines.push(access.assign(dst, &format!("__mem.load_u128({address})?")));
            } else {
                let helper = mem_helper("load", *bytes)?;
                lines.push(access.assign(dst, &format!("__mem.{helper}({address})? as u128")));
            }
        }
        Op::VStore {
            src,
            addr,
            offset,
            bytes,
        } => {
            track_reg(regs, src);
            track_reg(regs, addr);
            let address = address_expr(addr, *offset, access);
            if *bytes == 16 {
                lines.push(format!(
                    "__mem.store_u128({address}, {})?;",
                    access.read(src)
                ));
            } else {
                let helper = mem_helper("store", *bytes)?;
                lines.push(format!(
                    "__mem.{helper}({address}, {} as u64)?;",
                    access.read(src)
                ));
            }
        }
        Op::VMov { dst, src, bytes } => {
            track_reg(regs, dst);
            track_reg(regs, src);
            if *bytes >= 16 {
                lines.push(access.assign(dst, &access.read(src)));
            } else {
                lines.push(access.assign(
                    dst,
                    &format!("{} & {:#x}u128", access.read(src), lane_mask(*bytes)),
                ));
            }
        }
        Op::VDup { dst, src, esize, q } => {
            track_reg(regs, dst);
            track_reg(regs, src);
            let width = if *q { 16 } else { 8 };
            let repl = (0..width / *esize).fold(0u128, |acc, lane| {
                acc | (1u128 << (u32::from(lane) * u32::from(*esize) * 8))
            });
            lines.push(access.assign(
                dst,
                &format!(
                    "((({} as u64) & {:#x}) as u128).wrapping_mul({repl:#x}u128)",
                    access.read(src),
                    lane_mask(*esize)
                ),
            ));
        }
        Op::VInsGpr {
            dst,
            src,
            esize,
            index,
        } => {
            track_reg(regs, dst);
            track_reg(regs, src);
            let shift = u32::from(*index) * u32::from(*esize) * 8;
            let mask = lane_mask(*esize);
            lines.push(access.assign(
                dst,
                &format!(
                    "({} & !({mask:#x}u128 << {shift})) | (((({} as u64) as u128) & {mask:#x}u128) << {shift})",
                    access.read(dst),
                    access.read(src)
                ),
            ));
        }
        Op::VExtract {
            dst,
            src,
            esize,
            index,
            signed,
        } => {
            track_reg(regs, dst);
            track_reg(regs, src);
            let shift = u32::from(*index) * u32::from(*esize) * 8;
            let bits = u32::from(*esize) * 8;
            if *signed && bits < 64 {
                let up = 64 - bits;
                lines.push(access.assign(
                    dst,
                    &format!(
                        "(((({} >> {shift}) as u64) << {up}) as i64) >> {up}",
                        access.read(src)
                    ),
                ));
            } else {
                lines.push(access.assign(
                    dst,
                    &format!(
                        "((({} >> {shift}) as u64) & {:#x}) as i64",
                        access.read(src),
                        lane_mask(*esize)
                    ),
                ));
            }
        }
        Op::VInsLane {
            dst,
            src,
            esize,
            dst_index,
            src_index,
        } => {
            track_reg(regs, dst);
            track_reg(regs, src);
            let bits = u32::from(*esize) * 8;
            let mask = lane_mask(*esize);
            let to = u32::from(*dst_index) * bits;
            let from = u32::from(*src_index) * bits;
            lines.push(access.assign(
                dst,
                &format!(
                    "({} & !({mask:#x}u128 << {to})) | ((({} >> {from}) & {mask:#x}u128) << {to})",
                    access.read(dst),
                    access.read(src)
                ),
            ));
        }
        Op::VBitwise {
            dst,
            lhs,
            rhs,
            kind,
            q,
        } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            let lhs = access.read(lhs);
            let rhs = access.read(rhs);
            let expr = match kind.as_str() {
                "and" => format!("{lhs} & {rhs}"),
                "orr" => format!("{lhs} | {rhs}"),
                "eor" => format!("{lhs} ^ {rhs}"),
                "bic" => format!("{lhs} & !{rhs}"),
                "orn" => format!("{lhs} | !{rhs}"),
                other => {
                    return Err(PipelineError::Emit(format!(
                        "unsupported vector bitwise op {other}"
                    )))
                }
            };
            lines.push(access.assign(dst, &format!("({expr}){}", vector_width_mask(*q))));
        }
        Op::VBitSelect {
            dst,
            mask,
            set,
            clear,
            q,
        } => {
            track_reg(regs, dst);
            track_reg(regs, mask);
            track_reg(regs, set);
            track_reg(regs, clear);
            let mask = access.read(mask);
            lines.push(access.assign(
                dst,
                &format!(
                    "(({mask} & {}) | (!{mask} & {})){}",
                    access.read(set),
                    access.read(clear),
                    vector_width_mask(*q)
                ),
            ));
        }
        Op::ClzI64 { dst, src } => {
            track_reg(regs, dst);
            track_reg(regs, src);
            lines.push(access.assign(
                dst,
                &format!("({} as u64).leading_zeros() as i64", access.read(src)),
            ));
        }
        Op::RbitI64 { dst, src } => {
            track_reg(regs, dst);
            track_reg(regs, src);
            lines.push(access.assign(
                dst,
                &format!("({} as u64).reverse_bits() as i64", access.read(src)),
            ));
        }
        Op::RevI64 { dst, src } => {
            track_reg(regs, dst);
            track_reg(regs, src);
            lines.push(access.assign(
                dst,
                &format!("({} as u64).swap_bytes() as i64", access.read(src)),
            ));
        }
        Op::PcRel { dst, pc, offset } => {
            track_reg(regs, dst);
            lines.push(access.assign(dst, &format!("{pc} + {offset}")));
        }
        Op::LoadI8 { dst, addr, offset } => {
            track_reg(regs, dst);
            track_reg(regs, addr);
            emit_load(lines, dst, addr, *offset, "load_u8", access);
        }
        Op::LoadI16 { dst, addr, offset } => {
            track_reg(regs, dst);
            track_reg(regs, addr);
            emit_load(lines, dst, addr, *offset, "load_u16", access);
        }
        Op::LoadI32 { dst, addr, offset } => {
            track_reg(regs, dst);
            track_reg(regs, addr);
            emit_load(lines, dst, addr, *offset, "load_u32", access);
        }
        Op::LoadI64 { dst, addr, offset } => {
            track_reg(regs, dst);
            track_reg(regs, addr);
            emit_load(lines, dst, addr, *offset, "load_u64", access);
        }
        Op::StoreI8 { src, addr, offset } => {
            track_reg(regs, src);
            track_reg(regs, addr);
            emit_store(lines, src, addr, *offset, "store_u8", access);
        }
        Op::StoreI16 { src, addr, offset } => {
            track_reg(regs, src);
            track_reg(regs, addr);
            emit_store(lines, src, addr, *offset, "store_u16", access);
        }
        Op::StoreI32 { src, addr, offset } => {
            track_reg(regs, src);
            track_reg(regs, addr);
            emit_store(lines, src, addr, *offset, "store_u32", access);
        }
        Op::StoreI64 { src, addr, offset } => {
            track_reg(regs, src);
            track_reg(regs, addr);
            emit_store(lines, src, addr, *offset, "store_u64", access);
        }
        Op::ExclusiveLoad {
            dst,
            dst2,
            addr,
            size,
            pair,
            offset,
        } => {
            track_reg(regs, dst);
            if let Some(dst2) = dst2 {
                track_reg(regs, dst2);
            }
            track_reg(regs, addr);
            let address_expr = address_expr(addr, *offset, access);
            lines.push(format!(
                "let exclusive = recomp_abi::mem_load_exclusive({address_expr}, {size}, {pair})?;"
            ));
            lines.push(access.assign(dst, "exclusive.0 as i64"));
            if let Some(dst2) = dst2 {
                lines.push(access.assign(dst2, "exclusive.1 as i64"));
            }
        }
        Op::ExclusiveStore {
            src,
            src2,
            status,
            addr,
            size,
            pair,
            offset,
        } => {
            track_reg(regs, src);
            if let Some(src2) = src2 {
                track_reg(regs, src2);
            }
            track_reg(regs, addr);
            track_reg(regs, status);
            let address_expr = address_expr(addr, *offset, access);
            let second = src2
                .as_deref()
                .map(|src2| format!("{} as u64", access.read(src2)))
                .unwrap_or("0".to_string());
            lines.push(access.assign(
                status,
                &format!(
                    "recomp_abi::mem_store_exclusive({address_expr}, {size}, {pair}, {} as u64, {second})? as i64",
                    access.read(src)
                ),
            ));
        }
        Op::ClearExclusive => {
            lines.push("recomp_abi::clear_exclusive();".to_string());
        }
        Op::MemoryFence => {
            lines.push("recomp_abi::memory_fence();".to_string());
        }
        Op::AcquireFence => {
            lines.push("recomp_abi::acquire_fence();".to_string());
        }
        Op::ReleaseFence => {
            lines.push("recomp_abi::release_fence();".to_string());
        }
        Op::Br { target } => {
            lines.push(format!(
                "panic!({});",
                rust_string_literal(&format!("control-flow op in linear IR: br to {target}"))
            ));
        }
        Op::BrCond { cond, .. } => {
            *needs_flags = true;
            lines.push(format!(
                "panic!({});",
                rust_string_literal(&format!("control-flow op in linear IR: br_cond {cond}"))
            ));
        }
        Op::Call { target } => {
            let call_line = render_call_line(target);
            lines.push(call_line);
        }
        Op::Syscall { name, args } => {
            for arg in args {
                track_reg(regs, arg);
            }
            if recomp_services::kernel_svc_id(name).is_some() {
                track_reg(regs, "x0");
                track_reg(regs, "x1");
                track_reg(regs, "x2");
            }
            let behavior = config
                .stubs
                .get(name)
                .copied()
                .unwrap_or(StubBehavior::Panic);
            let call = render_syscall(name, behavior, args, access);
            lines.push(call);
        }
        Op::ReadSysReg { dst, reg } => {
            track_reg(regs, dst);
            let value = match reg.as_str() {
                // TPIDR_EL0 is the guest's own thread pointer; TPIDRRO_EL0
                // is the kernel-owned thread-local region (IPC buffer and
                // nn::os thread pointer).
                "tpidr_el0" => "recomp_abi::sysreg_read(recomp_abi::sysreg::TPIDR_EL0)",
                "tpidrro_el0" => "recomp_abi::sysreg_read(recomp_abi::sysreg::TPIDRRO_EL0)",
                "cntpct_el0" => "recomp_abi::sysreg_read(recomp_abi::sysreg::CNTPCT_EL0)",
                "cntfrq_el0" => "recomp_abi::sysreg_read(recomp_abi::sysreg::CNTFRQ_EL0)",
                "ctr_el0" => "recomp_abi::sysreg_read(recomp_abi::sysreg::CTR_EL0)",
                "fpcr" => "recomp_abi::sysreg_read(recomp_abi::sysreg::FPCR)",
                "fpsr" => "recomp_abi::sysreg_read(recomp_abi::sysreg::FPSR)",
                _ => "0",
            };
            lines.push(access.assign(dst, &format!("{value} as i64")));
        }
        Op::WriteSysReg { reg, src } => {
            track_reg(regs, src);
            let src = access.read(src);
            match reg.as_str() {
                "tpidr_el0" => lines.push(format!(
                    "recomp_abi::sysreg_write(recomp_abi::sysreg::TPIDR_EL0, {src} as u64);"
                )),
                "fpcr" => lines.push(format!(
                    "recomp_abi::sysreg_write(recomp_abi::sysreg::FPCR, {src} as u64);"
                )),
                "fpsr" => lines.push(format!(
                    "recomp_abi::sysreg_write(recomp_abi::sysreg::FPSR, {src} as u64);"
                )),
                // TPIDRRO_EL0 and read-only counters cannot be written.
                _ => lines.push(format!("let _ = {src};")),
            }
        }
        Op::ReadNzcv { dst } => {
            track_reg(regs, dst);
            *needs_flags = true;
            if access.is_direct() {
                let nzcv = access.flags_byte();
                lines.push(access.assign(
                    dst,
                    &format!(
                        "(((({nzcv} & 8) != 0) as i64) << 31) | (((({nzcv} & 4) != 0) as i64) << 30) | (((({nzcv} & 2) != 0) as i64) << 29) | (((({nzcv} & 1) != 0) as i64) << 28)"
                    ),
                ));
            } else {
                lines.push(format!(
                    "{dst} = ((flag_n as i64) << 31) | ((flag_z as i64) << 30) | ((flag_c as i64) << 29) | ((flag_v as i64) << 28);"
                ));
            }
        }
        Op::WriteNzcv { src } => {
            track_reg(regs, src);
            *needs_flags = true;
            let src = access.read(src);
            if access.is_direct() {
                lines.push(access.assign_nzcv(&format!("((({src} as u64) >> 28) & 0xf) as u8")));
            } else {
                let value = format!("({src} as u64)");
                lines.push(format!("let __recomp_nzcv = {value};"));
                lines.push("flag_n = (__recomp_nzcv & (1 << 31)) != 0;".to_string());
                lines.push("flag_z = (__recomp_nzcv & (1 << 30)) != 0;".to_string());
                lines.push("flag_c = (__recomp_nzcv & (1 << 29)) != 0;".to_string());
                lines.push("flag_v = (__recomp_nzcv & (1 << 28)) != 0;".to_string());
            }
        }
        Op::AddCarry {
            dst,
            lhs,
            rhs,
            subtract,
            set_flags,
            is_32,
        } => {
            track_reg(regs, dst);
            track_reg(regs, lhs);
            track_reg(regs, rhs);
            *needs_flags = true;
            let (bits, mask, signed) = if *is_32 {
                (32, "0xffff_ffffu64", "as u32 as i32 as i128")
            } else {
                (64, "u64::MAX", "as i64 as i128")
            };
            let invert = if *subtract { "!" } else { "" };
            let lhs = access.read(lhs);
            let rhs = access.read(rhs);
            let carry = if access.is_direct() {
                format!("((({}) & 2) != 0) as u64", access.flags_byte())
            } else {
                "flag_c as u64".to_string()
            };
            let mut line = format!(
                "{{ let __l = ({lhs} as u64) & {mask}; let __r = ({invert}({rhs} as u64)) & {mask}; let __c = {carry}; let __u = __l as u128 + __r as u128 + __c as u128; let __res = (__u as u64) & {mask};"
            );
            if *set_flags {
                if access.is_direct() {
                    line.push_str(&format!(
                        " let __s = (__l {signed}) + (__r {signed}) + __c as i128; let __nzcv = (((__res >> {}) & 1) as u8) << 3 | ((__res == 0) as u8) << 2 | (((__u >> {bits}) != 0) as u8) << 1 | (__s != (__res {signed})) as u8;",
                        bits - 1
                    ));
                } else {
                    line.push_str(&format!(
                        " let __s = (__l {signed}) + (__r {signed}) + __c as i128; flag_n = (__res >> {}) & 1 != 0; flag_z = __res == 0; flag_c = (__u >> {bits}) != 0; flag_v = __s != (__res {signed});",
                        bits - 1
                    ));
                }
            }
            if access.is_direct() {
                let nzcv = if *set_flags {
                    " (*ctx).nzcv = __nzcv;"
                } else {
                    ""
                };
                let target = reg_index(dst)
                    .map(|index| format!("(*ctx).x[{index}]"))
                    .unwrap_or_else(|| dst.to_string());
                line.push_str(&format!(" unsafe {{ {nzcv} {target} = __res as i64; }} }}"));
            } else {
                line.push_str(&format!(" {dst} = __res as i64; }}"));
            }
            lines.push(line);
        }
        Op::Simd {
            word,
            vector_reads,
            general_read,
            dst,
            reads_flags,
            writes_flags,
        } => {
            if dst.is_none() && !writes_flags {
                return Ok(());
            }
            let mut values = Vec::with_capacity(vector_reads.len());
            for reg in vector_reads {
                track_reg(regs, reg);
                let index = vreg_index(reg).ok_or_else(|| {
                    PipelineError::Emit(format!("simd operand {reg} is not a vector register"))
                })?;
                values.push(format!("({index}, {})", access.read(reg)));
            }
            let general = match general_read {
                Some(reg) => {
                    track_reg(regs, reg);
                    format!("{} as u64", access.read(reg))
                }
                None => "0".to_string(),
            };
            let flags = if *reads_flags {
                *needs_flags = true;
                access.flags_byte()
            } else {
                "0".to_string()
            };
            // The common instructions are decoded now, into a direct call;
            // `exec` would decode them on every run.
            if let Some(call) = recomp_simd::fast::specialize(*word) {
                use recomp_simd::fast::{Arg, Output};
                let mut args = Vec::with_capacity(call.args.len());
                for arg in &call.args {
                    args.push(match arg {
                        Arg::Vector(index) => {
                            let name = format!("v{index}");
                            track_reg(regs, &name);
                            access.read(&name)
                        }
                        Arg::General => general.clone(),
                        Arg::Flags => flags.to_string(),
                        Arg::Int(value) => format!("{value}u32"),
                        Arg::Bool(value) => value.to_string(),
                        Arg::Constant(value) => format!("{value:#x}u128"),
                    });
                }
                let expr = format!(
                    "recomp_abi::simd::fast::{}({})",
                    call.function,
                    args.join(", ")
                );
                match call.output {
                    Output::Value => {
                        if let Some(reg) = dst {
                            track_reg(regs, reg);
                            // A vector destination in direct mode is a
                            // `(*ctx).v[N]` field. Passing its address lets
                            // the inline host path store the whole register
                            // in one 16-byte access; the `u128` return value
                            // made the compiler split the result into two
                            // general registers and two stores.
                            let into = match call.function {
                                "vector3" => Some("vector3_into"),
                                "vector_element" => Some("vector_element_into"),
                                _ => None,
                            };
                            if let (Some(into), true, Some(index)) =
                                (into, access.is_direct(), vreg_index(reg))
                            {
                                lines.push(format!(
                                    "unsafe {{ recomp_abi::simd::fast::{into}(\
                                     core::ptr::addr_of_mut!((*ctx).v[{index}]), {}) }};",
                                    args.join(", ")
                                ));
                            } else if vreg_index(reg).is_some() {
                                lines.push(access.assign(reg, &expr));
                            } else {
                                lines.push(access.assign(reg, &format!("{expr} as u64 as i64")));
                            }
                        }
                    }
                    Output::Flags => {
                        *needs_flags = true;
                        if access.is_direct() {
                            lines.push(format!(
                                "{{ let __f = {expr}; {} }}",
                                access.assign_nzcv("__f")
                            ));
                        } else {
                            lines.push(format!(
                                "{{ let __f = {expr}; flag_n = __f & 8 != 0; flag_z = __f & 4 != 0; flag_c = __f & 2 != 0; flag_v = __f & 1 != 0; }}"
                            ));
                        }
                    }
                }
                return Ok(());
            }
            let mut line = format!(
                "{{ let __s = recomp_abi::simd::exec({word:#x}, &[{}], {general}, {flags})?;",
                values.join(", ")
            );
            if let Some(reg) = dst {
                track_reg(regs, reg);
                if vreg_index(reg).is_some() {
                    line.push_str(&format!(" {}", access.assign(reg, "__s.value")));
                } else {
                    line.push_str(&format!(
                        " {}",
                        access.assign(reg, "__s.value as u64 as i64")
                    ));
                }
            }
            if *writes_flags {
                *needs_flags = true;
                if access.is_direct() {
                    line.push_str(&format!(" {}", access.assign_nzcv("__s.nzcv")));
                } else {
                    line.push_str(" flag_n = __s.nzcv & 8 != 0; flag_z = __s.nzcv & 4 != 0; flag_c = __s.nzcv & 2 != 0; flag_v = __s.nzcv & 1 != 0;");
                }
            }
            line.push_str(" }");
            lines.push(line);
        }
        Op::Ret => {
            lines.push("return Ok(());".to_string());
        }
    }
    Ok(())
}

fn translate_terminator(
    block: &Block,
    access: RegisterAccess,
    regs: &mut Vec<String>,
    needs_flags: &mut bool,
) -> Result<RustTerminator, PipelineError> {
    match &block.terminator {
        Terminator::Br { target } => Ok(RustTerminator::Br {
            target: target.clone(),
        }),
        Terminator::BrCond {
            cond,
            then,
            else_target,
        } => {
            *needs_flags = true;
            let cond_expr = render_cond_expr(cond, access);
            Ok(RustTerminator::BrCond {
                cond_expr,
                cond: cond.clone(),
                then_label: then.clone(),
                else_label: else_target.clone(),
            })
        }
        Terminator::BrTest {
            reg,
            mask,
            zero,
            then,
            else_target,
        } => {
            track_reg(regs, reg);
            let op = if *zero { "==" } else { "!=" };
            Ok(RustTerminator::BrCond {
                cond_expr: Some(format!(
                    "(({} as u64) & {mask:#x}) {op} 0",
                    access.read(reg)
                )),
                cond: format!("test {reg}"),
                then_label: then.clone(),
                else_label: else_target.clone(),
            })
        }
        Terminator::Call {
            target,
            next,
            return_address,
        } => {
            let call_line = render_call_line(target);
            if return_address.is_some() {
                track_reg(regs, "x30");
            }
            Ok(RustTerminator::Call {
                call_line,
                next: next.clone(),
                return_address: *return_address,
            })
        }
        Terminator::BrIndirect { reg } => {
            track_reg(regs, reg);
            Ok(RustTerminator::BrIndirect { reg: reg.clone() })
        }
        Terminator::CallIndirect {
            reg,
            next,
            return_address,
        } => {
            track_reg(regs, reg);
            if return_address.is_some() {
                track_reg(regs, "x30");
            }
            Ok(RustTerminator::CallIndirect {
                reg: reg.clone(),
                next: next.clone(),
                return_address: *return_address,
            })
        }
        Terminator::Switch { reg, cases } => {
            track_reg(regs, reg);
            Ok(RustTerminator::Switch {
                reg: reg.clone(),
                cases: cases.clone(),
            })
        }
        Terminator::Trap { pc, word } => Ok(RustTerminator::Trap {
            pc: *pc,
            word: *word,
        }),
        Terminator::Ret => Ok(RustTerminator::Ret),
    }
}

/// Index of a guest vector register name (`v0`-`v31`).
pub(crate) fn vreg_index(name: &str) -> Option<usize> {
    let index: usize = name.strip_prefix('v')?.parse().ok()?;
    (index < 32).then_some(index)
}

/// Mask of the low `bytes` bytes.
fn lane_mask(bytes: u8) -> u128 {
    if bytes >= 16 {
        u128::MAX
    } else {
        (1u128 << (u32::from(bytes) * 8)) - 1
    }
}

/// Suffix that clears the upper half of a 64-bit (`q` false) vector result.
fn vector_width_mask(q: bool) -> &'static str {
    if q {
        ""
    } else {
        " & 0xffff_ffff_ffff_ffffu128"
    }
}

fn mem_helper(kind: &str, bytes: u8) -> Result<String, PipelineError> {
    let bits = match bytes {
        1 => 8,
        2 => 16,
        4 => 32,
        8 => 64,
        other => {
            return Err(PipelineError::Emit(format!(
                "unsupported {kind} width {other}"
            )))
        }
    };
    Ok(format!("{kind}_u{bits}"))
}

pub(crate) fn reg_index(name: &str) -> Option<usize> {
    let index: usize = name.strip_prefix('x')?.parse().ok()?;
    (index < 32).then_some(index)
}

/// Lines that write the function's live registers to the guest context.
/// The snapshot ABI writes them through `cpu_store`/`cpu_vstore` (a slice
/// literal per call site); the guest-context ABI stores straight into the
/// fields of the function's `ctx`. Direct register mode needs neither: every
/// op already wrote the context field.
pub(crate) fn spill_lines(regs: &[String], guest_context: bool, direct: bool) -> Vec<String> {
    let mut lines = Vec::new();
    if direct {
        return lines;
    }
    if guest_context {
        for reg in regs {
            if let Some(index) = reg_index(reg) {
                lines.push(format!("unsafe {{ (*ctx).x[{index}] = {reg}; }}"));
            } else if let Some(index) = vreg_index(reg) {
                lines.push(format!("unsafe {{ (*ctx).v[{index}] = {reg}; }}"));
            }
        }
        return lines;
    }
    let pairs: Vec<String> = regs
        .iter()
        .filter_map(|reg| reg_index(reg).map(|index| format!("({index}, {reg})")))
        .collect();
    if !pairs.is_empty() {
        lines.push(format!("recomp_abi::cpu_store(&[{}]);", pairs.join(", ")));
    }
    let vpairs: Vec<String> = regs
        .iter()
        .filter_map(|reg| vreg_index(reg).map(|index| format!("({index}, {reg})")))
        .collect();
    if !vpairs.is_empty() {
        lines.push(format!("recomp_abi::cpu_vstore(&[{}]);", vpairs.join(", ")));
    }
    lines
}

/// Lines that reload the function's live registers from the guest context
/// after a call. Direct register mode reads every register from the context
/// at its use, so there is nothing to reload.
pub(crate) fn reload_lines(regs: &[String], guest_context: bool, direct: bool) -> Vec<String> {
    let mut lines = Vec::new();
    if direct {
        return lines;
    }
    if guest_context {
        for reg in regs {
            if let Some(index) = reg_index(reg) {
                lines.push(format!("{reg} = unsafe {{ (*ctx).x[{index}] }};"));
            } else if let Some(index) = vreg_index(reg) {
                lines.push(format!("{reg} = unsafe {{ (*ctx).v[{index}] }};"));
            }
        }
        return lines;
    }
    let loads: Vec<String> = regs
        .iter()
        .filter_map(|reg| reg_index(reg).map(|index| format!("{reg} = __r[{index}];")))
        .collect();
    if !loads.is_empty() {
        lines.push(format!(
            "{{ let __r = recomp_abi::cpu_regs(); {} }}",
            loads.join(" ")
        ));
    }
    let vloads: Vec<String> = regs
        .iter()
        .filter_map(|reg| vreg_index(reg).map(|index| format!("{reg} = __v[{index}];")))
        .collect();
    if !vloads.is_empty() {
        lines.push(format!(
            "{{ let __v = recomp_abi::cpu_vregs(); {} }}",
            vloads.join(" ")
        ));
    }
    lines
}

fn render_call_line(target: &str) -> String {
    if let Some(addr) = crate::output::function_address(target) {
        format!("recomp_abi::dispatch({addr:#x})?;")
    } else if is_rust_ident(target) {
        format!("{target}()?;")
    } else {
        format!(
            "panic!({});",
            rust_string_literal(&format!("unsupported call target: {target}"))
        )
    }
}

fn address_expr(addr: &str, offset: i64, access: RegisterAccess) -> String {
    let addr = access.read(addr);
    match offset.signum() {
        1 => format!("({addr} as u64).wrapping_add({offset}u64)"),
        -1 => format!("({addr} as u64).wrapping_sub({}u64)", -offset),
        _ => format!("{addr} as u64"),
    }
}

/// Guest loads and stores go through the function's `__mem`
/// (`recomp_abi::memory()`, taken as it starts).
fn emit_load(
    lines: &mut Vec<String>,
    dst: &str,
    addr: &str,
    offset: i64,
    helper: &str,
    access: RegisterAccess,
) {
    let address_expr = address_expr(addr, offset, access);
    lines.push(access.assign(dst, &format!("__mem.{helper}({address_expr})? as i64")));
}

fn emit_store(
    lines: &mut Vec<String>,
    src: &str,
    addr: &str,
    offset: i64,
    helper: &str,
    access: RegisterAccess,
) {
    let address_expr = address_expr(addr, offset, access);
    lines.push(format!(
        "__mem.{helper}({address_expr}, {} as u64)?;",
        access.read(src)
    ));
}

fn render_cond_expr(cond: &str, access: RegisterAccess) -> Option<String> {
    let expr = match cond {
        "eq" => access.flag('z'),
        "ne" => format!("!{}", access.flag('z')),
        "cs" | "hs" => access.flag('c'),
        "cc" | "lo" => format!("!{}", access.flag('c')),
        "mi" => access.flag('n'),
        "pl" => format!("!{}", access.flag('n')),
        "vs" => access.flag('v'),
        "vc" => format!("!{}", access.flag('v')),
        "hi" => format!("{} && !{}", access.flag('c'), access.flag('z')),
        "ls" => format!("!{} || {}", access.flag('c'), access.flag('z')),
        "ge" => format!("{} == {}", access.flag('n'), access.flag('v')),
        "lt" => format!("{} != {}", access.flag('n'), access.flag('v')),
        "gt" => format!(
            "!{} && ({} == {})",
            access.flag('z'),
            access.flag('n'),
            access.flag('v')
        ),
        "le" => format!(
            "{} || ({} != {})",
            access.flag('z'),
            access.flag('n'),
            access.flag('v')
        ),
        "al" => "true".to_string(),
        _ => return None,
    };
    Some(expr)
}

fn is_rust_ident(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

pub(crate) fn rust_string_literal(value: &str) -> String {
    format!("{value:?}")
}

#[derive(Debug)]
pub struct RustFunction {
    pub name: String,
    /// The host implementation tried first (`host_functions`).
    pub host: Option<String>,
    pub regs: Vec<String>,
    pub needs_flags: bool,
    pub body: FunctionBody,
}

#[derive(Debug)]
pub enum FunctionBody {
    Linear(Vec<String>),
    Blocks(Vec<RustBlock>),
}

#[derive(Debug)]
pub struct RustBlock {
    pub label: String,
    pub lines: Vec<String>,
    pub terminator: RustTerminator,
}

#[derive(Debug)]
pub enum RustTerminator {
    Br {
        target: String,
    },
    BrCond {
        cond_expr: Option<String>,
        cond: String,
        then_label: String,
        else_label: String,
    },
    Call {
        call_line: String,
        next: String,
        return_address: Option<u64>,
    },
    BrIndirect {
        reg: String,
    },
    CallIndirect {
        reg: String,
        next: String,
        return_address: Option<u64>,
    },
    Switch {
        reg: String,
        cases: Vec<(u64, String)>,
    },
    Trap {
        pc: u64,
        word: u32,
    },
    Ret,
}

#[derive(Debug)]
pub struct RustProgram {
    pub title: String,
    pub abi_version: String,
    pub entry: String,
    pub functions: Vec<RustFunction>,
    pub performance_mode: PerformanceMode,
    pub graphics: crate::config::GraphicsBackend,
    pub parallel: bool,
    /// Whether generated code records its current block at every block
    /// (see `[runtime] block_tracking`).
    pub block_tracking: bool,
    /// The CPU the program is compiled for (`-C target-cpu`), if not the
    /// generic default.
    pub target_cpu: Option<String>,
    /// The Cargo target triple the program is cross-compiled for, if not the
    /// build host's (`[runtime] target`).
    pub target: Option<String>,
    /// Emit the guest-context ABI: functions take `*mut GuestContext` and
    /// access its fields directly.
    pub guest_context: bool,
    /// Lower each guest register access to a context field instead of a
    /// whole-function local plus prologue/call spills. Requires
    /// `guest_context`.
    pub direct_registers: bool,
    /// The script loaded when `RECOMP_SCRIPT` is not set.
    pub script: Option<crate::config::RuntimeAsset>,
    pub memory_layout: MemoryLayoutDescriptor,
    pub memory_image: Option<MemoryImageDescriptor>,
    pub ipc: Option<crate::config::IpcConfig>,
    pub initial_sp: Option<u64>,
    pub entry_address: Option<u64>,
}

impl RustProgram {
    pub fn entry_function(&self) -> Option<&RustFunction> {
        self.functions.iter().find(|func| func.name == self.entry)
    }
}

pub fn ensure_dir(path: &Path) -> Result<(), PipelineError> {
    fs::create_dir_all(path).map_err(PipelineError::Io)
}

fn absolute_path(path: &Path) -> Result<PathBuf, PipelineError> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

/// The toolchain path relative to the emitted project, or the absolute path
/// when the two share no root (as with a different drive on Windows). The
/// emitter writes either form with `/` separators, so an absolute fallback
/// is still valid in Cargo.toml.
fn runtime_relative_path(runtime_path: &Path, out_dir: &Path) -> PathBuf {
    diff_paths(runtime_path, out_dir).unwrap_or_else(|| runtime_path.to_path_buf())
}

fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    format!("{:x}", digest)
}
