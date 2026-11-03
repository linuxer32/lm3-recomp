use crate::homebrew::module::{
    BssInfo, ModuleBuild, ModuleJson, ModuleSegment, OffsetInfo, MODULE_SCHEMA_VERSION,
};
use crate::homebrew::nro::{parse_nro, NroModule};
use crate::homebrew::nso::{
    extract_segments, parse_nso, NsoModule, NsoSegmentData, NsoSegmentKind,
};
use crate::homebrew::romfs::{list_romfs_entries, RomfsEntry};
use crate::homebrew::util::hex_bytes;
use crate::output::{GeneratedFile, InputSummary};
use crate::provenance::{InputFormat, ProvenanceManifest};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const INTAKE_SCHEMA_VERSION: &str = "1";

#[derive(Debug)]
pub struct IntakeOptions {
    pub module_path: PathBuf,
    pub nso_paths: Vec<PathBuf>,
    pub provenance_path: PathBuf,
    pub out_dir: PathBuf,
    /// Apply MOD0 relocation tables to extracted NSO .text before writing
    /// segment files (static recompilation needs patched images).
    pub apply_relocations: bool,
    /// Resolve relocations only to discover code pointers and write the
    /// segments as loaded, for titles booted through their own rtld (it
    /// links the modules at startup and rejects pre-relocated images).
    pub link_at_runtime: bool,
}

#[derive(Debug)]
pub struct IntakeReport {
    pub out_dir: PathBuf,
    pub module_json_path: PathBuf,
    pub manifest_path: PathBuf,
    pub files_written: Vec<PathBuf>,
    pub relocations_applied: Vec<(String, crate::homebrew::mod0::RelocationReport)>,
    pub relocation_warnings: Vec<String>,
}

/// Module build, generated-file records, written paths, relocation report,
/// and an optional relocation warning.

#[derive(Debug, serde::Serialize)]
struct IntakeManifest {
    schema_version: String,
    tool: ToolInfo,
    modules: Vec<ModuleRecord>,
    assets: Vec<AssetRecord>,
    inputs: Vec<InputSummary>,
    generated_files: Vec<GeneratedFile>,
}

#[derive(Debug, serde::Serialize)]
struct ToolInfo {
    name: String,
    version: String,
}

#[derive(Debug, serde::Serialize)]
struct ModuleRecord {
    name: String,
    format: String,
    build_id: String,
    module_json_path: String,
}

#[derive(Debug, serde::Serialize, Clone)]
struct AssetRecord {
    kind: String,
    path: String,
    sha256: String,
    size: u64,
    source_offset: u64,
    source_size: u64,
}

pub fn intake_homebrew(options: IntakeOptions) -> Result<IntakeReport, String> {
    let module_path = absolute_path(&options.module_path)?;
    let nso_paths = options
        .nso_paths
        .iter()
        .map(|path| absolute_path(path))
        .collect::<Result<Vec<_>, _>>()?;
    let provenance_path = absolute_path(&options.provenance_path)?;
    let out_dir = absolute_path(&options.out_dir)?;

    let provenance_src =
        fs::read_to_string(&provenance_path).map_err(|err| format!("read provenance: {err}"))?;
    let provenance = ProvenanceManifest::parse(&provenance_src)?;
    let validation = provenance.validate(&provenance_path, &provenance_src)?;

    enforce_homebrew_formats(&validation.inputs)?;
    ensure_input_present(&validation.inputs, &module_path)?;
    for nso_path in &nso_paths {
        ensure_input_present(&validation.inputs, nso_path)?;
    }

    // The primary input may be either an NRO homebrew image or an NSO module
    // from a retail title's ExeFS. For retail titles the NSOs form one linked
    // program: apply their MOD0 relocations against a shared symbol space.
    let primary_bytes = fs::read(&module_path).map_err(|err| format!("read module: {err}"))?;
    let primary_is_nso = primary_bytes.starts_with(b"NSO0");
    let nro = if primary_is_nso {
        None
    } else {
        Some(parse_nro(&module_path)?)
    };

    let mut nso_modules = Vec::new();
    if primary_is_nso {
        nso_modules.push(parse_nso(&module_path)?);
    }
    for path in &nso_paths {
        nso_modules.push(parse_nso(path)?);
    }

    fs::create_dir_all(&out_dir)
        .map_err(|err| format!("create out dir {}: {err}", out_dir.display()))?;
    let segments_dir = out_dir.join("segments");
    let assets_dir = out_dir.join("assets");
    fs::create_dir_all(&segments_dir).map_err(|err| format!("create segments dir: {err}"))?;
    fs::create_dir_all(&assets_dir).map_err(|err| format!("create assets dir: {err}"))?;

    let mut generated_files = Vec::new();
    let mut files_written = Vec::new();
    let mut module_builds = Vec::new();
    let mut relocations_applied: Vec<(String, crate::homebrew::mod0::RelocationReport)> =
        Vec::new();
    let mut relocation_warnings: Vec<String> = Vec::new();

    // Deterministic module bases: the primary module at 0, then one 0x10000000
    // slot per additional NSO in command-line order.
    let mut next_base = 0u64;
    if let Some(nro) = &nro {
        let (build, files, written) = write_nro_segments(nro, &segments_dir)?;
        module_builds.push(build);
        generated_files.extend(files);
        files_written.extend(written);
        next_base = 0x1000_0000;
    }

    // NSO-only titles load like Horizon's loader places them: back to back
    // in command-line order (rtld, main, subsdk*, sdk), page aligned, from a
    // non-zero base so null pointers stay unmapped.
    if nro.is_none() {
        next_base = NSO_LOAD_BASE;
    }
    let link_modules: Vec<(&NsoModule, u64)> = nso_modules
        .iter()
        .map(|module| {
            let base = next_base;
            next_base = if nro.is_none() {
                (base + nso_image_size(module)).next_multiple_of(0x1000)
            } else {
                next_base + 0x1000_0000
            };
            (module, base)
        })
        .collect();
    let linked = if options.apply_relocations || options.link_at_runtime {
        write_linked_nso_segments(&link_modules, &segments_dir, options.link_at_runtime)?
    } else {
        link_modules
            .iter()
            .map(|(module, base)| write_unlinked_nso_segments(module, *base, &segments_dir))
            .collect::<Result<Vec<_>, _>>()?
    };

    for output in linked {
        if let Some(report) = output.relocations {
            relocations_applied.push((output.build.name.clone(), report));
        }
        if let Some(warning) = output.warning {
            relocation_warnings.push(format!("{}: {warning}", output.build.name));
        }
        module_builds.push(output.build);
        generated_files.extend(output.generated);
        files_written.extend(output.written);
    }

    let mut assets = Vec::new();
    if let Some(nro) = &nro {
        let (asset_files, asset_written) = extract_assets(nro, &out_dir, &assets_dir, &mut assets)?;
        generated_files.extend(asset_files);
        files_written.extend(asset_written);
    }

    module_builds.sort_by(|a, b| a.name.cmp(&b.name));
    let module_json = ModuleJson {
        schema_version: MODULE_SCHEMA_VERSION.to_string(),
        module_type: "homebrew".to_string(),
        modules: module_builds,
    };

    let module_json_path = out_dir.join("module.json");
    let module_json_src =
        serde_json::to_string_pretty(&module_json).map_err(|err| err.to_string())?;
    fs::write(&module_json_path, module_json_src.as_bytes())
        .map_err(|err| format!("write module.json: {err}"))?;
    files_written.push(module_json_path.clone());
    generated_files.push(GeneratedFile {
        path: "module.json".to_string(),
        sha256: sha256_bytes(module_json_src.as_bytes()),
        size: module_json_src.len() as u64,
    });

    let inputs = validation
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

    let module_records = module_json
        .modules
        .iter()
        .map(|module| ModuleRecord {
            name: module.name.clone(),
            format: module.format.clone(),
            build_id: module.build_id.clone(),
            module_json_path: "module.json".to_string(),
        })
        .collect::<Vec<_>>();

    assets.sort_by(|a, b| a.path.cmp(&b.path));
    generated_files.sort_by(|a, b| a.path.cmp(&b.path));

    let manifest = IntakeManifest {
        schema_version: INTAKE_SCHEMA_VERSION.to_string(),
        tool: ToolInfo {
            name: "recomp-pipeline".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
        modules: module_records,
        assets,
        inputs,
        generated_files,
    };

    let manifest_path = out_dir.join("manifest.json");
    let manifest_src = serde_json::to_string_pretty(&manifest).map_err(|err| err.to_string())?;
    fs::write(&manifest_path, manifest_src.as_bytes())
        .map_err(|err| format!("write manifest.json: {err}"))?;
    files_written.push(manifest_path.clone());

    Ok(IntakeReport {
        out_dir,
        module_json_path,
        manifest_path,
        files_written,
        relocations_applied,
        relocation_warnings,
    })
}

fn write_nro_segments(
    module: &NroModule,
    segments_dir: &Path,
) -> Result<(ModuleBuild, Vec<GeneratedFile>, Vec<PathBuf>), String> {
    let module_name = module
        .path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("nro")
        .to_string();
    let module_dir = segments_dir.join(&module_name);
    fs::create_dir_all(&module_dir).map_err(|err| format!("create module dir: {err}"))?;

    let bytes = fs::read(&module.path)
        .map_err(|err| format!("read NRO {}: {err}", module.path.display()))?;
    let mut segments = Vec::new();
    let mut generated = Vec::new();
    let mut written = Vec::new();

    for segment in &module.segments {
        let start = segment.file_offset as usize;
        let end = start
            .checked_add(segment.size as usize)
            .ok_or_else(|| "segment offset overflow".to_string())?;
        if end > bytes.len() {
            return Err(format!("NRO segment out of range: {}..{}", start, end));
        }
        let data = &bytes[start..end];
        let file_name = format!("{}.bin", segment.name);
        let output_rel = format!("segments/{module_name}/{file_name}");
        let output_path = module_dir.join(&file_name);
        fs::write(&output_path, data).map_err(|err| format!("write segment {file_name}: {err}"))?;
        written.push(output_path.clone());
        generated.push(GeneratedFile {
            path: output_rel.clone(),
            sha256: sha256_bytes(data),
            size: data.len() as u64,
        });
        segments.push(ModuleSegment {
            name: segment.name.clone(),
            file_offset: segment.file_offset as u64,
            file_size: segment.size as u64,
            memory_offset: segment.memory_offset as u64,
            memory_size: segment.size as u64,
            permissions: segment.permissions.as_str().to_string(),
            compressed: None,
            output_path: output_rel,
        });
    }

    let input_sha256 = sha256_path(&module.path)?;
    let input_size = bytes.len() as u64;
    let bss_offset = module
        .segments
        .iter()
        .find(|segment| segment.name == "data")
        .map(|segment| segment.memory_offset as u64 + segment.size as u64)
        .unwrap_or(0);

    let build = ModuleBuild {
        name: module_name,
        format: "nro".to_string(),
        base: 0,
        input_path: module.path.clone(),
        input_sha256,
        input_size,
        build_id: module.build_id_hex(),
        segments,
        bss: BssInfo {
            size: module.bss_size as u64,
            memory_offset: bss_offset,
        },
        embedded: None,
        dynstr: None,
        dynsym: None,
        code_pointers: Vec::new(),
    };

    Ok((build, generated, written))
}

struct LinkedNsoOutput {
    build: ModuleBuild,
    generated: Vec<GeneratedFile>,
    written: Vec<PathBuf>,
    relocations: Option<crate::homebrew::mod0::RelocationReport>,
    warning: Option<String>,
}

struct PreparedNso<'a> {
    module: &'a NsoModule,
    name: String,
    base: u64,
    segment_data: Vec<NsoSegmentData>,
    image: Vec<u8>,
    image_start: usize,
    mod0: Option<crate::homebrew::mod0::Mod0Header>,
}

fn prepare_nso(module: &NsoModule, base: u64) -> Result<PreparedNso<'_>, String> {
    let module_name = module
        .path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("nso")
        .to_string();
    let segment_data = extract_segments(module)?;
    if segment_data.is_empty() {
        return Err(format!("module {module_name} has no segments"));
    }
    let image_start = segment_data
        .iter()
        .map(|segment| segment.segment.memory_offset as usize)
        .min()
        .unwrap_or(0);
    let image_end = segment_data
        .iter()
        .map(|segment| segment.segment.memory_offset as usize + segment.data.len())
        .max()
        .unwrap_or(image_start)
        .max(image_start + module.bss_size as usize);
    let image = crate::homebrew::mod0::build_memory_image(&segment_data, image_end - image_start)?;

    let text = segment_data
        .iter()
        .find(|segment| segment.segment.kind == NsoSegmentKind::Text);
    let mod0 = text.and_then(|text| {
        let text_offset = text.segment.memory_offset as usize - image_start;
        crate::homebrew::mod0::find_mod0(&image[text_offset..])
            .and_then(|offset| crate::homebrew::mod0::parse_mod0(&image, text_offset + offset))
    });

    Ok(PreparedNso {
        module,
        name: module_name,
        base,
        segment_data,
        image,
        image_start,
        mod0,
    })
}

/// Build the shared symbol space exported by every defined dynamic symbol in
/// every linked module. Later duplicates keep the first definition, matching
/// load order.
fn collect_nso_exports(prepared: &[PreparedNso<'_>]) -> Result<BTreeMap<String, u64>, String> {
    let mut exports = BTreeMap::new();
    for module in prepared {
        let Some(header) = &module.mod0 else {
            continue;
        };
        let dynamic = crate::homebrew::mod0::parse_dynamic(&module.image, header)?;
        let symbols = crate::homebrew::mod0::parse_symbols(&module.image, &dynamic)?;
        for symbol in symbols {
            if symbol.name.is_empty() || !symbol.is_defined() {
                continue;
            }
            let address = if symbol.is_absolute() {
                symbol.value
            } else {
                module.base.wrapping_add(symbol.value)
            };
            exports.entry(symbol.name).or_insert(address);
        }
    }
    Ok(exports)
}

/// Link all NSO modules and write their relocated segment images.
fn write_linked_nso_segments(
    modules: &[(&NsoModule, u64)],
    segments_dir: &Path,
    link_at_runtime: bool,
) -> Result<Vec<LinkedNsoOutput>, String> {
    let prepared = modules
        .iter()
        .map(|(module, base)| prepare_nso(module, *base))
        .collect::<Result<Vec<_>, _>>()?;
    let exports = collect_nso_exports(&prepared)?;
    let text_ranges: Vec<(u64, u64)> = prepared.iter().flat_map(code_ranges).collect();

    let mut outputs = Vec::new();
    for mut module in prepared {
        let mut relocations = None;
        let mut warning = None;
        let loaded_image = link_at_runtime.then(|| module.image.clone());
        match &module.mod0 {
            Some(header) => {
                // The linker-script symbols __rel_dyn_* / __rel_plt_* are
                // local to each module: resolve them against this module's own
                // dynamic table rather than a global definition.
                let mut module_exports = exports.clone();
                if let Ok(dynamic) = crate::homebrew::mod0::parse_dynamic(&module.image, header) {
                    if let Some(rela) = dynamic.rela_offset {
                        module_exports.insert(
                            "__rel_dyn_start".to_string(),
                            module.base.wrapping_add(rela),
                        );
                        module_exports.insert(
                            "__rel_dyn_end".to_string(),
                            module.base.wrapping_add(rela + dynamic.rela_size),
                        );
                    }
                    if let Some(jmprel) = dynamic.jmprel_offset {
                        module_exports.insert(
                            "__rel_plt_start".to_string(),
                            module.base.wrapping_add(jmprel),
                        );
                        module_exports.insert(
                            "__rel_plt_end".to_string(),
                            module.base.wrapping_add(jmprel + dynamic.jmprel_size),
                        );
                    }
                }
                match crate::homebrew::mod0::apply_relocations(
                    &mut module.image,
                    header,
                    module.base,
                    &module_exports,
                ) {
                    Ok(report) => relocations = Some(report),
                    Err(err) => warning = Some(err),
                }
            }
            None => warning = Some("no MOD0 header; module left unrelocated".to_string()),
        }
        outputs.push(write_prepared_nso(
            module,
            segments_dir,
            relocations,
            warning,
            &text_ranges,
            loaded_image.as_deref(),
        )?);
    }
    Ok(outputs)
}

/// Where an NSO-only process image starts (Horizon's ASLR region begins at
/// 0x0800_0000).
const NSO_LOAD_BASE: u64 = 0x0800_0000;

/// Bytes an NSO occupies once loaded: its segments plus .bss after .data.
fn nso_image_size(module: &NsoModule) -> u64 {
    let segments_end = module
        .segments
        .iter()
        .map(|segment| u64::from(segment.memory_offset) + u64::from(segment.size))
        .max()
        .unwrap_or(0);
    segments_end + u64::from(module.bss_size)
}

fn write_unlinked_nso_segments(
    module: &NsoModule,
    base: u64,
    segments_dir: &Path,
) -> Result<LinkedNsoOutput, String> {
    let prepared = prepare_nso(module, base)?;
    let text_ranges = code_ranges(&prepared);
    write_prepared_nso(prepared, segments_dir, None, None, &text_ranges, None)
}

/// Executable ranges a relocated pointer may target. NSO text opens with an
/// 8-byte prologue and the MOD0 header, none of it callable; excluding it
/// keeps null and unresolved imports (0 + addend) out of a module loaded at
/// base 0 without dropping that module's real low-address functions.
fn code_ranges(module: &PreparedNso<'_>) -> Vec<(u64, u64)> {
    module
        .segment_data
        .iter()
        .filter(|segment| segment.segment.kind == NsoSegmentKind::Text)
        .map(|segment| {
            let start = module.base + segment.segment.memory_offset as u64;
            let header = crate::homebrew::mod0::find_mod0(&segment.data).map_or(0, |offset| {
                (offset + crate::homebrew::mod0::MOD0_HEADER_SIZE) as u64
            });
            (start + header, start + segment.data.len() as u64)
        })
        .collect()
}

/// Read the post-relocation value at every relocation site and keep those
/// that land in an executable segment. These are the program's real function
/// pointers; scanning raw data words would treat unrelated integers as code.
///
/// With `loaded_image` (the image before relocation), lazily bound imports
/// also contribute their PLT0 stub: an unrelocated JUMP_SLOT holds the
/// module-relative address of the trampoline rtld routes unresolved calls
/// through, which nothing calls directly.
fn collect_code_pointers(
    module: &PreparedNso<'_>,
    text_ranges: &[(u64, u64)],
    loaded_image: Option<&[u8]>,
) -> Vec<u64> {
    let Some(header) = &module.mod0 else {
        return Vec::new();
    };
    let Ok(dynamic) = crate::homebrew::mod0::parse_dynamic(&module.image, header) else {
        return Vec::new();
    };
    let Ok(relocations) = crate::homebrew::mod0::parse_relocations(&module.image, &dynamic) else {
        return Vec::new();
    };
    let in_text = |value: u64| {
        text_ranges
            .iter()
            .any(|(start, end)| value >= *start && value < *end)
    };
    let read = |image: &[u8], offset: usize| {
        image
            .get(offset..offset + 8)
            .map(|raw| u64::from_le_bytes(raw.try_into().unwrap_or([0; 8])))
    };
    let mut pointers = BTreeMap::new();
    // rtld calls each module's DT_INIT/DT_FINI through the dynamic table.
    for function in [dynamic.init_offset, dynamic.fini_offset]
        .into_iter()
        .flatten()
        .map(|offset| module.base.wrapping_add(offset))
        .filter(|address| in_text(*address))
    {
        pointers.insert(function, ());
    }
    for relocation in relocations {
        let offset = relocation.offset as usize;
        if let Some(value) = read(&module.image, offset).filter(|value| in_text(*value)) {
            pointers.insert(value, ());
        }
        if relocation.kind == crate::homebrew::mod0::R_AARCH64_JUMP_SLOT {
            if let Some(stub) = loaded_image
                .and_then(|image| read(image, offset))
                .filter(|raw| *raw != 0)
                .map(|raw| module.base.wrapping_add(raw))
                .filter(|stub| in_text(*stub))
            {
                pointers.insert(stub, ());
            }
        }
    }
    pointers.into_keys().collect()
}

fn write_prepared_nso(
    module: PreparedNso<'_>,
    segments_dir: &Path,
    relocations: Option<crate::homebrew::mod0::RelocationReport>,
    warning: Option<String>,
    text_ranges: &[(u64, u64)],
    segment_image: Option<&[u8]>,
) -> Result<LinkedNsoOutput, String> {
    let module_dir = segments_dir.join(&module.name);
    fs::create_dir_all(&module_dir).map_err(|err| format!("create module dir: {err}"))?;

    let mut segments = Vec::new();
    let mut generated = Vec::new();
    let mut written = Vec::new();

    for entry in &module.segment_data {
        let start = entry.segment.memory_offset as usize - module.image_start;
        let end = start + entry.data.len();
        // Code pointers come from the relocated image; segment files may be
        // written from the image as loaded instead.
        let data = segment_image
            .unwrap_or(&module.image)
            .get(start..end)
            .ok_or_else(|| {
                format!(
                    "segment {} outside module image",
                    entry.segment.memory_offset
                )
            })?
            .to_vec();
        let file_name = format!("{}.bin", segment_name(entry.segment.kind));
        let output_rel = format!("segments/{}/{file_name}", module.name);
        let output_path = module_dir.join(&file_name);
        fs::write(&output_path, &data)
            .map_err(|err| format!("write NSO segment {file_name}: {err}"))?;
        written.push(output_path.clone());
        generated.push(GeneratedFile {
            path: output_rel.clone(),
            sha256: sha256_bytes(&data),
            size: data.len() as u64,
        });
        segments.push(ModuleSegment {
            name: segment_name(entry.segment.kind).to_string(),
            file_offset: entry.segment.file_offset as u64,
            file_size: data.len() as u64,
            memory_offset: entry.segment.memory_offset as u64,
            memory_size: entry.segment.size as u64,
            permissions: entry.segment.permissions.as_str().to_string(),
            compressed: Some(entry.segment.compressed),
            output_path: output_rel,
        });
    }

    let code_pointers = collect_code_pointers(&module, text_ranges, segment_image);
    let input_sha256 = sha256_path(&module.module.path)?;
    let bss_offset = module
        .module
        .segments
        .iter()
        .find(|segment| segment.kind == NsoSegmentKind::Data)
        .map(|segment| segment.memory_offset as u64 + segment.size as u64)
        .unwrap_or(0);

    let build = ModuleBuild {
        name: module.name,
        format: "nso".to_string(),
        base: module.base,
        input_path: module.module.path.clone(),
        input_sha256,
        input_size: module.module.size,
        build_id: module.module.module_id_hex(),
        segments,
        bss: BssInfo {
            size: module.module.bss_size as u64,
            memory_offset: bss_offset,
        },
        embedded: Some(OffsetInfo {
            offset: module.module.embedded_offset as u64,
            size: module.module.embedded_size as u64,
        }),
        dynstr: Some(OffsetInfo {
            offset: module.module.dynstr_offset as u64,
            size: module.module.dynstr_size as u64,
        }),
        dynsym: Some(OffsetInfo {
            offset: module.module.dynsym_offset as u64,
            size: module.module.dynsym_size as u64,
        }),
        code_pointers,
    };

    Ok(LinkedNsoOutput {
        build,
        generated,
        written,
        relocations,
        warning,
    })
}

fn extract_assets(
    module: &NroModule,
    root_dir: &Path,
    assets_dir: &Path,
    records: &mut Vec<AssetRecord>,
) -> Result<(Vec<GeneratedFile>, Vec<PathBuf>), String> {
    let Some(assets) = module.assets.clone() else {
        return Ok((Vec::new(), Vec::new()));
    };
    let bytes = fs::read(&module.path)
        .map_err(|err| format!("read NRO {}: {err}", module.path.display()))?;
    let mut generated = Vec::new();
    let mut written = Vec::new();

    if assets.icon.size > 0 {
        let (path, info) = extract_asset(
            &bytes,
            &assets,
            assets.icon,
            root_dir,
            assets_dir,
            "icon.bin",
            "icon",
        )?;
        records.push(info);
        generated.push(path.generated_file);
        written.push(path.path);
    }

    if assets.nacp.size > 0 {
        let (path, info) = extract_asset(
            &bytes,
            &assets,
            assets.nacp,
            root_dir,
            assets_dir,
            "control.nacp",
            "nacp",
        )?;
        if info.size != 0x4000 {
            return Err(format!(
                "NACP size mismatch: expected 0x4000, got {}",
                info.size
            ));
        }
        records.push(info);
        generated.push(path.generated_file);
        written.push(path.path);
    }

    if assets.romfs.size > 0 {
        let romfs_dir = assets_dir.join("romfs");
        fs::create_dir_all(&romfs_dir).map_err(|err| format!("create romfs dir: {err}"))?;
        let (romfs_bytes, romfs_base_offset) = extract_asset_bytes(&bytes, &assets, assets.romfs)?;
        let entries = list_romfs_entries(&romfs_bytes)?;
        let (generated_entries, written_entries) = write_romfs_entries(
            &romfs_bytes,
            &entries,
            romfs_base_offset,
            root_dir,
            &romfs_dir,
            "romfs",
            records,
        )?;
        generated.extend(generated_entries);
        written.extend(written_entries);
    }

    Ok((generated, written))
}

struct AssetWrite {
    path: PathBuf,
    generated_file: GeneratedFile,
}

fn extract_asset(
    bytes: &[u8],
    assets: &crate::homebrew::nro::NroAssetHeader,
    section: crate::homebrew::nro::NroAssetSection,
    root_dir: &Path,
    out_dir: &Path,
    file_name: &str,
    kind: &str,
) -> Result<(AssetWrite, AssetRecord), String> {
    let start = assets
        .base_offset
        .checked_add(section.offset)
        .ok_or_else(|| "asset offset overflow".to_string())? as usize;
    let end = start
        .checked_add(section.size as usize)
        .ok_or_else(|| "asset size overflow".to_string())?;
    if end > bytes.len() {
        return Err(format!(
            "asset out of range: {}..{} (len={})",
            start,
            end,
            bytes.len()
        ));
    }
    let data = &bytes[start..end];
    let out_path = out_dir.join(file_name);
    fs::write(&out_path, data).map_err(|err| format!("write asset {file_name}: {err}"))?;

    let rel = out_path
        .strip_prefix(root_dir)
        .unwrap_or(&out_path)
        .to_string_lossy()
        .replace('\\', "/");

    let generated_file = GeneratedFile {
        path: rel.clone(),
        sha256: sha256_bytes(data),
        size: data.len() as u64,
    };
    let record = AssetRecord {
        kind: kind.to_string(),
        path: rel,
        sha256: generated_file.sha256.clone(),
        size: generated_file.size,
        source_offset: section.offset,
        source_size: section.size,
    };

    Ok((
        AssetWrite {
            path: out_path,
            generated_file,
        },
        record,
    ))
}

fn extract_asset_bytes(
    bytes: &[u8],
    assets: &crate::homebrew::nro::NroAssetHeader,
    section: crate::homebrew::nro::NroAssetSection,
) -> Result<(Vec<u8>, u64), String> {
    let start = assets
        .base_offset
        .checked_add(section.offset)
        .ok_or_else(|| "asset offset overflow".to_string())? as usize;
    let end = start
        .checked_add(section.size as usize)
        .ok_or_else(|| "asset size overflow".to_string())?;
    if end > bytes.len() {
        return Err(format!(
            "asset out of range: {}..{} (len={})",
            start,
            end,
            bytes.len()
        ));
    }
    Ok((bytes[start..end].to_vec(), start as u64))
}

fn write_romfs_entries(
    romfs_bytes: &[u8],
    entries: &[RomfsEntry],
    romfs_base_offset: u64,
    root_dir: &Path,
    romfs_dir: &Path,
    kind: &str,
    records: &mut Vec<AssetRecord>,
) -> Result<(Vec<GeneratedFile>, Vec<PathBuf>), String> {
    let mut generated = Vec::new();
    let mut written = Vec::new();
    for entry in entries {
        let rel_path = Path::new(&entry.path);
        if rel_path.is_absolute() {
            return Err(format!("romfs entry path is absolute: {}", entry.path));
        }
        for component in rel_path.components() {
            match component {
                std::path::Component::Normal(_) => {}
                _ => {
                    return Err(format!(
                        "romfs entry path contains invalid component: {}",
                        entry.path
                    ))
                }
            }
        }

        let out_path = romfs_dir.join(rel_path);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("create romfs dir {}: {err}", parent.display()))?;
        }

        let start = entry.data_offset as usize;
        let end = start
            .checked_add(entry.data_size as usize)
            .ok_or_else(|| "romfs file size overflow".to_string())?;
        if end > romfs_bytes.len() {
            return Err(format!(
                "romfs entry out of range: {}..{} (len={})",
                start,
                end,
                romfs_bytes.len()
            ));
        }
        let data = &romfs_bytes[start..end];
        fs::write(&out_path, data)
            .map_err(|err| format!("write romfs entry {}: {err}", out_path.display()))?;

        let rel = out_path
            .strip_prefix(root_dir)
            .unwrap_or(&out_path)
            .to_string_lossy()
            .replace('\\', "/");
        let generated_file = GeneratedFile {
            path: rel.clone(),
            sha256: sha256_bytes(data),
            size: data.len() as u64,
        };
        let record = AssetRecord {
            kind: kind.to_string(),
            path: rel,
            sha256: generated_file.sha256.clone(),
            size: generated_file.size,
            source_offset: romfs_base_offset
                .checked_add(entry.data_offset)
                .ok_or_else(|| "romfs source offset overflow".to_string())?,
            source_size: entry.data_size,
        };
        records.push(record);
        generated.push(generated_file);
        written.push(out_path);
    }

    Ok((generated, written))
}

fn ensure_input_present(
    inputs: &[crate::provenance::ValidatedInput],
    path: &Path,
) -> Result<(), String> {
    if inputs.iter().any(|input| input.path == path) {
        Ok(())
    } else {
        Err(format!(
            "input {} not listed in provenance metadata",
            path.display()
        ))
    }
}

fn enforce_homebrew_formats(inputs: &[crate::provenance::ValidatedInput]) -> Result<(), String> {
    let mut disallowed = BTreeMap::new();
    for input in inputs {
        match input.format {
            InputFormat::Nro0 | InputFormat::Nso0 | InputFormat::Nrr0 | InputFormat::Npdm => {}
            other => {
                disallowed
                    .entry(other.as_str())
                    .or_insert_with(Vec::new)
                    .push(input.path.clone());
            }
        }
    }
    if disallowed.is_empty() {
        return Ok(());
    }
    let mut message = String::from("disallowed input formats for homebrew intake:\n");
    for (format, paths) in disallowed {
        for path in paths {
            message.push_str(&format!("- {format}: {}\n", path.display()));
        }
    }
    Err(message)
}

fn segment_name(kind: NsoSegmentKind) -> &'static str {
    match kind {
        NsoSegmentKind::Text => "text",
        NsoSegmentKind::Rodata => "rodata",
        NsoSegmentKind::Data => "data",
    }
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

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    hex_bytes(&digest)
}

fn sha256_path(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|err| err.to_string())?;
    Ok(sha256_bytes(&bytes))
}
