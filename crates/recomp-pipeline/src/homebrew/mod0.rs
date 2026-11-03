//! MOD0 module-header and ELF relocation support for NSO executables.
//!
//! Nintendo's MOD0 header is the runtime linker's replacement for
//! `PT_DYNAMIC`. All fields are signed 32-bit offsets relative to the MOD0
//! magic (`nn::rocrt::ModuleHeader`):
//!
//! ```text
//! 0x00 magic "MOD0"
//! 0x04 .dynamic offset
//! 0x08 .bss start offset
//! 0x0C .bss end offset
//! 0x10 .eh_frame_hdr start offset
//! 0x14 .eh_frame_hdr end offset
//! 0x18 module-object offset (usually == .bss start)
//! ```
//!
//! `.dynamic` is a standard `Elf64_Dyn` array (u64 tag, u64 value). The
//! relocation tables (`DT_RELA`, `DT_JMPREL`) contain standard
//! `Elf64_Rela` entries. Retail NSOs use:
//!
//! - `R_AARCH64_RELATIVE` (1027): value = load base + addend
//! - `R_AARCH64_ABS64` (257): value = symbol + addend
//! - `R_AARCH64_GLOB_DAT` (1025) / `JUMP_SLOT` (1026): symbol address
//!
//! A static recompiler applies these itself instead of relying on rtld.

use crate::homebrew::nso::NsoSegmentData;

const MOD0_MAGIC: u32 = 0x3044_4F4D;
/// Magic plus six offset fields.
pub const MOD0_HEADER_SIZE: usize = 0x1C;

pub const DT_NULL: u64 = 0;
pub const DT_NEEDED: u64 = 1;
pub const DT_STRTAB: u64 = 5;
pub const DT_SYMTAB: u64 = 6;
pub const DT_RELA: u64 = 7;
pub const DT_RELASZ: u64 = 8;
pub const DT_RELAENT: u64 = 9;
pub const DT_STRSZ: u64 = 10;
pub const DT_SYMENT: u64 = 11;
pub const DT_SONAME: u64 = 14;
pub const DT_HASH: u64 = 4;
pub const DT_GNU_HASH: u64 = 0x6fff_fef5;
pub const DT_JMPREL: u64 = 23;
pub const DT_PLTRELSZ: u64 = 2;
pub const DT_INIT: u64 = 12;
pub const DT_FINI: u64 = 13;

pub const R_AARCH64_NONE: u32 = 0;
pub const R_AARCH64_ABS64: u32 = 257;
pub const R_AARCH64_GLOB_DAT: u32 = 1025;
pub const R_AARCH64_JUMP_SLOT: u32 = 1026;
pub const R_AARCH64_RELATIVE: u32 = 1027;

const SHN_UNDEF: u16 = 0;
const SHN_ABS: u16 = 0xfff1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mod0Header {
    /// Image-relative offset of the `.dynamic` array.
    pub dynamic_offset: u64,
    pub bss_start: u64,
    pub bss_end: u64,
    pub eh_frame_hdr_start: u64,
    pub eh_frame_hdr_end: u64,
    pub module_object: u64,
}

#[derive(Debug, Clone, Default)]
pub struct DynamicInfo {
    pub rela_offset: Option<u64>,
    pub rela_size: u64,
    pub rela_ent: u64,
    pub jmprel_offset: Option<u64>,
    pub jmprel_size: u64,
    pub symtab_offset: Option<u64>,
    pub syment: u64,
    pub strtab_offset: Option<u64>,
    pub strsz: u64,
    pub soname_offset: Option<u64>,
    /// Module initializer and finalizer (DT_INIT / DT_FINI), image-relative.
    pub init_offset: Option<u64>,
    pub fini_offset: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name_offset: u32,
    pub info: u8,
    pub other: u8,
    pub shndx: u16,
    pub value: u64,
    pub size: u64,
    pub name: String,
}

impl Symbol {
    pub fn is_defined(&self) -> bool {
        self.shndx != SHN_UNDEF
    }

    pub fn is_absolute(&self) -> bool {
        self.shndx == SHN_ABS
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Relocation {
    pub offset: u64,
    pub kind: u32,
    pub symbol: u32,
    pub addend: i64,
}

#[derive(Debug, Clone, Default)]
pub struct RelocationReport {
    pub relative: usize,
    pub symbol: usize,
    pub unresolved: usize,
    pub unresolved_names: Vec<String>,
    pub skipped: usize,
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let raw: [u8; 4] = bytes.get(offset..offset + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(raw))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    let raw: [u8; 8] = bytes.get(offset..offset + 8)?.try_into().ok()?;
    Some(u64::from_le_bytes(raw))
}

fn read_i32(bytes: &[u8], offset: usize) -> Option<i32> {
    read_u32(bytes, offset).map(|value| value as i32)
}

fn read_i64(bytes: &[u8], offset: usize) -> Option<i64> {
    read_u64(bytes, offset).map(|value| value as i64)
}

/// Resolve a signed 32-bit MOD0 field relative to the magic location.
fn resolve_field(image: &[u8], magic: usize, field: usize) -> Option<u64> {
    let relative = read_i32(image, field)? as i64;
    let absolute = magic as i64 + relative;
    (absolute >= 0).then_some(absolute as u64)
}

/// Parse a MOD0 header whose magic is at `offset` inside a memory image.
pub fn parse_mod0(image: &[u8], offset: usize) -> Option<Mod0Header> {
    if read_u32(image, offset)? != MOD0_MAGIC {
        return None;
    }
    Some(Mod0Header {
        dynamic_offset: resolve_field(image, offset, offset + 0x04)?,
        bss_start: resolve_field(image, offset, offset + 0x08)?,
        bss_end: resolve_field(image, offset, offset + 0x0C)?,
        eh_frame_hdr_start: resolve_field(image, offset, offset + 0x10)?,
        eh_frame_hdr_end: resolve_field(image, offset, offset + 0x14)?,
        module_object: resolve_field(image, offset, offset + 0x18)?,
    })
}

/// Locate the MOD0 header of a module image that starts at .text. Word 1 of
/// .text holds the MOD0 offset (rtld reads it the same way); most modules put
/// the header right after, while rtld keeps it in .rodata. Images without a
/// usable offset are scanned near the start.
pub fn find_mod0(image: &[u8]) -> Option<usize> {
    if let Some(offset) = read_u32(image, 4).map(|offset| offset as usize) {
        if offset.is_multiple_of(4) && read_u32(image, offset) == Some(MOD0_MAGIC) {
            return Some(offset);
        }
    }
    let limit = image.len().min(64);
    (0..limit / 4)
        .find(|&i| read_u32(image, i * 4) == Some(MOD0_MAGIC))
        .map(|i| i * 4)
}

/// Walk the `.dynamic` array referenced by a MOD0 header.
pub fn parse_dynamic(image: &[u8], header: &Mod0Header) -> Result<DynamicInfo, String> {
    let mut info = DynamicInfo {
        rela_size: 0,
        rela_ent: 24,
        jmprel_size: 0,
        syment: 24,
        strsz: 0,
        ..DynamicInfo::default()
    };
    let start = header.dynamic_offset as usize;
    if start >= image.len() {
        return Err(format!(
            "dynamic offset {:#x} outside image {:#x}",
            start,
            image.len()
        ));
    }
    for index in 0..4096 {
        let entry = start
            .checked_add(index * 16)
            .ok_or_else(|| "dynamic table overflow".to_string())?;
        let tag =
            read_u64(image, entry).ok_or_else(|| format!("dynamic entry {index} outside image"))?;
        let value = read_u64(image, entry + 8)
            .ok_or_else(|| format!("dynamic entry {index} outside image"))?;
        match tag {
            DT_NULL => break,
            DT_RELA => info.rela_offset = Some(value),
            DT_RELASZ => info.rela_size = value,
            DT_RELAENT if value != 0 => info.rela_ent = value,
            DT_JMPREL => info.jmprel_offset = Some(value),
            DT_PLTRELSZ => info.jmprel_size = value,
            DT_SYMTAB => info.symtab_offset = Some(value),
            DT_SYMENT if value != 0 => info.syment = value,
            DT_STRTAB => info.strtab_offset = Some(value),
            DT_STRSZ => info.strsz = value,
            DT_SONAME => info.soname_offset = Some(value),
            DT_INIT => info.init_offset = Some(value),
            DT_FINI => info.fini_offset = Some(value),
            _ => {}
        }
    }
    Ok(info)
}

fn read_cstr(image: &[u8], offset: usize) -> Option<String> {
    let rest = image.get(offset..)?;
    let end = rest.iter().position(|byte| *byte == 0)?;
    Some(String::from_utf8_lossy(&rest[..end]).to_string())
}

impl DynamicInfo {
    pub fn string(&self, image: &[u8], offset: u64) -> Option<String> {
        let strtab = self.strtab_offset? as usize;
        read_cstr(image, strtab.checked_add(offset as usize)?)
    }
}

/// Read the dynamic symbol table. The table extent is derived from the
/// distance to `.dynstr` (Nintendo emits symtab directly before strtab).
pub fn parse_symbols(image: &[u8], info: &DynamicInfo) -> Result<Vec<Symbol>, String> {
    let Some(symtab) = info.symtab_offset else {
        return Ok(Vec::new());
    };
    let Some(strtab) = info.strtab_offset else {
        return Ok(Vec::new());
    };
    if info.syment < 24 {
        return Err(format!("invalid DT_SYMENT {}", info.syment));
    }
    if strtab <= symtab {
        return Ok(Vec::new());
    }
    let count = ((strtab - symtab) / info.syment) as usize;
    let mut symbols = Vec::with_capacity(count);
    for index in 0..count {
        let base = symtab as usize + index * info.syment as usize;
        let name_offset =
            read_u32(image, base).ok_or_else(|| format!("symbol {index} outside image"))?;
        let info_byte = image
            .get(base + 4)
            .copied()
            .ok_or_else(|| format!("symbol {index} outside image"))?;
        let other = image
            .get(base + 5)
            .copied()
            .ok_or_else(|| format!("symbol {index} outside image"))?;
        let shndx = u16::from_le_bytes(
            image
                .get(base + 6..base + 8)
                .ok_or_else(|| format!("symbol {index} outside image"))?
                .try_into()
                .unwrap_or([0; 2]),
        );
        let value =
            read_u64(image, base + 8).ok_or_else(|| format!("symbol {index} outside image"))?;
        let size =
            read_u64(image, base + 16).ok_or_else(|| format!("symbol {index} outside image"))?;
        let name = info.string(image, name_offset as u64).unwrap_or_default();
        symbols.push(Symbol {
            name_offset,
            info: info_byte,
            other,
            shndx,
            value,
            size,
            name,
        });
    }
    Ok(symbols)
}

/// Collect `DT_RELA` and `DT_JMPREL` relocations in table order.
pub fn parse_relocations(image: &[u8], info: &DynamicInfo) -> Result<Vec<Relocation>, String> {
    let mut out = Vec::new();
    let mut read_table = |offset: u64, size: u64| -> Result<(), String> {
        if size == 0 {
            return Ok(());
        }
        let ent = if info.rela_ent == 0 {
            24
        } else {
            info.rela_ent
        };
        if ent < 24 {
            return Err(format!("invalid relocation entry size {ent}"));
        }
        let count = size / ent;
        for index in 0..count {
            let base = offset as usize + index as usize * ent as usize;
            let target =
                read_u64(image, base).ok_or_else(|| format!("relocation {index} outside image"))?;
            let r_info = read_u64(image, base + 8)
                .ok_or_else(|| format!("relocation {index} outside image"))?;
            let addend = read_i64(image, base + 16)
                .ok_or_else(|| format!("relocation {index} outside image"))?;
            out.push(Relocation {
                offset: target,
                kind: (r_info & 0xFFFF_FFFF) as u32,
                symbol: (r_info >> 32) as u32,
                addend,
            });
        }
        Ok(())
    };
    if let Some(offset) = info.rela_offset {
        read_table(offset, info.rela_size)?;
    }
    if let Some(offset) = info.jmprel_offset {
        read_table(offset, info.jmprel_size)?;
    }
    Ok(out)
}

/// Write a relocated 64-bit value into the image.
pub fn write_relocation(
    image: &mut [u8],
    relocation: &Relocation,
    value: u64,
) -> Result<(), String> {
    let target = relocation.offset as usize;
    if target + 8 > image.len() {
        return Err(format!(
            "relocation target {:#x} outside image {:#x}",
            relocation.offset,
            image.len()
        ));
    }
    image[target..target + 8].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// Apply one module's relocations in isolation. `exports` maps an imported
/// symbol name to its final address in the linked image.
pub fn apply_relocations(
    image: &mut [u8],
    header: &Mod0Header,
    base: u64,
    exports: &std::collections::BTreeMap<String, u64>,
) -> Result<RelocationReport, String> {
    let info = parse_dynamic(image, header)?;
    let symbols = parse_symbols(image, &info)?;
    let relocations = parse_relocations(image, &info)?;
    let mut report = RelocationReport::default();

    for relocation in &relocations {
        match relocation.kind {
            R_AARCH64_NONE => {
                report.skipped += 1;
            }
            R_AARCH64_RELATIVE => {
                let value = base.wrapping_add(relocation.addend as u64);
                write_relocation(image, relocation, value)?;
                report.relative += 1;
            }
            R_AARCH64_ABS64 | R_AARCH64_GLOB_DAT | R_AARCH64_JUMP_SLOT => {
                let symbol = symbols.get(relocation.symbol as usize);
                let symbol_value = match symbol {
                    Some(symbol) if symbol.is_defined() => {
                        let module_base = if symbol.is_absolute() { 0 } else { base };
                        module_base.wrapping_add(symbol.value)
                    }
                    Some(symbol) => match exports.get(&symbol.name) {
                        Some(address) => *address,
                        None => {
                            report.unresolved += 1;
                            if report.unresolved_names.len() < 32 && !symbol.name.is_empty() {
                                report.unresolved_names.push(symbol.name.clone());
                            }
                            0
                        }
                    },
                    None => {
                        report.unresolved += 1;
                        0
                    }
                };
                let value = symbol_value.wrapping_add(relocation.addend as u64);
                write_relocation(image, relocation, value)?;
                report.symbol += 1;
            }
            other => {
                report.skipped += 1;
                report
                    .unresolved_names
                    .push(format!("unsupported relocation type {other:#x}"));
            }
        }
    }
    Ok(report)
}

/// Assemble a module memory image from extracted NSO segments. Addresses are
/// module-relative (lowest segment memory offset becomes image offset 0).
pub fn build_memory_image(
    segments: &[NsoSegmentData],
    total_size: usize,
) -> Result<Vec<u8>, String> {
    let base = segments
        .iter()
        .map(|s| s.segment.memory_offset as usize)
        .min()
        .ok_or("no segments")?;
    let mut image = vec![0u8; total_size];
    for segment in segments {
        let start = segment.segment.memory_offset as usize - base;
        let end = start + segment.data.len();
        if end > image.len() {
            return Err(format!(
                "segment at {start:#x} exceeds image {total_size:#x}"
            ));
        }
        image[start..end].copy_from_slice(&segment.data);
    }
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u32(image: &mut [u8], offset: usize, value: u32) {
        image[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(image: &mut [u8], offset: usize, value: u64) {
        image[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn put_i32(image: &mut [u8], offset: usize, value: i32) {
        image[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// Build a minimal Nintendo-style module image with a dynamic table and
    /// one RELATIVE plus one symbol relocation.
    fn build_image() -> Vec<u8> {
        let mut image = vec![0u8; 0x400];
        // MOD0 at 0x08, all fields relative to the magic.
        put_u32(&mut image, 0x08, MOD0_MAGIC);
        put_i32(&mut image, 0x0C, 0x300 - 0x08); // dynamic
        put_i32(&mut image, 0x10, 0x380 - 0x08); // bss start
        put_i32(&mut image, 0x14, 0x400 - 0x08); // bss end
        put_i32(&mut image, 0x18, 0x380 - 0x08);
        put_i32(&mut image, 0x1C, 0x380 - 0x08);
        put_i32(&mut image, 0x20, 0x380 - 0x08);

        // Dynamic array at 0x300.
        let dyn_base = 0x300;
        put_u64(&mut image, dyn_base, DT_RELA);
        put_u64(&mut image, dyn_base + 8, 0x200);
        put_u64(&mut image, dyn_base + 16, DT_RELASZ);
        put_u64(&mut image, dyn_base + 24, 72);
        put_u64(&mut image, dyn_base + 32, DT_RELAENT);
        put_u64(&mut image, dyn_base + 40, 24);
        put_u64(&mut image, dyn_base + 48, DT_SYMTAB);
        put_u64(&mut image, dyn_base + 56, 0x100);
        put_u64(&mut image, dyn_base + 64, DT_SYMENT);
        put_u64(&mut image, dyn_base + 72, 24);
        put_u64(&mut image, dyn_base + 80, DT_STRTAB);
        put_u64(&mut image, dyn_base + 88, 0x148);
        put_u64(&mut image, dyn_base + 96, DT_NULL);
        put_u64(&mut image, dyn_base + 104, 0);

        // Symbols: [0] undef, [1] exported "target" (defined), [2] imported
        // "ext". The symbol count is derived from the gap to .dynstr.
        put_u32(&mut image, 0x100, 0);
        put_u32(&mut image, 0x118, 1); // name offset "target"
        image[0x11C] = 0x12;
        image[0x11E..0x120].copy_from_slice(&1_u16.to_le_bytes()); // defined
        put_u64(&mut image, 0x120, 0x280); // st_value
        put_u32(&mut image, 0x130, 8); // name offset "ext"
        put_u64(&mut image, 0x138, 0);
        // String table at 0x148, directly after three Elf64_Sym entries.
        image[0x148..0x148 + 12].copy_from_slice(b"\0target\0ext\0");
        // Rela entries at 0x200: RELATIVE, GLOB_DAT "ext", ABS64 "ext".
        put_u64(&mut image, 0x200, 0x360);
        put_u64(&mut image, 0x208, R_AARCH64_RELATIVE as u64);
        put_u64(&mut image, 0x210, 0x1234);
        put_u64(&mut image, 0x218, 0x368);
        put_u64(
            &mut image,
            0x220,
            ((2_u64) << 32) | R_AARCH64_GLOB_DAT as u64,
        );
        put_u64(&mut image, 0x228, 0);
        put_u64(&mut image, 0x230, 0x370);
        put_u64(&mut image, 0x238, ((2_u64) << 32) | R_AARCH64_ABS64 as u64);
        put_u64(&mut image, 0x240, 0x10);
        image
    }

    #[test]
    fn parses_mod0_header() {
        let image = build_image();
        let header = parse_mod0(&image, 0x08).expect("mod0 header");
        assert_eq!(header.dynamic_offset, 0x300);
        assert_eq!(header.bss_start, 0x380);
        assert_eq!(header.bss_end, 0x400);
        assert_eq!(header.module_object, 0x380);
    }

    #[test]
    fn parses_symbols_and_relocations() {
        let image = build_image();
        let header = parse_mod0(&image, 0x08).unwrap();
        let info = parse_dynamic(&image, &header).unwrap();
        let symbols = parse_symbols(&image, &info).unwrap();
        assert_eq!(symbols[1].name, "target");
        assert_eq!(symbols[1].value, 0x280);
        assert!(symbols[1].is_defined());
        assert_eq!(symbols[2].name, "ext");
        assert!(!symbols[2].is_defined());
        let relocs = parse_relocations(&image, &info).unwrap();
        assert_eq!(relocs.len(), 3);
        assert_eq!(relocs[0].kind, R_AARCH64_RELATIVE);
        assert_eq!(relocs[2].kind, R_AARCH64_ABS64);
        assert_eq!(relocs[2].symbol, 2);
    }

    #[test]
    fn applies_relative_and_symbol_relocations() {
        let mut image = build_image();
        let header = parse_mod0(&image, 0x08).unwrap();
        let mut exports = std::collections::BTreeMap::new();
        exports.insert("ext".to_string(), 0xDEAD_0000);
        let report = apply_relocations(&mut image, &header, 0x1000, &exports).unwrap();
        assert_eq!(report.relative, 1);
        assert_eq!(report.symbol, 2);
        assert_eq!(report.unresolved, 0);
        // RELATIVE: base + addend.
        assert_eq!(
            u64::from_le_bytes(image[0x360..0x368].try_into().unwrap()),
            0x2234
        );
        // GLOB_DAT: symbol address.
        assert_eq!(
            u64::from_le_bytes(image[0x368..0x370].try_into().unwrap()),
            0xDEAD_0000
        );
        // ABS64: symbol + addend.
        assert_eq!(
            u64::from_le_bytes(image[0x370..0x378].try_into().unwrap()),
            0xDEAD_0010
        );
    }

    #[test]
    fn reports_unresolved_symbols() {
        let mut image = build_image();
        let header = parse_mod0(&image, 0x08).unwrap();
        let report =
            apply_relocations(&mut image, &header, 0, &std::collections::BTreeMap::new()).unwrap();
        assert_eq!(report.unresolved, 2);
        assert_eq!(
            report.unresolved_names,
            vec!["ext".to_string(), "ext".to_string()]
        );
    }

    #[test]
    fn rejects_dynamic_outside_image() {
        let mut image = vec![0u8; 0x40];
        put_u32(&mut image, 0x08, MOD0_MAGIC);
        put_i32(&mut image, 0x0C, 0x1000);
        let header = parse_mod0(&image, 0x08).unwrap();
        assert!(parse_dynamic(&image, &header).is_err());
    }
}
