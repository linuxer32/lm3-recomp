#!/usr/bin/env python3
"""Generate the nso-boot sample: NSO with MOD0 tables + IPC boot sequence.

Layout (image-relative offsets; text memory offset is 0):
  text   : MOD0 header + relocation tables, then boot instructions
  rodata : filler
  data   : named port string, handle slot, HIPC request buffer, RomFS blob

Boot instructions:
  svcConnectToNamedPort("fs:GLOBAL") -> svcSendSyncRequest(OpenDataFileSystemByCurrentProcess)
  svcExitProcess(0)

All inputs are synthetic; no proprietary content.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"


def movz(rd: int, imm: int) -> int:
    assert imm <= 0xFFFF
    return 0xD2800000 | (imm << 6) | rd


def svc(imm: int) -> int:
    return 0xD4000000 | (imm << 5)


RET = 0xD65F03C0


def write_u32(buf: bytearray, offset: int, value: int) -> None:
    buf[offset : offset + 4] = struct.pack("<I", value)


def write_u64(buf: bytearray, offset: int, value: int) -> None:
    buf[offset : offset + 8] = struct.pack("<Q", value)


def build_romfs(files: list[tuple[str, bytes]]) -> bytearray:
    """RomFS image in the libnx romfs_header layout (u64 fields)."""
    dirs: list[tuple[int, str]] = [(0, "")]
    dir_table = bytearray()

    def dir_entry(parent: int, name: str) -> None:
        entry = bytearray(24)
        entry[0:8] = struct.pack("<Q", 0)
        entry[8:16] = struct.pack("<Q", 0)
        entry[16:20] = struct.pack("<I", parent)
        entry[20:24] = struct.pack("<I", len(name))
        entry.extend(name.encode())
        while len(entry) % 4:
            entry.append(0)
        dir_table.extend(entry)

    for parent, name in dirs:
        dir_entry(parent, name)

    file_table = bytearray()
    data = bytearray()
    for path, payload in files:
        segments = path.split("/")
        parent = 0
        for seg in segments[:-1]:
            match = next((i for i, (p, n) in enumerate(dirs) if p == parent and n == seg), None)
            if match is None:
                dirs.append((parent, seg))
                parent = len(dirs) - 1
                dir_entry(parent, seg)
            else:
                parent = match
        name = segments[-1]
        offset = len(data)
        data.extend(payload)
        entry = bytearray(24)
        entry[0:8] = struct.pack("<Q", offset)
        entry[8:16] = struct.pack("<Q", len(payload))
        entry[16:20] = struct.pack("<I", parent)
        entry[20:24] = struct.pack("<I", len(name))
        entry.extend(name.encode())
        while len(entry) % 4:
            entry.append(0)
        file_table.extend(entry)

    header_size = 0x58
    dir_hash_off = header_size
    dir_hash_size = 16
    dir_table_off = dir_hash_off + dir_hash_size
    file_hash_off = dir_table_off + len(dir_table)
    file_hash_size = 16
    file_table_off = file_hash_off + file_hash_size
    data_off = file_table_off + len(file_table)

    buf = bytearray(data_off + len(data))
    write_u64(buf, 0x00, header_size)
    write_u64(buf, 0x08, dir_hash_off)
    write_u64(buf, 0x10, dir_hash_size)
    write_u64(buf, 0x18, dir_table_off)
    write_u64(buf, 0x20, len(dir_table))
    write_u64(buf, 0x28, file_hash_off)
    write_u64(buf, 0x30, file_hash_size)
    write_u64(buf, 0x38, file_table_off)
    write_u64(buf, 0x40, len(file_table))
    write_u64(buf, 0x48, data_off)
    write_u64(buf, 0x50, len(data))
    buf[dir_hash_off : dir_hash_off + dir_hash_size] = bytes(dir_hash_size)
    buf[dir_table_off : dir_table_off + len(dir_table)] = dir_table
    buf[file_hash_off : file_hash_off + file_hash_size] = bytes(file_hash_size)
    buf[file_table_off : file_table_off + len(file_table)] = file_table
    buf[data_off : data_off + len(data)] = data
    return buf


def build_nro(path: Path) -> None:
    header_size = 0x80
    text = struct.pack("<5I", movz(0, 0x1000), svc(1), movz(0, 0), svc(7), RET)
    rodata = b"RODATA-SEGMENT"
    data = b"DATA-SEGMENT"
    text_off = header_size
    ro_off = text_off + len(text)
    data_off = ro_off + len(rodata)
    total = header_size + len(text) + len(rodata) + len(data)
    buf = bytearray(total)
    buf[0x10:0x14] = b"NRO0"
    write_u32(buf, 0x18, total)
    write_u32(buf, 0x20, text_off)
    write_u32(buf, 0x24, len(text))
    write_u32(buf, 0x28, ro_off)
    write_u32(buf, 0x2C, len(rodata))
    write_u32(buf, 0x30, data_off)
    write_u32(buf, 0x34, len(data))
    write_u32(buf, 0x38, 0x20)
    buf[0x40:0x60] = b"SYNTHETIC-NRO-BUILD-ID".ljust(0x20, b"0")
    buf[text_off:text_off + len(text)] = text
    buf[ro_off:ro_off + len(rodata)] = rodata
    buf[data_off:data_off + len(data)] = data
    path.write_bytes(buf)


def build_boot_nso(path: Path, romfs: bytes) -> None:
    header_size = 0x100
    text_mem = 0x0
    ro_mem = 0x1000
    data_mem = 0x2000

    # MOD0 header at text offset 0; tables packed after the header (+0x1C).
    # Field semantics: rel64 boundary = end of abs64 table, rel32 boundary =
    # end of rel64 table, abs64 boundary = end of rel32 table, ranges end =
    # end of exec ranges.
    text = bytearray(0x100)
    write_u32(text, 0x00, 0x3044_4F4D)  # "MOD0"
    write_u32(text, 0x04, 0x38)         # rel32 boundary (end of rel64 table)
    write_u32(text, 0x08, 0x28)         # rel64 boundary (end of abs64 table)
    write_u32(text, 0x0C, 0x40)         # abs64 boundary (end of rel32 table)
    write_u32(text, 0x10, 0x48)         # exec start
    write_u32(text, 0x14, 0x48)         # exec end
    write_u32(text, 0x18, 0x40)         # exec ranges end
    # abs64 table 0x1C..0x28: entry = (offset u32, value u64) storing the port
    # address at image offset 0x2010.
    write_u32(text, 0x1C, 0x2010)
    write_u64(text, 0x20, 0x2000)
    # rel64 table 0x28..0x38: entry at 0x28 (offset, addend) -> image 0x2030 += 0x40
    write_u32(text, 0x28, 0x2030)
    write_u64(text, 0x2C, 0x40)
    # rel32 table 0x38..0x40: entry (offset u32, addend u32) -> image 0x2040 += 0x20
    write_u32(text, 0x38, 0x2040)
    write_u32(text, 0x3C, 0x20)

    code = struct.pack(
        "<9I",
        movz(0, 0x2020),      # out handle pointer
        movz(1, 0x2000),      # port name pointer ("fs:GLOBAL")
        svc(0x1F),            # svcConnectToNamedPort
        movz(0, 0x1),         # session handle returned by the port
        movz(1, 0x2100),      # HIPC request buffer
        svc(0x21),            # svcSendSyncRequest (OpenDataFileSystemByCurrentProcess)
        movz(0, 0),           # exit code 0
        svc(0x07),            # svcExitProcess
        RET,
    )
    text[0x48:0x48 + len(code)] = code
    rodata = bytearray(b"RODATA-PAD" * 8)

    data = bytearray(0x2800 + len(romfs) + 0x400)
    data[0x00:0x10] = b"fs:GLOBAL\0\0\0\0\0\0\0"
    write_u64(data, 0x10, 0xDEAD)          # abs64 target slot
    write_u64(data, 0x20, 0)               # out-handle slot
    write_u64(data, 0x30, 0x100)           # rel64 target slot
    write_u32(data, 0x40, 5)               # rel32 target slot

    # HIPC request at image offset 0x2100 (data+0x100): header0 magic, header1
    # raw words, CMIF SFCI header with method 2 (OpenDataFileSystemByCurrentProcess).
    req = 0x100
    write_u32(data, req, 0x4000)                       # Header0Tag magic
    write_u32(data, req + 4, 5)                        # Header1Tag: raw_words=5
    write_u32(data, req + 8, 0x49434653)               # SFCI magic
    write_u32(data, req + 12, 1)                       # CMIF version
    write_u32(data, req + 16, 2)                       # command id
    write_u32(data, req + 20, 0)                       # token
    data[0x2800:0x2800 + len(romfs)] = romfs

    text_off = header_size
    ro_off = text_off + len(text)
    data_off = ro_off + len(rodata)
    total = header_size + len(text) + len(rodata) + len(data)
    buf = bytearray(total)
    buf[0x0:0x4] = b"NSO0"
    write_u32(buf, 0x8, 0)                 # reserved
    write_u32(buf, 0xC, 0)                 # flags: segments uncompressed
    write_u32(buf, 0x10, text_off)
    write_u32(buf, 0x14, text_mem)
    write_u32(buf, 0x18, len(text))
    write_u32(buf, 0x20, ro_off)
    write_u32(buf, 0x24, ro_mem)
    write_u32(buf, 0x28, len(rodata))
    write_u32(buf, 0x30, data_off)
    write_u32(buf, 0x34, data_mem)
    write_u32(buf, 0x38, len(data))
    write_u32(buf, 0x3C, 0)                # bss size
    buf[0x40:0x60] = b"SYNTHETIC-NSO-BOOT-ID".ljust(0x20, b"0")
    write_u32(buf, 0x60, len(text))
    write_u32(buf, 0x64, len(rodata))
    write_u32(buf, 0x68, len(data))
    buf[text_off:text_off + len(text)] = text
    buf[ro_off:ro_off + len(rodata)] = rodata
    buf[data_off:data_off + len(data)] = data
    path.write_bytes(buf)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    romfs = build_romfs([("data.bin", b"0123456789"), ("dir1/file.txt", b"hello romfs")])
    nro = INPUTS / "homebrew.nro"
    nso = INPUTS / "boot.nso"
    build_nro(nro)
    build_boot_nso(nso, romfs)

    lines = ['schema_version = "1"', "", "[title]", 'name = "nso-boot sample"',
             'title_id = "0100000000000000"', 'version = "0.0.0"', 'region = "US"', "",
             "[collection]", 'device = "synthetic"', 'collected_at = "2026-10-27"',
             'notes = "Generated inputs; no proprietary content."', "",
             "[collection.tool]", 'name = "generate.py"', 'version = "current"', ""]
    for path, fmt, role in [(nro, "nro0", "placeholder_module"), (nso, "nso0", "boot_executable")]:
        lines.append("[[inputs]]")
        lines.append(f'path = "{path.resolve()}"')
        lines.append(f'format = "{fmt}"')
        lines.append(f'sha256 = "{sha256(path)}"')
        lines.append(f"size = {path.stat().st_size}")
        lines.append(f'role = "{role}"')
        lines.append("")
    (ROOT / "provenance.toml").write_text("\n".join(lines))

    # title.toml: memory layout mirrors the NSO segment memory offsets; the
    # ipc section points at the RomFS blob embedded in .data (image base 0x2000
    # + 0x2800).
    romfs_base = 0x2000 + 0x2800
    title = f"""title = "nso-boot sample"
entry = "entry"
abi_version = "0.1.0"

[runtime]
performance_mode = "handheld"

[runtime.ipc]
romfs_base = {romfs_base:#x}
romfs_size = {len(romfs):#x}

[runtime.memory_layout]
regions = [
  {{ name = "text", base = 0x0, size = 0x1000, permissions = {{ read = true, write = false, execute = true }} }},
  {{ name = "rodata", base = 0x1000, size = 0x1000, permissions = {{ read = true, write = false, execute = false }} }},
  {{ name = "data", base = 0x2000, size = 0x4000, permissions = {{ read = true, write = true, execute = false }} }},
  {{ name = "heap", base = 0x10000, size = 0x4000, permissions = {{ read = true, write = true, execute = false }} }},
]
"""
    (ROOT / "title.toml").write_text(title)
    print(f"wrote {nso} ({nso.stat().st_size} bytes), romfs {len(romfs)} bytes")


if __name__ == "__main__":
    main()
