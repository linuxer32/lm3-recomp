# IPC / fs: emulation and NSO MOD0 relocation

This note describes the runtime service layer added for statically recompiled
Linux targets: kernel IPC SVC routing, a HIPC/CMIF message codec, `fs:`
dispatch backed by a RomFS window, and NSO MOD0 relocation applied during
intake.

## Scope and provenance

All emulation is protocol-driven (switchbrew.org: HIPC, Filesystem_services,
libnx `romfs_header`). No proprietary keys or retail code are embedded; the
`samples/nso-boot` inputs are generated locally by `generate.py`.

## MOD0 relocation (`recomp-pipeline::homebrew::mod0`)

libnx NSO images embed a MOD0 header at the start of `.text`. The runtime
loader normally applies its tables; a static recompiler must patch the image
before lifting code and data.

Header fields (relative to the header):

| offset | field              | meaning                       |
|--------|--------------------|-------------------------------|
| 0x00   | magic `MOD0`       | presence check                |
| 0x04   | rel32 boundary     | end of the rel64 table        |
| 0x08   | rel64 boundary     | end of the abs64 table        |
| 0x0C   | abs64 boundary     | end of the rel32 table        |
| 0x10   | exec start         | JIT/exec range start          |
| 0x14   | exec end           | JIT/exec range end            |
| 0x18   | exec ranges end    | end of exec-range table       |

Tables are packed right after the header (`header + 0x1C`) in this order:
abs64 (`offset u32`, `value u64`), rel64 (`offset u32`, `addend u64`), rel32
(`offset u32`, `addend u32`), exec ranges (`start u32`, `end u32`). Offsets
are relative to the combined memory image (text base 0), so tables can span
text/rodata/data/bss.

Intake applies tables when `--apply-relocations` is set (`apply_relocations`
in automation config). Images whose header does not validate against the
canonical layout (e.g. some Eden-produced NSO variants) produce a relocation
warning and fall back to unrelocated segments instead of failing intake.

## Lift entry addressing

NSO `.text` begins with the MOD0 header, so lifting from segment offset 0
decodes header words. `homebrew-lift --entry-address <hex>` seeds the entry
function past the header, and `--module-name <name>` lifts only that module's
segments (placeholder NRO modules from intake are excluded). Lifted
`module.json` now carries segment descriptors with `init_path` copies so the
pipeline emits runtime memory initialization (relocated `.data`, embedded
RomFS blobs, IPC buffers).

## HIPC / CMIF codec (`recomp-services::ipc`)

Wire layouts follow the Nintendo IPC message framing observed in the LM3
SDK binary (switchbrew.org/wiki/HIPC), which differs from the earlier
SPEC-070 draft diagrams:

- Message header (16 bytes): `Type` u32 at buffer + 0x00 (6 = request,
  4 = reply), `Options` u32 at +0x04, reserved/pid u64 at +0x08.
- HIPC `Header0` (u32 at buffer + 0x10): magic bits 0..15 (`0x4000`, or
  `0x0001` for domain-routed SDK builders), PointerCount bits 16..19,
  SendCount bits 20..23, ReceiveCount bits 24..27, ExchangeCount bits 28..31.
- HIPC `Header1` (u32 at buffer + 0x14): raw data words bits 0..9,
  ReceiveListCount bits 10..13, ReceiveListOffset (words from the descriptor
  region) bits 20..30, special bit 31.
- Optional special header u32 at buffer + 0x18: copy count bits 1..4, move
  count bits 5..8; handle descriptors follow as u32 words.
- Buffer descriptors are {offset, size} u32 pairs from buffer + 0x18, where
  offset is relative to the message buffer start. A pointer descriptor with
  offset 0 encodes an inline kernel handle (used in replies).
- The CMIF payload is located by scanning the message region for the SFCI
  magic (`0x49434653`) because SDK builders vary in descriptor placement:
  magic, version u32, method id u32 at magic + 0x08, inline parameters at
  magic + 0x0C. u64 parameters may sit at word-aligned addresses and are
  assembled from two u32 loads.
- Replies: message type 4 at buffer + 0x00; `Header0` magic plus handle
  pointer descriptors ({0, handle}) from buffer + 0x18; raw return data after
  the descriptors; an SFCO CMIF block (`0x4F434653`, version, status u32)
  aligned to 16 bytes after the raw data, matching the SDK parser.
- Receive-list entries are 8-byte {offset, size} pairs; service output is
  written into the inline buffer at the entry offset and the entry size word
  is rewritten to the bytes actually returned.

Deliberate simplifications vs. the kernel: domain routing and per-session
tokens are parsed but not enforced; `sm:` service manager registration is
not modeled; named ports resolve directly to service objects.

## RomFS backing modes

`IpcHub` serves RomFS either from an in-memory guest window (`romfs_base` /
`romfs_size`) or, when `[runtime.ipc] romfs_image` is configured, from a
host file. The host mode parses the Nintendo RomFS layout on first use and
caches the metadata: header size u64 at +0x00 marks the data region start;
directory table entries are 24-byte headers (parent index u32 at +0x00, name
length u32 at +0x14) with inline names padded to 4; file table entries are
32-byte headers (directory table byte offset u32 at +0x04, data offset u64 at
+0x08, size u64 at +0x10, name length u32 at +0x1C) with inline names padded
to 4; hash tables and tables sit at the end of the image. Multi-gigabyte
images never enter the guest address space: `IStorage`/`IFile` reads are
served from disk. `[runtime.ipc] ipc_buffer` points TPIDRRO_EL0 (the SDK's
IPC buffer) at a mapped guest region.

## Kernel SVC routing

`KernelSvcDispatcher` gains an attachable `IpcHub`:

- `svcConnectToNamedPort` (0x1F): resolves `fs:GLOBAL`/`fs:USER`/`fs:SYSTEM`
  to a session handle and writes it to the out-handle pointer; unknown ports
  return `0xC0A00009`.
- `svcSendSyncRequest` (0x21), `svcSendSyncRequestWithUserBuffer` (0x22),
  `svcSendAsyncRequestWithUserBuffer` (0x23): decode the request, dispatch,
  write the reply.
- `svcCloseHandle` (0x16): drops the object from the hub table.

Without an attached hub these SVCs keep the previous structural behavior
(succeed vacuously) so older samples still boot.

## fs: dispatch

Command IDs follow the FS v5-era CMIF table from switchbrew:

| object        | method | behavior                                  |
|---------------|--------|-------------------------------------------|
| `fsp-srv`     | 1      | `SetCurrentProcess` (no-op)               |
| `fsp-srv`     | 2      | `OpenDataFileSystemByCurrentProcess`      |
| `fsp-srv`     | 200    | `OpenDataStorageByCurrentProcess`         |
| `fsp-srv`     | 202    | `OpenDataStorageByDataId` (same window)   |
| `IFileSystem` | 1      | `OpenFile(path)` -> file storage          |
| `IFileSystem` | 2      | `OpenDirectory(path)` -> directory        |
| `IStorage`    | 0      | `Read(offset, buffer)` -> read size       |
| `IStorage`    | 1      | `GetSize`                                 |
| `IDirectory`  | 0      | `Read(entries)` -> count                  |
| `IDirectory`  | 1      | `GetEntryCount`                           |

Directory entries are encoded in the legacy layout: `path[256]`, `u64 size`,
`u32 type` (0 file, 1 directory), 268 bytes.

## RomFS window

The hub reads a RomFS image from a byte range of the recompiled process
memory image (`[runtime.ipc] romfs_base`/`romfs_size` in title configs;
`recomp_runtime::init_ipc_services(base, size)`). Parsing follows the libnx
`romfs_header` layout (u64 fields at 0x00..0x50: header size, dir hash,
dir table, file hash, file table, data). Directory/file tables are walked in
entry order; paths resolve `/dir/file` hierarchies.

## nso-boot demo

`samples/nso-boot` exercises the whole chain on Linux: intake (MOD0 patch),
lift (`--entry-address 48`, `--module-name boot`), pipeline emit with memory
image + `init_ipc_services`, then a boot sequence calling
`svcConnectToNamedPort("fs:GLOBAL")` and `svcSendSyncRequest`
(`OpenDataFileSystemByCurrentProcess`) before `svcExitProcess`. See
`samples/nso-boot/README.md` for commands and expected output.

## Known gaps

- LM3's Eden-produced NSO uses a non-libnx MOD0 table layout; intake reports
  a warning and lifts unrelocated segments. Static LM3 lifting of that dump
  needs either the dump's MOD0 variant decoded or a canonical libnx NSO.
- LM3's static modules contain no `svcSendSyncRequest` sites, so fs: method
  IDs are protocol-documentation-driven, not code-verified.
- `am:`/`gl:`/HID services are not emulated yet; only `fs:` ports resolve.
