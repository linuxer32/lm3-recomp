# nso-boot sample

Synthetic NSO boot image that exercises the static-recompilation pipeline
end to end on Linux:

1. **Intake** extracts NSO segments and applies the MOD0 relocation tables
   (`--apply-relocations`). Tables live in the combined text+rodata+data
   memory image, matching libnx semantics.
2. **Lift** decodes the boot instructions starting *after* the MOD0 header
   (`--entry-address 48`) and filters out the placeholder NRO module
   (`--module-name boot`).
3. **Run** emits a Rust project with a memory image (relocated `.data`,
   embedded RomFS blob), kernel SVC dispatch, and IPC/fs: emulation.

## Boot sequence

```asm
mov  x0, #0x2020   ; out-handle pointer
mov  x1, #0x2000   ; "fs:GLOBAL"
svc  #0x1F         ; svcConnectToNamedPort
mov  x0, #1        ; session handle
mov  x1, #0x2100   ; HIPC request buffer
svc  #0x21         ; svcSendSyncRequest (OpenDataFileSystemByCurrentProcess)
mov  x0, #0
svc  #0x07         ; svcExitProcess
```

The request buffer holds a HIPC header (magic bit 14) and a CMIF/SFCI
payload with method id 2. `fs:GLOBAL` replies with a filesystem object; the
runtime's IPC hub reads the RomFS blob embedded in `.data` (see
`[runtime.ipc]` in `title.toml`).

## MOD0 tables

Header at `.text` offset 0; tables packed after the header (`+0x1C`):

| table | region    | entry                         | effect                       |
|-------|-----------|-------------------------------|------------------------------|
| abs64 | 0x1C..0x28 | image 0x2010 <- `0x2000`      | port address patched         |
| rel64 | 0x28..0x38 | image 0x2030 += `0x40`        | slot `0x100` -> `0x140`      |
| rel32 | 0x38..0x40 | image 0x2040 += `0x20`        | slot `0x5` -> `0x25`         |

## Reproduce

```sh
python3 samples/nso-boot/generate.py

cargo run -p recomp-cli -- homebrew-intake \
  --module samples/nso-boot/inputs/homebrew.nro \
  --nso samples/nso-boot/inputs/boot.nso \
  --provenance samples/nso-boot/provenance.toml \
  --out-dir out/nso-boot-intake --apply-relocations

cargo run -p recomp-cli -- homebrew-lift \
  --module-json out/nso-boot-intake/module.json \
  --out-dir out/nso-boot-lift --entry entry \
  --entry-address 48 --module-name boot

cargo run -p recomp-cli -- run \
  --module out/nso-boot-lift/module.json \
  --config samples/nso-boot/title.toml \
  --provenance out/nso-boot-lift/provenance.toml \
  --out-dir out/nso-boot-project --runtime-path crates/recomp-runtime

cargo build --manifest-path out/nso-boot-project/Cargo.toml
./out/nso-boot-project/target/debug/recomp-nso-boot-sample
```

Expected output ends with `svcSendSyncRequest` dispatching method 2 against
`fs:GLOBAL` and a clean exit (code 0). All inputs are generated locally; the
sample contains no proprietary content.
