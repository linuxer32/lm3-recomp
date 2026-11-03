# Packaging Layout and Policy

This document defines the bundle layout, runtime asset lookup, and asset
separation rules for SwitchRecomp outputs. User-supplied title data (RomFS,
segments, scripts, saves) is never part of the repository or the `code/`
bundle.

## Bundle Layout (v1)
```
bundle/
  code/
    Cargo.toml
    .cargo/config.toml   # when the title sets [runtime] target_cpu
    chunks/              # large titles: workspace chunk crates
      chunk_0/
        Cargo.toml
        src/lib.rs
    src/
      main.rs
    manifest.json
  assets/
    README.txt
  metadata/
    provenance.toml
    bundle-manifest.json
```

- `code/` contains generated source, Cargo manifests (including the `chunks/`
  workspace members) and `.cargo/config.toml`. It is enough to build the
  program without the original project directory.
- `assets/` is reserved for user-supplied data and stays empty in
  distributions. `recomp-cli package` copies an assets tree only when the
  operator explicitly passes `--assets-dir`.
- `metadata/` contains provenance and checksum data required for auditing.
- Memory segment blobs (`segments/*.bin`) are extracted title bytes. They are
  **not** copied by `package`; place them at run time as described below.

## Reference Packaging Command
```
cargo run -p recomp-cli -- package \
  --project-dir out/minimal \
  --provenance samples/minimal/provenance.toml \
  --out-dir out/bundle-minimal
```

## Windows Bundle Layout (x86_64-pc-windows-gnu)

The generated project's Windows build is staged by
`scripts/build-windows-gnu.sh`; see `docs/windows-build.md`.

```
bundle-windows/
  <title>.exe
  SDL3.dll              # from the SDL3 mingw devel package's bin/
  vulkan-1.dll          # optional: System32 usually has the loader
  assets/               # user-supplied RomFS image or dump; never committed
  save/                 # created at run time, or RECOMP_SAVE_DIR elsewhere
```

`SDL3.dll` must be beside the executable (or on `PATH`). `vulkan-1.dll` is
loaded dynamically by `ash` and SDL; a GPU driver's copy in `System32` is
enough, and bundling the Vulkan SDK's `Bin/vulkan-1.dll` makes an offline
install self-contained. Do not commit either DLL or any title asset to this
repository.

## Runtime Asset Lookup

Generated programs no longer depend solely on the absolute build-machine
paths recorded at generation time. For a relative asset such as
`scripts/game.rhai`, `segments/data-0.bin`, or `romfs.bin`, the runtime
searches these roots in order:

1. `RECOMP_ASSETS_DIR`, when set and non-empty.
2. The executable's directory.
3. An `assets/` directory beside the executable.
4. The current working directory.

If nothing is found, the build-time path (for example
`out/minimal/segments/data-0.bin`) is used as before, so a development run
from the generated project keeps working.

Environment overrides:

| Variable | Effect |
|---|---|
| `RECOMP_ROMFS` | Host path to a RomFS image (or dump) served by `fsp-srv`. An empty value disables the image and falls back to the in-memory window, if configured. |
| `RECOMP_SCRIPT` | Host path to the script to load. An empty value loads no script; unset keeps the title's configured default. |
| `RECOMP_SAVE_DIR` | Host directory for save data. Unset keeps the existing default: a `save/` directory next to the RomFS image. |
| `RECOMP_ASSETS_DIR` | Root searched before the executable's directory and its `assets/` subdirectory. |

Memory segments have no per-file environment variable: package them under
`segments/` below any lookup root, e.g.
`<install-dir>/segments/data-0.bin` or
`<install-dir>/assets/segments/data-0.bin`.

### Recommended deployment

- **Linux/macOS**: install the executable and an `assets/` directory (or the
  individual files) in the same directory, or set `RECOMP_ASSETS_DIR`.
- **Windows**: place `romfs.bin`, `scripts/`, `segments/` and an optional
  `save/` next to the `.exe` (or under its `assets/` directory), and set
  `RECOMP_SAVE_DIR` when the install directory is read-only. Keep the
  packaged `SDL3.dll` and `vulkan-1.dll` beside the executable.
- **Android**: place the files in the APK's assets and set
  `RECOMP_ASSETS_DIR` to the extracted assets directory, and
  `RECOMP_SAVE_DIR` to the app-private storage directory. The APK builder is
  still a separate port workstream; see `docs/platform-port-plan.md`.

## Asset Separation Rules
- Never commit or distribute keys, RomFS dumps, segment blobs, or save data.
- `code/` contains generated source code and build metadata only.
- `assets/` is for operator-supplied data and must remain empty in
  distributions.
- `metadata/` contains provenance and checksum data required for auditing.
