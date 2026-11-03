# SPEC-090: Build, Packaging, and Distribution

## Status
Draft v0.6

## Rationale
- Added a Nix + devenv development scaffold and initial build outputs for exploratory work.
- Added a deterministic build manifest with input hashes for reproducible packaging.
- Made packaged projects buildable and runnable off the build machine: Cargo paths are portable, chunk crates and build config are bundled, and runtime assets resolve beside the executable or through environment overrides.

## Purpose
Define build outputs, packaging, and distribution policy.

## Goals
- Produce reproducible builds with clear versioning.
- Distribute only non-proprietary artifacts.

## Non-Goals
- Shipping proprietary assets or binaries.
- Providing a commercial installer.

## Build Outputs
- Recompiled native binary.
- Runtime library and support files.
- Metadata files for provenance and configuration.

## Packaging
- A bundle layout that separates code from user-supplied assets.
- A deterministic build manifest and checksums.

### Bundle Layout (v1)
```
bundle/
  code/
    Cargo.toml
    .cargo/config.toml
    chunks/
      chunk_0/
        Cargo.toml
        src/lib.rs
    src/
    manifest.json
  assets/
    README.txt
  metadata/
    provenance.toml
    bundle-manifest.json
```

- `code/` MUST include the chunk crates and `.cargo/config.toml` when the
  generated project has them, so the packaged project builds unchanged.
- `assets/` MUST NOT contain proprietary title data in distributions.
- Memory segment blobs MUST be supplied at run time, not copied from the
  generated project into `code/`.

### Runtime Asset Resolution
- Generated programs MUST prefer a user-supplied copy of a RomFS image,
  script, or memory segment beside the executable (or under
  `RECOMP_ASSETS_DIR`) over the absolute path recorded at generation time.
- `RECOMP_ROMFS`, `RECOMP_SCRIPT`, and `RECOMP_SAVE_DIR` MUST override the
  RomFS image, script, and save directory. An empty value disables the
  asset where the runtime defines that behavior.
- With no override or packaged copy present, the recorded build path MUST
  remain the fallback.

### Bundle Manifest
- `bundle-manifest.json` records checksum + size for every file in the bundle.
- The manifest is reproducible for identical inputs.

## Distribution Policy
- Open source code and tooling only.
- Clear instructions for user-supplied assets.

## Deliverables
- A packaging spec and reference implementation.
- A release checklist that includes legal compliance checks.

## Supporting Docs
- `docs/packaging.md`
- `docs/RELEASE-CHECKLIST.md`

## Open Questions
- What is the initial host platform support matrix?
- How should platform-specific dependencies be distributed?

## Acceptance Criteria
- A build that can be reproduced from the same inputs.
- A packaged output that runs when assets are supplied externally.
- A packaged project builds without the original `out/` project directory,
  including titles large enough to emit `chunks/`.
- A packaged output finds assets beside the executable or through
  `RECOMP_ASSETS_DIR`/`RECOMP_ROMFS`/`RECOMP_SCRIPT`/`RECOMP_SAVE_DIR`
  without depending on the build machine's absolute paths.
