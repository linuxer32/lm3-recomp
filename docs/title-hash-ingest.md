# Hashed Title Ingest Workflow

Use this workflow to obscure title names in local data paths by replacing clear-text title names
and filenames with hash-based names.

## What It Does
- Assigns a hash-based title id: `title-<sha256(title)[:16]>`.
- Renames or copies `game-data/<clear-title>/` to `game-data/<title-hash-id>/`.
- Renames XCI files to `<sha256(file)[:16]>.xci` under `xci/`.
- Renames gameplay videos to `<sha256(file)[:16]>-NNN.<ext>` under `gameplay-video/`.
- Updates a local decoder ring at `config/title-hash-registry.toml` (gitignored).
- Appends operation logs to `config/title-hash-tracking.log` (gitignored).

## Decoder Ring Files
- Tracked template: `config/title-hash-registry.example.toml`
- Local file (untracked): `config/title-hash-registry.toml`
- Local tracking log (untracked): `config/title-hash-tracking.log`

## Basic Usage
```bash
export GAME_DATA_ROOT=/path/to/game-data
scripts/ingest_hashed_title.sh \
  --title sample-title \
  --game-data-root $GAME_DATA_ROOT
```

This will:
- Map `sample-title` to `title-<hash-prefix>`.
- Rename:
  - `$GAME_DATA_ROOT/sample-title`
  - to `$GAME_DATA_ROOT/title-<hash-prefix>`
- Hash/rename files under:
  - `$GAME_DATA_ROOT/title-<hash-prefix>/xci/`
  - `$GAME_DATA_ROOT/title-<hash-prefix>/gameplay-video/`

## Explicit File Inputs
```bash
export GAME_DATA_ROOT=/path/to/game-data
scripts/ingest_hashed_title.sh \
  --title sample-title \
  --game-data-root $GAME_DATA_ROOT \
  --xci "$GAME_DATA_ROOT/sample-title/xci/original.xci" \
  --video "$GAME_DATA_ROOT/sample-title/gameplay-video/reference-raw.webm"
```

## Notes
- Default behavior is `move`; use `--copy` if you want to preserve original files.
- Re-running the script is safe for already-hashed names.
- Keep `config/title-hash-registry.toml` private; it is the clear-text decoder ring.
