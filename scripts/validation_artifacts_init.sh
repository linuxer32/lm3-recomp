#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'USAGE'
Usage:
  scripts/validation_artifacts_init.sh --out <path> [--label <label>]

Options:
  --out <path>     Output JSON path (required).
  --label <label>  Artifact label (default: title-sample-first-level).
USAGE
}

OUT=""
LABEL="title-sample-first-level"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage
      exit 2
      ;;
  esac
done

if [[ -z "$OUT" ]]; then
  echo "--out is required." >&2
  print_usage
  exit 2
fi

if [[ -e "$OUT" ]]; then
  echo "Output path already exists: $OUT" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"

cat <<JSON > "$OUT"
{
  "label": "${LABEL}",
  "xci_intake_manifest": "/path/to/inputs/intake/manifest.json",
  "pipeline_manifest": "/path/to/outputs/recompiled/manifest.json",
  "run_manifest": "/path/to/outputs/recompiled/run-manifest.json",
  "reference_config": "/path/to/validation/reference_video.toml",
  "capture_config": "/path/to/validation/capture_video.toml",
  "validation_config": "/path/to/validation/validation_config.toml",
  "out_dir": "/path/to/validation/reports/YYYY-MM-DD"
}
JSON

echo "Wrote artifact index template to $OUT"
