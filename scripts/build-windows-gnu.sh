#!/usr/bin/env bash
# Build a generated game project for Windows x86_64 (mingw) and stage the
# executable with the DLLs it loads at run time.
#
# The cross build needs:
#   rustup target add x86_64-pc-windows-gnu
#   x86_64-w64-mingw32-gcc (mingw-w64) on PATH
#   SDL3's mingw development package (libSDL3.dll.a, SDL3.dll)
# Vulkan needs no import library: ash and SDL load vulkan-1.dll at run time.
#
# See docs/windows-build.md for the download and layout.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/build-windows-gnu.sh --project-dir <generated-project> [options]

Options:
  --project-dir <dir>   Generated project (the directory with Cargo.toml).
  --release             Build the release game binary (default).
  --debug               Build the debug binary with a console window.
  --gui                 Use the Windows GUI subsystem even in a debug build.
  --fast-tls            Build with native Windows TLS (requires RUSTC_BOOTSTRAP).
  --sdl3-dir <dir>      SDL3 mingw package root, or its
                        x86_64-w64-mingw32 subdirectory. Defaults to
                        $SDL3_DIR, common vendor paths, or the mingw prefix.
  --vulkan-dll <file>   vulkan-1.dll to stage beside the executable.
                        Defaults to $VULKAN_SDK/Bin/vulkan-1.dll when present.
  --out-dir <dir>       Staging directory (default: <project>/dist/windows).
  -h, --help            Show this help.

The executable and SDL3.dll must sit in the same directory; Windows also
provides vulkan-1.dll in System32 when a GPU driver is installed.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
project_dir=""
profile=release
gui=0
fast_tls=0
sdl3_dir=${SDL3_DIR:-}
vulkan_dll=${VULKAN_DLL:-}
out_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir)
      [[ $# -ge 2 ]] || die "--project-dir needs a value"
      project_dir=$2
      shift 2
      ;;
    --release)
      profile=release
      shift
      ;;
    --debug)
      profile=debug
      shift
      ;;
    --gui)
      gui=1
      shift
      ;;
    --fast-tls)
      fast_tls=1
      shift
      ;;
    --sdl3-dir)
      [[ $# -ge 2 ]] || die "--sdl3-dir needs a value"
      sdl3_dir=$2
      shift 2
      ;;
    --vulkan-dll)
      [[ $# -ge 2 ]] || die "--vulkan-dll needs a value"
      vulkan_dll=$2
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || die "--out-dir needs a value"
      out_dir=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (try --help)"
      ;;
  esac
done

[[ -n "$project_dir" ]] || die "--project-dir is required"
manifest="$project_dir/Cargo.toml"
[[ -f "$manifest" ]] || die "$manifest not found"
if [[ "$profile" == release ]]; then
  gui=1
fi
target=x86_64-pc-windows-gnu

if ! rustup target list --installed 2>/dev/null | grep -qx "$target"; then
  die "the Rust target is not installed; run: rustup target add $target"
fi

# Cargo invokes the target linker by name; accept the MSYS2 spelling too.
linker=${CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER:-}
if [[ -z "$linker" ]]; then
  if command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
    linker=x86_64-w64-mingw32-gcc
  elif [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]] && command -v gcc >/dev/null 2>&1; then
    linker=gcc
  else
    die "x86_64-w64-mingw32-gcc not found; install mingw-w64"
  fi
fi
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="$linker"

# Normalize an SDL3 package layout to the directory holding include/ and lib/.
normalize_sdl3_dir() {
  local dir=$1
  [[ -n "$dir" && -d "$dir" ]] || return 1
  if [[ -d "$dir/x86_64-w64-mingw32" ]]; then
    dir="$dir/x86_64-w64-mingw32"
  elif [[ -d "$dir/SDL3-3.4.18/x86_64-w64-mingw32" ]]; then
    dir="$dir/SDL3-3.4.18/x86_64-w64-mingw32"
  elif [[ "$(basename "$dir")" == lib && -d "$dir/../include" ]]; then
    dir=$(cd "$dir/.." && pwd)
  fi
  if [[ -f "$dir/lib/libSDL3.dll.a" || -f "$dir/lib/libSDL3.a" ]]; then
    printf '%s\n' "$dir"
    return 0
  fi
  return 1
}

find_sdl3_dir() {
  local candidate
  local -a candidates=()
  [[ -n "$sdl3_dir" ]] && candidates+=("$sdl3_dir")
  for candidate in \
    "$repo_root"/third_party/SDL3 \
    "$repo_root"/third_party/sdl3 \
    "$repo_root"/vendor/SDL3 \
    "$repo_root"/vendor/sdl3 \
    /mingw64 \
    /usr/x86_64-w64-mingw32 \
    /usr/local/x86_64-w64-mingw32; do
    candidates+=("$candidate")
  done
  for candidate in "$repo_root"/third_party/SDL3-*/x86_64-w64-mingw32; do
    candidates+=("$candidate")
  done
  for candidate in "${candidates[@]}"; do
    if normalize_sdl3_dir "$candidate" >/dev/null 2>&1; then
      normalize_sdl3_dir "$candidate"
      return 0
    fi
  done
  return 1
}

if ! sdl3_dir=$(find_sdl3_dir); then
  cat >&2 <<'EOF'
error: an SDL3 mingw development package was not found.

Download the matching SDL3 release (the sdl3 crate 0.20 uses SDL 3.4.18):
  https://github.com/libsdl-org/SDL/releases/tag/release-3.4.18
  asset: SDL3-devel-3.4.18-mingw.zip

Unpack it and either pass --sdl3-dir or set SDL3_DIR. The layout needed is:
  <sdl3-dir>/x86_64-w64-mingw32/include/SDL3/SDL.h
  <sdl3-dir>/x86_64-w64-mingw32/lib/libSDL3.dll.a
  <sdl3-dir>/x86_64-w64-mingw32/bin/SDL3.dll

A system mingw-w64 SDL3 package (for example MSYS2's /mingw64) works too.
EOF
  exit 1
fi

sdl3_lib="$sdl3_dir/lib"
sdl3_dll="$sdl3_dir/bin/SDL3.dll"
[[ -f "$sdl3_lib/libSDL3.dll.a" || -f "$sdl3_lib/libSDL3.a" ]] ||
  die "$sdl3_lib has no libSDL3.dll.a or libSDL3.a"
[[ -f "$sdl3_dll" ]] ||
  echo "warning: $sdl3_dll is missing; stage SDL3.dll beside the executable before running it"

# sdl3-sys emits `-lSDL3`; give the linker the package's import library. A
# pre-existing RUSTFLAGS keeps its meaning, but paths with spaces should use
# CARGO_ENCODED_RUSTFLAGS instead.
if [[ $fast_tls -eq 1 ]]; then
  # Native TLS avoids Rust's TlsGetValue wrapper on every guest block/call.
  export RUSTC_BOOTSTRAP=1
  if [[ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]]; then
    export CARGO_ENCODED_RUSTFLAGS="${CARGO_ENCODED_RUSTFLAGS}"$'\x1f'"--cfg"$'\x1f'"recomp_fast_tls"
  elif [[ -n "${RUSTFLAGS:-}" ]]; then
    export RUSTFLAGS="$RUSTFLAGS --cfg recomp_fast_tls"
  else
    export RUSTFLAGS="--cfg recomp_fast_tls"
  fi
fi

if [[ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]]; then
  export CARGO_ENCODED_RUSTFLAGS="${CARGO_ENCODED_RUSTFLAGS}"$'\x1f'"-L"$'\x1f'"native=$sdl3_lib"
elif [[ -n "${RUSTFLAGS:-}" ]]; then
  export RUSTFLAGS="$RUSTFLAGS -L native=$sdl3_lib"
else
  export CARGO_ENCODED_RUSTFLAGS="-L"$'\x1f'"native=$sdl3_lib"
fi

# The generated package declares this feature; older projects need a
# regeneration. Without it a release build keeps the console window.
features=()
if [[ $gui -eq 1 ]]; then
  if grep -Eq '^windows-subsystem[[:space:]]*=[[:space:]]*\[\]' "$manifest"; then
    features=(--features windows-subsystem)
  else
    echo "warning: $manifest has no windows-subsystem feature; this was generated by an older pipeline and keeps a console window" >&2
  fi
fi

build_flags=(--manifest-path "$manifest" --target "$target")
if [[ "$profile" == release ]]; then
  build_flags+=(--release)
fi

echo "building $profile binary for $target"
echo "  SDL3:  $sdl3_dir"
echo "  linker: $(command -v "$linker" || printf '%s' "$linker")"
cargo build "${build_flags[@]}" "${features[@]}"

# The generated project has one binary named after its package.
package_name=$(awk '
  /^\[package\]/ { in_package = 1; next }
  /^\[/ { in_package = 0 }
  in_package && /^name[[:space:]]*=/ {
    gsub(/^[^=]*=[[:space:]]*"/, "")
    gsub(/".*$/, "")
    print
    exit
  }
' "$manifest")
[[ -n "$package_name" ]] || die "could not read the package name from $manifest"

target_dir=${CARGO_TARGET_DIR:-"$project_dir/target"}
exe="$target_dir/$target/$profile/$package_name.exe"
[[ -f "$exe" ]] || die "build finished but $exe is missing"

if [[ -z "$out_dir" ]]; then
  out_dir="$project_dir/dist/windows"
fi
mkdir -p "$out_dir"
cp "$exe" "$out_dir/"

if [[ -f "$sdl3_dll" ]]; then
  cp "$sdl3_dll" "$out_dir/SDL3.dll"
fi

if [[ -z "$vulkan_dll" ]]; then
  for candidate in \
    "${VULKAN_SDK:-}/Bin/vulkan-1.dll" \
    "${VULKAN_SDK:-}/bin/vulkan-1.dll" \
    "$repo_root/third_party/vulkan-1.dll" \
    /mingw64/bin/vulkan-1.dll \
    /usr/x86_64-w64-mingw32/bin/vulkan-1.dll; do
    if [[ -f "$candidate" ]]; then
      vulkan_dll=$candidate
      break
    fi
  done
fi
if [[ -n "$vulkan_dll" && -f "$vulkan_dll" ]]; then
  cp "$vulkan_dll" "$out_dir/vulkan-1.dll"
else
  echo "note: vulkan-1.dll not staged; Windows with a GPU driver has it in System32." >&2
  echo "      For an offline bundle, copy <Vulkan SDK>/Bin/vulkan-1.dll here." >&2
fi

# VP9 cutscenes are decoded through libvpx at runtime. Without this DLL the
# video surface is transparent and the castle/intro movies do not appear.
libvpx_dll="${LIBVPX_DLL:-}"
if [[ -z "$libvpx_dll" ]]; then
  for candidate in \
    "$repo_root/third_party/libvpx-1.dll" \
    /mingw64/bin/libvpx-1.dll \
    /usr/x86_64-w64-mingw32/bin/libvpx-1.dll \
    /usr/lib/x86_64-w64-mingw32/bin/libvpx-1.dll; do
    if [[ -f "$candidate" ]]; then
      libvpx_dll=$candidate
      break
    fi
  done
fi
if [[ -n "$libvpx_dll" && -f "$libvpx_dll" ]]; then
  cp "$libvpx_dll" "$out_dir/libvpx-1.dll"
else
  echo "warning: libvpx-1.dll not staged; VP9 cutscenes will not decode." >&2
  echo "         Install mingw-w64-x86_64-libvpx or set LIBVPX_DLL=<path>." >&2
fi

echo
echo "staged in $out_dir:"
ls -lh "$out_dir" | sed '1d'

if command -v x86_64-w64-mingw32-objdump >/dev/null 2>&1; then
  echo
  echo "the executable imports:"
  x86_64-w64-mingw32-objdump -p "$out_dir/$package_name.exe" |
    awk '/DLL Name/ { print "  " $3 }' | sort -u
fi
