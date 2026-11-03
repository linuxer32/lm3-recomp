#!/usr/bin/env bash
# Verify the shared-library contract the Android bootstrap expects.
set -euo pipefail

ANDROID_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "usage: scripts/check-game-lib.sh PATH/TO/librecomp_game.so"
}

if [[ $# -ne 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 2
fi
GAME_LIB="$1"
if [[ ! -f "$GAME_LIB" ]]; then
    echo "not a file: $GAME_LIB" >&2
    exit 1
fi

find_ndk() {
    local candidate
    for candidate in "${ANDROID_NDK_HOME:-}" "${ANDROID_NDK_ROOT:-}" "${ANDROID_NDK:-}"; do
        if [[ -n "$candidate" && -f "$candidate/build/cmake/android.toolchain.cmake" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    if [[ -d /opt/android-sdk/ndk/28.2.13676358 ]]; then
        printf '%s\n' /opt/android-sdk/ndk/28.2.13676358
        return 0
    fi
    find /opt/android-sdk/ndk -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1
}

NDK="$(find_ndk)"
case "$(uname -s)" in
    Linux*) HOST_TAG="linux-x86_64" ;;
    Darwin*) HOST_TAG="darwin-x86_64" ;;
    *) echo "unsupported host: $(uname -s)" >&2; exit 1 ;;
esac
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/$HOST_TAG"
READELF="$TOOLCHAIN/bin/llvm-readelf"
NM="$TOOLCHAIN/bin/llvm-nm"

ELF_HEADER="$("$READELF" -h "$GAME_LIB")"
if ! grep -q 'Machine:.*AArch64' <<< "$ELF_HEADER"; then
    echo "FAIL: $GAME_LIB is not an AArch64 ELF" >&2
    exit 1
fi

SYMBOLS="$("$NM" -D --defined-only "$GAME_LIB")"
if ! grep -Eq '[[:space:]]T[[:space:]]+recomp_android_main$' <<< "$SYMBOLS"; then
    echo "FAIL: $GAME_LIB does not export recomp_android_main" >&2
    echo "      See android/game-entry/ for the required wrapper." >&2
    exit 1
fi

if grep -Eq '[[:space:]]T[[:space:]]+SDL_main$' <<< "$SYMBOLS"; then
    echo "warning: $GAME_LIB also exports SDL_main; the bootstrap calls recomp_android_main only" >&2
fi

echo "OK: $(basename "$GAME_LIB")"
echo "  size:   $(du -h "$GAME_LIB" | cut -f1)"
echo "  sha256: $(sha256sum "$GAME_LIB" | cut -d' ' -f1)"
echo "  needed:"
"$READELF" -d "$GAME_LIB" | sed -n 's/.*Shared library: \[\(.*\)\]/    \1/p'
