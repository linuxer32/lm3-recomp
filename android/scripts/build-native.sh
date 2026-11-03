#!/usr/bin/env bash
# Build the Android native payload for arm64-v8a:
#   libmain.so          host bootstrap (exports SDL_main)
#   libSDL3.so          built by sdl3-sys from the pinned SDL3 source
#   librecomp_game.so   optional generated title / stub
#
# The output lands in app/src/main/jniLibs/arm64-v8a so both the Gradle
# project and scripts/build-apk.sh can consume it. Nothing here is committed.
set -euo pipefail

ANDROID_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="aarch64-linux-android"
API="${ANDROID_API_LEVEL:-24}"
PROFILE="release"
CARGO_PROFILE_FLAG="--release"
STUB_GAME=0
GAME_LIB=""

usage() {
    cat <<'EOF'
usage: scripts/build-native.sh [options]

options:
  --debug             build the Rust bootstrap with the debug profile
  --release           build with the release profile (default)
  --stub-game         build android/stub-game and package it as librecomp_game.so
  --game-lib PATH     copy PATH to lib/arm64-v8a/librecomp_game.so
  --ndk PATH          Android NDK root (default: $ANDROID_NDK_HOME or NDK 28.2)
  -h, --help          show this help

environment:
  ANDROID_NDK_HOME    NDK root; also ANDROID_NDK_ROOT and ANDROID_NDK are read
  ANDROID_API_LEVEL   minimum API for the NDK toolchain (default 24)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            PROFILE="debug"
            CARGO_PROFILE_FLAG=""
            shift
            ;;
        --release)
            PROFILE="release"
            CARGO_PROFILE_FLAG="--release"
            shift
            ;;
        --stub-game)
            STUB_GAME=1
            shift
            ;;
        --game-lib)
            GAME_LIB="${2:?--game-lib needs a path}"
            shift 2
            ;;
        --ndk)
            ANDROID_NDK_HOME="${2:?--ndk needs a path}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$GAME_LIB" && "$STUB_GAME" -eq 0 && -n "${RECOMP_GAME_LIB:-}" ]]; then
    GAME_LIB="$RECOMP_GAME_LIB"
fi
if [[ -n "$GAME_LIB" && "$STUB_GAME" -eq 1 ]]; then
    echo "--game-lib and --stub-game are mutually exclusive" >&2
    exit 2
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
    if [[ -d /opt/android-sdk/ndk ]]; then
        find /opt/android-sdk/ndk -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1
        return 0
    fi
    return 1
}

NDK="$(find_ndk)" || {
    echo "Android NDK not found; set ANDROID_NDK_HOME" >&2
    exit 1
}
case "$(uname -s)" in
    Linux*) HOST_TAG="linux-x86_64" ;;
    Darwin*) HOST_TAG="darwin-x86_64" ;;
    *) echo "unsupported host: $(uname -s)" >&2; exit 1 ;;
esac
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/$HOST_TAG"
CLANG="$TOOLCHAIN/bin/${TARGET}${API}-clang"
if [[ ! -x "$CLANG" ]]; then
    echo "NDK clang not found at $CLANG" >&2
    exit 1
fi

export ANDROID_NDK_HOME="$NDK"
export ANDROID_NDK_ROOT="$NDK"
export ANDROID_HOME="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-/opt/android-sdk}}"
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$ANDROID_HOME}"
export ANDROID_ABI="arm64-v8a"
export ANDROID_PLATFORM="android-$API"
export CMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake"
export CC_aarch64_linux_android="$CLANG"
export CXX_aarch64_linux_android="${TOOLCHAIN}/bin/${TARGET}${API}-clang++"
export AR_aarch64_linux_android="${TOOLCHAIN}/bin/llvm-ar"
export RANLIB_aarch64_linux_android="${TOOLCHAIN}/bin/llvm-ranlib"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CLANG"

JNI_DIR="$ANDROID_DIR/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$JNI_DIR"

# Do not silently package a stub left by an earlier run.
if [[ -z "$GAME_LIB" && "$STUB_GAME" -eq 0 ]]; then
    rm -f "$JNI_DIR/librecomp_game.so"
fi

echo "[android] NDK:   $NDK"
echo "[android] target: $TARGET (API $API), profile $PROFILE"

echo "[android] building bootstrap (libmain.so)"
cargo build \
    --manifest-path "$ANDROID_DIR/native/Cargo.toml" \
    --target "$TARGET" \
    $CARGO_PROFILE_FLAG

BOOTSTRAP="$ANDROID_DIR/native/target/$TARGET/$PROFILE/libmain.so"
if [[ ! -f "$BOOTSTRAP" ]]; then
    echo "missing bootstrap output: $BOOTSTRAP" >&2
    exit 1
fi
cp -f "$BOOTSTRAP" "$JNI_DIR/libmain.so"

SDL3_LIB="$ANDROID_DIR/native/target/$TARGET/$PROFILE/libSDL3.so"
if [[ ! -f "$SDL3_LIB" ]]; then
    # sdl3-sys copies the library to the profile directory, but keep this
    # fallback for build-script layout changes.
    SDL3_LIB="$(find "$ANDROID_DIR/native/target/$TARGET" -path '*/build/sdl3-sys-*/out/lib/libSDL3.so' -print -quit || true)"
fi
if [[ ! -f "$SDL3_LIB" ]]; then
    echo "SDL3 was not built by sdl3-sys; check the build-from-source feature" >&2
    exit 1
fi
cp -f "$SDL3_LIB" "$JNI_DIR/libSDL3.so"

if [[ "$STUB_GAME" -eq 1 ]]; then
    echo "[android] building non-proprietary stub game library"
    cargo build \
        --manifest-path "$ANDROID_DIR/stub-game/Cargo.toml" \
        --target "$TARGET" \
        $CARGO_PROFILE_FLAG
    STUB_LIB="$ANDROID_DIR/stub-game/target/$TARGET/$PROFILE/librecomp_game.so"
    if [[ ! -f "$STUB_LIB" ]]; then
        echo "missing stub output: $STUB_LIB" >&2
        exit 1
    fi
    cp -f "$STUB_LIB" "$JNI_DIR/librecomp_game.so"
    "$ANDROID_DIR/scripts/check-game-lib.sh" "$JNI_DIR/librecomp_game.so"
fi

if [[ -n "$GAME_LIB" ]]; then
    if [[ ! -f "$GAME_LIB" ]]; then
        echo "game library not found: $GAME_LIB" >&2
        exit 1
    fi
    "$ANDROID_DIR/scripts/check-game-lib.sh" "$GAME_LIB"
    cp -f "$GAME_LIB" "$JNI_DIR/librecomp_game.so"
    echo "[android] packaged $(basename "$GAME_LIB") as librecomp_game.so"
fi

echo "[android] extracting SDL3 Java sources"
"$ANDROID_DIR/scripts/prepare-sdl3-java.sh" --profile "$PROFILE"

echo "[android] native payload:"
ls -lh "$JNI_DIR"
echo "[android] SHA-256:"
sha256sum "$JNI_DIR"/*.so
