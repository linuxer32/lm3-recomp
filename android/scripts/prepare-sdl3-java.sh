#!/usr/bin/env bash
# Extract the official SDL3 Android Java sources from the sdl3-sys build output.
#
# sdl3-sys' build-from-source installs share/java/SDL3/SDL3-<version>-sources.jar
# next to libSDL3.so. SDL3's Android integration is Java (SDLActivity and the
# org.libsdl.app helpers) plus native JNI entry points in libSDL3.so. Keeping the
# Java in the generated tree avoids vendoring a copy that can drift from the
# linked C library, and avoids committing a third-party source dump.
set -euo pipefail

ANDROID_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="aarch64-linux-android"
PROFILE="release"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="${2:?--profile needs debug or release}"
            shift 2
            ;;
        -h|--help)
            echo "usage: scripts/prepare-sdl3-java.sh [--profile debug|release]"
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            exit 2
            ;;
    esac
done

JAVA_DIR="$ANDROID_DIR/app/src/main/java"
DEST="$JAVA_DIR/org/libsdl/app"
JAR="$(find "$ANDROID_DIR/native/target/$TARGET" \
    -path "*/build/sdl3-sys-*/out/share/java/SDL3/SDL3-*-sources.jar" \
    -print -quit 2>/dev/null || true)"

if [[ -z "$JAR" || ! -f "$JAR" ]]; then
    echo "SDL3 Java sources not found under native/target/$TARGET." >&2
    echo "Run scripts/build-native.sh first (it builds SDL3 from source)." >&2
    exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"
unzip -q "$JAR" 'org/libsdl/app/*.java' -d "$JAVA_DIR"

LICENSE="$(find "$ANDROID_DIR/native/target/$TARGET" \
    -path "*/build/sdl3-sys-*/out/share/licenses/SDL3/LICENSE.txt" \
    -print -quit 2>/dev/null || true)"
if [[ -n "$LICENSE" && -f "$LICENSE" ]]; then
    cp -f "$LICENSE" "$DEST/LICENSE-SDL3.txt"
fi

echo "[android] SDL3 Java sources: $DEST ($(find "$DEST" -name '*.java' | wc -l) files)"
