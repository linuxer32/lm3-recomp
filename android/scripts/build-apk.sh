#!/usr/bin/env bash
# Build a signed debug APK without Gradle, using the Android SDK build-tools.
#
# This is the reproducible fallback for machines without Gradle. The Gradle
# project in this directory is the Android Studio path; both consume the same
# app/src/main tree and the same native jniLibs output.
set -euo pipefail

ANDROID_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ANDROID_DIR/app"
BUILD_DIR="$ANDROID_DIR/build/apk"
PACKAGE="dev.switchrecomp.android"
MIN_SDK="${ANDROID_MIN_SDK:-24}"
TARGET_SDK="${ANDROID_TARGET_SDK:-35}"
VERSION_CODE="1"
VERSION_NAME="0.1.0"
DEBUG=0
STUB_GAME=0
GAME_LIB=""
ASSETS_DIR=""
SKIP_NATIVE=0
OUTPUT=""

usage() {
    cat <<'EOF'
usage: scripts/build-apk.sh [options]

options:
  --debug              build a debuggable APK and debug native profile
  --stub-game          package the non-proprietary stub as librecomp_game.so
  --game-lib PATH      package PATH as lib/arm64-v8a/librecomp_game.so
  --assets DIR         copy DIR into the APK assets/ tree (user-supplied only)
  --skip-native        reuse existing app/src/main/jniLibs
  --output PATH        APK output path (default: build/SwitchRecomp.apk)
  --sdk PATH           Android SDK root (default: $ANDROID_HOME or /opt/android-sdk)
  -h, --help           show this help

The APK is signed with a debug key generated at android/.debug.keystore. That
key is gitignored and is not a distribution key.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG=1
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
        --assets)
            ASSETS_DIR="${2:?--assets needs a directory}"
            shift 2
            ;;
        --skip-native)
            SKIP_NATIVE=1
            shift
            ;;
        --output)
            OUTPUT="${2:?--output needs a path}"
            shift 2
            ;;
        --sdk)
            ANDROID_HOME="${2:?--sdk needs a path}"
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

if [[ -n "$GAME_LIB" && "$STUB_GAME" -eq 1 ]]; then
    echo "--game-lib and --stub-game are mutually exclusive" >&2
    exit 2
fi

SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-/opt/android-sdk}}"
BUILD_TOOLS="$(find "$SDK/build-tools" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
if [[ -z "$BUILD_TOOLS" ]]; then
    echo "Android build-tools not found under $SDK" >&2
    exit 1
fi
AAPT2="$BUILD_TOOLS/aapt2"
D8="$BUILD_TOOLS/d8"
ZIPALIGN="$BUILD_TOOLS/zipalign"
APKSIGNER="$BUILD_TOOLS/apksigner"
for tool in "$AAPT2" "$D8" "$ZIPALIGN" "$APKSIGNER"; do
    if [[ ! -x "$tool" ]]; then
        echo "missing build tool: $tool" >&2
        exit 1
    fi
done

ANDROID_JAR="$SDK/platforms/android-$TARGET_SDK/android.jar"
if [[ ! -f "$ANDROID_JAR" ]]; then
    ANDROID_JAR="$(find "$SDK/platforms" -name android.jar | sort -V | tail -n 1)"
fi
if [[ ! -f "$ANDROID_JAR" ]]; then
    echo "Android platform android.jar not found under $SDK/platforms" >&2
    exit 1
fi

if [[ "$SKIP_NATIVE" -eq 0 ]]; then
    native_args=(--release)
    if [[ "$DEBUG" -eq 1 ]]; then
        native_args=(--debug)
    fi
    if [[ "$STUB_GAME" -eq 1 ]]; then
        native_args+=(--stub-game)
    fi
    if [[ -n "$GAME_LIB" ]]; then
        native_args+=(--game-lib "$GAME_LIB")
    fi
    "$ANDROID_DIR/scripts/build-native.sh" "${native_args[@]}"
else
    "$ANDROID_DIR/scripts/prepare-sdl3-java.sh" --profile "$([[ "$DEBUG" -eq 1 ]] && echo debug || echo release)"
fi

JNI_DIR="$APP_DIR/src/main/jniLibs/arm64-v8a"
if [[ ! -f "$JNI_DIR/libmain.so" || ! -f "$JNI_DIR/libSDL3.so" ]]; then
    echo "native payload incomplete in $JNI_DIR; run scripts/build-native.sh" >&2
    exit 1
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/assets" "$BUILD_DIR/gen" "$BUILD_DIR/classes" "$BUILD_DIR/dex" "$BUILD_DIR/apk"
if [[ -n "$ASSETS_DIR" ]]; then
    if [[ ! -d "$ASSETS_DIR" ]]; then
        echo "assets directory not found: $ASSETS_DIR" >&2
        exit 1
    fi
    cp -a "$ASSETS_DIR/." "$BUILD_DIR/assets/"
fi

# aapt2 link requires a package attribute; the source manifest deliberately
# has none so Gradle can use the namespace in app/build.gradle.kts.
awk -v pkg="$PACKAGE" '
    !injected && /<manifest / {
        sub(/<manifest /, "<manifest package=\"" pkg "\" ")
        injected = 1
    }
    { print }
' "$APP_DIR/src/main/AndroidManifest.xml" > "$BUILD_DIR/AndroidManifest.xml"

echo "[android] compiling resources"
"$AAPT2" compile --dir "$APP_DIR/src/main/res" -o "$BUILD_DIR/resources.zip"

echo "[android] linking resources and manifest"
LINK_ARGS=(
    -o "$BUILD_DIR/unsigned.apk"
    -I "$ANDROID_JAR"
    --manifest "$BUILD_DIR/AndroidManifest.xml"
    -R "$BUILD_DIR/resources.zip"
    -A "$BUILD_DIR/assets"
    --java "$BUILD_DIR/gen"
    --min-sdk-version "$MIN_SDK"
    --target-sdk-version "$TARGET_SDK"
    --version-code "$VERSION_CODE"
    --version-name "$VERSION_NAME"
    --auto-add-overlay
)
if [[ "$DEBUG" -eq 1 ]]; then
    LINK_ARGS+=(--debug-mode)
fi
"$AAPT2" link "${LINK_ARGS[@]}"

echo "[android] compiling Java (SDL3 bridge + RecompActivity)"
JAVA_FILES=()
while IFS= read -r -d '' file; do
    JAVA_FILES+=("$file")
done < <(find "$APP_DIR/src/main/java" "$BUILD_DIR/gen" -name '*.java' -print0)
if [[ ${#JAVA_FILES[@]} -eq 0 ]]; then
    echo "no Java sources found" >&2
    exit 1
fi
javac \
    --release 8 \
    -encoding UTF-8 \
    -Xlint:-options \
    -cp "$ANDROID_JAR" \
    -d "$BUILD_DIR/classes" \
    "${JAVA_FILES[@]}"

echo "[android] dexing"
CLASS_FILES=()
while IFS= read -r -d '' file; do
    CLASS_FILES+=("$file")
done < <(find "$BUILD_DIR/classes" -name '*.class' -print0)
"$D8" \
    --release \
    --min-api "$MIN_SDK" \
    --lib "$ANDROID_JAR" \
    --output "$BUILD_DIR/dex" \
    "${CLASS_FILES[@]}"

echo "[android] assembling APK"
cp "$BUILD_DIR/dex/classes.dex" "$BUILD_DIR/apk/classes.dex"
mkdir -p "$BUILD_DIR/apk/lib/arm64-v8a"
cp "$JNI_DIR"/*.so "$BUILD_DIR/apk/lib/arm64-v8a/"
(
    cd "$BUILD_DIR/apk"
    zip -q -r "$BUILD_DIR/unsigned.apk" classes.dex lib
)

"$ZIPALIGN" -f -p 4 "$BUILD_DIR/unsigned.apk" "$BUILD_DIR/aligned.apk"

KEYSTORE="${RECOMP_ANDROID_KEYSTORE:-$ANDROID_DIR/.debug.keystore}"
if [[ ! -f "$KEYSTORE" ]]; then
    echo "[android] generating debug keystore: $KEYSTORE"
    keytool -genkeypair \
        -keystore "$KEYSTORE" \
        -storepass android \
        -alias androiddebugkey \
        -keypass android \
        -keyalg RSA \
        -keysize 2048 \
        -validity 10000 \
        -dname "CN=Android Debug,O=Android,C=US" \
        >/dev/null 2>&1
fi

OUTPUT="${OUTPUT:-$ANDROID_DIR/build/SwitchRecomp.apk}"
mkdir -p "$(dirname "$OUTPUT")"
"$APKSIGNER" sign \
    --ks "$KEYSTORE" \
    --ks-key-alias androiddebugkey \
    --ks-pass pass:android \
    --key-pass pass:android \
    --min-sdk-version "$MIN_SDK" \
    --out "$OUTPUT" \
    "$BUILD_DIR/aligned.apk"

"$APKSIGNER" verify --min-sdk-version "$MIN_SDK" "$OUTPUT" >/dev/null
"$AAPT2" dump badging "$OUTPUT" > "$BUILD_DIR/badging.txt"

echo
echo "[android] APK: $OUTPUT"
echo "[android] size: $(du -h "$OUTPUT" | cut -f1)"
echo "[android] sha256: $(sha256sum "$OUTPUT" | cut -d' ' -f1)"
echo "[android] ABI/native entries:"
unzip -l "$OUTPUT" 'lib/*'
echo
grep -E "^(package|sdkVersion|targetSdkVersion|native-code|launchable-activity)" "$BUILD_DIR/badging.txt" || true
