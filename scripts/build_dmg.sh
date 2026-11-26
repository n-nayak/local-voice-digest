#!/usr/bin/env bash
set -euo pipefail

# Build LocalTranscribe.app and LocalTranscribe.dmg for macOS (Apple Silicon).
#
# Usage:
#   ./scripts/build_dmg.sh                  # Build SwiftUI .app + .dmg (uses local .venv in dev)
#   ./scripts/build_dmg.sh --self-contained # Also bundle standalone Python backend + ffmpeg inside .app

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_PKG="$ROOT_DIR/macos/LocalTranscribeApp"
DIST_DIR="$ROOT_DIR/dist"
APP_NAME="LocalTranscribe"
APP_BUNDLE="$DIST_DIR/${APP_NAME}.app"
DMG_PATH="$DIST_DIR/${APP_NAME}.dmg"

SELF_CONTAINED=false
for arg in "$@"; do
    case "$arg" in
        --self-contained) SELF_CONTAINED=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

echo "==> [1/4] Building SwiftUI macOS app (release)..."
swift build --package-path "$SWIFT_PKG" -c release
SWIFT_BIN="$SWIFT_PKG/.build/release/LocalTranscribeApp"

echo "==> [2/4] Assembling ${APP_NAME}.app bundle..."
rm -rf "$APP_BUNDLE" "$DMG_PATH"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"
cp "$SWIFT_BIN" "$APP_BUNDLE/Contents/MacOS/LocalTranscribeApp"
cp "$SWIFT_PKG/Info.plist" "$APP_BUNDLE/Contents/Info.plist"
if [ -f "$SWIFT_PKG/Assets/AppIcon.icns" ]; then
    cp "$SWIFT_PKG/Assets/AppIcon.icns" "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
fi
chmod +x "$APP_BUNDLE/Contents/MacOS/LocalTranscribeApp"

if [ "$SELF_CONTAINED" = true ]; then
    echo "==> [2b/4] Bundling self-contained Python MLX/Sherpa-ONNX backend & ffmpeg..."
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "Error: $PYTHON_BIN not found. Create the .venv first." >&2
        exit 1
    fi
    if ! "$PYTHON_BIN" -m PyInstaller --version >/dev/null 2>&1; then
        echo "Error: PyInstaller not installed in .venv. Run: uv pip install pyinstaller" >&2
        exit 1
    fi

    "$PYTHON_BIN" -m PyInstaller \
        --noconfirm \
        --onedir \
        --name transcribe-cli \
        --distpath "$DIST_DIR/pyinstaller_dist" \
        --workpath "$DIST_DIR/pyinstaller_build" \
        --specpath "$DIST_DIR/pyinstaller_build" \
        --collect-all mlx \
        --collect-all mlx_whisper \
        --collect-all mlx_lm \
        --collect-all sherpa_onnx \
        --collect-all opensmile \
        --collect-all soundfile \
        --exclude-module torch \
        --exclude-module torchaudio \
        --exclude-module whisperx \
        --exclude-module pyannote \
        --exclude-module speechbrain \
        "$ROOT_DIR/transcribe.py"

    mkdir -p "$APP_BUNDLE/Contents/Resources/backend"
    cp -R "$DIST_DIR/pyinstaller_dist/transcribe-cli/"* "$APP_BUNDLE/Contents/Resources/backend/"

    FFMPEG_BIN="$(command -v ffmpeg || true)"
    if [ -n "$FFMPEG_BIN" ]; then
        mkdir -p "$APP_BUNDLE/Contents/Resources/bin"
        cp "$FFMPEG_BIN" "$APP_BUNDLE/Contents/Resources/bin/ffmpeg"
        chmod +x "$APP_BUNDLE/Contents/Resources/bin/ffmpeg"
    else
        echo "Warning: ffmpeg not found on PATH; bundle a static arm64 ffmpeg into Contents/Resources/bin/ffmpeg." >&2
    fi
fi

echo "==> [3/4] Ad-hoc code signing ${APP_NAME}.app..."
codesign --deep --force --sign - "$APP_BUNDLE"

echo "==> [4/4] Creating drag-and-drop ${APP_NAME}.dmg..."
STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR"' EXIT
cp -R "$APP_BUNDLE" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$STAGING_DIR" \
    -ov \
    -format UDZO \
    "$DMG_PATH"

echo ""
echo "Built app: $APP_BUNDLE"
echo "Built DMG: $DMG_PATH"
