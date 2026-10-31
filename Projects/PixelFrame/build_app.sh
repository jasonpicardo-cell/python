#!/bin/bash
# Builds build/PixelFrame.app (arm64, release). Needs Xcode or Command Line Tools: xcode-select --install
set -euo pipefail
cd "$(dirname "$0")"

swift build -c release --arch arm64
BIN="$(swift build -c release --arch arm64 --show-bin-path)/PixelFrame"

APP="build/PixelFrame.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/PixelFrame"
strip -x "$APP/Contents/MacOS/PixelFrame"
cp Resources/Info.plist "$APP/Contents/Info.plist"
codesign --force --sign - "$APP"

echo "Built: $PWD/$APP"
echo "Run:   open \"$PWD/$APP\""
