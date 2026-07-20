#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
APP_NAME="IndexTTS Menu.app"
DIST="$ROOT/dist"
APP="$DIST/$APP_NAME"

cd "$ROOT"
swift build -c release

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$ROOT/.build/release/IndexTTSMenuBar" "$APP/Contents/MacOS/IndexTTSMenuBar"
cp "$ROOT/Packaging/Info.plist" "$APP/Contents/Info.plist"
chmod +x "$APP/Contents/MacOS/IndexTTSMenuBar"
/usr/bin/xattr -cr "$APP" 2>/dev/null || true

/usr/bin/codesign --force --deep --sign - "$APP"
/usr/bin/plutil -lint "$APP/Contents/Info.plist"

echo "Built: $APP"
