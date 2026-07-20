#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
SOURCE="$ROOT/dist/IndexTTS Menu.app"
TARGET="/Applications/IndexTTS Menu.app"

if [[ ! -d "$SOURCE" ]]; then
  "$ROOT/scripts/build_app.sh"
fi

rm -rf "$TARGET"
cp -R "$SOURCE" "$TARGET"
/usr/bin/xattr -cr "$TARGET" 2>/dev/null || true
/usr/bin/codesign --force --deep --sign - "$TARGET"
/usr/bin/codesign --verify --deep --strict "$TARGET"
echo "Installed: $TARGET"
