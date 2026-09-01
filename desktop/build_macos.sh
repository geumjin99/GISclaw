#!/usr/bin/env bash
# Build GISclaw.app and a .dmg on macOS. Run from the repository root:
#
#   bash desktop/build_macos.sh          # -> build/GISclaw-<version>-macos-<arch>.dmg
#
# The bundle carries its own Python (a relocatable build fetched by uv) with
# every package installed, plus the application. Nothing is compiled or
# frozen: the interpreter inside the app is an ordinary Python, so code the
# model writes can import anything that is installed.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(python3 -c "import re;print(re.search(r'APP_VERSION = \"([^\"]+)\"', open('app/server.py').read()).group(1))")
SHORT=${VERSION%%-*}
ARCH=$(uname -m | sed 's/x86_64/x64/')
OUT=build
APP="$OUT/GISclaw.app"
RES="$APP/Contents/Resources"
rm -rf "$OUT"
mkdir -p "$APP/Contents/MacOS" "$RES/gisclaw"

echo "== Python 3.11 (managed by uv), copied into the bundle"
export UV_PYTHON_PREFERENCE=only-managed
uv python install 3.11
PYBIN=$(uv python find 3.11)
PYROOT=$(python3 -c "import os,sys;print(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[1]))))" "$PYBIN")
cp -RL "$PYROOT" "$RES/python"
PY="$RES/python/bin/python3"
"$PY" -c "import sys; print(sys.version.split()[0], sys.prefix)"

echo "== Packages"
uv pip install --python "$PY" --break-system-packages -r desktop/requirements-desktop.txt
"$PY" -c "import geopandas, rasterio, pyproj, fiona, shapely, fastapi, webview; print('imports ok')"

echo "== Application files"
for d in app src desktop examples; do cp -R "$d" "$RES/gisclaw/$d"; done
for f in LICENSE COPYRIGHT DISCLAIMER.md THIRD_PARTY_NOTICES.md README.md CHANGELOG.md; do cp "$f" "$RES/gisclaw/"; done
find "$RES/gisclaw" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -f "$RES/gisclaw/app/server.log"

echo "== Launcher and bundle metadata"
cat > "$APP/Contents/MacOS/GISclaw" <<'LAUNCH'
#!/bin/bash
HERE="$(cd "$(dirname "$0")/.." && pwd)"
exec "$HERE/Resources/python/bin/python3" "$HERE/Resources/gisclaw/desktop/launcher.py"
LAUNCH
chmod +x "$APP/Contents/MacOS/GISclaw"

ICONSET=$OUT/icon.iconset
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s desktop/icon.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s*2)) $((s*2)) desktop/icon.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$RES/GISclaw.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>GISclaw</string>
  <key>CFBundleDisplayName</key><string>GISclaw</string>
  <key>CFBundleIdentifier</key><string>org.gisclaw.desktop</string>
  <key>CFBundleVersion</key><string>$SHORT</string>
  <key>CFBundleShortVersionString</key><string>$SHORT</string>
  <key>CFBundleExecutable</key><string>GISclaw</string>
  <key>CFBundleIconFile</key><string>GISclaw</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHumanReadableCopyright</key><string>Copyright (C) 2026 Han Jinzhen. AGPL-3.0-or-later.</string>
</dict></plist>
PLIST

echo "== Ad-hoc signature (no developer certificate)"
codesign --force --deep -s - "$APP"
codesign --verify --deep "$APP" && echo "signature ok"

echo "== Disk image"
DMGROOT=$OUT/dmgroot
mkdir -p "$DMGROOT"
cp -R "$APP" "$DMGROOT/"
ln -s /Applications "$DMGROOT/Applications"
DMG="$OUT/GISclaw-$VERSION-macos-$ARCH.dmg"
hdiutil create -volname "GISclaw" -srcfolder "$DMGROOT" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$DMGROOT" "$ICONSET"
ls -lh "$DMG"
