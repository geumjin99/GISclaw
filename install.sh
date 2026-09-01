#!/usr/bin/env bash
# GISclaw — install as a desktop application (macOS, Linux). No Docker.
#
#   bash install.sh
#
# Creates a private Python environment next to this file (.venv/), a `gisclaw`
# command, and on macOS a double-clickable GISclaw.app. Your data goes to the
# user data folder (see desktop/launcher.py), never into this folder.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# 1) uv — the installer that fetches Python and the wheels
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "uv is not on PATH; open a new terminal and run this again."; exit 1; }

# 2) Python 3.11 and the packages
say "Setting up Python 3.11…"
uv python install 3.11
uv venv --python 3.11 --quiet .venv
say "Installing packages (a few hundred MB; the first time takes a while)…"
uv pip install --python .venv/bin/python -r desktop/requirements-desktop.txt

# 3) the command
cat > gisclaw <<CMD
#!/usr/bin/env bash
exec "$HERE/.venv/bin/python" "$HERE/desktop/launcher.py" "\$@"
CMD
chmod +x gisclaw

# 4) macOS: an application bundle you can double-click and keep in the Dock
if [[ "$(uname)" == "Darwin" ]]; then
  APP="$HERE/GISclaw.app"
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
  cat > "$APP/Contents/MacOS/GISclaw" <<LAUNCH
#!/usr/bin/env bash
exec "$HERE/.venv/bin/python" "$HERE/desktop/launcher.py"
LAUNCH
  chmod +x "$APP/Contents/MacOS/GISclaw"
  cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>GISclaw</string>
  <key>CFBundleDisplayName</key><string>GISclaw</string>
  <key>CFBundleIdentifier</key><string>org.gisclaw.desktop</string>
  <key>CFBundleVersion</key><string>2.0.0</string>
  <key>CFBundleShortVersionString</key><string>2.0.0</string>
  <key>CFBundleExecutable</key><string>GISclaw</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
</dict></plist>
PLIST
  say "Done. Double-click GISclaw.app (or run ./gisclaw). To keep it: drag GISclaw.app to Applications."
else
  say "Done. Run ./gisclaw  (or ./gisclaw --browser to use your browser)."
fi
