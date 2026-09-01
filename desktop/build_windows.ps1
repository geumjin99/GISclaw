# Build the Windows installer. Run from the repository root in PowerShell:
#
#   pwsh desktop/build_windows.ps1     # -> build\GISclaw-<version>-windows-x64-setup.exe
#
# Needs uv and Inno Setup 6 (both present on GitHub's windows runners). The
# installer carries its own Python (a relocatable build fetched by uv) with
# every package installed, plus the application, and puts shortcuts in the
# Start menu and on the Desktop.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$version = (Select-String -Path app\server.py -Pattern 'APP_VERSION = "([^"]+)"').Matches[0].Groups[1].Value
$short = $version.Split("-")[0]
$out = "build"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Path "$out\GISclaw\gisclaw" | Out-Null

Write-Host "== Python 3.11 (managed by uv), copied into the package"
$env:UV_PYTHON_PREFERENCE = "only-managed"
uv python install 3.11
$pybin = (uv python find 3.11).Trim()
$pyroot = (Get-Item (Resolve-Path $pybin)).Directory.FullName
Copy-Item -Recurse -Path $pyroot -Destination "$out\GISclaw\python"
$py = "$out\GISclaw\python\python.exe"
& $py -c "import sys; print(sys.version.split()[0], sys.prefix)"

Write-Host "== Packages"
uv pip install --python $py --break-system-packages -r desktop\requirements-desktop.txt
& $py -c "import geopandas, rasterio, pyproj, fiona, shapely, fastapi, webview; print('imports ok')"

Write-Host "== Application files"
foreach ($d in "app", "src", "desktop", "examples") { Copy-Item -Recurse $d "$out\GISclaw\gisclaw\$d" }
foreach ($f in "LICENSE", "COPYRIGHT", "DISCLAIMER.md", "THIRD_PARTY_NOTICES.md", "README.md", "CHANGELOG.md") { Copy-Item $f "$out\GISclaw\gisclaw\" }
Get-ChildItem -Recurse -Directory -Filter __pycache__ "$out\GISclaw\gisclaw" | Remove-Item -Recurse -Force
Remove-Item "$out\GISclaw\gisclaw\app\server.log" -ErrorAction SilentlyContinue

Write-Host "== Installer"
$iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
& $iscc "/DMyAppVersion=$short" "/DMyAppFullVersion=$version" desktop\windows\gisclaw.iss
Get-ChildItem "$out\*.exe" | ForEach-Object { Write-Host $_.FullName ("{0:N0} MB" -f ($_.Length / 1MB)) }
