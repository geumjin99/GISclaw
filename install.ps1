# GISclaw — install as a desktop application on Windows. No Docker.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# Creates a private Python environment next to this file (.venv\), a GISclaw.cmd
# launcher, and a Desktop shortcut. Your data goes to %LOCALAPPDATA%\GISclaw.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "`nInstalling uv (Python package manager)..." -ForegroundColor Cyan
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "`nSetting up Python 3.11..." -ForegroundColor Cyan
uv python install 3.11
uv venv --python 3.11 --quiet .venv
Write-Host "`nInstalling packages (a few hundred MB; the first time takes a while)..." -ForegroundColor Cyan
uv pip install --python .venv\Scripts\python.exe -r desktop\requirements-desktop.txt

$here = (Get-Location).Path
@"
@echo off
"$here\.venv\Scripts\pythonw.exe" "$here\desktop\launcher.py" %*
"@ | Set-Content -Path "$here\GISclaw.cmd" -Encoding ASCII

$shell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = $shell.CreateShortcut("$desktop\GISclaw.lnk")
$lnk.TargetPath = "$here\.venv\Scripts\pythonw.exe"
$lnk.Arguments = "`"$here\desktop\launcher.py`""
$lnk.WorkingDirectory = $here
$lnk.Save()

Write-Host "`nDone. Use the GISclaw shortcut on your Desktop, or run GISclaw.cmd." -ForegroundColor Green
Write-Host "Windows may show a SmartScreen notice the first time: More info -> Run anyway."
