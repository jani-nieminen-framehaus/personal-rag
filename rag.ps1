# =============================================================================
# rag.ps1 — PowerShell shiv launcher.
# Same as rag.bat but with proper error handling and colored output.
# First run: creates the venv + installs deps. After that: just runs cli.py.
#
# Usage:
#     .\rag.ps1                              (no args -> "rag start": bring up
#                                              the whole stack and open the GUI)
#     .\rag.ps1 ask "What was YaRN about?"
#     .\rag.ps1 ingest --markdown .\samples
#     .\rag.ps1 eval
#
# If you want `rag` on PATH (run from anywhere), create a Windows shortcut
# that points to rag.bat — that's the binary-feel.
# =============================================================================
$ErrorActionPreference = 'Stop'

$repo  = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv  = Join-Path $repo '.venv'
$pyExe = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path $pyExe)) {
    Write-Host "[rag] first run - creating venv and installing deps (this takes a few minutes)" -ForegroundColor Yellow
    & python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed - is Python 3.12 on PATH?" }
    & $pyExe -m pip install --upgrade pip | Out-Null
    & $pyExe -m pip install -r (Join-Path $repo 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    Write-Host "[rag] setup complete." -ForegroundColor Green
}

# No args -> default to "start" so a desktop shortcut (or a bare
# `rag.ps1` from the shell) brings up the whole stack.
if ($args.Count -eq 0) {
    & $pyExe (Join-Path $repo 'cli.py') start
} else {
    & $pyExe (Join-Path $repo 'cli.py') @args
}
exit $LASTEXITCODE
