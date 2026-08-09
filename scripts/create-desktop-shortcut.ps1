# =============================================================================
# create-desktop-shortcut.ps1 - pin a one-click "start the rag stack" shortcut
# on the user's desktop.
#
# The shortcut points at rag.bat, which defaults to `rag start` when invoked
# with no arguments. Double-clicking it brings up Ollama + Qdrant + the rag GUI
# in the background and opens the browser to http://localhost:8420.
#
# Idempotent: re-running just updates the existing shortcut.
# =============================================================================

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$TargetExe = Join-Path $RepoRoot 'rag.bat'
$ShortcutName = 'rag.lnk'

if (-not (Test-Path $TargetExe)) {
    Write-Host "ERROR: $TargetExe not found" -ForegroundColor Red
    exit 1
}

# Resolve the desktop folder for the current user.
$DesktopDir = [Environment]::GetFolderPath('Desktop')
if (-not (Test-Path $DesktopDir)) {
    Write-Host "ERROR: desktop folder not found at $DesktopDir" -ForegroundColor Red
    exit 1
}
$ShortcutPath = Join-Path $DesktopDir $ShortcutName

# Build the shortcut via WScript.Shell (the only PowerShell-native way
# without an extra dependency).
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($ShortcutPath)
$sc.TargetPath = $TargetExe
$sc.WorkingDirectory = $RepoRoot
$sc.WindowStyle = 7  # Minimized - the rag.bat window pops up briefly
                    # and then the orchestrator takes over; the user
                    # shouldn't see a flicker.
$sc.IconLocation = "shell32.dll,12"   # a generic app icon
$sc.Description = "Bring up the rag stack (Ollama + Qdrant + GUI) and open the browser."
$sc.Save()

Write-Host ""
Write-Host "  desktop shortcut created" -ForegroundColor Green
Write-Host "  path        : $ShortcutPath"
Write-Host "  target      : $TargetExe"
Write-Host "  behaviour   : double-click -> rag stack comes up + browser opens"
Write-Host ""
Write-Host "  to remove later:  del '$ShortcutPath'" -ForegroundColor Gray
Write-Host ""
