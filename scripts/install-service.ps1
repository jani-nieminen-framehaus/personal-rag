# =============================================================================
# install-service.ps1 — register rag in Windows Task Scheduler.
#
# Idempotent: running it twice just updates the existing task. To remove
# the auto-launch, run scripts\uninstall-service.ps1.
#
# The scheduled task runs `rag start --no-browser` at user logon. That
# orchestrator:
#   1. Checks Ollama — starts it if not reachable.
#   2. Checks Qdrant — starts it if not reachable.
#   3. Spawns the rag GUI server (FastAPI on 127.0.0.1:8420) as a
#      detached process so closing the scheduled-task process tree
#      doesn't kill the server.
#   4. Returns. The server keeps running.
#
# After this script completes, every time you log in to Windows, the
# whole stack comes up in the background. Find the GUI with
# `rag status` (terminal) or just open http://localhost:8420 in a browser.
# =============================================================================

$ErrorActionPreference = 'Stop'

$TaskName    = 'rag-gui'
$Description = 'rag personal RAG stack (Ollama + Qdrant + GUI on 127.0.0.1:8420)'
$RepoRoot    = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonExe   = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$CliPath     = Join-Path $RepoRoot 'cli.py'
$VenvDir     = Join-Path $RepoRoot '.venv'

# Sanity: venv must exist (created by rag.ps1 / rag.bat on first run).
if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: venv not found at $VenvDir" -ForegroundColor Red
    Write-Host "Run rag.ps1 once (or: python -m venv .venv && pip install -r requirements.txt)" -ForegroundColor Yellow
    exit 1
}

# Build the action — we run `rag start --no-browser` from the repo root
# so config.yaml is found and the state file lands in the user's home.
# `start` does the orchestration (Ollama + Qdrant + rag server).
$ActionArgs = '-m cli start --no-browser'
$Action     = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $ActionArgs `
    -WorkingDirectory $RepoRoot

# Trigger: at user logon. DelayStart so we don't race the desktop
# (Ollama and Qdrant take a moment to be reachable).
$Trigger = New-ScheduledTaskTrigger -AtLogOn
$Trigger.Delay = '00:00:15'   # 15s after logon

# Settings: restart on failure up to 3 times, 30s apart, run only
# when the user is logged on (interactive session so the browser can
# open localhost). If the user prefers the always-on story, they can
# switch the principal to LocalSystem later via Task Scheduler UI.
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)  # 0 = no time limit

# Principal: current user, interactive only.
$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().User) `
    -LogonType Interactive `
    -RunLevel Highest

# Register (or update).
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description $Description `
        -Force | Out-Null
} catch {
    Write-Host "ERROR: failed to register scheduled task: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  rag stack scheduled task installed" -ForegroundColor Green
Write-Host "  name        : $TaskName"
Write-Host "  python      : $PythonExe"
Write-Host "  args        : $ActionArgs"
Write-Host "  working dir : $RepoRoot"
Write-Host "  trigger     : AtLogOn (+15s delay)"
Write-Host "  restart     : up to 3x on failure"
Write-Host ""
Write-Host "  on next logon the orchestrator will start:" -ForegroundColor Gray
Write-Host "    1. Ollama (if not running) -> http://localhost:11434" -ForegroundColor Gray
Write-Host "    2. Qdrant (if not running) -> http://localhost:7333" -ForegroundColor Gray
Write-Host "    3. rag GUI server          -> http://localhost:8420" -ForegroundColor Gray
Write-Host ""
Write-Host "  next steps:" -ForegroundColor Cyan
Write-Host "    1. Start it now (no need to log out):" -ForegroundColor Gray
Write-Host "         Start-ScheduledTask -TaskName $TaskName" -ForegroundColor Gray
Write-Host "    2. Find the URL once it's up:" -ForegroundColor Gray
Write-Host "         rag status" -ForegroundColor Gray
Write-Host "         rag url" -ForegroundColor Gray
Write-Host "    3. To remove the auto-launch later:" -ForegroundColor Gray
Write-Host "         scripts\uninstall-service.ps1" -ForegroundColor Gray
Write-Host ""
