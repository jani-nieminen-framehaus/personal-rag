# =============================================================================
# start_all.ps1 - rag boot orchestrator.
#
# Idempotent. Checks each backend (Ollama, Qdrant, rag server). Starts the
# ones that aren't running. Polls the rag server until /api/health says
# ready. Opens the default browser to the GUI URL.
#
# This is the "double-click and go" entry point. Pin a shortcut to this
# script on the desktop / taskbar and you get the full stack in one click.
#
# On logon, scripts/install-service.ps1 schedules `python -m cli serve`
# directly (no browser, no console), so the server is also auto-launched.
# This script is for manual restarts / wake-from-sleep / cold boot.
#
# Configuration (overridable via env vars or function parameters):
#   RAG_REPO          path to the rag repo (default: parent of this script)
#   RAG_VENV          path to the venv (default: $RAG_REPO\.venv)
#   RAG_PYTHON        python.exe inside the venv (default: $RAG_VENV\Scripts\python.exe)
#   RAG_HOST          GUI bind host (default: 127.0.0.1)
#   RAG_PORT          GUI bind port (default: 8420)
#   OLLAMA_URL        Ollama health probe (default: http://localhost:11434/api/tags)
#   OLLAMA_BIN        ollama binary (default: ollama on PATH)
#   QDRANT_URL        Qdrant health probe (default: http://localhost:7333/)
#   QDRANT_BIN        qdrant.exe (default: C:\Tools\qdrant\qdrant.exe)
#   QDRANT_ARGS       extra args for qdrant.exe (default: --storage-snapshots-dir C:\qdrant\storage)
#   WAIT_TIMEOUT_S    max seconds to wait for the rag server (default: 120)
# =============================================================================

[CmdletBinding()]
param(
    [switch]$NoBrowser   # Don't open the browser at the end (for headless / smoke).
)

$ErrorActionPreference = 'Stop'

# Shared constants + helpers (TaskName, RagPort, StateFile, Read-StateJson,
# Test-PidAlive). Python mirror: service_state.py at the repo root.
. (Join-Path $PSScriptRoot '_config.ps1')

# -- config -----------------------------------------------------------------

$RepoRoot    = if ($env:RAG_REPO) { $env:RAG_REPO } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$VenvDir     = if ($env:RAG_VENV) { $env:RAG_VENV } else { Join-Path $RepoRoot '.venv' }
$PythonExe   = if ($env:RAG_PYTHON) { $env:RAG_PYTHON } else { Join-Path $VenvDir 'Scripts\python.exe' }
$BindHost    = if ($env:RAG_HOST) { $env:RAG_HOST } else { '127.0.0.1' }
$Port        = $RagPort
$OllamaUrl   = if ($env:OLLAMA_URL) { $env:OLLAMA_URL } else { 'http://localhost:11434/api/tags' }
$OllamaBin   = if ($env:OLLAMA_BIN) { $env:OLLAMA_BIN } else { 'ollama' }
$QdrantUrl   = if ($env:QDRANT_URL) { $env:QDRANT_URL } else { 'http://localhost:7333/' }
$QdrantBin   = if ($env:QDRANT_BIN) { $env:QDRANT_BIN } else { 'C:\Tools\qdrant\qdrant.exe' }
$QdrantArgs  = if ($env:QDRANT_ARGS) { $env:QDRANT_ARGS } else { '--storage-snapshots-dir C:\qdrant\storage' }
$WaitTimeout = if ($env:WAIT_TIMEOUT_S) { [int]$env:WAIT_TIMEOUT_S } else { 120 }

$GuiUrl = "http://localhost:$Port"

# -- helpers ----------------------------------------------------------------

function Write-Stage($n, $name) {
    Write-Host ""
    Write-Host "  [$n] $name" -ForegroundColor Cyan
}

function Test-Healthy {
    param([string]$Url, [int]$TimeoutS = 10)
    try {
        $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutS -ErrorAction Stop
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400)
    } catch {
        return $false
    }
}

function Wait-Healthy {
    param([string]$Url, [int]$TimeoutS, [string]$Label)
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        if (Test-Healthy -Url $Url) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

# Read-StateJson / Test-PidAlive come from _config.ps1.

function Is-RagServer-Running {
    $state = Read-StateJson
    if ($state -and $state.pid) {
        if (Test-PidAlive -ProcessId ([int]$state.pid)) { return $true }
    }
    return $false
}

# -- preflight --------------------------------------------------------------

if (-not (Test-Path $PythonExe)) {
    Write-Host "  [fatal] venv not found at $VenvDir" -ForegroundColor Red
    Write-Host "  run rag.bat once (or: python -m venv .venv && pip install -r requirements.txt)" -ForegroundColor Yellow
    exit 1
}

# -- 1. rag server ----------------------------------------------------------

Write-Stage 1 "rag server"
if (Is-RagServer-Running) {
    Write-Host "    already running (pid=$((Read-StateJson).pid)) -> $GuiUrl" -ForegroundColor Green
} else {
    Write-Host "    not running, starting..."

    # Preferred: trigger the Task Scheduler task if it's installed. The
    # task runs in its own session, fully detached, so closing the
    # orchestrator (or its parent terminal) won't kill the server.
    $taskTriggered = $false
    try {
        $null = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
        Start-ScheduledTask -TaskName $TaskName
        $taskTriggered = $true
        Write-Host "    triggered scheduled task '$TaskName'" -ForegroundColor Gray
    } catch {
        # No scheduled task installed - fall back to a manual spawn.
    }

    if (-not $taskTriggered) {
        Write-Host "    (no scheduled task found; spawning detached process)"
        # Use cmd.exe to fully detach - Start-Process alone keeps the
        # child in our job object on Windows. The double-quoted empty
        # string is the "title" arg that cmd /c requires.
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $PythonExe
        $psi.Arguments = "-m cli serve"
        $psi.WorkingDirectory = $RepoRoot
        $psi.UseShellExecute = $true
        $psi.WindowStyle = 'Hidden'
        [System.Diagnostics.Process]::Start($psi) | Out-Null
    }

    # Poll the health endpoint. Default timeout is 120s, which is
    # enough for the embedder + reranker to load on first run.
    $ready = Wait-Healthy -Url "$GuiUrl/api/health" -TimeoutS $WaitTimeout -Label "rag"
    if (-not $ready) {
        Write-Host "    [fatal] rag server did not become ready within $WaitTimeout s" -ForegroundColor Red
        Write-Host "    check $StateFile for the state file, or run rag serve manually" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "    ready -> $GuiUrl" -ForegroundColor Green
}

# -- 2. Ollama --------------------------------------------------------------

Write-Stage 2 "Ollama (generator backend)"
if (Test-Healthy -Url $OllamaUrl) {
    Write-Host "    already running -> $OllamaUrl" -ForegroundColor Green
} else {
    Write-Host "    not reachable, attempting to start..."
    # On Windows, Ollama installs a service that auto-starts at logon.
    # If that's not the case, fall back to `ollama serve` in a hidden window.
    $started = $false
    try {
        Start-Process -FilePath $OllamaBin -ArgumentList 'serve' -WindowStyle Hidden -ErrorAction Stop
        $started = $true
    } catch {
        Write-Host "    [warn] could not spawn '$OllamaBin serve': $_" -ForegroundColor Yellow
    }
    if ($started) {
        $ok = Wait-Healthy -Url $OllamaUrl -TimeoutS 30 -Label "ollama"
        if ($ok) { Write-Host "    ready -> $OllamaUrl" -ForegroundColor Green }
        else     { Write-Host "    [warn] Ollama did not respond within 30s" -ForegroundColor Yellow }
    }
}

# -- 3. Qdrant --------------------------------------------------------------

Write-Stage 3 "Qdrant (vector store)"
if (Test-Healthy -Url $QdrantUrl) {
    Write-Host "    already running -> $QdrantUrl" -ForegroundColor Green
} else {
    Write-Host "    not reachable, attempting to start..."
    if (-not (Test-Path $QdrantBin)) {
        Write-Host "    [warn] qdrant.exe not found at $QdrantBin" -ForegroundColor Yellow
        Write-Host "    skip; install Qdrant to $QdrantBin or set \$env:QDRANT_BIN" -ForegroundColor Yellow
    } else {
        try {
            $qargs = $QdrantArgs -split ' '
            Start-Process -FilePath $QdrantBin -ArgumentList $qargs -WindowStyle Hidden -ErrorAction Stop
            $ok = Wait-Healthy -Url $QdrantUrl -TimeoutS 30 -Label "qdrant"
            if ($ok) { Write-Host "    ready -> $QdrantUrl" -ForegroundColor Green }
            else     { Write-Host "    [warn] Qdrant did not respond within 30s" -ForegroundColor Yellow }
        } catch {
            Write-Host "    [warn] could not start $QdrantBin : $_" -ForegroundColor Yellow
        }
    }
}

# -- 4. open browser --------------------------------------------------------

Write-Stage 4 "open GUI"
if (-not $NoBrowser) {
    Write-Host "    opening $GuiUrl in default browser" -ForegroundColor Cyan
    Start-Process $GuiUrl
} else {
    Write-Host "    skip (NoBrowser). GUI: $GuiUrl" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "  rag stack ready." -ForegroundColor Green
Write-Host "  url : $GuiUrl" -ForegroundColor Green
Write-Host "  tip : rag status / rag stats" -ForegroundColor Gray
Write-Host ""
exit 0
