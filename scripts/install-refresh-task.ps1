# =============================================================================
# install-refresh-task.ps1 — schedule `rag refresh` so the index keeps itself
# current without you remembering to do it.
#
# Idempotent: any existing task of this name is removed first, so re-running
# with a different -At just moves the schedule.
#
# What gets scheduled:
#     <repo>\.venv\Scripts\python.exe -m cli refresh
#
# `refresh` re-ingests only the files whose CONTENT changed since the last
# run, working from the `sources:` list in config.yaml. An unchanged corpus
# costs a directory walk and one SHA-256 per file — no embedding model is
# loaded — so a nightly run over a quiet corpus is milliseconds and no VRAM.
#
# IT NEVER DELETES. `--prune` (drop index entries whose file is gone from
# disk) is deliberately NOT scheduled: it is the one irreversible thing here
# and it asks for confirmation. Run it by hand when you want it, after
# `rag refresh --dry-run` has shown you exactly what it would remove.
#
# Remove the schedule with scripts\uninstall-refresh-task.ps1.
# =============================================================================
param(
    # Time of day to run, local time. Anything Get-Date parses: '03:00',
    # '3am', '23:30'. Only the time part is used; the recurrence is daily.
    [string]$At = '03:00'
)

$ErrorActionPreference = 'Stop'

# Shared constants ($RefreshTaskName etc.). Python mirror: service_state.py.
. (Join-Path $PSScriptRoot '_config.ps1')

$Description = 'rag: re-ingest changed files into the index. Never deletes — --prune stays manual.'
$RepoRoot    = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonExe   = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$VenvDir     = Join-Path $RepoRoot '.venv'

# Sanity: venv must exist (created by rag.ps1 / rag.bat on first run).
# Checked here, at install time, rather than letting the task discover it at
# 03:00 — and the task runs python directly rather than through rag.ps1 so a
# missing venv fails loudly instead of quietly starting a multi-GB pip install
# in the middle of the night.
if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: venv not found at $VenvDir" -ForegroundColor Red
    Write-Host "Run rag.ps1 once (or: python -m venv .venv && pip install -r requirements.txt)" -ForegroundColor Yellow
    exit 1
}

# Parse -At up front so a typo is an error here and not a task that silently
# runs at midnight.
try {
    $RunAt = Get-Date $At
} catch {
    Write-Host "ERROR: -At '$At' is not a time I can parse. Try '03:00' or '3am'." -ForegroundColor Red
    exit 1
}

# A task that can never do anything is worse than no task at all: it looks
# installed. `refresh` works exclusively from the `sources:` list, which ships
# empty, so check before promising the user a working schedule. Advisory only
# — a probe that fails must not block the install.
$SourceCount = $null
$ProbeError  = $null
try {
    Push-Location $RepoRoot
    $probe = 'from core.pipeline import load_config, configured_sources; print(len(configured_sources(load_config())))'
    $out = & $PythonExe -c $probe 2>&1
    if ($LASTEXITCODE -eq 0) {
        $SourceCount = [int](($out | Select-Object -Last 1) -as [string]).Trim()
    } else {
        $ProbeError = ($out | Out-String).Trim()
    }
} catch {
    $ProbeError = $_.Exception.Message
} finally {
    Pop-Location
}

# The action. Same shape as install-service.ps1: the venv interpreter, run
# from the repo root so config.yaml and the metadata DB resolve.
$ActionArgs = '-m cli refresh'
$Action     = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $ActionArgs `
    -WorkingDirectory $RepoRoot

$Trigger = New-ScheduledTaskTrigger -Daily -At $RunAt

# StartWhenAvailable: if the machine was off at 03:00, run once it is back
# rather than skipping the day entirely.
#
# Battery settings mirror install-service.ps1. A desktop behind a UPS reads as
# "on battery" during a mains blip, and being stopped mid-ingest is worse than
# the power draw: it leaves files half-indexed.
#
# ExecutionTimeLimit is deliberately NOT unlimited. Task Scheduler will not
# start a second instance while one is running, so a single wedged run would
# silently cancel every future refresh. Four hours is far beyond any real
# incremental run; after that, being killed is the correct outcome.
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4)

# Principal: the current user, interactive, UNELEVATED.
#
# -RunLevel Limited, not Highest. The P2 audit found the existing GUI task
# asking for elevation it does not need, and a scheduled task that inherits an
# elevated token hands that token to everything it spawns — here, an ingest
# pipeline that parses arbitrary PDFs, EPUBs and HTML from disk. Reading your
# own notes and writing to your own Qdrant needs nothing an admin has.
$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().User) `
    -LogonType Interactive `
    -RunLevel Limited

# Idempotence: remove any existing task of this name first, so a re-run is a
# clean re-register rather than an update of whatever was there before.
$existing = Get-ScheduledTask -TaskName $RefreshTaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  replacing the existing '$RefreshTaskName' task" -ForegroundColor Gray
    Unregister-ScheduledTask -TaskName $RefreshTaskName -Confirm:$false
}

try {
    Register-ScheduledTask `
        -TaskName $RefreshTaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description $Description | Out-Null
} catch {
    Write-Host "ERROR: failed to register scheduled task: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  rag refresh scheduled task installed" -ForegroundColor Green
Write-Host "  name        : $RefreshTaskName"
Write-Host "  python      : $PythonExe"
Write-Host "  args        : $ActionArgs"
Write-Host "  working dir : $RepoRoot"
Write-Host ("  trigger     : daily at {0}" -f $RunAt.ToString('HH:mm'))
Write-Host "  run level   : Limited (not elevated — it does not need to be)"
Write-Host ""
Write-Host "  it re-ingests only what changed. it never deletes:" -ForegroundColor Gray
Write-Host "    --prune is NOT scheduled; run it by hand after --dry-run" -ForegroundColor Gray
Write-Host ""

if ($null -ne $SourceCount -and $SourceCount -eq 0) {
    Write-Host "  WARNING: config.yaml has no sources, so this task will do nothing." -ForegroundColor Yellow
    Write-Host "  Add entries under the 'sources:' key, e.g.:" -ForegroundColor Yellow
    Write-Host "      sources:" -ForegroundColor Gray
    Write-Host "        - type: markdown" -ForegroundColor Gray
    Write-Host "          path: D:/notes" -ForegroundColor Gray
    Write-Host ""
} elseif ($null -ne $SourceCount) {
    Write-Host "  tracking $SourceCount configured source(s) from config.yaml" -ForegroundColor Gray
    Write-Host ""
} elseif ($ProbeError) {
    Write-Host "  WARNING: could not read the 'sources:' list from config.yaml —" -ForegroundColor Yellow
    Write-Host "  the task will hit the same error. Fix it and check with 'rag refresh --dry-run':" -ForegroundColor Yellow
    Write-Host "      $ProbeError" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "  next steps:" -ForegroundColor Cyan
Write-Host "    1. See what it WOULD do, without writing anything:" -ForegroundColor Gray
Write-Host "         rag refresh --dry-run" -ForegroundColor Gray
Write-Host "    2. Run it now instead of waiting for tonight:" -ForegroundColor Gray
Write-Host "         Start-ScheduledTask -TaskName $RefreshTaskName" -ForegroundColor Gray
Write-Host "    3. Check how the last run went (0 = clean):" -ForegroundColor Gray
Write-Host "         Get-ScheduledTaskInfo -TaskName $RefreshTaskName" -ForegroundColor Gray
Write-Host "    4. To remove the schedule later:" -ForegroundColor Gray
Write-Host "         scripts\uninstall-refresh-task.ps1" -ForegroundColor Gray
Write-Host ""
