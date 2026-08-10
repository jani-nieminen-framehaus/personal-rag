# =============================================================================
# uninstall-service.ps1 — remove the rag GUI scheduled task.
# Also stops the running service if it's up.
# =============================================================================

$ErrorActionPreference = 'Stop'

# Shared constants + helpers ($TaskName, $StateFile, Read-StateJson).
# Python mirror: service_state.py at the repo root.
. (Join-Path $PSScriptRoot '_config.ps1')

# Stop the service if it's running. The PID from the state file is
# authoritative and works on every PowerShell version; the command-line
# match is a fallback that needs PS 7+ ($_.CommandLine is $null on 5.1).
$stopped = $false
$statePid = (Read-StateJson).pid
if ($statePid) {
    $proc = Get-Process -Id $statePid -ErrorAction SilentlyContinue
    if ($proc -and $proc.ProcessName -match 'python') {
        Write-Host "stopping running rag service (pid $statePid)..."
        Stop-Process -Id $statePid -Force -ErrorAction SilentlyContinue
        $stopped = $true
    }
}
if (-not $stopped) {
    # Matches both launch forms: "python -m cli serve" and "python cli.py serve".
    $proc = Get-Process -Name python -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'cli(\.py)?\s+serve' } |
        Select-Object -First 1
    if ($proc) {
        Write-Host "stopping running rag service (pid $($proc.Id))..."
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $stopped = $true
    }
}

# Remove the scheduled task.
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "  scheduled task '$TaskName' removed" -ForegroundColor Green
} else {
    Write-Host "  no scheduled task named '$TaskName' found — nothing to remove"
}

# Clean up the state file — but never yank it out from under a server that
# is still alive (that makes `rag status` report "not running" while the
# server keeps serving).
if (Test-Path $stateFile) {
    $stillAlive = $false
    if ($statePid -and -not $stopped) {
        $stillAlive = [bool](Get-Process -Id $statePid -ErrorAction SilentlyContinue)
    }
    if ($stillAlive) {
        Write-Host "  state file kept — server (pid $statePid) is still running; stop it first" -ForegroundColor Yellow
    } else {
        Remove-Item $stateFile -Force -ErrorAction SilentlyContinue
        Write-Host "  state file $stateFile removed"
    }
}

Write-Host ""
Write-Host "  rag GUI is no longer auto-launched. To bring it back:" -ForegroundColor Cyan
Write-Host "       scripts\install-service.ps1" -ForegroundColor Gray
