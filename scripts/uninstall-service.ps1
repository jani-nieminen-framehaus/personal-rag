# =============================================================================
# uninstall-service.ps1 — remove the rag GUI scheduled task.
# Also stops the running service if it's up.
# =============================================================================

$ErrorActionPreference = 'Stop'
$TaskName = 'rag-gui'

# Stop the service if it's running.
$proc = Get-Process -Name python -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'cli serve' } |
    Select-Object -First 1
if ($proc) {
    Write-Host "stopping running rag service (pid $($proc.Id))..."
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}

# Remove the scheduled task.
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "  scheduled task '$TaskName' removed" -ForegroundColor Green
} else {
    Write-Host "  no scheduled task named '$TaskName' found — nothing to remove"
}

# Clean up the state file.
$stateFile = Join-Path $env:USERPROFILE '.rag\state.json'
if (Test-Path $stateFile) {
    Remove-Item $stateFile -Force -ErrorAction SilentlyContinue
    Write-Host "  state file $stateFile removed"
}

Write-Host ""
Write-Host "  rag GUI is no longer auto-launched. To bring it back:" -ForegroundColor Cyan
Write-Host "       scripts\install-service.ps1" -ForegroundColor Gray
