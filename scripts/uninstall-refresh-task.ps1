# =============================================================================
# uninstall-refresh-task.ps1 — remove the nightly `rag refresh` scheduled task.
#
# Removes the SCHEDULE only. Nothing is deleted from the index or the metadata
# DB here, and stopping the task just means the index stops keeping itself
# current. (What a refresh itself removes is one changed file's old chunks,
# replaced by the re-ingested ones; --prune, which removes entries for files
# gone from disk, was never scheduled.) Bring it back with
# scripts\install-refresh-task.ps1.
# =============================================================================

$ErrorActionPreference = 'Stop'

# Shared constants ($RefreshTaskName). Python mirror: service_state.py.
. (Join-Path $PSScriptRoot '_config.ps1')

$existing = Get-ScheduledTask -TaskName $RefreshTaskName -ErrorAction SilentlyContinue
if (-not $existing) {
    Write-Host "  no scheduled task named '$RefreshTaskName' found — nothing to remove"
    exit 0
}

# Unregistering terminates a run in progress. Stop it first so it is a request
# rather than a surprise, and say so: an ingest killed mid-batch can leave a
# file half-indexed, which the next `rag refresh` picks up and redoes.
if ($existing.State -eq 'Running') {
    Write-Host "  the task is running right now — stopping it first" -ForegroundColor Yellow
    Stop-ScheduledTask -TaskName $RefreshTaskName -ErrorAction SilentlyContinue
    Write-Host "  if it was mid-ingest, run 'rag refresh' once by hand to finish the job" -ForegroundColor Gray
}

Unregister-ScheduledTask -TaskName $RefreshTaskName -Confirm:$false
Write-Host "  scheduled task '$RefreshTaskName' removed" -ForegroundColor Green
Write-Host ""
Write-Host "  its log is left where it is: $RefreshLogFile" -ForegroundColor Gray
Write-Host ""
Write-Host "  the index is untouched — it just stops updating itself. To bring it back:" -ForegroundColor Cyan
Write-Host "       scripts\install-refresh-task.ps1" -ForegroundColor Gray
Write-Host "  or refresh by hand whenever you like:" -ForegroundColor Cyan
Write-Host "       rag refresh" -ForegroundColor Gray
