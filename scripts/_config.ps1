# =============================================================================
# _config.ps1 — shared constants + helpers for the rag scripts.
#
# Dot-source from any script in this directory:
#     . (Join-Path $PSScriptRoot '_config.ps1')
#
# Python mirror of these constants: service_state.py at the repo root.
# Keep the two in sync.
# =============================================================================

$Script:TaskName  = 'rag-gui'
$Script:RagPort   = if ($env:RAG_PORT) { [int]$env:RAG_PORT } else { 8420 }
$Script:StateFile = Join-Path $env:USERPROFILE '.rag\state.json'

function Read-StateJson {
    if (-not (Test-Path $Script:StateFile)) { return $null }
    try {
        return Get-Content $Script:StateFile -Raw | ConvertFrom-Json
    } catch { return $null }
}

function Test-PidAlive {
    param([int]$ProcessId)
    if ($ProcessId -le 0) { return $false }
    $sig = @'
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetExitCodeProcess(IntPtr h, out uint code);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool CloseHandle(IntPtr h);
}
'@
    if (-not ('Win32' -as [type])) {
        Add-Type -TypeDefinition $sig
    }
    $h = [Win32]::OpenProcess(0x1000, $false, [uint32]$ProcessId)
    if ($h -eq [IntPtr]::Zero) { return $false }
    try {
        $code = 0
        [void][Win32]::GetExitCodeProcess($h, [ref]$code)
        return $code -eq 259   # STILL_ACTIVE
    } finally {
        [void][Win32]::CloseHandle($h)
    }
}
