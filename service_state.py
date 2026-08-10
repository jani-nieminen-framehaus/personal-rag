"""Single source of truth for the GUI service state file.

`~/.rag/state.json` is written by serve.py on startup, read by `rag status`
/ `rag url` / `rag open` and the tray icon, and cleared on clean shutdown.
The PID in the file guards against clearing state that belongs to another
server instance.

The PowerShell mirror of these constants and helpers lives in
`scripts/_config.ps1` — keep the two in sync.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

STATE_DIR = Path.home() / ".rag"
STATE_FILE = STATE_DIR / "state.json"
DEFAULT_PORT = 8420


def read_state(path: Path | None = None) -> dict | None:
    """Parsed state file, or None when absent or invalid."""
    p = path or STATE_FILE
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_state(payload: dict, path: Path | None = None) -> None:
    """Atomic write (temp + rename) so a crash can't leave a torn file."""
    p = path or STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)


def clear_state(owner_pid: int | None = None, path: Path | None = None) -> bool:
    """Delete the state file. Returns True if it was deleted.

    With `owner_pid`, only delete when the file's pid matches — never yank
    state out from under a different (still running) server instance."""
    p = path or STATE_FILE
    if owner_pid is not None:
        data = read_state(p)
        if not data or data.get("pid") != owner_pid:
            return False
    try:
        p.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def pid_alive(pid: int) -> bool:
    """Best-effort check: is the given PID still running on this host?

    On Windows, `os.kill(pid, 0)` raises ValueError (the signal-0 trick
    is a Unix idiom). We fall back to the Win32 OpenProcess API.
    On Unix, signal 0 is the standard check.
    """
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        # Win32 OpenProcess. Returns 0 (NULL handle) if the process
        # doesn't exist or we don't have access. PROCESS_QUERY_LIMITED_
        # INFORMATION is enough to check existence without elevated rights.
        try:
            import ctypes
            from ctypes import wintypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid))
            if handle == 0:
                return False
            try:
                # OpenProcess alone succeeds for a just-exited process whose
                # kernel object hasn't been reaped — ask for the exit code
                # to distinguish "running" from "zombie".
                code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    # Unix: signal 0 is the standard "is the process alive" check.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, we just don't own it
    except OSError:
        return False
