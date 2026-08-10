"""Unit tests for the FastAPI server + service state + status/url CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# -- service state read/write ----------------------------------------------

def test_state_round_trip(tmp_path: Path):
    """service_state.write_state → read_state round-trips through the
    actual repo helpers (atomic temp+rename write, tolerant read)."""
    import service_state
    state = {"port": 8420, "pid": 1234, "started_at": "2026-08-09T10:00:00Z",
             "url": "http://localhost:8420", "host": "127.0.0.1"}
    state_file = tmp_path / "state.json"
    service_state.write_state(state, path=state_file)
    assert service_state.read_state(path=state_file) == state
    assert list(tmp_path.glob("*.tmp")) == [], "atomic write must not leave temp files"


def test_read_state_tolerates_torn_file(tmp_path: Path):
    """A truncated/garbage state file reads as None, never raises."""
    import service_state
    state_file = tmp_path / "state.json"
    state_file.write_text('{"port": 84', encoding="utf-8")
    assert service_state.read_state(path=state_file) is None


def test_clear_state_respects_ownership(tmp_path: Path):
    """clear_state(owner_pid=...) must refuse to delete another server's file."""
    import service_state
    state_file = tmp_path / "state.json"
    service_state.write_state({"pid": 1234}, path=state_file)
    assert service_state.clear_state(owner_pid=9999, path=state_file) is False
    assert state_file.exists()
    assert service_state.clear_state(owner_pid=1234, path=state_file) is True
    assert not state_file.exists()


def test_status_no_state_file(tmp_path: Path, monkeypatch, capsys):
    """`rag status` with no state file says so and exits 0 (it's a read command)."""
    from cli import _read_service_state
    # Point the SERVICE_STATE_FILE constant at a non-existent path
    monkeypatch.setattr("cli.SERVICE_STATE_FILE", tmp_path / "nope.json")
    s = _read_service_state()
    assert s is None


def test_url_exits_nonzero_when_not_running(tmp_path: Path, monkeypatch, capsys):
    """`rag url` exits 1 when no service is running."""
    from click.testing import CliRunner
    from cli import cli as cli_group
    monkeypatch.setattr("cli.SERVICE_STATE_FILE", tmp_path / "nope.json")
    runner = CliRunner()
    result = runner.invoke(cli_group, ["url"])
    assert result.exit_code == 1


# -- pid_alive check --------------------------------------------------------

def test_pid_alive_handles_zero_and_negative():
    """0 and negative PIDs must be treated as not-alive."""
    from cli import _pid_alive
    assert _pid_alive(0) is False
    assert _pid_alive(-1) is False


def test_pid_alive_for_current_process():
    """The current process is alive; the function should return True."""
    from cli import _pid_alive
    import os
    assert _pid_alive(os.getpid()) is True


def test_pid_alive_for_dead_process():
    """A process that has exited must report not-alive.

    Regression: the previous os.kill(pid, 0) probe returned silently for
    a just-exited PID on Windows, so `rag status` reported a crashed
    server as running.
    """
    import subprocess
    import sys
    from cli import _pid_alive
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert _pid_alive(p.pid) is False


# -- FastAPI app shape -----------------------------------------------------

def test_fastapi_app_exists_with_expected_routes():
    """The serve module's app must expose the core API routes."""
    import serve
    app = serve.app
    paths = {r.path for r in app.routes}
    # UI
    assert "/" in paths
    # API
    assert "/api/health" in paths
    assert "/api/ask" in paths
    assert "/api/eval" in paths
    assert "/api/topics" in paths


def test_ask_request_validation():
    """An empty query is rejected by Pydantic (min_length=1)."""
    from pydantic import ValidationError
    from serve import AskRequest
    with pytest.raises(ValidationError):
        AskRequest(query="")


# -- rag start / rag open (P2) ----------------------------------------------

def test_start_and_open_commands_registered():
    """`rag start` and `rag open` are registered subcommands of the CLI."""
    from click.testing import CliRunner
    from cli import cli as cli_group
    runner = CliRunner()
    res = runner.invoke(cli_group, ["--help"])
    assert "start" in res.output
    assert "open" in res.output


def test_open_exits_nonzero_when_not_running(tmp_path, monkeypatch):
    """`rag open` exits 1 when the GUI is not running."""
    from click.testing import CliRunner
    from cli import cli as cli_group
    monkeypatch.setattr("cli.SERVICE_STATE_FILE", tmp_path / "nope.json")
    runner = CliRunner()
    res = runner.invoke(cli_group, ["open"])
    assert res.exit_code == 1
    assert "not running" in res.output


def test_open_invokes_webbrowser_when_running(tmp_path, monkeypatch):
    """`rag open` calls webbrowser.open with the URL from the state file."""
    import json
    from click.testing import CliRunner
    from cli import cli as cli_group
    state = {"port": 8420, "pid": os.getpid(), "started_at": "2026-01-01T00:00:00Z",
             "url": "http://localhost:8420", "host": "127.0.0.1"}
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr("cli.SERVICE_STATE_FILE", state_file)
    captured: dict = {}
    monkeypatch.setattr("cli.webbrowser.open", lambda url, *a, **kw: captured.setdefault("url", url))
    runner = CliRunner()
    res = runner.invoke(cli_group, ["open"])
    assert res.exit_code == 0, res.output
    assert captured.get("url") == "http://localhost:8420"


def test_start_dispatches_to_powershell_on_windows(monkeypatch):
    """On Windows, `rag start --no-browser` invokes start_all.ps1 with the
    -NoBrowser flag. subprocess.call is mocked; the real script exists in
    this repo, so the dispatch actually fires and we assert the call shape."""
    from click.testing import CliRunner
    from cli import cli as cli_group

    captured: dict = {}

    def fake_call(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return 0

    monkeypatch.setattr("cli.subprocess.call", fake_call)
    monkeypatch.setattr("cli.sys.platform", "win32")
    res = CliRunner().invoke(cli_group, ["start", "--no-browser"])

    assert res.exit_code == 0, res.output
    cmd = captured.get("cmd")
    assert cmd, "subprocess.call was never invoked"
    assert cmd[0] == "powershell.exe"
    assert any(str(a).endswith("start_all.ps1") for a in cmd)
    assert "-NoBrowser" in cmd


def test_start_script_file_exists():
    """The PowerShell orchestrator script must be present on disk."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "scripts" / "start_all.ps1"
    assert p.is_file(), f"missing: {p}"


def test_create_desktop_shortcut_script_exists():
    """The desktop-shortcut script must be present on disk."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "scripts" / "create-desktop-shortcut.ps1"
    assert p.is_file(), f"missing: {p}"


def test_library_endpoints_registered():
    """The four P2 library endpoints must be present on the FastAPI app."""
    import serve
    app = serve.app
    paths = {getattr(r, "path", "") for r in app.routes}
    for p in ("/api/sources", "/api/citations", "/api/eval-runs", "/api/stats"):
        assert p in paths, f"{p} missing; routes: {sorted(paths)}"


def test_rag_bat_defaults_to_start_when_no_args():
    """`rag.bat` with no args should call `python cli.py start` (desktop-shortcut UX)."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "rag.bat"
    content = p.read_text(encoding="utf-8")
    # The no-args branch should invoke `start` so the desktop shortcut
    # brings up the whole stack.
    assert "cli.py\" start" in content
    assert "if \"%*\"==\"\"" in content


def test_rag_ps1_defaults_to_start_when_no_args():
    """`rag.ps1` with no args should also default to `start`."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "rag.ps1"
    content = p.read_text(encoding="utf-8")
    assert "cli.py') start" in content


# -- /api/ingest (P2) -------------------------------------------------------

def test_api_ingest_route_registered():
    """The /api/ingest endpoint must exist (P2 GUI upload)."""
    import serve
    app = serve.app
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/ingest" in paths, f"/api/ingest missing; routes: {paths}"


def test_api_ingest_rejects_empty_upload(serve_state):
    """POST /api/ingest with no files returns 400."""
    from fastapi.testclient import TestClient
    import serve
    # Force the server into the "ready" state without loading the real
    # ABCs (which would download models and take minutes).
    serve_state.ready = True
    client = TestClient(serve.app)
    # No files key at all.
    r = client.post("/api/ingest", files=[])
    # FastAPI returns 422 for missing required parameter; 400 for
    # empty list. Either is acceptable for "no files".
    assert r.status_code in (400, 422)


def test_api_ingest_rejects_when_not_ready(serve_state):
    """POST /api/ingest returns 503 when the server hasn't finished startup."""
    from fastapi.testclient import TestClient
    import serve
    # Make sure ready is False.
    serve_state.ready = False
    client = TestClient(serve.app)
    # Use a dummy in-memory file; the ready check happens first.
    r = client.post(
        "/api/ingest",
        files=[("files", ("test.md", b"# hello\nbody\n"))],
    )
    assert r.status_code == 503


class _NopEmbedder:
    def dim(self): return 4
    def embed(self, texts): return [[0.0] * 4 for _ in texts]
    def embed_query(self, t): return [0.0] * 4
    def embed_documents(self, texts): return [[0.0] * 4 for _ in texts]


class _NopStore:
    collection = "test"
    def ensure_collection(self, recreate=False, expected_dense_dim=None): pass
    def upsert_chunks(self, chunks, vectors): return len(chunks)


def test_api_ingest_handles_unsupported_extensions(serve_state):
    """Unsupported file types are reported in `skipped`, not in `files`."""
    from fastapi.testclient import TestClient
    import serve
    serve_state.ready = True
    serve_state.embedder = _NopEmbedder()
    serve_state.store = _NopStore()
    client = TestClient(serve.app)
    r = client.post(
        "/api/ingest",
        files=[("files", ("foo.txt", b"hi"))],  # not supported
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ingested"] == 0
    assert "foo.txt" in j["skipped"]
    assert j["chunks"] == 0


def test_api_ingest_uploads_markdown(serve_state):
    """A markdown upload should be accepted and 'ingested' via the nop store."""
    from fastapi.testclient import TestClient
    import serve
    serve_state.ready = True
    serve_state.embedder = _NopEmbedder()
    serve_state.store = _NopStore()
    client = TestClient(serve.app)
    r = client.post(
        "/api/ingest",
        files=[("files", ("notes.md", b"# Hello\nbody of the note\n"))],
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ingested"] == 1
    assert j["chunks"] >= 1
    assert "notes.md" in j["files"]
    assert "md" in j["types"]


def test_static_dir_is_mounted():
    """The static dir must be served at /static/* (UI is there)."""
    import serve
    app = serve.app
    # The mount path shows up in app.routes under StaticFiles
    has_static = any(getattr(r, "path", "").startswith("/static") for r in app.routes)
    assert has_static, f"static mount missing; routes: {[(r.path, type(r).__name__) for r in app.routes]}"


def test_index_html_exists():
    """The UI HTML must exist on disk; otherwise / would 404."""
    from pathlib import Path
    html = Path(__file__).resolve().parent.parent / "static" / "index.html"
    assert html.is_file(), f"UI HTML missing: {html}"
    assert "<html" in html.read_text(encoding="utf-8").lower()
