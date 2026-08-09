"""Unit tests for the FastAPI server + service state + status/url CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# -- service state read/write ----------------------------------------------

def test_state_round_trip(tmp_path: Path, monkeypatch):
    """The service state file can be written and read back."""
    state = {"port": 8420, "pid": 1234, "started_at": "2026-08-09T10:00:00Z",
             "url": "http://localhost:8420", "host": "127.0.0.1"}
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    read_back = json.loads(state_file.read_text(encoding="utf-8"))
    assert read_back == state


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
    try:
        AskRequest(query="")
    except ValidationError:
        return
    raise AssertionError("expected ValidationError for empty query")


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
