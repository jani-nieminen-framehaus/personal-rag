"""Pytest config: sys.path shim + shared fixtures.

The sys.path insert stays even though pyproject.toml sets pythonpath —
it keeps direct `python tests/test_x.py` runs working outside pytest.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def serve_state():
    """Hand out serve.S for mutation and restore ready/embedder/store after
    the test — replaces the hand-rolled try/finally blocks that made the
    server tests order-dependent."""
    import serve

    saved = (serve.S.ready, serve.S.embedder, serve.S.store)
    yield serve.S
    serve.S.ready, serve.S.embedder, serve.S.store = saved


@pytest.fixture
def rmtree_retry():
    """Windows can hold a handle on files (SQLite DBs especially) briefly
    after close, making a bare rmtree flaky. Retry, then best-effort."""
    def _rm(path, attempts: int = 5, delay: float = 0.1) -> None:
        for _ in range(attempts):
            try:
                shutil.rmtree(path)
                return
            except OSError:
                time.sleep(delay)
        shutil.rmtree(path, ignore_errors=True)
    return _rm
