"""FastAPI server for the RAG GUI.

Run via:
    python cli.py serve             # binds 127.0.0.1:8420 by default
    RAG_PORT=9000 python cli.py serve
    RAG_HOST=0.0.0.0 python cli.py serve  # expose to LAN

Auto-launched on user logon via the Task Scheduler entry created by
scripts/install-service.ps1. The state file at
%USERPROFILE%\\.rag\\state.json records the port, PID, and URL so
`rag status` / `rag url` can find the running server.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.pipeline import (
    ask as ask_pipeline,
    load_config,
    make_embedder,
    make_generator,
    make_reranker,
    make_store,
)


log = logging.getLogger("rag.serve")

# Persistent state: hard-coded so the user never has to remember
# "what port is the server on" — they always go to :8420.
DEFAULT_PORT = 8420
DEFAULT_HOST = "127.0.0.1"

# State file lives in the user's home dir so the CLI commands
# (`rag status`, `rag url`) can find it from any CWD.
STATE_DIR = Path.home() / ".rag"
STATE_FILE = STATE_DIR / "state.json"

STATIC_DIR = Path(__file__).resolve().parent / "static"


# -----------------------------------------------------------------------------
# Singletons (ABCs are expensive to build — load once at startup)
# -----------------------------------------------------------------------------

class _Singletons:
    config: dict | None = None
    embedder = None
    store = None
    reranker = None
    generator = None
    ready: bool = False
    started_at: str | None = None


S = _Singletons()


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "localhost"


def _state_write(host: str, port: int) -> None:
    """Write the service state file so `rag status` / `rag url` can find
    the running server. Best-effort — if it fails (e.g. perms), we
    just log; the server still runs."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        S.started_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "port": port,
            "pid": os.getpid(),
            "started_at": S.started_at,
            "url": f"http://localhost:{port}",
            "lan_url": f"http://{_hostname()}.local:{port}" if host == "0.0.0.0" else None,
            "host": host,
        }
        STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("could not write state file %s: %s", STATE_FILE, e)


def _state_clear() -> None:
    """Remove the state file on clean shutdown. Idempotent."""
    try:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
    except OSError:
        pass


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load config, construct ABCs, write state file.
    Shutdown: clear state file (only if our PID still owns it)."""
    host = os.environ.get("RAG_HOST", DEFAULT_HOST)
    port = int(os.environ.get("RAG_PORT", DEFAULT_PORT))

    log.info("loading config + ABCs (this takes a few seconds the first time)...")
    S.config = load_config()
    S.embedder = make_embedder(S.config)
    S.store = make_store(S.config)
    S.reranker = make_reranker(S.config)
    S.generator = make_generator(S.config)
    S.ready = True
    log.info("ABCs loaded; ready for traffic")

    _state_write(host, port)
    log.info("=" * 60)
    log.info("  rag serve ready")
    log.info("  URL:  http://localhost:%d", port)
    if host == "0.0.0.0":
        log.info("  LAN:  http://%s.local:%d  (or your rig's IP)", _hostname(), port)
    log.info("  state: %s", STATE_FILE)
    log.info("=" * 60)

    yield

    # On shutdown, only clear if our PID still owns the state file
    # (don't yank it out from under a restart).
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if data.get("pid") == os.getpid():
                _state_clear()
        except (OSError, json.JSONDecodeError, ValueError):
            pass


app = FastAPI(title="rag", version="0.1.0", lifespan=lifespan)

# Static UI files (HTML/CSS/JS) — mounted at /static/*
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """Serve the chat UI."""
    idx = STATIC_DIR / "index.html"
    if not idx.is_file():
        raise HTTPException(503, "UI not built (static/index.html missing)")
    return FileResponse(idx)


@app.get("/api/health")
async def health():
    return {
        "ready": S.ready,
        "url": f"http://localhost:{int(os.environ.get('RAG_PORT', DEFAULT_PORT))}",
        "started_at": S.started_at,
    }


# -- /api/ask ----------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    topic: str | None = None
    top_k_dense: int | None = Field(None, ge=1, le=100)
    top_k_final: int | None = Field(None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    dense_hits: list[dict]


@app.post("/api/ask", response_model=AskResponse)
async def api_ask(req: AskRequest):
    if not S.ready:
        raise HTTPException(503, "server not ready")
    pipeline_cfg = (S.config or {}).get("pipeline", {})
    top_k_dense = req.top_k_dense or pipeline_cfg.get("top_k_dense", 20)
    top_k_final = req.top_k_final or pipeline_cfg.get("top_k_final", 5)
    result = ask_pipeline(
        req.query,
        embedder=S.embedder,
        store=S.store,
        reranker=S.reranker,
        generator=S.generator,
        top_k_dense=top_k_dense,
        top_k_final=top_k_final,
        topic=req.topic,
    )
    return AskResponse(
        answer=result.answer,
        citations=result.citations,
        dense_hits=result.dense_hits,
    )


# -- /api/eval ---------------------------------------------------------------

@app.get("/api/eval")
async def api_eval():
    """Run the eval harness. Heavy (loads the embedder), so this is a
    separate endpoint that the user can hit explicitly. The pipeline
    keeps the embedder warm in S so this doesn't reload."""
    if not S.ready:
        raise HTTPException(503, "server not ready")
    # Lazy import so /api/ask stays fast.
    from eval import run_ragas
    golden = Path(__file__).resolve().parent / "eval" / "golden_set.jsonl"
    if not golden.is_file():
        raise HTTPException(404, f"golden set not found: {golden}")
    metrics = run_ragas.run(
        golden_path=golden,
        embedder=S.embedder,
        store=S.store,
        reranker=S.reranker,
        generator=None,  # retrieval-only eval
        top_k_dense=(S.config or {}).get("pipeline", {}).get("top_k_dense", 20),
        top_k_final=(S.config or {}).get("pipeline", {}).get("top_k_final", 5),
    )
    return metrics


# -- /api/topics ------------------------------------------------------------

@app.get("/api/topics")
async def api_topics():
    """Return the distinct topics present in the index. The /api/ask
    filter dropdown uses this."""
    if not S.ready:
        raise HTTPException(503, "server not ready")
    # Lightweight: scroll the collection, dedupe topics from payloads.
    # We don't bother caching; this is fast for a few thousand points.
    seen: set[str] = set()
    offset = None
    while True:
        points, offset = S.store.client.scroll(
            collection_name=S.store.collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            t = (p.payload or {}).get("topic")
            if t:
                seen.add(t)
        if offset is None:
            break
    return sorted(seen)


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def run() -> None:
    """CLI entry: parse env, start uvicorn. Blocking."""
    host = os.environ.get("RAG_HOST", DEFAULT_HOST)
    port = int(os.environ.get("RAG_PORT", DEFAULT_PORT))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(
        "serve:app",
        host=host,
        port=port,
        log_level="info",
        # No reload — the ABCs load once at startup.
        reload=False,
        # Access log on (default) so the user can see who hit what.
    )


if __name__ == "__main__":
    run()
