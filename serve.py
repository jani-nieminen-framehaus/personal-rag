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
import shutil
import socket
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.pipeline import (
    ask as ask_pipeline,
    ingest as ingest_pipeline,
    load_config,
    make_embedder,
    make_generator,
    make_metadata,
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
    metadata = None
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
        # Atomic write (temp + rename): a crash mid-write must not leave a
        # truncated file — shutdown would silently skip cleanup and
        # `rag status` would misreport until the file is deleted by hand.
        tmp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(STATE_FILE)
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
    S.metadata = make_metadata(S.config)
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
    hybrid: bool = Field(False, description="Use hybrid dense+sparse search.")
    session_id: str | None = Field(
        None, max_length=64,
        description="Session ID for conversation memory. If provided, this turn is stored.",
    )


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    dense_hits: list[dict]


@app.post("/api/ask", response_model=AskResponse)
def api_ask(req: AskRequest):
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
        metadata=S.metadata,
        hybrid=req.hybrid,
    )
    response = AskResponse(
        answer=result.answer,
        citations=result.citations,
        dense_hits=result.dense_hits,
    )

    # P2 conversation memory: store the turn in the DB when session_id is given.
    if req.session_id and S.metadata is not None:
        try:
            S.metadata.record_turn(
                session_id=req.session_id,
                query=req.query,
                answer=result.answer,
                citations=result.citations,
            )
        except Exception as e:
            log.warning("session: record_turn failed: %s", e)

    return response


# -- /api/eval ---------------------------------------------------------------

@app.get("/api/eval")
def api_eval():
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
        metadata=S.metadata,
    )
    return metrics


# -- /api/ingest ------------------------------------------------------------
#
# P2 GUI ingest: file upload from the browser, no terminal required.
# Accepts one or more uploaded files (multipart/form-data), saves them
# to a temp dir, and runs the appropriate ingester on the saved files.
#
# Supported types (detected by extension):
#   .pdf       -> PdfDirIngester (per-page chunks)
#   .md / .markdown / .py -> MarkdownDirIngester (heading/AST chunks)
#
# All chunks land in the same Qdrant collection, so PDFs and notes
# coexist. Re-ingesting the same file is a no-op (deterministic
# chunk_ids).

@app.post("/api/ingest")
async def api_ingest(files: list[UploadFile] = File(...)):
    """Upload one or more files and ingest them into the index.

    Returns {ingested: N, chunks: M, files: [name, ...], skipped: [...]}.
    Skipped files are bad-extension / unreadable / not found.
    """
    if not S.ready:
        raise HTTPException(503, "server not ready")
    if not files:
        raise HTTPException(400, "no files uploaded")

    # Save the uploads to a temp dir, partitioned by type so the
    # right ingester can pick each one up. Cleanup in finally.
    work = Path(tempfile.mkdtemp(prefix="rag-ingest-"))
    try:
        saved: list[Path] = []
        skipped: list[str] = []
        for f in files:
            name = Path(f.filename or "").name
            if not name:
                skipped.append("<empty>")
                continue
            ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
            if ext not in {"pdf", "md", "markdown", "py", "epub"}:
                skipped.append(name)
                continue
            # Bucket by extension so each ingester sees a clean dir.
            bucket = work / ext
            bucket.mkdir(exist_ok=True)
            dst = bucket / name
            try:
                content = await f.read()
                dst.write_bytes(content)
                saved.append(dst)
            except Exception as e:
                log.warning("ingest: failed to save %s: %s", name, e)
                skipped.append(name)

        # Run the right ingester on each bucket.
        chunking = (S.config or {}).get("chunking", {})
        ing = (S.config or {}).get("ingest", {})
        total = 0
        ran_for: list[str] = []
        for ext, ing_cls, doc_type in (
            ("pdf", "PdfDirIngester", "pdf"),
            ("md", "MarkdownDirIngester", "markdown"),
            ("markdown", "MarkdownDirIngester", "markdown"),
            ("py", "MarkdownDirIngester", "markdown"),
            ("epub", "EpubDirIngester", "epub"),
        ):
            bucket = work / ext
            if not bucket.is_dir():
                continue
            files_here = list(bucket.iterdir())
            if not files_here:
                continue
            ran_for.append(ext)
            if ing_cls == "PdfDirIngester":
                from ingest.pdf_dir import PdfDirIngester
                inst = PdfDirIngester(
                    path=str(bucket),
                    target_tokens=chunking.get("target_tokens", 768),
                    overlap_pct=chunking.get("overlap_pct", 12),
                    min_chunk_tokens=chunking.get("min_chunk_tokens", 32),
                    default_topic=ing.get("default_topic", "default"),
                    max_chunks_per_doc=chunking.get("max_chunks_per_doc", 2000),
                )
            elif ing_cls == "EpubDirIngester":
                from ingest.epub_dir import EpubDirIngester
                inst = EpubDirIngester(
                    path=str(bucket),
                    target_tokens=chunking.get("target_tokens", 768),
                    overlap_pct=chunking.get("overlap_pct", 12),
                    min_chunk_tokens=chunking.get("min_chunk_tokens", 32),
                    default_topic=ing.get("default_topic", "default"),
                    max_chunks_per_doc=chunking.get("max_chunks_per_doc", 2000),
                )
            else:
                from ingest.markdown_dir import MarkdownDirIngester
                inst = MarkdownDirIngester(
                    root=str(bucket),
                    target_tokens=chunking.get("target_tokens", 768),
                    overlap_pct=chunking.get("overlap_pct", 12),
                    min_chunk_tokens=chunking.get("min_chunk_tokens", 32),
                    default_topic=ing.get("default_topic", "default"),
                    frontmatter_topic_key=ing.get("markdown", {}).get("frontmatter_topic_key", "topic"),
                    max_chunks_per_doc=chunking.get("max_chunks_per_doc", 2000),
                )
            # Don't recreate the collection when called from the GUI -
            # it might already hold the user's previous ingests.
            try:
                n = ingest_pipeline(
                    inst, S.embedder, S.store,
                    recreate=False, batch_size=None, metadata=S.metadata,
                )
                total += n
            except Exception as e:
                log.exception("ingest: %s ingester failed: %s", ext, e)
                raise HTTPException(500, f"{ext} ingest failed: {e}")
        return {
            "ingested": len(saved),
            "chunks": total,
            "files": [p.name for p in saved],
            "skipped": skipped,
            "types": ran_for,
        }
    finally:
        # Best-effort cleanup of the temp work dir.
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass


# -- /api/topics ------------------------------------------------------------

@app.get("/api/topics")
def api_topics():
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


# -- /api/library: sources / citations / eval-runs / stats (P2) ------------
#
# All four read from the metadata store. The GUI Library tab calls them
# in parallel on tab activation. Each one returns [] (or a sensible empty
# payload) when the metadata store is disabled or empty.

def _ensure_metadata():
    """Return the metadata store, or raise 503 if disabled."""
    if not S.ready:
        raise HTTPException(503, "server not ready")
    if S.metadata is None:
        raise HTTPException(503, "metadata store disabled in config.yaml")
    return S.metadata


@app.get("/api/sources")
def api_sources(limit: int = 100):
    md = _ensure_metadata()
    return md.get_sources(limit=limit)


@app.get("/api/citations")
def api_citations(limit: int = 50, source_path: str | None = None):
    md = _ensure_metadata()
    return md.get_citations(limit=limit, source_path=source_path)


@app.get("/api/eval-runs")
def api_eval_runs(limit: int = 20):
    md = _ensure_metadata()
    return md.get_eval_runs(limit=limit)


@app.get("/api/stats")
def api_stats():
    md = _ensure_metadata()
    return md.get_stats()


# -- /api/sessions -----------------------------------------------------------

class SessionCreateRequest(BaseModel):
    title: str | None = None


@app.post("/api/sessions/{session_id}")
def api_create_session(session_id: str, body: SessionCreateRequest | None = None):
    """Create or update a session. Idempotent."""
    md = _ensure_metadata()
    title = body.title if body else None
    md.create_session(session_id, title=title)
    return {"id": session_id, "title": title or ""}


@app.get("/api/sessions")
def api_list_sessions(limit: int = 20):
    """List recent sessions, most recent first."""
    md = _ensure_metadata()
    return md.get_sessions(limit=limit)


@app.get("/api/sessions/{session_id}/turns")
def api_get_turns(session_id: str, limit: int = 50):
    """Get all turns for a session, oldest first (chronological)."""
    md = _ensure_metadata()
    return md.get_turns(session_id, limit=limit)


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
