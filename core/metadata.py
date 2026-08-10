"""SQLite metadata layer for the RAG.

Tracks three things that Qdrant doesn't surface cleanly:

- `sources`  : one row per ingested FILE (chunk_count, content_hash, when
              it was ingested). Lets you answer "what files have I
              indexed?" and "did this file change since last ingest?".
- `citations`: append-only log of every chunk the model returned as
              evidence for an answer. Lets you answer "what do I cite
              most often?" and "when did I last look at this file?".
- `eval_runs`: every eval invocation, with the headline metrics. Lets
              you track retrieval quality over time without re-running.

The store is a thin wrapper over `sqlite3` (stdlib). One file,
default at `./metadata.sqlite3` (override via `metadata.path` in
config.yaml). Idempotent schema creation on open. Threadsafe via
`check_same_thread=False` + a per-call lock so the FastAPI handler
threads don't trip each other.

Wired into:
- `core.pipeline.ingest()`  - record_source for each unique file.
- `core.pipeline.ask()`     - record_citation per returned citation.
- `eval/run_ragas.run()`    - record_eval_run after the harness finishes.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_path   TEXT PRIMARY KEY,
    doc_type      TEXT,
    topic         TEXT,
    ingested_at   TIMESTAMP NOT NULL,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    content_hash  TEXT
);

CREATE TABLE IF NOT EXISTS citations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at     TIMESTAMP NOT NULL,
    query        TEXT NOT NULL,
    chunk_id     TEXT NOT NULL,
    source_path  TEXT,
    rank         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_citations_asked_at  ON citations(asked_at);
CREATE INDEX IF NOT EXISTS idx_citations_source    ON citations(source_path);
CREATE INDEX IF NOT EXISTS idx_citations_chunk     ON citations(chunk_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at              TIMESTAMP NOT NULL,
    n_questions         INTEGER NOT NULL,
    recall_at_5         REAL,
    mrr                 REAL,
    recall_at_dense     REAL,
    faithfulness_proxy  REAL
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_ran_at ON eval_runs(ran_at);
"""


def _now_iso() -> str:
    """UTC timestamp in ISO-8601, second precision. Matches serve.py's state file."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -----------------------------------------------------------------------------
# Store
# -----------------------------------------------------------------------------

class MetadataStore:
    """Thin SQLite wrapper. Open with a path, call the record / read methods, close.

    The connection uses `check_same_thread=False` so the FastAPI
    handlers (which run in a threadpool) can share it. We serialize
    writes through a single lock - sqlite3 doesn't allow concurrent
    writes from the same connection anyway. Reads are lock-free.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None
        )
        # Pragmas: WAL for concurrent readers + a small fsync window.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writes ---------------------------------------------------------------

    def record_source(
        self,
        source_path: str,
        doc_type: str,
        topic: str,
        chunk_count: int,
        content_hash: str = "",
    ) -> None:
        """Upsert a source row. Idempotent - re-ingesting the same file just
        refreshes chunk_count, content_hash, and ingested_at."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sources (source_path, doc_type, topic, ingested_at, chunk_count, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    doc_type     = excluded.doc_type,
                    topic        = excluded.topic,
                    ingested_at  = excluded.ingested_at,
                    chunk_count  = excluded.chunk_count,
                    content_hash = excluded.content_hash
                """,
                (source_path, doc_type, topic, _now_iso(), int(chunk_count), content_hash),
            )

    def record_citations(
        self,
        query: str,
        citations: Iterable[dict[str, Any]],
    ) -> int:
        """Append a citation log row per citation in the iterable.

        `citations` is the list-of-dicts that `pipeline.ask()` returns
        (each dict has at least `chunk_id`, `source_path`, `n`). Returns
        the number of rows written.
        """
        rows = [
            (_now_iso(), query, c["chunk_id"], c.get("source_path", ""), int(c["n"]))
            for c in citations
        ]
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO citations (asked_at, query, chunk_id, source_path, rank)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def record_eval_run(
        self,
        n_questions: int,
        recall_at_5: float,
        mrr: float,
        recall_at_dense: float | None = None,
        faithfulness_proxy: float | None = None,
    ) -> None:
        """Append one row to `eval_runs`. The eval harness calls this once per run."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO eval_runs
                    (ran_at, n_questions, recall_at_5, mrr, recall_at_dense, faithfulness_proxy)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_iso(),
                    int(n_questions),
                    float(recall_at_5),
                    float(mrr),
                    None if recall_at_dense is None else float(recall_at_dense),
                    None if faithfulness_proxy is None else float(faithfulness_proxy),
                ),
            )

    # -- reads ----------------------------------------------------------------

    def get_sources(self, limit: int = 50) -> list[dict[str, Any]]:
        """All sources, most recent first. Capped by `limit`."""
        cur = self._conn.execute(
            "SELECT source_path, doc_type, topic, ingested_at, chunk_count, content_hash "
            "FROM sources ORDER BY ingested_at DESC LIMIT ?",
            (int(limit),),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_citations(
        self, limit: int = 50, source_path: str | None = None
    ) -> list[dict[str, Any]]:
        """Recent citations, optionally filtered to one source_path. Most recent first."""
        if source_path:
            cur = self._conn.execute(
                "SELECT id, asked_at, query, chunk_id, source_path, rank "
                "FROM citations WHERE source_path = ? ORDER BY asked_at DESC, id DESC LIMIT ?",
                (source_path, int(limit)),
            )
        else:
            cur = self._conn.execute(
                "SELECT id, asked_at, query, chunk_id, source_path, rank "
                "FROM citations ORDER BY asked_at DESC, id DESC LIMIT ?",
                (int(limit),),
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_eval_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent eval runs, most recent first."""
        cur = self._conn.execute(
            "SELECT id, ran_at, n_questions, recall_at_5, mrr, recall_at_dense, faithfulness_proxy "
            "FROM eval_runs ORDER BY ran_at DESC, id DESC LIMIT ?",
            (int(limit),),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_stats(self) -> dict[str, Any]:
        """Aggregate counts and the top-cited source. One row, fast."""
        out: dict[str, Any] = {}
        for key, table in (
            ("total_sources", "sources"),
            ("total_citations", "citations"),
            ("total_eval_runs", "eval_runs"),
        ):
            cur = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
            out[key] = int(cur.fetchone()[0])
        # Top-cited source_path (by citation row count, not unique chunk).
        cur = self._conn.execute(
            "SELECT source_path, COUNT(*) AS n FROM citations "
            "WHERE source_path IS NOT NULL AND source_path != '' "
            "GROUP BY source_path ORDER BY n DESC, source_path ASC LIMIT 1"
        )
        row = cur.fetchone()
        out["top_cited_source"] = row[0] if row else None
        out["top_cited_count"] = int(row[1]) if row else 0
        return out


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def hash_text(text: str) -> str:
    """Short content hash for source rows. First 16 hex chars of SHA-256.

    Truncating to 16 chars (64 bits) keeps the column compact and is
    still collision-resistant at the scale of a personal knowledge base.
    """
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
