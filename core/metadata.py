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
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Schema & Migrations
# -----------------------------------------------------------------------------

_LATEST_SCHEMA_VERSION = 2

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

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    created_at  TIMESTAMP NOT NULL,
    updated_at  TIMESTAMP NOT NULL,
    title       TEXT,
    turn_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);

CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    asked_at    TIMESTAMP NOT NULL,
    query       TEXT NOT NULL,
    answer      TEXT NOT NULL,
    citations   TEXT
);
CREATE INDEX IF NOT EXISTS idx_turns_session  ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_asked_at ON turns(asked_at);
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
        self._migrate()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # Every additive migration to date, as (table, column, type) triples.
    # v0 -> v1: eval_runs gains params (JSON) + faithfulness_method.
    # v1 -> v2: sources gains file_hash (raw-file hash for change detection).
    # _SCHEMA stays at the v0 shape so fresh and existing DBs take the exact
    # same path through here: a fresh DB applies all three triples in one
    # open (0 -> 2); a DB already at v1 applies only the third (1 -> 2).
    _MIGRATIONS: tuple[tuple[str, str, str], ...] = (
        ("eval_runs", "params", "TEXT"),
        ("eval_runs", "faithfulness_method", "TEXT"),
        ("sources", "file_hash", "TEXT"),
    )

    def _migrate(self) -> None:
        """Additive migrations, versioned via PRAGMA user_version.

        One declarative list of (table, column, type) triples covers every
        migration to date, applied idempotently in a single loop, with
        user_version set to _LATEST_SCHEMA_VERSION afterwards. That keeps
        exactly one code path for a fresh database (0 -> 2, adding all
        three columns) and an existing v1 database (1 -> 2, adding only
        the third) — the property the original v1 migration was careful
        to preserve.

        The migration is deliberately TOLERANT rather than transactional.
        The connection autocommits (`isolation_level=None`) and `self._lock`
        is a threading.Lock, which does nothing across processes — and this
        deployment runs the FastAPI server as a scheduled task alongside CLI
        commands against the same metadata.sqlite3. So two concurrent opens
        of a stale DB, or a crash between the ALTERs and the PRAGMA, can leave
        a column present with user_version still behind. Checking table_info
        before each ADD COLUMN makes that state self-repairing on the next
        open instead of a permanent `duplicate column name: ...` out of
        __init__ that needs manual sqlite surgery. (A BEGIN IMMEDIATE
        transaction would close the race but could NOT repair a database
        that is already half-migrated.)
        """
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version >= _LATEST_SCHEMA_VERSION:
                return
            existing_by_table: dict[str, set[str]] = {}
            for table, column, decl in self._MIGRATIONS:
                if table not in existing_by_table:
                    existing_by_table[table] = {
                        row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")
                    }
                if column in existing_by_table[table]:
                    continue
                try:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                except sqlite3.OperationalError as e:
                    # Another process won the race between our table_info
                    # read and this ALTER. The column exists either way,
                    # which is all we need — anything else re-raises.
                    if "duplicate column" not in str(e).lower():
                        raise
                    log.debug("metadata: %s.%s already added concurrently", table, column)
                else:
                    existing_by_table[table].add(column)
            # Set the version regardless of which ALTERs we actually ran, so
            # a half-migrated database converges to _LATEST_SCHEMA_VERSION
            # on this open.
            self._conn.execute(f"PRAGMA user_version = {_LATEST_SCHEMA_VERSION}")

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
        file_hash: str | None = None,
    ) -> None:
        """Upsert a source row. Idempotent - re-ingesting the same file just
        refreshes chunk_count, content_hash, file_hash, and ingested_at.

        `file_hash` is the exception: passing None - or "", which is what
        `hash_file` returns for a file it could not read - leaves whatever is
        already stored alone. It is the input `rag refresh` uses to decide
        whether a file needs re-ingesting at all, so a caller that simply
        doesn't know it must not be able to destroy it."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sources (source_path, doc_type, topic, ingested_at, chunk_count, content_hash, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    doc_type     = excluded.doc_type,
                    topic        = excluded.topic,
                    ingested_at  = excluded.ingested_at,
                    chunk_count  = excluded.chunk_count,
                    content_hash = excluded.content_hash,
                    -- Not `excluded.file_hash`: omitting the argument must
                    -- PRESERVE a stored hash, not erase it. Otherwise any
                    -- caller that doesn't hash (a plain `rag ingest`) nulls
                    -- the hash `rag refresh` relies on, and the next refresh
                    -- sees the file as changed again.
                    -- NULLIF as well as COALESCE, because hash_file returns
                    -- "" for a file it cannot read and "" is not NULL: a
                    -- transient read failure would otherwise overwrite a good
                    -- hash with empty, and that file would then re-ingest on
                    -- every single run.
                    file_hash    = COALESCE(NULLIF(excluded.file_hash, ''), sources.file_hash)
                """,
                (source_path, doc_type, topic, _now_iso(), int(chunk_count), content_hash, file_hash),
            )

    # SQLite's parameter limit is 999 on builds older than 3.32. A refresh of
    # a large source can exceed that, so the IN clause is batched.
    _CLEAR_BATCH = 500

    def clear_file_hashes(self, source_paths: Iterable[str]) -> int:
        """Set file_hash back to NULL for these rows, so the next refresh
        treats them as changed. Returns the number of rows updated.

        Used when an ingest fails part-way. `pipeline.ingest` records every
        source it managed to write before aborting, with the hash of the WHOLE
        file - so a file big enough to span more than one embed batch ends up
        with a row claiming the new hash while only the first batch is
        indexed. Left alone, every later refresh calls that file unchanged and
        never revisits it, and the un-indexed tail is silently lost.

        `record_source(file_hash="")` cannot do this: the upsert deliberately
        PRESERVES a stored hash when the caller passes None or "", so that a
        transient read failure cannot erase a good one. Clearing has to be an
        explicit, separate act.
        """
        paths = [str(p) for p in source_paths]
        if not paths:
            return 0
        total = 0
        with self._lock:
            for i in range(0, len(paths), self._CLEAR_BATCH):
                batch = paths[i:i + self._CLEAR_BATCH]
                cur = self._conn.execute(
                    "UPDATE sources SET file_hash = NULL WHERE source_path IN "
                    f"({','.join('?' * len(batch))})",
                    batch,
                )
                total += cur.rowcount
        return total

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
        params: dict | None = None,
        faithfulness_method: str | None = None,
    ) -> None:
        """Append one row to `eval_runs`. The eval harness calls this once per run."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO eval_runs
                    (ran_at, n_questions, recall_at_5, mrr, recall_at_dense,
                     faithfulness_proxy, params, faithfulness_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_iso(),
                    int(n_questions),
                    float(recall_at_5),
                    float(mrr),
                    None if recall_at_dense is None else float(recall_at_dense),
                    None if faithfulness_proxy is None else float(faithfulness_proxy),
                    None if params is None else json.dumps(params, ensure_ascii=False),
                    faithfulness_method,
                ),
            )

    # -- reads ----------------------------------------------------------------

    def get_sources(self, limit: int = 50) -> list[dict[str, Any]]:
        """All sources, most recent first. Capped by `limit`."""
        cur = self._conn.execute(
            "SELECT source_path, doc_type, topic, ingested_at, chunk_count, content_hash, file_hash "
            "FROM sources ORDER BY ingested_at DESC LIMIT ?",
            (int(limit),),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def source_hashes(self) -> dict[str, str | None]:
        """Every recorded source_path mapped to its file_hash. Refresh's
        change-detection input — one query, no limit."""
        cur = self._conn.execute("SELECT source_path, file_hash FROM sources")
        return {row[0]: row[1] for row in cur.fetchall()}

    def delete_all_sources(self) -> int:
        """Empty the `sources` table. Returns the number of rows removed.

        Exists for exactly one caller: an ingest that DROPPED the collection.
        Every row is a claim that a file's chunks are in the index, so once the
        collection is gone every row is false — and the falsehood is not inert.
        `rag refresh` compares a file against its recorded hash, matches, and
        reports the index up to date, which makes the one command that could
        restore the corpus the one command that refuses to.

        Deleting rather than clearing the hashes: the chunks are gone, not
        stale, and a row that survives would also keep the file listed by
        `rag sources` as indexed when it is not.
        """
        with self._lock:
            cur = self._conn.execute("DELETE FROM sources")
            return int(cur.rowcount or 0)

    def delete_source(self, source_path: str) -> bool:
        """Remove one source row. Returns True if a row was deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sources WHERE source_path = ?", (source_path,)
            )
            return cur.rowcount > 0

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
            "SELECT id, ran_at, n_questions, recall_at_5, mrr, recall_at_dense, faithfulness_proxy, params, faithfulness_method "
            "FROM eval_runs ORDER BY ran_at DESC, id DESC LIMIT ?",
            (int(limit),),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        for r in rows:
            r["params"] = json.loads(r["params"]) if r.get("params") else None
        return rows

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

    # -- conversation sessions ------------------------------------------------

    def create_session(self, session_id: str, title: str | None = None) -> None:
        """Create a new session, or no-op if it already exists (idempotent)."""
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (id, created_at, updated_at, title, turn_count)
                VALUES (?, ?, ?, ?, 0)
                ON CONFLICT(id) DO NOTHING
                """,
                (session_id, now, now, title or ""),
            )

    def record_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        citations: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        """Append one turn to a session. Creates the session if it doesn't exist.

        `citations` is stored as JSON so it can be replayed in the UI.
        """
        now = _now_iso()
        citations_json = json.dumps(list(citations)) if citations else "[]"
        with self._lock:
            # Ensure the session exists.
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (id, created_at, updated_at, title, turn_count) "
                "VALUES (?, ?, ?, '', 0)",
                (session_id, now, now),
            )
            self._conn.execute(
                """
                INSERT INTO turns (session_id, asked_at, query, answer, citations)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, now, query, answer, citations_json),
            )
            self._conn.execute(
                """
                UPDATE sessions SET updated_at = ?, turn_count = turn_count + 1
                WHERE id = ?
                """,
                (now, session_id),
            )

    def get_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent sessions, most recent first."""
        cur = self._conn.execute(
            "SELECT id, created_at, updated_at, title, turn_count "
            "FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (int(limit),),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_turns(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """All turns for one session, oldest first (chronological order)."""
        cur = self._conn.execute(
            "SELECT id, session_id, asked_at, query, answer, citations "
            "FROM turns WHERE session_id = ? ORDER BY asked_at ASC LIMIT ?",
            (session_id, int(limit)),
        )
        cols = [d[0] for d in cur.description]
        rows = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            # Decode citations JSON.
            if d.get("citations"):
                try:
                    d["citations"] = json.loads(d["citations"])
                except Exception:
                    d["citations"] = []
            rows.append(d)
        return rows


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


def hash_file(path: str | Path) -> str:
    """Hash a file's raw bytes. Returns "" when the file cannot be read.

    Raw bytes, not chunked text: the point is to decide whether to parse a
    file at all, so this must not require parsing it. Read in blocks so a
    500 MB PDF does not land in memory.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()[:16]
