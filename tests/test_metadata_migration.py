"""Schema migrations: v0 -> v1 adds eval_runs.params + faithfulness_method,
v1 -> v2 adds sources.file_hash. Both are applied on open, idempotently, and
must survive a database that is already populated — which is what the one
running on this machine is."""
from __future__ import annotations

import sqlite3

from core.metadata import MetadataStore


def _columns(db_path, table):
    conn = sqlite3.connect(db_path)
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def test_fresh_db_reaches_the_latest_version_with_the_new_columns(tmp_path):
    db = tmp_path / "meta.sqlite3"
    md = MetadataStore(str(db))
    md.close()
    cols = _columns(str(db), "eval_runs")
    assert "params" in cols and "faithfulness_method" in cols
    conn = sqlite3.connect(str(db))
    # A fresh DB now converges straight to the latest schema version (2),
    # since _migrate applies every migration up to _LATEST_SCHEMA_VERSION
    # in one open — see test_v2_adds_file_hash_and_reaches_version_2 below.
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    conn.close()


def test_reopen_is_idempotent(tmp_path):
    db = tmp_path / "meta.sqlite3"
    MetadataStore(str(db)).close()
    MetadataStore(str(db)).close()  # second open must not ALTER again / raise


def test_half_migrated_db_is_repaired(tmp_path):
    """A crash — or a second process opening the same v0 DB — between the
    ALTERs and the PRAGMA leaves the columns present with user_version
    still 0. `isolation_level=None` autocommits each statement and
    `self._lock` is a threading.Lock (useless across processes), so this
    is reachable on this deployment: the FastAPI scheduled task and a CLI
    command share one metadata.sqlite3. Every later open must still
    succeed instead of raising 'duplicate column name: params' out of
    MetadataStore.__init__ and needing manual sqlite surgery.
    """
    db = tmp_path / "meta.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE eval_runs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            ran_at              TIMESTAMP NOT NULL,
            n_questions         INTEGER NOT NULL,
            recall_at_5         REAL,
            mrr                 REAL,
            recall_at_dense     REAL,
            faithfulness_proxy  REAL
        );
        """
    )
    # Half-applied migration: one of the two columns landed, version never set.
    conn.execute("ALTER TABLE eval_runs ADD COLUMN params TEXT")
    conn.commit()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    conn.close()

    md = MetadataStore(str(db))  # must not raise
    md.record_eval_run(n_questions=1, recall_at_5=1.0, mrr=1.0,
                       params={"k": 1}, faithfulness_method="nli")
    md.close()

    cols = _columns(str(db), "eval_runs")
    assert "params" in cols and "faithfulness_method" in cols
    conn = sqlite3.connect(str(db))
    # Repair converges to the latest schema version (2), not just 1 — the
    # migration loop also picks up sources.file_hash on this same open.
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    conn.close()


def test_record_and_read_params_roundtrip(tmp_path):
    md = MetadataStore(str(tmp_path / "meta.sqlite3"))
    md.record_eval_run(
        n_questions=5, recall_at_5=0.8, mrr=0.7,
        params={"dense_weight": 0.5, "reranker": "bge"},
        faithfulness_method="nli",
    )
    runs = md.get_eval_runs()
    md.close()
    assert runs[0]["params"] == {"dense_weight": 0.5, "reranker": "bge"}
    assert runs[0]["faithfulness_method"] == "nli"


def test_record_without_params_stays_none(tmp_path):
    md = MetadataStore(str(tmp_path / "meta.sqlite3"))
    md.record_eval_run(n_questions=1, recall_at_5=1.0, mrr=1.0)
    runs = md.get_eval_runs()
    md.close()
    assert runs[0]["params"] is None
    assert runs[0]["faithfulness_method"] is None


def test_v2_adds_file_hash_and_reaches_version_2(tmp_path):
    db = tmp_path / "meta.sqlite3"
    MetadataStore(str(db)).close()
    assert "file_hash" in _columns(str(db), "sources")
    conn = sqlite3.connect(str(db))
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    conn.close()


# The `sources` table as it was before v2 — no file_hash. This is the shape
# `ALTER TABLE sources ADD COLUMN file_hash` actually runs against, and the
# shape sitting in the live metadata.sqlite3 on this machine.
_PRE_V2_SOURCES = """
CREATE TABLE sources (
    source_path   TEXT PRIMARY KEY,
    doc_type      TEXT,
    topic         TEXT,
    ingested_at   TIMESTAMP NOT NULL,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    content_hash  TEXT
);
"""

_V0_EVAL_RUNS = """
CREATE TABLE eval_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at              TIMESTAMP NOT NULL,
    n_questions         INTEGER NOT NULL,
    recall_at_5         REAL,
    mrr                 REAL,
    recall_at_dense     REAL,
    faithfulness_proxy  REAL
);
"""


def _populate_sources(conn):
    conn.executemany(
        "INSERT INTO sources (source_path, doc_type, topic, ingested_at, "
        "chunk_count, content_hash) VALUES (?, ?, ?, ?, ?, ?)",
        [("/a.md", "markdown", "notes", "2026-01-01T00:00:00+00:00", 3, "aaa"),
         ("/b.pdf", "pdf", "books", "2026-01-02T00:00:00+00:00", 9, "bbb")],
    )
    conn.commit()


def test_a_populated_v0_database_gains_file_hash_without_losing_rows(tmp_path):
    """The migration that will actually run on this machine.

    The live metadata.sqlite3 here is at user_version 0 with a `sources` table
    that has no file_hash column and two rows already in it. This test builds
    exactly that and opens a MetadataStore over it, so `ALTER TABLE sources ADD
    COLUMN file_hash` runs against a POPULATED pre-v2 table — which is the one
    thing the old version of this test never did: it built a v2 database and
    rewound PRAGMA user_version to 1, asserting a property it did not exercise.
    """
    db = tmp_path / "meta.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.executescript(_PRE_V2_SOURCES + _V0_EVAL_RUNS)
    _populate_sources(conn)
    assert "file_hash" not in [r[1] for r in conn.execute("PRAGMA table_info(sources)")]
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    conn.close()

    md = MetadataStore(str(db))
    rows = {r["source_path"]: r for r in md.get_sources()}
    # Present is not enough — refresh WRITES through this column, and reads it
    # back to decide whether to re-ingest.
    md.record_source("/a.md", "markdown", "notes", 3, "aaa",
                     file_hash="feedfacefeedface")
    hashes = md.source_hashes()
    md.close()

    assert set(rows) == {"/a.md", "/b.pdf"}, "the migration lost rows"
    assert rows["/b.pdf"]["chunk_count"] == 9
    assert rows["/b.pdf"]["content_hash"] == "bbb"
    # NULL, not "": unknown, so refresh re-ingests it once and then tracks it.
    assert rows["/b.pdf"]["file_hash"] is None
    assert hashes["/a.md"] == "feedfacefeedface"
    assert hashes["/b.pdf"] is None

    assert "file_hash" in _columns(str(db), "sources")
    # v0, so the eval_runs half of the migration runs on this same open too.
    assert "params" in _columns(str(db), "eval_runs")
    assert "faithfulness_method" in _columns(str(db), "eval_runs")
    conn = sqlite3.connect(str(db))
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    conn.close()


def test_a_populated_v1_database_upgrades_to_v2(tmp_path):
    """A genuine v1: eval_runs already has its two columns, `sources` does not
    have file_hash, and the version says so. Only the third migration should
    have anything to do."""
    db = tmp_path / "meta.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.executescript(_PRE_V2_SOURCES + _V0_EVAL_RUNS)
    conn.execute("ALTER TABLE eval_runs ADD COLUMN params TEXT")
    conn.execute("ALTER TABLE eval_runs ADD COLUMN faithfulness_method TEXT")
    _populate_sources(conn)
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    md = MetadataStore(str(db))
    rows = {r["source_path"]: r for r in md.get_sources()}
    md.close()

    assert set(rows) == {"/a.md", "/b.pdf"}
    assert rows["/a.md"]["content_hash"] == "aaa"
    assert rows["/a.md"]["file_hash"] is None
    assert "file_hash" in _columns(str(db), "sources")
    conn = sqlite3.connect(str(db))
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    conn.close()


def test_a_half_migrated_sources_table_is_repaired(tmp_path):
    """The `sources` counterpart of the eval_runs case above: file_hash landed
    but the PRAGMA never did. Every later open must still succeed rather than
    raising 'duplicate column name: file_hash' out of __init__."""
    db = tmp_path / "meta.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.executescript(_PRE_V2_SOURCES + _V0_EVAL_RUNS)
    conn.execute("ALTER TABLE sources ADD COLUMN file_hash TEXT")
    _populate_sources(conn)
    conn.close()

    md = MetadataStore(str(db))          # must not raise
    rows = md.get_sources()
    md.close()

    assert len(rows) == 2
    conn = sqlite3.connect(str(db))
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    conn.close()


def test_hash_file_changes_with_content(tmp_path):
    from core.metadata import hash_file

    f = tmp_path / "x.md"
    f.write_text("hello", encoding="utf-8")
    first = hash_file(f)
    assert first
    assert hash_file(f) == first          # stable
    f.write_text("goodbye", encoding="utf-8")
    assert hash_file(f) != first          # content-sensitive


def test_hash_file_missing_returns_empty(tmp_path):
    from core.metadata import hash_file

    assert hash_file(tmp_path / "nope.md") == ""


def test_source_hashes_and_delete_source(tmp_path):
    md = MetadataStore(str(tmp_path / "meta.sqlite3"))
    md.record_source(source_path="/a.md", doc_type="markdown", topic="t",
                     chunk_count=1, content_hash="c1", file_hash="f1")
    md.record_source(source_path="/b.md", doc_type="markdown", topic="t",
                     chunk_count=1, content_hash="c2")
    assert md.source_hashes() == {"/a.md": "f1", "/b.md": None}
    assert md.delete_source("/a.md") is True
    assert md.delete_source("/a.md") is False      # already gone
    assert set(md.source_hashes()) == {"/b.md"}
    md.close()
