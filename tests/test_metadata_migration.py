"""eval_runs migration: user_version 0 -> 1 adds params + faithfulness_method."""
from __future__ import annotations

import sqlite3

from core.metadata import MetadataStore


def _columns(db_path, table):
    conn = sqlite3.connect(db_path)
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def test_fresh_db_is_version_1_with_new_columns(tmp_path):
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


def test_v1_database_upgrades_to_v2(tmp_path):
    """A database already at v1 must gain file_hash without losing rows."""
    db = tmp_path / "meta.sqlite3"
    md = MetadataStore(str(db))
    md.record_source(source_path="/a.md", doc_type="markdown", topic="t",
                     chunk_count=3, content_hash="abc")
    md.close()
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    md2 = MetadataStore(str(db))
    rows = md2.get_sources()
    md2.close()
    assert len(rows) == 1
    assert rows[0]["content_hash"] == "abc"
    assert rows[0]["file_hash"] is None


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
