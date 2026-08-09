"""Unit tests for the SQLite metadata layer.

We use temp files (not `:memory:`) so the tests exercise the same
disk-backed path the production code uses. The file is per-test
(tmp_path) so they're isolated.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from core.metadata import MetadataStore, hash_text


@pytest.fixture
def store(tmp_path: Path) -> MetadataStore:
    s = MetadataStore(tmp_path / "test.sqlite3")
    yield s
    s.close()


# ---- schema + open --------------------------------------------------------

def test_open_creates_schema(store: MetadataStore):
    """An empty dir + a fresh path should produce all 3 tables + indexes."""
    cur = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in cur.fetchall()}
    assert {"sources", "citations", "eval_runs"} <= tables


def test_open_idempotent(tmp_path: Path):
    """Opening the same file twice should not error or lose data."""
    p = tmp_path / "x.sqlite3"
    a = MetadataStore(p)
    a.record_source("/a.md", "markdown", "ml", 3, "abc")
    a.close()
    b = MetadataStore(p)
    rows = b.get_sources()
    assert len(rows) == 1
    assert rows[0]["source_path"] == "/a.md"
    b.close()


# ---- record_source --------------------------------------------------------

def test_record_source_insert(store: MetadataStore):
    store.record_source(
        source_path="notes/photography/exposure.md",
        doc_type="markdown",
        topic="photography",
        chunk_count=4,
        content_hash="abc123",
    )
    rows = store.get_sources()
    assert len(rows) == 1
    r = rows[0]
    assert r["source_path"] == "notes/photography/exposure.md"
    assert r["doc_type"] == "markdown"
    assert r["topic"] == "photography"
    assert r["chunk_count"] == 4
    assert r["content_hash"] == "abc123"
    # ingested_at parses as a valid ISO timestamp.
    datetime.fromisoformat(r["ingested_at"])


def test_record_source_upsert(store: MetadataStore):
    """Re-recording the same path refreshes fields (idempotent ingest)."""
    store.record_source("notes/x.md", "markdown", "ml", 3, "hash1")
    store.record_source("notes/x.md", "markdown", "ml", 5, "hash2")
    rows = store.get_sources()
    assert len(rows) == 1
    assert rows[0]["chunk_count"] == 5
    assert rows[0]["content_hash"] == "hash2"


def test_record_source_returns_no_metadata_db_state(store: MetadataStore):
    """record_source returns None (no row counts) — it's fire-and-log."""
    result = store.record_source("x.md", "markdown", "ml", 1, "h")
    assert result is None


# ---- record_citations -----------------------------------------------------

def test_record_citations_writes_one_row_per_citation(store: MetadataStore):
    citations = [
        {"n": 1, "chunk_id": "c1", "source_path": "a.md"},
        {"n": 2, "chunk_id": "c2", "source_path": "a.md"},
        {"n": 3, "chunk_id": "c3", "source_path": "b.md"},
    ]
    n = store.record_citations("What is exposure?", citations)
    assert n == 3
    rows = store.get_citations()
    assert len(rows) == 3
    # Most recent first.
    assert rows[0]["chunk_id"] == "c3"


def test_record_citations_empty_iterable_returns_zero(store: MetadataStore):
    n = store.record_citations("q", [])
    assert n == 0
    assert store.get_citations() == []


def test_get_citations_filter_by_source(store: MetadataStore):
    store.record_citations("q1", [{"n": 1, "chunk_id": "c1", "source_path": "a.md"}])
    store.record_citations("q2", [{"n": 1, "chunk_id": "c2", "source_path": "b.md"}])
    store.record_citations("q3", [{"n": 1, "chunk_id": "c3", "source_path": "a.md"}])
    only_a = store.get_citations(source_path="a.md")
    assert len(only_a) == 2
    assert {r["chunk_id"] for r in only_a} == {"c1", "c3"}


def test_get_citations_limit(store: MetadataStore):
    for i in range(10):
        store.record_citations(f"q{i}", [{"n": 1, "chunk_id": f"c{i}", "source_path": "a.md"}])
    assert len(store.get_citations(limit=3)) == 3


# ---- record_eval_run ------------------------------------------------------

def test_record_eval_run_writes_row(store: MetadataStore):
    store.record_eval_run(
        n_questions=9, recall_at_5=0.89, mrr=0.92,
        recall_at_dense=0.78, faithfulness_proxy=0.65,
    )
    rows = store.get_eval_runs()
    assert len(rows) == 1
    r = rows[0]
    assert r["n_questions"] == 9
    assert r["recall_at_5"] == 0.89
    assert r["mrr"] == 0.92
    assert r["recall_at_dense"] == 0.78
    assert r["faithfulness_proxy"] == 0.65


def test_record_eval_run_allows_none_optionals(store: MetadataStore):
    """recall_at_dense and faithfulness_proxy are optional (None = NULL)."""
    store.record_eval_run(n_questions=5, recall_at_5=0.5, mrr=0.5)
    r = store.get_eval_runs()[0]
    assert r["recall_at_dense"] is None
    assert r["faithfulness_proxy"] is None


# ---- get_stats ------------------------------------------------------------

def test_get_stats_empty_db(store: MetadataStore):
    s = store.get_stats()
    assert s["total_sources"] == 0
    assert s["total_citations"] == 0
    assert s["total_eval_runs"] == 0
    assert s["top_cited_source"] is None
    assert s["top_cited_count"] == 0


def test_get_stats_with_data(store: MetadataStore):
    store.record_source("a.md", "markdown", "ml", 3, "h1")
    store.record_source("b.md", "markdown", "code", 5, "h2")
    store.record_citations("q", [
        {"n": 1, "chunk_id": "c1", "source_path": "a.md"},
        {"n": 2, "chunk_id": "c2", "source_path": "a.md"},
        {"n": 3, "chunk_id": "c3", "source_path": "b.md"},
    ])
    store.record_eval_run(n_questions=9, recall_at_5=0.8, mrr=0.9)
    s = store.get_stats()
    assert s["total_sources"] == 2
    assert s["total_citations"] == 3
    assert s["total_eval_runs"] == 1
    # a.md is cited 2x, b.md 1x → a.md is the top.
    assert s["top_cited_source"] == "a.md"
    assert s["top_cited_count"] == 2


def test_get_stats_ignores_empty_source_paths(store: MetadataStore):
    """Citation rows with empty source_path shouldn't count toward 'top cited'."""
    store.record_citations("q", [
        {"n": 1, "chunk_id": "c1", "source_path": ""},
        {"n": 2, "chunk_id": "c2", "source_path": "real.md"},
    ])
    s = store.get_stats()
    assert s["top_cited_source"] == "real.md"
    assert s["top_cited_count"] == 1


# ---- hash_text ------------------------------------------------------------

def test_hash_text_deterministic():
    """Same text → same hash, every time."""
    assert hash_text("hello world") == hash_text("hello world")


def test_hash_text_distinguishes():
    """Different text → different hash."""
    assert hash_text("alpha") != hash_text("beta")


def test_hash_text_empty_string():
    """Empty input returns empty (we never want a NULL in the column)."""
    assert hash_text("") == ""


def test_hash_text_short_format():
    """The hash is a 16-char prefix of the SHA-256 hex digest."""
    h = hash_text("anything")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
