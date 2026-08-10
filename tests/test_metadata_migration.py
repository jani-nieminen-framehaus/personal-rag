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
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    conn.close()


def test_reopen_is_idempotent(tmp_path):
    db = tmp_path / "meta.sqlite3"
    MetadataStore(str(db)).close()
    MetadataStore(str(db)).close()  # second open must not ALTER again / raise


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
