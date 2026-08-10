# tests/test_eval_sweep_support.py
"""run() extensions for the sweep: params recording + section match mode."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from eval import run_ragas


def _golden(tmp_path, row):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return p


def _result(citations):
    r = MagicMock()
    r.answer = "a"
    r.citations = citations
    # Controller correction: dense_hits carry the same keys as citations
    # (source_path, section, chunk_id, ...) — NOT chunk_id alone. See
    # core/pipeline.py:476-484, where each dense hit is built with
    # source_path and section. Aliasing dense_ids to retrieved_ids would
    # silently turn recall_at_dense into a post-rerank metric.
    r.dense_hits = [dict(c) for c in citations]
    return r


def test_extra_params_recorded(monkeypatch, tmp_path):
    golden = _golden(tmp_path, {"question": "q", "relevant_chunk_ids": ["c1"]})
    cite = {"chunk_id": "c1", "text": "t", "source_path": "/a.md", "section": "S"}
    monkeypatch.setattr(run_ragas, "ask_pipeline", lambda *a, **kw: _result([cite]))
    metadata = MagicMock()

    run_ragas.run(
        golden, embedder=MagicMock(), store=MagicMock(), reranker=MagicMock(),
        generator=None, metadata=metadata,
        extra_params={"dense_weight": 0.7, "reranker": "bge"},
    )
    kwargs = metadata.record_eval_run.call_args.kwargs
    assert kwargs["params"]["dense_weight"] == 0.7
    assert kwargs["params"]["top_k_dense"] == 20  # base params merged in


def test_hybrid_and_dense_weight_forwarded(monkeypatch, tmp_path):
    golden = _golden(tmp_path, {"question": "q", "relevant_chunk_ids": ["c1"]})
    seen = {}

    def fake_ask(*a, **kw):
        seen.update(kw)
        return _result([{"chunk_id": "c1", "text": "t",
                         "source_path": "/a.md", "section": "S"}])

    monkeypatch.setattr(run_ragas, "ask_pipeline", fake_ask)
    run_ragas.run(golden, embedder=MagicMock(), store=MagicMock(),
                  reranker=MagicMock(), generator=None,
                  hybrid=True, dense_weight=0.3)
    assert seen["hybrid"] is True
    assert seen["dense_weight"] == 0.3


def test_section_match_mode(monkeypatch, tmp_path):
    """chunk_ids drift across chunking configs; section match must not."""
    golden = _golden(tmp_path, {
        "question": "q",
        "relevant_chunk_ids": ["stale-id-from-other-chunking"],
        "relevant_refs": [{"source_path": "/a.md", "section": "S"}],
    })
    cite = {"chunk_id": "fresh-id", "text": "t", "source_path": "/a.md", "section": "S"}
    monkeypatch.setattr(run_ragas, "ask_pipeline", lambda *a, **kw: _result([cite]))

    m_chunk = run_ragas.run(golden, embedder=MagicMock(), store=MagicMock(),
                            reranker=MagicMock(), generator=None)
    m_sect = run_ragas.run(golden, embedder=MagicMock(), store=MagicMock(),
                           reranker=MagicMock(), generator=None,
                           match_mode="section")
    assert m_chunk["recall_at_5"] == 0.0   # stale id no longer exists
    assert m_sect["recall_at_5"] == 1.0    # section survives re-chunking


def test_section_match_mode_requires_relevant_refs(monkeypatch, tmp_path):
    """Controller-directed requirement: a golden row with no relevant_refs
    under match_mode="section" must fail loudly (ValueError naming the
    offending question), not silently score 0.0. eval/golden_set.jsonl has
    no relevant_refs today — silently scoring those rows 0.0 would make a
    whole chunking sweep look uniformly terrible for the wrong reason."""
    golden = _golden(tmp_path, {
        "question": "which question lacks relevant_refs",
        "relevant_chunk_ids": ["c1"],
    })
    cite = {"chunk_id": "c1", "text": "t", "source_path": "/a.md", "section": "S"}
    monkeypatch.setattr(run_ragas, "ask_pipeline", lambda *a, **kw: _result([cite]))

    with pytest.raises(ValueError, match="which question lacks relevant_refs"):
        run_ragas.run(golden, embedder=MagicMock(), store=MagicMock(),
                      reranker=MagicMock(), generator=None,
                      match_mode="section")
