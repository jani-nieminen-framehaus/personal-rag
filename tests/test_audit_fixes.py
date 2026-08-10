"""Regression tests for the 2026-08-10 P2 audit fixes (Wave A).

Covers, one group per verified finding:
- hybrid IDF cache: invalidated by ingest (success AND failure paths).
- search_hybrid: true weighted RRF — rank-based, raw score scales can't leak.
- ingest: a source failing mid-stream records partial metadata and re-raises.
- embed_query: blank/whitespace queries raise instead of returning junk.
- cli ingest: --populate-sparse failure exits non-zero; --recreate prompts
  unless --yes.
- serve: state file written atomically, no .tmp left behind.
- run_ragas: golden loader rejects non-list relevant_chunk_ids; faithfulness
  rows and summary carry a faithfulness_method tag.
- build_golden: drifted references make main() return 1.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

from core.interfaces import Chunk


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        parent_id=f"parent-{cid}",
        text=f"text for {cid}",
        source_path=f"/src/{cid}.md",
        topic="test",
        doc_type="markdown",
        section=f"Section {cid}",
    )


def _fake_embedder() -> MagicMock:
    emb = MagicMock()
    emb.dim.return_value = 4
    emb.batch_size = 32
    emb.embed.side_effect = lambda texts: [[0.0] * 4 for _ in texts]
    return emb


# -- hybrid cache invalidation ------------------------------------------------

def test_ingest_invalidates_hybrid_cache():
    """Re-ingest must drop the cached IDF map — a stale map silently shifts
    hybrid scores against the new corpus."""
    from core import pipeline

    pipeline._hybrid_cache.update({"vocab": {"x": 0}, "idf": {"x": 1.0}, "built": True})

    class OneChunkIngester:
        def iter_chunks(self):
            yield _chunk("c1")

    store = MagicMock()
    store.collection = "kb"

    pipeline.ingest(OneChunkIngester(), _fake_embedder(), store)

    assert pipeline._hybrid_cache["built"] is False
    assert pipeline._hybrid_cache["vocab"] is None
    assert pipeline._hybrid_cache["idf"] is None


def test_ingest_failure_records_partial_sources_and_invalidates():
    """A source exploding mid-stream must leave metadata matching what
    actually landed in the collection, and still invalidate the cache."""
    from core import pipeline

    pipeline._hybrid_cache.update({"vocab": {"x": 0}, "idf": {"x": 1.0}, "built": True})

    class ExplodingIngester:
        def iter_chunks(self):
            yield _chunk("ok-1")
            raise RuntimeError("bad source")

    store = MagicMock()
    store.collection = "kb"
    metadata = MagicMock()

    with pytest.raises(RuntimeError, match="bad source"):
        pipeline.ingest(
            ExplodingIngester(), _fake_embedder(), store,
            metadata=metadata, batch_size=1,
        )

    # The chunk that flushed before the failure is recorded.
    metadata.record_source.assert_called_once()
    assert pipeline._hybrid_cache["built"] is False


# -- weighted RRF fusion -------------------------------------------------------

def test_search_hybrid_is_rank_based_not_score_based():
    """Raw score magnitudes must not leak into the fusion: a 12345-vs-0.01
    sparse score gap with mirrored ranks yields symmetric fused scores."""
    from store.qdrant_store import QdrantStore

    store = QdrantStore.__new__(QdrantStore)  # skip __init__/client
    a, b = _chunk("A"), _chunk("B")
    store.search_dense = lambda qv, top_k=20: [(a, 0.90), (b, 0.89)]
    store._search_sparse = lambda qs, top_k=20: [(b, 12345.0), (a, 0.01)]

    hits = store.search_hybrid(
        query_vector=[0.0], query_sparse=MagicMock(), top_k=2, dense_weight=0.5,
    )

    k = 60
    scores = {c.chunk_id: hs for c, _ds, hs in hits}
    assert scores["A"] == pytest.approx(0.5 / (k + 1) + 0.5 / (k + 2))
    assert scores["B"] == pytest.approx(0.5 / (k + 2) + 0.5 / (k + 1))
    assert scores["A"] == pytest.approx(scores["B"])


def test_search_hybrid_absent_branch_contributes_zero():
    """A chunk missing from one branch gets no phantom contribution from it."""
    from store.qdrant_store import QdrantStore

    store = QdrantStore.__new__(QdrantStore)
    a, c = _chunk("A"), _chunk("C")
    store.search_dense = lambda qv, top_k=20: [(a, 0.9), (c, 0.5)]
    store._search_sparse = lambda qs, top_k=20: [(a, 3.0)]

    hits = store.search_hybrid(
        query_vector=[0.0], query_sparse=MagicMock(), top_k=5, dense_weight=0.5,
    )

    k = 60
    scores = {ch.chunk_id: hs for ch, _ds, hs in hits}
    assert scores["A"] == pytest.approx(0.5 / (k + 1) + 0.5 / (k + 1))
    assert scores["C"] == pytest.approx(0.5 / (k + 2))
    assert hits[0][0].chunk_id == "A"


# -- embed_query blank guard ---------------------------------------------------

def test_embed_query_raises_on_blank_and_whitespace(monkeypatch):
    """'' returned a zero vector (undefined under cosine) and '   ' embedded
    just the instruction prefix — both must raise now."""
    try:
        import torch  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "torch", MagicMock())
    from providers.embed_qwen3 import Qwen3Embedder

    emb = Qwen3Embedder.__new__(Qwen3Embedder)  # guard runs before any attrs
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(ValueError, match="empty or whitespace"):
            emb.embed_query(bad)


# -- cli ingest: sparse failure + recreate gate --------------------------------

def _patch_cli_factories(monkeypatch, store):
    import cli as cli_mod
    monkeypatch.setattr(cli_mod, "make_embedder", lambda cfg: MagicMock())
    monkeypatch.setattr(cli_mod, "make_store", lambda cfg: store)
    monkeypatch.setattr(cli_mod, "make_metadata", lambda cfg: None)
    fake_ingest = MagicMock(return_value=0)
    monkeypatch.setattr(cli_mod, "ingest_pipeline", fake_ingest)
    return cli_mod, fake_ingest


def test_populate_sparse_failure_exits_nonzero(monkeypatch, tmp_path):
    """Sparse population failing after a successful ingest must NOT report
    plain success — the index is in a half-state."""
    from click.testing import CliRunner

    store = MagicMock()
    store.collection = "kb"
    store.enable_hybrid.side_effect = RuntimeError("qdrant down")
    cli_mod, _ = _patch_cli_factories(monkeypatch, store)

    notes = tmp_path / "notes"
    notes.mkdir()
    res = CliRunner().invoke(
        cli_mod.cli, ["ingest", "--markdown", str(notes), "--populate-sparse"],
    )

    assert res.exit_code == 1
    assert "sparse population failed" in res.output
    assert "dense-only" in res.output


def test_recreate_prompts_and_aborts_on_no(monkeypatch, tmp_path):
    from click.testing import CliRunner

    store = MagicMock()
    store.collection = "kb"
    cli_mod, fake_ingest = _patch_cli_factories(monkeypatch, store)

    notes = tmp_path / "notes"
    notes.mkdir()
    res = CliRunner().invoke(
        cli_mod.cli, ["ingest", "--markdown", str(notes), "--recreate"],
        input="n\n",
    )

    assert res.exit_code != 0
    fake_ingest.assert_not_called()


def test_recreate_with_yes_skips_prompt(monkeypatch, tmp_path):
    from click.testing import CliRunner

    store = MagicMock()
    store.collection = "kb"
    cli_mod, fake_ingest = _patch_cli_factories(monkeypatch, store)

    notes = tmp_path / "notes"
    notes.mkdir()
    res = CliRunner().invoke(
        cli_mod.cli, ["ingest", "--markdown", str(notes), "--recreate", "--yes"],
    )

    assert res.exit_code == 0
    fake_ingest.assert_called_once()


# -- serve: atomic state write -------------------------------------------------

def test_state_write_is_atomic_and_leaves_no_tmp(monkeypatch, tmp_path):
    import serve

    state_file = tmp_path / "state.json"
    monkeypatch.setattr(serve, "STATE_DIR", tmp_path)
    monkeypatch.setattr(serve, "STATE_FILE", state_file)

    serve._state_write("127.0.0.1", 8420)

    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["port"] == 8420
    assert data["pid"] == os.getpid()
    assert list(tmp_path.glob("*.tmp")) == []


# -- run_ragas: golden loader type check + method tag --------------------------

def test_load_golden_rejects_non_list_ids(tmp_path):
    """A bare string would iterate as characters and silently zero recall."""
    from eval.run_ragas import load_golden

    p = tmp_path / "golden.jsonl"
    p.write_text(
        json.dumps({"question": "q", "relevant_chunk_ids": "id1,id2"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="list of strings"):
        load_golden(p)


def test_load_golden_accepts_list_ids(tmp_path):
    from eval.run_ragas import load_golden

    p = tmp_path / "golden.jsonl"
    p.write_text(
        json.dumps({"question": "q", "relevant_chunk_ids": ["id1", "id2"]}) + "\n",
        encoding="utf-8",
    )
    assert load_golden(p)[0]["relevant_chunk_ids"] == ["id1", "id2"]


def _golden_file(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        json.dumps({"question": "q", "relevant_chunk_ids": ["c1"]}) + "\n",
        encoding="utf-8",
    )
    return p


def _fake_ask_result():
    r = MagicMock()
    r.answer = "the answer"
    r.citations = [{"chunk_id": "c1", "text": "the answer text"}]
    r.dense_hits = [{"chunk_id": "c1"}]
    return r


def test_faithfulness_method_tagged_lexical(monkeypatch, tmp_path):
    from eval import run_ragas

    monkeypatch.setattr(run_ragas, "ask_pipeline", lambda *a, **kw: _fake_ask_result())
    summary = run_ragas.run(
        _golden_file(tmp_path),
        embedder=MagicMock(), store=MagicMock(), reranker=MagicMock(),
        generator=object(), metadata=None,
    )
    assert summary["per_question"][0]["faithfulness_method"] == "lexical"
    assert summary["faithfulness_method"] == "lexical"


def test_faithfulness_method_tagged_nli(monkeypatch, tmp_path):
    from eval import run_ragas

    monkeypatch.setattr(run_ragas, "ask_pipeline", lambda *a, **kw: _fake_ask_result())
    nli = MagicMock()
    nli.score.return_value = 0.9
    summary = run_ragas.run(
        _golden_file(tmp_path),
        embedder=MagicMock(), store=MagicMock(), reranker=MagicMock(),
        generator=object(), metadata=None, nli_faithfulness=nli,
    )
    assert summary["per_question"][0]["faithfulness_method"] == "nli"
    assert summary["faithfulness_method"] == "nli"


# -- build_golden: drifted refs fail the build ---------------------------------

def test_build_golden_fails_on_drifted_refs(monkeypatch, tmp_path):
    from eval import build_golden

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"question": "q1", "relevant": [{"path": "gone.md", "section": "S"}]}) + "\n",
        encoding="utf-8",
    )
    golden_out = tmp_path / "golden.jsonl"
    monkeypatch.setattr(build_golden, "SAMPLES", tmp_path)
    monkeypatch.setattr(build_golden, "QUESTIONS", questions)
    monkeypatch.setattr(build_golden, "GOLDEN", golden_out)
    monkeypatch.setattr(build_golden, "build_index", lambda: {})

    assert build_golden.main() == 1
    # Still written, for inspecting the drift.
    assert golden_out.is_file()


def test_build_golden_passes_when_refs_resolve(monkeypatch, tmp_path):
    from eval import build_golden

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"question": "q1", "relevant": [{"path": "a.md", "section": "S"}]}) + "\n",
        encoding="utf-8",
    )
    golden_out = tmp_path / "golden.jsonl"
    monkeypatch.setattr(build_golden, "SAMPLES", tmp_path)
    monkeypatch.setattr(build_golden, "QUESTIONS", questions)
    monkeypatch.setattr(build_golden, "GOLDEN", golden_out)
    monkeypatch.setattr(build_golden, "build_index", lambda: {("a.md", "S"): ["cid1"]})

    assert build_golden.main() == 0
    entry = json.loads(golden_out.read_text(encoding="utf-8").strip())
    assert entry["relevant_chunk_ids"] == ["cid1"]
