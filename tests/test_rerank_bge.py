"""Unit tests for the BGE reranker.

We mock the CrossEncoder to keep the test fast and CI-friendly
(no model download, no GPU). The mock returns predetermined scores
so we can assert the reorder + top_k behavior deterministically.

The test venv does NOT have `sentence_transformers` installed (the
real .venv does, but the test venv is a slim subset for fast CI).
The `with patch("sentence_transformers.CrossEncoder", ...)` lines
require the module to be importable, so we inject a stub into
sys.modules at the top of this file. The stub has just enough
surface for `patch` to find the attribute and for the reranker's
lazy `from sentence_transformers import CrossEncoder` to succeed
inside the patch context (BgeReranker does the import lazily in
`__init__`, so we don't trigger it at module import time).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Inject a `sentence_transformers` stub into sys.modules so the
# `with patch("sentence_transformers.CrossEncoder", ...)` lines in
# the tests below can find the attribute. The CrossEncoder class
# itself is replaced by the per-test MagicMock, so this stub is
# only a placeholder.
if "sentence_transformers" not in sys.modules:
    _stub = types.ModuleType("sentence_transformers")
    _stub.CrossEncoder = MagicMock()  # placeholder; tests override it
    sys.modules["sentence_transformers"] = _stub

from core.interfaces import Chunk
from providers.rerank_bge import BgeReranker


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"cid-{idx}",
        parent_id="parent-1",
        text=text,
        source_path="notes/test.md",
        topic="test",
        doc_type="markdown",
        section=f"sec-{idx}",
    )


def _build_reranker_with_rank_mock(rank_return):
    """Construct a BgeReranker whose CrossEncoder.rank() returns `rank_return`."""
    fake_model = MagicMock()
    fake_model.rank = MagicMock(return_value=rank_return)
    with patch("sentence_transformers.CrossEncoder", return_value=fake_model):
        rr = BgeReranker(model="fake/model", device="cpu")
    return rr, fake_model


# ---- rank() API (newer CrossEncoder) ---------------------------------------

def test_rerank_dict_response_newer_api():
    """Newer CrossEncoder.rank() returns a list of dicts with corpus_id + score."""
    chunks = [_chunk("alpha doc", 0), _chunk("beta doc", 1), _chunk("gamma doc", 2)]
    # rank() returns the top-k after sorting internally. Mock returns the
    # top-2 (matching the top_k the caller asked for): gamma then alpha.
    rank_return = [
        {"corpus_id": 2, "score": 0.95},
        {"corpus_id": 0, "score": 0.70},
    ]
    rr, model = _build_reranker_with_rank_mock(rank_return)

    out = rr.rerank("q", chunks, top_k=2)
    # Top-2: gamma (idx 2) then alpha (idx 0). Beta is cut.
    assert [c.chunk_id for c in out] == ["cid-2", "cid-0"]
    # The model's rank was called once with the right query + top_k.
    model.rank.assert_called_once()
    kwargs = model.rank.call_args.kwargs
    assert kwargs["query"] == "q"
    assert kwargs["top_k"] == 2
    assert kwargs["documents"] == ["alpha doc", "beta doc", "gamma doc"]


def test_rerank_tuple_response_older_api():
    """Older CrossEncoder returns (corpus_id, score) tuples."""
    chunks = [_chunk("a", 0), _chunk("b", 1), _chunk("c", 2)]
    rank_return = [(1, 0.9), (0, 0.5), (2, 0.1)]
    rr, _ = _build_reranker_with_rank_mock(rank_return)
    out = rr.rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == ["cid-1", "cid-0", "cid-2"]


# ---- edge cases ------------------------------------------------------------

def test_rerank_empty_chunks_returns_empty():
    rr, model = _build_reranker_with_rank_mock([])
    assert rr.rerank("q", [], top_k=5) == []
    # No model call expected.
    model.rank.assert_not_called()


def test_rerank_top_k_zero_returns_empty():
    rr, model = _build_reranker_with_rank_mock([])
    chunks = [_chunk("x", 0)]
    assert rr.rerank("q", chunks, top_k=0) == []
    model.rank.assert_not_called()


def test_rerank_top_k_larger_than_chunks_caps_at_chunks():
    """If top_k > len(chunks), we should still get all chunks back, sorted."""
    chunks = [_chunk("a", 0), _chunk("b", 1)]
    # Mock returns top-2 since we asked for min(10, 2) = 2.
    rank_return = [(1, 0.8), (0, 0.4)]
    rr, model = _build_reranker_with_rank_mock(rank_return)
    out = rr.rerank("q", chunks, top_k=10)
    assert [c.chunk_id for c in out] == ["cid-1", "cid-0"]
    # Asked for 10 but capped to 2 before calling the model.
    kwargs = model.rank.call_args.kwargs
    assert kwargs["top_k"] == 2


def test_rerank_preserves_chunk_objects_unchanged():
    """The reranker must not mutate the input chunks (only reorder)."""
    chunks = [_chunk("a", 0), _chunk("b", 1), _chunk("c", 2)]
    rank_return = [{"corpus_id": 2, "score": 0.9}, {"corpus_id": 1, "score": 0.5}]
    rr, _ = _build_reranker_with_rank_mock(rank_return)
    out = rr.rerank("q", chunks, top_k=2)
    # Returned chunks are the same objects, in the new order.
    assert out[0] is chunks[2]
    assert out[1] is chunks[1]
    # Section / text / etc. unchanged.
    assert out[0].text == "c"
    assert out[0].section == "sec-2"


# ---- fallback to predict() when rank() is missing --------------------------

def test_rerank_falls_back_to_predict_when_rank_missing():
    """If a model class doesn't expose rank(), we fall back to predict() + sort."""
    chunks = [_chunk("a", 0), _chunk("b", 1), _chunk("c", 2)]
    fake_model = MagicMock()
    # No `rank` attribute at all — accessing it should raise AttributeError.
    del fake_model.rank
    # predict() returns scores in input order: c=highest, a=mid, b=lowest.
    import numpy as np
    fake_model.predict = MagicMock(return_value=np.array([0.4, 0.1, 0.9]))

    with patch("sentence_transformers.CrossEncoder", return_value=fake_model):
        rr = BgeReranker(model="fake/model", device="cpu")
    out = rr.rerank("q", chunks, top_k=2)
    # Sorted by score desc: c (0.9), a (0.4). b cut.
    assert [c.chunk_id for c in out] == ["cid-2", "cid-0"]
    fake_model.predict.assert_called_once()
    # Pairs were (query, chunk_text) — verify the first arg shape.
    pairs = fake_model.predict.call_args.args[0]
    assert pairs[0] == ("q", "a")
    assert pairs[2] == ("q", "c")
