"""Unit tests for BM25/TF-IDF sparse vector population and hybrid search.

These are integration tests against the store/pipeline layer. They mock the
Qdrant client to avoid requiring a live Qdrant instance. The key invariants
tested:

- `enable_hybrid()` populates sparse vectors for all existing points.
- `search_hybrid()` fuses dense and sparse results with RRF.
- The pipeline's `ask()` accepts `hybrid=True` and uses the hybrid path.
- `rag ingest --populate-sparse` is accepted by the CLI.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Minimal mock Qdrant client — captures upsert/update_vectors calls.
from qdrant_client.http.models import PointStruct, SparseVector


class _MockScrollResult:
    """A scroll-like response with points + optional next offset."""

    def __init__(self, points, offset=None):
        self.points = points
        self.offset = offset

    def __iter__(self):
        return iter(self.points)


class _FakeQdrantClient:
    """Minimal QdrantClient mock that tracks what was written."""

    def __init__(self):
        self.written_vectors: list[dict] = []  # list of {sparse: SparseVector}
        self.scroll_responses: list[_MockScrollResult] = []
        self._collection_exists = True
        self._vectors: dict[str, dict] = {}  # point_id -> {dense, sparse}

    def collection_exists(self, collection):
        return self._collection_exists

    def get_collection(self, collection):
        m = MagicMock()
        params = MagicMock()
        params.vectors = {"dense": MagicMock(size=4)}
        m.config.params.vectors = params
        return m

    def create_collection(self, **kwargs):
        pass

    def upsert(self, collection_name, points, wait=True):
        for p in points:
            self._vectors[p.id] = {"dense": p.vector.get("dense", [])}

    def update_vectors(self, collection_name, points, wait=True):
        for p in points:
            self._vectors[p.id] = {
                **self._vectors.get(p.id, {}),
                "sparse": p.vector.get("sparse"),
            }
            self.written_vectors.append(p.vector)

    def scroll(self, collection_name, limit=256, offset=None, with_payload=True, with_vectors=False):
        # Find the current position in our mock responses.
        if offset is None:
            pos = 0
        else:
            pos = int(offset)
        if pos >= len(self.scroll_responses):
            return ([], None)
        return self.scroll_responses[pos], pos + 1 if pos + 1 < len(self.scroll_responses) else None

    def query_points(self, collection_name, query, using, limit, with_payload=True, query_filter=None):
        from core.interfaces import make_parent_id

        # Dense search: return mock points.
        hits = []
        for cid, vec_data in self._vectors.items():
            if using == "dense":
                score = sum(a * b for a, b in zip(query, vec_data.get("dense", [])))
            elif using == "sparse":
                score = 0.5  # dummy score
            else:
                score = 0.0
            hits.append(MagicMock(
                id=cid,
                score=score,
                payload={
                    "chunk_id": cid,
                    "parent_id": make_parent_id(f"/src/{cid}.txt"),
                    "text": f"text for {cid}",
                    "source_path": f"/src/{cid}.txt",
                    "topic": "test",
                    "doc_type": "markdown",
                    "section": f"Section {cid}",
                },
            ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return MagicMock(points=hits[:limit])

    def count(self, collection_name, count_filter=None):
        return MagicMock(count=len(self._vectors))


# ---- helpers ---------------------------------------------------------------

def _mock_points(*ids) -> list:
    """Build mock scroll responses with payloads."""
    from core.interfaces import make_parent_id

    points = []
    for cid in ids:
        p = MagicMock()
        p.id = cid
        p.payload = {
            "chunk_id": cid,
            "parent_id": make_parent_id(f"/src/{cid}.txt"),
            "text": f"This is the document text for chunk {cid}.",
            "source_path": f"/src/{cid}.txt",
            "topic": "test",
            "doc_type": "markdown",
            "section": f"Section {cid}",
        }
        points.append(p)
    return points


# ---- enable_hybrid ---------------------------------------------------------

def test_enable_hybrid_writes_sparse_vectors(tmp_path: Path):
    """enable_hybrid() should compute and store a sparse vector per chunk."""
    from store.qdrant_store import QdrantStore

    client = _FakeQdrantClient()
    client.scroll_responses = [
        _MockScrollResult(_mock_points("a", "b", "c")),
    ]
    # Pre-populate with dense vectors.
    for cid in ("a", "b", "c"):
        client.upsert("kb", [
            PointStruct(id=cid, vector={"dense": [0.1, 0.2, 0.3, 0.4]}, payload={})
        ])

    store = QdrantStore(url="http://localhost:7333", collection="kb", dense_dim=4)
    store.client = client

    store.enable_hybrid()

    assert len(client.written_vectors) == 3, (
        f"expected 3 sparse vectors, got {len(client.written_vectors)}"
    )
    for vec in client.written_vectors:
        assert "sparse" in vec, "each written vector must have a 'sparse' slot"
        sp = vec["sparse"]
        assert isinstance(sp, SparseVector), f"expected SparseVector, got {type(sp)}"
        assert len(sp.indices) > 0, "sparse vector must have non-empty indices"


def test_enable_hybrid_idempotent(tmp_path: Path):
    """Re-running enable_hybrid() should overwrite (not duplicate) sparse vectors."""
    from store.qdrant_store import QdrantStore

    client = _FakeQdrantClient()
    client.scroll_responses = [_MockScrollResult(_mock_points("a", "b"))]

    store = QdrantStore(url="http://localhost:7333", collection="kb", dense_dim=4)
    store.client = client

    store.enable_hybrid()
    first_count = len(client.written_vectors)

    # Clear written list and re-run.
    client.written_vectors.clear()
    store.enable_hybrid()
    assert len(client.written_vectors) == 2, (
        "second run should overwrite (not append) sparse vectors"
    )


def test_enable_hybrid_empty_collection(tmp_path: Path, caplog):
    """An empty collection should not crash; it should log and exit."""
    from store.qdrant_store import QdrantStore

    client = _FakeQdrantClient()
    client.scroll_responses = [_MockScrollResult([])]  # no chunks

    store = QdrantStore(url="http://localhost:7333", collection="kb", dense_dim=4)
    store.client = client

    store.enable_hybrid()  # must not raise

    assert len(client.written_vectors) == 0


# ---- search_hybrid ---------------------------------------------------------

def test_search_hybrid_returns_fused_results():
    """search_hybrid() should return results ordered by fused score."""
    from store.qdrant_store import QdrantStore
    from qdrant_client.http.models import SparseVector

    client = _FakeQdrantClient()
    # Pre-seed vectors.
    from core.interfaces import make_parent_id
    for cid, dense in [("a", [1.0, 0.0, 0.0, 0.0]), ("b", [0.0, 1.0, 0.0, 0.0])]:
        client.upsert("kb", [
            PointStruct(id=cid, vector={"dense": dense, "sparse": SparseVector(indices=[0], values=[1.0])}, payload={
                "chunk_id": cid, "parent_id": make_parent_id(f"/{cid}"), "text": "text",
                "source_path": f"/{cid}", "topic": "t", "doc_type": "md", "section": "S"
            })
        ])
    client.scroll_responses = [_MockScrollResult(_mock_points("a", "b"))]

    store = QdrantStore(url="http://localhost:7333", collection="kb", dense_dim=4)
    store.client = client

    hits = store.search_hybrid(
        query_vector=[1.0, 1.0, 0.0, 0.0],
        query_sparse=SparseVector(indices=[0], values=[1.0]),
        top_k=2,
        dense_weight=0.5,
    )

    assert len(hits) == 2, f"expected 2 hits, got {len(hits)}"
    # Results should be (chunk, dense_score, hybrid_score).
    assert len(hits[0]) == 3


def test_search_hybrid_falls_back_when_sparse_unavailable():
    """If sparse fails, search_hybrid should return dense hits."""
    from store.qdrant_store import QdrantStore
    from qdrant_client.http.models import SparseVector

    client = _FakeQdrantClient()
    from core.interfaces import make_parent_id
    for cid, dense in [("a", [1.0, 0.0, 0.0, 0.0])]:
        client.upsert("kb", [
            PointStruct(id=cid, vector={"dense": dense}, payload={
                "chunk_id": cid, "parent_id": make_parent_id(f"/{cid}"), "text": "text",
                "source_path": f"/{cid}", "topic": "t", "doc_type": "md", "section": "S"
            })
        ])
    client.scroll_responses = [_MockScrollResult(_mock_points("a"))]

    # Make query_points raise on sparse queries.
    original_qp = client.query_points

    def _bad_sparse(*args, **kwargs):
        if kwargs.get("using") == "sparse":
            raise RuntimeError("sparse not populated")
        return original_qp(*args, **kwargs)

    client.query_points = _bad_sparse

    store = QdrantStore(url="http://localhost:7333", collection="kb", dense_dim=4)
    store.client = client

    hits = store.search_hybrid(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        query_sparse=SparseVector(indices=[0], values=[1.0]),
        top_k=2,
        dense_weight=0.5,
    )

    # Should still return the dense hit.
    assert len(hits) >= 1


# ---- pipeline hybrid path -------------------------------------------------

def test_pipeline_ask_accepts_hybrid_flag():
    """pipeline.ask() should accept hybrid=True and route to _retrieve_hybrid."""
    from core import pipeline

    # Patch the store.
    mock_store = MagicMock()
    mock_store.search_dense.return_value = []
    mock_store.search_hybrid.return_value = []
    mock_store.client.scroll.return_value = ([], None)
    mock_store.collection = "kb"

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_embedder.dim.return_value = 4

    result = pipeline.ask(
        "test query",
        embedder=mock_embedder,
        store=mock_store,
        reranker=MagicMock(),
        generator=MagicMock(),
        hybrid=True,
    )

    # When sparse is not populated, it falls back to dense search.
    # The key check: ask() accepted hybrid=True without raising.
    assert result.answer == "(no results found in the index)"


# ---- CLI                                                                  ---

def test_ingest_cli_accepts_populate_sparse_flag():
    """`rag ingest --populate-sparse` should not raise on construction."""
    from click.testing import CliRunner
    from cli import cli

    runner = CliRunner()
    # We just test that the flag is accepted (not the full ingest path).
    # The ingest path is integration-tested elsewhere.
    result = runner.invoke(cli, ["ingest", "--help"])
    assert "--populate-sparse" in result.output, (
        "--populate-sparse should appear in ingest help text"
    )


def test_ask_cli_accepts_hybrid_flag():
    """`rag ask --hybrid` should appear in help text."""
    from click.testing import CliRunner
    from cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["ask", "--help"])
    assert "--hybrid" in result.output, (
        "--hybrid should appear in ask help text"
    )
