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

    # Patch the store. The pipeline reaches the corpus only through the
    # VectorStore surface (iter_texts), never through a raw client.
    mock_store = MagicMock()
    mock_store.search_dense.return_value = []
    mock_store.search_hybrid.return_value = []
    mock_store.iter_texts.return_value = []
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


# ---- hybrid topic post-filter (must not confound the top_k_dense axis) ----


@pytest.fixture
def clean_hybrid_cache():
    """The corpus IDF map is a module global. Drop it either side of a
    test so a populated cache can't leak into (or out of) this file."""
    from core import pipeline
    pipeline.invalidate_hybrid_cache()
    yield
    pipeline.invalidate_hybrid_cache()


class _TopicCorpusStore:
    """Fake store whose in-topic chunks are sparse among out-of-topic ones.

    Retrieval is deterministic (corpus order) so the only thing that can
    change the size of the returned list is the retrieval budget itself.
    """

    collection = "kb"

    def __init__(self, n_chunks: int, in_topic_every: int = 3, topic: str = "target"):
        from core.interfaces import Chunk, make_parent_id

        self.chunks = [
            Chunk(
                chunk_id=f"c{i}",
                parent_id=make_parent_id(f"/src/{i}.md"),
                text=f"retrieval passage about exposure compensation number {i}",
                source_path=f"/src/{i}.md",
                topic=topic if i % in_topic_every == 0 else "other",
                doc_type="markdown",
                section=f"Section {i}",
            )
            for i in range(n_chunks)
        ]
        self.hybrid_budgets: list[int] = []
        self.dense_budgets: list[int] = []
        self.hybrid_raises = False

    def iter_texts(self):
        for c in self.chunks:
            yield c.chunk_id, c.text

    def search_dense(self, vector, top_k):
        self.dense_budgets.append(top_k)
        return [(c, 1.0) for c in self.chunks[:top_k]]

    def search_hybrid(self, query_vector, query_sparse, top_k=20, dense_weight=0.5):
        self.hybrid_budgets.append(top_k)
        if self.hybrid_raises:
            raise RuntimeError("sparse slot not populated")
        return [(c, 1.0, 0.01) for c in self.chunks[:top_k]]


_QUERY = "exposure compensation passage"


def test_hybrid_topic_filter_still_returns_a_full_top_k(clean_hybrid_cache):
    """The post-filter ran AFTER retrieving top_k, so a topic-filtered
    hybrid query returned only the in-topic survivors of a top_k slice —
    while the dense path pushes the filter into the store and returns a
    full top_k. Over-fetching before the filter closes that gap."""
    from core import pipeline

    store = _TopicCorpusStore(n_chunks=200)
    hits = pipeline._retrieve_hybrid(
        _QUERY, [0.1, 0.2, 0.3, 0.4], store, top_k=5, topic="target")

    assert len(hits) == 5, f"expected a full top_k of in-topic hits, got {len(hits)}"
    assert all(c.topic == "target" for c, _ in hits)
    assert store.hybrid_budgets[0] > 5, "the filtered branch must over-fetch"


def test_hybrid_topic_filter_does_not_confound_the_top_k_dense_axis(clean_hybrid_cache):
    """THE confound: `top_k_dense` is a SWEPT AXIS. With a post-filter over
    a top_k slice, 40 beat 20 partly because more in-topic hits happened to
    survive the filter — nothing to do with retrieval quality. After the
    fix each budget yields its full budget of in-topic hits, so the axis
    measures ranking again."""
    from core import pipeline

    counts = {}
    for k in (20, 40):
        pipeline.invalidate_hybrid_cache()
        store = _TopicCorpusStore(n_chunks=600)
        hits = pipeline._retrieve_hybrid(
            _QUERY, [0.1, 0.2, 0.3, 0.4], store, top_k=k, topic="target")
        assert all(c.topic == "target" for c, _ in hits)
        counts[k] = len(hits)

    assert counts == {20: 20, 40: 40}


def test_hybrid_topic_filter_result_size_is_stable_as_the_pool_grows(clean_hybrid_cache):
    """top_k held constant, candidate pool grows 10x: the number of
    returned in-topic chunks must not move."""
    from core import pipeline

    sizes = []
    for n_chunks in (60, 600):
        pipeline.invalidate_hybrid_cache()
        store = _TopicCorpusStore(n_chunks=n_chunks)
        hits = pipeline._retrieve_hybrid(
            _QUERY, [0.1, 0.2, 0.3, 0.4], store, top_k=5, topic="target")
        sizes.append(len(hits))

    assert sizes == [5, 5], f"result size moved with the pool size: {sizes}"


def test_hybrid_topic_filter_applies_to_the_dense_fallback(clean_hybrid_cache):
    """When search_hybrid raises, _retrieve_hybrid falls back to dense.
    That fallback must over-fetch AND honour the topic — it used to return
    an unfiltered top_k slice, i.e. mostly out-of-topic chunks."""
    from core import pipeline

    store = _TopicCorpusStore(n_chunks=200)
    store.hybrid_raises = True
    hits = pipeline._retrieve_hybrid(
        _QUERY, [0.1, 0.2, 0.3, 0.4], store, top_k=5, topic="target")

    assert len(hits) == 5
    assert all(c.topic == "target" for c, _ in hits)
    assert store.dense_budgets[0] > 5, "the dense fallback must over-fetch too"


def test_hybrid_without_topic_does_not_over_fetch(clean_hybrid_cache):
    """No topic filter means nothing is discarded, so the retrieval budget
    must stay exactly top_k — over-fetching unconditionally would change
    what the reranker sees on every untopiced query."""
    from core import pipeline

    store = _TopicCorpusStore(n_chunks=200)
    hits = pipeline._retrieve_hybrid(
        _QUERY, [0.1, 0.2, 0.3, 0.4], store, top_k=5, topic=None)

    assert store.hybrid_budgets == [5]
    assert len(hits) == 5


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
