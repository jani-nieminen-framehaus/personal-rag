"""Tests for the Wave B structural fixes.

- The pipeline depends only on the VectorStore ABC: a from-scratch
  InMemoryStore can drive ingest() and ask() end-to-end without qdrant,
  and the concrete-store import can't silently return to pipeline.py.
- chunking_params() is the single source of chunking defaults.
- Zeal ingest isolates per-page parse/chunk failures.
- Markdown ingest truncates at the chunk cap instead of dropping the file.
"""
from __future__ import annotations

import inspect
import logging
import sqlite3
from pathlib import Path

import pytest

from core.interfaces import Chunk, VectorStore


# -- pipeline ↔ store decoupling ----------------------------------------------

def test_pipeline_has_no_concrete_store_import():
    """The audit's highest-leverage structural finding: pipeline.py imported
    QdrantStore directly. Keep it out."""
    import core.pipeline
    src = inspect.getsource(core.pipeline)
    assert "from store.qdrant_store" not in src
    assert "import store.qdrant_store" not in src


class InMemoryStore(VectorStore):
    """Minimal dense-only VectorStore — just enough to drive the pipeline."""

    def __init__(self):
        self.collection = "mem"
        self.points: dict[str, tuple[Chunk, list[float]]] = {}

    def ensure_collection(self, recreate=False, expected_dense_dim=None):
        if recreate:
            self.points.clear()

    def count(self):
        return len(self.points)

    def upsert_chunks(self, chunks, vectors):
        for c, v in zip(chunks, vectors):
            self.points[c.chunk_id] = (c, v)
        return len(chunks)

    def search_dense(self, vector, top_k):
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))
        scored = sorted(
            ((c, dot(vector, v)) for c, v in self.points.values()),
            key=lambda t: t[1], reverse=True,
        )
        return scored[:top_k]

    def search_with_filter(self, vector, top_k, topic=None):
        hits = self.search_dense(vector, top_k=max(len(self.points), 1))
        if topic:
            hits = [(c, s) for c, s in hits if c.topic == topic]
        return hits[:top_k]

    def iter_payloads(self):
        for c, _v in self.points.values():
            yield c.chunk_id, c.to_payload()


class _TinyEmbedder:
    batch_size = 8

    def dim(self):
        return 3

    def embed(self, texts):
        # Deterministic toy vectors: length-ish features, no ML anywhere.
        return [[1.0, float(len(t) % 5), float(t.count("a") % 3)] for t in texts]

    def embed_documents(self, texts):
        return self.embed(texts)

    def embed_query(self, text):
        return self.embed([text])[0]


class _EchoGenerator:
    def generate(self, prompt):
        return "answer based on sources [1]"


def _mk_chunk(cid: str, text: str, topic: str = "test") -> Chunk:
    return Chunk(
        chunk_id=cid, parent_id=f"p-{cid}", text=text,
        source_path=f"/src/{cid}.md", topic=topic,
        doc_type="markdown", section=f"S {cid}",
    )


def test_pipeline_runs_end_to_end_on_custom_store():
    """A store that is *only* the ABC surface can serve the whole pipeline:
    swap-ability is real, not aspirational."""
    from core import pipeline
    from core.pipeline import PassthroughReranker

    class TwoChunkIngester:
        def iter_chunks(self):
            yield _mk_chunk("c1", "alpha notes about cameras")
            yield _mk_chunk("c2", "beta notes about lenses")

    store = InMemoryStore()
    embedder = _TinyEmbedder()

    n = pipeline.ingest(TwoChunkIngester(), embedder, store)
    assert n == 2
    assert store.count() == 2

    result = pipeline.ask(
        "what about cameras?",
        embedder=embedder,
        store=store,
        reranker=PassthroughReranker(),
        generator=_EchoGenerator(),
        top_k_dense=5,
        top_k_final=2,
    )
    assert result.answer == "answer based on sources [1]"
    assert result.citations, "citations must come from the custom store"
    assert {c["chunk_id"] for c in result.citations} <= {"c1", "c2"}


def test_pipeline_hybrid_falls_back_to_dense_on_dense_only_store():
    """VectorStore.search_hybrid raises by default; the pipeline must catch
    it and still answer from the dense branch."""
    from core import pipeline
    from core.pipeline import PassthroughReranker

    class OneChunkIngester:
        def iter_chunks(self):
            yield _mk_chunk("c1", "alpha notes about cameras and aperture")

    store = InMemoryStore()
    embedder = _TinyEmbedder()
    pipeline.ingest(OneChunkIngester(), embedder, store)
    pipeline.invalidate_hybrid_cache()

    result = pipeline.ask(
        "cameras aperture alpha",
        embedder=embedder,
        store=store,
        reranker=PassthroughReranker(),
        generator=_EchoGenerator(),
        hybrid=True,
    )
    assert result.citations, "hybrid on a dense-only store must fall back, not fail"


# -- chunking_params -----------------------------------------------------------

def test_chunking_params_defaults():
    from core.pipeline import chunking_params

    cp = chunking_params({})
    assert cp == {
        "target_tokens": 768,
        "overlap_pct": 12,
        "min_chunk_tokens": 32,
        "default_topic": "default",
        "max_chunks_per_doc": 2000,
    }


def test_chunking_params_overrides():
    from core.pipeline import chunking_params

    cp = chunking_params({
        "chunking": {"target_tokens": 256, "max_chunks_per_doc": 10},
        "ingest": {"default_topic": "photo"},
    })
    assert cp["target_tokens"] == 256
    assert cp["max_chunks_per_doc"] == 10
    assert cp["default_topic"] == "photo"
    assert cp["overlap_pct"] == 12  # untouched default


# -- zeal: per-page isolation --------------------------------------------------

def _make_two_page_docset(root: Path) -> None:
    docs = root / "Contents" / "Resources" / "Documents"
    docs.mkdir(parents=True)
    (docs / "page_a.html").write_text(
        "<html><body><h1>Page A</h1><p>alpha content for testing pages</p></body></html>",
        encoding="utf-8",
    )
    (docs / "page_b.html").write_text(
        "<html><body><h1>Page B</h1><p>beta content for testing pages</p></body></html>",
        encoding="utf-8",
    )
    idx = root / "Contents" / "Resources" / "docSet.dsidx"
    with sqlite3.connect(str(idx)) as conn:
        conn.execute("CREATE TABLE searchIndex (id INTEGER, name TEXT, type TEXT, path TEXT)")
        conn.executemany(
            "INSERT INTO searchIndex VALUES (?, ?, ?, ?)",
            [(1, "Page A", "Page", "page_a.html"), (2, "Page B", "Page", "page_b.html")],
        )
        conn.commit()


def test_zeal_one_bad_page_does_not_kill_the_docset(tmp_path, monkeypatch, caplog):
    """A docset has ~50k pages; one broken page must be skipped, not fatal."""
    import ingest.zeal_docsets as zd

    root = tmp_path / "Test.docset"
    root.mkdir()
    _make_two_page_docset(root)

    real_html_to_text = zd._html_to_text

    def exploding(html):
        if "beta" in html:
            raise RuntimeError("malformed page")
        return real_html_to_text(html)

    monkeypatch.setattr(zd, "_html_to_text", exploding)

    zi = zd.ZealIngester(
        docset_path=root,
        target_tokens=768, overlap_pct=12, min_chunk_tokens=1,
        default_topic="default",
    )
    with caplog.at_level(logging.WARNING):
        chunks = list(zi.iter_chunks())  # must not raise

    assert chunks, "the healthy page must still be ingested"
    assert all("page_a" in c.source_path for c in chunks)
    assert any("skipping page" in r.message for r in caplog.records)


# -- markdown: truncate, not drop ---------------------------------------------

def test_markdown_truncates_at_cap_instead_of_dropping(tmp_path, caplog):
    from ingest.markdown_dir import MarkdownDirIngester

    body = "\n\n".join(
        f"## Section {i}\n\n" + ("word " * 40) for i in range(6)
    )
    (tmp_path / "big.md").write_text("# Big\n\n" + body, encoding="utf-8")

    mi = MarkdownDirIngester(
        root=tmp_path,
        target_tokens=16, overlap_pct=0, min_chunk_tokens=1,
        default_topic="t", max_chunks_per_doc=2,
    )
    with caplog.at_level(logging.WARNING):
        chunks = list(mi.iter_chunks())

    assert len(chunks) == 2, "must yield exactly the cap, not zero"
    assert any("truncating" in r.message for r in caplog.records)
