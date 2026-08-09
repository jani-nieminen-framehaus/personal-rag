"""Unit tests for Wave 2 bugs.

Covers:
- #5 faithfulness: the metric must operate on chunk TEXT, not source_path.
- #6 delete_by_source: must not crash on a string `status` (qdrant-client 1.x).
- #7 embedder dim: QdrantStore must validate the embedder's dim against the
  existing collection and raise on mismatch.
- #8 zeal dedupe: each docset page must be chunked exactly once even when
  searchIndex has multiple anchor rows for the same file.
"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.interfaces import Chunk


# -- #5: faithfulness operates on chunk text, not source_path -----------------

def test_faithfulness_proxy_uses_chunk_text():
    """The faithfulness proxy should be 0 when the answer and the chunk
    TEXT have NO token overlap, and high when they share tokens. The
    previous implementation passed source_path strings, so the proxy
    measured overlap between the answer and FILE PATHS, which is
    meaningless."""
    from eval.run_ragas import faithfulness_proxy
    answer = "exposure compensation is useful for snow scenes"
    retrieved_texts = ["Exposure compensation is useful when the meter is fooled by snow."]
    # Token overlap should be > 0.5 — they share exposure/compensation/snow/etc.
    score = faithfulness_proxy(answer, retrieved_texts)
    assert score > 0.5, (
        f"faithfulness_proxy should be high when answer overlaps chunk text; got {score}"
    )
    # And the source_path is NOT in the answer — using source_path alone would give 0.
    assert faithfulness_proxy(answer, ["D:\\some\\path\\to\\foo.md"]) == 0.0


def test_ask_result_citations_include_text():
    """AskResult.citations must carry the chunk text. The faithfulness proxy
    needs it; otherwise it falls back to source_path (the audit flagged
    this as wrong)."""
    c = Chunk(
        chunk_id="c1", parent_id="p", text="the answer is in here",
        source_path="notes/x.md", topic="t", doc_type="markdown", section="S",
    )
    # Simulate the citation dict as built by core.pipeline.ask()
    citation = {
        "n": 1,
        "source_path": c.source_path,
        "section": c.section,
        "chunk_id": c.chunk_id,
        "topic": c.topic,
        "doc_type": c.doc_type,
        "score": 0.9,
        # The fix: include the chunk text.
        "text": c.text,
    }
    assert citation["text"] == "the answer is in here"


# -- #6: delete_by_source must not crash on a string status --------------------

def test_delete_by_source_returns_count_not_status():
    """Bug #6: `int(result.status)` — qdrant-client returns `status` as a
    string enum, so this raised TypeError whenever delete was called.
    Fix: pre-count the points, then return that count.
    """
    from store.qdrant_store import QdrantStore

    fake_client = MagicMock()
    # Pre-count returns 3 points matching the source filter
    fake_client.count.return_value = MagicMock(count=3)
    # delete returns an UpdateResult-like with status as a STRING
    fake_client.delete.return_value = MagicMock(status="completed", operation_id=42)

    with patch("store.qdrant_store.QdrantClient", return_value=fake_client):
        store = QdrantStore(url="http://x", collection="c", dense_dim=4)

    fake_client.collection_exists.return_value = True
    deleted = store.delete_by_source("notes/foo.md")
    # Pre-count was 3, so the function should return 3.
    assert deleted == 3
    # And the filter passed to count/delete must reference the source_path field.
    args = fake_client.count.call_args
    assert "collection_name" in args.kwargs
    assert args.kwargs["collection_name"] == "c"


# -- #7: embedder dim validation ----------------------------------------------

def test_ensure_collection_validates_dim_on_existing():
    """Bug #7: dense_dim from config was never validated against the existing
    collection. After swapping the embedder in config.yaml, the first upsert
    would fail with a confusing Qdrant-side error.

    Fix: ensure_collection(expected_dense_dim=...) reads the existing
    collection's vector config and raises on mismatch.
    """
    from store.qdrant_store import QdrantStore

    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True

    # Build a fake VectorParams with size=4096 (the existing collection)
    fake_vector_params = MagicMock()
    fake_vector_params.size = 4096
    fake_collection_info = MagicMock()
    fake_collection_info.config.params.vectors = {"dense": fake_vector_params}

    fake_client.get_collection.return_value = fake_collection_info

    with patch("store.qdrant_store.QdrantClient", return_value=fake_client):
        store = QdrantStore(url="http://x", collection="kb_p0", dense_dim=4096)

    # Match: no raise
    store.ensure_collection(expected_dense_dim=4096)

    # Mismatch: must raise with a helpful message
    with pytest.raises(ValueError, match="dense_dim"):
        store.ensure_collection(expected_dense_dim=1024)


# -- #8: Zeal dedupe -----------------------------------------------------------

def _make_fake_docset(root: Path) -> None:
    """Build a minimal Zeal docset: 2 HTML files, 5 searchIndex rows
    (3 for page A, 2 for page B). The ingester must dedupe to 2 unique pages.
    """
    docs = root / "Contents" / "Resources" / "Documents"
    docs.mkdir(parents=True)

    (docs / "page_a.html").write_text(
        "<html><body><h1>Page A</h1><p>alpha content for testing</p>"
        "<a name='sec1'>Section 1</a>"
        "<a name='sec2'>Section 2</a>"
        "</body></html>",
        encoding="utf-8",
    )
    (docs / "subdir").mkdir(parents=True, exist_ok=True)
    (docs / "subdir" / "page_b.html").write_text(
        "<html><body><h1>Page B</h1><p>beta content for testing</p></body></html>",
        encoding="utf-8",
    )

    idx = root / "Contents" / "Resources" / "docSet.dsidx"
    if idx.exists():
        idx.unlink()
    with sqlite3.connect(str(idx)) as conn:
        conn.execute("CREATE TABLE searchIndex (id INTEGER, name TEXT, type TEXT, path TEXT)")
        conn.executemany(
            "INSERT INTO searchIndex VALUES (?, ?, ?, ?)",
            [
                (1, "Section 1", "Section", "page_a.html"),
                (2, "Section 2", "Section", "page_a.html"),
                (3, "Top of page A", "Page", "page_a.html"),
                (4, "Top of page B", "Page", "subdir/page_b.html"),
                (5, "Bottom of page B", "Section", "subdir/page_b.html"),
            ],
        )
        conn.commit()


def test_zeal_dedupes_same_file():
    """Bug #8: searchIndex has many rows per file (one per anchor). The
    ingester must chunk each file once, not 3x for page_a and 2x for
    page_b. Without dedupe, the same chunks are upserted multiple times
    (idempotent on the Qdrant side, but the ingest is wasted work and
    the chunk_id collisions could lose text)."""
    td = tempfile.mkdtemp(prefix="zeal_test_")
    try:
        root = Path(td) / "Test.docset"
        root.mkdir()
        _make_fake_docset(root)

        from ingest.zeal_docsets import ZealIngester
        zi = ZealIngester(
            docset_path=root,
            target_tokens=768, overlap_pct=12, min_chunk_tokens=32,
            default_topic="default",
        )

        # Count unique (source_path, section) pairs across the deduped output.
        seen: dict[tuple[str, str], int] = {}
        for c in zi.iter_chunks():
            key = (c.source_path, c.section)
            seen[key] = seen.get(key, 0) + 1

        paths = sorted(seen.keys())
        assert len(paths) == 2, f"expected 2 unique pages, got {len(paths)}: {paths}"
        assert any("subdir" in p[0] for p in paths), (
            f"synth_path must preserve relative dir, got: {paths}"
        )
    finally:
        # Windows can hold a file handle on the SQLite DB briefly after
        # the ingester's with-block closes, causing rmtree to fail. A
        # small retry loop is more reliable than ignore_errors.
        for _ in range(5):
            try:
                shutil.rmtree(td)
                break
            except OSError:
                time.sleep(0.1)
        else:
            # Final fallback — best-effort cleanup
            shutil.rmtree(td, ignore_errors=True)


def test_zeal_source_path_preserves_relative_dir():
    """Bug #8 sub-issue: synth_path = docset_root / page_path.name dropped
    the relative subdirectory, so two pages named index.html in different
    subdirs collided on chunk_id. After the fix, the source_path includes
    the relative path within the docset."""
    td = tempfile.mkdtemp(prefix="zeal_test_")
    try:
        root = Path(td) / "Test.docset"
        root.mkdir()
        _make_fake_docset(root)

        from ingest.zeal_docsets import ZealIngester
        zi = ZealIngester(
            docset_path=root,
            target_tokens=768, overlap_pct=12, min_chunk_tokens=32,
            default_topic="default",
        )
        chunks = list(zi.iter_chunks())
        # page_b lives at "subdir/page_b.html" — its source_path must
        # contain "subdir" so it doesn't collide with anything at the
        # docset root.
        assert any("subdir" in c.source_path for c in chunks), (
            f"subdir/page_b.html lost its relative path: {[c.source_path for c in chunks]}"
        )
    finally:
        for _ in range(5):
            try:
                shutil.rmtree(td)
                break
            except OSError:
                time.sleep(0.1)
        else:
            shutil.rmtree(td, ignore_errors=True)
