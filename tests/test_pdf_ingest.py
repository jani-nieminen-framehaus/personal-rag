"""Unit tests for the PDF ingester.

We build a tiny test PDF in tmp_path using pymupdf itself, so the
test is self-contained — no external fixture files, no internet.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# pymupdf is an optional dep (only needed for the PDF ingester).
pymupdf = pytest.importorskip("pymupdf")

from ingest.pdf_dir import PdfDirIngester  # noqa: E402  (import after skip)


# ---- helpers ---------------------------------------------------------------

def _build_pdf(path: Path, pages: list[str], title: str = "Test Book") -> None:
    """Create a minimal PDF at `path` with one page per element in `pages`.

    Uses pymupdf's document/page API directly — no LaTeX, no
    external tools. Pages use `insert_textbox` (wraps text in a
    rect) so we can produce arbitrarily long pages without
    pymupdf silently truncating at the page edge.
    """
    doc = pymupdf.open()
    try:
        for body in pages:
            # Make pages tall enough that even 200x repeated sentences fit.
            # insert_textbox wraps text inside the rect.
            page = doc.new_page(width=612, height=8000)
            page.insert_textbox(
                pymupdf.Rect(72, 72, 540, 7900),
                body,
                fontsize=11,
            )
        doc.set_metadata({"title": title})
        doc.save(str(path))
    finally:
        doc.close()


def _ingester(path: Path, **kwargs) -> PdfDirIngester:
    return PdfDirIngester(
        path=path,
        target_tokens=kwargs.get("target_tokens", 768),
        overlap_pct=kwargs.get("overlap_pct", 12),
        min_chunk_tokens=kwargs.get("min_chunk_tokens", 32),
        default_topic=kwargs.get("default_topic", "default"),
    )


# ---- single-file ingest ---------------------------------------------------

def test_ingest_single_pdf_yields_one_chunk_per_page(tmp_path: Path):
    pdf = tmp_path / "book.pdf"
    _build_pdf(pdf, ["First page text.", "Second page text.", "Third page text."])
    chunks = list(_ingester(pdf).iter_chunks())
    assert len(chunks) == 3, f"expected 3 chunks, got {len(chunks)}"
    # Each chunk has the expected metadata.
    for i, c in enumerate(chunks, start=1):
        assert c.doc_type == "pdf"
        assert c.section == f"Page {i}"
        assert c.source_path == str(pdf)
        assert c.topic == "default"  # single-file ingest → default_topic
        assert c.extra["page"] == i
        # The text should contain the page body.
        assert "page text" in c.text.lower()


def test_ingest_single_pdf_chunk_id_is_deterministic(tmp_path: Path):
    """Re-ingesting the same PDF should produce the same chunk_ids."""
    pdf = tmp_path / "book.pdf"
    _build_pdf(pdf, ["Page A.", "Page B."])
    ing = _ingester(pdf)
    a = [c.chunk_id for c in ing.iter_chunks()]
    b = [c.chunk_id for c in ing.iter_chunks()]
    assert a == b, "chunk_ids must be deterministic across runs"


def test_ingest_single_pdf_id_path_uses_filename_only(tmp_path: Path):
    """For single-file ingest, id_path is the filename — same id regardless of where you clone."""
    pdf = tmp_path / "alpha.pdf"
    _build_pdf(pdf, ["body"])
    chunks = list(_ingester(pdf).iter_chunks())
    # Same filename, different tmp_path → same chunk_id (id_path is just
    # the filename for single-file ingest, not the absolute path).
    other_dir = tmp_path / "other_location"
    other_dir.mkdir()
    other = other_dir / "alpha.pdf"
    _build_pdf(other, ["body"])
    other_chunks = list(_ingester(other).iter_chunks())
    assert chunks[0].chunk_id == other_chunks[0].chunk_id, (
        f"id_path should be filename-only for single-file ingest, "
        f"got {chunks[0].chunk_id} vs {other_chunks[0].chunk_id}"
    )


# ---- directory ingest -----------------------------------------------------

def test_ingest_directory_walks_recursively(tmp_path: Path):
    lib = tmp_path / "library"
    (lib / "photography").mkdir(parents=True)
    (lib / "photography" / "exposure.pdf").write_bytes(b"")  # placeholder
    _build_pdf(lib / "photography" / "exposure.pdf", ["On exposure."])
    (lib / "code").mkdir()
    _build_pdf(lib / "code" / "patterns.pdf", ["On patterns."])
    chunks = list(_ingester(lib).iter_chunks())
    assert len(chunks) == 2
    by_path = {Path(c.source_path).name: c for c in chunks}
    assert by_path["exposure.pdf"].topic == "photography"
    assert by_path["patterns.pdf"].topic == "code"


def test_ingest_directory_topic_uses_immediate_parent(tmp_path: Path):
    """Bug-safety: root/a/b/c/book.pdf should be topic 'c' (immediate parent), not 'a'."""
    # The file needs a subdir between the root and the file so the
    # immediate-parent rule actually fires. A file directly in root
    # has no parent and falls to default — that's a separate test.
    lib = tmp_path / "lib"
    lib.mkdir()
    nested = lib / "a" / "b" / "c"
    nested.mkdir(parents=True)
    _build_pdf(nested / "book.pdf", ["body"])
    chunks = list(_ingester(lib).iter_chunks())
    assert len(chunks) == 1
    assert chunks[0].topic == "c"


def test_ingest_directory_topic_root_level_file_uses_default(tmp_path: Path):
    """A PDF directly in the root has no parent — falls back to default_topic."""
    lib = tmp_path / "lib"
    lib.mkdir()
    _build_pdf(lib / "lone.pdf", ["body"])
    chunks = list(_ingester(lib, default_topic="books").iter_chunks())
    assert chunks[0].topic == "books"


def test_ingest_directory_skips_hidden_dirs(tmp_path: Path):
    """`.git/foo.pdf` should not be picked up."""
    lib = tmp_path / "library"
    hidden = lib / ".secret"
    hidden.mkdir(parents=True)
    _build_pdf(hidden / "skip.pdf", ["should not be indexed"])
    (lib / "photography").mkdir()
    _build_pdf(lib / "photography" / "keep.pdf", ["should be indexed"])
    chunks = list(_ingester(lib).iter_chunks())
    names = {Path(c.source_path).name for c in chunks}
    assert "keep.pdf" in names
    assert "skip.pdf" not in names


def test_ingest_directory_skips_non_pdf_files(tmp_path: Path):
    """Only .pdf is picked up; .txt etc. ignored."""
    lib = tmp_path / "lib"
    lib.mkdir()
    _build_pdf(lib / "real.pdf", ["body"])
    (lib / "notes.txt").write_text("not a pdf", encoding="utf-8")
    chunks = list(_ingester(lib).iter_chunks())
    assert len(chunks) == 1
    assert Path(chunks[0].source_path).name == "real.pdf"


# ---- long pages get split -------------------------------------------------

def test_long_page_is_split_into_multiple_chunks(tmp_path: Path):
    """A page whose text exceeds target_tokens should be split by the chunker.

    Each split piece keeps the same `section` and increments chunk_index
    so the ids remain deterministic.
    """
    pdf = tmp_path / "long.pdf"
    # ~2000 tokens of repeated text — well above target=128.
    long_body = ("This is a sentence of test text. " * 200).strip()
    _build_pdf(pdf, [long_body])
    chunks = list(_ingester(pdf, target_tokens=128, overlap_pct=12, min_chunk_tokens=32).iter_chunks())
    # Should produce more than 1 chunk from the single long page.
    assert len(chunks) > 1
    # All chunks from the same page share `section == "Page 1"`.
    assert all(c.section == "Page 1" for c in chunks)
    # Their chunk_ids are unique and deterministic.
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)


# ---- constructor validation ----------------------------------------------

def test_constructor_rejects_nonexistent_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _ingester(tmp_path / "nope.pdf")


def test_constructor_rejects_non_pdf_file(tmp_path: Path):
    f = tmp_path / "notes.txt"
    f.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="not a .pdf"):
        _ingester(f)


def test_constructor_accepts_directory_with_no_pdfs(tmp_path: Path):
    """An empty dir should not crash on construction; iter_chunks() yields nothing."""
    lib = tmp_path / "empty_lib"
    lib.mkdir()
    chunks = list(_ingester(lib).iter_chunks())
    assert chunks == []


# ---- bad PDF is skipped, not fatal ----------------------------------------

def test_corrupt_pdf_is_skipped_not_fatal(tmp_path: Path, caplog):
    """A malformed PDF in a directory should log a warning, not kill ingest."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "broken.pdf").write_bytes(b"this is not a real pdf at all")
    _build_pdf(lib / "good.pdf", ["real text"])
    chunks = list(_ingester(lib).iter_chunks())
    # The good one is indexed; the bad one is skipped.
    assert len(chunks) == 1
    assert Path(chunks[0].source_path).name == "good.pdf"
