"""Ingesters can be restricted to a subset of files, so refresh can
re-ingest just what changed instead of the whole tree.

Note: This module guards optional-dependency skips (pytest.importorskip) INSIDE
each test function rather than at module level. This is deliberate — unlike
test_pdf_ingest.py, the markdown tests have NO optional dependency, and a
module-level skip would prevent them from running. Test functions that need
optional deps call importorskip themselves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add tests dir to path so we can import conftest
sys.path.insert(0, str(Path(__file__).parent))

# Import build_epub from conftest (pytest special module handling)
import conftest as conftest_module
build_epub = conftest_module.build_epub

from ingest.markdown_dir import MarkdownDirIngester

CH = dict(target_tokens=768, overlap_pct=12, min_chunk_tokens=8,
          default_topic="test")


def _notes(tmp_path):
    (tmp_path / "keep.md").write_text("# Keep\n\nKeep body text here.\n", encoding="utf-8")
    (tmp_path / "skip.md").write_text("# Skip\n\nSkip body text here.\n", encoding="utf-8")
    return tmp_path


def test_markdown_without_only_paths_all_files_are_ingested(tmp_path):
    root = _notes(tmp_path)
    got = {c.source_path for c in MarkdownDirIngester(root=root, **CH).iter_chunks()}
    assert len(got) == 2


def test_markdown_only_paths_restricts_to_the_named_file(tmp_path):
    root = _notes(tmp_path)
    ing = MarkdownDirIngester(root=root, only_paths={root / "keep.md"}, **CH)
    got = {c.source_path for c in ing.iter_chunks()}
    assert len(got) == 1
    assert got.pop().endswith("keep.md")


def test_markdown_empty_only_paths_ingests_nothing(tmp_path):
    """An empty set means 'no files', not 'no filter' — refresh relies on
    this distinction when a source has no changes."""
    root = _notes(tmp_path)
    ing = MarkdownDirIngester(root=root, only_paths=set(), **CH)
    assert list(ing.iter_chunks()) == []


# ---- PDF ingester tests ---------------------------------------------------

def _build_pdf(path: Path, pages: list[str], title: str = "Test Book") -> None:
    """Create a minimal PDF at `path` with one page per element in `pages`."""
    import pymupdf
    doc = pymupdf.open()
    try:
        for body in pages:
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


def _pdfs(tmp_path):
    _build_pdf(tmp_path / "keep.pdf", ["Keep page text."])
    _build_pdf(tmp_path / "skip.pdf", ["Skip page text."])
    return tmp_path


def test_pdf_without_only_paths_all_files_are_ingested(tmp_path):
    pytest.importorskip("pymupdf")
    from ingest.pdf_dir import PdfDirIngester
    root = _pdfs(tmp_path)
    got = {c.source_path for c in PdfDirIngester(path=root, **CH).iter_chunks()}
    assert len(got) == 2


def test_pdf_only_paths_restricts_to_the_named_file(tmp_path):
    pytest.importorskip("pymupdf")
    from ingest.pdf_dir import PdfDirIngester
    root = _pdfs(tmp_path)
    ing = PdfDirIngester(path=root, only_paths={root / "keep.pdf"}, **CH)
    got = {c.source_path for c in ing.iter_chunks()}
    assert len(got) == 1
    assert got.pop().endswith("keep.pdf")


def test_pdf_empty_only_paths_ingests_nothing(tmp_path):
    """An empty set means 'no files', not 'no filter' — refresh relies on
    this distinction when a source has no changes."""
    pytest.importorskip("pymupdf")
    from ingest.pdf_dir import PdfDirIngester
    root = _pdfs(tmp_path)
    ing = PdfDirIngester(path=root, only_paths=set(), **CH)
    assert list(ing.iter_chunks()) == []


# ---- EPUB ingester tests --------------------------------------------------

def _epubs(tmp_path):
    build_epub(
        tmp_path / "keep.epub",
        [{"id": "ch1", "title": "Keep", "body": "Keep chapter text."}],
    )
    build_epub(
        tmp_path / "skip.epub",
        [{"id": "ch1", "title": "Skip", "body": "Skip chapter text."}],
    )
    return tmp_path


def test_epub_without_only_paths_all_files_are_ingested(tmp_path):
    pytest.importorskip("ebooklib")
    from ingest.epub_dir import EpubDirIngester
    root = _epubs(tmp_path)
    got = {c.source_path for c in EpubDirIngester(path=root, **CH).iter_chunks()}
    assert len(got) == 2


def test_epub_only_paths_restricts_to_the_named_file(tmp_path):
    pytest.importorskip("ebooklib")
    from ingest.epub_dir import EpubDirIngester
    root = _epubs(tmp_path)
    ing = EpubDirIngester(path=root, only_paths={root / "keep.epub"}, **CH)
    got = {c.source_path for c in ing.iter_chunks()}
    assert len(got) == 1
    assert got.pop().endswith("keep.epub")


def test_epub_empty_only_paths_ingests_nothing(tmp_path):
    """An empty set means 'no files', not 'no filter' — refresh relies on
    this distinction when a source has no changes."""
    pytest.importorskip("ebooklib")
    from ingest.epub_dir import EpubDirIngester
    root = _epubs(tmp_path)
    ing = EpubDirIngester(path=root, only_paths=set(), **CH)
    assert list(ing.iter_chunks()) == []
