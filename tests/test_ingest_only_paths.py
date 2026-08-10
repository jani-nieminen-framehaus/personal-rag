"""Ingesters can be restricted to a subset of files, so refresh can
re-ingest just what changed instead of the whole tree."""
from __future__ import annotations

from pathlib import Path

import pytest

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
    import pymupdf  # noqa: F401 (used via pymupdf.open below)
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
    (tmp_path / "keep.pdf").write_bytes(b"")  # placeholder
    _build_pdf(tmp_path / "keep.pdf", ["Keep page text."])
    (tmp_path / "skip.pdf").write_bytes(b"")  # placeholder
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

def _build_epub(path: Path, chapters: list[str], title: str = "Test Book") -> None:
    """Create a minimal EPUB at `path` with one chapter per element in `chapters`."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier(str(path))
    book.set_title(title)
    book.set_language('en')

    c1 = epub.EpubAuthor('Test Author')
    book.add_author(c1)

    for i, chapter_text in enumerate(chapters, start=1):
        c = epub.EpubHtml()
        c.file_name = f'chap_{i:02d}.xhtml'
        c.title = f'Chapter {i}'
        c.content = chapter_text
        book.add_item(c)
        book.toc.append(c)

    book.spine = ['nav'] + [item for item in book.items if isinstance(item, epub.EpubHtml)]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(path), book)


def _epubs(tmp_path):
    (tmp_path / "keep.epub").write_bytes(b"")  # placeholder
    _build_epub(tmp_path / "keep.epub", ["Keep chapter text."])
    (tmp_path / "skip.epub").write_bytes(b"")  # placeholder
    _build_epub(tmp_path / "skip.epub", ["Skip chapter text."])
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
