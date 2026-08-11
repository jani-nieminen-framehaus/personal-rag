"""Ingesters can be restricted to a subset of files, so refresh can
re-ingest just what changed instead of the whole tree.

Note: This module guards optional-dependency skips (pytest.importorskip) INSIDE
each test function rather than at module level. This is deliberate — unlike
test_pdf_ingest.py, the markdown tests have NO optional dependency, and a
module-level skip would prevent them from running. Test functions that need
optional deps call importorskip themselves.
"""
from __future__ import annotations

import zipfile
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

def _build_epub(
    path: Path,
    chapters: list[dict],
    title: str = "Test Book",
) -> None:
    """Create a minimal valid EPUB at `path` with one or more chapters.

    Args:
        path: destination .epub file path.
        chapters: list of dicts, each with keys `id` (str), `title` (str),
                  and `body` (str).  These become OEBPS/chN.xhtml files.

    Uses zipfile + XML strings to hand-write a valid EPUB, deliberately
    avoiding ebooklib's writer API which has structural bugs.
    """
    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. mimetype — must be first, uncompressed, no extra attrs.
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        # 2. META-INF/container.xml — tells ebooklib where the OPF is.
        container = """\
<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0"
           xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf"
              media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
        zf.writestr("META-INF/container.xml", container)

        # 3. OEBPS/content.opf (EPUB 2.0.1 — ebooklib reads this most reliably)
        manifest_items = "\n".join(
            f'    <item id="{c["id"]}" href="{c["id"]}.xhtml" '
            f'media-type="application/xhtml+xml"/>'
            for c in chapters
        )
        manifest_items += (
            f'\n    <item id="ncx" href="toc.ncx" '
            f'media-type="application/x-dtbncx+xml"/>'
        )
        spine_items = "\n".join(
            f'    <itemref idref="{c["id"]}"/>' for c in chapters
        )
        opf = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" '
            'unique-identifier="uid">\n'
            '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
            f'    <dc:title>{title}</dc:title>\n'
            '    <dc:language>en</dc:language>\n'
            '    <dc:identifier id="uid">urn:uuid:test-book-001</dc:identifier>\n'
            '  </metadata>\n'
            '  <manifest>\n'
            f'{manifest_items}\n'
            '  </manifest>\n'
            '  <spine toc="ncx">\n'
            f'{spine_items}\n'
            '  </spine>\n'
            '</package>'
        )
        zf.writestr("OEBPS/content.opf", opf)

        # 4. OEBPS/toc.ncx (required for EPUB 2.0.1)
        nav_points = "\n".join(
            f'  <navPoint id="np{i+1}" playOrder="{i+1}">'
            f'<navLabel><text>{c["title"]}</text></navLabel>'
            f'<content src="{c["id"]}.xhtml"/></navPoint>'
            for i, c in enumerate(chapters)
        )
        ncx = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">\n'
            '  <head>\n'
            '    <meta name="dtb:uid" content="urn:uuid:test-book-001"/>\n'
            '  </head>\n'
            f'  <docTitle><text>{title}</text></docTitle>\n'
            '  <navMap>\n'
            f'{nav_points}\n'
            '  </navMap>\n'
            '</ncx>'
        )
        zf.writestr("OEBPS/toc.ncx", ncx)

        # 5. OEBPS/chN.xhtml — one per chapter
        for c in chapters:
            body_escaped = (
                c["body"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            xhtml = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>{c["title"]}</title></head>
<body>
<h1>{c["title"]}</h1>
<p>{body_escaped}</p>
</body>
</html>"""
            zf.writestr(f"OEBPS/{c['id']}.xhtml", xhtml)


def _epubs(tmp_path):
    _build_epub(
        tmp_path / "keep.epub",
        [{"id": "ch1", "title": "Keep", "body": "Keep chapter text."}],
    )
    _build_epub(
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
