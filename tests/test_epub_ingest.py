"""Unit tests for the EPUB ingester.

We build a tiny test EPUB in tmp_path using Python's zipfile + XML strings,
so the test is fully self-contained — no external fixture files, no internet.
ebooklib is an optional dep (skipped if not installed).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

# ebooklib is optional.
pytest.importorskip("ebooklib")

from ingest.epub_dir import EpubDirIngester  # noqa: E402  (import after skip)


# ---- helpers ---------------------------------------------------------------

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

    The EPUB uses the simplest possible OPF manifest.  Items are declared
    in the spine in chapter order so ebooklib returns them that way.
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


def _ingester(path: Path, **kwargs) -> EpubDirIngester:
    return EpubDirIngester(
        path=path,
        target_tokens=kwargs.get("target_tokens", 768),
        overlap_pct=kwargs.get("overlap_pct", 12),
        min_chunk_tokens=kwargs.get("min_chunk_tokens", 32),
        default_topic=kwargs.get("default_topic", "default"),
    )


# ---- single-file ingest ---------------------------------------------------

def test_ingest_single_epub_yields_one_chunk_per_chapter(tmp_path: Path):
    epub = tmp_path / "book.epub"
    _build_epub(
        epub,
        [
            {"id": "ch1", "title": "Chapter One", "body": "First chapter content."},
            {"id": "ch2", "title": "Chapter Two", "body": "Second chapter content."},
        ],
    )
    chunks = list(_ingester(epub).iter_chunks())
    assert len(chunks) == 2, f"expected 2 chunks, got {len(chunks)}"
    assert chunks[0].doc_type == "epub"
    assert chunks[0].section == "Chapter One"
    assert chunks[1].section == "Chapter Two"
    assert chunks[0].source_path == str(epub)
    assert chunks[0].topic == "default"  # single-file → default_topic


def test_ingest_single_epub_chunk_id_is_deterministic(tmp_path: Path):
    epub = tmp_path / "book.epub"
    _build_epub(epub, [{"id": "ch1", "title": "First", "body": "Body text."}])
    a = [c.chunk_id for c in _ingester(epub).iter_chunks()]
    b = [c.chunk_id for c in _ingester(epub).iter_chunks()]
    assert a == b, "chunk_ids must be deterministic"


def test_ingest_single_epub_id_path_is_filename(tmp_path: Path):
    """Single-file ingest uses the filename as id_path.  This means
    alpha.epub and beta.epub (different filenames) produce different
    chunk_ids even with identical content — correct behaviour for
    filename-based deduplication.

    The determinism guarantee (same file → same chunk_ids) is tested
    separately; this test confirms the id_path is NOT the full
    absolute path."""
    epub_a = tmp_path / "alpha.epub"
    epub_b = tmp_path / "beta.epub"
    _build_epub(epub_a, [{"id": "ch1", "title": "Chapter", "body": "Same body text here."}])
    _build_epub(epub_b, [{"id": "ch1", "title": "Chapter", "body": "Same body text here."}])

    chunks_a = [c.chunk_id for c in _ingester(epub_a).iter_chunks()]
    chunks_b = [c.chunk_id for c in _ingester(epub_b).iter_chunks()]
    # Different filenames → different chunk_ids.
    assert chunks_a != chunks_b
    # source_path must be the absolute path.
    chunks_a_list = list(_ingester(epub_a).iter_chunks())
    assert "alpha.epub" in chunks_a_list[0].source_path


def test_ingest_single_epub_stores_chapter_metadata_in_extra(tmp_path: Path):
    epub = tmp_path / "book.epub"
    _build_epub(
        epub,
        [
            {"id": "ch1", "title": "Intro", "body": "Short intro."},
            {"id": "ch2", "title": "Main", "body": "Long main body."},
        ],
    )
    chunks = list(_ingester(epub).iter_chunks())
    assert chunks[0].extra["chapter_index"] == 0
    assert chunks[0].extra["chapter_title"] == "Intro"
    assert chunks[1].extra["chapter_index"] == 1
    assert chunks[1].extra["chapter_title"] == "Main"


# ---- directory ingest -----------------------------------------------------

def test_ingest_directory_walks_recursively(tmp_path: Path):
    lib = tmp_path / "library"
    lib.mkdir(parents=True)
    (lib / "fiction").mkdir()
    _build_epub(
        lib / "fiction" / "novel.epub",
        [{"id": "ch1", "title": "Chapter One", "body": "The story begins here."}],
    )
    (lib / "tech").mkdir()
    _build_epub(
        lib / "tech" / "guide.epub",
        [{"id": "ch1", "title": "Guide", "body": "Instructions for the guide."}],
    )
    chunks = list(_ingester(lib).iter_chunks())
    assert len(chunks) == 2
    by_name = {Path(c.source_path).name: c for c in chunks}
    assert by_name["novel.epub"].topic == "fiction"
    assert by_name["guide.epub"].topic == "tech"


def test_ingest_directory_topic_uses_immediate_parent(tmp_path: Path):
    lib = tmp_path / "lib"
    lib.mkdir(parents=True)
    nested = lib / "a" / "b" / "c"
    nested.mkdir(parents=True)
    _build_epub(
        nested / "deep.epub",
        [{"id": "ch1", "title": "Deep", "body": "This is the deep chapter."}],
    )
    chunks = list(_ingester(lib).iter_chunks())
    assert len(chunks) == 1
    assert chunks[0].topic == "c", f"expected 'c', got {chunks[0].topic!r}"


def test_ingest_directory_topic_root_level_file_uses_default(tmp_path: Path):
    lib = tmp_path / "lib"
    lib.mkdir()
    _build_epub(
        lib / "lone.epub",
        [{"id": "ch1", "title": "Lone", "body": "This is the lone chapter."}],
    )
    chunks = list(_ingester(lib, default_topic="books").iter_chunks())
    assert chunks[0].topic == "books"


def test_ingest_directory_skips_hidden_dirs(tmp_path: Path):
    lib = tmp_path / "library"
    hidden = lib / ".secret"
    hidden.mkdir(parents=True)
    _build_epub(
        hidden / "skip.epub",
        [{"id": "ch1", "title": "Skip", "body": "This chapter is skipped."}],
    )
    (lib / "public").mkdir()
    _build_epub(
        lib / "public" / "keep.epub",
        [{"id": "ch1", "title": "Keep", "body": "This chapter should be indexed."}],
    )
    chunks = list(_ingester(lib).iter_chunks())
    names = {Path(c.source_path).name for c in chunks}
    assert "keep.epub" in names
    assert "skip.epub" not in names


def test_ingest_directory_skips_non_epub_files(tmp_path: Path):
    lib = tmp_path / "lib"
    lib.mkdir()
    _build_epub(
        lib / "real.epub",
        [{"id": "ch1", "title": "Real", "body": "This is the real content."}],
    )
    (lib / "notes.txt").write_text("not an epub", encoding="utf-8")
    chunks = list(_ingester(lib).iter_chunks())
    assert len(chunks) == 1
    assert Path(chunks[0].source_path).name == "real.epub"


# ---- long chapters get split -----------------------------------------------

def test_long_chapter_is_split_into_multiple_chunks(tmp_path: Path):
    epub = tmp_path / "long.epub"
    # ~2000 tokens of repeated text — well above target=128.
    long_body = ("This is a sentence of test text. " * 200).strip()
    _build_epub(
        epub,
        [{"id": "ch1", "title": "Long Chapter", "body": long_body}],
    )
    chunks = list(
        _ingester(epub, target_tokens=128, overlap_pct=12, min_chunk_tokens=32).iter_chunks()
    )
    # Should produce more than 1 chunk from the single long chapter.
    assert len(chunks) > 1, f"expected splitting, got {len(chunks)} chunk(s)"
    # All share the same section (chapter title).
    assert all(c.section == "Long Chapter" for c in chunks)
    # Chunk IDs are unique.
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)


# ---- constructor validation -----------------------------------------------

def test_constructor_rejects_nonexistent_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _ingester(tmp_path / "nope.epub")


def test_constructor_rejects_non_epub_file(tmp_path: Path):
    f = tmp_path / "notes.txt"
    f.write_text("not an epub", encoding="utf-8")
    with pytest.raises(ValueError, match="not an .epub"):
        _ingester(f)


def test_constructor_accepts_directory_with_no_epubs(tmp_path: Path):
    """An empty dir should not crash on construction; iter_chunks() yields nothing."""
    empty = tmp_path / "empty"
    empty.mkdir()
    chunks = list(_ingester(empty).iter_chunks())
    assert chunks == []


# ---- bad EPUB is skipped, not fatal --------------------------------------

def test_corrupt_epub_is_skipped_not_fatal(tmp_path: Path, caplog):
    """A malformed ZIP in a directory should log a warning, not kill ingest."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "broken.epub").write_bytes(b"not a valid epub zip at all")
    _build_epub(
        lib / "good.epub",
        [{"id": "ch1", "title": "Good", "body": "Real content."}],
    )
    chunks = list(_ingester(lib).iter_chunks())
    assert len(chunks) == 1
    assert Path(chunks[0].source_path).name == "good.epub"
