"""Pytest config: sys.path shim + shared fixtures.

The sys.path insert stays even though pyproject.toml sets pythonpath —
it keeps direct `python tests/test_x.py` runs working outside pytest.
"""
from __future__ import annotations

import shutil
import sys
import time
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def serve_state():
    """Hand out serve.S for mutation and restore ready/embedder/store after
    the test — replaces the hand-rolled try/finally blocks that made the
    server tests order-dependent."""
    import serve

    saved = (serve.S.ready, serve.S.embedder, serve.S.store)
    yield serve.S
    serve.S.ready, serve.S.embedder, serve.S.store = saved


@pytest.fixture
def rmtree_retry():
    """Windows can hold a handle on files (SQLite DBs especially) briefly
    after close, making a bare rmtree flaky. Retry, then best-effort."""
    def _rm(path, attempts: int = 5, delay: float = 0.1) -> None:
        for _ in range(attempts):
            try:
                shutil.rmtree(path)
                return
            except OSError:
                time.sleep(delay)
        shutil.rmtree(path, ignore_errors=True)
    return _rm


def build_epub(
    path: Path,
    chapters: list[dict],
    title: str = "Test Book",
) -> None:
    """Create a minimal valid EPUB at `path` with one or more chapters.

    Args:
        path: destination .epub file path.
        chapters: list of dicts, each with keys `id` (str), `title` (str),
                  and `body` (str).  These become OEBPS/chN.xhtml files.

    Uses zipfile + XML strings to hand-write a valid EPUB. Does not call
    ebooklib's writer API, which raised AttributeError on the (nonexistent)
    EpubAuthor class.
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
