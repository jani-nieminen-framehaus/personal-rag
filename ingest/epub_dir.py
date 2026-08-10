"""EPUB ingester.

Walks a directory of `.epub` files (or a single EPUB), extracts the text
per chapter via `ebooklib`, and yields one or more Chunks per chapter.
Long chapters are split by token count using the same sliding-window strategy
as the markdown chunker (see `core.chunker._split_by_tokens`).

Topic resolution (mirrors `pdf_dir.py`):
  1. Parent directory name (relative to the ingester root), lowercased.
  2. `default_topic` from config.

Source path: absolute on the Chunk, but the id_path used for
chunk_id hashing is root-relative so the same logical file hashes
to the same id regardless of where the repo is cloned.

Doc type: `"epub"`. Section: chapter title (from EPUB NCX/Nav) or
`"Chapter N"` as a fallback. Extra payload: `{"chapter_index": N,
"chapter_title": "..."}`.

ebooklib is imported lazily so the rest of the system can run on a
machine that doesn't have it installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from core.chunker import _split_by_tokens, count_tokens
from core.interfaces import Chunk, Ingester, make_chunk_id, make_parent_id
from core.walk import iter_source_files


log = logging.getLogger(__name__)


class EpubDirIngester(Ingester):
    """Ingest one EPUB file or a directory tree of EPUBs.

    Args:
        path: a single `.epub` file OR a directory to walk recursively.
              If a file: that file is the only source. id_path is just
              the filename. Topic falls back to default_topic.
              If a directory: walked recursively; topic from each
              file's immediate parent dir.
        target_tokens / overlap_pct / min_chunk_tokens: chunk sizing.
            Same semantics as the markdown chunker.
        default_topic: fallback when no parent dir matches.
        max_chunks_per_doc: safety cap.
        skip_hidden: skip `.dotdir/*` style files.
    """

    def __init__(
        self,
        path: str | Path,
        target_tokens: int,
        overlap_pct: int,
        min_chunk_tokens: int,
        default_topic: str,
        max_chunks_per_doc: int = 2000,
        skip_hidden: bool = True,
        only_paths: set[Path] | None = None,
    ):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"epub path does not exist: {self.path}")
        self.is_file = self.path.is_file()
        self.is_dir = self.path.is_dir()
        if not (self.is_file or self.is_dir):
            raise ValueError(f"epub path is neither file nor directory: {self.path}")
        if self.is_file and self.path.suffix.lower() not in (".epub", ".EPUB"):
            raise ValueError(f"file is not an .epub: {self.path}")
        self.target_tokens = target_tokens
        self.overlap_pct = overlap_pct
        self.min_chunk_tokens = min_chunk_tokens
        self.default_topic = default_topic
        self.max_chunks_per_doc = max_chunks_per_doc
        self.skip_hidden = skip_hidden
        self.only_paths = only_paths

    @property
    def name(self) -> str:
        return f"epub_dir:{self.path}"

    def iter_chunks(self) -> Iterator[Chunk]:
        files = self._collect_files()
        log.info("epub: found %d file(s) under %s", len(files), self.path)
        for file in files:
            try:
                yield from self._iter_one_epub(file)
            except Exception as e:
                # Don't let one bad EPUB kill the whole ingest.
                log.warning("epub: failed to read %s: %s — skipping", file, e)

    # -- helpers --------------------------------------------------------------

    def _collect_files(self) -> list[Path]:
        return iter_source_files(
            self.path, {".epub"},
            skip_hidden=self.skip_hidden, only_paths=self.only_paths,
        )

    def _iter_one_epub(self, file: Path) -> Iterator[Chunk]:
        # Lazy import so epub is an optional dep.
        try:
            from ebooklib import epub as epub_module  # type: ignore[import-not-found]
            import ebooklib  # type: ignore[import-not-found]
            from bs4 import BeautifulSoup  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "ebooklib is not installed. Run `pip install ebooklib beautifulsoup4` "
                "to enable EPUB ingest."
            ) from e

        # id_path for chunk_id hashing. Root-relative when ingesting
        # a directory, just the filename when ingesting a single file.
        if self.is_dir:
            try:
                id_path = file.relative_to(self.path).as_posix()
            except ValueError:
                id_path = file.name
        else:
            id_path = file.name

        topic = self._resolve_topic(file)
        parent_id = make_parent_id(id_path)

        book = epub_module.read_epub(str(file))

        # Collect chapters in spine order — the spine defines reading order
        # and is guaranteed to exist in every valid EPUB.
        # ebooklib's spine is a list of (idref, is_linear) tuples.
        chapters: list[tuple[int, str, str]] = []  # (idx, title, text)

        for item_ref in book.spine:
            if item_ref is None:
                continue
            # Spine items are (idref, is_linear) tuples.
            if isinstance(item_ref, tuple):
                item_id = item_ref[0]
            else:
                item_id = item_ref

            try:
                item_obj = book.get_item_with_id(item_id)
            except Exception:
                continue
            if item_obj is None:
                continue
            if item_obj.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            content = item_obj.get_content()
            if not content:
                continue
            soup = BeautifulSoup(content, "html.parser")
            # Extract title: try <title>, then <h1>, then None.
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else None
            if not title:
                h1 = soup.find("h1")
                if h1:
                    title = h1.get_text(strip=True)
            text = soup.get_text(separator=" ", strip=True)
            if not text or len(text) < 10:
                # Likely a nav or spacer page with no real content.
                continue
            # Clean up excessive whitespace.
            text = " ".join(text.split())
            chapters.append((len(chapters), title or "", text))

        log.info("epub: %s has %d chapter(s)", file, len(chapters))

        chunk_index = 0
        for chapter_idx, chapter_title, text in chapters:
            section = chapter_title or f"Chapter {chapter_idx + 1}"

            if count_tokens(text) <= self.target_tokens:
                pieces = [text]
            else:
                pieces = _split_by_tokens(
                    text, self.target_tokens, self.overlap_pct, self.min_chunk_tokens
                )

            for piece in pieces:
                if chunk_index >= self.max_chunks_per_doc:
                    log.warning(
                        "epub: %s exceeded max_chunks_per_doc=%d — truncating",
                        file, self.max_chunks_per_doc,
                    )
                    return
                cid = make_chunk_id(id_path, section, chunk_index)
                extra: dict = {
                    "chapter_index": chapter_idx,
                    "chapter_title": chapter_title,
                }
                yield Chunk(
                    chunk_id=cid,
                    parent_id=parent_id,
                    text=piece,
                    source_path=str(file),
                    topic=topic,
                    doc_type="epub",
                    section=section,
                    extra=extra,
                )
                chunk_index += 1

        log.debug("epub: %s yielded %d chunks across %d chapters", file, chunk_index, len(chapters))

    def _resolve_topic(self, file: Path) -> str:
        if not self.is_dir:
            return self.default_topic
        try:
            rel = file.relative_to(self.path)
            parts = rel.parts
            if len(parts) >= 2:
                return parts[-2].lower()
        except ValueError:
            pass
        return self.default_topic
