"""PDF ingester.

Walks a directory of `.pdf` files (or a single PDF), extracts the text
per page via pymupdf, and yields one or more Chunks per page. Long
pages are split by token count using the same sliding-window strategy
as the markdown chunker (see `core.chunker._split_by_tokens`).

Topic resolution (mirrors `markdown_dir.py`):
  1. Frontmatter — N/A, PDFs don't have one.
  2. Parent directory name (relative to the ingester root), lowercased.
  3. `default_topic` from config.

Source path: absolute on the Chunk, but the id_path used for
chunk_id hashing is root-relative so the same logical file hashes
to the same id regardless of where the repo is cloned.

Doc type: `"pdf"`. Section: `"Page N"` (or the PDF's own page label
if it has one — common in books with roman-numbered front matter).
Extra payload: `{"page": N, "page_label": "..."}` (label only
included when the PDF has one).

P1 note: this ingester does text extraction only. Scanned PDFs
(where `get_text()` returns empty) produce no chunks for that page
and a warning is logged. OCR via Tesseract can be layered on top
later — the chunker doesn't care how the text got into `page.text`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from core.chunker import _split_by_tokens, count_tokens
from core.interfaces import Chunk, Ingester, make_chunk_id, make_parent_id


log = logging.getLogger(__name__)


class PdfDirIngester(Ingester):
    """Ingest one PDF file or a directory tree of PDFs.

    Args:
        path: a single `.pdf` file OR a directory to walk recursively.
              If a file: that file is the only source. id_path is just
              the filename. Topic falls back to default_topic.
              If a directory: walked recursively; topic from each
              file's immediate parent dir (like MarkdownDirIngester).
        target_tokens / overlap_pct / min_chunk_tokens: chunk sizing.
            Same semantics as the markdown chunker.
        default_topic: fallback when no parent dir matches.
        max_chunks_per_doc: safety cap (PDFs can be huge; 5000 is
            generous — a 500-page book produces ~500 chunks).
        skip_hidden: skip `.dotdir/*` style files.
    """

    def __init__(
        self,
        path: str | Path,
        target_tokens: int,
        overlap_pct: int,
        min_chunk_tokens: int,
        default_topic: str,
        max_chunks_per_doc: int = 5000,
        skip_hidden: bool = True,
    ):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"pdf path does not exist: {self.path}")
        self.is_file = self.path.is_file()
        self.is_dir = self.path.is_dir()
        if not (self.is_file or self.is_dir):
            raise ValueError(f"pdf path is neither file nor directory: {self.path}")
        if self.is_file and self.path.suffix.lower() != ".pdf":
            raise ValueError(f"file is not a .pdf: {self.path}")
        self.target_tokens = target_tokens
        self.overlap_pct = overlap_pct
        self.min_chunk_tokens = min_chunk_tokens
        self.default_topic = default_topic
        self.max_chunks_per_doc = max_chunks_per_doc
        self.skip_hidden = skip_hidden

    @property
    def name(self) -> str:
        return f"pdf_dir:{self.path}"

    def iter_chunks(self) -> Iterator[Chunk]:
        files = self._collect_files()
        log.info("pdf: found %d file(s) under %s", len(files), self.path)
        for file in files:
            try:
                yield from self._iter_one_pdf(file)
            except Exception as e:
                # Don't let one bad PDF kill the whole ingest. Log + skip.
                log.warning("pdf: failed to read %s: %s — skipping", file, e)

    # -- helpers --------------------------------------------------------------

    def _collect_files(self) -> list[Path]:
        if self.is_file:
            return [self.path]
        out: list[Path] = []
        for p in self.path.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() != ".pdf":
                continue
            if self.skip_hidden and any(
                part.startswith(".") for part in p.relative_to(self.path).parts
            ):
                continue
            out.append(p)
        return sorted(out)

    def _iter_one_pdf(self, file: Path) -> Iterator[Chunk]:
        # pymupdf is imported lazily so the rest of the system can
        # run on a machine that doesn't have it installed. Modern
        # pymupdf prefers the `pymupdf` name; the legacy `fitz` shim
        # still works but emits a deprecation warning.
        try:
            import pymupdf  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "pymupdf is not installed. Run `pip install pymupdf` to enable PDF ingest."
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

        doc = pymupdf.open(str(file))
        try:
            chunk_index = 0
            page_count = len(doc)
            for page_num, page in enumerate(doc, start=1):
                text = (page.get_text("text") or "").strip()
                if not text:
                    # Scanned page or image-only page — nothing to embed.
                    log.debug("pdf: %s page %d is empty (no text layer)", file, page_num)
                    continue

                # Section: prefer the PDF's own page label (often
                # absent, but books sometimes use roman numerals or
                # chapter-prefixed labels). Fall back to "Page N".
                label = (page.get_label() or "").strip()
                section = f"Page {label}" if label else f"Page {page_num}"

                # Token-aware split for long pages.
                if count_tokens(text) <= self.target_tokens:
                    pieces = [text]
                else:
                    pieces = _split_by_tokens(
                        text, self.target_tokens, self.overlap_pct, self.min_chunk_tokens
                    )

                for piece in pieces:
                    if chunk_index >= self.max_chunks_per_doc:
                        log.warning(
                            "pdf: %s exceeded max_chunks_per_doc=%d — truncating",
                            file, self.max_chunks_per_doc,
                        )
                        return
                    cid = make_chunk_id(id_path, section, chunk_index)
                    extra: dict = {"page": page_num}
                    if label:
                        extra["page_label"] = label
                    yield Chunk(
                        chunk_id=cid,
                        parent_id=parent_id,
                        text=piece,
                        source_path=str(file),
                        topic=topic,
                        doc_type="pdf",
                        section=section,
                        extra=extra,
                    )
                    chunk_index += 1
            log.debug("pdf: %s yielded %d chunks across %d pages", file, chunk_index, page_count)
        finally:
            doc.close()

    def _resolve_topic(self, file: Path) -> str:
        # When ingesting a single file, there's no parent-dir tree to mine.
        if not self.is_dir:
            return self.default_topic
        # Same convention as MarkdownDirIngester: the IMMEDIATE parent
        # dir name, relative to the ingester root.
        try:
            rel = file.relative_to(self.path)
            parts = rel.parts
            if len(parts) >= 2:
                return parts[-2].lower()
        except ValueError:
            pass
        return self.default_topic
