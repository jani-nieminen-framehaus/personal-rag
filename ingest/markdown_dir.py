"""Ingest a directory of personal notes.

Walks the directory recursively, picks up `.md` / `.markdown` / `.py` files,
and yields `Chunk` objects via `core.chunker.chunk_file`. The actual chunking
strategy is decided in the chunker (heading-aware for Markdown, AST-based for
Python, whole-file fallback for anything else).

The class is named `MarkdownDirIngester` for historical reasons (the original
taxonomy); the P0 walk also picks up Python files so you can ingest a
mixed notes+code tree in one pass. PDFs and other formats will be their own
ingester in P1.

Topic resolution:
    1. YAML frontmatter `topic:` key (configurable; only applied to markdown)
    2. Parent directory name (relative to the ingester root)
    3. The configured default topic
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from core.chunker import chunk_file
from core.interfaces import Chunk, Ingester
from core.walk import iter_source_files


log = logging.getLogger(__name__)


class MarkdownDirIngester(Ingester):
    def __init__(
        self,
        root: str | Path,
        target_tokens: int,
        overlap_pct: int,
        min_chunk_tokens: int,
        default_topic: str,
        frontmatter_topic_key: str = "topic",
        max_chunks_per_doc: int = 2000,
        skip_hidden: bool = True,
        only_paths: set[Path] | None = None,
    ):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"markdown root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"markdown root is not a directory: {self.root}")
        self.target_tokens = target_tokens
        self.overlap_pct = overlap_pct
        self.min_chunk_tokens = min_chunk_tokens
        self.default_topic = default_topic
        self.frontmatter_topic_key = frontmatter_topic_key
        self.max_chunks_per_doc = max_chunks_per_doc
        self.skip_hidden = skip_hidden
        self.only_paths = only_paths

    @property
    def name(self) -> str:
        return f"markdown_dir:{self.root}"

    def iter_chunks(self) -> Iterator[Chunk]:
        files = sorted(self._walk())
        log.info("markdown: found %d files under %s", len(files), self.root)
        for path in files:
            # Root-relative, forward-slash path. Used for chunk_id so the
            # same logical file hashes to the same id regardless of where
            # the repo is cloned. Bug #1 from the audit.
            try:
                id_path = path.relative_to(self.root).as_posix()
            except ValueError:
                # File is outside the ingester root (shouldn't happen, but
                # fall back to the absolute path rather than crash).
                id_path = str(path)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                log.warning("cannot read %s: %s", path, e)
                continue
            try:
                chunks = chunk_file(
                    path,
                    text,
                    target=self.target_tokens,
                    overlap_pct=self.overlap_pct,
                    min_size=self.min_chunk_tokens,
                    topic_resolver=self._resolve_topic,
                    id_path=id_path,
                )
            except Exception as e:
                log.warning("chunking failed for %s: %s — skipping", path, e)
                continue
            if len(chunks) > self.max_chunks_per_doc:
                # Truncate, don't drop: pdf_dir/epub_dir keep the first N
                # chunks at the cap, and a partially indexed file beats a
                # silently missing one.
                log.warning(
                    "truncating %s — produced %d chunks (cap=%d); keeping the first %d",
                    path, len(chunks), self.max_chunks_per_doc, self.max_chunks_per_doc,
                )
                chunks = chunks[: self.max_chunks_per_doc]
            for c in chunks:
                yield c

    # -- helpers --------------------------------------------------------------

    def _walk(self) -> list[Path]:
        return iter_source_files(
            self.root, {".md", ".markdown", ".py"},
            skip_hidden=self.skip_hidden, only_paths=self.only_paths,
        )

    def _resolve_topic(self, path: Path, frontmatter: dict) -> str:
        # 1. Frontmatter wins.
        fm_topic = frontmatter.get(self.frontmatter_topic_key) if frontmatter else None
        if isinstance(fm_topic, str) and fm_topic.strip():
            return fm_topic.strip().lower()
        # 2. IMMEDIATE parent directory name (relative to ingester root).
        #    Bug #4 from the audit: previously this used parts[0] which
        #    returned the top-level directory under the ingester root, not
        #    the file's actual parent. With the old code, samples/notes/code/
        #    foo.py resolved to topic "notes" instead of "code".
        try:
            rel = path.relative_to(self.root)
            parts = rel.parts
            if len(parts) >= 2:
                return parts[-2].lower()
        except ValueError:
            pass
        # 3. Default.
        return self.default_topic
