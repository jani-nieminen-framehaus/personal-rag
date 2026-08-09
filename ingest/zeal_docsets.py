"""Ingest a Zeal docset (Dash-compatible schema).

A docset is a directory shaped like:
    <docset_root>/
        Contents/
            Resources/
                docSet.dsidx            ← SQLite index
                Documents/              ← actual HTML pages

The index is read-only — we never write to it. Page content is the HTML file
at `<docset_root>/Contents/Resources/Documents/<path>` where `<path>` comes
from the `path` column of the `searchIndex` table.

Topic resolution: the docset's directory name (e.g. `Python.docset` → `python`).
Sections: each page's `name` (the page title) — and for pages with HTML
h1/h2/h3, the chunker produces sub-chunks under that title.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup

from core.chunker import chunk_whole_file
from core.interfaces import Chunk, Ingester


log = logging.getLogger(__name__)


class ZealIngester(Ingester):
    def __init__(
        self,
        docset_path: str | Path,
        target_tokens: int,
        overlap_pct: int,
        min_chunk_tokens: int,
        default_topic: str,
        sqlite_filename: str = "docSet.dsidx",
        pages_dirname: str = "Contents/Resources/Documents",
        max_pages: int = 50000,
    ):
        self.docset_root = Path(docset_path).resolve()
        if not self.docset_root.is_dir():
            raise NotADirectoryError(f"docset path is not a directory: {self.docset_root}")
        if not self.docset_root.name.lower().endswith(".docset"):
            log.warning("path %s does not end in .docset — proceeding anyway", self.docset_root)
        self.sqlite_path = self.docset_root / "Contents" / "Resources" / sqlite_filename
        if not self.sqlite_path.is_file():
            raise FileNotFoundError(f"docset sqlite index not found: {self.sqlite_path}")
        self.pages_root = self.docset_root / pages_dirname
        if not self.pages_root.is_dir():
            raise FileNotFoundError(f"docset pages dir not found: {self.pages_root}")
        self.target_tokens = target_tokens
        self.overlap_pct = overlap_pct
        self.min_chunk_tokens = min_chunk_tokens
        self.default_topic = default_topic
        self.max_pages = max_pages

    @property
    def name(self) -> str:
        return f"zeal:{self.docset_root}"

    def iter_chunks(self) -> Iterator[Chunk]:
        topic = self._docset_topic()
        log.info("zeal: opening %s as topic=%r", self.sqlite_path, topic)
        # Open the index read-only — never write to a docset.
        with sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            table = self._pick_table(cur)
            if table is None:
                log.error("no usable page table in %s", self.sqlite_path)
                return
            cols = self._column_map(cur, table)
            if "path" not in cols:
                log.error("table %s has no `path` column; got %s", table, list(cols.keys()))
                return
            name_col = cols.get("name")

            cur.execute(f"SELECT {', '.join(cols)} FROM {table} LIMIT ?", (self.max_pages,))
            rows = cur.fetchall()
            log.info("zeal: %d rows from %s", len(rows), table)

            # Bug #8 fix: dedupe by relative path. The searchIndex table
            # has multiple rows per file (one per anchor / section), so
            # naively processing every row would re-read and re-chunk the
            # same file repeatedly. The first `name` we see becomes the
            # page title; later rows for the same `path` are ignored.
            seen_paths: dict[str, str] = {}  # rel -> first name
            for row in rows:
                rel = row["path"]
                if not rel:
                    continue
                rel = rel.replace("\\", "/")
                if rel in seen_paths:
                    continue
                seen_paths[rel] = (row[name_col] if name_col else None) or Path(rel).stem

            log.info("zeal: %d unique pages after dedupe", len(seen_paths))

            for rel, section in seen_paths.items():
                page_path = self.pages_root / rel
                if not page_path.is_file():
                    log.debug("missing page file: %s", page_path)
                    continue
                try:
                    html = page_path.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    log.debug("read failed: %s — %s", page_path, e)
                    continue
                text = _html_to_text(html)
                if not text.strip():
                    continue
                # Synthetic "path" so Chunk.source_path is meaningful.
                # Bug #8 fix: use the FULL relative path (with subdirs),
                # not just page_path.name — two pages named index.html in
                # different subdirs used to collide on chunk_id.
                synth_path = self.docset_root / rel
                for c in chunk_whole_file(
                    path=synth_path,
                    text=text,
                    target=self.target_tokens,
                    overlap_pct=self.overlap_pct,
                    min_size=self.min_chunk_tokens,
                    topic_resolver=lambda *_: topic,
                    doc_type="zeal",
                ):
                    c.section = str(section)
                    yield c

    # -- helpers --------------------------------------------------------------

    def _docset_topic(self) -> str:
        n = self.docset_root.name
        if n.lower().endswith(".docset"):
            n = n[:-7]
        n = n.strip().lower().replace(" ", "-")
        return n or self.default_topic

    def _pick_table(self, cur: sqlite3.Cursor) -> str | None:
        """Most docsets use `searchIndex`; some use `pages`. Try searchIndex
        first (it's the standard Dash schema)."""
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0].lower(): r[0] for r in cur.fetchall()}
        for cand in ("searchindex", "pages", "search_index"):
            if cand in tables:
                return tables[cand]
        return None

    def _column_map(self, cur: sqlite3.Cursor, table: str) -> dict[str, str]:
        """Map expected column names to their actual case in the table."""
        cur.execute(f"PRAGMA table_info({table})")
        cols = {row[1].lower(): row[1] for row in cur.fetchall()}
        return cols


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text. We don't try to preserve markdown semantics
    here — Zeal pages are messy and the generator just needs the words."""
    soup = BeautifulSoup(html, "lxml")
    # Drop script/style first.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse blank lines.
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)
