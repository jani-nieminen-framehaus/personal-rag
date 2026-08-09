"""Structure-aware chunking.

Three strategies, picked by file extension or explicit doc_type:
- Markdown: frontmatter-aware, split on headings, then token-aware sub-split
  inside large sections with ~12% overlap.
- Python: AST-based split on function/async-function/class/method boundaries,
  with token-aware sub-split on huge definitions.
- Other (TS/JS/Go/Rust/...): whole-file as one chunk, log a one-line warning.

Chunk IDs are deterministic (UUID5) so re-ingest is idempotent.

Token counting uses tiktoken cl100k as a fast, model-agnostic proxy. If you
later need exact Qwen3 token counts, swap `count_tokens` to use the embedding
model's tokenizer — every other call site stays the same.
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import tiktoken
import frontmatter

from core.interfaces import Chunk, make_chunk_id, make_parent_id


log = logging.getLogger(__name__)


# ---------- Token counting (cheap proxy) -------------------------------------

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Token count via cl100k. Fast and good enough for chunk sizing."""
    if not text:
        return 0
    return len(_ENC.encode(text))


def _split_by_tokens(text: str, target: int, overlap_pct: int, min_size: int) -> list[str]:
    """Sliding-window split on a single string by token count.

    If text fits in `target`, returns it as-is. Otherwise walks token windows
    with `overlap_pct`% overlap and emits the windows. Last window is padded
    by extending backwards if it's smaller than min_size.
    """
    tokens = _ENC.encode(text)
    n = len(tokens)
    if n <= target:
        return [text]

    overlap = max(1, (target * overlap_pct) // 100)
    step = max(1, target - overlap)
    pieces: list[str] = []
    start = 0
    while start < n:
        end = min(n, start + target)
        # If the last piece is too small, extend it backwards.
        if n - end < min_size and pieces:
            new_start = max(0, n - target)
            piece_tokens = tokens[new_start:n]
            pieces[-1] = _ENC.decode(piece_tokens)
            break
        piece_tokens = tokens[start:end]
        pieces.append(_ENC.decode(piece_tokens))
        if end == n:
            break
        start += step
    return pieces


# ---------- Markdown ----------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class MdSection:
    heading: str         # last heading text seen; "(top)" if before first heading
    level: int           # 0 if no heading
    body: str            # text under that heading (inclusive of the heading line)


def _split_markdown_sections(md: str) -> list[MdSection]:
    """Walk the markdown by heading lines. Preserve heading lines in the body
    so the LLM still has structural context."""
    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [MdSection(heading="(top)", level=0, body=md)]

    sections: list[MdSection] = []
    # Anything before the first heading.
    pre = md[: matches[0].start()].rstrip()
    if pre.strip():
        sections.append(MdSection(heading="(top)", level=0, body=pre))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        level = len(m.group(1))
        heading_text = m.group(2).strip()
        body = md[start:end].rstrip()
        sections.append(MdSection(heading=heading_text, level=level, body=body))
    return sections


def chunk_markdown(
    path: Path,
    text: str,
    target: int,
    overlap_pct: int,
    min_size: int,
    topic_resolver: Callable[[Path, dict], str],
) -> list[Chunk]:
    """Chunk one markdown file. Honors YAML frontmatter for topic."""
    fm = frontmatter.loads(text)
    body = fm.content
    meta = dict(fm.metadata or {})
    topic = topic_resolver(path, meta)

    parent_id = make_parent_id(str(path))
    sections = _split_markdown_sections(body)
    chunks: list[Chunk] = []
    chunk_index = 0

    for sec in sections:
        if count_tokens(sec.body) <= target:
            piece = sec.body
            pieces = [piece]
        else:
            pieces = _split_by_tokens(sec.body, target, overlap_pct, min_size)

        for piece in pieces:
            cid = make_chunk_id(str(path), sec.heading, chunk_index)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    parent_id=parent_id,
                    text=piece,
                    source_path=str(path),
                    topic=topic,
                    doc_type="markdown",
                    section=sec.heading,
                    extra={"heading_level": sec.level} if sec.level else {},
                )
            )
            chunk_index += 1
    return chunks


# ---------- Python (AST) -----------------------------------------------------

class _PyChunker(ast.NodeVisitor):
    """Collect top-level + class-level definitions as separate spans."""

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        self.spans: list[tuple[str, int, int]] = []  # (name, start_line, end_line)

    def visit(self, node):  # type: ignore[override]
        # We don't recurse with super().visit() — we walk manually so we can
        # group methods under their class name.
        if isinstance(node, ast.ClassDef):
            self._emit_class(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._emit_function(node, kind="function")
        # Recurse for nested defs (def inside def, methods are handled in _emit_class).
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def _emit_class(self, node: ast.ClassDef) -> None:
        self.spans.append((f"class {node.name}", node.lineno, node.end_lineno or node.lineno))
        # Emit methods as separate spans so they can be retrieved independently.
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._emit_function(child, kind="method", class_name=node.name)

    def _emit_function(self, node, kind: str, class_name: str | None = None) -> None:
        name = f"{class_name}.{node.name}" if class_name else node.name
        self.spans.append((f"{kind} {name}", node.lineno, node.end_lineno or node.lineno))


def chunk_python(
    path: Path,
    source: str,
    target: int,
    overlap_pct: int,
    min_size: int,
    topic_resolver: Callable[[Path, dict], str],
) -> list[Chunk]:
    """Chunk a .py file by AST node boundaries.

    Falls back to one whole-file chunk if parsing fails (logs a warning).
    """
    parent_id = make_parent_id(str(path))
    topic = topic_resolver(path, {})
    chunks: list[Chunk] = []
    lines = source.splitlines(keepends=True)

    try:
        chunker = _PyChunker(source)
        spans = chunker.spans
    except SyntaxError as e:
        log.warning("python parse failed for %s: %s — falling back to whole-file chunk", path, e)
        spans = [("(module)", 1, len(lines))]

    if not spans:
        # File had no top-level defs/class — emit one chunk for the whole file.
        spans = [("(module)", 1, len(lines))]

    chunk_index = 0
    for name, start, end in spans:
        # ast line numbers are 1-based and inclusive.
        body = "".join(lines[start - 1: end])
        if count_tokens(body) <= target:
            pieces = [body]
        else:
            pieces = _split_by_tokens(body, target, overlap_pct, min_size)
        for piece in pieces:
            cid = make_chunk_id(str(path), name, chunk_index)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    parent_id=parent_id,
                    text=piece,
                    source_path=str(path),
                    topic=topic,
                    doc_type="code_python",
                    section=name,
                    extra={"line_start": start, "line_end": end},
                )
            )
            chunk_index += 1
    return chunks


# ---------- Whole-file fallback ----------------------------------------------

def chunk_whole_file(
    path: Path,
    text: str,
    target: int,
    overlap_pct: int,
    min_size: int,
    topic_resolver: Callable[[Path, dict], str],
    doc_type: str,
) -> list[Chunk]:
    """Treat the whole file as one (possibly token-split) chunk. Used for any
    non-Markdown, non-Python text source — code in other languages, plain text,
    Zeal pages, etc. The caller sets doc_type."""
    parent_id = make_parent_id(str(path))
    topic = topic_resolver(path, {})
    if count_tokens(text) <= target:
        pieces = [text]
    else:
        pieces = _split_by_tokens(text, target, overlap_pct, min_size)

    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        cid = make_chunk_id(str(path), doc_type, i)
        chunks.append(
            Chunk(
                chunk_id=cid,
                parent_id=parent_id,
                text=piece,
                source_path=str(path),
                topic=topic,
                doc_type=doc_type,
                section=doc_type,
            )
        )
    return chunks


# ---------- Dispatch ---------------------------------------------------------

def chunk_file(
    path: Path,
    text: str,
    *,
    target: int,
    overlap_pct: int,
    min_size: int,
    topic_resolver: Callable[[Path, dict], str],
) -> list[Chunk]:
    """Pick the chunking strategy by extension."""
    ext = path.suffix.lower()
    if ext in {".md", ".markdown"}:
        return chunk_markdown(path, text, target, overlap_pct, min_size, topic_resolver)
    if ext == ".py":
        return chunk_python(path, text, target, overlap_pct, min_size, topic_resolver)
    # Fallback: whole-file. log so the user knows what's happening.
    log.warning("whole-file chunking for %s (no specialized chunker)", path)
    return chunk_whole_file(path, text, target, overlap_pct, min_size, topic_resolver, doc_type=f"text_{ext.lstrip('.') or 'plain'}")
