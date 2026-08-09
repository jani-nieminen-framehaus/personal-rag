"""Structure-aware chunking.

Three strategies, picked by file extension or explicit doc_type:
- Markdown: frontmatter-aware, split on headings, then token-aware sub-split
  inside large sections with ~12% overlap.
- Python: AST-based split on function/async-function/class/method boundaries,
  with token-aware sub-split on huge definitions.
- Other (TS/JS/Go/Rust/...): whole-file as one chunk, log a one-line warning.

Chunk IDs are deterministic (UUID5) so re-ingest is idempotent. The caller
(typically an Ingester) passes `id_path` — the root-relative, forward-slash
path of the file — so chunk_ids are stable across clones of the same repo.
Without `id_path`, ids use the absolute path and change on every clone
(see tests/test_chunker.py::test_chunk_id_stable_across_paths).

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

    Algorithm:
      1. Compute piece boundaries via a forward walk with `overlap_pct`% stride.
      2. If the last piece is smaller than `min_size`, MERGE it into the
         previous piece (extending the previous to cover the tail). This
         preserves the tail's content — the previous bug was to OVERWRITE
         the last piece with the last `target` tokens, silently dropping
         everything between the original last piece and the new one.

    If text fits in `target`, returns it as-is.
    """
    tokens = _ENC.encode(text)
    n = len(tokens)
    if n <= target:
        return [text]

    overlap = max(1, (target * overlap_pct) // 100)
    step = max(1, target - overlap)

    # Pass 1: compute piece boundaries.
    boundaries: list[tuple[int, int]] = []
    i = 0
    while True:
        end = min(n, i + target)
        boundaries.append((i, end))
        if end == n:
            break
        i += step

    # Pass 2: if the last piece is too small, merge it into the previous.
    # This is the fix for the silent data-loss bug. The previous piece's
    # end extends to cover the small tail.
    if len(boundaries) >= 2 and (boundaries[-1][1] - boundaries[-1][0]) < min_size:
        prev_start, _ = boundaries[-2]
        _, last_end = boundaries[-1]
        boundaries[-2] = (prev_start, last_end)
        boundaries.pop()

    return [_ENC.decode(tokens[s:e]) for s, e in boundaries]


# ---------- Markdown ----------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
# A fenced code block line, per CommonMark: 0-3 leading spaces, then 3+
# backticks or tildes, then optionally an info string (e.g. "python" or "bash").
# Closing fences have nothing after the marker (whitespace only).
_FENCE_LINE_RE = re.compile(r"^(\s{0,3})(```+|~~~+)([^\n]*)$")


def _find_real_headings(md: str) -> list[re.Match]:
    """Find heading lines, but skip any that are inside a fenced code block.

    Bug #11 fix: the previous regex matched `# comment` lines inside
    ```bash ... ``` blocks, causing false splits. We walk the markdown
    line by line, toggling a "in-code" flag when we see a fence opener/
    closer, and only consider headings outside fences.
    """
    matches: list[re.Match] = []
    in_code = False
    fence_char: str | None = None
    fence_len: int = 0
    pos = 0
    for line in md.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        fence_match = _FENCE_LINE_RE.match(line)
        if fence_match:
            indent = fence_match.group(1)
            marker = fence_match.group(2)
            # Per CommonMark, fenced code blocks can have up to 3 spaces
            # of leading indent and the fence must be at least 3 chars
            # long. (Longer is fine; matches the same family.)
            if len(indent) > 3:
                continue
            char = marker[0]
            n = len(marker)
            if not in_code:
                if n >= 3:
                    in_code = True
                    fence_char = char
                    fence_len = n
                continue
            # In code — does this line close the block?
            if char == fence_char and n >= fence_len:
                in_code = False
                fence_char = None
                fence_len = 0
            continue
        if in_code:
            continue
        h = _HEADING_RE.match(line)
        if h:
            matches.append(_FakeMatch(line_start + h.start(), h.group(1), h.group(2)))
    return matches


class _FakeMatch:
    """Mimics re.Match enough for our consumer (start, group(1), group(2))."""
    def __init__(self, start: int, hashes: str, heading: str):
        self._start = start
        self._hashes = hashes
        self._heading = heading
    def start(self) -> int: return self._start
    def group(self, n: int) -> str:
        return {1: self._hashes, 2: self._heading}[n]


@dataclass
class MdSection:
    heading: str         # last heading text seen; "(top)" if before first heading
    level: int           # 0 if no heading
    body: str            # text under that heading (inclusive of the heading line)


def _split_markdown_sections(md: str) -> list[MdSection]:
    """Walk the markdown by heading lines. Preserve heading lines in the body
    so the LLM still has structural context. Headings inside fenced code
    blocks are ignored (audit #11)."""
    matches = _find_real_headings(md)
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
    *,
    id_path: str | None = None,
) -> list[Chunk]:
    """Chunk one markdown file. Honors YAML frontmatter for topic.

    `id_path` is the root-relative, forward-slash path used for chunk_id
    computation. Defaults to str(path) (absolute) if not provided.
    """
    fm = frontmatter.loads(text)
    body = fm.content
    meta = dict(fm.metadata or {})
    topic = topic_resolver(path, meta)

    id_path = id_path or str(path)
    parent_id = make_parent_id(id_path)
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
            cid = make_chunk_id(id_path, sec.heading, chunk_index)
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

    # Bug #12 fix: tiny chunks (under min_size tokens) get merged into
    # the next chunk so nothing ships as a sub-min fragment. We walk
    # left-to-right, absorbing small chunks into their successor.
    if len(chunks) >= 2 and min_size > 0:
        i = 0
        while i < len(chunks) - 1:
            if count_tokens(chunks[i].text) < min_size:
                # Absorb this small chunk into the next one.
                chunks[i + 1].text = chunks[i].text + "\n\n" + chunks[i + 1].text
                del chunks[i]
                # Don't advance i — the new chunks[i] is the merged
                # chunk, and may itself still be too small (e.g. a
                # chain of three tiny sections).
            else:
                i += 1
        # The last chunk, if still too small, has no successor to absorb
        # into. Prepend it to its predecessor instead.
        if len(chunks) >= 2 and count_tokens(chunks[-1].text) < min_size:
            chunks[-2].text = chunks[-2].text + "\n\n" + chunks[-1].text
            chunks.pop()

    return chunks


# ---------- Python (AST) -----------------------------------------------------

class _PyChunker:
    """Walk a Python AST and emit one span per top-level def/class + one
    span per class method (so methods are independently retrievable).

    Behaviour:
      - One span per top-level function or class definition.
      - Inside a class, one span per method (named "method ClassName.method").
      - Nested functions are NOT emitted as separate spans — they appear
        inside the parent function's text. (The previous implementation
        walked every node and emitted nested defs separately, which
        caused the same text to appear in two chunks.)
      - Decorators are included: span start is the first decorator's
        lineno (not the `def` line) so `@decorator` isn't lost.
      - If the file has module-level code before the first def (imports,
        docstring), a "(module prelude)" span covers that range.

    After the fix the previous dead-code bug — __init__ never called
    self.visit(self.tree), so spans was always empty — is gone. The class
    walks the body directly in __init__.
    """

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        self.spans: list[tuple[str, int, int]] = []  # (name, start_line, end_line)
        self._walk(self.tree.body)
        self._add_prelude()

    def _walk(self, body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                self._emit_class(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._emit_function(node, class_name=None)
            # Other top-level statements (imports, assignments) are NOT
            # emitted as spans; they appear in (module prelude) if any,
            # or are part of the previous span's tail.

    def _emit_class(self, node: ast.ClassDef) -> None:
        start = self._start_line(node)
        end = node.end_lineno or start
        self.spans.append((f"class {node.name}", start, end))
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._emit_function(child, class_name=node.name)

    def _emit_function(self, node, class_name: str | None) -> None:
        name = f"{class_name}.{node.name}" if class_name else node.name
        kind = "method" if class_name else "function"
        start = self._start_line(node)
        end = node.end_lineno or start
        self.spans.append((f"{kind} {name}", start, end))

    def _add_prelude(self) -> None:
        """If there's module-level code before the first span, add a
        (module prelude) span so imports + docstring are retrievable."""
        if not self.spans:
            return
        first_start = self.spans[0][1]  # the start_line of the first span
        if first_start > 1:
            self.spans.insert(0, ("(module prelude)", 1, first_start - 1))

    @staticmethod
    def _start_line(node) -> int:
        """If the node has decorators, start at the first one so the span
        includes the decorator."""
        if node.decorator_list:
            return min(d.lineno for d in node.decorator_list)
        return node.lineno


def chunk_python(
    path: Path,
    source: str,
    target: int,
    overlap_pct: int,
    min_size: int,
    topic_resolver: Callable[[Path, dict], str],
    *,
    id_path: str | None = None,
) -> list[Chunk]:
    """Chunk a .py file by AST node boundaries.

    Falls back to one whole-file chunk if parsing fails (logs a warning).
    """
    id_path = id_path or str(path)
    parent_id = make_parent_id(id_path)
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
            cid = make_chunk_id(id_path, name, chunk_index)
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
    *,
    id_path: str | None = None,
) -> list[Chunk]:
    """Treat the whole file as one (possibly token-split) chunk. Used for any
    non-Markdown, non-Python text source — code in other languages, plain text,
    Zeal pages, etc. The caller sets doc_type."""
    id_path = id_path or str(path)
    parent_id = make_parent_id(id_path)
    topic = topic_resolver(path, {})
    if count_tokens(text) <= target:
        pieces = [text]
    else:
        pieces = _split_by_tokens(text, target, overlap_pct, min_size)

    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        cid = make_chunk_id(id_path, doc_type, i)
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
    id_path: str | None = None,
) -> list[Chunk]:
    """Pick the chunking strategy by extension.

    `id_path` is the root-relative, forward-slash path used for chunk_id
    computation. If None, defaults to str(path). Pass a normalized path
    from the ingester to make chunk_ids stable across clones.
    """
    ext = path.suffix.lower()
    if ext in {".md", ".markdown"}:
        return chunk_markdown(path, text, target, overlap_pct, min_size, topic_resolver, id_path=id_path)
    if ext == ".py":
        return chunk_python(path, text, target, overlap_pct, min_size, topic_resolver, id_path=id_path)
    # Fallback: whole-file. log so the user knows what's happening.
    log.warning("whole-file chunking for %s (no specialized chunker)", path)
    return chunk_whole_file(
        path, text, target, overlap_pct, min_size, topic_resolver,
        doc_type=f"text_{ext.lstrip('.') or 'plain'}",
        id_path=id_path,
    )
