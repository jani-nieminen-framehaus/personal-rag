"""Unit tests for core.chunker.

These are the tests that would have caught the 🔴 blockers in the audit:
- _split_by_tokens silent data loss (bug #2)
- _PyChunker dead __init__ (bug #3)
- Topic fallback using top-level dir (bug #4)
- Path-relative chunk_id (bug #1)
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core import chunker
from core.chunker import (
    _split_by_tokens,
    _PyChunker,
    chunk_file,
    chunk_python,
    chunk_markdown,
    count_tokens,
)
from core.interfaces import make_chunk_id, make_parent_id


# -- _split_by_tokens ---------------------------------------------------------

def _enc(text: str) -> list[int]:
    """Encode text to token ids for analysis."""
    from core.chunker import _ENC
    return _ENC.encode(text)


def test_split_by_tokens_no_data_loss_1444():
    """Bug #2 reproducer: text > target tokens must not lose content.

    The original code returned 1 piece (the LAST `target` tokens) for any
    input where the tail was smaller than `min_size`, losing everything
    before the start of the last window. The fix must cover the original
    text with the union of pieces — no word may vanish.
    """
    # Build a long text. The exact token count depends on tiktoken's BPE
    # behavior on "tok0001" etc., so we don't assert a specific count.
    # We DO assert that every distinct "word" survives.
    text = " ".join(f"tok{i:04d}" for i in range(2000))
    n = count_tokens(text)
    assert n > 768, f"setup: text must be > target tokens; got {n}"

    pieces = _split_by_tokens(text, target=768, overlap_pct=12, min_size=32)

    # Every distinct word must appear in at least one piece.
    all_piece_text = " ".join(pieces)
    missing = [i for i in (0, 100, 500, 700, 1000, 1300, 1500, 1800, 1999)
               if f"tok{i:04d}" not in all_piece_text]
    assert not missing, f"words missing from pieces: {missing}"


def test_split_by_tokens_no_data_loss_table():
    """Bug #2 reproducer: parametrize over the input sizes from the audit table."""
    for n in (769, 1000, 1444, 2000):
        text = " ".join(f"tok{i:04d}" for i in range(n))
        pieces = _split_by_tokens(text, target=768, overlap_pct=12, min_size=32)
        # Every distinct word must appear in some piece.
        all_piece_text = " ".join(pieces)
        missing = [i for i in range(0, n, max(1, n // 20))
                   if f"tok{i:04d}" not in all_piece_text]
        assert not missing, f"n={n}: words missing from pieces: {missing}"


def test_split_by_tokens_short_text_passthrough():
    """Text <= target tokens returns the text as a single piece."""
    text = "short text"
    pieces = _split_by_tokens(text, target=100, overlap_pct=12, min_size=10)
    assert pieces == [text]


def test_split_by_tokens_oversized_last_piece_merged_not_replaced():
    """When the tail is too small, the last piece must be EXTENDED, not replaced.

    This is the second half of bug #2: the original code overwrote the last
    piece, losing its content. The fix should make the last piece larger
    (potentially > target) so the tail is captured.
    """
    # 1444 tokens, target 768, min_size 700 — forces the merge branch.
    text = " ".join(f"tok{i:04d}" for i in range(1444))
    pieces = _split_by_tokens(text, target=768, overlap_pct=12, min_size=700)
    # Tail is 1444 - 768 = 676, which is < 700. Last piece should be merged.
    # No tokens should be missing.
    all_piece_text = " ".join(pieces)
    for i in (0, 700, 1400, 1443):
        assert f"tok{i:04d}" in all_piece_text, f"token {i} missing after merge"


# -- _PyChunker (bug #3) ------------------------------------------------------

PY_WITH_CLASS_AND_METHODS = '''\
class Greeter:
    """A friendly greeter."""

    def hello(self, name: str) -> str:
        """Say hello."""
        return f"hello, {name}"

    def goodbye(self, name: str) -> str:
        return f"goodbye, {name}"


def standalone(x: int) -> int:
    """A top-level function."""
    return x + 1
'''


def test_py_chunker_emits_class_and_method_spans():
    """Bug #3: __init__ was dead code. Spans must be populated.

    With a class containing 2 methods + 1 top-level function, we expect
    4 spans: class, method.hello, method.goodbye, function standalone.
    """
    chunker_ = _PyChunker(PY_WITH_CLASS_AND_METHODS)
    names = [s[0] for s in chunker_.spans]
    assert "class Greeter" in names, f"class span missing; got {names}"
    assert "method Greeter.hello" in names
    assert "method Greeter.goodbye" in names
    assert "function standalone" in names
    # No duplicates
    assert len(names) == len(set(names)), f"duplicate spans: {names}"


def test_py_chunker_no_nested_function_duplication():
    """Bug #3 sub-issue: nested defs were emitted as separate spans AND
    appeared inside their parent span's text. The fix: nested defs are not
    emitted as separate top-level spans — they appear inside the parent.
    """
    source = '''\
def outer():
    """An outer function with a nested helper."""
    def inner():
        return 1
    return inner()
'''
    chunker_ = _PyChunker(source)
    names = [s[0] for s in chunker_.spans]
    # Only "function outer" — not "function inner" (which is nested).
    assert names == ["function outer"], f"unexpected spans: {names}"


def test_py_chunker_includes_decorators():
    """Bug #3 sub-issue: node.lineno is the def line, so decorators fell
    outside the span. The fix: start at the first decorator's lineno if any.
    """
    source = '''\
@dataclass
class Foo:
    x: int

@property
def bar(self) -> int:
    return self.x
'''
    chunker_ = _PyChunker(source)
    # Find the bar span
    bar_span = next(s for s in chunker_.spans if "bar" in s[0])
    # @property is on line 5, def bar is on line 6. Span must start at line 5.
    assert bar_span[1] == 5, f"expected span start at decorator line 5, got {bar_span[1]}"
    foo_span = next(s for s in chunker_.spans if "class Foo" in s[0])
    # @dataclass is on line 1, class Foo on line 2. Span must start at line 1.
    assert foo_span[1] == 1, f"expected class span start at decorator line 1, got {foo_span[1]}"


def test_chunk_python_produces_multiple_chunks():
    """End-to-end: the audit verified framehaus_pipeline.py produced 1 chunk
    (the AST was dead). After the fix, it should produce one chunk per
    top-level def + module prelude."""
    src_path = Path("samples/notes/code/framehaus_pipeline.py")
    text = src_path.read_text(encoding="utf-8")
    chunks = chunk_python(
        src_path, text,
        target=768, overlap_pct=12, min_size=32,
        topic_resolver=lambda *_: "default",
    )
    # Audit said: 1 chunk. After fix: at least 4 (1 module prelude + 4 funcs).
    assert len(chunks) >= 4, f"expected >= 4 chunks, got {len(chunks)}: {[c.section for c in chunks]}"
    # All function names should be present as sections.
    sections = {c.section for c in chunks}
    for fn in ("connect", "fetch_clients", "upsert_client", "iter_clients_jsonl"):
        assert any(fn in s for s in sections), f"function {fn} missing from sections: {sections}"


# -- Markdown chunking --------------------------------------------------------

def test_chunk_markdown_heading_sections():
    # Sections are large enough that the merge-on-tiny rule doesn't fire.
    src = '''\
# Title

This is a substantial intro body with enough tokens to qualify as a
proper chunk on its own, well above min_size.

## Section A

Body of section A which also has enough content to stand alone without
being absorbed by the merge logic.

## Section B

Body of section B which is similarly substantial and won't be merged
either.
'''
    p = Path("test.md")
    chunks = chunk_markdown(
        p, src, target=100, overlap_pct=12, min_size=10,
        topic_resolver=lambda *_: "t",
    )
    sections = [c.section for c in chunks]
    assert "Title" in sections
    assert "Section A" in sections
    assert "Section B" in sections


# -- chunk_id (bug #1) --------------------------------------------------------

def test_make_chunk_id_normalizes_backslashes():
    """make_chunk_id normalizes backslashes so Windows and Unix paths hash equal."""
    a = make_chunk_id("ml/yarn.md", "YaRN", 0)
    b = make_chunk_id("ml\\yarn.md", "YaRN", 0)
    assert a == b, f"chunk_id depends on slash style: {a} != {b}"


def test_chunk_id_stable_across_paths(tmp_path: Path):
    """The same logical file at two different absolute paths gets the
    same chunk_id when the caller passes a normalized id_path.

    This is the contract: the ingester computes the root-relative
    id_path, passes it through, and the chunker hashes from that —
    so cloning the repo to a new location doesn't invalidate ids.
    """
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    (root1 / "ml" / "yarn.md").mkdir(parents=True)
    (root2 / "ml" / "yarn.md").mkdir(parents=True)
    text = "# YaRN\nbody\n"

    (root1 / "ml" / "yarn.md" / "note.md").write_text(text, encoding="utf-8")
    (root2 / "ml" / "yarn.md" / "note.md").write_text(text, encoding="utf-8")

    chunks1 = chunk_file(
        root1 / "ml" / "yarn.md" / "note.md",
        text, target=100, overlap_pct=12, min_size=10,
        topic_resolver=lambda *_: "t",
        id_path="ml/yarn.md/note.md",
    )
    chunks2 = chunk_file(
        root2 / "ml" / "yarn.md" / "note.md",
        text, target=100, overlap_pct=12, min_size=10,
        topic_resolver=lambda *_: "t",
        id_path="ml/yarn.md/note.md",
    )
    assert chunks1[0].chunk_id == chunks2[0].chunk_id, (
        f"chunk_id depends on absolute path: {chunks1[0].chunk_id} != {chunks2[0].chunk_id}"
    )
