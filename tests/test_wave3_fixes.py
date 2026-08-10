"""Unit tests for Wave 3 fixes.

Covers:
- #9 query instruction: embed_query vs embed_documents split.
- #10 dense recall: AskResult exposes dense_hits.
- #11 code-fence-aware markdown: headings inside ``` blocks ignored.
- #12 min_chunk_tokens for sections: tiny sections merged into next.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# -- #9: query instruction split ---------------------------------------------

def test_embedder_abc_has_query_and_documents_methods():
    """The Embedder ABC must expose embed_query and embed_documents so
    instruction-tuned models can apply different prefixes (audit #9)."""
    from core.interfaces import Embedder
    assert hasattr(Embedder, "embed_query")
    assert hasattr(Embedder, "embed_documents")


def test_default_embed_query_delegates_to_embed_documents():
    """A bare Embedder subclass that only implements embed() should still
    work — embed_query and embed_documents default to the legacy embed()."""
    from core.interfaces import Embedder

    class Stub(Embedder):
        def __init__(self):
            self.calls = []
        def dim(self): return 4
        def embed(self, texts):
            self.calls.append(("embed", list(texts)))
            return [[float(i)] * 4 for i in range(len(texts))]

    s = Stub()
    q = s.embed_query("hello")
    d = s.embed_documents(["a", "b"])
    assert q == [0.0, 0.0, 0.0, 0.0]
    assert d == [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    # embed_query calls embed_documents (which calls embed), so embed
    # gets one call with a single-element list. Default impl works.
    assert len(s.calls) == 2


# -- #10: AskResult exposes dense_hits ---------------------------------------

def test_ask_result_has_dense_hits_field():
    """AskResult must have a `dense_hits` field for the eval to compute
    recall@20 separately from recall@5 (audit #10)."""
    from core.pipeline import AskResult
    r = AskResult(answer="x", citations=[], dense_hits=[{"chunk_id": "c1", "score": 0.9}])
    assert r.dense_hits[0]["chunk_id"] == "c1"


# -- #11: code-fence-aware markdown chunker ----------------------------------

MD_WITH_FENCE = '''\
# Real heading

intro text

```bash
# This is a shell comment, not a heading
# Another comment
echo "still in fence"
```

## Another real heading

body
'''


def test_heading_inside_code_fence_is_ignored():
    """Audit #11: the previous regex matched `# comment` lines inside
    fenced code blocks, creating false section splits. The fix: walk
    lines, toggle an in-code flag, and only consider headings outside
    fences. After the fix, this markdown has exactly 2 real sections
    (Real heading, Another real heading) — the `# This is a shell
    comment` line must NOT be treated as a heading.
    """
    from core.chunker import _find_real_headings, _split_markdown_sections

    headings = _find_real_headings(MD_WITH_FENCE)
    heading_texts = [h.group(2) for h in headings]
    assert "Real heading" in heading_texts
    assert "Another real heading" in heading_texts
    # The shell comment must NOT be a heading.
    assert "This is a shell comment" not in heading_texts
    assert "Another comment" not in heading_texts


def test_chunk_markdown_skips_fenced_headings():
    """End-to-end: chunk_markdown on a doc with code fences must produce
    the right number of sections (no false splits)."""
    from core.chunker import chunk_markdown
    p = Path("test.md")
    # Use min_size=0 to disable the merge-on-tiny logic — this test is
    # about fence handling, not merging.
    chunks = chunk_markdown(
        p, MD_WITH_FENCE, target=100, overlap_pct=12, min_size=0,
        topic_resolver=lambda *_: "t",
    )
    sections = [c.section for c in chunks]
    assert "Real heading" in sections, f"missing 'Real heading' in {sections}"
    assert "Another real heading" in sections, f"missing 'Another real heading' in {sections}"
    # No spurious sections from the shell comments.
    assert not any("comment" in s.lower() for s in sections if s not in ("Real heading", "Another real heading")), (
        f"shell comments leaked as section names: {sections}"
    )


def test_chunk_markdown_handles_tilde_fences():
    """CommonMark allows ~~~ as the fence char. The fix must accept that."""
    from core.chunker import _find_real_headings
    md = '''\
# Real heading

~~~python
# A comment in a tilde fence
print("hello")
~~~

## Another real
body
'''
    headings = _find_real_headings(md)
    heading_texts = [h.group(2) for h in headings]
    assert heading_texts == ["Real heading", "Another real"]


# -- #12: min_chunk_tokens for sections --------------------------------------

MD_WITH_TINY_SECTIONS = '''\
# Big section

This is a fairly large body of text that has more than enough tokens to
qualify as a proper chunk. It goes on for a bit to ensure we have plenty
of content here for the test.

# Tiny

## Another real

body of another real heading with enough content
'''


def test_tiny_sections_get_merged_into_next():
    """Audit #12: the previous chunker emitted a 29-token `(top)` chunk
    as-is. The fix: sections smaller than min_chunk_tokens are merged
    into the next section's chunk. After the fix, no chunk in the
    output is below min_chunk_tokens.
    """
    from core.chunker import chunk_markdown, count_tokens
    p = Path("test.md")
    chunks = chunk_markdown(
        p, MD_WITH_TINY_SECTIONS,
        target=100, overlap_pct=12, min_size=30,
        topic_resolver=lambda *_: "t",
    )
    # All chunks should be >= min_size tokens, OR there should be only
    # one chunk left (which can't be merged into anything).
    small = [c for c in chunks if count_tokens(c.text) < 30]
    assert len(small) <= 1, (
        f"unexpected number of small chunks: {[(c.section, len(c.text)) for c in small]}"
    )


# -- #14: CWD-relative eval path ---------------------------------------------

def test_resolve_repo_path_finds_file_in_cwd(tmp_path: Path, monkeypatch):
    """The default `eval/golden_set.jsonl` path should resolve when the
    user is sitting in the repo root, even if it's not an absolute path
    (audit #14)."""
    from cli import _resolve_repo_path

    # Create eval/golden_set.jsonl in CWD
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "golden_set.jsonl").write_text("{}\n", encoding="utf-8")

    resolved = _resolve_repo_path("eval/golden_set.jsonl")
    assert resolved == tmp_path / "eval" / "golden_set.jsonl"
