"""Unit tests for ingesters — covers topic resolution (bug #4)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ingest.markdown_dir import MarkdownDirIngester


def _ingester(tmp_path: Path) -> MarkdownDirIngester:
    """A real ingester rooted at tmp_path; ingest happens on demand."""
    return MarkdownDirIngester(
        root=tmp_path,
        target_tokens=768,
        overlap_pct=12,
        min_chunk_tokens=32,
        default_topic="default",
    )


def test_topic_fallback_uses_immediate_parent(tmp_path: Path):
    """Bug #4: parts[0] returned the TOP-level dir under the ingester root.
    The docstring and README promise the IMMEDIATE parent.

    Layout:
        tmp_path/
            notes/
                code/
                    foo.py
                    bar.md
            photography/
                exposure.md

    foo.py should resolve to topic='code', bar.md to 'code', exposure.md to 'photography'.
    """
    (tmp_path / "notes" / "code").mkdir(parents=True)
    (tmp_path / "notes" / "code" / "foo.py").write_text("def x(): return 1\n", encoding="utf-8")
    (tmp_path / "notes" / "code" / "bar.md").write_text("# Bar\nbody\n", encoding="utf-8")
    (tmp_path / "photography").mkdir()
    (tmp_path / "photography" / "exposure.md").write_text("# Exposure\nbody\n", encoding="utf-8")

    ing = _ingester(tmp_path)
    chunks = list(ing.iter_chunks())
    by_basename = {c.source_path.rsplit("\\", 1)[-1]: c.topic for c in chunks}
    assert by_basename.get("foo.py") == "code", f"foo.py should be 'code', got {by_basename}"
    assert by_basename.get("bar.md") == "code", f"bar.md should be 'code', got {by_basename}"
    assert by_basename.get("exposure.md") == "photography", f"exposure.md should be 'photography', got {by_basename}"


def test_topic_fallback_root_level_file(tmp_path: Path):
    """A file directly in the ingester root has no parent — falls to default."""
    (tmp_path / "lone.md").write_text("# Lone\nbody\n", encoding="utf-8")
    ing = _ingester(tmp_path)
    chunks = list(ing.iter_chunks())
    assert len(chunks) >= 1
    assert chunks[0].topic == "default"


def test_topic_frontmatter_wins(tmp_path: Path):
    """Frontmatter topic beats the parent-dir fallback."""
    (tmp_path / "notes" / "code").mkdir(parents=True)
    (tmp_path / "notes" / "code" / "foo.md").write_text(
        "---\ntopic: photography\n---\n# Foo\nbody\n", encoding="utf-8"
    )
    ing = _ingester(tmp_path)
    chunks = list(ing.iter_chunks())
    assert chunks[0].topic == "photography"


def test_topic_nested_dirs_use_immediate_parent(tmp_path: Path):
    """A file at tmp_path/A/B/C/foo.md should be topic 'C' (immediate parent),
    not 'A' (top-level under root). Topics are lowercased for consistency."""
    (tmp_path / "A" / "B" / "C").mkdir(parents=True)
    (tmp_path / "A" / "B" / "C" / "deep.md").write_text("# Deep\nbody\n", encoding="utf-8")
    ing = _ingester(tmp_path)
    chunks = list(ing.iter_chunks())
    assert chunks[0].topic == "c", f"expected 'c' (immediate parent, lowercased), got {chunks[0].topic!r}"
