"""The one file walk, extracted from three near-identical copies."""
from __future__ import annotations

from pathlib import Path

from core.walk import iter_source_files


def _tree(root: Path) -> None:
    (root / "a.md").write_text("a", encoding="utf-8")
    (root / "b.txt").write_text("b", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "c.md").write_text("c", encoding="utf-8")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "d.md").write_text("d", encoding="utf-8")


def test_filters_by_suffix_and_recurses(tmp_path):
    _tree(tmp_path)
    got = iter_source_files(tmp_path, {".md"})
    assert [p.name for p in got] == ["a.md", "c.md"]


def test_skips_hidden_directories_by_default(tmp_path):
    _tree(tmp_path)
    assert all(".hidden" not in p.parts for p in iter_source_files(tmp_path, {".md"}))


def test_skip_hidden_false_includes_them(tmp_path):
    _tree(tmp_path)
    got = iter_source_files(tmp_path, {".md"}, skip_hidden=False)
    assert any(".hidden" in p.parts for p in got)


def test_suffix_match_is_case_insensitive(tmp_path):
    (tmp_path / "SHOUT.MD").write_text("x", encoding="utf-8")
    assert len(iter_source_files(tmp_path, {".md"})) == 1


def test_result_is_sorted(tmp_path):
    for name in ("z.md", "a.md", "m.md"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    got = iter_source_files(tmp_path, {".md"})
    assert got == sorted(got)


def test_only_paths_restricts_the_result(tmp_path):
    _tree(tmp_path)
    target = tmp_path / "a.md"
    got = iter_source_files(tmp_path, {".md"}, only_paths={target})
    assert got == [target]


def test_only_paths_ignores_entries_that_do_not_match_the_walk(tmp_path):
    """A path in only_paths that isn't under root (or has the wrong suffix)
    must not sneak into the result."""
    _tree(tmp_path)
    outsider = tmp_path / "b.txt"
    got = iter_source_files(tmp_path, {".md"}, only_paths={outsider})
    assert got == []


def test_empty_only_paths_means_no_files_not_no_filter(tmp_path):
    """set() must mean "restrict to nothing", NOT "no restriction". Task 5's
    refresh relies on this: collapsing the two would make a no-change refresh
    silently re-ingest the whole corpus."""
    _tree(tmp_path)
    assert iter_source_files(tmp_path, {".md"}) != []      # files DO exist
    assert iter_source_files(tmp_path, {".md"}, only_paths=set()) == []


def test_single_file_root_returns_that_file(tmp_path):
    f = tmp_path / "only.pdf"
    f.write_text("x", encoding="utf-8")
    assert iter_source_files(f, {".pdf"}) == [f]


def test_missing_root_returns_empty(tmp_path):
    assert iter_source_files(tmp_path / "nope", {".md"}) == []
