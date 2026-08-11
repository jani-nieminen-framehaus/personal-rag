# P3.2 Living Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The index keeps itself current — `rag refresh` re-ingests only what actually changed, a scheduled task runs it unattended, and `rag forget` erases a source or topic on demand.

**Architecture:** A `sources:` list in `config.yaml` records what the index is supposed to contain. Schema v2 adds a raw-bytes `file_hash` per source file, so change detection costs one file read instead of a full parse+chunk. A new `core/refresh.py` plans the work (new / changed / unchanged / vanished) before doing any of it, so the common "nothing changed" case never constructs an embedder. The three near-identical directory walks are extracted to `core/walk.py` first, because refresh needs to enumerate a source's files without instantiating its ingester.

**Tech Stack:** Existing only — click, sqlite3, qdrant-client (mocked in tests), pytest. No new dependencies.

## Global Constraints

- Repo `D:\Tinkering sideprojects\rag`, branch `feat/p3.2-living-index` (already created; spec committed at `ffbe561`).
- Test command from repo root: `".venv_tests\Scripts\python.exe" -m pytest tests/ -q`. Baseline: **204 passed, 1 skipped, 1 warning**. The skip is `test_epub_ingest.py` (ebooklib absent) and the warning is a pre-existing Starlette deprecation in `tests/test_serve.py` — both expected, neither is yours to fix.
- No test may require Ollama, Qdrant, GPU models, or the network.
- **No path may delete data without an explicit flag or confirmation.** `--prune` and `rag forget` are the only deleting paths; both must be opt-in.
- **An absent source root means "unknown", never "empty".** An unplugged drive or unmounted share must never cause a prune. This is the single most important safety property in this plan.
- A GateGuard hook DENIES the FIRST Edit/Write call per file with a "present these facts" message. Expected friction: retry the identical call once. Do not rewrite the edit or try to disable the hook.
- Commit messages must end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_019QXgDZsgWNRnwg2ijiRocm`
- Never `git add -A` — there is an untracked `.superpowers/` scratch directory that must never be committed. Stage files explicitly.

## File Structure

| File | Responsibility |
| --- | --- |
| `core/walk.py` (new) | The one file-enumeration function; suffix filtering, hidden-file skipping, optional `only_paths` restriction. |
| `core/metadata.py` (modify) | Schema v2 (`sources.file_hash`), `hash_file()`, hash lookup and row deletion for refresh. |
| `ingest/{markdown,pdf,epub}_dir.py` (modify) | Use the extracted walk; accept `only_paths`. |
| `core/refresh.py` (new) | Plan and apply a refresh. Knows nothing about click. |
| `cli.py` (modify) | `rag refresh` and `rag forget` command surfaces. |
| `scripts/install-refresh-task.ps1`, `uninstall-refresh-task.ps1` (new) | Scheduled task registration, reusing `_config.ps1`. |

---

### Task 1: Extract the shared file walk

**Files:**
- Create: `core/walk.py`
- Modify: `ingest/markdown_dir.py` (`_walk`, ~line 105), `ingest/pdf_dir.py` (`_collect_files`, ~line 99), `ingest/epub_dir.py` (`_collect_files`, ~line 94)
- Test: `tests/test_walk.py` (create)

**Interfaces:**
- Produces: `core.walk.iter_source_files(root: Path, suffixes: set[str], skip_hidden: bool = True, only_paths: set[Path] | None = None) -> list[Path]` — returns a **sorted** list of matching files. Tasks 3 and 5 both consume this.

Note the current three copies differ: markdown matches `{".md", ".markdown", ".py"}` and returns unsorted; pdf and epub match one suffix each and return sorted. The extracted version always sorts — that makes markdown ingest order deterministic, which is an improvement, not a regression.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_walk.py
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


def test_single_file_root_returns_that_file(tmp_path):
    f = tmp_path / "only.pdf"
    f.write_text("x", encoding="utf-8")
    assert iter_source_files(f, {".pdf"}) == [f]


def test_missing_root_returns_empty(tmp_path):
    assert iter_source_files(tmp_path / "nope", {".md"}) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `".venv_tests\Scripts\python.exe" -m pytest tests/test_walk.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.walk'`

- [ ] **Step 3: Implement `core/walk.py`**

```python
"""The one file-enumeration function.

Extracted from three near-identical copies in ingest/{markdown,pdf,epub}_dir.py
(the original audit's cross-cutting finding #9). `rag refresh` needs to
enumerate a source's files WITHOUT instantiating its ingester, which is what
finally forced the extraction.
"""
from __future__ import annotations

from pathlib import Path


def iter_source_files(
    root: str | Path,
    suffixes: set[str],
    skip_hidden: bool = True,
    only_paths: set[Path] | None = None,
) -> list[Path]:
    """Every file under `root` whose suffix is in `suffixes`, sorted.

    Args:
        root: a directory to walk, or a single file (returned as-is if it
            matches the suffix filter).
        suffixes: lower-case suffixes including the dot, e.g. {".md", ".py"}.
        skip_hidden: skip anything under a dot-prefixed directory, and
            dot-prefixed files themselves.
        only_paths: when set, restrict the result to these paths. Entries that
            the walk would not have produced anyway are ignored, so a caller
            cannot smuggle in a file from outside `root` or of the wrong type.

    A missing root yields an empty list rather than raising: callers that care
    about the difference between "no files" and "root absent" must check the
    root themselves. `core/refresh.py` depends on making exactly that
    distinction before it prunes anything.
    """
    root = Path(root).resolve()
    wanted = {s.lower() for s in suffixes}

    if root.is_file():
        candidates = [root] if root.suffix.lower() in wanted else []
    elif root.is_dir():
        candidates = [
            p for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in wanted
            and not (
                skip_hidden
                and any(part.startswith(".") for part in p.relative_to(root).parts)
            )
        ]
    else:
        candidates = []

    if only_paths is not None:
        wanted_paths = {Path(p).resolve() for p in only_paths}
        candidates = [p for p in candidates if p.resolve() in wanted_paths]

    return sorted(candidates)
```

- [ ] **Step 4: Switch the three ingesters**

In each, replace the body of the walk method with a delegation, keeping the method name so nothing else changes:

`ingest/markdown_dir.py`:
```python
    def _walk(self) -> list[Path]:
        return iter_source_files(
            self.root, {".md", ".markdown", ".py"},
            skip_hidden=self.skip_hidden, only_paths=self.only_paths,
        )
```
`ingest/pdf_dir.py`:
```python
    def _collect_files(self) -> list[Path]:
        return iter_source_files(
            self.path, {".pdf"},
            skip_hidden=self.skip_hidden, only_paths=self.only_paths,
        )
```
`ingest/epub_dir.py`: identical but `{".epub"}`.

`self.only_paths` does not exist yet — add `self.only_paths = None` in each `__init__` for now; Task 3 makes it a real constructor parameter. Add `from core.walk import iter_source_files` to each module's imports. Delete the now-dead loop bodies; do not leave them commented out.

Note the pdf/epub versions previously special-cased `if self.is_file: return [self.path]` — `iter_source_files` handles a file root itself, so that branch goes away. Verify the existing single-file PDF/EPUB tests still pass.

- [ ] **Step 5: Run the new test file, then the full suite**

Both must be green. Existing ingest tests are the real check here — this is a refactor, so any behavioural change shows up there.

- [ ] **Step 6: Commit**

```bash
git add core/walk.py ingest/markdown_dir.py ingest/pdf_dir.py ingest/epub_dir.py tests/test_walk.py
git commit -m "refactor(ingest): extract the shared file walk to core/walk.py"
```

---

### Task 2: Schema v2 — raw-file hashing

**Files:**
- Modify: `core/metadata.py` (`_LATEST_SCHEMA_VERSION` at line 44, `_migrate` at ~137, `record_source` at ~184, `get_sources` at ~269, `hash_text` at ~417)
- Test: `tests/test_metadata_migration.py` (append)

**Interfaces:**
- Consumes: the existing tolerant migration pattern — `_migrate` reads `PRAGMA table_info` per column, skips columns already present, and sets `user_version` regardless, so a half-migrated database self-repairs. **Extend that same loop; do not invent a second mechanism.**
- Produces: `_LATEST_SCHEMA_VERSION = 2`; `sources.file_hash TEXT`; `hash_file(path: str | Path) -> str` (module-level, beside `hash_text`); `record_source(..., file_hash: str | None = None)`; `get_sources()` rows include `file_hash`; `source_hashes() -> dict[str, str | None]` mapping every recorded `source_path` to its `file_hash`; `delete_source(source_path: str) -> bool` removing one row. Task 5 consumes `hash_file`, `source_hashes` and `delete_source`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_metadata_migration.py`)

```python
def test_v2_adds_file_hash_and_reaches_version_2(tmp_path):
    db = tmp_path / "meta.sqlite3"
    MetadataStore(str(db)).close()
    assert "file_hash" in _columns(str(db), "sources")
    conn = sqlite3.connect(str(db))
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    conn.close()


def test_v1_database_upgrades_to_v2(tmp_path):
    """A database already at v1 must gain file_hash without losing rows."""
    db = tmp_path / "meta.sqlite3"
    md = MetadataStore(str(db))
    md.record_source(source_path="/a.md", doc_type="markdown", topic="t",
                     chunk_count=3, content_hash="abc")
    md.close()
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    md2 = MetadataStore(str(db))
    rows = md2.get_sources()
    md2.close()
    assert len(rows) == 1
    assert rows[0]["content_hash"] == "abc"
    assert rows[0]["file_hash"] is None


def test_hash_file_changes_with_content(tmp_path):
    from core.metadata import hash_file

    f = tmp_path / "x.md"
    f.write_text("hello", encoding="utf-8")
    first = hash_file(f)
    assert first
    assert hash_file(f) == first          # stable
    f.write_text("goodbye", encoding="utf-8")
    assert hash_file(f) != first          # content-sensitive


def test_hash_file_missing_returns_empty(tmp_path):
    from core.metadata import hash_file

    assert hash_file(tmp_path / "nope.md") == ""


def test_source_hashes_and_delete_source(tmp_path):
    md = MetadataStore(str(tmp_path / "meta.sqlite3"))
    md.record_source(source_path="/a.md", doc_type="markdown", topic="t",
                     chunk_count=1, content_hash="c1", file_hash="f1")
    md.record_source(source_path="/b.md", doc_type="markdown", topic="t",
                     chunk_count=1, content_hash="c2")
    assert md.source_hashes() == {"/a.md": "f1", "/b.md": None}
    assert md.delete_source("/a.md") is True
    assert md.delete_source("/a.md") is False      # already gone
    assert set(md.source_hashes()) == {"/b.md"}
    md.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `".venv_tests\Scripts\python.exe" -m pytest tests/test_metadata_migration.py -q`
Expected: FAIL — no `file_hash` column, `hash_file` undefined, unexpected kwarg.

- [ ] **Step 3: Implement**

Set `_LATEST_SCHEMA_VERSION = 2`. In `_migrate`, the per-column loop already exists for v1's two columns — extend the same structure so `sources` gains `file_hash`. The loop must remain tolerant: read `PRAGMA table_info(sources)`, skip `file_hash` if present, and still set `user_version` at the end.

Add beside `hash_text`:

```python
def hash_file(path: str | Path) -> str:
    """Hash a file's raw bytes. Returns "" when the file cannot be read.

    Raw bytes, not chunked text: the point is to decide whether to parse a
    file at all, so this must not require parsing it. Read in blocks so a
    500 MB PDF does not land in memory.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()[:16]
```

(`Path` must be imported in `core/metadata.py` — check and add if absent.)

Add `file_hash: str | None = None` to `record_source`'s signature and to its INSERT/upsert column list. Add `file_hash` to `get_sources`'s SELECT. Then:

```python
    def source_hashes(self) -> dict[str, str | None]:
        """Every recorded source_path mapped to its file_hash. Refresh's
        change-detection input — one query, no limit."""
        cur = self._conn.execute("SELECT source_path, file_hash FROM sources")
        return {row[0]: row[1] for row in cur.fetchall()}

    def delete_source(self, source_path: str) -> bool:
        """Remove one source row. Returns True if a row was deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sources WHERE source_path = ?", (source_path,)
            )
            return cur.rowcount > 0
```

- [ ] **Step 4: Run the migration tests, then the full suite.** Both green.

- [ ] **Step 5: Commit**

```bash
git add core/metadata.py tests/test_metadata_migration.py
git commit -m "feat(metadata): schema v2 - raw-file hashing for change detection"
```

---

### Task 3: `only_paths` on the directory ingesters

**Files:**
- Modify: `ingest/markdown_dir.py` (`__init__` ~line 32), `ingest/pdf_dir.py` (~57), `ingest/epub_dir.py` (~52)
- Test: `tests/test_ingest_only_paths.py` (create)

**Interfaces:**
- Consumes: Task 1's `iter_source_files(..., only_paths=...)`, already wired into each walk method.
- Produces: each of the three constructors accepts `only_paths: set[Path] | None = None` as a trailing keyword parameter, stored as `self.only_paths`. Task 5 constructs ingesters with it.

`ZealIngester` is deliberately excluded — a docset is one all-or-nothing unit.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_only_paths.py
"""Ingesters can be restricted to a subset of files, so refresh can
re-ingest just what changed instead of the whole tree."""
from __future__ import annotations

import pytest

from ingest.markdown_dir import MarkdownDirIngester

CH = dict(target_tokens=768, overlap_pct=12, min_chunk_tokens=8,
          default_topic="test")


def _notes(tmp_path):
    (tmp_path / "keep.md").write_text("# Keep\n\nKeep body text here.\n", encoding="utf-8")
    (tmp_path / "skip.md").write_text("# Skip\n\nSkip body text here.\n", encoding="utf-8")
    return tmp_path


def test_without_only_paths_all_files_are_ingested(tmp_path):
    root = _notes(tmp_path)
    got = {c.source_path for c in MarkdownDirIngester(root=root, **CH).iter_chunks()}
    assert len(got) == 2


def test_only_paths_restricts_to_the_named_file(tmp_path):
    root = _notes(tmp_path)
    ing = MarkdownDirIngester(root=root, only_paths={root / "keep.md"}, **CH)
    got = {c.source_path for c in ing.iter_chunks()}
    assert len(got) == 1
    assert got.pop().endswith("keep.md")


def test_empty_only_paths_ingests_nothing(tmp_path):
    """An empty set means 'no files', not 'no filter' — refresh relies on
    this distinction when a source has no changes."""
    root = _notes(tmp_path)
    ing = MarkdownDirIngester(root=root, only_paths=set(), **CH)
    assert list(ing.iter_chunks()) == []
```

Add the equivalent three tests for `PdfDirIngester` and `EpubDirIngester` guarded by `pytest.importorskip("pymupdf")` / `pytest.importorskip("ebooklib")`, matching how the existing PDF/EPUB tests handle their optional dependencies. Read `tests/test_pdf_ingest.py` for the established pattern before writing them.

- [ ] **Step 2: Run to verify failure** — `TypeError: unexpected keyword argument 'only_paths'`.

- [ ] **Step 3: Implement** — add the parameter to each constructor and assign `self.only_paths = only_paths` (replacing the `= None` placeholder Task 1 left). Change nothing else.

- [ ] **Step 4: Full suite green.**

- [ ] **Step 5: Commit**

```bash
git add ingest/markdown_dir.py ingest/pdf_dir.py ingest/epub_dir.py tests/test_ingest_only_paths.py
git commit -m "feat(ingest): optional only_paths restriction on the directory ingesters"
```

---

### Task 4: Configured sources

**Files:**
- Modify: `core/pipeline.py` (add beside `chunking_params`), `config.yaml`
- Test: `tests/test_configured_sources.py` (create)

**Interfaces:**
- Produces: `core.pipeline.configured_sources(cfg: dict) -> list[dict]` returning validated entries `{"type": str, "path": Path}`. Valid types: `markdown`, `pdf`, `epub`, `zeal`. Task 5 consumes it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_configured_sources.py
"""config.yaml's sources: list — the record of what the index should contain."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.pipeline import configured_sources


def test_absent_or_empty_yields_nothing():
    assert configured_sources({}) == []
    assert configured_sources({"sources": None}) == []
    assert configured_sources({"sources": []}) == []


def test_entries_are_normalised(tmp_path):
    cfg = {"sources": [{"type": "markdown", "path": str(tmp_path)}]}
    got = configured_sources(cfg)
    assert got == [{"type": "markdown", "path": Path(tmp_path).resolve()}]


def test_unknown_type_is_rejected_by_name():
    with pytest.raises(ValueError, match="pdfs"):
        configured_sources({"sources": [{"type": "pdfs", "path": "/x"}]})


def test_missing_key_is_rejected():
    with pytest.raises(ValueError, match="path"):
        configured_sources({"sources": [{"type": "markdown"}]})


def test_non_mapping_entry_is_rejected():
    with pytest.raises(ValueError, match="mapping"):
        configured_sources({"sources": ["D:/notes"]})
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement** in `core/pipeline.py`, directly below `chunking_params`:

```python
SOURCE_TYPES = ("markdown", "pdf", "epub", "zeal")


def configured_sources(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The `sources:` list from config.yaml, validated and normalised.

    This is the record of what the index is SUPPOSED to contain. `rag ingest`
    stays invocation-driven for one-offs; `rag refresh` works from here and
    never guesses.

    Raises ValueError on a malformed entry rather than skipping it — a typo in
    a source type must not silently mean "that corpus is no longer indexed".
    """
    raw = cfg.get("sources") or []
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"sources[{i}]: each entry must be a mapping of type + path")
        if "type" not in entry or "path" not in entry:
            raise ValueError(f"sources[{i}]: needs both 'type' and 'path'")
        if entry["type"] not in SOURCE_TYPES:
            raise ValueError(
                f"sources[{i}]: unknown type {entry['type']!r}; "
                f"expected one of {', '.join(SOURCE_TYPES)}"
            )
        out.append({"type": entry["type"], "path": Path(entry["path"]).resolve()})
    return out
```

- [ ] **Step 4: Add the config block.** Append to `config.yaml`:

```yaml
# ---- Sources -----------------------------------------------------------------
# What the index is supposed to contain. `rag refresh` re-ingests only the
# files here whose content changed; `rag ingest --markdown X` still works for
# one-offs. Leave empty until you know what you want tracked.
sources: []
#  - type: markdown          # markdown | pdf | epub | zeal
#    path: D:/notes
#  - type: pdf
#    path: D:/papers
```

- [ ] **Step 5: Full suite green, then commit**

```bash
git add core/pipeline.py config.yaml tests/test_configured_sources.py
git commit -m "feat(config): a sources list so the index records what it should contain"
```

---

### Task 5: The refresh engine

**Files:**
- Create: `core/refresh.py`
- Test: `tests/test_refresh.py` (create)

**Interfaces:**
- Consumes: `core.walk.iter_source_files` (Task 1); `metadata.source_hashes()`, `metadata.delete_source()`, `core.metadata.hash_file` (Task 2); `only_paths=` constructors (Task 3); `core.pipeline.configured_sources` (Task 4); plus existing `chunking_params`, `pipeline.ingest`, `store.delete_by_source`.
- Produces:
  - `SUFFIXES: dict[str, set[str]]` mapping source type → suffixes (`markdown` → `{".md", ".markdown", ".py"}`, `pdf` → `{".pdf"}`, `epub` → `{".epub"}`).
  - `@dataclass SourcePlan` with fields `type: str`, `path: Path`, `new: list[Path]`, `changed: list[Path]`, `unchanged: int`, `vanished: list[str]`, `root_missing: bool`, and a property `has_work -> bool` (True when `new` or `changed` is non-empty).
  - `@dataclass RefreshPlan` with `sources: list[SourcePlan]` and properties `has_work -> bool`, `has_missing_roots -> bool`.
  - `plan_refresh(cfg, metadata) -> RefreshPlan` — filesystem + database reads only, never mutates.
  - `run_refresh(cfg, metadata, store, embedder_factory, *, prune=False, dry_run=False) -> RefreshPlan` — applies the plan and returns it. `embedder_factory` is a zero-argument callable, invoked **only if** there is work to do.
- Task 6 consumes `plan_refresh`, `run_refresh` and the dataclasses.

**Zeal:** the docset's `.dsidx` file is the hashed unit (`<docset>/Contents/Resources/docSet.dsidx`). It is "changed" or not; there is no per-page granularity.

**Vanished detection:** a recorded `source_path` counts as vanished only when its source root is present AND the file is gone. When `root_missing` is True, `vanished` MUST be empty — an unplugged drive is not an emptied corpus.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_refresh.py
"""Refresh: plan before acting, skip what hasn't changed, never delete by accident."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core import refresh
from core.metadata import hash_file


def _cfg(tmp_path, stype="markdown"):
    return {"sources": [{"type": stype, "path": str(tmp_path)}],
            "chunking": {"target_tokens": 768, "overlap_pct": 12,
                         "min_chunk_tokens": 8},
            "ingest": {"default_topic": "test"}}


def _md(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(f"# {name}\n\n{body}\n", encoding="utf-8")
    return p


def _meta(hashes):
    m = MagicMock()
    m.source_hashes.return_value = dict(hashes)
    m.delete_source.return_value = True
    return m


# -- planning -----------------------------------------------------------------

def test_new_file_is_planned_as_new(tmp_path):
    f = _md(tmp_path, "a.md", "body")
    plan = refresh.plan_refresh(_cfg(tmp_path), _meta({}))
    assert plan.sources[0].new == [f]
    assert plan.sources[0].changed == []
    assert plan.has_work is True


def test_unchanged_file_is_skipped(tmp_path):
    f = _md(tmp_path, "a.md", "body")
    plan = refresh.plan_refresh(_cfg(tmp_path), _meta({str(f): hash_file(f)}))
    assert plan.sources[0].new == []
    assert plan.sources[0].changed == []
    assert plan.sources[0].unchanged == 1
    assert plan.has_work is False


def test_edited_file_is_planned_as_changed(tmp_path):
    f = _md(tmp_path, "a.md", "body")
    stale = _meta({str(f): "0000000000000000"})
    plan = refresh.plan_refresh(_cfg(tmp_path), stale)
    assert plan.sources[0].changed == [f]


def test_row_without_a_file_hash_is_treated_as_changed(tmp_path):
    """Rows written before schema v2 have file_hash NULL — re-ingest once."""
    f = _md(tmp_path, "a.md", "body")
    plan = refresh.plan_refresh(_cfg(tmp_path), _meta({str(f): None}))
    assert plan.sources[0].changed == [f]


def test_vanished_file_is_detected(tmp_path):
    _md(tmp_path, "a.md", "body")
    gone = str(tmp_path / "gone.md")
    plan = refresh.plan_refresh(_cfg(tmp_path), _meta({gone: "abc"}))
    assert plan.sources[0].vanished == [gone]


def test_missing_root_reports_nothing_vanished(tmp_path):
    """THE safety property: an unplugged drive must not look like an emptied
    corpus, or --prune would wipe the index."""
    missing = tmp_path / "not_mounted"
    cfg = {"sources": [{"type": "markdown", "path": str(missing)}]}
    recorded = str(missing / "a.md")
    plan = refresh.plan_refresh(cfg, _meta({recorded: "abc"}))
    sp = plan.sources[0]
    assert sp.root_missing is True
    assert sp.vanished == []
    assert plan.has_missing_roots is True


def test_only_files_under_this_root_are_considered(tmp_path):
    """A source row belonging to a different root must not be called vanished."""
    other = str(Path("/somewhere/else/x.md"))
    _md(tmp_path, "a.md", "body")
    plan = refresh.plan_refresh(_cfg(tmp_path), _meta({other: "abc"}))
    assert plan.sources[0].vanished == []


# -- applying -----------------------------------------------------------------

def test_no_work_means_no_embedder_is_ever_built(tmp_path):
    """The common case must be nearly free — no GPU model load."""
    f = _md(tmp_path, "a.md", "body")
    factory = MagicMock()
    refresh.run_refresh(_cfg(tmp_path), _meta({str(f): hash_file(f)}),
                        store=MagicMock(), embedder_factory=factory)
    factory.assert_not_called()


def test_work_builds_the_embedder_once(tmp_path, monkeypatch):
    _md(tmp_path, "a.md", "body")
    factory = MagicMock()
    monkeypatch.setattr(refresh, "ingest_pipeline", MagicMock(return_value=1))
    refresh.run_refresh(_cfg(tmp_path), _meta({}),
                        store=MagicMock(), embedder_factory=factory)
    assert factory.call_count == 1


def test_ingest_is_restricted_to_the_changed_files(tmp_path, monkeypatch):
    keep = _md(tmp_path, "keep.md", "body")
    other = _md(tmp_path, "other.md", "body")
    fake = MagicMock(return_value=1)
    monkeypatch.setattr(refresh, "ingest_pipeline", fake)
    refresh.run_refresh(_cfg(tmp_path),
                        _meta({str(other): hash_file(other)}),
                        store=MagicMock(), embedder_factory=MagicMock())
    ingester = fake.call_args.args[0]
    assert ingester.only_paths == {keep}


def test_dry_run_changes_nothing(tmp_path, monkeypatch):
    _md(tmp_path, "a.md", "body")
    fake = MagicMock()
    monkeypatch.setattr(refresh, "ingest_pipeline", fake)
    store = MagicMock()
    factory = MagicMock()
    plan = refresh.run_refresh(_cfg(tmp_path), _meta({}), store=store,
                               embedder_factory=factory, dry_run=True)
    fake.assert_not_called()
    factory.assert_not_called()
    store.delete_by_source.assert_not_called()
    assert plan.has_work is True          # it still REPORTS the work


def test_prune_deletes_vanished_files_only_when_asked(tmp_path, monkeypatch):
    _md(tmp_path, "a.md", "body")
    gone = str(tmp_path / "gone.md")
    monkeypatch.setattr(refresh, "ingest_pipeline", MagicMock(return_value=1))
    store = MagicMock()
    meta = _meta({gone: "abc"})

    refresh.run_refresh(_cfg(tmp_path), meta, store=store,
                        embedder_factory=MagicMock(), prune=False)
    store.delete_by_source.assert_not_called()

    refresh.run_refresh(_cfg(tmp_path), _meta({gone: "abc"}), store=store,
                        embedder_factory=MagicMock(), prune=True)
    store.delete_by_source.assert_called_once_with(gone)


def test_prune_never_touches_a_missing_root(tmp_path):
    missing = tmp_path / "not_mounted"
    cfg = {"sources": [{"type": "markdown", "path": str(missing)}]}
    store = MagicMock()
    refresh.run_refresh(cfg, _meta({str(missing / "a.md"): "abc"}),
                        store=store, embedder_factory=MagicMock(), prune=True)
    store.delete_by_source.assert_not_called()


def test_refresh_invalidates_the_hybrid_cache(tmp_path, monkeypatch):
    """The corpus changed; a stale IDF map would skew the next hybrid query."""
    from core import pipeline

    _md(tmp_path, "a.md", "body")
    monkeypatch.setattr(refresh, "ingest_pipeline", MagicMock(return_value=1))
    pipeline._hybrid_cache.update({"vocab": {"x": 0}, "idf": {"x": 1.0}, "built": True})
    refresh.run_refresh(_cfg(tmp_path), _meta({}), store=MagicMock(),
                        embedder_factory=MagicMock())
    assert pipeline._hybrid_cache["built"] is False
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement `core/refresh.py`**

Structure it as: `SUFFIXES` map; `_zeal_index(path)` returning the `.dsidx` path; `_plan_one(entry, recorded)` building one `SourcePlan`; `plan_refresh` looping entries; `_make_ingester(entry, cfg, only_paths)` constructing the right ingester with `chunking_params(cfg)`; `run_refresh` applying.

Key requirements the tests pin, in prose so you write the code rather than transcribe it:

- `plan_refresh` reads `metadata.source_hashes()` once, then per source: if the root does not exist, set `root_missing=True`, leave `new`/`changed`/`vanished` empty, and continue. Otherwise enumerate via `iter_source_files` (Zeal: the single `.dsidx` path), and for each file compare `hash_file(f)` against the recorded hash — absent from the map means **new**, present-but-different or `None` means **changed**, equal means **unchanged** (count only).
- `vanished` is every recorded `source_path` that resolves under this root and no longer exists on disk. Compare resolved paths; do not compare strings naively.
- `run_refresh` calls `plan_refresh` first. If `dry_run`, return the plan untouched. If the plan has work, call `embedder_factory()` **once** and reuse the embedder for every source. If there is no work and no prune to do, never call it.
- For each source with work, build the ingester with `only_paths=set(new + changed)` and call `ingest_pipeline(ingester, embedder, store, metadata=metadata)`. Import it as `from core.pipeline import ingest as ingest_pipeline` at module level so tests can monkeypatch `refresh.ingest_pipeline`.
- Zeal ingests whole (no `only_paths`), and its recorded `source_path` values are per page, so its `vanished` is always empty — the docset either changed or it did not.
- When `prune` and not `dry_run`, for each `vanished` path call `store.delete_by_source(path)` then `metadata.delete_source(path)`.
- Finally, call `pipeline.invalidate_hybrid_cache()` whenever anything was ingested or pruned.
- Log a WARNING naming each missing root.

- [ ] **Step 4: Full suite green.**

- [ ] **Step 5: Commit**

```bash
git add core/refresh.py tests/test_refresh.py
git commit -m "feat(refresh): plan-then-apply engine with a missing-root prune guard"
```

---

### Task 6: `rag refresh` and `rag forget`

**Files:**
- Modify: `cli.py`
- Test: `tests/test_refresh_cli.py` (create)

**Interfaces:**
- Consumes: Task 5's `plan_refresh` / `run_refresh` / `RefreshPlan`; `metadata.get_sources()`, `metadata.delete_source()`; `store.delete_by_source()`.
- Produces: `rag refresh [--prune] [--dry-run] [--yes]`; `rag forget (--source PATH | --topic NAME) [--yes]`.

Follow the established command shape: `@cli.command()`, `@click.pass_context`, `cfg = ctx.obj["config"]`, lazy imports of heavy modules inside the body, and `metadata.close()` on every return path.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_refresh_cli.py
"""CLI surface for refresh and forget. Deleting paths must be opt-in."""
from __future__ import annotations

from unittest.mock import MagicMock

from click.testing import CliRunner


def _patch(monkeypatch, store=None, metadata=None):
    import cli as cli_mod
    monkeypatch.setattr(cli_mod, "make_embedder", lambda cfg: MagicMock())
    monkeypatch.setattr(cli_mod, "make_store", lambda cfg: store or MagicMock())
    monkeypatch.setattr(cli_mod, "make_metadata", lambda cfg: metadata or MagicMock())
    return cli_mod


def test_refresh_reports_when_nothing_changed(monkeypatch):
    cli_mod = _patch(monkeypatch)
    plan = MagicMock(has_work=False, has_missing_roots=False, sources=[])
    monkeypatch.setattr("core.refresh.run_refresh", lambda *a, **k: plan)
    res = CliRunner().invoke(cli_mod.cli, ["refresh"])
    assert res.exit_code == 0
    assert "up to date" in res.output.lower()


def test_refresh_exits_nonzero_when_a_root_is_missing(monkeypatch):
    """A source that vanished from disk is an operational problem, not a no-op."""
    cli_mod = _patch(monkeypatch)
    plan = MagicMock(has_work=False, has_missing_roots=True, sources=[])
    monkeypatch.setattr("core.refresh.run_refresh", lambda *a, **k: plan)
    res = CliRunner().invoke(cli_mod.cli, ["refresh"])
    assert res.exit_code == 1


def test_refresh_prune_requires_confirmation(monkeypatch):
    cli_mod = _patch(monkeypatch)
    called = {}
    def fake(*a, **k):
        called.update(k)
        return MagicMock(has_work=False, has_missing_roots=False, sources=[])
    monkeypatch.setattr("core.refresh.run_refresh", fake)
    res = CliRunner().invoke(cli_mod.cli, ["refresh", "--prune"], input="n\n")
    assert res.exit_code != 0
    assert called == {}          # aborted before doing anything


def test_refresh_prune_with_yes_skips_the_prompt(monkeypatch):
    cli_mod = _patch(monkeypatch)
    seen = {}
    def fake(*a, **k):
        seen.update(k)
        return MagicMock(has_work=False, has_missing_roots=False, sources=[])
    monkeypatch.setattr("core.refresh.run_refresh", fake)
    res = CliRunner().invoke(cli_mod.cli, ["refresh", "--prune", "--yes"])
    assert res.exit_code == 0
    assert seen["prune"] is True


def test_forget_source_deletes_chunks_and_row(monkeypatch):
    store, meta = MagicMock(), MagicMock()
    store.delete_by_source.return_value = 7
    meta.get_sources.return_value = [{"source_path": "/a.md", "topic": "t"}]
    cli_mod = _patch(monkeypatch, store=store, metadata=meta)
    res = CliRunner().invoke(cli_mod.cli, ["forget", "--source", "/a.md", "--yes"])
    assert res.exit_code == 0
    store.delete_by_source.assert_called_once_with("/a.md")
    meta.delete_source.assert_called_once_with("/a.md")


def test_forget_topic_deletes_every_source_with_that_topic(monkeypatch):
    store, meta = MagicMock(), MagicMock()
    store.delete_by_source.return_value = 1
    meta.get_sources.return_value = [
        {"source_path": "/a.md", "topic": "photography"},
        {"source_path": "/b.md", "topic": "photography"},
        {"source_path": "/c.md", "topic": "ml"},
    ]
    cli_mod = _patch(monkeypatch, store=store, metadata=meta)
    res = CliRunner().invoke(cli_mod.cli, ["forget", "--topic", "photography", "--yes"])
    assert res.exit_code == 0
    deleted = {c.args[0] for c in store.delete_by_source.call_args_list}
    assert deleted == {"/a.md", "/b.md"}


def test_forget_without_confirmation_deletes_nothing(monkeypatch):
    store, meta = MagicMock(), MagicMock()
    meta.get_sources.return_value = [{"source_path": "/a.md", "topic": "t"}]
    cli_mod = _patch(monkeypatch, store=store, metadata=meta)
    res = CliRunner().invoke(cli_mod.cli, ["forget", "--source", "/a.md"], input="n\n")
    assert res.exit_code != 0
    store.delete_by_source.assert_not_called()


def test_forget_requires_exactly_one_selector(monkeypatch):
    cli_mod = _patch(monkeypatch)
    both = CliRunner().invoke(cli_mod.cli, ["forget", "--source", "/a", "--topic", "t"])
    neither = CliRunner().invoke(cli_mod.cli, ["forget"])
    assert both.exit_code != 0 and neither.exit_code != 0


def test_forget_unknown_target_exits_nonzero(monkeypatch):
    store, meta = MagicMock(), MagicMock()
    meta.get_sources.return_value = []
    cli_mod = _patch(monkeypatch, store=store, metadata=meta)
    res = CliRunner().invoke(cli_mod.cli, ["forget", "--source", "/nope.md", "--yes"])
    assert res.exit_code == 1
    store.delete_by_source.assert_not_called()
```

- [ ] **Step 2: Run to verify failure** — no such command.

- [ ] **Step 3: Implement both commands in `cli.py`.**

`refresh`: build store and metadata (not the embedder — pass `lambda: make_embedder(cfg)` as the factory so it stays unbuilt when there is no work). Confirm before `--prune` unless `--yes`, naming what will be removed. Print a per-source summary line (new / changed / unchanged / vanished). Exit 1 if `plan.has_missing_roots`, after printing which roots. Close metadata on every path.

`forget`: exactly one of `--source` / `--topic` (raise `click.UsageError` otherwise). Resolve the target to a list of source paths via `metadata.get_sources()`; if empty, print a clear message and `sys.exit(1)`. Show what will be deleted, `click.confirm(..., abort=True)` unless `--yes`, then for each path `store.delete_by_source(path)` and `metadata.delete_source(path)`, printing the total chunks removed. Invalidate the hybrid cache afterwards (`from core.pipeline import invalidate_hybrid_cache`).

Note `get_sources()` defaults to `limit=50` — pass a large limit explicitly so forget cannot silently miss sources beyond the first fifty.

- [ ] **Step 4: Full suite green.**

- [ ] **Step 5: Commit**

```bash
git add cli.py tests/test_refresh_cli.py
git commit -m "feat(cli): rag refresh and rag forget"
```

---

### Task 7: Scheduled task, eval index size, and docs

**Files:**
- Create: `scripts/install-refresh-task.ps1`, `scripts/uninstall-refresh-task.ps1`
- Modify: `scripts/_config.ps1`, `eval/run_ragas.py`, `README.md`, `docs/superpowers/specs/2026-08-10-p3-runbook.md`
- Test: `tests/test_refresh_cli.py` (append one test for the eval change)

**Interfaces:**
- Consumes: `scripts/_config.ps1`'s existing shared constants; `eval/run_ragas.run`'s existing `run_params` dict.
- Produces: `$Script:RefreshTaskName = 'rag-refresh'` in `_config.ps1`; the two scripts; `params["index_points"]` recorded per eval run.

- [ ] **Step 1: Write the failing test** (append to `tests/test_refresh_cli.py`)

```python
def test_eval_records_the_index_size(monkeypatch, tmp_path):
    """Without this, a recall trend across months reads as degradation when
    it is really just a growing corpus."""
    import json
    from unittest.mock import MagicMock
    from eval import run_ragas

    golden = tmp_path / "g.jsonl"
    golden.write_text(
        json.dumps({"question": "q", "relevant_chunk_ids": ["c1"]}) + "\n",
        encoding="utf-8")

    result = MagicMock()
    result.answer = "a"
    result.citations = [{"chunk_id": "c1", "text": "t",
                         "source_path": "/a.md", "section": "S"}]
    result.dense_hits = list(result.citations)
    monkeypatch.setattr(run_ragas, "ask_pipeline", lambda *a, **k: result)

    store = MagicMock()
    store.count.return_value = 4242
    metadata = MagicMock()
    run_ragas.run(golden, embedder=MagicMock(), store=store,
                  reranker=MagicMock(), generator=None, metadata=metadata)
    assert metadata.record_eval_run.call_args.kwargs["params"]["index_points"] == 4242
```

- [ ] **Step 2: Run to verify failure** — KeyError.

- [ ] **Step 3: Implement the eval change** — in `eval/run_ragas.py`, add `index_points` to the `run_params` dict, reading `store.count()` inside a `try/except Exception` that falls back to `None` (a store that cannot report a count must not break an eval run).

- [ ] **Step 4: Write the PowerShell scripts.**

Add to `scripts/_config.ps1`: `$Script:RefreshTaskName = 'rag-refresh'`.

`scripts/install-refresh-task.ps1` — dot-source `_config.ps1`; register a scheduled task named `$Script:RefreshTaskName` running the repo's `rag.ps1 refresh` daily at a `-At` time taken from a `[string]$At = '03:00'` parameter; `-RunLevel Limited` (**not** `Highest` — the P2 audit found the existing task requesting elevation it does not need); unregister any existing task of that name first so re-running is idempotent; print what it registered.

`scripts/uninstall-refresh-task.ps1` — dot-source `_config.ps1`; unregister the task if present, report either way. Model both on the existing `install-service.ps1` / `uninstall-service.ps1` for structure and style.

- [ ] **Step 5: Documentation.**

In `README.md`, directly after the "Ingesting your own notes" section, add a short "Keeping the index current" section: the `sources:` config block, `rag refresh`, `rag refresh --prune`, `rag forget`, and the scheduled task. State plainly that prune and forget are the only commands that delete, and that a missing source root is treated as "unknown" rather than "empty" so an unplugged drive cannot cause a prune.

In the P3.1 runbook, add one line noting that `rag refresh` now keeps the corpus current between eval runs, and that a materially changed corpus is the trigger to re-run `rag golden generate` and the sweep.

- [ ] **Step 6: Full suite green.**

- [ ] **Step 7: Commit**

```bash
git add scripts/_config.ps1 scripts/install-refresh-task.ps1 scripts/uninstall-refresh-task.ps1 eval/run_ragas.py README.md docs/superpowers/specs/2026-08-10-p3-runbook.md tests/test_refresh_cli.py
git commit -m "feat(ops): scheduled refresh task, eval index size, docs"
```

---

## Self-Review (done at authoring time)

- **Spec coverage:** sources list (T4), file_hash + v2 migration (T2), shared walk (T3's enabler, T1), `rag refresh` with prune and dry-run (T5, T6), `rag forget` (T6), scheduled task (T7), eval index size (T7), missing-root guard (T5 and T6 both test it), hybrid cache invalidation (T5). No spec section is unimplemented.
- **Ordering:** T1 must precede T3 (the walk gains `only_paths` before the constructors expose it); T2 before T5 (hashing); T4 before T5 (source list); T5 before T6 (engine before CLI). T7 is independent and last.
- **Type consistency:** `only_paths: set[Path] | None` is identical in T1, T3 and T5. `source_hashes() -> dict[str, str | None]` in T2 matches T5's `_meta` fixture. `SourcePlan` / `RefreshPlan` field names in T5 match the attributes T6's tests access (`has_work`, `has_missing_roots`, `sources`).
- **Known deviation from the current code:** the extracted walk sorts markdown results, which the existing `_walk` did not. Deliberate — deterministic ingest order — and called out in T1.
- **Placeholder scan:** clean. T5's Step 3 is prose rather than a full code block by design: its behaviour is pinned by fourteen tests written out in full, and transcribing an implementation would invite the implementer to match my structure rather than the tests. Every other code step is runnable as written.
