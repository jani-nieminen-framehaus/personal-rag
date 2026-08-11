"""P3.2 whole-branch review fixes.

These findings live in the JOINS between the seven tasks — the places a
per-task review structurally cannot see, because each task's own reviewer
only ever looked at one side of the seam:

- `ingest --recreate` drops the collection while the catalog keeps claiming
  hashes for it, and `rag refresh` — the one thing that could restore the
  index — reads those hashes and reports "up to date".
- refresh deletes every changed file's chunks before ingesting any of them,
  so one poison file takes its siblings down with it, nightly.
- `metadata.path` is cwd-relative while config resolution is repo-anchored.
- a `rag serve` process never learns the corpus moved under it.
- `rag forget` is undone by the next refresh, silently.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from core import pipeline, refresh
from core.metadata import MetadataStore, hash_file
from tests.test_refresh import _MemStore, _TinyEmbedder, _cfg, _md, _multi_cfg


@pytest.fixture
def meta_store(tmp_path):
    s = MetadataStore(tmp_path / "meta.sqlite3")
    yield s
    s.close()


def _markdown_ingester(root):
    from ingest.markdown_dir import MarkdownDirIngester
    return MarkdownDirIngester(
        root=root, target_tokens=768, overlap_pct=12,
        min_chunk_tokens=8, default_topic="test",
    )


# -----------------------------------------------------------------------------
# CRITICAL — `ingest --recreate` strands the index while refresh insists
# nothing is wrong.
#
# `--recreate` drops the WHOLE collection and re-ingests only the sources named
# on that command line. Every other configured source keeps a `sources` row
# claiming a valid hash for chunks that no longer exist, so the next refresh
# matches file to hash and prints "index is up to date". The store's own dim
# check tells the user to run `rag ingest --recreate` when the embedder
# changes — following that advice silently loses every source not named.
# -----------------------------------------------------------------------------

def test_recreate_clears_the_catalog_of_the_sources_it_dropped(tmp_path, meta_store):
    """The collection is now empty, so a row claiming a hash is simply false."""
    notes = tmp_path / "notes"
    notes.mkdir()
    _md(notes, "a.md", "alpha " * 40)
    # A row for a source that is NOT part of this ingest — a PDF library, a
    # docset, anything the user indexed last month.
    meta_store.record_source("D:/papers/old.pdf", "pdf", "papers", 12,
                             "content", file_hash="deadbeefdeadbeef")

    pipeline.ingest(_markdown_ingester(notes), _TinyEmbedder(), _MemStore(),
                    recreate=True, metadata=meta_store)

    rows = {r["source_path"] for r in meta_store.get_sources()}
    assert "D:/papers/old.pdf" not in rows, "the catalog still claims dropped chunks"
    assert str(notes / "a.md") in rows, "and what WAS re-ingested is recorded"


def test_a_plain_ingest_leaves_the_catalog_alone(tmp_path, meta_store):
    """Only a drop justifies a wipe. An ordinary ingest adds."""
    notes = tmp_path / "notes"
    notes.mkdir()
    _md(notes, "a.md", "alpha " * 40)
    meta_store.record_source("D:/papers/old.pdf", "pdf", "papers", 12,
                             "content", file_hash="deadbeefdeadbeef")

    pipeline.ingest(_markdown_ingester(notes), _TinyEmbedder(), _MemStore(),
                    recreate=False, metadata=meta_store)

    rows = {r["source_path"] for r in meta_store.get_sources()}
    assert "D:/papers/old.pdf" in rows


def test_recreate_without_a_metadata_store_is_still_fine():
    """`eval --sweep` recreates a scratch collection with metadata=None."""
    notes_store = _MemStore()
    pipeline.ingest(MagicMock(iter_chunks=lambda: iter(())), _TinyEmbedder(),
                    notes_store, recreate=True, metadata=None)


def test_cli_recreate_clears_the_catalog(tmp_path, monkeypatch, meta_store):
    """End to end through the command the store's dim error tells you to run."""
    import cli as cli_mod

    notes = tmp_path / "notes"
    notes.mkdir()
    _md(notes, "a.md", "alpha " * 40)
    meta_store.record_source("D:/papers/old.pdf", "pdf", "papers", 12,
                             "content", file_hash="deadbeefdeadbeef")

    monkeypatch.setattr(cli_mod, "make_embedder", lambda cfg: _TinyEmbedder())
    monkeypatch.setattr(cli_mod, "make_store", lambda cfg: _MemStore())
    monkeypatch.setattr(cli_mod, "make_metadata", lambda cfg: meta_store)
    monkeypatch.setattr(meta_store, "close", lambda: None)

    res = CliRunner().invoke(
        cli_mod.cli, ["ingest", "--markdown", str(notes), "--recreate", "--yes"])
    assert res.exit_code == 0, res.output

    rows = {r["source_path"] for r in meta_store.get_sources()}
    assert "D:/papers/old.pdf" not in rows


def test_an_empty_index_is_never_up_to_date(tmp_path, meta_store):
    """The index-side counterpart of the disk-side guard in `_plan_one`: a
    store holding zero points while the catalog holds rows is UNKNOWN, not
    up to date. Reached by a dropped collection, a wiped Qdrant volume, or a
    `--recreate` that died before it wrote anything back."""
    notes = tmp_path / "notes"
    notes.mkdir()
    f = _md(notes, "a.md", "alpha " * 40)
    cfg = _cfg(notes)
    store = _MemStore()

    refresh.run_refresh(cfg, meta_store, store=store, embedder_factory=_TinyEmbedder)
    assert refresh.plan_refresh(cfg, meta_store, store).has_work is False

    store.points.clear()                       # the collection is gone
    plan = refresh.plan_refresh(cfg, meta_store, store)
    assert plan.has_work is True
    assert plan.sources[0].changed == [f]
    assert plan.sources[0].unchanged == 0


def test_an_empty_index_with_an_empty_catalog_is_not_suspicious(tmp_path, meta_store):
    """A brand-new install has both. Nothing to warn about, nothing to redo."""
    notes = tmp_path / "notes"
    notes.mkdir()
    plan = refresh.plan_refresh(_cfg(notes), meta_store, _MemStore())
    assert plan.has_work is False


def test_a_store_that_cannot_report_a_count_does_not_block_the_refresh(tmp_path, meta_store):
    """Fail open. A count endpoint that is unreachable must not turn every
    refresh into a full re-embed of the corpus."""
    notes = tmp_path / "notes"
    notes.mkdir()
    _md(notes, "a.md", "alpha " * 40)
    cfg = _cfg(notes)
    store = _MemStore()
    refresh.run_refresh(cfg, meta_store, store=store, embedder_factory=_TinyEmbedder)

    class _Mute(_MemStore):
        def count(self):
            raise RuntimeError("qdrant unreachable")

    mute = _Mute()
    mute.points = store.points
    assert refresh.plan_refresh(cfg, meta_store, mute).has_work is False
    # ...and so does a store with no count() at all.
    assert refresh.plan_refresh(cfg, meta_store, object()).has_work is False


def test_recreate_naming_one_source_leaves_work_for_the_others(tmp_path, meta_store):
    """The full sequence, as it actually happens: swap the embedding model,
    hit the dim error, follow the error's own advice with the one source you
    were thinking about — and the next refresh has to notice the rest."""
    notes = tmp_path / "notes"
    notes.mkdir()
    papers = tmp_path / "papers"
    papers.mkdir()
    _md(notes, "a.md", "alpha " * 40)
    b = _md(papers, "b.md", "beta " * 40)
    cfg = _multi_cfg([{"type": "markdown", "path": str(notes)},
                      {"type": "markdown", "path": str(papers)}])
    store = _MemStore()

    refresh.run_refresh(cfg, meta_store, store=store, embedder_factory=_TinyEmbedder)
    assert refresh.plan_refresh(cfg, meta_store, store).has_work is False

    # `rag ingest --markdown <notes> --recreate`: the whole collection goes,
    # and only `notes` comes back.
    pipeline.ingest(_markdown_ingester(notes), _TinyEmbedder(), store,
                    recreate=True, metadata=meta_store)
    assert not any(c.source_path == str(b) for c in store.points.values())

    plan = refresh.plan_refresh(cfg, meta_store, store)
    assert plan.has_work is True, "refresh called a stranded index up to date"
    assert plan.sources[1].new == [b]
