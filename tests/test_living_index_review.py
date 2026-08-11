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


# -----------------------------------------------------------------------------
# IMPORTANT — poison-file amplification, on the unattended path
#
# `_refresh_one` deleted every changed file's chunks before ingesting any of
# them, so one repeatably-failing file deleted every OTHER changed file in the
# source and did not restore them. `_mark_for_reingest` then cleared their
# hashes, so the next nightly run deleted them again. A permanently poisoned
# file made the hole permanent.
# -----------------------------------------------------------------------------

class _PoisonStore(_MemStore):
    """A store that refuses any upsert carrying chunks from a named file.

    Stands in for an outsized document that OOMs the embedder or a write the
    backend rejects: the file fails the same way on every single run, which is
    what turns a one-run hole into a permanent one.
    """

    def __init__(self):
        super().__init__()
        self.poison: set[str] = set()

    def upsert_chunks(self, chunks, vectors):
        if any(c.source_path in self.poison for c in chunks):
            raise RuntimeError("that file OOMs the embedder")
        return super().upsert_chunks(chunks, vectors)


def _poisoned(tmp_path, meta_store):
    """Two indexed markdown files, then one of them turns poisonous and both
    are edited — the state a nightly refresh walks into."""
    notes = tmp_path / "notes"
    notes.mkdir()
    good = _md(notes, "good.md", "gamma " * 60)
    bad = _md(notes, "bad.md", "delta " * 60)
    cfg = _cfg(notes)
    store = _PoisonStore()

    refresh.run_refresh(cfg, meta_store, store=store, embedder_factory=_TinyEmbedder)
    assert any(c.source_path == str(good) for c in store.points.values())

    store.poison = {str(bad)}
    good.write_text("# good.md\n\n" + "epsilon " * 60 + "\n", encoding="utf-8")
    bad.write_text("# bad.md\n\n" + "zeta " * 60 + "\n", encoding="utf-8")
    return cfg, store, good, bad


def test_one_poison_file_does_not_take_its_siblings_with_it(tmp_path, meta_store):
    cfg, store, good, bad = _poisoned(tmp_path, meta_store)

    plan = refresh.run_refresh(cfg, meta_store, store=store,
                               embedder_factory=_TinyEmbedder)

    assert plan.sources[0].error, "the run still has to report the failure"
    surviving = {c.source_path for c in store.points.values()}
    assert str(good) in surviving, "the healthy sibling was deleted and not restored"
    assert any("epsilon" in c.text for c in store.points.values()), \
        "and it holds the NEW content, not a stale copy"


def test_a_poison_file_does_not_clear_a_healthy_siblings_hash(tmp_path, meta_store):
    """Otherwise the next run deletes and re-embeds the sibling all over
    again — every night, forever, for a file that is perfectly fine."""
    cfg, store, good, bad = _poisoned(tmp_path, meta_store)

    refresh.run_refresh(cfg, meta_store, store=store, embedder_factory=_TinyEmbedder)

    hashes = meta_store.source_hashes()
    assert hashes[str(good)] == hash_file(good), "the healthy file's hash was cleared"
    replan = refresh.plan_refresh(cfg, meta_store, store)
    assert replan.sources[0].changed == [bad]
    assert replan.sources[0].unchanged == 1


def test_the_failure_is_still_reported_against_the_source(tmp_path, meta_store):
    """Per-file isolation must not quietly swallow the failure: the scheduled
    task's exit code is the only thing the operator sees."""
    cfg, store, good, bad = _poisoned(tmp_path, meta_store)

    plan = refresh.run_refresh(cfg, meta_store, store=store,
                               embedder_factory=_TinyEmbedder)

    assert plan.has_errors is True
    assert "OOMs the embedder" in (plan.sources[0].error or "")
    assert str(bad) in (plan.sources[0].error or ""), \
        "and it names the file, which is the whole point of isolating them"


# -----------------------------------------------------------------------------
# IMPORTANT — `metadata.path` was cwd-relative while config resolution is
# repo-anchored
#
# `rag.ps1` tells the user to put `rag` on PATH and run it from anywhere. Run
# from any directory other than the repo, `make_metadata` opened — or CREATED —
# an empty database there. Everything still "worked": refresh saw the whole
# corpus as new and started a full re-embed, reporting success throughout. The
# branch's headline promise inverted into hours of GPU time, silently.
# -----------------------------------------------------------------------------

def test_metadata_path_is_anchored_to_the_config_not_the_cwd(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg_file = repo / "config.yaml"
    cfg_file.write_text(
        "metadata:\n  enabled: true\n  path: ./metadata.sqlite3\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    cfg = pipeline.load_config(cfg_file)
    md = pipeline.make_metadata(cfg)
    try:
        assert md.path == repo / "metadata.sqlite3"
    finally:
        md.close()
    assert not (elsewhere / "metadata.sqlite3").exists(), \
        "an empty database was created in the cwd; the corpus now reads as new"


def test_an_absolute_metadata_path_is_left_exactly_as_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    absolute = (tmp_path / "elsewhere" / "meta.sqlite3").resolve()
    cfg = {"metadata": {"path": str(absolute)}}
    assert pipeline.resolve_metadata_path(cfg) == absolute


def test_a_config_dict_with_no_file_behind_it_falls_back_to_the_repo(tmp_path, monkeypatch):
    """Hand-built config dicts (tests, embedded callers) have no config file to
    anchor to. The repo root is the right fallback: it is where config.yaml and
    the owner's live metadata.sqlite3 both are — never the cwd."""
    monkeypatch.chdir(tmp_path)
    resolved = pipeline.resolve_metadata_path({"metadata": {"path": "./metadata.sqlite3"}})
    assert resolved == pipeline.DEFAULT_CONFIG_PATH.parent / "metadata.sqlite3"
    assert resolved.parent != tmp_path


def test_the_catalog_commands_resolve_the_same_path(tmp_path, monkeypatch):
    """`rag sources` / `rag stats` open the store themselves. A second
    resolution is a second answer — and pointing them at an empty DB in the
    cwd would report a corpus of zero files that is sitting there indexed."""
    import cli as cli_mod

    repo = tmp_path / "repo"
    repo.mkdir()
    cfg_file = repo / "config.yaml"
    cfg_file.write_text("metadata:\n  path: ./meta.sqlite3\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    md = cli_mod._open_metadata(pipeline.load_config(cfg_file))
    try:
        assert md.path == repo / "meta.sqlite3"
    finally:
        md.close()
