"""Refresh: plan before acting, skip what hasn't changed, never delete by accident."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core import refresh
from core.metadata import MetadataStore, hash_file


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


# -- convergence: the feature only works if a refresh settles -----------------
#
# The three tests below are NOT in the task brief. They exist because the
# brief's fourteen all mock `ingest_pipeline` away, which hides the fact that
# nothing in the ingest path ever wrote `file_hash` — so a refresh would have
# re-ingested the same unchanged files forever. See the report for detail.


class _TinyEmbedder:
    """Deterministic toy vectors. No model, no GPU, no network."""
    batch_size = 8

    def dim(self):
        return 3

    def embed(self, texts):
        return [[1.0, float(len(t) % 5), float(t.count("a") % 3)] for t in texts]

    def embed_documents(self, texts):
        return self.embed(texts)

    def embed_query(self, text):
        return self.embed([text])[0]


class _MemStore:
    """Just enough VectorStore surface to drive `pipeline.ingest`."""

    def __init__(self):
        self.collection = "mem"
        self.points: dict[str, object] = {}
        self.deleted: list[str] = []

    def ensure_collection(self, recreate=False, expected_dense_dim=None):
        if recreate:
            self.points.clear()

    def upsert_chunks(self, chunks, vectors):
        for c in chunks:
            self.points[c.chunk_id] = c
        return len(chunks)

    def delete_by_source(self, source_path: str) -> int:
        self.deleted.append(source_path)
        gone = [k for k, c in self.points.items() if c.source_path == source_path]
        for k in gone:
            del self.points[k]
        return len(gone)


@pytest.fixture
def meta_store(tmp_path):
    s = MetadataStore(tmp_path / "meta.sqlite3")
    yield s
    s.close()


def test_ingest_records_the_file_hash(tmp_path, meta_store):
    """Correction A. Nothing wrote `file_hash`, so every refresh re-ingested
    everything: `run_refresh` ingests through `pipeline.ingest`, whose
    `_record_sources()` omitted the argument and left the column NULL."""
    from core import pipeline
    from ingest.markdown_dir import MarkdownDirIngester

    notes = tmp_path / "notes"
    notes.mkdir()
    f = _md(notes, "a.md", "body text " * 40)
    ing = MarkdownDirIngester(
        root=notes, target_tokens=768, overlap_pct=12,
        min_chunk_tokens=8, default_topic="test",
    )
    pipeline.ingest(ing, _TinyEmbedder(), _MemStore(), metadata=meta_store)

    rows = meta_store.get_sources()
    assert len(rows) == 1
    assert rows[0]["file_hash"] == hash_file(f)
    assert rows[0]["file_hash"]          # non-empty


def test_record_source_without_a_file_hash_preserves_the_stored_one(meta_store):
    """Correction B. The upsert set `file_hash = excluded.file_hash`
    unconditionally, so a plain `rag ingest` (which omits it) erased a hash
    refresh had just recorded."""
    meta_store.record_source("/a.md", "markdown", "t", 3, "ch", file_hash="deadbeefdeadbeef")
    meta_store.record_source("/a.md", "markdown", "t", 4, "ch2")

    rows = meta_store.get_sources()
    assert len(rows) == 1
    assert rows[0]["file_hash"] == "deadbeefdeadbeef"
    assert rows[0]["chunk_count"] == 4     # everything else still upserts


def test_a_second_refresh_over_unchanged_markdown_does_nothing(tmp_path, meta_store):
    """End-to-end convergence: real ingest, real metadata, no mocks in the
    hash path. Without correction A this loops forever."""
    notes = tmp_path / "notes"
    notes.mkdir()
    _md(notes, "a.md", "body text " * 40)
    cfg = _cfg(notes)
    store = _MemStore()

    first = refresh.run_refresh(cfg, meta_store, store=store,
                                embedder_factory=_TinyEmbedder)
    assert first.has_work is True
    assert store.points

    # Planned separately so a regression fails here, on the claim itself,
    # rather than somewhere inside a re-ingest that should never have started.
    replan = refresh.plan_refresh(cfg, meta_store)
    assert replan.has_work is False
    assert replan.sources[0].unchanged == 1

    factory = MagicMock(side_effect=_TinyEmbedder)
    second = refresh.run_refresh(cfg, meta_store, store=store,
                                 embedder_factory=factory)
    assert second.has_work is False
    factory.assert_not_called()


def test_prune_removes_the_metadata_row_so_it_stays_pruned(tmp_path, meta_store):
    """Deleting the vectors but leaving the `sources` row would report the same
    file as vanished on every run forever — the same non-convergence as A/C."""
    notes = tmp_path / "notes"
    notes.mkdir()
    _md(notes, "a.md", "body text " * 40)
    doomed = _md(notes, "b.md", "other text " * 40)
    cfg = _cfg(notes)
    store = _MemStore()

    refresh.run_refresh(cfg, meta_store, store=store, embedder_factory=_TinyEmbedder)
    assert len(meta_store.get_sources()) == 2

    doomed.unlink()
    plan = refresh.run_refresh(cfg, meta_store, store=store,
                               embedder_factory=_TinyEmbedder, prune=True)
    assert plan.sources[0].vanished == [str(doomed)]
    assert store.deleted == [str(doomed)]
    assert [r["source_path"] for r in meta_store.get_sources()] == [str(notes / "a.md")]

    again = refresh.run_refresh(cfg, meta_store, store=store,
                                embedder_factory=_TinyEmbedder, prune=True)
    assert again.sources[0].vanished == []
    assert store.deleted == [str(doomed)]


# -- zeal ---------------------------------------------------------------------

def _make_fake_docset(root: Path) -> None:
    docs = root / "Contents" / "Resources" / "Documents"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "page_a.html").write_text(
        "<html><body><h1>Page A</h1><p>alpha content for testing</p></body></html>",
        encoding="utf-8",
    )
    idx = root / "Contents" / "Resources" / "docSet.dsidx"
    with sqlite3.connect(str(idx)) as conn:
        conn.execute("CREATE TABLE searchIndex (id INTEGER, name TEXT, type TEXT, path TEXT)")
        conn.execute("INSERT INTO searchIndex VALUES (1, 'Top of page A', 'Page', 'page_a.html')")
        conn.commit()


def test_zeal_docset_is_reingested_once_then_never_again(tmp_path, meta_store, monkeypatch):
    """Correction C. `sources` rows for a docset are per PAGE, so nothing was
    ever keyed by the `.dsidx` the plan hashes — every refresh re-ingested a
    docset that can hold ~50k pages. `run_refresh` records a marker row."""
    docset = tmp_path / "Test.docset"
    docset.mkdir()
    _make_fake_docset(docset)
    cfg = _cfg(docset, stype="zeal")

    ingested: list[object] = []
    monkeypatch.setattr(
        refresh, "ingest_pipeline",
        lambda ing, *a, **k: (ingested.append(ing), 1)[1],
    )

    first = refresh.run_refresh(cfg, meta_store, store=MagicMock(),
                                embedder_factory=MagicMock())
    assert first.has_work is True
    assert len(ingested) == 1

    second = refresh.run_refresh(cfg, meta_store, store=MagicMock(),
                                 embedder_factory=MagicMock())
    assert second.has_work is False
    assert second.sources[0].unchanged == 1
    assert len(ingested) == 1


def test_zeal_page_rows_are_never_reported_vanished(tmp_path):
    """A docset is all-or-nothing: its per-page rows are not the unit refresh
    tracks, and calling them vanished would let --prune shred the docset."""
    docset = tmp_path / "Test.docset"
    docset.mkdir()
    _make_fake_docset(docset)
    stale_page = str(docset / "Contents" / "Resources" / "Documents" / "removed.html")
    plan = refresh.plan_refresh(_cfg(docset, stype="zeal"), _meta({stale_page: "abc"}))
    assert plan.sources[0].vanished == []


def test_present_but_null_zeal_keys_resolve_to_the_defaults(tmp_path):
    """`sqlite_filename:` with nothing after it is a PRESENT key whose value is
    None, not a missing key — so `.get(key, default)` hands None straight to the
    ingester, which does `Path / None` and raises TypeError.

    The planner resolved it fine (`... or DEFAULT`) while the engine's own
    re-ingest path crashed on the same config. One resolution, one answer."""
    docset = tmp_path / "Test.docset"
    docset.mkdir()
    _make_fake_docset(docset)
    cfg = _cfg(docset, stype="zeal")
    cfg["ingest"]["zeal"] = {"sqlite_filename": None, "pages_dirname": None}

    assert refresh.zeal_index_name(cfg) == refresh.DEFAULT_ZEAL_INDEX
    assert refresh.zeal_pages_dirname(cfg) == refresh.DEFAULT_ZEAL_PAGES

    plan = refresh.plan_refresh(cfg, _meta({}))
    assert plan.sources[0].root_missing is False
    # The re-ingest path — where the crash was.
    ingester = refresh._make_ingester(plan.sources[0], cfg, set())
    assert ingester.sqlite_path == (
        Path(docset).resolve() / "Contents" / "Resources" / refresh.DEFAULT_ZEAL_INDEX
    )
    assert ingester.pages_root == Path(docset).resolve() / refresh.DEFAULT_ZEAL_PAGES


def test_a_missing_docset_is_a_missing_root(tmp_path):
    cfg = {"sources": [{"type": "zeal", "path": str(tmp_path / "Gone.docset")}]}
    plan = refresh.plan_refresh(cfg, _meta({str(tmp_path / "Gone.docset" / "p.html"): "abc"}))
    assert plan.sources[0].root_missing is True
    assert plan.sources[0].vanished == []
    assert plan.has_work is False


# -- review round 1: overlapping and nested roots ------------------------------
#
# No test covered two sources whose roots overlap, and both Critical findings
# lived in exactly that blind spot. `vanished` was computed from prefix +
# existence alone, with no notion of which source OWNS a recorded row.


def _multi_cfg(sources):
    return {"sources": sources,
            "chunking": {"target_tokens": 768, "overlap_pct": 12,
                         "min_chunk_tokens": 8},
            "ingest": {"default_topic": "test"}}


def test_a_missing_nested_source_is_not_pruned_by_its_present_parent(tmp_path, monkeypatch):
    """Critical 1. An unreachable subtree BELOW a present root - a junction to
    a NAS, a removable volume mounted into a folder. The parent enumerates
    fine, so its own root_missing guard never fires, and it used to claim the
    child's still-existing files as vanished and delete them."""
    docs = tmp_path / "docs"
    docs.mkdir()
    _md(docs, "kept.md", "body")
    nas = docs / "nas"                      # never created: the unmounted share
    cfg = _multi_cfg([{"type": "markdown", "path": str(docs)},
                      {"type": "markdown", "path": str(nas)}])
    recorded = str(nas / "report.md")

    plan = refresh.plan_refresh(cfg, _meta({recorded: "abc"}))
    parent, child = plan.sources
    assert child.root_missing is True
    assert child.vanished == []
    assert parent.root_missing is False
    assert parent.vanished == []            # the bug: used to be [recorded]

    monkeypatch.setattr(refresh, "ingest_pipeline", MagicMock(return_value=1))
    store = MagicMock()
    refresh.run_refresh(cfg, _meta({recorded: "abc"}), store=store,
                        embedder_factory=MagicMock(), prune=True)
    store.delete_by_source.assert_not_called()


def test_a_docset_under_a_markdown_root_is_never_swept_up(tmp_path, monkeypatch):
    """Critical 2. ZealIngester records a page as <docset>/<rel> while the file
    lives at <docset>/Contents/Resources/Documents/<rel>, so every one of a
    docset's ~50k rows names a path that never exists. A parent source used to
    sweep them all into `vanished` and destroy the real vectors - and the
    .dsidx marker survived, so the docset then read `unchanged` forever and was
    never rebuilt. The docset is NOT configured here: the suffix filter is what
    has to save it."""
    docs = tmp_path / "docs"
    docs.mkdir()
    _md(docs, "kept.md", "body")
    docset = docs / "Test.docset"
    docset.mkdir()
    _make_fake_docset(docset)

    page_row = str(docset / "page_a.html")
    assert not Path(page_row).exists()      # the synthetic path, as recorded
    assert (docset / "Contents" / "Resources" / "Documents" / "page_a.html").is_file()

    cfg = _multi_cfg([{"type": "markdown", "path": str(docs)}])
    plan = refresh.plan_refresh(cfg, _meta({page_row: "abc"}))
    assert plan.sources[0].vanished == []

    monkeypatch.setattr(refresh, "ingest_pipeline", MagicMock(return_value=1))
    store = MagicMock()
    refresh.run_refresh(cfg, _meta({page_row: "abc"}), store=store,
                        embedder_factory=MagicMock(), prune=True)
    store.delete_by_source.assert_not_called()


def test_a_root_that_enumerates_nothing_refuses_to_prune(tmp_path):
    """A present root that yields zero files while the index holds rows for it
    is "unknown", not "emptied". Could be a permissions change, a sync client
    mid-reset, or a mount point whose volume is gone. Deliberate removal is
    what `rag forget` is for; silent mass deletion is never the safe default."""
    docs = tmp_path / "docs"
    docs.mkdir()                            # present, but empty
    recorded = str(docs / "was_here.md")
    cfg = _multi_cfg([{"type": "markdown", "path": str(docs)}])

    plan = refresh.plan_refresh(cfg, _meta({recorded: "abc"}))
    sp = plan.sources[0]
    assert sp.vanished == []
    assert sp.prune_blocked                 # and it says why
    assert plan.has_blocked_prunes is True

    store = MagicMock()
    refresh.run_refresh(cfg, _meta({recorded: "abc"}), store=store,
                        embedder_factory=MagicMock(), prune=True)
    store.delete_by_source.assert_not_called()


def test_an_empty_root_with_no_recorded_rows_is_not_flagged(tmp_path):
    """A genuinely new, still-empty source is not suspicious - no warning."""
    docs = tmp_path / "docs"
    docs.mkdir()
    plan = refresh.plan_refresh(_multi_cfg([{"type": "markdown", "path": str(docs)}]),
                                _meta({}))
    assert plan.sources[0].prune_blocked is None
    assert plan.has_blocked_prunes is False


# -- review round 1: replacing a changed file ----------------------------------

def test_editing_a_file_shorter_leaves_no_orphaned_chunks(tmp_path, meta_store):
    """chunk_id is deterministic on position, so a re-ingest overwrites
    same-position chunks but orphans everything past the new end of the file.
    Deleted text stayed queryable and kept turning up in citations."""
    notes = tmp_path / "notes"
    notes.mkdir()
    f = notes / "a.md"
    f.write_text(
        "# One\n\n" + "keep text " * 40
        + "\n\n## Two\n\n" + "zzzremoved " * 40
        + "\n\n## Three\n\n" + "zzzremoved " * 40 + "\n",
        encoding="utf-8",
    )
    cfg = _cfg(notes)
    store = _MemStore()

    refresh.run_refresh(cfg, meta_store, store=store, embedder_factory=_TinyEmbedder)
    assert any("zzzremoved" in c.text for c in store.points.values())

    f.write_text("# One\n\n" + "keep text " * 40 + "\n", encoding="utf-8")
    refresh.run_refresh(cfg, meta_store, store=store, embedder_factory=_TinyEmbedder)

    assert store.points, "the surviving content must still be indexed"
    assert not any("zzzremoved" in c.text for c in store.points.values())


def test_a_new_file_is_not_deleted_before_it_is_ingested(tmp_path, monkeypatch):
    """Only `changed` paths have old chunks to clear. Calling delete for a new
    path is a pointless store round trip per file on a first ingest."""
    _md(tmp_path, "a.md", "body")
    monkeypatch.setattr(refresh, "ingest_pipeline", MagicMock(return_value=1))
    store = MagicMock()
    refresh.run_refresh(_cfg(tmp_path), _meta({}), store=store,
                        embedder_factory=MagicMock())
    store.delete_by_source.assert_not_called()


# -- review round 1: per-source error isolation --------------------------------

def test_one_failing_source_does_not_stop_the_others(tmp_path, monkeypatch):
    """Unattended scheduled task: a single corrupt PDF must not mean no later
    source refreshes, no prune runs, and a stale IDF map left behind chunks
    that were already written."""
    from core import pipeline

    a = tmp_path / "a"
    a.mkdir()
    _md(a, "a.md", "body")
    b = tmp_path / "b"
    b.mkdir()
    _md(b, "b.md", "body")
    cfg = _multi_cfg([{"type": "markdown", "path": str(a)},
                      {"type": "markdown", "path": str(b)}])

    seen: list[Path] = []

    def flaky(ingester, *args, **kwargs):
        seen.append(Path(ingester.root))
        if Path(ingester.root) == a.resolve():
            raise RuntimeError("corrupt file")
        return 1

    monkeypatch.setattr(refresh, "ingest_pipeline", flaky)
    pipeline._hybrid_cache.update({"vocab": {"x": 0}, "idf": {"x": 1.0}, "built": True})

    plan = refresh.run_refresh(cfg, _meta({}), store=MagicMock(),
                               embedder_factory=MagicMock())

    assert len(seen) == 2, "the second source never got its turn"
    assert "corrupt file" in (plan.sources[0].error or "")
    assert plan.sources[1].error is None
    assert plan.has_errors is True
    assert pipeline._hybrid_cache["built"] is False


def test_a_source_whose_ingest_failed_is_not_pruned(tmp_path, monkeypatch):
    """Something is wrong with that source; deleting under it is the one
    irreversible thing in this file. Skipping one run costs nothing."""
    docs = tmp_path / "docs"
    docs.mkdir()
    _md(docs, "a.md", "body")
    gone = str(docs / "gone.md")
    cfg = _multi_cfg([{"type": "markdown", "path": str(docs)}])

    monkeypatch.setattr(refresh, "ingest_pipeline",
                        MagicMock(side_effect=RuntimeError("locked")))
    store = MagicMock()
    plan = refresh.run_refresh(cfg, _meta({gone: "abc"}), store=store,
                               embedder_factory=MagicMock(), prune=True)

    assert plan.sources[0].vanished == [gone]     # still REPORTED
    store.delete_by_source.assert_not_called()    # but not acted on


# -- review round 1: an unreadable file is not a changed file ------------------

def test_an_unreadable_file_is_not_reported_as_changed(tmp_path, monkeypatch):
    """hash_file returns "" when it cannot read the file. Treating that as a
    hash made a permanently locked-but-listed file re-ingest on every single
    run - the exact perpetual work this feature exists to avoid."""
    f = _md(tmp_path, "a.md", "body")
    monkeypatch.setattr(refresh, "hash_file", lambda p: "")

    plan = refresh.plan_refresh(_cfg(tmp_path), _meta({str(f): "abc"}))
    sp = plan.sources[0]
    assert sp.changed == []
    assert sp.new == []
    assert sp.unreadable == [f]
    assert plan.has_work is False


def test_record_source_with_an_empty_file_hash_preserves_the_stored_one(meta_store):
    """COALESCE guards NULL; "" is not NULL. A transient read failure would
    otherwise overwrite a perfectly good stored hash with empty."""
    meta_store.record_source("/a.md", "markdown", "t", 3, "ch",
                             file_hash="deadbeefdeadbeef")
    meta_store.record_source("/a.md", "markdown", "t", 4, "ch2", file_hash="")

    rows = meta_store.get_sources()
    assert len(rows) == 1
    assert rows[0]["file_hash"] == "deadbeefdeadbeef"
    assert rows[0]["chunk_count"] == 4


def test_record_zeal_marker_is_public_for_task_6(tmp_path, meta_store):
    """Task 6 calls this after a manual `rag ingest --zeal`, so a docset the
    owner ingested by hand does not read as new on the next refresh."""
    docset = tmp_path / "Test.docset"
    docset.mkdir()
    _make_fake_docset(docset)

    refresh.record_zeal_marker(meta_store, docset, "docSet.dsidx")

    idx = docset / "Contents" / "Resources" / "docSet.dsidx"
    rows = {r["source_path"]: r for r in meta_store.get_sources()}
    assert rows[str(idx)]["file_hash"] == hash_file(idx)
    assert rows[str(idx)]["doc_type"] == refresh.ZEAL_MARKER_DOC_TYPE


# -- review round 2: a failed ingest must not leave a file "unchanged" ---------
#
# pipeline.ingest appends to its `sources` accumulator only after a batch is
# upserted, and its abort path records those sources before re-raising - with
# the hash of the WHOLE file. So a file spanning more than one embed batch can
# end up with a row claiming the new hash while only the first batch is
# indexed. Every later refresh then reads it as unchanged. Since refresh now
# deletes a changed file's old chunks first, the un-indexed tail is not stale,
# it is gone.


class _SmallBatchEmbedder(_TinyEmbedder):
    """Flushes every 2 chunks, so a handful of chunks spans several batches."""
    batch_size = 2


class _FlakyStore(_MemStore):
    """Fails the (fail_after + 1)-th upsert, i.e. part-way through a source."""

    def __init__(self):
        super().__init__()
        self.upserts = 0
        self.fail_after: int | None = None

    def arm(self, fail_after: int) -> None:
        self.upserts = 0
        self.fail_after = fail_after

    def upsert_chunks(self, chunks, vectors):
        self.upserts += 1
        if self.fail_after is not None and self.upserts > self.fail_after:
            raise RuntimeError("qdrant went away mid-ingest")
        return super().upsert_chunks(chunks, vectors)


def _many_sections(name, marker):
    body = " ".join([marker] * 60)
    return ("# Top\n\n" + body + "\n\n"
            + "\n\n".join(f"## Sec {i}\n\n{body}" for i in range(6)) + "\n")


def test_a_failed_ingest_forgets_the_hash_of_what_it_touched(tmp_path, meta_store):
    """The row would otherwise claim the new full-file hash over a partly
    indexed file whose old chunks this run already deleted - permanently, and
    silently, because every later refresh calls it unchanged."""
    notes = tmp_path / "notes"
    notes.mkdir()
    f = notes / "a.md"
    f.write_text(_many_sections("a.md", "alpha"), encoding="utf-8")
    cfg = _cfg(notes)
    store = _FlakyStore()

    # A clean run first, so there is a good row with a real hash.
    refresh.run_refresh(cfg, meta_store, store=store, embedder_factory=_TinyEmbedder)
    assert meta_store.get_sources()[0]["file_hash"]
    assert refresh.plan_refresh(cfg, meta_store).has_work is False
    full = len(store.points)
    assert full >= 4, "the file must span several embed batches for this to bite"

    # Now edit it, and die part-way through the re-ingest.
    f.write_text(_many_sections("a.md", "beta"), encoding="utf-8")
    store.arm(fail_after=1)
    plan = refresh.run_refresh(cfg, meta_store, store=store,
                               embedder_factory=_SmallBatchEmbedder)

    assert plan.sources[0].error                      # the run did say so
    assert len(store.points) < full                   # and the tail really is gone

    row = meta_store.get_sources()[0]
    assert row["source_path"] == str(f)
    assert row["file_hash"] is None                   # forgotten, not claimed

    # The property that actually matters: the next refresh rebuilds it.
    replan = refresh.plan_refresh(cfg, meta_store)
    assert replan.sources[0].changed == [f]
    assert replan.sources[0].unchanged == 0
    assert replan.has_work is True


def test_a_failing_source_does_not_clear_a_healthy_one(tmp_path, meta_store):
    """Only the failed source's own paths are forgotten."""
    good = tmp_path / "good"
    good.mkdir()
    gf = good / "g.md"
    gf.write_text(_many_sections("g.md", "gamma"), encoding="utf-8")
    bad = tmp_path / "bad"
    bad.mkdir()
    bf = bad / "b.md"
    bf.write_text(_many_sections("b.md", "delta"), encoding="utf-8")

    cfg = _multi_cfg([{"type": "markdown", "path": str(bad)},
                      {"type": "markdown", "path": str(good)}])

    class _OneSourceFails(_MemStore):
        def upsert_chunks(self, chunks, vectors):
            if any(c.source_path == str(bf) for c in chunks):
                raise RuntimeError("that one file is cursed")
            return super().upsert_chunks(chunks, vectors)

    # Clean run: both sources land, both rows get hashes.
    refresh.run_refresh(cfg, meta_store, store=_MemStore(),
                        embedder_factory=_TinyEmbedder)
    before = {r["source_path"]: r["file_hash"] for r in meta_store.get_sources()}
    assert before[str(bf)] and before[str(gf)]

    # Edit both; the bad source fails, the good one still runs.
    bf.write_text(_many_sections("b.md", "epsilon"), encoding="utf-8")
    gf.write_text(_many_sections("g.md", "zeta"), encoding="utf-8")
    plan = refresh.run_refresh(cfg, meta_store, store=_OneSourceFails(),
                               embedder_factory=_SmallBatchEmbedder)

    assert plan.sources[0].error
    assert plan.sources[1].error is None

    after = {r["source_path"]: r["file_hash"] for r in meta_store.get_sources()}
    assert after[str(bf)] is None                     # forgotten
    assert after[str(gf)]                             # untouched, and refreshed
    assert after[str(gf)] != before[str(gf)]

    replan = refresh.plan_refresh(cfg, meta_store)
    assert replan.sources[0].changed == [bf]          # rebuilt next run
    assert replan.sources[1].unchanged == 1           # left alone


def test_clear_file_hashes_only_touches_the_named_rows(meta_store):
    meta_store.record_source("/a.md", "markdown", "t", 1, "c", file_hash="aaaaaaaaaaaaaaaa")
    meta_store.record_source("/b.md", "markdown", "t", 1, "c", file_hash="bbbbbbbbbbbbbbbb")

    assert meta_store.clear_file_hashes(["/a.md", "/nonexistent.md"]) == 1
    assert meta_store.clear_file_hashes([]) == 0

    rows = {r["source_path"]: r["file_hash"] for r in meta_store.get_sources()}
    assert rows["/a.md"] is None
    assert rows["/b.md"] == "bbbbbbbbbbbbbbbb"


def test_clear_file_hashes_handles_more_paths_than_sqlite_allows_variables(meta_store):
    """SQLite's parameter limit is 999 on older builds; the IN clause batches."""
    paths = [f"/bulk/{i}.md" for i in range(1200)]
    for p in paths:
        meta_store.record_source(p, "markdown", "t", 1, "c", file_hash="cccccccccccccccc")

    assert meta_store.clear_file_hashes(paths) == 1200
    assert all(h is None for h in meta_store.source_hashes().values())
