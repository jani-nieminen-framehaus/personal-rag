"""CLI surface for refresh and forget. Deleting paths must be opt-in."""
from __future__ import annotations

import sqlite3
from pathlib import Path
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


# -- controller correction A: a partly-failed refresh must not look clean ------

def _source(**kw):
    """A SourcePlan-shaped stand-in. Real dataclass, so the CLI reads the same
    attribute names it will read in production."""
    from core.refresh import SourcePlan
    kw.setdefault("type", "markdown")
    kw.setdefault("path", Path("D:/notes"))
    return SourcePlan(**kw)


def test_refresh_exits_nonzero_when_a_source_failed(monkeypatch):
    """Task 5 made refresh resilient: one source raising is recorded and the run
    continues. Without this, a partly-failed scheduled refresh returns 0 and
    reads exactly like a clean one."""
    cli_mod = _patch(monkeypatch)
    bad = _source(path=Path("D:/papers"), type="pdf", error="RuntimeError: bad xref")
    plan = MagicMock(has_work=True, has_missing_roots=False, sources=[bad])
    monkeypatch.setattr("core.refresh.run_refresh", lambda *a, **k: plan)
    res = CliRunner().invoke(cli_mod.cli, ["refresh"])
    assert res.exit_code == 1
    assert "bad xref" in res.output
    assert "papers" in res.output


def test_refresh_blocked_prune_is_reported_but_is_not_a_failure(monkeypatch):
    """A blocked prune is the safety property doing its job — say so verbatim,
    exit 0."""
    cli_mod = _patch(monkeypatch)
    blocked = _source(prune_blocked="the root is present but enumerated no files")
    plan = MagicMock(has_work=False, has_missing_roots=False, sources=[blocked])
    monkeypatch.setattr("core.refresh.run_refresh", lambda *a, **k: plan)
    res = CliRunner().invoke(cli_mod.cli, ["refresh", "--prune", "--yes"])
    assert res.exit_code == 0
    assert "the root is present but enumerated no files" in res.output


def test_refresh_summary_shows_the_per_source_counts(monkeypatch):
    cli_mod = _patch(monkeypatch)
    s = _source(new=[Path("a.md")], changed=[Path("b.md"), Path("c.md")],
                unchanged=41, vanished=["d.md"], unreadable=[Path("e.md")])
    plan = MagicMock(has_work=True, has_missing_roots=False, sources=[s])
    monkeypatch.setattr("core.refresh.run_refresh", lambda *a, **k: plan)
    res = CliRunner().invoke(cli_mod.cli, ["refresh"])
    assert res.exit_code == 0
    out = res.output
    for token in ("new 1", "changed 2", "unchanged 41", "vanished 1", "unreadable"):
        assert token in out, f"{token!r} missing from:\n{out}"


def test_refresh_says_pruned_once_the_rows_are_actually_gone(monkeypatch):
    """`vanished` is what the plan FOUND; after a real --prune those rows are
    already deleted, and the summary has to say so rather than leave the
    operator wondering whether anything happened."""
    cli_mod = _patch(monkeypatch)
    s = _source(unchanged=9, vanished=["d.md", "e.md"])
    plan = MagicMock(has_work=False, has_missing_roots=False, sources=[s])
    monkeypatch.setattr("core.refresh.run_refresh", lambda *a, **k: plan)

    done = CliRunner().invoke(cli_mod.cli, ["refresh", "--prune", "--yes"])
    assert done.exit_code == 0
    assert "pruned 2" in done.output
    assert "removed 2 vanished file(s)" in done.output

    plain = CliRunner().invoke(cli_mod.cli, ["refresh"])
    assert "vanished 2" in plain.output
    assert "--prune" in plain.output          # the hint, not a claim of removal


def test_refresh_does_not_claim_to_have_pruned_a_blocked_source(monkeypatch):
    """A blocked source is skipped by the prune loop, so its rows are still
    there — the planning word is the honest one."""
    cli_mod = _patch(monkeypatch)
    s = _source(vanished=["d.md"], prune_blocked="the source root is not present")
    plan = MagicMock(has_work=False, has_missing_roots=False, sources=[s])
    monkeypatch.setattr("core.refresh.run_refresh", lambda *a, **k: plan)
    res = CliRunner().invoke(cli_mod.cli, ["refresh", "--prune", "--yes"])
    assert "vanished 1" in res.output
    assert "pruned" not in res.output
    assert "removed 0 vanished file(s)" in res.output


def test_refresh_does_not_build_an_embedder_when_there_is_no_work(monkeypatch):
    """The whole promise of refresh: an unchanged corpus costs a directory walk,
    not several GB of VRAM. A factory that is CALLED instead of PASSED defeats
    it while every other test still passes."""
    cli_mod = _patch(monkeypatch)
    built = []
    monkeypatch.setattr(cli_mod, "make_embedder", lambda cfg: built.append(cfg))

    def fake(cfg, metadata, store, embedder_factory, **k):
        assert callable(embedder_factory)
        return MagicMock(has_work=False, has_missing_roots=False, sources=[])

    monkeypatch.setattr("core.refresh.run_refresh", fake)
    res = CliRunner().invoke(cli_mod.cli, ["refresh"])
    assert res.exit_code == 0, res.output
    assert built == []


def test_refresh_dry_run_passes_dry_run_through(monkeypatch):
    cli_mod = _patch(monkeypatch)
    seen = {}
    def fake(*a, **k):
        seen.update(k)
        return MagicMock(has_work=True, has_missing_roots=False, sources=[])
    monkeypatch.setattr("core.refresh.run_refresh", fake)
    res = CliRunner().invoke(cli_mod.cli, ["refresh", "--dry-run"])
    assert res.exit_code == 0
    assert seen["dry_run"] is True
    assert "dry run" in res.output.lower()


def test_forget_invalidates_the_hybrid_cache(monkeypatch):
    """A cached IDF map built over the pre-deletion corpus skews the next
    hybrid query."""
    store, meta = MagicMock(), MagicMock()
    store.delete_by_source.return_value = 3
    meta.get_sources.return_value = [{"source_path": "/a.md", "topic": "t"}]
    cli_mod = _patch(monkeypatch, store=store, metadata=meta)
    calls = []
    monkeypatch.setattr("core.pipeline.invalidate_hybrid_cache", lambda: calls.append(1))
    res = CliRunner().invoke(cli_mod.cli, ["forget", "--source", "/a.md", "--yes"])
    assert res.exit_code == 0
    assert calls == [1]


def test_forget_asks_for_every_source_not_just_the_first_fifty(monkeypatch):
    """get_sources() defaults to limit=50. Taking the default would delete a
    subset and report success."""
    store, meta = MagicMock(), MagicMock()
    store.delete_by_source.return_value = 1
    meta.get_sources.return_value = [
        {"source_path": f"/n{i}.md", "topic": "big"} for i in range(120)
    ]
    cli_mod = _patch(monkeypatch, store=store, metadata=meta)
    res = CliRunner().invoke(cli_mod.cli, ["forget", "--topic", "big", "--yes"])
    assert res.exit_code == 0
    assert meta.get_sources.call_args.kwargs["limit"] > 120
    assert store.delete_by_source.call_count == 120


def test_forget_closes_metadata_even_when_it_aborts(monkeypatch):
    store, meta = MagicMock(), MagicMock()
    meta.get_sources.return_value = [{"source_path": "/a.md", "topic": "t"}]
    cli_mod = _patch(monkeypatch, store=store, metadata=meta)
    declined = CliRunner().invoke(cli_mod.cli, ["forget", "--source", "/a.md"], input="n\n")
    assert declined.exit_code != 0
    meta.close.assert_called_once()

    meta.reset_mock()
    meta.get_sources.return_value = []
    missing = CliRunner().invoke(cli_mod.cli, ["forget", "--source", "/x.md", "--yes"])
    assert missing.exit_code == 1
    meta.close.assert_called_once()


# -- controller correction B: a manual --zeal ingest must leave the marker -----

def _make_fake_docset(root: Path) -> None:
    docs = root / "Contents" / "Resources" / "Documents"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "page_a.html").write_text(
        "<html><body><h1>Page A</h1><p>alpha</p></body></html>", encoding="utf-8")
    idx = root / "Contents" / "Resources" / "docSet.dsidx"
    with sqlite3.connect(str(idx)) as conn:
        conn.execute("CREATE TABLE searchIndex (id INTEGER, name TEXT, type TEXT, path TEXT)")
        conn.execute("INSERT INTO searchIndex VALUES (1, 'Page A', 'Page', 'page_a.html')")
        conn.commit()


def test_zeal_ingest_records_the_dsidx_marker(tmp_path, monkeypatch):
    """Ingest writes one `sources` row per PAGE, so nothing is keyed by the
    `.dsidx` that refresh hashes. Without the marker, the first refresh after a
    manual `rag ingest --zeal` calls the docset new and re-embeds every page."""
    docset = tmp_path / "Test.docset"
    docset.mkdir()
    _make_fake_docset(docset)

    meta = MagicMock()
    cli_mod = _patch(monkeypatch, metadata=meta)
    monkeypatch.setattr(cli_mod, "ingest_pipeline", lambda *a, **k: 4)

    res = CliRunner().invoke(cli_mod.cli, ["ingest", "--zeal", str(docset)])
    assert res.exit_code == 0, res.output

    idx = (docset / "Contents" / "Resources" / "docSet.dsidx").resolve()
    recorded = {c.kwargs.get("source_path") for c in meta.record_source.call_args_list}
    assert str(idx) in recorded, recorded
    marker = next(c for c in meta.record_source.call_args_list
                  if c.kwargs.get("source_path") == str(idx))
    # The hash is the whole point — an empty one reads as "unknown" and
    # re-ingests anyway.
    assert marker.kwargs["file_hash"]
    # Written before the store is closed, or it is not written at all.
    assert meta.close.called


def test_zeal_marker_is_not_written_when_the_ingest_fails(tmp_path, monkeypatch):
    """A marker over a half-written docset would make the next refresh call it
    unchanged."""
    docset = tmp_path / "Test.docset"
    docset.mkdir()
    _make_fake_docset(docset)

    meta = MagicMock()
    cli_mod = _patch(monkeypatch, metadata=meta)

    def boom(*a, **k):
        raise RuntimeError("embedder died")

    monkeypatch.setattr(cli_mod, "ingest_pipeline", boom)
    res = CliRunner().invoke(cli_mod.cli, ["ingest", "--zeal", str(docset)])
    assert res.exit_code != 0
    idx = str((docset / "Contents" / "Resources" / "docSet.dsidx").resolve())
    recorded = {c.kwargs.get("source_path") for c in meta.record_source.call_args_list}
    assert idx not in recorded
