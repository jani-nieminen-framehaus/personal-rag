"""Whole-branch review fixes: config surface for the swept knobs, honest
sweep exit codes, chunking-sweep scope warning, blank-query handling.

These are the findings a per-task review structurally could not see —
each one spans two or more of the ten tasks.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

import cli as cli_mod


REPO = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# The swept knobs need a config surface, or the sweep winner cannot be applied
# -----------------------------------------------------------------------------

def _write_cfg(tmp_path: Path, pipeline_block: dict) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"pipeline": pipeline_block}), encoding="utf-8")
    return str(p)


@pytest.fixture
def stub_abcs(monkeypatch):
    """Neutralise the expensive factories; the CLI wiring is what's under test."""
    for name in ("make_embedder", "make_store", "make_reranker", "make_generator"):
        monkeypatch.setattr(cli_mod, name, lambda cfg: MagicMock())
    monkeypatch.setattr(cli_mod, "make_metadata", lambda cfg: None)


@pytest.fixture
def captured_ask(monkeypatch):
    """Capture the kwargs `rag ask` hands to the pipeline."""
    seen: dict = {}

    def fake_ask(query, **kw):
        seen.update(kw)
        seen["query"] = query
        return cli_mod.pipeline.AskResult(answer="ok", citations=[])

    monkeypatch.setattr(cli_mod, "ask_pipeline", fake_ask)
    return seen


def test_config_yaml_exposes_hybrid_and_dense_weight():
    """`dense_weight` lived only inside the eval.sweep grid, so a sweep
    winner of 0.3 or 0.7 could never be applied to live queries — ask()'s
    default of 0.5 was pinned forever. Same for `hybrid`."""
    cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    pipeline_cfg = cfg["pipeline"]
    assert "hybrid" in pipeline_cfg, "pipeline.hybrid missing from config.yaml"
    assert "dense_weight" in pipeline_cfg, "pipeline.dense_weight missing from config.yaml"
    assert pipeline_cfg["hybrid"] is False
    assert pipeline_cfg["dense_weight"] == 0.5


def test_ask_reads_hybrid_and_dense_weight_from_config(tmp_path, stub_abcs, captured_ask):
    cfg = _write_cfg(tmp_path, {"hybrid": True, "dense_weight": 0.7})
    res = CliRunner().invoke(cli_mod.cli, ["-c", cfg, "ask", "q"])
    assert res.exit_code == 0, res.output
    assert captured_ask["hybrid"] is True
    assert captured_ask["dense_weight"] == 0.7


def test_ask_flags_override_config(tmp_path, stub_abcs, captured_ask):
    cfg = _write_cfg(tmp_path, {"hybrid": True, "dense_weight": 0.7})
    res = CliRunner().invoke(
        cli_mod.cli, ["-c", cfg, "ask", "q", "--no-hybrid", "--dense-weight", "0.3"])
    assert res.exit_code == 0, res.output
    assert captured_ask["hybrid"] is False
    assert captured_ask["dense_weight"] == 0.3


def test_ask_hybrid_flag_still_works(tmp_path, stub_abcs, captured_ask):
    """--hybrid must keep working as a bare opt-in flag."""
    cfg = _write_cfg(tmp_path, {})
    res = CliRunner().invoke(cli_mod.cli, ["-c", cfg, "ask", "q", "--hybrid"])
    assert res.exit_code == 0, res.output
    assert captured_ask["hybrid"] is True


def test_ask_defaults_when_pipeline_block_is_absent(tmp_path, stub_abcs, captured_ask):
    """The defensive `(cfg.get("pipeline") or {})` idiom: a config with a
    null pipeline block must not blow up."""
    p = tmp_path / "config.yaml"
    p.write_text("pipeline:\n", encoding="utf-8")  # -> None, not {}
    res = CliRunner().invoke(cli_mod.cli, ["-c", str(p), "ask", "q"])
    assert res.exit_code == 0, res.output
    assert captured_ask["hybrid"] is False
    assert captured_ask["dense_weight"] == 0.5


def test_eval_forwards_hybrid_and_dense_weight(tmp_path, stub_abcs):
    """Without this, the runbook's baseline (step 3) and the sweep (step 4)
    measure different retrieval stacks, so the "measured delta" compares
    incomparable numbers."""
    cfg = _write_cfg(tmp_path, {"hybrid": True, "dense_weight": 0.7})
    golden = tmp_path / "g.jsonl"
    golden.write_text("", encoding="utf-8")
    with patch("eval.run_ragas.run", return_value={"n_questions": 0}) as run_mock:
        res = CliRunner().invoke(
            cli_mod.cli, ["-c", cfg, "eval", "--golden", str(golden), "--json"])
    assert res.exit_code == 0, res.output
    kwargs = run_mock.call_args.kwargs
    assert kwargs["hybrid"] is True
    assert kwargs["dense_weight"] == 0.7


def test_eval_flags_override_config(tmp_path, stub_abcs):
    cfg = _write_cfg(tmp_path, {"hybrid": True, "dense_weight": 0.7})
    golden = tmp_path / "g.jsonl"
    golden.write_text("", encoding="utf-8")
    with patch("eval.run_ragas.run", return_value={"n_questions": 0}) as run_mock:
        res = CliRunner().invoke(cli_mod.cli, [
            "-c", cfg, "eval", "--golden", str(golden), "--json",
            "--no-hybrid", "--dense-weight", "0.3"])
    assert res.exit_code == 0, res.output
    kwargs = run_mock.call_args.kwargs
    assert kwargs["hybrid"] is False
    assert kwargs["dense_weight"] == 0.3


# -----------------------------------------------------------------------------
# build_golden: relevant_refs + the chunking_params drift it exists to prevent
# -----------------------------------------------------------------------------

def test_build_golden_emits_relevant_refs(monkeypatch, tmp_path):
    """Without relevant_refs, section mode raises on every row of the
    samples golden set, so a chunking sweep over it cannot run at all."""
    import json
    from eval import build_golden

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"question": "q1", "topic": "ml",
                    "relevant": [{"path": "ml/a.md", "section": "S"}]}) + "\n",
        encoding="utf-8")
    golden_out = tmp_path / "golden.jsonl"
    monkeypatch.setattr(build_golden, "SAMPLES", tmp_path)
    monkeypatch.setattr(build_golden, "QUESTIONS", questions)
    monkeypatch.setattr(build_golden, "GOLDEN", golden_out)
    monkeypatch.setattr(build_golden, "build_index", lambda: {("ml/a.md", "S"): ["cid1"]})

    assert build_golden.main() == 0
    entry = json.loads(golden_out.read_text(encoding="utf-8").strip())
    assert entry["relevant_chunk_ids"] == ["cid1"]
    assert entry["relevant_refs"] == [
        {"source_path": str(tmp_path / "ml" / "a.md"), "section": "S"}
    ]


def test_build_golden_ref_source_path_matches_the_live_chunker():
    """The section key run_ragas matches on is f'{source_path}::{section}',
    built from live citations — and the live chunker stamps the ABSOLUTE
    path. build_index() keys its own map by a repo-relative posix path, so
    emitting that form would silently score every section-mode row 0.0.
    This test pins the emitted form to what the ingester actually produces.
    """
    from eval import build_golden
    from ingest.markdown_dir import MarkdownDirIngester

    cp = build_golden._load_chunking_config()
    ingester = MarkdownDirIngester(
        root=build_golden.SAMPLES,
        target_tokens=cp["target_tokens"],
        overlap_pct=cp["overlap_pct"],
        min_chunk_tokens=cp["min_chunk_tokens"],
        default_topic=cp["default_topic"],
        max_chunks_per_doc=cp["max_chunks_per_doc"],
    )
    live = {c.source_path for c in ingester.iter_chunks()}
    assert live, "samples/notes produced no chunks"

    index = build_golden.build_index()
    emitted = {str(build_golden.SAMPLES / rel_path) for rel_path, _section in index}
    assert emitted <= live, (
        "build_golden would emit a source_path the live pipeline never "
        f"produces; e.g. {sorted(emitted - live)[:2]}"
    )


def test_build_index_passes_max_chunks_per_doc(monkeypatch):
    """chunking_params() returns max_chunks_per_doc and every other call
    site passes it. build_golden did not — the exact drift the resolver
    exists to prevent."""
    from eval import build_golden

    seen: dict = {}

    class _Spy:
        def __init__(self, **kw):
            seen.update(kw)

        def iter_chunks(self):
            return iter(())

    monkeypatch.setattr(build_golden, "MarkdownDirIngester", _Spy)
    build_golden.build_index()
    assert "max_chunks_per_doc" in seen, "build_golden dropped max_chunks_per_doc"
    assert seen["max_chunks_per_doc"] == build_golden._load_chunking_config()["max_chunks_per_doc"]


# -- the API surface ---------------------------------------------------------

def _fake_result():
    r = MagicMock()
    r.answer = "ok"
    r.citations = []
    r.dense_hits = []
    return r


def test_api_ask_falls_back_to_config_for_hybrid_and_dense_weight(serve_state, monkeypatch):
    from fastapi.testclient import TestClient
    import serve

    seen: dict = {}

    def fake_ask(query, **kw):
        seen.update(kw)
        return _fake_result()

    monkeypatch.setattr(serve, "ask_pipeline", fake_ask)
    serve_state.ready = True
    # serve_state restores ready/embedder/store only — patch config through
    # monkeypatch so it can't leak into the other server tests.
    monkeypatch.setattr(serve.S, "config", {"pipeline": {"hybrid": True, "dense_weight": 0.3}})

    r = TestClient(serve.app).post("/api/ask", json={"query": "hello"})
    assert r.status_code == 200, r.text
    assert seen["hybrid"] is True
    assert seen["dense_weight"] == 0.3


def test_api_ask_request_fields_override_config(serve_state, monkeypatch):
    from fastapi.testclient import TestClient
    import serve

    seen: dict = {}

    def fake_ask(query, **kw):
        seen.update(kw)
        return _fake_result()

    monkeypatch.setattr(serve, "ask_pipeline", fake_ask)
    serve_state.ready = True
    # serve_state restores ready/embedder/store only — patch config through
    # monkeypatch so it can't leak into the other server tests.
    monkeypatch.setattr(serve.S, "config", {"pipeline": {"hybrid": True, "dense_weight": 0.3}})

    r = TestClient(serve.app).post(
        "/api/ask", json={"query": "hello", "hybrid": False, "dense_weight": 0.7})
    assert r.status_code == 200, r.text
    assert seen["hybrid"] is False
    assert seen["dense_weight"] == 0.7
