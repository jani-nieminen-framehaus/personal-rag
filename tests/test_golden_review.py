"""Curation state machine: y/n/e transitions, resume, stats."""
from __future__ import annotations

import json

from eval import golden_review


CAND = {"question": "Draft?", "relevant_chunk_ids": ["c1"],
        "relevant_refs": [{"source_path": "/a.md", "section": "S"}],
        "topic": "notes", "preview": "…", "status": "candidate"}


def test_accept_produces_golden_row_without_status_or_preview():
    updated, golden = golden_review.apply_decision(dict(CAND), "y")
    assert updated["status"] == "accepted"
    assert golden["question"] == "Draft?"
    assert golden["relevant_chunk_ids"] == ["c1"]
    assert golden["relevant_refs"][0]["section"] == "S"
    assert "status" not in golden and "preview" not in golden


def test_reject_produces_no_golden_row():
    updated, golden = golden_review.apply_decision(dict(CAND), "n")
    assert updated["status"] == "rejected"
    assert golden is None


def test_edit_replaces_question_then_accepts():
    updated, golden = golden_review.apply_decision(dict(CAND), "e", edited="Better?")
    assert updated["status"] == "accepted"
    assert golden["question"] == "Better?"


def test_pending_filters_only_candidates():
    rows = [dict(CAND), {**CAND, "status": "accepted"}, {**CAND, "status": "rejected"}]
    assert len(golden_review.pending(rows)) == 1


def test_rows_roundtrip(tmp_path):
    p = tmp_path / "c.jsonl"
    golden_review.save_rows(p, [dict(CAND)])
    assert golden_review.load_rows(p) == [CAND]


def test_stats_counts(tmp_path):
    cands = tmp_path / "c.jsonl"
    golden = tmp_path / "g.jsonl"
    golden_review.save_rows(cands, [dict(CAND), {**CAND, "status": "rejected"}])
    golden.write_text(json.dumps({"question": "q", "relevant_chunk_ids": ["c1"],
                                  "topic": "notes"}) + "\n", encoding="utf-8")
    s = golden_review.stats(cands, golden)
    assert s["candidates_pending"] == 1
    assert s["rejected"] == 1
    assert s["accepted_total"] == 1
    assert s["accepted_by_topic"] == {"notes": 1}


def test_golden_cli_group_registered_with_subcommands():
    """--help smoke test: without this, nothing verifies the `golden` group
    is actually wired into cli.py — the pure-function tests above would all
    pass even if the CLI group were never registered."""
    from click.testing import CliRunner
    from cli import cli

    result = CliRunner().invoke(cli, ["golden", "--help"])
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "review" in result.output
    assert "stats" in result.output
