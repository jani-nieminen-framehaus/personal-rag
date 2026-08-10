"""Curation state machine: y/n/e transitions, resume, stats."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

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


# -- regression: session-level persistence (fix round 1) ----------------------
#
# save_rows() is atomic per-write, but the review SESSION wasn't protected:
# the two save_rows() calls only ran after the loop finished normally or hit
# "q". A Ctrl-C (or any uncaught exception) mid-loop skipped both calls and
# discarded every decision made earlier in that session, not just the
# in-flight one. These tests drive the real `rag golden review` command
# against real tmp_path files (module-level GOLDEN_CANDIDATES/GOLDEN_REAL
# monkeypatched to point there) to lock in that every save happens on every
# exit path.

def _two_candidates():
    return [
        {**dict(CAND), "question": "Q1", "relevant_chunk_ids": ["c1"]},
        {**dict(CAND), "question": "Q2", "relevant_chunk_ids": ["c2"]},
    ]


def _patch_golden_paths(monkeypatch, cands_path, golden_path):
    import cli as cli_mod
    monkeypatch.setattr(cli_mod, "GOLDEN_CANDIDATES", cands_path)
    monkeypatch.setattr(cli_mod, "GOLDEN_REAL", golden_path)
    return cli_mod


def test_review_cli_yes_then_quit_persists_and_leaves_rest_pending(tmp_path, monkeypatch):
    """y then q: the accepted row lands in the golden file, its candidate
    status is persisted as "accepted", and the un-reviewed candidate is
    still "candidate" (i.e. still pending on the next `rag golden review`)."""
    from click.testing import CliRunner

    cands_path = tmp_path / "cands.jsonl"
    golden_path = tmp_path / "golden.jsonl"
    golden_review.save_rows(cands_path, _two_candidates())
    cli_mod = _patch_golden_paths(monkeypatch, cands_path, golden_path)

    result = CliRunner().invoke(cli_mod.cli, ["golden", "review"], input="y\nq\n")

    assert result.exit_code == 0, result.output
    saved_cands = golden_review.load_rows(cands_path)
    assert saved_cands[0]["question"] == "Q1"
    assert saved_cands[0]["status"] == "accepted"
    assert saved_cands[1]["question"] == "Q2"
    assert saved_cands[1]["status"] == "candidate"
    saved_golden = golden_review.load_rows(golden_path)
    assert len(saved_golden) == 1
    assert saved_golden[0]["question"] == "Q1"


def test_review_cli_persists_earlier_decisions_on_keyboard_interrupt(tmp_path, monkeypatch):
    """This is the test that actually proves the finding is fixed: a
    KeyboardInterrupt raised while prompting for the SECOND candidate must
    not discard the "y" decision already made on the first one."""
    from click.testing import CliRunner

    cands_path = tmp_path / "cands.jsonl"
    golden_path = tmp_path / "golden.jsonl"
    golden_review.save_rows(cands_path, _two_candidates())
    cli_mod = _patch_golden_paths(monkeypatch, cands_path, golden_path)
    monkeypatch.setattr(cli_mod.click, "prompt", MagicMock(side_effect=["y", KeyboardInterrupt()]))

    result = CliRunner().invoke(cli_mod.cli, ["golden", "review"])

    # Click converts the propagated KeyboardInterrupt into Abort -> exit 1,
    # "Aborted!" — this only happens if the interrupt was allowed to
    # propagate (not swallowed) after saving.
    assert result.exit_code == 1
    assert "Aborted!" in result.output
    saved_cands = golden_review.load_rows(cands_path)
    assert saved_cands[0]["question"] == "Q1"
    assert saved_cands[0]["status"] == "accepted"
    assert saved_cands[1]["question"] == "Q2"
    assert saved_cands[1]["status"] == "candidate"
    saved_golden = golden_review.load_rows(golden_path)
    assert len(saved_golden) == 1
    assert saved_golden[0]["question"] == "Q1"
