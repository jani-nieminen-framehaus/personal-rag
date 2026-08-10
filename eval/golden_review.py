"""Curation for golden candidates: pure state transitions + JSONL IO.

The CLI drives the interaction; everything testable lives here."""
from __future__ import annotations

import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    tmp.replace(path)


def pending(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == "candidate"]


def apply_decision(candidate: dict, decision: str, edited: str | None = None):
    """y = accept, n = reject, e = accept with edited question.

    Returns (updated_candidate, golden_row_or_None). The golden row drops
    curation-only fields (status, preview).

    MUTATION CONTRACT: this function mutates `candidate` in place (setting
    its "status", and its "question" on an edit) AND returns it as
    `updated_candidate`. This is deliberate, not an oversight: the CLI's
    `review` command keeps the full `rows` list (which contains this same
    dict by reference) and writes it back to `golden_candidates.jsonl`
    wholesale after the review pass. The in-place mutation is what makes
    that write-back reflect each row's new status. Do NOT "fix" this into
    a pure function that only returns a new dict — doing so would silently
    break status persistence across `rag golden review` sessions.
    """
    if decision == "n":
        candidate["status"] = "rejected"
        return candidate, None
    if decision == "e":
        if not edited or not edited.strip():
            raise ValueError("edit decision requires a non-empty question")
        candidate["question"] = edited.strip()
    elif decision != "y":
        raise ValueError(f"unknown decision {decision!r}")
    candidate["status"] = "accepted"
    golden = {k: v for k, v in candidate.items() if k not in ("status", "preview")}
    return candidate, golden


def stats(candidates_path: Path, golden_path: Path) -> dict:
    cands = load_rows(candidates_path)
    accepted = load_rows(golden_path)
    by_topic: dict[str, int] = {}
    for row in accepted:
        t = row.get("topic", "default")
        by_topic[t] = by_topic.get(t, 0) + 1
    return {
        "candidates_pending": len(pending(cands)),
        "rejected": sum(1 for r in cands if r.get("status") == "rejected"),
        "accepted_total": len(accepted),
        "accepted_by_topic": by_topic,
    }
