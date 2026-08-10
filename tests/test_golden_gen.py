"""Candidate generation: stratified sampling, LLM drafting, resume-safe append."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from eval import golden_gen


def _payload(cid, topic="notes"):
    return {"chunk_id": cid, "text": f"Some passage text for {cid}.",
            "source_path": f"/src/{cid}.md", "topic": topic,
            "doc_type": "markdown", "section": f"Sec {cid}"}


def _store(payloads):
    store = MagicMock()
    store.iter_payloads.return_value = iter([(p["chunk_id"], p) for p in payloads])
    return store


def test_sample_chunks_stratifies_across_topics():
    payloads = [_payload(f"a{i}", "alpha") for i in range(10)] + \
               [_payload(f"b{i}", "beta") for i in range(10)]
    picked = golden_gen.sample_chunks(_store(payloads), n=6)
    topics = [p["topic"] for p in picked]
    assert len(picked) == 6
    assert topics.count("alpha") == 3 and topics.count("beta") == 3


def test_sample_chunks_topic_filter():
    payloads = [_payload("a1", "alpha"), _payload("b1", "beta")]
    picked = golden_gen.sample_chunks(_store(payloads), n=5, topics=["beta"])
    assert [p["chunk_id"] for p in picked] == ["b1"]


def test_draft_question_strips_and_rejects_empty():
    gen = MagicMock()
    gen.generate.return_value = "  What is exposure compensation?  \n"
    assert golden_gen.draft_question(gen, "text") == "What is exposure compensation?"
    gen.generate.return_value = "   "
    assert golden_gen.draft_question(gen, "text") is None


def test_generate_candidates_appends_and_resumes(tmp_path):
    out = tmp_path / "cands.jsonl"
    gen = MagicMock()
    gen.generate.return_value = "A question?"
    store = _store([_payload("c1"), _payload("c2")])
    n1 = golden_gen.generate_candidates(store, gen, out, n=2)
    assert n1 == 2
    # Re-run with the same corpus: everything already present -> 0 new rows.
    store2 = _store([_payload("c1"), _payload("c2")])
    n2 = golden_gen.generate_candidates(store2, gen, out, n=2)
    assert n2 == 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["status"] == "candidate"
    # Check that both expected sections are present (order may vary based on shuffle)
    sections = [row["relevant_refs"][0]["section"] for row in rows]
    assert "Sec c1" in sections and "Sec c2" in sections


def test_generate_candidates_skips_failed_drafts(tmp_path):
    out = tmp_path / "cands.jsonl"
    gen = MagicMock()
    gen.generate.side_effect = ["", "Good question?"]
    n = golden_gen.generate_candidates(_store([_payload("c1"), _payload("c2")]), gen, out, n=2)
    assert n == 1
