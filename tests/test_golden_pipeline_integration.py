"""Seam test: the golden-set chain must actually compose.

Every stage of P3.1 is unit-tested in isolation and each passed its own
review — but nothing asserted that

    generate → curate → persist → load → score

fits together. Rename one field anywhere along that chain (say
`relevant_refs` -> `refs`, or `source_path` -> `path`) and every stage
still passes its own tests while `rag eval` silently reports 0.0. This is
the test that catches that, with no network, no Ollama, no Qdrant, and no
models.

The pieces are deliberately joined the way production joins them:
citations are built from the STORE PAYLOAD, exactly as pipeline.ask()
builds them, never from the golden row. If both sides were built from the
golden row, a renamed field would be mirrored on both sides and the test
would pass while the real eval scored zero.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from core.pipeline import AskResult
from eval import golden_gen, golden_review, run_ragas


# A miniature "live index": the payload shape VectorStore.iter_payloads yields.
CORPUS = [
    {
        "chunk_id": "chunk-aaa",
        "text": "Exposure compensation shifts the meter's target exposure.",
        "source_path": r"D:\notes\photography\exposure.md",
        "topic": "photography",
        "doc_type": "markdown",
        "section": "Exposure compensation",
    },
    {
        "chunk_id": "chunk-bbb",
        "text": "Reciprocal rank fusion merges two ranked lists by rank, not score.",
        "source_path": r"D:\notes\ml\hybrid.md",
        "topic": "ml",
        "doc_type": "markdown",
        "section": "Reciprocal rank fusion",
    },
]


class _FakeStore:
    """Just enough VectorStore for golden_gen.sample_chunks()."""

    def iter_payloads(self):
        return iter([(p["chunk_id"], p) for p in CORPUS])


class _FakeGenerator:
    """Drafts a distinct question per passage, offline."""

    def generate(self, prompt: str) -> str:
        for p in CORPUS:
            if p["text"] in prompt:
                return f"What does the note say about {p['section']}?"
        raise AssertionError(f"drafting prompt carried no known passage: {prompt!r}")


def _citations_from_payload(payload: dict) -> list[dict]:
    """Mirror of the citation dict pipeline.ask() builds (core/pipeline.py)."""
    return [{
        "n": 1,
        "source_path": payload["source_path"],
        "section": payload["section"],
        "chunk_id": payload["chunk_id"],
        "topic": payload["topic"],
        "doc_type": payload["doc_type"],
        "score": 0.9,
        "text": payload["text"],
    }]


def test_golden_chain_generate_curate_persist_load_score(tmp_path, monkeypatch):
    candidates = tmp_path / "golden_candidates.jsonl"
    golden_real = tmp_path / "golden_real.jsonl"

    # 1. generate ------------------------------------------------------------
    written = golden_gen.generate_candidates(
        _FakeStore(), _FakeGenerator(), candidates, n=2)
    assert written == 2, "both corpus chunks should yield a candidate"

    # 2. curate: accept one as-is, accept the other with an edited question --
    rows = golden_review.load_rows(candidates)
    assert len(golden_review.pending(rows)) == 2
    accepted: list[dict] = []
    edited_question = "Restated: how does rank fusion combine two result lists?"
    for i, row in enumerate(rows):
        decision, edit = ("y", None) if i == 0 else ("e", edited_question)
        _, golden_row = golden_review.apply_decision(row, decision, edited=edit)
        assert golden_row is not None
        # Curation-only fields must not leak into the golden set.
        assert "status" not in golden_row and "preview" not in golden_row
        accepted.append(golden_row)

    # 3. persist -------------------------------------------------------------
    golden_review.save_rows(golden_real, accepted)

    # 4. load ----------------------------------------------------------------
    loaded = run_ragas.load_golden(golden_real)
    assert len(loaded) == 2
    assert edited_question in {row["question"] for row in loaded}, (
        "the curator's edit must survive persist -> load")

    # Provenance: map each question back to the chunk it was drafted from,
    # via relevant_chunk_ids. Retrieval is then stubbed from the CORPUS
    # payload, so relevant_refs is genuinely being checked against an
    # independently built citation.
    by_chunk = {p["chunk_id"]: p for p in CORPUS}
    payload_for_question = {
        row["question"]: by_chunk[row["relevant_chunk_ids"][0]] for row in loaded
    }

    # 5. score ---------------------------------------------------------------
    seen_topics: list[str | None] = []

    def fake_ask(query, **kw):
        seen_topics.append(kw.get("topic"))
        payload = payload_for_question[query]
        citations = _citations_from_payload(payload)
        return AskResult(answer="", citations=citations,
                         dense_hits=[dict(c) for c in citations])

    monkeypatch.setattr(run_ragas, "ask_pipeline", fake_ask)

    section_metrics = run_ragas.run(
        golden_real, embedder=MagicMock(), store=MagicMock(),
        reranker=MagicMock(), generator=None, match_mode="section")

    assert section_metrics["n_questions"] == 2
    assert section_metrics["recall_at_5"] == 1.0, (
        "section-mode recall collapsed — a field name drifted somewhere in "
        "generate -> curate -> persist -> load -> score")
    assert section_metrics["mrr"] == 1.0

    # The topic stamped at generation time must reach retrieval: it is what
    # drives the topic-filtered (over-fetching) hybrid path.
    assert set(seen_topics) == {"photography", "ml"}

    # The provenance chunk_id survives the same chain, so the default
    # match mode scores just as cleanly.
    chunk_metrics = run_ragas.run(
        golden_real, embedder=MagicMock(), store=MagicMock(),
        reranker=MagicMock(), generator=None, match_mode="chunk_id")
    assert chunk_metrics["recall_at_5"] == 1.0
    assert chunk_metrics["mrr"] == 1.0


def test_golden_chain_detects_a_renamed_ref_field(tmp_path, monkeypatch):
    """Guard on the guard: if the citation's source_path stops lining up
    with the persisted relevant_refs, section mode must go to 0.0 — i.e.
    the test above is genuinely sensitive to the drift it claims to catch.
    """
    candidates = tmp_path / "c.jsonl"
    golden_real = tmp_path / "g.jsonl"
    golden_gen.generate_candidates(_FakeStore(), _FakeGenerator(), candidates, n=2)
    rows = golden_review.load_rows(candidates)
    accepted = [golden_review.apply_decision(r, "y")[1] for r in rows]
    golden_review.save_rows(golden_real, accepted)

    loaded = run_ragas.load_golden(golden_real)
    by_chunk = {p["chunk_id"]: p for p in CORPUS}
    payload_for_question = {
        row["question"]: by_chunk[row["relevant_chunk_ids"][0]] for row in loaded
    }

    def drifted_ask(query, **kw):
        payload = dict(payload_for_question[query])
        # Simulate the repo-relative vs absolute source_path mismatch.
        payload["source_path"] = "notes/photography/exposure.md"
        citations = _citations_from_payload(payload)
        return AskResult(answer="", citations=citations,
                         dense_hits=[dict(c) for c in citations])

    monkeypatch.setattr(run_ragas, "ask_pipeline", drifted_ask)
    metrics = run_ragas.run(
        golden_real, embedder=MagicMock(), store=MagicMock(),
        reranker=MagicMock(), generator=None, match_mode="section")
    assert metrics["recall_at_5"] == 0.0


def test_golden_chain_fails_loudly_if_refs_stop_being_emitted(tmp_path, monkeypatch):
    """The other half of the drift story: if golden_gen ever stopped
    emitting `relevant_refs` (renamed key, dropped field), the persisted
    set must make section mode RAISE rather than quietly score 0.0."""
    import pytest

    candidates = tmp_path / "c.jsonl"
    golden_real = tmp_path / "g.jsonl"
    golden_gen.generate_candidates(_FakeStore(), _FakeGenerator(), candidates, n=2)
    rows = golden_review.load_rows(candidates)
    accepted = []
    for r in rows:
        _, golden_row = golden_review.apply_decision(r, "y")
        golden_row.pop("relevant_refs")     # the rename, simulated at persist time
        accepted.append(golden_row)
    golden_review.save_rows(golden_real, accepted)

    monkeypatch.setattr(run_ragas, "ask_pipeline",
                        lambda *a, **kw: AskResult(answer="", citations=[]))
    with pytest.raises(ValueError, match="relevant_refs"):
        run_ragas.run(golden_real, embedder=MagicMock(), store=MagicMock(),
                      reranker=MagicMock(), generator=None, match_mode="section")
