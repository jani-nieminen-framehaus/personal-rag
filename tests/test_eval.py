"""Unit tests for the eval harness (`eval/run_ragas.py`).

Currently covers:
- `print_report` doesn't crash on Windows consoles with a non-UTF-8
  code page (the `✓` / `✗` glyphs are unicode). Regression test
  for the cp1252 crash we hit in the live smoke pass.
- `run` records the size of the index each run was measured against,
  so a recall trend across months can tell a real regression from a
  corpus that simply grew.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

from eval import run_ragas


# -- print_report encoding --------------------------------------------------

def test_print_report_survives_cp1252_stdout():
    """`rag eval` on a default Windows console uses cp1252, which can't
    encode the ✓ / · / ✗ glyphs. The report must reconfigure stdout
    to UTF-8 before printing them, or the whole eval crashes AFTER
    the metrics are computed (super annoying for the user).

    We simulate the cp1252 console by reconfiguring sys.stdout's
    encoding, then restore the original encoding in finally. This
    doesn't replace the underlying file object, so subsequent tests
    can still print.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is None:
        # Python < 3.7 or non-TTY stdout - nothing to test.
        return
    original_encoding = sys.stdout.encoding
    try:
        reconfigure(encoding="cp1252")
        metrics = {
            "n_questions": 3,
            "recall_at_5": 0.66,
            "recall_at_dense": 0.83,
            "mrr": 0.75,
            "per_question": [
                {"question": "q1", "recall_at_5": 1.0, "recall_at_dense": 1.0, "mrr": 1.0},
                {"question": "q2", "recall_at_5": 0.0, "recall_at_dense": 0.5, "mrr": 0.0},
                {"question": "q3", "recall_at_5": 0.5, "recall_at_dense": 1.0, "mrr": 0.5},
            ],
        }
        # If this raises UnicodeEncodeError, the fix is missing.
        run_ragas.print_report(metrics)
    finally:
        try:
            reconfigure(encoding=original_encoding or "utf-8")
        except Exception:
            pass


def test_print_report_handles_optional_faithfulness(capsys):
    """The faithfulness line is printed exactly when the metric is present."""
    metrics = {
        "n_questions": 1,
        "recall_at_5": 1.0,
        "recall_at_dense": 1.0,
        "mrr": 1.0,
        "per_question": [
            {"question": "q", "recall_at_5": 1.0, "recall_at_dense": 1.0, "mrr": 1.0},
        ],
    }
    run_ragas.print_report(metrics)
    out = capsys.readouterr().out
    assert "faithfulness" not in out.lower()

    metrics["faithfulness_proxy"] = 0.42
    run_ragas.print_report(metrics)
    out = capsys.readouterr().out
    assert "faithfulness" in out.lower()
    assert "0.420" in out


# -- recorded run params ------------------------------------------------------

def _one_question_run(tmp_path, monkeypatch, store):
    """Drive `run` over a single golden row with the pipeline stubbed out.

    Returns the MagicMock metadata store so the caller can read what was
    recorded. Nothing here touches Ollama, Qdrant or the network.
    """
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        json.dumps({"question": "q", "relevant_chunk_ids": ["c1"]}) + "\n",
        encoding="utf-8",
    )

    result = MagicMock()
    result.answer = "a"
    result.citations = [{"chunk_id": "c1", "text": "t",
                         "source_path": "/a.md", "section": "S"}]
    result.dense_hits = list(result.citations)
    monkeypatch.setattr(run_ragas, "ask_pipeline", lambda *a, **k: result)

    metadata = MagicMock()
    run_ragas.run(golden, embedder=MagicMock(), store=store,
                  reranker=MagicMock(), generator=None, metadata=metadata)
    return metadata


def test_eval_records_the_index_size(monkeypatch, tmp_path):
    """Without this, a recall trend across months reads as degradation when
    it is really just a growing corpus — retrieval gets harder as more
    near-duplicates compete for the same few slots. The README already warns
    about the trap; the recorded row is what lets you check it after the fact."""
    store = MagicMock()
    store.count.return_value = 4242
    metadata = _one_question_run(tmp_path, monkeypatch, store)
    assert metadata.record_eval_run.call_args.kwargs["params"]["index_points"] == 4242


def test_index_size_of_a_store_that_cannot_count_is_unknown_not_fatal(monkeypatch, tmp_path):
    """A store that cannot report a count must not break an eval run: the
    metrics are the point, the corpus size is context. Recorded as None so a
    later reader can tell "nobody asked" apart from "the index was empty"."""
    store = MagicMock()
    store.count.side_effect = RuntimeError("qdrant said no")
    metadata = _one_question_run(tmp_path, monkeypatch, store)
    assert metadata.record_eval_run.call_args.kwargs["params"]["index_points"] is None
