"""Unit tests for the eval harness (`eval/run_ragas.py`).

Currently covers:
- `print_report` doesn't crash on Windows consoles with a non-UTF-8
  code page (the `✓` / `✗` glyphs are unicode). Regression test
  for the cp1252 crash we hit in the live smoke pass.
"""
from __future__ import annotations

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
    if not hasattr(sys := __import__("sys"), "stdout"):
        pytest.skip("no sys.stdout")  # type: ignore[name-defined]
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


def test_print_report_handles_optional_faithfulness():
    """The faithfulness_proxy line is only printed when the metric is present."""
    metrics = {
        "n_questions": 1,
        "recall_at_5": 1.0,
        "recall_at_dense": 1.0,
        "mrr": 1.0,
        "per_question": [
            {"question": "q", "recall_at_5": 1.0, "recall_at_dense": 1.0, "mrr": 1.0},
        ],
    }
    # Without faithfulness - no KeyError on the lookup.
    run_ragas.print_report(metrics)

    # With faithfulness - should still print.
    metrics["faithfulness_proxy"] = 0.42
    run_ragas.print_report(metrics)
