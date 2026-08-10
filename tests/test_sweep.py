"""Sweep grid construction, per-combo isolation, reranker reuse."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from eval import sweep


def test_build_grid_defaults():
    grid = sweep.build_grid({})
    # 3 dense_weights x 2 top_k_dense x 1 top_k_final x 2 rerankers = 12
    assert len(grid) == 12
    assert {"dense_weight", "top_k_dense", "top_k_final", "reranker"} == set(grid[0])


def test_build_grid_config_override():
    cfg = {"eval": {"sweep": {"dense_weight": [0.5], "top_k_dense": [10],
                              "top_k_final": [5], "reranker": ["passthrough"]}}}
    grid = sweep.build_grid(cfg)
    assert grid == [{"dense_weight": 0.5, "top_k_dense": 10,
                     "top_k_final": 5, "reranker": "passthrough"}]


def test_run_sweep_isolates_combo_failures(tmp_path):
    cfg = {"eval": {"sweep": {"dense_weight": [0.3, 0.7], "top_k_dense": [20],
                              "top_k_final": [5], "reranker": ["passthrough"]}}}
    calls = []

    def fake_run(golden_path, **kw):
        calls.append(kw)
        if kw["dense_weight"] == 0.3:
            raise RuntimeError("boom")
        return {"recall_at_5": 0.9, "mrr": 0.8}

    with patch.object(sweep.run_ragas, "run", side_effect=fake_run):
        results = sweep.run_sweep(Path("g.jsonl"), cfg,
                                  embedder=MagicMock(), store=MagicMock())
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"].startswith("failed")]
    assert len(ok) == 1 and len(failed) == 1
    assert ok[0]["recall_at_5"] == 0.9
    assert results[0]["status"] == "ok"  # ok rows sort first
    assert all(kw["hybrid"] is True for kw in calls)


def test_run_sweep_builds_each_reranker_once():
    cfg = {"eval": {"sweep": {"dense_weight": [0.3, 0.7], "top_k_dense": [20],
                              "top_k_final": [5], "reranker": ["bge"]}}}
    with patch.object(sweep.run_ragas, "run",
                      return_value={"recall_at_5": 1.0, "mrr": 1.0}), \
         patch.object(sweep, "_make_reranker_for") as mk:
        mk.return_value = MagicMock()
        sweep.run_sweep(Path("g.jsonl"), cfg, embedder=MagicMock(), store=MagicMock())
    assert mk.call_count == 1  # 2 combos, same reranker instance reused


def test_format_table_contains_params_and_metrics():
    results = [{"params": {"dense_weight": 0.5, "top_k_dense": 20,
                           "top_k_final": 5, "reranker": "bge"},
                "status": "ok", "recall_at_5": 0.91, "mrr": 0.85}]
    table = sweep.format_table(results)
    assert "0.91" in table and "bge" in table
