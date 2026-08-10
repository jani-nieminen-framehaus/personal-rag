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


def test_cli_eval_sweep_prints_table(monkeypatch):
    from click.testing import CliRunner
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "make_embedder", lambda cfg: MagicMock())
    monkeypatch.setattr(cli_mod, "make_store", lambda cfg: MagicMock())
    monkeypatch.setattr(cli_mod, "make_metadata", lambda cfg: None)

    fake_results = [{"params": {"dense_weight": 0.5, "top_k_dense": 20,
                                "top_k_final": 5, "reranker": "bge"},
                     "status": "ok", "recall_at_5": 0.91, "mrr": 0.85}]
    with patch("eval.sweep.run_sweep", return_value=fake_results):
        res = CliRunner().invoke(cli_mod.cli, ["eval", "--sweep"])
    assert res.exit_code == 0
    assert "recall@5" in res.output and "0.91" in res.output


def test_chunking_sweep_uses_scratch_collections_and_drops_them(tmp_path):
    cfg = {"store": {"class": "store.qdrant_store.QdrantStore",
                     "url": "http://localhost:6333", "collection": "kb",
                     "dense_dim": 4}}
    made = []

    def fake_make_store(c):
        s = MagicMock()
        s.collection = c["store"]["collection"]
        made.append(s)
        return s

    with patch.object(sweep, "make_store", side_effect=fake_make_store), \
         patch.object(sweep, "ingest_pipeline", return_value=3), \
         patch.object(sweep.run_ragas, "run",
                      return_value={"recall_at_5": 0.8, "mrr": 0.7}) as run_mock:
        results = sweep.run_chunking_sweep(
            Path("g.jsonl"), cfg, markdown_root=str(tmp_path),
            embedder=MagicMock(), target_tokens_list=[512, 1024])

    assert [s.collection for s in made] == ["kb_tune_512", "kb_tune_1024"]
    for s in made:
        s.drop.assert_called_once()          # cleaned up even on success
    assert all(c.kwargs["match_mode"] == "section"
               for c in run_mock.call_args_list)
    assert {r["params"]["target_tokens"] for r in results} == {512, 1024}


def test_chunking_sweep_drops_scratch_on_failure(tmp_path):
    cfg = {"store": {"class": "x", "url": "u", "collection": "kb", "dense_dim": 4}}
    scratch = MagicMock()
    scratch.collection = "kb_tune_512"
    with patch.object(sweep, "make_store", return_value=scratch), \
         patch.object(sweep, "ingest_pipeline", side_effect=RuntimeError("embed died")):
        results = sweep.run_chunking_sweep(
            Path("g.jsonl"), cfg, markdown_root=str(tmp_path),
            embedder=MagicMock(), target_tokens_list=[512])
    scratch.drop.assert_called_once()
    assert results[0]["status"].startswith("failed")
