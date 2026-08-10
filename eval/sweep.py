"""Query-time eval sweep: grid over retrieval knobs, per-combo isolation.

No re-ingestion here — chunking sweeps live in their own heavy mode."""
from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

from core.pipeline import PassthroughReranker, make_reranker
from eval import run_ragas

log = logging.getLogger(__name__)

DEFAULT_SWEEP: dict[str, list] = {
    "dense_weight": [0.3, 0.5, 0.7],
    "top_k_dense": [20, 40],
    "top_k_final": [5],
    "reranker": ["passthrough", "bge"],
}


def build_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    axes = {**DEFAULT_SWEEP, **((cfg.get("eval") or {}).get("sweep") or {})}
    keys = ["dense_weight", "top_k_dense", "top_k_final", "reranker"]
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(axes[k] for k in keys))]


def _make_reranker_for(name: str, cfg: dict[str, Any]):
    if name == "passthrough":
        return PassthroughReranker()
    if name == "bge":
        return make_reranker(cfg)
    raise ValueError(f"unknown reranker axis value: {name!r}")


def run_sweep(
    golden_path: Path,
    cfg: dict[str, Any],
    embedder,
    store,
    metadata=None,
    generator=None,
    nli=None,
) -> list[dict[str, Any]]:
    rerankers: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    grid = build_grid(cfg)
    for i, combo in enumerate(grid, 1):
        log.info("sweep [%d/%d]: %s", i, len(grid), combo)
        try:
            if combo["reranker"] not in rerankers:
                rerankers[combo["reranker"]] = _make_reranker_for(combo["reranker"], cfg)
            metrics = run_ragas.run(
                golden_path,
                embedder=embedder,
                store=store,
                reranker=rerankers[combo["reranker"]],
                generator=generator,
                top_k_dense=combo["top_k_dense"],
                top_k_final=combo["top_k_final"],
                metadata=metadata,
                nli_faithfulness=nli,
                hybrid=True,
                dense_weight=combo["dense_weight"],
                extra_params=dict(combo),
            )
            results.append({"params": combo, "status": "ok",
                            "recall_at_5": metrics["recall_at_5"],
                            "mrr": metrics["mrr"]})
        except Exception as e:
            log.warning("sweep: combo %s failed: %s", combo, e)
            results.append({"params": combo, "status": f"failed: {e}",
                            "recall_at_5": None, "mrr": None})
    results.sort(key=lambda r: (r["status"] != "ok",
                                -(r["recall_at_5"] or 0.0),
                                -(r["mrr"] or 0.0)))
    return results


def format_table(results: list[dict[str, Any]]) -> str:
    header = f'{"dense_w":>7}  {"k_dense":>7}  {"k_final":>7}  {"reranker":<11}  {"recall@5":>8}  {"mrr":>6}  status'
    lines = [header, "-" * len(header)]
    for r in results:
        p = r["params"]
        r5 = "-" if r["recall_at_5"] is None else f'{r["recall_at_5"]:.2f}'
        mrr = "-" if r["mrr"] is None else f'{r["mrr"]:.2f}'
        lines.append(f'{p["dense_weight"]:>7}  {p["top_k_dense"]:>7}  '
                     f'{p["top_k_final"]:>7}  {p["reranker"]:<11}  '
                     f'{r5:>8}  {mrr:>6}  {r["status"]}')
    return "\n".join(lines)
