"""Query-time eval sweep: grid over retrieval knobs, per-combo isolation.

Also includes the chunking sweep heavy mode — re-ingesting into throwaway
scratch collections, since chunk_ids change with chunk size."""
from __future__ import annotations

import copy
import itertools
import logging
from pathlib import Path
from typing import Any

from core.pipeline import (
    PassthroughReranker,
    make_reranker,
    make_store,
    chunking_params,
    ingest as ingest_pipeline,
)
from eval import run_ragas
from ingest.markdown_dir import MarkdownDirIngester

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


# -----------------------------------------------------------------------------
# Chunking sweep (heavy mode): scratch collections, re-ingestion per combo
# -----------------------------------------------------------------------------

DEFAULT_TARGET_TOKENS = [512, 768, 1024]


def run_chunking_sweep(
    golden_path: Path,
    cfg: dict[str, Any],
    markdown_root: str,
    embedder,
    target_tokens_list: list[int] | None = None,
    metadata=None,
) -> list[dict[str, Any]]:
    """Grid over target_tokens using scratch collections (kb_tune_<n>).

    The live collection is never touched. Metrics use section-level
    matching because chunk_ids are chunking-dependent (UUID5 over
    path/section/chunk_index). Markdown-only in P3.1."""
    results: list[dict[str, Any]] = []
    for tt in (target_tokens_list or DEFAULT_TARGET_TOKENS):
        combo = {"target_tokens": tt}
        scratch_cfg = copy.deepcopy(cfg)
        scratch_cfg["store"]["collection"] = f"kb_tune_{tt}"
        scratch_cfg.setdefault("chunking", {})["target_tokens"] = tt
        scratch = None
        try:
            scratch = make_store(scratch_cfg)
            cp = chunking_params(scratch_cfg)
            ingester = MarkdownDirIngester(
                root=markdown_root,
                target_tokens=cp["target_tokens"],
                overlap_pct=cp["overlap_pct"],
                min_chunk_tokens=cp["min_chunk_tokens"],
                default_topic=cp["default_topic"],
                max_chunks_per_doc=cp["max_chunks_per_doc"],
            )
            ingest_pipeline(ingester, embedder, scratch, recreate=True)
            metrics = run_ragas.run(
                golden_path, embedder=embedder, store=scratch,
                reranker=PassthroughReranker(), generator=None,
                metadata=metadata, match_mode="section",
                extra_params=dict(combo, sweep="chunking"),
            )
            results.append({"params": combo, "status": "ok",
                            "recall_at_5": metrics["recall_at_5"],
                            "mrr": metrics["mrr"]})
        except Exception as e:
            log.warning("chunking sweep: target_tokens=%d failed: %s", tt, e)
            results.append({"params": combo, "status": f"failed: {e}",
                            "recall_at_5": None, "mrr": None})
        finally:
            if scratch is not None:
                try:
                    scratch.drop()
                except Exception as e:
                    log.warning("chunking sweep: could not drop %s: %s",
                                scratch.collection, e)
    results.sort(key=lambda r: (r["status"] != "ok",
                                -(r["recall_at_5"] or 0.0)))
    return results


def format_chunking_table(results: list[dict[str, Any]]) -> str:
    header = f'{"target_tokens":>13}  {"recall@5":>8}  {"mrr":>6}  status'
    lines = [header, "-" * len(header)]
    for r in results:
        r5 = "-" if r["recall_at_5"] is None else f'{r["recall_at_5"]:.2f}'
        mrr = "-" if r["mrr"] is None else f'{r["mrr"]:.2f}'
        lines.append(f'{r["params"]["target_tokens"]:>13}  {r5:>8}  {mrr:>6}  {r["status"]}')
    return "\n".join(lines)
