"""Eval harness — recall@5, MRR, optional naive faithfulness.

Filename kept as `run_ragas.py` per the project taxonomy. There is no Ragas
framework dependency — metrics are computed directly from retrieval results.

Golden set format (`eval/golden_set.jsonl`, one JSON object per line):
    {
      "question": "What is exposure compensation?",
      "relevant_chunk_ids": ["uuid5-...", "uuid5-..."],
      "topic": "photography"      // optional, restricts the search
    }

Metrics:
    recall@k   : fraction of relevant chunks that appear in the top-k.
    MRR        : mean reciprocal rank of the FIRST relevant chunk (1/rank).
    faithfulness (optional): token-overlap heuristic between the generated
                answer and the union of retrieved chunk texts. NOT a real
                faithfulness measure — it catches the trivial "model ignored
                the sources" case, nothing more. The P1 plan is to swap in
                an NLI-based metric.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from core.pipeline import ask as ask_pipeline
from core.interfaces import Embedder, Reranker, Generator
from store.qdrant_store import QdrantStore


log = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def load_golden(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"golden set not found: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if "question" not in obj or "relevant_chunk_ids" not in obj:
                raise ValueError(f"line {ln}: missing question or relevant_chunk_ids")
            out.append(obj)
    return out


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: Iterable[str], k: int) -> float:
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & rel) / len(rel)


def mrr(retrieved_ids: list[str], relevant_ids: Iterable[str]) -> float:
    rel = set(relevant_ids)
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def faithfulness_proxy(answer: str, retrieved_texts: list[str]) -> float:
    """Token-overlap between the answer and the union of retrieved chunks.
    Returns 0..1. Crude but cheap; replace with an NLI-based scorer in P1."""
    if not answer or not retrieved_texts:
        return 0.0
    ans_t = _tokens(answer)
    if not ans_t:
        return 0.0
    src_t: set[str] = set()
    for t in retrieved_texts:
        src_t |= _tokens(t)
    if not src_t:
        return 0.0
    return len(ans_t & src_t) / len(ans_t)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run(
    golden_path: Path,
    embedder: Embedder,
    store: QdrantStore,
    reranker: Reranker,
    generator: Generator | None,
    top_k_dense: int = 20,
    top_k_final: int = 5,
) -> dict[str, Any]:
    """Compute aggregate metrics over the golden set."""
    gold = load_golden(golden_path)
    log.info("eval: %d questions from %s", len(gold), golden_path)

    recall_sum = 0.0
    mrr_sum = 0.0
    faithfulness_sum = 0.0
    faithfulness_count = 0
    per_question: list[dict[str, Any]] = []

    for i, item in enumerate(gold, 1):
        q = item["question"]
        rel = item["relevant_chunk_ids"]
        topic = item.get("topic")

        # Retrieve the top-k via the actual pipeline.
        result = ask_pipeline(
            q,
            embedder=embedder,
            store=store,
            reranker=reranker,
            generator=generator if generator else _SilentGenerator(),
            top_k_dense=top_k_dense,
            top_k_final=top_k_final,
            topic=topic,
        )
        retrieved_ids = [c["chunk_id"] for c in result.citations]
        # Audit #10: also compute dense-stage recall@k (pre-rerank). When
        # a real reranker lands in P1, this distinguishes "dense stage
        # missed the chunk entirely" from "reranker demoted a good hit".
        dense_ids = [c["chunk_id"] for c in result.dense_hits]
        r_dense = recall_at_k(dense_ids, rel, top_k_dense)
        r5 = recall_at_k(retrieved_ids, rel, 5)
        m = mrr(retrieved_ids, rel)
        recall_sum += r5
        mrr_sum += m
        log.info(
            "[%d/%d] %r  recall@%d(dense)=%.2f  recall@5(post-rerank)=%.2f  mrr=%.2f",
            i, len(gold), q, top_k_dense, r_dense, r5, m,
        )

        row: dict[str, Any] = {
            "question": q,
            "retrieved": retrieved_ids[:5],
            "relevant": rel,
            "recall_at_5": r5,
            "recall_at_dense": r_dense,
            "mrr": m,
        }
        if generator is not None:
            # Bug #5 fix: pass chunk TEXT (not source_path) so the proxy
            # measures actual answer↔content overlap. The previous code
            # measured answer↔file_path overlap, which was meaningless.
            f = faithfulness_proxy(result.answer, [c["text"] for c in result.citations])
            faithfulness_sum += f
            faithfulness_count += 1
            row["faithfulness_proxy"] = f
        per_question.append(row)

    n = max(1, len(gold))
    recall_dense_sum = sum(r.get("recall_at_dense", 0.0) for r in per_question)
    summary = {
        "n_questions": len(gold),
        "recall_at_5": round(recall_sum / n, 4),
        "recall_at_dense": round(recall_dense_sum / n, 4),
        "mrr": round(mrr_sum / n, 4),
        "per_question": per_question,
    }
    if faithfulness_count:
        summary["faithfulness_proxy"] = round(faithfulness_sum / faithfulness_count, 4)
    return summary


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def print_report(metrics: dict[str, Any]) -> None:
    print("=" * 60)
    print(f"  RAG eval — {metrics['n_questions']} questions")
    print("=" * 60)
    print(f"  recall@dense (pre-rerank) : {metrics.get('recall_at_dense', 0.0):.3f}")
    print(f"  recall@5 (post-rerank)    : {metrics['recall_at_5']:.3f}")
    print(f"  MRR                       : {metrics['mrr']:.3f}")
    if "faithfulness_proxy" in metrics:
        print(f"  faithfulness (proxy)      : {metrics['faithfulness_proxy']:.3f}")
    print("=" * 60)
    print("  per-question:")
    for row in metrics["per_question"]:
        ok = "✓" if row["recall_at_5"] >= 1.0 else ("·" if row["recall_at_5"] > 0 else "✗")
        rd = row.get("recall_at_dense", 0.0)
        print(
            f"    {ok} recall@dense={rd:.2f}  recall@5={row['recall_at_5']:.2f}  "
            f"mrr={row['mrr']:.2f}  q={row['question']!r}"
        )


# -----------------------------------------------------------------------------
# Helper: silent generator for retrieval-only runs
# -----------------------------------------------------------------------------

class _SilentGenerator(Generator):
    """No-op generator so the pipeline can be exercised end-to-end without
    spending tokens on the LLM during eval that only measures retrieval."""

    def generate(self, prompt: str) -> str:
        return ""


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--golden", default="eval/golden_set.jsonl")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    from core.pipeline import load_config, make_embedder, make_reranker, make_store

    cfg = load_config(args.config)
    embedder = make_embedder(cfg)
    store = make_store(cfg)
    reranker = make_reranker(cfg)
    metrics = run(Path(args.golden), embedder, store, reranker, generator=None)
    print_report(metrics)
