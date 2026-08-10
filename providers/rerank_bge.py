"""BGE cross-encoder reranker.

Flips the pipeline's rerank stage from no-op to real. Uses
`sentence_transformers.CrossEncoder` so the runtime path matches the
embedder (and we get the same HF cache + device placement).

Default model: BAAI/bge-reranker-base (~280M params, ~0.5 GB fp16).
That fits comfortably alongside the embedder on a 24 GB card with
Ollama unloaded. To upgrade, uncomment the v2-gemma line in
config.yaml — it's a stronger model but ~2.6 B params (~5 GB fp16).

VRAM: with `device: cuda`, the model is loaded once at startup and
stays resident in the FastAPI lifespan, just like the embedder.

The cross-encoder is given (query, chunk_text) pairs and returns a
relevance score. We pair against `chunk.text` (NOT the source path)
so the score reflects semantic relevance to the question.
"""
from __future__ import annotations

import logging
from typing import Any

from core.interfaces import Chunk, Reranker


log = logging.getLogger(__name__)


# Default model — small + fast, fine for P1.
# Heavier `bge-reranker-v2-gemma` is available via config swap.
DEFAULT_MODEL = "BAAI/bge-reranker-base"


class BgeReranker(Reranker):
    """BGE cross-encoder reranker.

    Reads its config from the `reranker:` section of config.yaml. The
    factory (`core.pipeline.make_reranker`) inspects the constructor
    signature and passes only matching kwargs, so config can have
    comments / future keys without breaking instantiation.

    The constructor is intentionally lazy about loading: it builds a
    CrossEncoder on first instantiation. Re-ranking batches are sized
    to the model's batch_size config (default 32).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        # Imported lazily so the rest of the system can run on machines
        # where sentence-transformers isn't installed.
        from sentence_transformers import CrossEncoder

        log.info("loading %s on %s (max_length=%d, batch_size=%d)",
                 model, device, max_length, batch_size)
        self.model_name = model
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = CrossEncoder(
            model,
            max_length=max_length,
            device=device,
        )
        self._has_rank = hasattr(self._model, "rank")
        if not self._has_rank:
            log.info("reranker: CrossEncoder has no rank() (older "
                     "sentence-transformers) — using predict() fallback")
        log.info("reranker ready (model=%s, device=%s)", model, device)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Score (query, chunk) pairs and return the top_k best, preserving order ties by input order.

        Edge cases:
        - `chunks` empty → return [].
        - `top_k >= len(chunks)` → return the sorted list (no slicing needed).
        - `top_k <= 0` → treated as 0; return [].

        The CrossEncoder's `rank()` method is the canonical way to do
        this; it returns (corpus_id, score) tuples sorted by descending
        score, and respects `top_k`. We map corpus_id back to the
        input chunk.
        """
        if not chunks or top_k <= 0:
            return []
        if top_k > len(chunks):
            top_k = len(chunks)

        if self._has_rank:
            ranked = self._model.rank(
                query=query,
                documents=[c.text for c in chunks],
                top_k=top_k,
                batch_size=self.batch_size,
                show_progress_bar=False,
                return_documents=False,
            )
        else:
            # `rank()` is a convenience wrapper that some older
            # CrossEncoder versions don't have (decided once in
            # __init__). Fall back to predict() and sort manually.
            # Same semantics, slightly more code.
            pairs: list[tuple[str, str]] = [(query, c.text) for c in chunks]
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            indexed = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
            ranked = [(idx, float(score)) for idx, score in indexed[:top_k]]

        out: list[Chunk] = []
        for hit in ranked:
            # `rank()` returns either a list of dicts (newer API) or
            # a list of tuples (older API). Handle both.
            if isinstance(hit, dict):
                corpus_id = int(hit["corpus_id"])
            else:
                corpus_id = int(hit[0])
            out.append(chunks[corpus_id])
        return out
