"""BGE-reranker-v2 — P1 stub.

P0 uses core.pipeline.PassthroughReranker (no-op). When you're ready for P1:
    1. `pip install FlagEmbedding` (or transformers + the BGE reranker
       model from HF).
    2. Implement rerank() — score (query, chunk) pairs and return the top-k
       sorted by descending score.
    3. Flip `reranker.class` in config.yaml to providers.rerank_bge.BgeReranker.

The chunk schema is already what rerankers need (text + metadata), so the
swap is local to this file.
"""
from __future__ import annotations

import logging

from core.interfaces import Reranker, Chunk


log = logging.getLogger(__name__)


class BgeReranker(Reranker):
    """Stub. Raises NotImplementedError."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "BgeReranker is a P1 stub. P0 uses core.pipeline.PassthroughReranker."
        )

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        raise NotImplementedError
