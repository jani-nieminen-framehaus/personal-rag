"""BGE-M3 embedder — P1 stub.

BGE-M3 is interesting for P1 because it produces dense + sparse (lexical) +
multi-vector (ColBERT-style) representations from a single forward pass. The
plan was to swap it in to enable hybrid retrieval without a separate BM25
index. Implementation deferred — see PLAN.md §9.
"""
from __future__ import annotations

import logging

from core.interfaces import Embedder


log = logging.getLogger(__name__)


class BgeM3Embedder(Embedder):
    """Stub. Raises NotImplementedError on any method call.

    To enable in P1:
        1. `pip install FlagEmbedding` (or use the HF AutoModel path).
        2. Implement embed() to return dense vectors for the dense store AND
           emit sparse vectors via the BGE-M3 lexical weights for Qdrant's
           `sparse` named vector.
        3. Wire `core.pipeline.retrieve_hybrid` to combine dense + sparse
           with RRF.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "BgeM3Embedder is a P1 stub. "
            "Switch `embedder.class` in config.yaml to providers.embed_qwen3.Qwen3Embedder for P0."
        )

    def dim(self) -> int:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
