"""Qwen3-Embedding-8B embedder, Q4 via bitsandbytes.

P0 default. Loaded via sentence-transformers so the path matches every other
HF model you might swap in. The model has custom code on the Hub
(`trust_remote_code=True`) and a sentence-transformers config that the loader
honors automatically.

VRAM at Q4: ~5–6 GB. The compute dtype is bfloat16 by default — change in
config.yaml if your GPU prefers float16.

First-run note: this downloads ~5 GB on first import. Subsequent runs are
instant (HF cache). The CLI also surfaces this in its setup hint.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from core.interfaces import Embedder


log = logging.getLogger(__name__)


def _build_quant_config(quant: str, compute_dtype: str) -> dict[str, Any] | None:
    """Map config.quant to a HF BitsAndBytesConfig dict.

    Returns None when quant == "none" (no quantization)."""
    if quant == "none":
        return None
    if quant not in {"q4", "q8"}:
        raise ValueError(f"unsupported quant={quant!r}; expected q4 | q8 | none")
    # bitsandbytes is an optional dep — only needed when quantizing.
    from transformers import BitsAndBytesConfig

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[compute_dtype]
    if quant == "q4":
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        }
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True,
        )
    }


class Qwen3Embedder(Embedder):
    """Qwen3-Embedding-8B via sentence-transformers.

    Reads its own config from the EmbedderConfig — see `core/pipeline.py` for
    the factory that builds instances from config.yaml.

    Audit #9: the model is instruction-tuned. The Qwen3-Embedding model
    card recommends a query instruction prefix for asymmetric retrieval
    (query and document spaces differ). We default to the standard
    "web search" instruction, which is what Qwen3-Embedding was trained
    for; override via `query_instruction` in config.yaml.
    """

    # Standard Qwen3-Embedding retrieval-query instruction. Override via config.
    DEFAULT_QUERY_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-8B",
        quant: str = "q4",
        compute_dtype: str = "bfloat16",
        device: str = "cuda",
        batch_size: int = 8,
        normalize: bool = True,
        max_seq_length: int = 8192,
        query_instruction: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_instruction = query_instruction if query_instruction is not None else self.DEFAULT_QUERY_INSTRUCTION

        log.info("loading %s on %s (quant=%s, compute=%s)", model, device, quant, compute_dtype)
        model_kwargs = _build_quant_config(quant, compute_dtype) or {}
        # trust_remote_code is needed because Qwen3-Embedding ships custom modules.
        self._model = SentenceTransformer(
            model,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        # Cap seq length to keep memory bounded on long chunks.
        self._model.max_seq_length = max_seq_length
        # Probe dimensionality once.
        probe = self._model.encode(["probe"], normalize_embeddings=normalize, convert_to_numpy=False)
        self._dim = len(probe[0])
        log.info("embedder ready (dim=%d, max_seq_length=%d)", self._dim, self._model.max_seq_length)

    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with the retrieval instruction prefix.

        Documents (chunks at ingest time) are embedded WITHOUT the
        prefix; queries (one at retrieval time) are embedded WITH it.
        This asymmetry measurably improves recall on Qwen3-Embedding.
        """
        if not text.strip():
            # A zero vector has no direction — cosine against it is undefined,
            # and whitespace-only input would embed just the instruction
            # prefix. Both silently return junk matches; fail loudly instead.
            raise ValueError("embed_query: query text is empty or whitespace-only")
        prompted = f"{self.query_instruction}\n{text}"
        return self.embed_documents([prompted])[0]
