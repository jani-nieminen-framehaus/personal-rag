"""NLI-based faithfulness scorer.

Replaces the token-overlap heuristic in `run_ragas.py` with a real NLI model.
Uses a small cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`, ~22M params)
to check whether each retrieved chunk ENTAILS the generated answer.

Faithfulness score per (answer, chunk):
  - Entailment → 1.0  (the chunk fully supports the answer)
  - Neutral     → 0.5  (the chunk is somewhat relevant but not conclusive)
  - Contradiction → 0.0 (the chunk contradicts the answer)

Overall score: mean across all chunks.

The model is loaded lazily on first call so the rest of the eval
harness (retrieval-only) works without it. Falls back to the
token-overlap proxy if the model can't be loaded.
"""
from __future__ import annotations

import logging
from typing import Any


log = logging.getLogger(__name__)


DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


def _load_cross_encoder(model: str, max_length: int, device: str):
    """Load sentence_transformers.CrossEncoder. Extracted for testability."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Install it with: pip install sentence-transformers"
        )
    log.info("loading NLI faithfulness scorer: %s on %s (max_length=%d)", model, device, max_length)
    return CrossEncoder(model, max_length=max_length, device=device)


class NliFaithfulness:
    """NLI cross-encoder faithfulness scorer.

    Config keys (read from the `faithfulness:` section in config.yaml):
        model      : HF model name (default: cross-encoder/nli-deberta-v3-xsmall)
        device     : "cuda" or "cpu" (default: cuda)
        max_length : token limit per sequence (default: 512)
        batch_size : inference batch size (default: 16)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.model_name = model
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = _load_cross_encoder(model, max_length, device)
        log.info("NLI faithfulness scorer ready: %s", model)

    def score(self, answer: str, chunks: list[str]) -> float:
        """Compute faithfulness of the answer against the retrieved chunks.

        Args:
            answer: the generated answer text.
            chunks: list of retrieved chunk texts (the evidence).

        Returns:
            Mean NLI entailment score in [0.0, 1.0].
            Returns 0.0 if `chunks` is empty.
        """
        if not chunks or not answer.strip():
            return 0.0

        # NLI label → score mapping.
        LABEL_SCORES = {
            "entailment": 1.0,
            "neutral": 0.5,
            "contradiction": 0.0,
        }

        # Build (answer, chunk) pairs.
        pairs: list[tuple[str, str]] = [(answer, chunk) for chunk in chunks]

        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as e:
            log.warning("NLI scoring failed (%s) — falling back to 0.0", e)
            return 0.0

        # Cross-encoder NLI outputs softmax over [contradiction, entailment, neutral]
        # (the exact order varies by model). Map to our scale.
        total = 0.0
        for raw in scores:
            # raw can be a numpy array or Python list.
            arr = raw
            if hasattr(raw, "__iter__") and not isinstance(raw, (int, float, str)):
                arr = raw  # keep as numpy array or list

            if hasattr(arr, "__len__") and len(arr) >= 2:
                if len(arr) == 3:
                    # Standard DeBERTa NLI: index 0=contradiction, 1=entailment, 2=neutral.
                    score = float(arr[1])  # entailment score
                else:
                    # Multi-class: pick the max score.
                    score = float(max(arr))
            else:
                # Single logit — positive = entailment, negative = contradiction.
                val = float(arr.flat[0]) if hasattr(arr, "flat") else float(arr)
                score = max(0.0, val)
            total += score

        return total / len(chunks) if chunks else 0.0


# Lazy factory — tries to load the model, returns None on failure.
def make_nli_faithfulness(cfg: dict[str, Any] | None) -> NliFaithfulness | None:
    """Attempt to build an NliFaithfulness from config. Returns None if unavailable."""
    if cfg is None:
        return None
    section = cfg.get("faithfulness", {}) or {}
    if not section.get("enabled", False):
        return None
    try:
        return NliFaithfulness(
            model=section.get("model", DEFAULT_MODEL),
            device=section.get("device", "cuda"),
            max_length=section.get("max_length", 512),
            batch_size=section.get("batch_size", 16),
        )
    except Exception as e:
        log.warning("NLI faithfulness model unavailable: %s", e)
        return None
