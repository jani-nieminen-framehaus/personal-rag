"""Unit tests for the NLI faithfulness scorer.

Tests cover:
- NliFaithfulness.score() with mocked model output.
- make_nli_faithfulness() factory (enabled / disabled / load-failure paths).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class _FakeNliModel:
    """Fake NLI cross-encoder that returns controlled softmax outputs.

    output = [contradiction, entailment, neutral] scores.
    """

    def __init__(self, scores_per_pair):
        self._scores = scores_per_pair

    def predict(self, pairs, batch_size=16, show_progress_bar=False, convert_to_numpy=False):
        import numpy as np
        return [np.array(s, dtype=float) for s in self._scores]


# We patch inside NliFaithfulness.__init__ where the import happens.
_PATCH_TARGET = "eval.nli_faithfulness.CrossEncoderLoader"


class TestNliFaithfulness:
    def _make_scorer(self, fake_model):
        """Import and instantiate NliFaithfulness with a patched CrossEncoder loader."""
        from eval import nli_faithfulness
        import importlib

        # Replace the lazy import function.
        original = nli_faithfulness._load_cross_encoder
        nli_faithfulness._load_cross_encoder = lambda model, max_length, device: fake_model
        try:
            return nli_faithfulness.NliFaithfulness(model="fake/nli", device="cpu")
        finally:
            nli_faithfulness._load_cross_encoder = original

    def test_score_returns_mean_entailment(self):
        """Faithfulness score should be the mean entailment probability."""
        scorer = self._make_scorer(_FakeNliModel([
            [0.1, 0.8, 0.1],   # entailment=0.8
            [0.2, 0.7, 0.1],   # entailment=0.7
        ]))
        result = scorer.score(
            "The capital of France is Paris.",
            ["France is a country in Europe.", "Paris is the capital."],
        )
        assert result == pytest.approx(0.75), f"expected 0.75, got {result}"

    def test_score_empty_chunks_returns_zero(self):
        """No chunks → faithfulness 0."""
        scorer = self._make_scorer(MagicMock())
        assert scorer.score("answer", []) == 0.0

    def test_score_empty_answer_returns_zero(self):
        """Empty/whitespace answer → faithfulness 0."""
        scorer = self._make_scorer(MagicMock())
        assert scorer.score("  ", ["chunk1", "chunk2"]) == 0.0

    def test_score_model_predict_failure_falls_back_to_zero(self):
        """If predict() raises, score() returns 0.0 (not an exception)."""
        bad_model = MagicMock()
        bad_model.predict.side_effect = RuntimeError("GPU OOM")
        scorer = self._make_scorer(bad_model)
        assert scorer.score("answer", ["chunk"]) == 0.0

    def test_score_single_logit_positive(self):
        """Single positive logit → raw value."""
        scorer = self._make_scorer(_FakeNliModel([[2.5]]))
        result = scorer.score("answer", ["chunk"])
        assert result == pytest.approx(2.5), f"expected 2.5, got {result}"

    def test_score_single_logit_negative_clipped(self):
        """Negative logit → clipped to 0."""
        scorer = self._make_scorer(_FakeNliModel([[-1.0]]))
        result = scorer.score("answer", ["chunk"])
        assert result == 0.0, f"expected 0.0, got {result}"


class TestMakeNliFaithfulness:
    def test_disabled_returns_none(self):
        """When faithfulness.enabled=False in config, factory returns None."""
        from eval.nli_faithfulness import make_nli_faithfulness
        cfg = {"faithfulness": {"enabled": False}}
        assert make_nli_faithfulness(cfg) is None

    def test_no_faithfulness_section_returns_none(self):
        """No faithfulness section → None."""
        from eval.nli_faithfulness import make_nli_faithfulness
        assert make_nli_faithfulness({}) is None
        assert make_nli_faithfulness(None) is None

    def test_enabled_loads_scorer(self):
        """When faithfulness.enabled=True, factory returns a scorer."""
        from eval import nli_faithfulness
        original = nli_faithfulness._load_cross_encoder
        nli_faithfulness._load_cross_encoder = lambda *a, **kw: MagicMock()
        try:
            from eval.nli_faithfulness import make_nli_faithfulness
            cfg = {"faithfulness": {"enabled": True, "model": "fake/nli", "device": "cpu"}}
            scorer = make_nli_faithfulness(cfg)
            assert scorer is not None
        finally:
            nli_faithfulness._load_cross_encoder = original

    def test_load_failure_returns_none(self):
        """When CrossEncoder import fails, factory returns None."""
        from eval import nli_faithfulness
        original = nli_faithfulness._load_cross_encoder
        nli_faithfulness._load_cross_encoder = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("not found")
        )
        try:
            from eval.nli_faithfulness import make_nli_faithfulness
            cfg = {"faithfulness": {"enabled": True}}
            assert make_nli_faithfulness(cfg) is None
        finally:
            nli_faithfulness._load_cross_encoder = original
