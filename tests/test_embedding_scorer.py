"""Tests for the embedding-based scorer.

All OpenAI embedding calls are mocked — no API key required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.scoring.embedding_scorer import (
    _cosine_similarity,
    explain_score,
    score_with_embeddings,
)

# ── Cosine similarity unit tests ─────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert _cosine_similarity(a, b) == 0.0


# ── score_with_embeddings ────────────────────────────────────────────────


def _build_mock_embeddings(
    user_vec: list[float],
    high_vecs: list[list[float]],
    low_vecs: list[list[float]],
) -> MagicMock:
    """Build a mock embeddings model with controlled vectors."""
    mock_model = MagicMock()
    mock_model.embed_query.return_value = user_vec
    mock_model.embed_documents.side_effect = [high_vecs, low_vecs]
    return mock_model


class TestScoreWithEmbeddings:
    def test_short_transcript_returns_neutral(self):
        result = score_with_embeddings("hello world", min_words=15)
        assert result["score"] == 3.0
        assert result["confidence"] == 0.0
        assert "warning" in result

    def test_happy_path_high_e(self):
        """User vector closer to high-E references → score > 3."""
        # Realistic: all-positive components so cos_sim > 0
        user = [0.9, 0.1, 0.3]
        highs = [[0.9, 0.1, 0.3]] * 7   # very similar to user
        lows = [[0.1, 0.9, 0.3]] * 7    # dissimilar to user
        mock_emb = _build_mock_embeddings(user, highs, lows)

        transcript = " ".join(["word"] * 20)  # 20 words to pass min_words
        with patch(
            "src.scoring.embedding_scorer._get_embeddings_model",
            return_value=mock_emb,
        ):
            result = score_with_embeddings(transcript, min_words=15)

        assert result["method"] == "embedding"
        assert result["score"] > 3.0
        assert result["high_similarity"] > result["low_similarity"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_happy_path_low_e(self):
        """User vector closer to low-E references → score < 3."""
        user = [0.1, 0.9, 0.3]
        highs = [[0.9, 0.1, 0.3]] * 7   # dissimilar to user
        lows = [[0.1, 0.9, 0.3]] * 7    # very similar to user
        mock_emb = _build_mock_embeddings(user, highs, lows)

        transcript = " ".join(["word"] * 20)
        with patch(
            "src.scoring.embedding_scorer._get_embeddings_model",
            return_value=mock_emb,
        ):
            result = score_with_embeddings(transcript, min_words=15)

        assert result["score"] < 3.0
        assert result["low_similarity"] > result["high_similarity"]

    def test_api_failure_returns_fallback(self):
        mock_emb = MagicMock()
        mock_emb.embed_query.side_effect = RuntimeError("API down")

        transcript = " ".join(["word"] * 20)
        with patch(
            "src.scoring.embedding_scorer._get_embeddings_model",
            return_value=mock_emb,
        ):
            result = score_with_embeddings(transcript, min_words=15)

        assert result["score"] == 3.0
        assert "error" in result

    def test_score_clamped(self):
        """Extreme vectors shouldn't push score outside [1, 5]."""
        user = [1.0, 0.0, 0.0]
        highs = [[1.0, 0.0, 0.0]] * 7  # perfect alignment
        lows = [[0.0, 1.0, 0.0]] * 7   # orthogonal
        mock_emb = _build_mock_embeddings(user, highs, lows)

        transcript = " ".join(["word"] * 20)
        with patch(
            "src.scoring.embedding_scorer._get_embeddings_model",
            return_value=mock_emb,
        ):
            result = score_with_embeddings(transcript, min_words=15)

        assert 1.0 <= result["score"] <= 5.0

    def test_result_structure(self):
        user = [0.5, 0.5, 0.5]
        highs = [[0.5, 0.5, 0.5]] * 7
        lows = [[0.5, 0.5, 0.5]] * 7
        mock_emb = _build_mock_embeddings(user, highs, lows)

        transcript = " ".join(["word"] * 20)
        with patch(
            "src.scoring.embedding_scorer._get_embeddings_model",
            return_value=mock_emb,
        ):
            result = score_with_embeddings(transcript, min_words=15)

        assert "method" in result
        assert "score" in result
        assert "classification" in result
        assert "confidence" in result
        assert "high_similarity" in result
        assert "low_similarity" in result


# ── explain_score ─────────────────────────────────────────────────────────


class TestExplainScore:
    def test_normal_result(self):
        result = {
            "method": "embedding",
            "score": 3.8,
            "classification": "High",
            "confidence": 0.6,
            "high_similarity": 0.85,
            "low_similarity": 0.72,
            "balance": 0.08,
        }
        text = explain_score(result)
        assert "3.80" in text
        assert "High" in text

    def test_error_result(self):
        result = {"error": "API failed"}
        text = explain_score(result)
        assert "FAILED" in text

    def test_warning_result(self):
        result = {"warning": "Too short"}
        text = explain_score(result)
        assert "Too short" in text
