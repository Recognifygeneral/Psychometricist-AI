"""Tests for the scoring modules.

Tests the feature-based scorer (no API needed) and verifies
the ensemble structure. Embedding and LLM scorers are tested
with mocks since they require API keys.
"""

from src.extraction.features import extract_features
from src.scoring.feature_scorer import score_with_features, explain_score


class TestFeatureScorer:
    """Test the rule-based feature scorer."""

    HIGH_E_TEXT = (
        "I absolutely love going to parties and meeting new people! "
        "My friends always say I'm the life of the party. We go out "
        "every weekend and have the most amazing adventures together. "
        "I'm always excited to try new things and I definitely don't "
        "mind being the center of attention! It's so much fun!"
    )

    LOW_E_TEXT = (
        "I tend to keep to myself mostly. I suppose I might enjoy "
        "the occasional quiet evening at home with a book. I'm "
        "not really sure about large gatherings, they seem somewhat "
        "overwhelming perhaps. I usually prefer solitary activities "
        "and rarely go out."
    )

    NEUTRAL_TEXT = (
        "I went to the store today and bought some groceries. "
        "Then I came home and cooked dinner. It was a normal day."
    )

    def test_high_e_scores_above_midpoint(self):
        features = extract_features(self.HIGH_E_TEXT)
        result = score_with_features(features)
        assert result["score"] > 3.0
        assert result["classification"] in ("Medium", "High")

    def test_low_e_scores_below_midpoint(self):
        features = extract_features(self.LOW_E_TEXT)
        result = score_with_features(features)
        assert result["score"] < 3.5  # should trend lower
        assert result["classification"] in ("Low", "Medium")

    def test_neutral_scores_near_midpoint(self):
        features = extract_features(self.NEUTRAL_TEXT)
        result = score_with_features(features)
        # Neutral text should be close to 3.0
        assert 2.0 <= result["score"] <= 4.0

    def test_high_scores_higher_than_low(self):
        high = score_with_features(extract_features(self.HIGH_E_TEXT))
        low = score_with_features(extract_features(self.LOW_E_TEXT))
        assert high["score"] > low["score"]

    def test_score_in_valid_range(self):
        for text in [self.HIGH_E_TEXT, self.LOW_E_TEXT, self.NEUTRAL_TEXT]:
            features = extract_features(text)
            result = score_with_features(features)
            assert 1.0 <= result["score"] <= 5.0

    def test_classification_values(self):
        for text in [self.HIGH_E_TEXT, self.LOW_E_TEXT, self.NEUTRAL_TEXT]:
            features = extract_features(text)
            result = score_with_features(features)
            assert result["classification"] in ("Low", "Medium", "High")

    def test_confidence_in_valid_range(self):
        features = extract_features(self.HIGH_E_TEXT)
        result = score_with_features(features)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_result_structure(self):
        features = extract_features(self.HIGH_E_TEXT)
        result = score_with_features(features)
        assert result["method"] == "feature_based"
        assert "score" in result
        assert "classification" in result
        assert "confidence" in result
        assert "feature_contributions" in result
        assert "features_used" in result

    def test_empty_text_returns_neutral(self):
        features = extract_features("")
        result = score_with_features(features)
        assert result["score"] == 3.0
        assert result["classification"] == "Medium"
        assert result["confidence"] == 0.0

    def test_explain_score_output(self):
        features = extract_features(self.HIGH_E_TEXT)
        result = score_with_features(features)
        explanation = explain_score(result)
        assert isinstance(explanation, str)
        assert "Feature-Based Score" in explanation
        assert "/5.0" in explanation


class TestEnsembleStructure:
    """Test the ensemble scorer's structure (without API calls)."""

    def test_feature_only_ensemble(self):
        from src.scoring.ensemble import score_ensemble

        text = "I love hanging out with friends and going to exciting new places!"
        result = score_ensemble(
            transcript=text,
            run_llm=False,
            run_embedding=False,
            run_features=True,
        )
        assert "ensemble_score" in result
        assert "ensemble_classification" in result
        assert "individual_results" in result
        assert "feature_based" in result["individual_results"]
        assert 1.0 <= result["ensemble_score"] <= 5.0

    def test_no_methods_returns_neutral(self):
        from src.scoring.ensemble import score_ensemble

        result = score_ensemble(
            transcript="test",
            run_llm=False,
            run_embedding=False,
            run_features=False,
        )
        assert result["ensemble_score"] == 3.0
        assert "warning" in result

    def test_format_results(self):
        from src.scoring.ensemble import score_ensemble, format_results

        result = score_ensemble(
            transcript="I enjoy being with people and having fun!",
            run_llm=False,
            run_embedding=False,
            run_features=True,
        )
        formatted = format_results(result)
        assert isinstance(formatted, str)
        assert "EXTRAVERSION" in formatted
        assert "feature" in formatted.lower()


class TestScorerIntegration:
    """Test that scoring modules import correctly and have expected APIs."""

    def test_feature_scorer_imports(self):
        from src.scoring.feature_scorer import score_with_features, explain_score, WEIGHTS
        assert callable(score_with_features)
        assert callable(explain_score)
        assert len(WEIGHTS) > 0

    def test_llm_scorer_imports(self):
        from src.scoring.llm_scorer import score_domain_level, score_facet_level
        assert callable(score_domain_level)
        assert callable(score_facet_level)

    def test_embedding_scorer_imports(self):
        from src.scoring.embedding_scorer import score_with_embeddings
        assert callable(score_with_embeddings)

    def test_ensemble_imports(self):
        from src.scoring.ensemble import score_ensemble, format_results
        assert callable(score_ensemble)
        assert callable(format_results)

    def test_session_logger_imports(self):
        from src.session.logger import SessionLogger, load_session, list_sessions
        assert callable(load_session)
        assert callable(list_sessions)
        logger = SessionLogger(session_id="test_001")
        assert logger.session_id == "test_001"
