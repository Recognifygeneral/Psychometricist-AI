"""Tests for LLM-based scorers (domain + facet level).

All OpenAI calls are mocked — no API key required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.scoring.llm_scorer import (
    _parse_json,
    _response_text,
    explain_score,
    score_domain_level,
    score_facet_level,
)

# ── Helpers ───────────────────────────────────────────────────────────────

VALID_DOMAIN_JSON = json.dumps({
    "score": 4.2,
    "classification": "High",
    "confidence": 0.85,
    "evidence": "The speaker uses enthusiastic language and social references.",
})

VALID_FACET_JSON = json.dumps({
    "facet_scores": [
        {"facet_code": "E1", "facet_name": "Friendliness", "score": 4.0, "evidence": "Warm."},
        {"facet_code": "E2", "facet_name": "Gregariousness", "score": 3.5, "evidence": "Social."},
        {"facet_code": "E3", "facet_name": "Assertiveness", "score": 3.0, "evidence": "Moderate."},
        {"facet_code": "E4", "facet_name": "Activity Level", "score": 3.0, "evidence": "Average."},
        {"facet_code": "E5", "facet_name": "Excitement-Seeking", "score": 4.0, "evidence": "High."},
        {"facet_code": "E6", "facet_name": "Cheerfulness", "score": 4.5, "evidence": "Positive."},
    ]
})


def _mock_llm_response(content: str) -> MagicMock:
    """Build a mock ChatOpenAI that returns a single AIMessage."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=content)
    return mock_llm


# ── _parse_json ───────────────────────────────────────────────────────────


class TestParseJson:
    def test_plain_json(self):
        result = _parse_json('{"score": 3.0}')
        assert result["score"] == 3.0

    def test_markdown_fenced_json(self):
        raw = '```json\n{"score": 3.0}\n```'
        result = _parse_json(raw)
        assert result["score"] == 3.0

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError, match="Expecting value"):
            _parse_json("not json at all")


class TestResponseText:
    def test_string_passthrough(self):
        assert _response_text("hello") == "hello"

    def test_dict_serialized(self):
        result = _response_text({"key": "val"})
        assert '"key"' in result


# ── score_domain_level ────────────────────────────────────────────────────


class TestScoreDomainLevel:
    def test_happy_path(self):
        mock_llm = _mock_llm_response(VALID_DOMAIN_JSON)
        with patch("src.scoring.llm_scorer.get_chat_llm", return_value=mock_llm):
            result = score_domain_level("I love hanging out with friends.")

        assert result["method"] == "llm_domain"
        assert result["score"] == 4.2
        assert result["classification"] == "High"
        assert 0.0 <= result["confidence"] <= 1.0
        assert "evidence" in result

    def test_empty_transcript_returns_neutral(self):
        result = score_domain_level("")
        assert result["score"] == 3.0
        assert result["confidence"] == 0.0

    def test_whitespace_transcript_returns_neutral(self):
        result = score_domain_level("   \n  ")
        assert result["score"] == 3.0

    def test_parse_error_returns_neutral(self):
        mock_llm = _mock_llm_response("This is not JSON")
        with patch("src.scoring.llm_scorer.get_chat_llm", return_value=mock_llm):
            result = score_domain_level("Some transcript text.")

        assert result["score"] == 3.0
        assert "error" in result

    def test_api_failure_returns_neutral(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API down")
        with patch("src.scoring.llm_scorer.get_chat_llm", return_value=mock_llm):
            result = score_domain_level("Some transcript text.")

        assert result["score"] == 3.0
        assert "error" in result

    def test_score_clamped_to_valid_range(self):
        """Scores outside [1,5] should be clamped."""
        out_of_range = json.dumps({
            "score": 9.0, "classification": "High",
            "confidence": 0.5, "evidence": "Very extraverted.",
        })
        mock_llm = _mock_llm_response(out_of_range)
        with patch("src.scoring.llm_scorer.get_chat_llm", return_value=mock_llm):
            result = score_domain_level("Very social person.")

        assert result["score"] == 5.0

    def test_confidence_clamped(self):
        bad_conf = json.dumps({
            "score": 3.0, "classification": "Medium",
            "confidence": 2.5, "evidence": "Moderate.",
        })
        mock_llm = _mock_llm_response(bad_conf)
        with patch("src.scoring.llm_scorer.get_chat_llm", return_value=mock_llm):
            result = score_domain_level("Moderate person.")

        assert result["confidence"] <= 1.0


# ── score_facet_level ─────────────────────────────────────────────────────


class TestScoreFacetLevel:
    def test_happy_path(self):
        mock_llm = _mock_llm_response(VALID_FACET_JSON)
        with patch("src.scoring.llm_scorer.get_chat_llm", return_value=mock_llm):
            result = score_facet_level("I enjoy social activities.")

        assert result["method"] == "llm_facet"
        assert len(result["facet_scores"]) == 6
        assert 1.0 <= result["overall_score"] <= 5.0

    def test_empty_transcript(self):
        result = score_facet_level("")
        assert result["facet_scores"] == []
        assert result["overall_score"] == 3.0

    def test_parse_error(self):
        mock_llm = _mock_llm_response("Bad output")
        with patch("src.scoring.llm_scorer.get_chat_llm", return_value=mock_llm):
            result = score_facet_level("Some text.")

        assert "error" in result
        assert result["overall_score"] == 3.0

    def test_api_failure(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API error")
        with patch("src.scoring.llm_scorer.get_chat_llm", return_value=mock_llm):
            result = score_facet_level("Some text.")

        assert "error" in result


# ── explain_score ─────────────────────────────────────────────────────────


class TestExplainScore:
    def test_domain_explanation(self):
        result = {
            "method": "llm_domain",
            "score": 3.5,
            "classification": "Medium",
            "confidence": 0.7,
            "evidence": "Mixed signals.",
        }
        text = explain_score(result)
        assert "3.50" in text
        assert "Medium" in text

    def test_facet_explanation(self):
        result = {
            "method": "llm_facet",
            "overall_score": 3.5,
            "classification": "Medium",
            "facet_scores": [
                {"facet_code": "E1", "facet_name": "Friendliness",
                 "score": 4.0, "evidence": "Warm."},
            ],
        }
        text = explain_score(result)
        assert "E1" in text
        assert "Friendliness" in text
