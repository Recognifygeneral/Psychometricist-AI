"""LLM-based scorers for Extraversion domain and facet ratings."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.settings import LLM_MODEL_NAME, NEUTRAL_SCORE, classify_extraversion

logger = logging.getLogger(__name__)

DOMAIN_SCORER_PROMPT = """\
You are an expert personality psychologist scoring an interview transcript
for EXTRAVERSION.

Return valid JSON only:
{
  "score": 3.5,
  "classification": "Medium",
  "confidence": 0.7,
  "evidence": "Brief evidence summary."
}

Rules:
- score is in [1.0, 5.0]
- confidence is in [0.0, 1.0]
- use behavioral evidence from the transcript
"""

FACET_SCORER_PROMPT = """\
You are an expert personality psychologist. Score each Extraversion facet
on a 1-5 scale.

Return valid JSON only:
{
  "facet_scores": [
    {"facet_code": "E1", "facet_name": "Friendliness", "score": 3.5, "evidence": "..."},
    {"facet_code": "E2", "facet_name": "Gregariousness", "score": 3.0, "evidence": "..."},
    {"facet_code": "E3", "facet_name": "Assertiveness", "score": 3.0, "evidence": "..."},
    {"facet_code": "E4", "facet_name": "Activity Level", "score": 3.0, "evidence": "..."},
    {"facet_code": "E5", "facet_name": "Excitement-Seeking", "score": 3.0, "evidence": "..."},
    {"facet_code": "E6", "facet_name": "Cheerfulness", "score": 3.0, "evidence": "..."}
  ]
}
"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0)


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse JSON from model output, stripping markdown fences if needed."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def _response_text(content: Any) -> str:
    """Normalize LangChain message content into a text string."""
    if isinstance(content, str):
        return content
    return json.dumps(content)


def score_domain_level(transcript: str) -> dict[str, Any]:
    """Score overall Extraversion from user transcript text."""
    if not transcript.strip():
        return {
            "method": "llm_domain",
            "score": NEUTRAL_SCORE,
            "classification": classify_extraversion(NEUTRAL_SCORE),
            "confidence": 0.0,
            "evidence": "Empty transcript - no evidence to score.",
        }

    try:
        llm = _get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=DOMAIN_SCORER_PROMPT),
                HumanMessage(
                    content=f"INTERVIEW TRANSCRIPT (user responses only):\n\n{transcript}"
                ),
            ]
        )

        parsed = _parse_json(_response_text(response.content))
        score = max(1.0, min(5.0, float(parsed["score"])))
        confidence = round(float(parsed.get("confidence", 0.5)), 3)
        confidence = max(0.0, min(1.0, confidence))

        return {
            "method": "llm_domain",
            "score": round(score, 2),
            "classification": classify_extraversion(score),
            "confidence": confidence,
            "evidence": parsed.get("evidence", ""),
        }

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning("LLM domain scorer parse error: %s", e)
        return {
            "method": "llm_domain",
            "score": NEUTRAL_SCORE,
            "classification": classify_extraversion(NEUTRAL_SCORE),
            "confidence": 0.0,
            "evidence": "LLM parse error. Defaulting to neutral.",
            "error": f"LLM parse error: {e}",
        }

    except Exception as e:
        logger.warning("LLM domain scorer failed: %s", e)
        return {
            "method": "llm_domain",
            "score": NEUTRAL_SCORE,
            "classification": classify_extraversion(NEUTRAL_SCORE),
            "confidence": 0.0,
            "error": f"LLM scorer failed: {str(e)}",
        }


def score_facet_level(transcript: str) -> dict[str, Any]:
    """Score each Extraversion facet (secondary analysis mode)."""
    if not transcript.strip():
        return {
            "method": "llm_facet",
            "facet_scores": [],
            "overall_score": NEUTRAL_SCORE,
            "classification": classify_extraversion(NEUTRAL_SCORE),
        }

    try:
        llm = _get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=FACET_SCORER_PROMPT),
                HumanMessage(
                    content=f"INTERVIEW TRANSCRIPT (user responses only):\n\n{transcript}"
                ),
            ]
        )

        parsed = _parse_json(_response_text(response.content))
        facet_scores = parsed.get("facet_scores", [])
        scores = [float(fs["score"]) for fs in facet_scores if "score" in fs]
        overall = sum(scores) / len(scores) if scores else NEUTRAL_SCORE

        return {
            "method": "llm_facet",
            "facet_scores": facet_scores,
            "overall_score": round(overall, 2),
            "classification": classify_extraversion(overall),
        }

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning("LLM facet scorer parse error: %s", e)
        return {
            "method": "llm_facet",
            "facet_scores": [],
            "overall_score": NEUTRAL_SCORE,
            "classification": classify_extraversion(NEUTRAL_SCORE),
            "error": f"LLM facet parse error: {str(e)}",
        }

    except Exception as e:
        logger.warning("LLM facet scorer failed: %s", e)
        return {
            "method": "llm_facet",
            "facet_scores": [],
            "overall_score": NEUTRAL_SCORE,
            "classification": classify_extraversion(NEUTRAL_SCORE),
            "error": str(e),
        }


def explain_score(result: dict[str, Any]) -> str:
    """Return a human-readable explanation."""
    method = result.get("method", "llm")
    if method == "llm_facet":
        lines = [
            f"LLM Facet-Level Score: {result['overall_score']:.2f}/5.0 -> {result['classification']}"
        ]
        for fs in result.get("facet_scores", []):
            lines.append(
                f"  {fs['facet_code']} {fs['facet_name']}: {float(fs['score']):.1f} - {fs['evidence']}"
            )
        return "\n".join(lines)

    return (
        f"LLM Domain-Level Score: {result['score']:.2f}/5.0 -> {result['classification']}\n"
        f"Confidence: {result['confidence']:.1%}\n"
        f"Evidence: {result.get('evidence', 'N/A')}"
    )
