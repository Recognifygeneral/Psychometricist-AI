"""Scorer agent that runs ensemble scoring and persists session outputs."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from src.extraction.features import extract_features
from src.models.state import AssessmentState
from src.scoring.ensemble import format_results, score_ensemble
from src.session.logger import SessionLogger
from src.settings import NEUTRAL_SCORE, classify_extraversion

logger = logging.getLogger(__name__)


def _incomplete_assessment_response() -> dict[str, Any]:
    """Return a neutral fallback payload when no transcript is available."""
    return {
        "scoring_results": {},
        "overall_score": NEUTRAL_SCORE,
        "classification": classify_extraversion(NEUTRAL_SCORE),
        "confidence": 0.0,
        "messages": [
            AIMessage(
                content="I wasn't able to gather enough information to "
                "produce a reliable score. The assessment is incomplete."
            )
        ],
    }


def _extract_facet_scores(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract optional facet-level details from LLM secondary output."""
    facet_data = results.get("individual_results", {}).get("llm_facet")
    if facet_data and facet_data.get("facet_scores"):
        return facet_data["facet_scores"]
    return []


def _save_session_log(state: AssessmentState, results: dict[str, Any]) -> str:
    """Persist session log and return a status line for the summary output."""
    session_id = state.get("session_id", "unknown")
    transcript = state.get("transcript", "")
    features = extract_features(transcript)

    try:
        session_logger = SessionLogger(session_id=session_id)
        for record in state.get("turn_records", []):
            session_logger.turns.append(dict(record))
        session_logger.set_metadata("transcript_features", features.to_dict())
        session_logger.set_metadata("transcript_word_count", features.word_count)
        session_logger.log_scoring(results)
        log_path = session_logger.save()
        return f"Session log saved -> {log_path.name}"
    except Exception as e:
        logger.warning("Session log failed for session %s: %s", session_id, e)
        return f"Session log failed: {e}"


def scorer_node(state: AssessmentState) -> dict[str, Any]:
    """LangGraph node: run ensemble scoring, save session output, and respond."""
    transcript = state.get("transcript", "")
    if not transcript.strip():
        return _incomplete_assessment_response()

    features = extract_features(transcript)
    results = score_ensemble(
        transcript=transcript,
        features=features,
        run_llm=True,
        run_embedding=True,
        run_features=True,
        run_facet_level=True,  # secondary: per-facet for analysis
    )
    summary_text = format_results(results)
    summary_text += f"\n\n{_save_session_log(state, results)}"

    return {
        "scoring_results": results,
        "overall_score": results.get("ensemble_score", NEUTRAL_SCORE),
        "classification": results.get(
            "ensemble_classification",
            classify_extraversion(NEUTRAL_SCORE),
        ),
        "confidence": results.get("ensemble_confidence", 0.0),
        "facet_scores": _extract_facet_scores(results),
        "messages": [AIMessage(content=summary_text)],
    }
