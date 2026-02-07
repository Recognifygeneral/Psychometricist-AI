"""Scorer agent — runs multi-method ensemble scoring on the interview transcript.

Revised design (aligned with project spec):
  - Primary output: domain-level classification (Low/Medium/High)
  - Runs THREE scoring methods: feature-based, embedding, LLM
  - Combines via confidence-weighted ensemble
  - Saves structured session log
  - Per-facet LLM scoring kept as optional secondary output

Scientific framing: by comparing scoring methods we assess convergent
validity BETWEEN methods (a core research question).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.extraction.features import extract_features
from src.models.state import AssessmentState
from src.scoring.ensemble import score_ensemble, format_results
from src.session.logger import SessionLogger


def scorer_node(state: AssessmentState) -> dict:
    """LangGraph node: run ensemble scoring and produce final results.

    Steps:
      1. Extract features from full transcript
      2. Run all scoring methods (feature, embedding, LLM)
      3. Combine into ensemble score
      4. Save session log
      5. Return state updates

    Returns state updates with scoring_results, overall_score,
    classification, confidence, and summary message.
    """
    transcript = state.get("transcript", "")

    if not transcript.strip():
        return {
            "scoring_results": {},
            "overall_score": 3.0,
            "classification": "Medium",
            "confidence": 0.0,
            "messages": [
                AIMessage(
                    content="I wasn't able to gather enough information to "
                    "produce a reliable score. The assessment is incomplete."
                )
            ],
        }

    # ── 1. Extract features from full transcript ──────────────────────
    features = extract_features(transcript)

    # ── 2. Run ensemble scoring ───────────────────────────────────────
    results = score_ensemble(
        transcript=transcript,
        features=features,
        run_llm=True,
        run_embedding=True,
        run_features=True,
        run_facet_level=True,  # secondary: per-facet for analysis
    )

    # ── 3. Format results for display ─────────────────────────────────
    summary_text = format_results(results)

    # ── 4. Extract facet scores if available ──────────────────────────
    facet_scores = []
    facet_data = results.get("individual_results", {}).get("llm_facet")
    if facet_data and facet_data.get("facet_scores"):
        facet_scores = facet_data["facet_scores"]

    # ── 5. Save session log ───────────────────────────────────────────
    session_id = state.get("session_id", "unknown")
    try:
        logger = SessionLogger(session_id=session_id)

        # Reconstruct turns from turn_records
        for record in state.get("turn_records", []):
            logger.turns.append(record)

        # Attach full-transcript features
        logger.set_metadata("transcript_features", features.to_dict())
        logger.set_metadata("transcript_word_count", features.word_count)
        logger.log_scoring(results)

        log_path = logger.save()
        summary_text += f"\n\n📄 Session log saved → {log_path.name}"
    except Exception as e:
        summary_text += f"\n\n⚠ Session log failed: {e}"

    # ── 6. Return state updates ───────────────────────────────────────
    return {
        "scoring_results": results,
        "overall_score": results.get("ensemble_score", 3.0),
        "classification": results.get("ensemble_classification", "Medium"),
        "confidence": results.get("ensemble_confidence", 0.0),
        "facet_scores": facet_scores,
        "messages": [AIMessage(content=summary_text)],
    }
