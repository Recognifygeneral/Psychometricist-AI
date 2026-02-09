"""Factory helpers for creating workflow state payloads."""

from __future__ import annotations

from typing import Any

DEFAULT_MAX_TURNS = 10


def new_assessment_state(
    session_id: str,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict[str, Any]:
    """Return a fresh assessment state dict used by CLI and web entrypoints."""
    return {
        "session_id": session_id,
        "probes_used": [],
        "transcript": "",
        "turn_records": [],
        "turn_features": [],
        "scoring_results": {},
        "overall_score": 0.0,
        "classification": "",
        "confidence": 0.0,
        "facet_scores": [],
        "turn_count": 0,
        "max_turns": max_turns,
        "done": False,
    }
