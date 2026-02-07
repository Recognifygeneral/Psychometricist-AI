"""Shared state definitions for the LangGraph psychometric assessment workflow.

Design philosophy (from revised project spec):
  "This is a scientific experiment first, software system second."

The state is deliberately flat and simple for the MVP:
  - No facet-level routing (just a pool of probes)
  - Primary output is a domain-level classification (Low/Medium/High)
  - Per-turn feature extraction stored for analysis
  - Multi-method scoring results stored for comparison
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import MessagesState


class FacetScore(TypedDict):
    """Score for a single Extraversion facet (secondary / optional)."""

    facet_code: str
    facet_name: str
    score: float  # 1.0 – 5.0
    evidence: str  # brief justification from the Scorer


class TurnRecord(TypedDict, total=False):
    """Record of a single interview turn for session logging."""

    turn_number: int
    timestamp: str  # ISO 8601
    probe_id: str
    ai_message: str
    user_message: str
    features: dict[str, Any]  # extracted linguistic features


class AssessmentState(MessagesState):
    """Full shared state for the psychometric interview workflow.

    Extends MessagesState (which provides `messages: list[AnyMessage]`
    with the `add_messages` reducer) with psychometric-specific fields.
    """

    # --- Session identity ---
    session_id: str  # unique session identifier

    # --- Probe tracking ---
    probes_used: Annotated[list[str], operator.add]  # probe IDs already asked

    # --- Human input (set by interrupt/resume) ---
    user_input: str  # latest raw user input from interrupt()

    # --- Transcript for scoring ---
    transcript: str  # user-only plain-text transcript (accumulated, overwrite)

    # --- Per-turn data (for session logging & analysis) ---
    turn_records: list[TurnRecord]  # accumulated turn-by-turn data
    turn_features: list[dict]  # extracted features per turn (overwrite)

    # --- Scoring output (multi-method) ---
    scoring_results: dict  # full ensemble output (all methods)
    overall_score: float  # ensemble score (1–5)
    classification: str  # "Low", "Medium", or "High" (ensemble)
    confidence: float  # ensemble confidence (0–1)

    # --- Legacy: facet-level scores (optional secondary output) ---
    facet_scores: list[FacetScore]  # one per facet, from LLM facet scorer

    # --- Control flow ---
    turn_count: int  # how many interviewer–user exchanges
    max_turns: int  # upper limit before forcing scoring (fixed, not adaptive)
    done: bool  # True when max_turns hit
