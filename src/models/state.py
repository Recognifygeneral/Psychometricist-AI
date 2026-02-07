"""Shared state definitions for the LangGraph psychometric assessment workflow."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import MessagesState


class FacetScore(TypedDict):
    """Score for a single Extraversion facet."""

    facet_code: str
    facet_name: str
    score: float  # 1.0 – 5.0
    evidence: str  # brief justification from the Scorer


class AssessmentState(MessagesState):
    """Full shared state for the psychometric interview workflow.

    Extends MessagesState (which provides `messages: list[AnyMessage]`
    with the `add_messages` reducer) with psychometric-specific fields.
    """

    # --- Navigator / routing state ---
    current_facet: str  # facet code currently being explored, e.g. "E1"
    explored_facets: Annotated[list[str], operator.add]  # facet codes finished
    probes_used: Annotated[list[str], operator.add]  # probe IDs already asked

    # --- Human input (set by interrupt/resume) ---
    user_input: str  # latest raw user input from interrupt()

    # --- Transcript for scoring ---
    transcript: str  # user-only plain-text transcript (accumulated, overwrite)

    # --- Scoring output ---
    facet_scores: list[FacetScore]  # one per facet, filled by Scorer
    overall_score: float  # mean of facet scores
    classification: str  # "Low", "Medium", or "High"

    # --- Control flow ---
    turn_count: int  # how many interviewer–user exchanges
    max_turns: int  # upper limit before forcing scoring
    done: bool  # True when all facets explored or max_turns hit
