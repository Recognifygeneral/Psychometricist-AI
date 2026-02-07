"""LangGraph workflow — wires Interviewer and Scorer into a stateful graph.

Revised design (aligned with project spec):
  - Fixed number of turns (NOT adaptive)
  - Flat probe pool (no facet rotation for MVP)
  - Per-turn feature extraction
  - Multi-method ensemble scoring

Flow:
    START → router → interviewer → human_turn → update_state → router → …
                  ↘ scorer → END   (when turn_count >= max_turns)

The human_turn node uses LangGraph's `interrupt()` to pause execution
and wait for real user input, which the CLI runner resumes via
`Command(resume=...)`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.agents.interviewer import interviewer_node
from src.agents.scorer import scorer_node
from src.extraction.features import extract_features
from src.models.state import AssessmentState

# Fixed session length — NOT adaptive (spec requirement)
MAX_TURNS = 10


# ── Graph nodes ───────────────────────────────────────────────────────────


def router(state: AssessmentState) -> Command:
    """Decide whether to continue interviewing or move to scoring."""
    done = state.get("done", False)
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", MAX_TURNS)

    if done or turn_count >= max_turns:
        return Command(update={"done": True}, goto="scorer")
    return Command(goto="interviewer")


def human_turn(state: AssessmentState) -> dict:
    """Pause execution and wait for user input via interrupt()."""
    user_input: str = interrupt("Waiting for user response…")
    return {"user_input": user_input}


def update_state(state: AssessmentState) -> dict:
    """Process the user's response: update transcript, extract features.

    This node runs AFTER each human turn and BEFORE routing back.
    It extracts linguistic features from the user's response for
    per-turn analysis and accumulates the transcript.
    """
    from langchain_core.messages import HumanMessage

    user_text: str = state.get("user_input", "")
    turn_count = state.get("turn_count", 0)
    transcript = state.get("transcript", "")
    turn_records = state.get("turn_records", [])
    turn_features_list = state.get("turn_features", [])

    # Accumulate transcript
    transcript += f"\n[Turn {turn_count + 1}] {user_text}"
    turn_count += 1

    # Extract linguistic features for this turn
    features = extract_features(user_text)
    features_dict = features.to_dict()

    # Get the last AI message for the turn record
    messages = state.get("messages", [])
    last_ai_text = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            last_ai_text = msg.content
            break

    # Build turn record
    turn_record = {
        "turn_number": turn_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ai_message": last_ai_text,
        "user_message": user_text,
        "features": features_dict,
    }
    turn_records = turn_records + [turn_record]
    turn_features_list = turn_features_list + [features_dict]

    # Check if we've hit max turns
    max_turns = state.get("max_turns", MAX_TURNS)
    done = turn_count >= max_turns

    return {
        "messages": [HumanMessage(content=user_text)],
        "transcript": transcript,
        "turn_count": turn_count,
        "turn_records": turn_records,
        "turn_features": turn_features_list,
        "done": done,
    }


# ── Build the graph ───────────────────────────────────────────────────────


def build_graph():
    """Construct and compile the assessment StateGraph."""
    graph = StateGraph(AssessmentState)

    graph.add_node("router", router)
    graph.add_node("interviewer", interviewer_node)
    graph.add_node("human_turn", human_turn)
    graph.add_node("update_state", update_state)
    graph.add_node("scorer", scorer_node)

    graph.add_edge(START, "router")
    # router uses Command to go to "interviewer" or "scorer"
    graph.add_edge("interviewer", "human_turn")
    graph.add_edge("human_turn", "update_state")
    graph.add_edge("update_state", "router")
    graph.add_edge("scorer", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
