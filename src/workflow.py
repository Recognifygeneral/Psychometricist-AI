"""LangGraph workflow — wires Interviewer and Scorer into a stateful graph.

Flow:
    START → router → interviewer → human_turn → update_state → router → …
                  ↘ scorer → END   (when done==True)

The human_turn node uses LangGraph's `interrupt()` to pause execution
and wait for real user input, which the CLI runner resumes via
`Command(resume=...)`.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.agents.interviewer import interviewer_node
from src.agents.scorer import scorer_node
from src.models.state import AssessmentState

# Ordered facet sequence for the MVP (sequential exploration)
FACET_ORDER = ["E1", "E2", "E3", "E4", "E5", "E6"]
TURNS_PER_FACET = 2
MAX_TURNS = len(FACET_ORDER) * TURNS_PER_FACET


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
    """Process the user's response: update transcript, manage facet rotation."""
    from langchain_core.messages import HumanMessage

    user_text: str = state.get("user_input", "")
    turn_count = state.get("turn_count", 0)
    current_facet = state.get("current_facet", "E1")
    explored_facets = state.get("explored_facets", [])
    transcript = state.get("transcript", "")

    # Accumulate transcript
    transcript += f"\n[Turn {turn_count + 1}] {user_text}"
    turn_count += 1

    # Facet rotation: advance every TURNS_PER_FACET turns
    facet_idx = FACET_ORDER.index(current_facet) if current_facet in FACET_ORDER else 0
    turns_on_current = turn_count - facet_idx * TURNS_PER_FACET

    new_explored: list[str] = []
    if turns_on_current >= TURNS_PER_FACET:
        new_explored = [current_facet]
        next_idx = facet_idx + 1
        if next_idx < len(FACET_ORDER):
            current_facet = FACET_ORDER[next_idx]
        else:
            # All facets covered
            return {
                "messages": [HumanMessage(content=user_text)],
                "transcript": transcript,
                "turn_count": turn_count,
                "current_facet": current_facet,
                "explored_facets": new_explored,
                "done": True,
            }

    return {
        "messages": [HumanMessage(content=user_text)],
        "transcript": transcript,
        "turn_count": turn_count,
        "current_facet": current_facet,
        "explored_facets": new_explored,
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
