"""Interviewer agent — conducts the conversational psychometric interview.

The Interviewer receives the current assessment state (which facet to
explore, which probes remain) and produces a warm, open-ended question
that naturally elicits personality-relevant behavior without the user
realizing they are being assessed.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.graph_client import get_unused_probe, get_facets_for_trait
from src.models.state import AssessmentState

SYSTEM_PROMPT = """\
You are a warm, curious, and empathetic conversational interviewer.
Your goal is to have a natural, engaging conversation that helps you
understand the person you are talking to.

RULES — follow strictly:
1. NEVER ask yes/no or closed-ended questions.
2. NEVER mention personality traits, psychology, tests, or assessments.
3. Ask ONE open-ended question at a time.
4. Use the probe below as *inspiration* — rephrase it in your own words
   so it feels natural and conversational, not like a survey.
5. If continuing the conversation, briefly acknowledge what the user
   said before transitioning to the next question.
6. Keep your replies concise (2-4 sentences max).
7. Be genuinely interested — use follow-up cues when appropriate.

CURRENT PROBE (for inspiration only — do NOT read it verbatim):
{probe_text}

TARGET BEHAVIOR to elicit:
{target_behavior}

FACET BEING EXPLORED: {facet_name}
TURN: {turn} of {max_turns}
"""

OPENING_PROMPT = """\
You are a warm, curious, and empathetic conversational interviewer.
This is the very first message of the conversation. Greet the user
warmly, introduce yourself briefly as someone who loves getting to know
people through conversation, and ask your first open-ended question.

Use this probe as inspiration (do NOT read it verbatim):
{probe_text}

Keep it to 2-3 sentences. Be natural. Do NOT mention psychology or assessments.
"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-5.2", temperature=0.7)


def interviewer_node(state: AssessmentState) -> dict:
    """LangGraph node: generate the next interviewer question.

    Returns a dict of state updates (messages, probes_used, etc.).
    """
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", 12)
    current_facet = state.get("current_facet", "E1")
    probes_used = state.get("probes_used", [])

    # ── Fetch probe from graph (Neo4j or local JSON fallback) ─────────
    probe = get_unused_probe(current_facet, probes_used)
    facets = get_facets_for_trait()
    facet_map = {f["code"]: f["name"] for f in facets}

    facet_name = facet_map.get(current_facet, current_facet)

    # Fallback if no more probes for this facet
    if probe is None:
        probe = {
            "id": f"fallback_{current_facet}",
            "text": f"Tell me more about how you experience {facet_name.lower()} in your daily life.",
            "target_behavior": f"General {facet_name.lower()} behavior",
        }

    # ── Build prompt ──────────────────────────────────────────────────
    is_opening = turn_count == 0

    if is_opening:
        system_text = OPENING_PROMPT.format(probe_text=probe["text"])
    else:
        system_text = SYSTEM_PROMPT.format(
            probe_text=probe["text"],
            target_behavior=probe["target_behavior"],
            facet_name=facet_name,
            turn=turn_count + 1,
            max_turns=max_turns,
        )

    # Build message list: system + conversation history (last 6 msgs)
    history = state.get("messages", [])
    recent = history[-6:] if len(history) > 6 else history
    prompt_messages = [SystemMessage(content=system_text)] + list(recent)

    # ── Call LLM ──────────────────────────────────────────────────────
    llm = _get_llm()
    response: AIMessage = llm.invoke(prompt_messages)

    # ── Return state updates ──────────────────────────────────────────
    return {
        "messages": [response],
        "probes_used": [probe["id"]],
    }
