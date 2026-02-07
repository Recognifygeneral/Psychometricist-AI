"""Interviewer agent — conducts the conversational psychometric interview.

Revised design (aligned with project spec):
  - Flat probe pool — draws from ALL probes regardless of facet
  - No facet routing or facet-aware state management
  - Fixed session length; probe selection is sequential from pool
  - Warm, open-ended, behaviorally-focused questioning

The Interviewer's job is to elicit natural, personality-relevant
conversation without the user realizing they're being assessed.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.graph_client import get_all_probes
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


def _select_probe(probes_used: list[str]) -> dict | None:
    """Select the next probe from the flat pool, skipping already-used ones.

    Returns None if all probes are exhausted.
    """
    all_probes = get_all_probes()
    for probe in all_probes:
        if probe["id"] not in probes_used:
            return probe
    return None


def interviewer_node(state: AssessmentState) -> dict:
    """LangGraph node: generate the next interviewer question.

    Draws from a flat pool of probes (no facet routing).
    Returns a dict of state updates (messages, probes_used, etc.).
    """
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", 10)
    probes_used = state.get("probes_used", [])

    # ── Select probe from flat pool ───────────────────────────────────
    probe = _select_probe(probes_used)

    # Fallback if pool exhausted
    if probe is None:
        probe = {
            "id": f"fallback_{turn_count}",
            "text": "Tell me about something that's been on your mind lately.",
            "target_behavior": "General personality expression",
        }

    # ── Build prompt ──────────────────────────────────────────────────
    is_opening = turn_count == 0

    if is_opening:
        system_text = OPENING_PROMPT.format(probe_text=probe["text"])
    else:
        system_text = SYSTEM_PROMPT.format(
            probe_text=probe["text"],
            target_behavior=probe["target_behavior"],
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
