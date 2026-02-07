"""Scorer agent — evaluates the interview transcript and produces psychometric scores.

The Scorer receives the accumulated transcript (user messages only) and
uses GPT to rate the user on each Extraversion facet, then computes an
overall domain score and a classification (Low / Medium / High).
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.graph.graph_client import get_all_data_for_scoring
from src.models.state import AssessmentState, FacetScore

SCORER_SYSTEM_PROMPT = """\
You are an expert psychometric scorer. You will receive:
1. A description of the Extraversion trait and its six facets.
2. The full transcript of a conversational interview (user responses only).

Your task: rate the user on EACH of the six Extraversion facets on a
1–5 scale, where:
  1 = Very Low (strong introversion signals)
  2 = Low
  3 = Average / Neutral
  4 = High
  5 = Very High (strong extraversion signals)

For each facet, cite specific evidence from the transcript.

FACET DEFINITIONS:
{facet_definitions}

SCORING GUIDELINES:
- Base your ratings on behavioral evidence in the transcript, not on
  what the user explicitly claims about themselves.
- Consider: enthusiasm, social references, assertive language,
  activity descriptions, excitement-seeking anecdotes, positive emotion
  expressions.
- Absence of evidence for a facet should default to 3 (neutral), NOT 1.
- Be calibrated: most people score 2.5–3.5; reserve 1 and 5 for
  extreme and clear evidence.

RESPOND WITH VALID JSON ONLY — no markdown, no commentary:
{{
  "facet_scores": [
    {{"facet_code": "E1", "facet_name": "Friendliness", "score": 3.5, "evidence": "..."}},
    ...
  ]
}}
"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-5.2", temperature=0.0)


def _build_facet_definitions(scoring_data: dict) -> str:
    """Format facet info for the system prompt."""
    lines = []
    for f in scoring_data["facets"]:
        items_text = "; ".join(
            f'"{it["text"]}" ({"+" if it["keying"] == "+" else "−"})'
            for it in f["items"]
        )
        features_text = ", ".join(
            lf["description"] for lf in f["linguistic_features"]
        )
        lines.append(
            f"• {f['code']} {f['name']}: {f['description']}\n"
            f"  IPIP items: {items_text or 'N/A'}\n"
            f"  Linguistic markers: {features_text or 'N/A'}"
        )
    return "\n".join(lines)


def _classify(score: float) -> str:
    if score <= 2.3:
        return "Low"
    elif score <= 3.6:
        return "Medium"
    else:
        return "High"


def scorer_node(state: AssessmentState) -> dict:
    """LangGraph node: score the transcript and produce final results.

    Returns state updates with facet_scores, overall_score, classification.
    """
    transcript = state.get("transcript", "")

    if not transcript.strip():
        # Edge case: nothing to score
        return {
            "facet_scores": [],
            "overall_score": 3.0,
            "classification": "Medium",
            "messages": [
                AIMessage(
                    content="I wasn't able to gather enough information to "
                    "produce a reliable score. The assessment is incomplete."
                )
            ],
        }

    # ── Fetch graph context (Neo4j or local JSON fallback) ───────────
    scoring_data = get_all_data_for_scoring()

    facet_defs = _build_facet_definitions(scoring_data)

    # ── Build prompt ──────────────────────────────────────────────────
    system = SystemMessage(
        content=SCORER_SYSTEM_PROMPT.format(facet_definitions=facet_defs)
    )
    user_msg = HumanMessage(
        content=f"INTERVIEW TRANSCRIPT (user responses only):\n\n{transcript}"
    )

    # ── Call LLM ──────────────────────────────────────────────────────
    llm = _get_llm()
    response: AIMessage = llm.invoke([system, user_msg])

    # ── Parse response ────────────────────────────────────────────────
    try:
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(raw)
        facet_scores: list[FacetScore] = parsed["facet_scores"]
    except (json.JSONDecodeError, KeyError):
        # Fallback: could not parse — return neutral scores
        facet_scores = [
            {
                "facet_code": f"E{i}",
                "facet_name": name,
                "score": 3.0,
                "evidence": "Scoring parse error — defaulting to neutral.",
            }
            for i, name in enumerate(
                [
                    "Friendliness",
                    "Gregariousness",
                    "Assertiveness",
                    "Activity Level",
                    "Excitement-Seeking",
                    "Cheerfulness",
                ],
                start=1,
            )
        ]

    # ── Compute aggregate ─────────────────────────────────────────────
    scores_numeric = [fs["score"] for fs in facet_scores]
    overall = sum(scores_numeric) / len(scores_numeric) if scores_numeric else 3.0
    classification = _classify(overall)

    # ── Build summary message ─────────────────────────────────────────
    summary_lines = ["## Extraversion Assessment Results\n"]
    for fs in facet_scores:
        bar = "█" * int(fs["score"]) + "░" * (5 - int(fs["score"]))
        summary_lines.append(
            f"**{fs['facet_code']} {fs['facet_name']}**: "
            f"{fs['score']:.1f}/5.0  {bar}\n"
            f"  _{fs['evidence']}_\n"
        )
    summary_lines.append(f"\n**Overall Extraversion**: {overall:.2f}/5.0")
    summary_lines.append(f"**Classification**: {classification}")
    summary_text = "\n".join(summary_lines)

    return {
        "facet_scores": facet_scores,
        "overall_score": round(overall, 2),
        "classification": classification,
        "messages": [AIMessage(content=summary_text)],
    }
