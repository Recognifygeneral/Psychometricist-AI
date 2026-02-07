"""LLM-based scorer — simplified domain-level classification.

Replaces the original per-facet scorer with a streamlined approach:
one overall Extraversion classification with confidence and evidence.

Scientific framing: the LLM acts as a "human rater analogue" —
an informed judge who reads the transcript and forms a holistic
impression, similar to how personality researchers do consensus
coding of interview transcripts.

Per-facet scoring is retained as an OPTIONAL secondary mode.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_SCORER_PROMPT = """\
You are an expert personality psychologist scoring an interview transcript
for EXTRAVERSION — the tendency toward sociability, assertiveness,
positive emotionality, and engagement with the external world.

You will receive the user's responses from a conversational interview.

Your task: produce a SINGLE overall Extraversion rating.

SCORING SCALE:
  1.0 = Very Low Extraversion (consistently introverted signals)
  2.0 = Low Extraversion
  3.0 = Average / Neutral (ambiguous or mixed signals)
  4.0 = High Extraversion
  5.0 = Very High Extraversion (consistently extraverted signals)

BEHAVIORAL INDICATORS TO LOOK FOR:
  Higher Extraversion:
    - References to social activities, friends, groups, parties
    - Enthusiastic, energetic, positive emotional tone
    - Assertive, confident language; willingness to lead
    - Excitement-seeking, adventure, spontaneity
    - Longer, more elaborate responses with vivid descriptions

  Lower Extraversion:
    - Preference for solitude, quiet activities, small groups
    - Reserved, cautious, tentative language
    - Hedging, qualifying statements
    - Comfort-seeking, routine preference
    - Shorter, more measured responses

IMPORTANT GUIDELINES:
  - Rate based on BEHAVIORAL EVIDENCE, not self-claims
  - Absence of evidence → default to 3.0 (neutral), NOT 1.0
  - Most people score 2.5–3.5; reserve extremes for clear evidence
  - Consider CONSISTENCY — scattered signals → closer to 3.0

RESPOND WITH VALID JSON ONLY — no markdown, no commentary:
{{
  "score": 3.5,
  "classification": "Medium",
  "confidence": 0.7,
  "evidence": "Brief 2-3 sentence summary of key behavioral evidence."
}}

Classification rules:
  score <= 2.3 → "Low"
  2.3 < score <= 3.6 → "Medium"
  score > 3.6 → "High"

Confidence (0.0–1.0):
  0.0–0.3 = very uncertain (short/ambiguous transcript)
  0.4–0.6 = moderate certainty
  0.7–1.0 = high certainty (clear, consistent evidence)
"""


FACET_SCORER_PROMPT = """\
You are an expert personality psychologist. Rate the user on EACH of
the six Extraversion facets on a 1–5 scale.

FACET DEFINITIONS:
  E1 Friendliness: Warmth and comfort in social encounters with strangers
  E2 Gregariousness: Preference for being in groups vs. being alone
  E3 Assertiveness: Taking charge, expressing opinions confidently
  E4 Activity Level: Pace of life, energy expenditure, busyness
  E5 Excitement-Seeking: Need for stimulation, novelty, adventure
  E6 Cheerfulness: Frequency of positive emotions, optimism

RESPOND WITH VALID JSON ONLY:
{{
  "facet_scores": [
    {{"facet_code": "E1", "facet_name": "Friendliness", "score": 3.5, "evidence": "..."}},
    {{"facet_code": "E2", "facet_name": "Gregariousness", "score": 3.0, "evidence": "..."}},
    {{"facet_code": "E3", "facet_name": "Assertiveness", "score": 3.0, "evidence": "..."}},
    {{"facet_code": "E4", "facet_name": "Activity Level", "score": 3.0, "evidence": "..."}},
    {{"facet_code": "E5", "facet_name": "Excitement-Seeking", "score": 3.0, "evidence": "..."}},
    {{"facet_code": "E6", "facet_name": "Cheerfulness", "score": 3.0, "evidence": "..."}}
  ]
}}
"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-5.2", temperature=0.0)


def _classify(score: float) -> str:
    if score <= 2.3:
        return "Low"
    elif score <= 3.6:
        return "Medium"
    else:
        return "High"


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def score_domain_level(transcript: str) -> dict[str, Any]:
    """Score overall Extraversion from the transcript.

    This is the PRIMARY scoring mode for the MVP.

    Parameters
    ----------
    transcript : str
        User-only text from the interview.

    Returns
    -------
    dict with keys: method, score, classification, confidence, evidence
    """
    if not transcript.strip():
        return {
            "method": "llm_domain",
            "score": 3.0,
            "classification": "Medium",
            "confidence": 0.0,
            "evidence": "Empty transcript — no evidence to score.",
        }

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=DOMAIN_SCORER_PROMPT),
            HumanMessage(content=f"INTERVIEW TRANSCRIPT (user responses only):\n\n{transcript}"),
        ])

        parsed = _parse_json(response.content)
        score = max(1.0, min(5.0, float(parsed["score"])))

        return {
            "method": "llm_domain",
            "score": round(score, 2),
            "classification": parsed.get("classification", _classify(score)),
            "confidence": round(float(parsed.get("confidence", 0.5)), 3),
            "evidence": parsed.get("evidence", ""),
        }

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {
            "method": "llm_domain",
            "score": 3.0,
            "classification": "Medium",
            "confidence": 0.0,
            "evidence": f"LLM parse error: {e}. Defaulting to neutral.",
        }

    except Exception as e:
        return {
            "method": "llm_domain",
            "score": 3.0,
            "classification": "Medium",
            "confidence": 0.0,
            "error": f"LLM scorer failed: {str(e)}",
        }


def score_facet_level(transcript: str) -> dict[str, Any]:
    """Score each Extraversion facet (OPTIONAL secondary mode).

    Parameters
    ----------
    transcript : str
        User-only text from the interview.

    Returns
    -------
    dict with keys: method, facet_scores, overall_score, classification
    """
    if not transcript.strip():
        return {
            "method": "llm_facet",
            "facet_scores": [],
            "overall_score": 3.0,
            "classification": "Medium",
        }

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=FACET_SCORER_PROMPT),
            HumanMessage(content=f"INTERVIEW TRANSCRIPT (user responses only):\n\n{transcript}"),
        ])

        parsed = _parse_json(response.content)
        facet_scores = parsed["facet_scores"]

        scores = [fs["score"] for fs in facet_scores]
        overall = sum(scores) / len(scores) if scores else 3.0

        return {
            "method": "llm_facet",
            "facet_scores": facet_scores,
            "overall_score": round(overall, 2),
            "classification": _classify(overall),
        }

    except Exception as e:
        return {
            "method": "llm_facet",
            "facet_scores": [],
            "overall_score": 3.0,
            "classification": "Medium",
            "error": str(e),
        }


def explain_score(result: dict) -> str:
    """Human-readable explanation."""
    method = result.get("method", "llm")
    if method == "llm_facet":
        lines = [f"LLM Facet-Level Score: {result['overall_score']:.2f}/5.0 → {result['classification']}"]
        for fs in result.get("facet_scores", []):
            lines.append(f"  {fs['facet_code']} {fs['facet_name']}: {fs['score']:.1f} — {fs['evidence']}")
        return "\n".join(lines)
    else:
        return (
            f"LLM Domain-Level Score: {result['score']:.2f}/5.0 → {result['classification']}\n"
            f"Confidence: {result['confidence']:.1%}\n"
            f"Evidence: {result.get('evidence', 'N/A')}"
        )
