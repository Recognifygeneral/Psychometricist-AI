"""End-to-end test with mocked LLM — verifies the full interview loop
without requiring an OpenAI API key.

Tests the simplified workflow (flat probe pool, ensemble scoring).

Run:  PYTHONPATH=. python tests/test_e2e_mock.py
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

# ── Mock responses ────────────────────────────────────────────────────────

INTERVIEWER_RESPONSES = [
    "Hi there! I'd love to get to know you. Tell me, what's it like for you when you walk into a room full of people you've never met before?",
    "That's really interesting! What does your ideal weekend look like — especially in terms of who you'd spend it with?",
    "I love that! When you think about projects or group situations, do you tend to step up and take charge?",
    "How fascinating! When a group of friends can't decide what to do, what role do you usually play?",
    "Great to hear! Walk me through a typical day for you — how do you spend your time and energy?",
    "Sounds busy! What's something exciting or adventurous you've done recently?",
    "That's wonderful! Tell me about something that made you really happy recently.",
    "I can tell that meant a lot! How would your closest friends describe your general mood?",
    "Beautiful way to put it! When you're at a social event, do you find yourself connecting with many different people?",
    "What a great conversation this has been! One last question — what energizes you more, quiet time or lively gatherings?",
]

# LLM domain-level scorer response
LLM_DOMAIN_RESPONSE = json.dumps({
    "score": 3.75,
    "classification": "High",
    "confidence": 0.8,
    "evidence": "Consistent social orientation, positive emotion, and assertive language."
})

# LLM facet-level scorer response
LLM_FACET_RESPONSE = json.dumps({
    "facet_scores": [
        {"facet_code": "E1", "facet_name": "Friendliness", "score": 4.0, "evidence": "Warm descriptions"},
        {"facet_code": "E2", "facet_name": "Gregariousness", "score": 3.5, "evidence": "Prefers groups"},
        {"facet_code": "E3", "facet_name": "Assertiveness", "score": 4.5, "evidence": "Takes charge"},
        {"facet_code": "E4", "facet_name": "Activity Level", "score": 3.0, "evidence": "Moderate pace"},
        {"facet_code": "E5", "facet_name": "Excitement-Seeking", "score": 3.5, "evidence": "Some thrills"},
        {"facet_code": "E6", "facet_name": "Cheerfulness", "score": 4.0, "evidence": "Positive affect"},
    ]
})

USER_RESPONSES = [
    "I usually walk right up to people and start chatting. I find it exciting to meet new faces!",
    "My ideal weekend involves brunch with friends, a group hike, and then maybe a dinner party at my place.",
    "Yes, absolutely! At work I often volunteer to lead the team meetings and organize project timelines.",
    "I'm usually the one suggesting options and rallying everyone to make a decision. I don't like sitting around.",
    "I wake up early, hit the gym, then it's meetings and deep work. I try to squeeze in coffee with colleagues.",
    "I went bungee jumping last month! I'm always looking for the next adventure. Skydiving is on my list.",
    "My friend's surprise birthday party — seeing their face light up made my whole week!",
    "They'd say I'm basically always smiling and bringing good energy. My nickname is 'Sunshine'.",
    "Oh definitely, I try to talk to everyone! I love hearing different people's stories.",
    "A lively gathering, no question! I feed off that social energy.",
]


# ── Mock LLM factory ─────────────────────────────────────────────────────

_interviewer_call_count = 0


def mock_interviewer_invoke(messages):
    """Return pre-scripted interviewer responses."""
    global _interviewer_call_count
    idx = _interviewer_call_count
    _interviewer_call_count += 1
    if idx < len(INTERVIEWER_RESPONSES):
        return AIMessage(content=INTERVIEWER_RESPONSES[idx])
    return AIMessage(content="Tell me more about yourself.")


_llm_scorer_call_count = 0


def mock_llm_scorer_invoke(messages):
    """Return domain or facet scorer response based on call order."""
    global _llm_scorer_call_count
    _llm_scorer_call_count += 1
    # First call is domain-level, second is facet-level
    if _llm_scorer_call_count <= 1:
        return AIMessage(content=LLM_DOMAIN_RESPONSE)
    return AIMessage(content=LLM_FACET_RESPONSE)


# ── Test ──────────────────────────────────────────────────────────────────


def test_full_interview_loop():
    """Simulate a complete 10-turn interview + ensemble scoring with mocked LLMs."""
    global _interviewer_call_count, _llm_scorer_call_count
    _interviewer_call_count = 0
    _llm_scorer_call_count = 0

    from langgraph.types import Command
    from src.workflow import build_graph, MAX_TURNS

    # Mock the embedding scorer to avoid API calls
    mock_embedding_result = {
        "method": "embedding",
        "score": 3.8,
        "classification": "High",
        "confidence": 0.6,
        "high_similarity": 0.85,
        "low_similarity": 0.72,
        "balance": 0.08,
    }

    with patch("src.agents.interviewer.ChatOpenAI") as MockInterviewerLLM, \
         patch("src.scoring.llm_scorer.ChatOpenAI") as MockScorerLLM, \
         patch("src.scoring.embedding_scorer.score_with_embeddings", return_value=mock_embedding_result):

        mock_interviewer = MagicMock()
        mock_interviewer.invoke = mock_interviewer_invoke
        MockInterviewerLLM.return_value = mock_interviewer

        mock_scorer = MagicMock()
        mock_scorer.invoke = mock_llm_scorer_invoke
        MockScorerLLM.return_value = mock_scorer

        graph = build_graph()
        config = {"configurable": {"thread_id": "test-e2e-001"}}

        initial_state = {
            "session_id": "test-e2e-001",
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
            "max_turns": MAX_TURNS,
            "done": False,
        }

        # First invocation
        result = graph.invoke(initial_state, config)

        turn = 0
        while True:
            if result.get("classification"):
                break
            if turn >= len(USER_RESPONSES):
                print(f"WARN: Ran out of user responses at turn {turn}")
                break
            result = graph.invoke(Command(resume=USER_RESPONSES[turn]), config)
            turn += 1

    # ── Assertions ────────────────────────────────────────────────────
    print(f"\nTurns completed: {turn}")
    print(f"Classification:  {result.get('classification')}")
    print(f"Overall score:   {result.get('overall_score')}")
    print(f"Confidence:      {result.get('confidence')}")

    assert result.get("classification") in ("Low", "Medium", "High"), \
        f"Expected classification, got: {result.get('classification')}"
    assert result.get("overall_score", 0) > 0, "Expected a positive score"
    assert 0.0 <= result.get("confidence", -1) <= 1.0, "Confidence out of range"

    # Check scoring_results has multi-method data
    scoring = result.get("scoring_results", {})
    assert "ensemble_score" in scoring, "Missing ensemble_score"
    assert "individual_results" in scoring, "Missing individual_results"

    individual = scoring["individual_results"]
    assert "feature_based" in individual, "Missing feature_based scorer"
    print(f"Feature score:   {individual['feature_based'].get('score')}")

    # Check transcript accumulated
    transcript = result.get("transcript", "")
    assert len(transcript) > 100, f"Transcript too short: {len(transcript)} chars"

    # Check turn records were created
    turn_records = result.get("turn_records", [])
    assert len(turn_records) > 0, "No turn records found"

    # Check turn features were extracted
    turn_features = result.get("turn_features", [])
    assert len(turn_features) > 0, "No turn features found"
    assert "word_count" in turn_features[0], "Turn features missing word_count"

    print(f"Transcript length: {len(transcript)} chars")
    print(f"Turn records: {len(turn_records)}")
    print(f"Turn features: {len(turn_features)}")

    print("\n✓ End-to-end test PASSED!")


if __name__ == "__main__":
    test_full_interview_loop()
