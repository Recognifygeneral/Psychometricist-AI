"""End-to-end test with mocked LLM — verifies the full interview loop
without requiring an OpenAI API key.

Run:  PYTHONPATH=. python tests/test_e2e_mock.py
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock
import json

from langchain_core.messages import AIMessage

# ── Mock responses ────────────────────────────────────────────────────────

INTERVIEWER_RESPONSES = [
    "Hi there! I'd love to get to know you. Tell me, what's it like for you when you walk into a room full of people you've never met before?",
    "That's really interesting! What does your ideal weekend look like — especially in terms of who you'd spend it with?",
    "I love that! When you think about projects or group situations, do you tend to step up and take charge? Tell me about a time that happened.",
    "How fascinating! When a group of friends can't decide what to do, what role do you usually play in making that decision?",
    "Great to hear! Walk me through a typical day for you — how do you spend your time and energy?",
    "Sounds like a full day! What's something exciting or adventurous you've done recently, or something you'd love to try?",
    "That's wonderful! Tell me about something that made you really happy recently.",
    "I can tell that meant a lot! How would your closest friends describe your general mood?",
    "Beautiful way to put it! When you're at a social event, do you find yourself connecting with many different people?",
    "That's great! What energizes you more — a quiet evening alone or a lively gathering with friends?",
    "Interesting! What's something you're really passionate about that you could talk about for hours?",
    "What a great conversation this has been! One last question — if you could plan any social event, what would it look like?",
]

SCORER_RESPONSE = json.dumps({
    "facet_scores": [
        {"facet_code": "E1", "facet_name": "Friendliness", "score": 4.0, "evidence": "Warm descriptions of meeting new people"},
        {"facet_code": "E2", "facet_name": "Gregariousness", "score": 3.5, "evidence": "Mixed preference for alone time vs groups"},
        {"facet_code": "E3", "facet_name": "Assertiveness", "score": 4.5, "evidence": "Consistently takes leadership roles"},
        {"facet_code": "E4", "facet_name": "Activity Level", "score": 3.0, "evidence": "Moderate pace of life described"},
        {"facet_code": "E5", "facet_name": "Excitement-Seeking", "score": 3.5, "evidence": "Some thrill-seeking but not extreme"},
        {"facet_code": "E6", "facet_name": "Cheerfulness", "score": 4.0, "evidence": "Frequent positive emotion expressions"},
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
    "Travel and food culture! I can talk about my trips and favorite restaurants endlessly.",
    "A huge outdoor festival with music, food, and all my friends together. That's my dream event!",
]

_call_count = 0


def mock_invoke(messages):
    """Return pre-scripted AI responses based on call order."""
    global _call_count
    idx = _call_count
    _call_count += 1

    # The last call is the Scorer
    if idx >= len(INTERVIEWER_RESPONSES):
        return AIMessage(content=SCORER_RESPONSE)

    return AIMessage(content=INTERVIEWER_RESPONSES[idx])


# ── Test ──────────────────────────────────────────────────────────────────


def test_full_interview_loop():
    """Simulate a complete 12-turn interview + scoring with mocked LLM."""
    global _call_count
    _call_count = 0

    from langgraph.types import Command
    from src.workflow import build_graph, FACET_ORDER, MAX_TURNS

    # Patch ChatOpenAI to return mock responses
    with patch("src.agents.interviewer.ChatOpenAI") as MockInterviewerLLM, \
         patch("src.agents.scorer.ChatOpenAI") as MockScorerLLM:

        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke
        MockInterviewerLLM.return_value = mock_llm
        MockScorerLLM.return_value = mock_llm

        graph = build_graph()
        config = {"configurable": {"thread_id": "test-001"}}

        initial_state = {
            "current_facet": FACET_ORDER[0],
            "explored_facets": [],
            "probes_used": [],
            "transcript": "",
            "facet_scores": [],
            "overall_score": 0.0,
            "classification": "",
            "turn_count": 0,
            "max_turns": MAX_TURNS,
            "done": False,
        }

        # First invocation — should hit router → interviewer → human_turn (interrupt)
        result = graph.invoke(initial_state, config)

        turn = 0
        while True:
            # Check if scoring is done
            if result.get("classification"):
                break

            if turn >= len(USER_RESPONSES):
                print(f"WARN: Ran out of user responses at turn {turn}")
                break

            # Resume with next user response
            result = graph.invoke(
                Command(resume=USER_RESPONSES[turn]),
                config,
            )
            turn += 1

    # ── Assertions ────────────────────────────────────────────────────
    print(f"\nTurns completed: {turn}")
    print(f"Classification:  {result.get('classification')}")
    print(f"Overall score:   {result.get('overall_score')}")

    assert result.get("classification") in ("Low", "Medium", "High"), \
        f"Expected classification, got: {result.get('classification')}"
    assert result.get("overall_score", 0) > 0, "Expected a positive score"
    assert result.get("facet_scores"), "Expected facet_scores to be populated"
    assert len(result["facet_scores"]) == 6, f"Expected 6 facet scores, got {len(result['facet_scores'])}"

    # Check transcript accumulated
    transcript = result.get("transcript", "")
    assert len(transcript) > 100, f"Transcript too short: {len(transcript)} chars"

    print(f"Transcript length: {len(transcript)} chars")
    print(f"Facet scores: {[(fs['facet_code'], fs['score']) for fs in result['facet_scores']]}")

    print("\n✓ End-to-end test PASSED!")


if __name__ == "__main__":
    test_full_interview_loop()
