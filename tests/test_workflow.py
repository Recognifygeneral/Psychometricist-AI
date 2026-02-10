"""Tests for LangGraph workflow node functions (router, human_turn, update_state).

These tests exercise the pure logic of each node in isolation,
without building the full graph or calling any LLM.
"""

from __future__ import annotations

from src.models.initial_state import new_assessment_state
from src.workflow import MAX_TURNS, router, update_state


class TestRouter:
    """Test the router node that decides interviewer vs. scorer."""

    def test_routes_to_interviewer_when_turns_remain(self):
        state = new_assessment_state(session_id="r-1", max_turns=10)
        cmd = router(state)
        assert cmd.goto == "interviewer"

    def test_routes_to_scorer_when_max_turns_reached(self):
        state = new_assessment_state(session_id="r-2", max_turns=10)
        state["turn_count"] = 10
        cmd = router(state)
        assert cmd.goto == "scorer"

    def test_routes_to_scorer_when_done_flag_set(self):
        state = new_assessment_state(session_id="r-3", max_turns=10)
        state["done"] = True
        cmd = router(state)
        assert cmd.goto == "scorer"

    def test_routes_to_interviewer_at_boundary_minus_one(self):
        state = new_assessment_state(session_id="r-4", max_turns=5)
        state["turn_count"] = 4
        cmd = router(state)
        assert cmd.goto == "interviewer"

    def test_routes_to_scorer_at_exact_boundary(self):
        state = new_assessment_state(session_id="r-5", max_turns=5)
        state["turn_count"] = 5
        cmd = router(state)
        assert cmd.goto == "scorer"


class TestUpdateState:
    """Test the update_state node that processes user responses."""

    def _make_state_with_input(self, user_text: str, turn_count: int = 0) -> dict:
        state = new_assessment_state(session_id="u-1", max_turns=10)
        state["user_input"] = user_text
        state["turn_count"] = turn_count
        state["messages"] = []
        return state

    def test_increments_turn_count(self):
        state = self._make_state_with_input("Hello!", turn_count=0)
        result = update_state(state)
        assert result["turn_count"] == 1

    def test_accumulates_transcript(self):
        state = self._make_state_with_input("I love parties!", turn_count=2)
        result = update_state(state)
        assert "[Turn 3]" in result["transcript"]
        assert "I love parties!" in result["transcript"]

    def test_extracts_features_per_turn(self):
        state = self._make_state_with_input(
            "I love going to exciting parties with friends!",
            turn_count=0,
        )
        result = update_state(state)
        assert len(result["turn_features"]) == 1
        assert "word_count" in result["turn_features"][0]
        assert result["turn_features"][0]["word_count"] > 0

    def test_creates_turn_record(self):
        state = self._make_state_with_input("Hello world", turn_count=0)
        result = update_state(state)
        assert len(result["turn_records"]) == 1
        record = result["turn_records"][0]
        assert record["turn_number"] == 1
        assert record["user_message"] == "Hello world"
        assert "timestamp" in record
        assert "features" in record

    def test_sets_done_at_max_turns(self):
        state = self._make_state_with_input("Last answer", turn_count=9)
        state["max_turns"] = 10
        result = update_state(state)
        assert result["done"] is True

    def test_not_done_before_max_turns(self):
        state = self._make_state_with_input("Mid answer", turn_count=4)
        state["max_turns"] = 10
        result = update_state(state)
        assert result["done"] is False

    def test_appends_human_message(self):
        state = self._make_state_with_input("Test message", turn_count=0)
        result = update_state(state)
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Test message"


class TestMaxTurnsConstant:
    def test_max_turns_is_ten(self):
        assert MAX_TURNS == 10
