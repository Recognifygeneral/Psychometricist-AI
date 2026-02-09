"""API tests for the FastAPI web entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import web.app as web_app


@dataclass
class FakeGraph:
    """Small graph stub for API route tests."""

    fail_on_resume: bool = False
    complete_on_resume: bool = False

    def invoke(self, payload: Any, _config: dict) -> dict[str, Any]:
        is_start_call = isinstance(payload, dict)
        if is_start_call:
            return {
                "messages": [SimpleNamespace(content="Welcome! Tell me about yourself.")],
                "turn_count": 0,
                "max_turns": 10,
                "classification": "",
            }

        if self.fail_on_resume:
            raise RuntimeError("resume failed")

        response: dict[str, Any] = {
            "messages": [SimpleNamespace(content="Thanks, tell me more.")],
            "turn_count": 1,
            "max_turns": 10,
            "classification": "",
        }
        if self.complete_on_resume:
            response.update(
                {
                    "classification": "High",
                    "overall_score": 3.8,
                    "confidence": 0.7,
                    "scoring_results": {"ensemble_score": 3.8},
                    "facet_scores": [],
                }
            )
        return response


@pytest.fixture
def client_with_fake_graph(monkeypatch):
    fake = FakeGraph()
    monkeypatch.setattr(web_app, "graph", fake)
    return TestClient(web_app.app), fake


def test_start_session_returns_initial_ai_prompt(client_with_fake_graph):
    client, _ = client_with_fake_graph
    res = client.post("/api/start")
    data = res.json()

    assert res.status_code == 200
    assert data["status"] == "in-progress"
    assert data["session_id"]
    assert data["ai_message"]
    assert data["turn"] == 0


def test_respond_rejects_empty_message(client_with_fake_graph):
    client, _ = client_with_fake_graph
    res = client.post(
        "/api/respond",
        json={"session_id": "abc12345", "message": "   "},
    )
    assert res.status_code == 400
    assert "cannot be empty" in res.json()["detail"].lower()


def test_respond_rejects_too_long_message(client_with_fake_graph):
    client, _ = client_with_fake_graph
    long_text = "a" * (web_app.MAX_MESSAGE_CHARS + 1)
    res = client.post(
        "/api/respond",
        json={"session_id": "abc12345", "message": long_text},
    )
    assert res.status_code == 400
    assert "too long" in res.json()["detail"].lower()


def test_respond_rejects_invalid_session_id(client_with_fake_graph):
    client, _ = client_with_fake_graph
    res = client.post(
        "/api/respond",
        json={"session_id": "bad session id", "message": "hello"},
    )
    assert res.status_code == 422


def test_respond_returns_safe_error_when_resume_fails(monkeypatch):
    fake = FakeGraph(fail_on_resume=True)
    monkeypatch.setattr(web_app, "graph", fake)
    client = TestClient(web_app.app)

    res = client.post(
        "/api/respond",
        json={"session_id": "abc12345", "message": "hello"},
    )
    assert res.status_code == 400
    assert "start a new session" in res.json()["detail"].lower()


def test_respond_returns_complete_status(monkeypatch):
    fake = FakeGraph(complete_on_resume=True)
    monkeypatch.setattr(web_app, "graph", fake)
    client = TestClient(web_app.app)

    res = client.post(
        "/api/respond",
        json={"session_id": "abc12345", "message": "hello"},
    )
    data = res.json()
    assert res.status_code == 200
    assert data["status"] == "complete"
    assert data["classification"] == "High"
    assert data["overall_score"] == 3.8
