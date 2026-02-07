"""FastAPI backend for the psychometric interview web interface.

Endpoints:
    POST /api/start       → start a new interview session, returns first AI message
    POST /api/respond     → send user message, returns next AI message
    GET  /                → serve the chat UI
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.workflow import build_graph, MAX_TURNS

app = FastAPI(title="AI Psychometricist", version="0.2.0")

# Build the graph once at startup
graph = build_graph()

# ── Request / Response models ─────────────────────────────────────────────


class StartRequest(BaseModel):
    pass


class RespondRequest(BaseModel):
    session_id: str
    message: str


class MessageResponse(BaseModel):
    session_id: str
    ai_message: str
    turn: int
    max_turns: int
    status: str  # "in-progress" | "complete"
    # Only present when status == "complete"
    overall_score: float | None = None
    classification: str | None = None
    confidence: float | None = None
    scoring_results: dict[str, Any] | None = None
    facet_scores: list[dict] | None = None


# ── Helpers ───────────────────────────────────────────────────────────────


def _extract_response(result: dict, session_id: str) -> MessageResponse:
    """Build a MessageResponse from the graph state."""
    messages = result.get("messages", [])
    last_ai = messages[-1].content if messages else ""

    classification = result.get("classification", "")
    is_complete = bool(classification)

    return MessageResponse(
        session_id=session_id,
        ai_message=last_ai,
        turn=result.get("turn_count", 0),
        max_turns=result.get("max_turns", MAX_TURNS),
        status="complete" if is_complete else "in-progress",
        overall_score=result.get("overall_score") if is_complete else None,
        classification=classification if is_complete else None,
        confidence=result.get("confidence") if is_complete else None,
        scoring_results=result.get("scoring_results") if is_complete else None,
        facet_scores=result.get("facet_scores") if is_complete else None,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.post("/api/start", response_model=MessageResponse)
def start_session():
    """Start a new interview session."""
    session_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "session_id": session_id,
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

    result = graph.invoke(initial_state, config)
    return _extract_response(result, session_id)


@app.post("/api/respond", response_model=MessageResponse)
def respond(req: RespondRequest):
    """Send a user response and get the next interviewer question (or final scores)."""
    config = {"configurable": {"thread_id": req.session_id}}

    try:
        result = graph.invoke(Command(resume=req.message), config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _extract_response(result, req.session_id)


# ── Serve static frontend ────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the chat interface."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
