"""FastAPI backend for the psychometric interview web interface."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.logging_config import setup_logging
from src.models.initial_state import new_assessment_state
from src.workflow import MAX_TURNS, build_graph

load_dotenv()
setup_logging()

# Railway sometimes stores env values with trailing whitespace after copy/paste.
for key in ("OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
    value = os.environ.get(key)
    if value:
        os.environ[key] = value.strip()

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Psychometricist", version="0.2.0")
graph = build_graph()
MAX_MESSAGE_CHARS = 4000


# ── Request logging middleware ────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, and response time."""
    import time
    import uuid as _uuid

    request_id = _uuid.uuid4().hex[:8]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s (%.0fms) [rid=%s]",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ── Health check ──────────────────────────────────────────────────────────

@app.get("/health")
def health_check() -> JSONResponse:
    """Lightweight health probe for deployment platforms."""
    return JSONResponse({"status": "ok"})


class RespondRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    message: str


class MessageResponse(BaseModel):
    session_id: str
    ai_message: str
    turn: int
    max_turns: int
    status: str
    overall_score: float | None = None
    classification: str | None = None
    confidence: float | None = None
    scoring_results: dict[str, Any] | None = None
    facet_scores: list[dict] | None = None


def _extract_response(result: dict[str, Any], session_id: str) -> MessageResponse:
    """Build API response from graph state."""
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


@app.post("/api/start", response_model=MessageResponse)
def start_session() -> MessageResponse:
    """Start a new interview session."""
    session_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": session_id}}
    initial_state = new_assessment_state(session_id=session_id, max_turns=MAX_TURNS)

    result = graph.invoke(initial_state, config)
    return _extract_response(result, session_id)


@app.post("/api/respond", response_model=MessageResponse)
def respond(req: RespondRequest) -> MessageResponse:
    """Send a user response and return the next state."""
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(user_message) > MAX_MESSAGE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long. Maximum length is {MAX_MESSAGE_CHARS} characters.",
        )

    config = {"configurable": {"thread_id": req.session_id}}

    try:
        result = graph.invoke(Command(resume=user_message), config)
    except Exception:
        logger.exception("Failed to resume session %s", req.session_id)
        raise HTTPException(
            status_code=400,
            detail="Unable to continue this session. Start a new session and try again.",
        ) from None

    return _extract_response(result, req.session_id)


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def serve_ui() -> HTMLResponse:
    """Serve the chat interface."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    print(f"Starting web interface on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
