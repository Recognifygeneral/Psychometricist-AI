"""Session logger — structured JSON logging of each interview session.

Captures every detail needed for scientific analysis:
  - Per-turn: timestamp, AI prompt, user response, extracted features
  - Session-level: scoring results from all methods, metadata
  - Saved as a JSON file per session for easy post-hoc analysis

Output directory: data/sessions/
File format: {session_id}_{timestamp}.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.extraction.features import LinguisticFeatures

SESSIONS_DIR = Path(__file__).resolve().parents[2] / "data" / "sessions"


def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


class SessionLogger:
    """Accumulates session data and writes a structured JSON log.

    Usage:
        logger = SessionLogger(session_id="abc123")
        logger.log_turn(
            turn_number=1,
            ai_message="Hi! Tell me about...",
            user_message="Well, I usually...",
            probe_id="probe_E1_1",
            features=extract_features(user_text),
        )
        ...
        logger.log_scoring(ensemble_results)
        logger.save()
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: str | None = None
        self.turns: list[dict[str, Any]] = []
        self.scoring: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {
            "trait": "Extraversion",
            "instrument": "IPIP 50-item Big Five Markers",
            "model": "gpt-5.2",
        }

    def log_turn(
        self,
        turn_number: int,
        ai_message: str,
        user_message: str,
        probe_id: str | None = None,
        features: LinguisticFeatures | None = None,
    ) -> None:
        """Record a single interview turn.

        Parameters
        ----------
        turn_number : int
            The 1-based turn index.
        ai_message : str
            The interviewer's question/prompt.
        user_message : str
            The user's response.
        probe_id : str, optional
            The ID of the probe used for this turn.
        features : LinguisticFeatures, optional
            Extracted linguistic features for this turn.
        """
        turn_record = {
            "turn_number": turn_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "probe_id": probe_id,
            "ai_message": ai_message,
            "user_message": user_message,
            "features": features.to_dict() if features else None,
        }
        self.turns.append(turn_record)

    def log_scoring(self, scoring_results: dict[str, Any]) -> None:
        """Record the final scoring output.

        Parameters
        ----------
        scoring_results : dict
            Output from ensemble.score_ensemble().
        """
        self.scoring = scoring_results
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def log_self_report(self, self_report_score: float) -> None:
        """Attach the self-report validation score.

        Parameters
        ----------
        self_report_score : float
            The user's IPIP self-report Extraversion score (1–5).
        """
        self.scoring["self_report_score"] = self_report_score

    def set_metadata(self, key: str, value: Any) -> None:
        """Add arbitrary metadata to the session log."""
        self.metadata[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full session log to a dictionary."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
            "total_turns": len(self.turns),
            "turns": self.turns,
            "scoring": self.scoring,
        }

    def save(self) -> Path:
        """Write the session log to a JSON file.

        Returns
        -------
        Path
            Absolute path to the saved log file.
        """
        _ensure_dir()

        if self.completed_at is None:
            self.completed_at = datetime.now(timezone.utc).isoformat()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_id}_{timestamp}.json"
        filepath = SESSIONS_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        return filepath

    def summary(self) -> str:
        """One-line summary of the session."""
        score = self.scoring.get("ensemble_score", "?")
        cls = self.scoring.get("ensemble_classification", "?")
        return (
            f"Session {self.session_id}: {len(self.turns)} turns, "
            f"score={score}, classification={cls}"
        )


def load_session(filepath: str | Path) -> dict[str, Any]:
    """Load a session log from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to the session JSON file.

    Returns
    -------
    dict
        The full session log data.
    """
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> list[Path]:
    """List all session log files, newest first.

    Returns
    -------
    list[Path]
        Sorted list of session JSON file paths.
    """
    _ensure_dir()
    files = list(SESSIONS_DIR.glob("*.json"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def load_all_sessions() -> list[dict[str, Any]]:
    """Load all session logs for analysis.

    Returns
    -------
    list[dict]
        All session records, newest first.
    """
    return [load_session(p) for p in list_sessions()]
