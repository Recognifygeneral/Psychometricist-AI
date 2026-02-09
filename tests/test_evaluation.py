"""Tests for evaluation/self-report data plumbing."""

from __future__ import annotations

import csv
import json

from src.evaluation import self_report


def test_attach_self_report_to_session_updates_log(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "abc123_20260209_120000.json"
    session_file.write_text(
        json.dumps(
            {
                "session_id": "abc123",
                "scoring": {
                    "ensemble_score": 3.8,
                    "ensemble_classification": "High",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(self_report, "SESSIONS_DIR", session_dir)

    attached, ai_score, ai_classification = self_report.attach_self_report_to_session(
        "abc123", 4.25
    )

    assert attached is True
    assert ai_score == 3.8
    assert ai_classification == "High"

    saved = json.loads(session_file.read_text(encoding="utf-8"))
    assert saved["scoring"]["self_report_score"] == 4.25


def test_save_result_writes_ai_fields(tmp_path, monkeypatch):
    results_path = tmp_path / "pilot_results.csv"
    monkeypatch.setattr(self_report, "RESULTS_PATH", results_path)

    self_report.save_result(
        user_id="participant-001",
        self_report_score=3.4,
        ai_score=3.8,
        ai_classification="High",
    )

    with open(results_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["user_id"] == "participant-001"
    assert rows[0]["self_report_score"] == "3.4"
    assert rows[0]["ai_score"] == "3.8"
    assert rows[0]["ai_classification"] == "High"


def test_attach_self_report_rejects_invalid_session_id(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    monkeypatch.setattr(self_report, "SESSIONS_DIR", session_dir)

    attached, ai_score, ai_classification = self_report.attach_self_report_to_session(
        "../bad-session",
        3.0,
    )

    assert attached is False
    assert ai_score is None
    assert ai_classification == ""
