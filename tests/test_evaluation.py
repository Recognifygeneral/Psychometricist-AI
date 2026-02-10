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


# ── _compute_metrics tests ─────────────────────────────────────────────


def test_compute_metrics_happy_path():
    from src.evaluation.compare import _compute_metrics

    self_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    ai_scores = [1.5, 2.5, 3.0, 3.5, 4.5]
    metrics = _compute_metrics(self_scores, ai_scores)

    assert metrics["n"] == 5
    assert 0.9 < metrics["pearson_r"] <= 1.0
    assert metrics["mae"] > 0
    assert 0 <= metrics["classification_agreement"] <= 1


def test_compute_metrics_too_few_scores():
    from src.evaluation.compare import _compute_metrics

    metrics = _compute_metrics([3.0], [3.0])
    assert "error" in metrics


def test_compute_metrics_perfect_agreement():
    from src.evaluation.compare import _compute_metrics

    scores = [1.0, 3.0, 5.0]
    metrics = _compute_metrics(scores, scores)
    assert metrics["pearson_r"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["classification_agreement"] == 1.0
