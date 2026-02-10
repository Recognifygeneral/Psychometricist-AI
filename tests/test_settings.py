"""Tests for shared settings and threshold classification."""

from __future__ import annotations

import src.settings as settings


def test_classify_extraversion_uses_thresholds():
    assert settings.classify_extraversion(2.0) == "Low"
    assert settings.classify_extraversion(3.0) == "Medium"
    assert settings.classify_extraversion(4.0) == "High"


def test_invalid_threshold_env_falls_back(monkeypatch):
    monkeypatch.setenv("LOW_EXTRAVERSION_THRESHOLD", "not-a-number")
    monkeypatch.setenv("HIGH_EXTRAVERSION_THRESHOLD", "still-not-a-number")
    settings.reset()

    assert settings.LOW_EXTRAVERSION_THRESHOLD == 2.3
    assert settings.HIGH_EXTRAVERSION_THRESHOLD == 3.6

    # Restore canonical thresholds so later imports in the same test run
    # keep the expected behavior.
    monkeypatch.setenv("LOW_EXTRAVERSION_THRESHOLD", "2.3")
    monkeypatch.setenv("HIGH_EXTRAVERSION_THRESHOLD", "3.6")
    settings.reset()


def test_lazy_model_name(monkeypatch):
    """Model names are read lazily — monkeypatch works without reload."""
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-test-model")
    settings.reset()

    assert settings.LLM_MODEL_NAME == "gpt-test-model"

    # Restore default
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    settings.reset()
