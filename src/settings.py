"""Project-wide settings and shared scoring constants."""

from __future__ import annotations

import os
from typing import Final


def _float_env(name: str, default: float) -> float:
    """Parse float environment values with a safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

# Shared Extraversion score anchors
NEUTRAL_SCORE: Final[float] = 3.0
LOW_EXTRAVERSION_THRESHOLD: Final[float] = _float_env("LOW_EXTRAVERSION_THRESHOLD", 2.3)
HIGH_EXTRAVERSION_THRESHOLD: Final[float] = _float_env("HIGH_EXTRAVERSION_THRESHOLD", 3.6)

# Shared model configuration (overridable via environment variables)
LLM_MODEL_NAME: Final[str] = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2")
EMBEDDING_MODEL_NAME: Final[str] = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def classify_extraversion(score: float) -> str:
    """Map a numeric Extraversion score to Low / Medium / High."""
    if score <= LOW_EXTRAVERSION_THRESHOLD:
        return "Low"
    if score <= HIGH_EXTRAVERSION_THRESHOLD:
        return "Medium"
    return "High"
