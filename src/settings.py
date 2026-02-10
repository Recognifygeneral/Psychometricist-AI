"""Project-wide settings and shared scoring constants.

All environment-dependent values are read **lazily** on first access
(not at import time) and cached via ``functools.lru_cache``.  Call
``reset()`` in tests to clear the cache after changing env vars —
no ``importlib.reload`` required.
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Final


def _float_env(name: str, default: float) -> float:
    """Parse float environment values with a safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ── Constant (never changes at runtime) ──────────────────────────────────
NEUTRAL_SCORE: Final[float] = 3.0


# ── Lazy settings cache ──────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_settings() -> dict[str, object]:
    """Read env-dependent settings once and cache the result."""
    return {
        "LOW_EXTRAVERSION_THRESHOLD": _float_env("LOW_EXTRAVERSION_THRESHOLD", 2.3),
        "HIGH_EXTRAVERSION_THRESHOLD": _float_env("HIGH_EXTRAVERSION_THRESHOLD", 3.6),
        "LLM_MODEL_NAME": os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2"),
        "EMBEDDING_MODEL_NAME": os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
    }


def reset() -> None:
    """Clear the cached settings — call from tests after monkeypatching env vars."""
    _load_settings.cache_clear()


# Type declarations for static analysis (not set at runtime so
# ``__getattr__`` is invoked on attribute access).
if TYPE_CHECKING:
    LOW_EXTRAVERSION_THRESHOLD: float
    HIGH_EXTRAVERSION_THRESHOLD: float
    LLM_MODEL_NAME: str
    EMBEDDING_MODEL_NAME: str


def __getattr__(name: str) -> object:
    """PEP 562 module-level ``__getattr__`` — provides lazy env reads."""
    settings = _load_settings()
    if name in settings:
        return settings[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── Public helpers ────────────────────────────────────────────────────────


def classify_extraversion(score: float) -> str:
    """Map a numeric Extraversion score to Low / Medium / High."""
    s = _load_settings()
    if score <= s["LOW_EXTRAVERSION_THRESHOLD"]:  # type: ignore[operator]
        return "Low"
    if score <= s["HIGH_EXTRAVERSION_THRESHOLD"]:  # type: ignore[operator]
        return "Medium"
    return "High"
