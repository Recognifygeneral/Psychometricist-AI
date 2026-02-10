"""Centralized path constants for the project.

All data / session / result file paths are defined here so that every
module imports from a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
SESSIONS_DIR: Path = DATA_DIR / "sessions"
RESULTS_PATH: Path = DATA_DIR / "pilot_results.csv"
IPIP_DATA_PATH: Path = DATA_DIR / "ipip_extraversion.json"
