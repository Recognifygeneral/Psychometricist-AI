"""CLI self-report workflow for IPIP Extraversion items."""

from __future__ import annotations

import csv
import json
import re
import uuid
from pathlib import Path

from src.settings import classify_extraversion

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ipip_extraversion.json"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "pilot_results.csv"
SESSIONS_DIR = Path(__file__).resolve().parents[2] / "data" / "sessions"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _load_items() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def _reverse_score(raw: int) -> int:
    """Reverse-score a 1-5 item: score = 6 - raw."""
    return 6 - raw


def _find_latest_session_log(session_id: str) -> Path | None:
    """Return the newest session file for a given session ID."""
    if not SESSIONS_DIR.exists():
        return None
    matches = sorted(
        SESSIONS_DIR.glob(f"{session_id}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _is_valid_session_id(session_id: str) -> bool:
    return bool(SESSION_ID_PATTERN.fullmatch(session_id))


def attach_self_report_to_session(
    session_id: str,
    self_report_score: float,
) -> tuple[bool, float | None, str]:
    """Attach self-report score to a saved interview session log.

    Returns (attached, ai_score, ai_classification).
    """
    if not _is_valid_session_id(session_id):
        return False, None, ""

    path = _find_latest_session_log(session_id)
    if path is None:
        return False, None, ""

    with open(path, encoding="utf-8") as f:
        session = json.load(f)

    scoring = session.setdefault("scoring", {})
    scoring["self_report_score"] = round(self_report_score, 2)

    ai_score = scoring.get("ensemble_score")
    ai_classification = str(scoring.get("ensemble_classification", ""))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)

    if isinstance(ai_score, (int, float)):
        return True, float(ai_score), ai_classification
    return True, None, ai_classification


def administer() -> tuple[str, float, list[dict]]:
    """Run questionnaire interactively. Returns (user_id, score, responses)."""
    items = _load_items()

    print("\n" + "=" * 60)
    print("  IPIP Extraversion Self-Report Questionnaire")
    print("  Rate each statement from 1 to 5:")
    print("    1 = Very Inaccurate")
    print("    2 = Moderately Inaccurate")
    print("    3 = Neither Accurate Nor Inaccurate")
    print("    4 = Moderately Accurate")
    print("    5 = Very Accurate")
    print("=" * 60 + "\n")

    user_id = input("Enter your user ID (or press Enter for auto-generated): ").strip()
    if not user_id:
        user_id = str(uuid.uuid4())[:8]
    print(f"User ID: {user_id}\n")

    responses: list[dict] = []
    for i, item in enumerate(items, 1):
        while True:
            try:
                raw = int(input(f"  {i}. \"{item['text']}\"  [1-5]: "))
                if 1 <= raw <= 5:
                    break
                print("    Please enter a number between 1 and 5.")
            except ValueError:
                print("    Please enter a number between 1 and 5.")

        scored = raw if item["keying"] == "+" else _reverse_score(raw)
        responses.append(
            {
                "position": item["position"],
                "text": item["text"],
                "keying": item["keying"],
                "raw": raw,
                "scored": scored,
            }
        )

    domain_score = sum(r["scored"] for r in responses) / len(responses)
    return user_id, domain_score, responses


def save_result(
    user_id: str,
    self_report_score: float,
    ai_score: float | None = None,
    ai_classification: str = "",
) -> None:
    """Append a result row to pilot CSV."""
    file_exists = RESULTS_PATH.exists()

    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "self_report_score", "ai_score", "ai_classification"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "user_id": user_id,
                "self_report_score": round(self_report_score, 2),
                "ai_score": round(ai_score, 2) if ai_score is not None else "",
                "ai_classification": ai_classification,
            }
        )

    print(f"\nResult saved -> {RESULTS_PATH}")


def main() -> None:
    user_id, score, _responses = administer()

    band = classify_extraversion(score)
    label = f"{band} Extraversion"

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  User ID        : {user_id}")
    print(f"  Domain Score   : {score:.2f} / 5.00")
    print(f"  Classification : {label}")
    print("=" * 60)

    ai_score: float | None = None
    ai_classification = ""

    session_id = input(
        "\nOptional: enter interview session ID to link this score (or press Enter to skip): "
    ).strip()
    if session_id:
        if not _is_valid_session_id(session_id):
            print("Invalid session ID format. Use letters, numbers, underscore, or hyphen only.")
        else:
            attached, ai_score, ai_classification = attach_self_report_to_session(
                session_id,
                score,
            )
            if attached:
                print(f"Linked self-report to session {session_id}.")
            else:
                print(f"No session log found for session ID: {session_id}")

    save_it = input("\nSave to pilot results CSV? [Y/n]: ").strip().lower()
    if save_it in ("", "y", "yes"):
        save_result(user_id, score, ai_score=ai_score, ai_classification=ai_classification)


if __name__ == "__main__":
    main()
