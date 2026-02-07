"""Self-report evaluation — administer the IPIP Extraversion items as a
standard Likert-scale questionnaire via CLI.

Usage:
    python -m src.evaluation.self_report
    # or via pyproject entry-point:  self-report
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ipip_extraversion.json"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "pilot_results.csv"


def _load_items() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def _reverse_score(raw: int) -> int:
    """Reverse-score a 1-5 item: score = 6 - raw."""
    return 6 - raw


def administer() -> tuple[str, float, list[dict]]:
    """Run the questionnaire interactively. Returns (user_id, score, responses)."""
    items = _load_items()

    print("\n" + "═" * 60)
    print("  IPIP Extraversion Self-Report Questionnaire")
    print("  Rate each statement from 1 to 5:")
    print("    1 = Very Inaccurate")
    print("    2 = Moderately Inaccurate")
    print("    3 = Neither Accurate Nor Inaccurate")
    print("    4 = Moderately Accurate")
    print("    5 = Very Accurate")
    print("═" * 60 + "\n")

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
        responses.append({
            "position": item["position"],
            "text": item["text"],
            "keying": item["keying"],
            "raw": raw,
            "scored": scored,
        })

    domain_score = sum(r["scored"] for r in responses) / len(responses)
    return user_id, domain_score, responses


def save_result(user_id: str, self_report_score: float, ai_score: float | None = None) -> None:
    """Append a result row to the pilot CSV."""
    file_exists = RESULTS_PATH.exists()

    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "user_id", "self_report_score", "ai_score", "ai_classification",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "user_id": user_id,
            "self_report_score": round(self_report_score, 2),
            "ai_score": round(ai_score, 2) if ai_score is not None else "",
            "ai_classification": "",
        })

    print(f"\nResult saved → {RESULTS_PATH}")


def main() -> None:
    user_id, score, responses = administer()

    print("\n" + "═" * 60)
    print("  RESULTS")
    print("═" * 60)
    print(f"  User ID        : {user_id}")
    print(f"  Domain Score   : {score:.2f} / 5.00")
    if score <= 2.3:
        label = "Low Extraversion"
    elif score <= 3.6:
        label = "Medium Extraversion"
    else:
        label = "High Extraversion"
    print(f"  Classification : {label}")
    print("═" * 60)

    save_it = input("\nSave to pilot results CSV? [Y/n]: ").strip().lower()
    if save_it in ("", "y", "yes"):
        save_result(user_id, score)


if __name__ == "__main__":
    main()
