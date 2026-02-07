"""Correlation comparison — compute agreement between self-report and AI scores.

Usage:
    python -m src.evaluation.compare
    # or via pyproject entry-point:  compare

Reads data/pilot_results.csv and computes:
  • Pearson r  (linear correlation)
  • Spearman ρ (rank correlation — robust to non-linearity)
  • Mean absolute error (MAE)
  • Classification agreement (% matching Low/Medium/High label)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "pilot_results.csv"


def _classify(score: float) -> str:
    if score <= 2.3:
        return "Low"
    elif score <= 3.6:
        return "Medium"
    else:
        return "High"


def load_paired_scores() -> tuple[list[float], list[float], list[str]]:
    """Load rows that have both self_report and AI scores.

    Returns (self_scores, ai_scores, user_ids).
    """
    self_scores: list[float] = []
    ai_scores: list[float] = []
    user_ids: list[str] = []

    with open(RESULTS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sr = row.get("self_report_score", "").strip()
            ai = row.get("ai_score", "").strip()
            if sr and ai:
                self_scores.append(float(sr))
                ai_scores.append(float(ai))
                user_ids.append(row["user_id"])

    return self_scores, ai_scores, user_ids


def analyze() -> dict:
    """Compute correlation and agreement metrics."""
    self_scores, ai_scores, user_ids = load_paired_scores()
    n = len(self_scores)

    if n < 2:
        return {"error": f"Need at least 2 paired scores, found {n}."}

    sr = np.array(self_scores)
    ai = np.array(ai_scores)

    # Pearson r
    pearson_r, pearson_p = stats.pearsonr(sr, ai)

    # Spearman ρ
    spearman_rho, spearman_p = stats.spearmanr(sr, ai)

    # Mean Absolute Error
    mae = float(np.mean(np.abs(sr - ai)))

    # Classification agreement
    sr_labels = [_classify(s) for s in self_scores]
    ai_labels = [_classify(s) for s in ai_scores]
    agreement = sum(a == b for a, b in zip(sr_labels, ai_labels)) / n

    return {
        "n": n,
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 4),
        "spearman_rho": round(spearman_rho, 4),
        "spearman_p": round(spearman_p, 4),
        "mae": round(mae, 4),
        "classification_agreement": round(agreement, 4),
        "self_report_mean": round(float(np.mean(sr)), 2),
        "ai_mean": round(float(np.mean(ai)), 2),
        "self_report_std": round(float(np.std(sr, ddof=1)), 2),
        "ai_std": round(float(np.std(ai, ddof=1)), 2),
    }


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"No results file found at {RESULTS_PATH}")
        print("Run self-report and AI interviews first.")
        return

    results = analyze()

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("\n" + "═" * 60)
    print("  CORRELATION ANALYSIS — Self-Report vs. AI Scores")
    print("═" * 60)
    print(f"  Paired observations     : {results['n']}")
    print(f"  Self-report  mean (SD)  : {results['self_report_mean']} ({results['self_report_std']})")
    print(f"  AI score     mean (SD)  : {results['ai_mean']} ({results['ai_std']})")
    print()
    print(f"  Pearson r               : {results['pearson_r']:+.4f}   (p = {results['pearson_p']:.4f})")
    print(f"  Spearman ρ              : {results['spearman_rho']:+.4f}   (p = {results['spearman_p']:.4f})")
    print(f"  Mean Absolute Error     : {results['mae']:.4f}")
    print(f"  Classification Agreement: {results['classification_agreement']:.1%}")
    print("═" * 60)

    # Interpretation
    r = abs(results["pearson_r"])
    if r >= 0.7:
        interp = "Strong convergent validity — AI scores highly correlated with self-report."
    elif r >= 0.5:
        interp = "Moderate convergent validity — promising, but room for improvement."
    elif r >= 0.3:
        interp = "Weak convergent validity — some signal, but significant calibration needed."
    else:
        interp = "Poor convergent validity — AI scores do not reliably track self-report."

    print(f"\n  Interpretation: {interp}")
    print()


if __name__ == "__main__":
    main()
