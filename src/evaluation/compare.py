"""Correlation comparison — compute agreement between self-report and AI scores.

Revised to support multi-method comparison:
  - Loads session logs from data/sessions/
  - Compares each scoring method against self-report
  - Also reads legacy pilot_results.csv for backward compatibility

Usage:
    python -m src.evaluation.compare
    # or via pyproject entry-point:  compare
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "pilot_results.csv"
SESSIONS_DIR = Path(__file__).resolve().parents[2] / "data" / "sessions"


def _classify(score: float) -> str:
    if score <= 2.3:
        return "Low"
    elif score <= 3.6:
        return "Medium"
    else:
        return "High"


def load_paired_scores() -> tuple[list[float], list[float], list[str]]:
    """Load rows that have both self_report and AI scores (legacy CSV).

    Returns (self_scores, ai_scores, user_ids).
    """
    self_scores: list[float] = []
    ai_scores: list[float] = []
    user_ids: list[str] = []

    if not RESULTS_PATH.exists():
        return self_scores, ai_scores, user_ids

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


def load_session_scores() -> dict[str, list[dict]]:
    """Load multi-method scores from session logs.

    Returns a dict mapping method names to lists of
    {session_id, score, self_report_score} dicts.
    """
    if not SESSIONS_DIR.exists():
        return {}

    method_data: dict[str, list[dict]] = {}

    for path in SESSIONS_DIR.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            session = json.load(f)

        scoring = session.get("scoring", {})
        sr = scoring.get("self_report_score")
        if sr is None:
            continue

        individual = scoring.get("individual_results", {})
        for method, result in individual.items():
            if method == "llm_facet":
                continue
            score = result.get("score")
            if score is not None:
                if method not in method_data:
                    method_data[method] = []
                method_data[method].append({
                    "session_id": session.get("session_id", ""),
                    "score": float(score),
                    "self_report_score": float(sr),
                })

        # Also record ensemble
        ens_score = scoring.get("ensemble_score")
        if ens_score is not None:
            if "ensemble" not in method_data:
                method_data["ensemble"] = []
            method_data["ensemble"].append({
                "session_id": session.get("session_id", ""),
                "score": float(ens_score),
                "self_report_score": float(sr),
            })

    return method_data


def _compute_metrics(self_scores: list[float], ai_scores: list[float]) -> dict:
    """Compute correlation and agreement metrics for a pair of score lists."""
    n = len(self_scores)
    if n < 2:
        return {"error": f"Need at least 2 paired scores, found {n}."}

    sr = np.array(self_scores)
    ai = np.array(ai_scores)

    pearson_r, pearson_p = stats.pearsonr(sr, ai)
    spearman_rho, spearman_p = stats.spearmanr(sr, ai)
    mae = float(np.mean(np.abs(sr - ai)))

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
        "self_report_std": round(float(np.std(sr, ddof=1)), 2) if n > 1 else 0.0,
        "ai_std": round(float(np.std(ai, ddof=1)), 2) if n > 1 else 0.0,
    }


def analyze() -> dict:
    """Compute correlation metrics (legacy CSV mode)."""
    self_scores, ai_scores, _ = load_paired_scores()
    return _compute_metrics(self_scores, ai_scores)


def analyze_multi_method() -> dict[str, dict]:
    """Compute correlation metrics for each scoring method.

    Returns {method_name: metrics_dict} for all methods with
    paired self-report data.
    """
    method_data = load_session_scores()
    results = {}

    for method, entries in method_data.items():
        sr = [e["self_report_score"] for e in entries]
        ai = [e["score"] for e in entries]
        results[method] = _compute_metrics(sr, ai)

    # Also include legacy CSV if available
    self_scores, ai_scores, _ = load_paired_scores()
    if len(self_scores) >= 2:
        results["legacy_csv"] = _compute_metrics(self_scores, ai_scores)

    return results


def _interpret_r(r: float) -> str:
    r = abs(r)
    if r >= 0.7:
        return "Strong convergent validity"
    elif r >= 0.5:
        return "Moderate convergent validity"
    elif r >= 0.3:
        return "Weak convergent validity"
    else:
        return "Poor convergent validity"


def main() -> None:
    # Try multi-method analysis first
    multi = analyze_multi_method()

    if multi:
        print("\n" + "═" * 70)
        print("  MULTI-METHOD CORRELATION ANALYSIS — Self-Report vs. AI")
        print("═" * 70)

        for method, metrics in multi.items():
            if "error" in metrics:
                print(f"\n  {method}: {metrics['error']}")
                continue

            print(f"\n  ── {method.upper()} {'─' * (50 - len(method))}")
            print(f"  N = {metrics['n']}")
            print(f"  Pearson r  = {metrics['pearson_r']:+.4f}  (p = {metrics['pearson_p']:.4f})")
            print(f"  Spearman ρ = {metrics['spearman_rho']:+.4f}  (p = {metrics['spearman_p']:.4f})")
            print(f"  MAE        = {metrics['mae']:.4f}")
            print(f"  Class. Agreement = {metrics['classification_agreement']:.1%}")
            print(f"  → {_interpret_r(metrics['pearson_r'])}")

        print("\n" + "═" * 70)
        return

    # Fall back to legacy CSV
    if not RESULTS_PATH.exists():
        print(f"No results found.")
        print(f"  - No session logs in {SESSIONS_DIR}")
        print(f"  - No CSV at {RESULTS_PATH}")
        print("Run interviews and self-reports first.")
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
    print(f"\n  → {_interpret_r(results['pearson_r'])}")
    print()


if __name__ == "__main__":
    main()
