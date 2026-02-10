"""Ensemble scorer — combines all scoring methods into a final assessment.

The ensemble approach is central to the project's scientific framing:
by comparing LLM-based, embedding-based, and feature-based scores,
we can assess convergent validity BETWEEN methods and identify which
approach (or combination) best approximates traditional self-report.

Fusion strategies:
  1. Confidence-weighted mean (default) — weight each method by its
     own confidence estimate
  2. Majority vote on classification — Low/Medium/High plurality
  3. Simple arithmetic mean (as a baseline)

The ensemble also runs all methods and stores individual results
for post-hoc analysis.
"""

from __future__ import annotations

from typing import Any

from src.extraction.features import LinguisticFeatures, extract_features
from src.scoring.embedding_scorer import score_with_embeddings
from src.scoring.feature_scorer import score_with_features
from src.scoring.llm_scorer import score_domain_level, score_facet_level
from src.settings import NEUTRAL_SCORE, classify_extraversion


def _majority_vote(classifications: list[str]) -> str:
    """Return the most common classification (ties → "Medium")."""
    counts: dict[str, int] = {}
    for c in classifications:
        counts[c] = counts.get(c, 0) + 1
    if not counts:
        return "Medium"
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    if len(winners) == 1:
        return winners[0]
    # Tie-breaking: prefer Medium > High > Low (conservative)
    for pref in ["Medium", "High", "Low"]:
        if pref in winners:
            return pref
    return "Medium"


def score_ensemble(
    transcript: str,
    features: LinguisticFeatures | None = None,
    run_llm: bool = True,
    run_embedding: bool = True,
    run_features: bool = True,
    run_facet_level: bool = False,
) -> dict[str, Any]:
    """Run all scoring methods and combine results.

    Parameters
    ----------
    transcript : str
        User-only text from the interview.
    features : LinguisticFeatures, optional
        Pre-extracted features. If None, will be extracted from transcript.
    run_llm : bool
        Whether to run LLM domain-level scorer (requires API).
    run_embedding : bool
        Whether to run embedding scorer (requires API).
    run_features : bool
        Whether to run feature-based scorer (local, no API).
    run_facet_level : bool
        Whether to also run per-facet LLM scoring (optional, extra API call).

    Returns
    -------
    dict
        {
            "ensemble_score": float,
            "ensemble_classification": str,
            "ensemble_confidence": float,
            "fusion_method": str,
            "individual_results": {
                "feature_based": {...},
                "embedding": {...},
                "llm_domain": {...},
                "llm_facet": {...},  # if requested
            },
            "classification_votes": {"Low": n, "Medium": n, "High": n},
        }
    """
    individual: dict[str, dict] = {}
    scores: list[float] = []
    confidences: list[float] = []
    classifications: list[str] = []

    # ── 1. Feature-based scoring (always local, fast) ─────────────────
    if run_features:
        if features is None:
            features = extract_features(transcript)
        feat_result = score_with_features(features)
        individual["feature_based"] = feat_result
        scores.append(feat_result["score"])
        confidences.append(feat_result["confidence"])
        classifications.append(feat_result["classification"])

    # ── 2. Embedding scoring (requires OpenAI API) ────────────────────
    if run_embedding:
        emb_result = score_with_embeddings(transcript)
        individual["embedding"] = emb_result
        if "error" not in emb_result:
            scores.append(emb_result["score"])
            confidences.append(emb_result["confidence"])
            classifications.append(emb_result["classification"])

    # ── 3. LLM domain-level scoring (requires OpenAI API) ────────────
    if run_llm:
        llm_result = score_domain_level(transcript)
        individual["llm_domain"] = llm_result
        if "error" not in llm_result:
            scores.append(llm_result["score"])
            confidences.append(llm_result["confidence"])
            classifications.append(llm_result["classification"])

    # ── 4. Optional: LLM facet-level scoring ──────────────────────────
    if run_facet_level:
        facet_result = score_facet_level(transcript)
        individual["llm_facet"] = facet_result

    # ── Fusion ────────────────────────────────────────────────────────
    if not scores:
        return {
            "ensemble_score": NEUTRAL_SCORE,
            "ensemble_classification": classify_extraversion(NEUTRAL_SCORE),
            "ensemble_confidence": 0.0,
            "fusion_method": "none",
            "individual_results": individual,
            "classification_votes": {},
            "warning": "No scoring methods produced results.",
        }

    # Confidence-weighted mean
    total_weight = sum(confidences)
    if total_weight > 0:
        weighted_score = sum(s * c for s, c in zip(scores, confidences, strict=True)) / total_weight
        fusion_method = "confidence_weighted_mean"
    else:
        weighted_score = sum(scores) / len(scores)
        fusion_method = "arithmetic_mean"

    ensemble_score = max(1.0, min(5.0, weighted_score))
    ensemble_classification = classify_extraversion(ensemble_score)

    # Ensemble confidence: mean of individual confidences,
    # boosted if methods agree
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    all_agree = len(set(classifications)) == 1
    agreement_bonus = 0.15 if all_agree else 0.0
    ensemble_confidence = min(1.0, mean_confidence + agreement_bonus)

    # Classification votes
    vote_counts: dict[str, int] = {}
    for c in classifications:
        vote_counts[c] = vote_counts.get(c, 0) + 1

    majority = _majority_vote(classifications)

    return {
        "ensemble_score": round(ensemble_score, 2),
        "ensemble_classification": ensemble_classification,
        "majority_vote_classification": majority,
        "ensemble_confidence": round(ensemble_confidence, 3),
        "fusion_method": fusion_method,
        "methods_used": len(scores),
        "methods_agree": all_agree,
        "individual_results": individual,
        "classification_votes": vote_counts,
        "individual_scores": {
            method: result.get("score", None)
            for method, result in individual.items()
            if method != "llm_facet"
        },
    }


def format_results(result: dict) -> str:
    """Format ensemble results for display.

    Parameters
    ----------
    result : dict
        Output of score_ensemble().

    Returns
    -------
    str
        Human-readable multi-line report.
    """
    lines = [
        "═" * 60,
        "  EXTRAVERSION ASSESSMENT — MULTI-METHOD RESULTS",
        "═" * 60,
        "",
        f"  ENSEMBLE SCORE:          {result['ensemble_score']:.2f} / 5.00",
        f"  CLASSIFICATION:          {result['ensemble_classification']}",
        f"  MAJORITY VOTE:           {result['majority_vote_classification']}",
        f"  CONFIDENCE:              {result['ensemble_confidence']:.1%}",
        f"  METHODS AGREE:           {'Yes' if result['methods_agree'] else 'No'}",
        f"  FUSION METHOD:           {result['fusion_method']}",
        "",
        "─" * 60,
        "  INDIVIDUAL METHOD SCORES",
        "─" * 60,
    ]

    for method, res in result.get("individual_results", {}).items():
        if method == "llm_facet":
            continue
        score = res.get("score", "N/A")
        cls = res.get("classification", "N/A")
        conf = res.get("confidence", 0)
        status = "✓" if "error" not in res and "warning" not in res else "⚠"
        if isinstance(score, (int, float)):
            lines.append(f"  {status} {method:20s}  {score:.2f}/5.0  {cls:8s}  conf={conf:.1%}")
        else:
            note = res.get("error", res.get("warning", ""))
            lines.append(f"  ⚠ {method:20s}  —        —         {note}")

    # Facet detail if available
    facet_data = result.get("individual_results", {}).get("llm_facet")
    if facet_data and facet_data.get("facet_scores"):
        lines.append("")
        lines.append("─" * 60)
        lines.append("  LLM FACET-LEVEL DETAIL (secondary)")
        lines.append("─" * 60)
        for fs in facet_data["facet_scores"]:
            bar = "█" * int(fs["score"]) + "░" * (5 - int(fs["score"]))
            lines.append(f"  {fs['facet_code']} {fs['facet_name']:20s} {fs['score']:.1f}  {bar}")

    # Classification votes
    votes = result.get("classification_votes", {})
    if votes:
        lines.append("")
        lines.append(f"  VOTES: {votes}")

    lines.append("")
    lines.append("═" * 60)
    return "\n".join(lines)


def explain_individual_result(result: dict) -> str:
    """Format a single scorer's result dict for human display.

    Dispatches to the per-method formatter based on ``result["method"]``.
    This is the single public entry-point; prefer it over importing
    ``explain_score`` from individual scorer modules.
    """
    from src.scoring.embedding_scorer import explain_score as _emb_explain
    from src.scoring.feature_scorer import explain_score as _feat_explain
    from src.scoring.llm_scorer import explain_score as _llm_explain

    method = result.get("method", "")
    if method == "feature_based":
        return _feat_explain(result)
    if method == "embedding":
        return _emb_explain(result)
    if method in ("llm_domain", "llm_facet"):
        return _llm_explain(result)
    return f"Unknown method: {method}"
