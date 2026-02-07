"""Feature-based scorer — rule-based scoring using extracted linguistic features.

Maps numeric linguistic features to an Extraversion score using
empirically-grounded weights from personality-language research.

This scorer requires NO API calls — it runs entirely locally and is
fully deterministic, making it ideal as a reproducible baseline.

Weight sources and rationale:
  - Pennebaker & King (1999): positive emotion and social words
    predict Extraversion; hedging predicts Introversion/Neuroticism
  - Mairesse et al. (2007): assertive language, word count, and
    first-person plural pronouns correlate with Extraversion
  - Yarkoni (2010): positive emotion words r=.24, social words r=.22
  - Schwartz et al. (2013): "party", "love", excitement words
    are top unigram predictors of Extraversion on Facebook
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.extraction.features import LinguisticFeatures


@dataclass
class FeatureWeight:
    """A single feature's scoring parameters."""

    feature_name: str       # key in LinguisticFeatures.scoring_vector()
    neutral_baseline: float  # expected value for an average (3.0) person
    direction: float         # +1.0 = higher → more E; -1.0 = higher → less E
    weight: float            # magnitude of influence
    rationale: str           # brief scientific justification


# ═══════════════════════════════════════════════════════════════════════════
# SCORING WEIGHTS
# Each weight defines: (feature, neutral, direction, magnitude, rationale)
#
# The score formula is:
#   score = 3.0 + Σ (weight * direction * (feature_value - neutral))
#   score = clip(score, 1.0, 5.0)
#
# Weights are calibrated so typical interview features produce
# scores in the 2.0–4.0 range, with extremes reserved for
# very clear signals.
# ═══════════════════════════════════════════════════════════════════════════
WEIGHTS: list[FeatureWeight] = [
    FeatureWeight(
        "positive_emotion_ratio", 0.04, +1.0, 12.0,
        "Extraverts use ~50% more positive emotion words (r=.24, Yarkoni 2010)"
    ),
    FeatureWeight(
        "negative_emotion_ratio", 0.03, -1.0, 8.0,
        "Negative emotion inversely associated with E (r=-.15, Yarkoni 2010)"
    ),
    FeatureWeight(
        "social_reference_ratio", 0.04, +1.0, 14.0,
        "Social words are top predictors of E (r=.22, Pennebaker & King 1999)"
    ),
    FeatureWeight(
        "first_person_plural_ratio", 0.015, +1.0, 10.0,
        "We/us pronouns signal group orientation (Pennebaker & King 1999)"
    ),
    FeatureWeight(
        "assertive_ratio", 0.025, +1.0, 10.0,
        "Assertive language maps to E3 facet (Mairesse et al., 2007)"
    ),
    FeatureWeight(
        "hedging_ratio", 0.04, -1.0, 8.0,
        "Tentative/hedging language inversely correlated with E (r=-.18)"
    ),
    FeatureWeight(
        "excitement_ratio", 0.01, +1.0, 15.0,
        "Excitement/adventure words map to E5 facet (Schwartz et al., 2013)"
    ),
    FeatureWeight(
        "exclamation_ratio", 0.08, +1.0, 3.0,
        "Exclamation frequency correlates with enthusiasm/E6"
    ),
    FeatureWeight(
        "word_count", 25.0, +1.0, 0.008,
        "Response length: extraverts produce longer verbal output (Mairesse 2007)"
    ),
    FeatureWeight(
        "lexical_diversity", 0.65, +1.0, 2.0,
        "Vocabulary richness slightly associated with E (Yarkoni 2010)"
    ),
]


def _classify(score: float) -> str:
    """Classify a 1–5 score into Low / Medium / High."""
    if score <= 2.3:
        return "Low"
    elif score <= 3.6:
        return "Medium"
    else:
        return "High"


def _compute_confidence(score: float) -> float:
    """Estimate confidence based on distance from the neutral midpoint.

    Scores near 3.0 → low confidence (ambiguous).
    Scores near 1.0 or 5.0 → high confidence (clear signal).

    Returns a value in [0.0, 1.0].
    """
    distance = abs(score - 3.0)  # 0.0 – 2.0
    return min(1.0, distance / 1.5)  # saturates at ±1.5 from midpoint


def score_with_features(features: LinguisticFeatures) -> dict[str, Any]:
    """Score Extraversion using extracted linguistic features.

    Parameters
    ----------
    features : LinguisticFeatures
        Features extracted from the user's interview responses.

    Returns
    -------
    dict
        {
            "method": "feature_based",
            "score": float (1.0–5.0),
            "classification": "Low" | "Medium" | "High",
            "confidence": float (0.0–1.0),
            "feature_contributions": {feature_name: contribution, ...},
            "features_used": {feature_name: value, ...},
        }
    """
    if features.word_count == 0:
        return {
            "method": "feature_based",
            "score": 3.0,
            "classification": "Medium",
            "confidence": 0.0,
            "feature_contributions": {},
            "features_used": {},
            "warning": "No text to analyze — defaulting to neutral.",
        }

    scoring_vec = features.scoring_vector()
    contributions: dict[str, float] = {}
    total = 0.0

    for w in WEIGHTS:
        value = scoring_vec.get(w.feature_name, 0.0)
        contribution = w.weight * w.direction * (value - w.neutral_baseline)
        contributions[w.feature_name] = round(contribution, 4)
        total += contribution

    raw_score = 3.0 + total
    score = max(1.0, min(5.0, raw_score))
    classification = _classify(score)
    confidence = _compute_confidence(score)

    return {
        "method": "feature_based",
        "score": round(score, 2),
        "classification": classification,
        "confidence": round(confidence, 3),
        "feature_contributions": contributions,
        "features_used": {k: round(v, 4) for k, v in scoring_vec.items()},
        "raw_unclipped_score": round(raw_score, 4),
    }


def explain_score(result: dict) -> str:
    """Generate a human-readable explanation of the feature-based score.

    Parameters
    ----------
    result : dict
        Output of score_with_features().

    Returns
    -------
    str
        Multi-line explanation string.
    """
    lines = [
        f"Feature-Based Score: {result['score']:.2f}/5.0 → {result['classification']}",
        f"Confidence: {result['confidence']:.1%}",
        "",
        "Feature contributions (positive = more extraverted):",
    ]

    contribs = result.get("feature_contributions", {})
    # Sort by absolute magnitude
    sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)

    for name, contrib in sorted_contribs:
        if abs(contrib) < 0.001:
            continue
        direction = "↑" if contrib > 0 else "↓"
        value = result.get("features_used", {}).get(name, "?")
        lines.append(f"  {direction} {name}: {contrib:+.3f}  (raw: {value})")

    return "\n".join(lines)
