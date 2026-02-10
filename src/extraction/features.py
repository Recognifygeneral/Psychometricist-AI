"""Linguistic feature extraction from interview responses.

Extracts numeric features from user text that are empirically associated
with Extraversion (Pennebaker & King, 1999; Mairesse et al., 2007;
Yarkoni, 2010; Schwartz et al., 2013).

Usage:
    from src.extraction.features import extract_features

    features = extract_features("I love going to parties with friends!")
    # → {"word_count": 8, "positive_emotion_ratio": 0.125, ...}

All ratio features are word-count-normalized (0.0–1.0 range).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from src.extraction.word_lists import (
    ASSERTIVE_LANGUAGE,
    ASSERTIVE_PHRASES,
    EXCITEMENT_WORDS,
    FIRST_PERSON_PLURAL,
    FIRST_PERSON_SINGULAR,
    HEDGE_PHRASES,
    HEDGING_LANGUAGE,
    NEGATIVE_EMOTION,
    POSITIVE_EMOTION,
    SOCIAL_REFERENCES,
)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, preserving contractions."""
    # Normalize curly/smart quotes to straight apostrophes
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text.lower())


def _count_sentences(text: str) -> int:
    """Count sentences using punctuation-based heuristic."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'[.!?]+(?:\s|$)', text.strip())
    # Filter out empty strings
    return max(1, len([s for s in sentences if s.strip()]))


def _count_matches(words: list[str], word_set: frozenset[str]) -> int:
    """Count how many words appear in the given set."""
    return sum(1 for w in words if w in word_set)


def _count_phrase_matches(text: str, phrases: tuple[str, ...]) -> int:
    """Count occurrences of multi-word phrases in the text."""
    lower = text.lower()
    return sum(lower.count(phrase) for phrase in phrases)


@dataclass
class LinguisticFeatures:
    """Container for all extracted linguistic features.

    All ratio fields are normalized by word count (0.0–1.0).
    Raw counts are also available for transparency.
    """

    # ── Raw counts ────────────────────────────────────────────────────
    word_count: int = 0
    sentence_count: int = 0
    unique_word_count: int = 0

    # Punctuation counts
    exclamation_count: int = 0
    question_count: int = 0

    # Category hit counts
    positive_emotion_count: int = 0
    negative_emotion_count: int = 0
    social_reference_count: int = 0
    first_person_singular_count: int = 0
    first_person_plural_count: int = 0
    assertive_count: int = 0
    hedging_count: int = 0
    excitement_count: int = 0

    # Phrase counts (multi-word)
    hedge_phrase_count: int = 0
    assertive_phrase_count: int = 0

    # ── Derived ratios ────────────────────────────────────────────────
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    lexical_diversity: float = 0.0  # unique words / total words (TTR)

    positive_emotion_ratio: float = 0.0
    negative_emotion_ratio: float = 0.0
    social_reference_ratio: float = 0.0
    first_person_singular_ratio: float = 0.0
    first_person_plural_ratio: float = 0.0
    assertive_ratio: float = 0.0
    hedging_ratio: float = 0.0
    excitement_ratio: float = 0.0
    exclamation_ratio: float = 0.0
    question_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for JSON serialization."""
        return asdict(self)

    def scoring_vector(self) -> dict[str, float]:
        """Return only the ratio features used for scoring."""
        return {
            "positive_emotion_ratio": self.positive_emotion_ratio,
            "negative_emotion_ratio": self.negative_emotion_ratio,
            "social_reference_ratio": self.social_reference_ratio,
            "first_person_singular_ratio": self.first_person_singular_ratio,
            "first_person_plural_ratio": self.first_person_plural_ratio,
            "assertive_ratio": self.assertive_ratio,
            "hedging_ratio": self.hedging_ratio,
            "excitement_ratio": self.excitement_ratio,
            "exclamation_ratio": self.exclamation_ratio,
            "avg_sentence_length": self.avg_sentence_length,
            "lexical_diversity": self.lexical_diversity,
            "word_count": float(self.word_count),
        }


def extract_features(text: str) -> LinguisticFeatures:
    """Extract all linguistic features from a text string.

    Parameters
    ----------
    text : str
        The raw user text (single turn or full transcript).

    Returns
    -------
    LinguisticFeatures
        Dataclass with all raw counts and derived ratios.
    """
    if not text or not text.strip():
        return LinguisticFeatures()

    words = _tokenize(text)
    word_count = len(words)

    if word_count == 0:
        return LinguisticFeatures()

    sentence_count = _count_sentences(text)
    unique_words = set(words)
    unique_word_count = len(unique_words)

    # ── Punctuation ───────────────────────────────────────────────────
    exclamation_count = text.count("!")
    question_count = text.count("?")

    # ── Category word counts ──────────────────────────────────────────
    positive_emotion_count = _count_matches(words, POSITIVE_EMOTION)
    negative_emotion_count = _count_matches(words, NEGATIVE_EMOTION)
    social_reference_count = _count_matches(words, SOCIAL_REFERENCES)
    first_person_singular_count = _count_matches(words, FIRST_PERSON_SINGULAR)
    first_person_plural_count = _count_matches(words, FIRST_PERSON_PLURAL)
    assertive_count = _count_matches(words, ASSERTIVE_LANGUAGE)
    hedging_count = _count_matches(words, HEDGING_LANGUAGE)
    excitement_count = _count_matches(words, EXCITEMENT_WORDS)

    # ── Multi-word phrase matches ─────────────────────────────────────
    hedge_phrase_count = _count_phrase_matches(text, HEDGE_PHRASES)
    assertive_phrase_count = _count_phrase_matches(text, ASSERTIVE_PHRASES)

    # Add phrase counts to single-word counts for combined ratios
    total_hedge = hedging_count + hedge_phrase_count
    total_assertive = assertive_count + assertive_phrase_count

    # ── Derived metrics ───────────────────────────────────────────────
    avg_word_length = sum(len(w.replace("'", "")) for w in words) / word_count
    avg_sentence_length = word_count / sentence_count
    lexical_diversity = unique_word_count / word_count

    # ── Ratios (normalized by word count) ─────────────────────────────
    def _ratio(count: int) -> float:
        return count / word_count

    return LinguisticFeatures(
        # Raw counts
        word_count=word_count,
        sentence_count=sentence_count,
        unique_word_count=unique_word_count,
        exclamation_count=exclamation_count,
        question_count=question_count,
        positive_emotion_count=positive_emotion_count,
        negative_emotion_count=negative_emotion_count,
        social_reference_count=social_reference_count,
        first_person_singular_count=first_person_singular_count,
        first_person_plural_count=first_person_plural_count,
        assertive_count=total_assertive,
        hedging_count=total_hedge,
        excitement_count=excitement_count,
        hedge_phrase_count=hedge_phrase_count,
        assertive_phrase_count=assertive_phrase_count,
        # Derived
        avg_word_length=round(avg_word_length, 3),
        avg_sentence_length=round(avg_sentence_length, 2),
        lexical_diversity=round(lexical_diversity, 3),
        # Ratios
        positive_emotion_ratio=round(_ratio(positive_emotion_count), 4),
        negative_emotion_ratio=round(_ratio(negative_emotion_count), 4),
        social_reference_ratio=round(_ratio(social_reference_count), 4),
        first_person_singular_ratio=round(_ratio(first_person_singular_count), 4),
        first_person_plural_ratio=round(_ratio(first_person_plural_count), 4),
        assertive_ratio=round(_ratio(total_assertive), 4),
        hedging_ratio=round(_ratio(total_hedge), 4),
        excitement_ratio=round(_ratio(excitement_count), 4),
        exclamation_ratio=round(exclamation_count / sentence_count, 4),
        question_ratio=round(question_count / sentence_count, 4),
    )


def extract_features_multi(turns: list[str]) -> LinguisticFeatures:
    """Extract features from the full concatenated transcript.

    Parameters
    ----------
    turns : list[str]
        Individual user messages from the interview.

    Returns
    -------
    LinguisticFeatures
        Features extracted from the concatenated text.
    """
    combined = " ".join(t.strip() for t in turns if t.strip())
    return extract_features(combined)


def aggregate_turn_features(turn_features: list[LinguisticFeatures]) -> dict[str, float]:
    """Compute mean feature values across all turns.

    Useful for seeing per-turn trends vs. whole-transcript features.

    Parameters
    ----------
    turn_features : list[LinguisticFeatures]
        Features extracted from each individual turn.

    Returns
    -------
    dict[str, float]
        Mean value of each scoring feature.
    """
    if not turn_features:
        return {}

    n = len(turn_features)
    vectors = [tf.scoring_vector() for tf in turn_features]
    keys = vectors[0].keys()

    return {
        key: round(sum(v[key] for v in vectors) / n, 4)
        for key in keys
    }
