"""Embedding-based scorer — cosine similarity to Extraversion reference texts.

Encodes the user's interview responses and compares them against
curated reference descriptions of high- and low-Extraversion behavior.

Uses OpenAI's text-embedding-3-small model (same API key as the
Interviewer LLM). Falls back gracefully if the API is unavailable.

Scientific basis:
  Embedding similarity captures semantic content that goes beyond
  individual word matches. It can detect themes like "social energy",
  "excitement-seeking", or "quiet preference" even when expressed
  in novel language not covered by the word lists.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE TEXTS
# Curated vignettes covering the six Extraversion facets.
# Each represents a prototypical response pattern from a person
# at the given pole of the Extraversion dimension.
# ═══════════════════════════════════════════════════════════════════════════

HIGH_EXTRAVERSION_REFS: list[str] = [
    # E1 Friendliness / Warmth
    "I love meeting new people and I'm usually the one who walks up to "
    "strangers at a party to introduce myself. Making new friends comes "
    "naturally to me and I genuinely enjoy getting to know people from "
    "all walks of life. I feel really comfortable around people I've "
    "just met.",

    # E2 Gregariousness
    "My ideal weekend is packed with social activities. I love being "
    "surrounded by a big group of friends, whether we're going out to "
    "eat, hitting up a concert, or just hanging out at someone's place. "
    "I get restless when I spend too much time alone and I always prefer "
    "doing things with other people around.",

    # E3 Assertiveness
    "I naturally take charge in group situations. When a decision needs "
    "to be made, I'm usually the one who speaks up and proposes a plan. "
    "I feel confident expressing my opinions even when they're unpopular "
    "and I enjoy leading projects and organizing people.",

    # E4 Activity Level
    "I'm always on the go — my schedule is packed from morning to night "
    "and I love it that way. I have tons of energy and I can't stand "
    "sitting still for too long. My friends always say I'm the most "
    "active person they know. I juggle multiple hobbies and activities.",

    # E5 Excitement-Seeking
    "I'm always looking for the next adventure or thrill. I love trying "
    "new things — skydiving, traveling to exotic places, going to loud "
    "concerts. I get bored easily with routine and I'm drawn to "
    "anything that gets my adrenaline pumping. Life is too short to "
    "play it safe.",

    # E6 Cheerfulness / Positive Emotions
    "I'm a genuinely happy person — my friends always describe me as "
    "the cheerful one who lights up the room. I find joy in small things "
    "and I tend to see the bright side of every situation. I laugh a lot "
    "and I love making other people laugh too. My default mood is upbeat "
    "and optimistic.",

    # Mixed / general high-E
    "I absolutely love my social life — between work events, friend "
    "gatherings, family dinners, and weekend trips, I'm always surrounded "
    "by people. I draw energy from conversations and feel most alive "
    "when I'm in a lively group. I'm spontaneous, enthusiastic, and "
    "I rarely say no to a good time.",
]

LOW_EXTRAVERSION_REFS: list[str] = [
    # E1 (low) Reserve with strangers
    "I find it quite difficult to approach people I don't know. At "
    "parties or gatherings, I usually stick to the people I came with. "
    "It takes me a long time to open up to new people and I feel "
    "uncomfortable in situations where I'm expected to make small talk "
    "with strangers.",

    # E2 (low) Preference for solitude
    "I much prefer spending time alone or with just one or two close "
    "friends. Large social gatherings drain my energy and I need a lot "
    "of quiet time to recharge afterward. My ideal weekend involves "
    "staying home with a good book, watching a movie by myself, or "
    "going for a quiet walk alone.",

    # E3 (low) Deference / non-assertive
    "I tend to go along with what the group decides rather than "
    "voicing my own opinion. I don't enjoy being in charge and I'd "
    "rather someone else take the lead. I keep my opinions to myself "
    "most of the time, especially in groups where I might be "
    "disagreed with.",

    # E4 (low) Low activity / calm pace
    "I have a pretty relaxed pace of life. I don't feel the need to "
    "constantly be doing things and I'm perfectly content with a slow, "
    "quiet day. My friends might call me lazy but I just don't have "
    "that restless energy that some people have. I value stillness.",

    # E5 (low) Comfort-seeking / risk-averse
    "I prefer familiar routines and predictable situations. New or "
    "unfamiliar experiences make me anxious rather than excited. I "
    "don't seek out thrills or adrenaline — I'd much rather do "
    "something I've done before that I know I enjoy. I'm cautious "
    "and I think carefully before trying anything new.",

    # E6 (low) Subdued / flat affect
    "I wouldn't describe myself as a particularly cheerful person. "
    "I don't get overly excited about things and my emotional range "
    "is fairly narrow. I'm more of a calm, even-keeled person. "
    "My friends wouldn't call me the life of the party by any stretch.",

    # Mixed / general low-E
    "I'm a private person who keeps to myself. I have a small circle "
    "of close friends and that's enough for me. I find social situations "
    "tiring and I often prefer my own company. I'm quiet, reflective, "
    "and I think before I speak. I rarely draw attention to myself.",
]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _classify(score: float) -> str:
    if score <= 2.3:
        return "Low"
    elif score <= 3.6:
        return "Medium"
    else:
        return "High"


def _get_embeddings_model():
    """Lazily import and create the OpenAI embeddings model."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def score_with_embeddings(
    transcript: str,
    min_words: int = 15,
) -> dict[str, Any]:
    """Score Extraversion using embedding cosine similarity.

    Parameters
    ----------
    transcript : str
        The full user transcript (concatenated interview responses).
    min_words : int
        Minimum word count for reliable scoring.

    Returns
    -------
    dict
        {
            "method": "embedding",
            "score": float (1.0–5.0),
            "classification": "Low" | "Medium" | "High",
            "confidence": float (0.0–1.0),
            "high_similarity": float,
            "low_similarity": float,
        }
    """
    word_count = len(transcript.split())
    if word_count < min_words:
        return {
            "method": "embedding",
            "score": 3.0,
            "classification": "Medium",
            "confidence": 0.0,
            "high_similarity": 0.0,
            "low_similarity": 0.0,
            "warning": f"Transcript too short ({word_count} words < {min_words}). "
                       "Defaulting to neutral.",
        }

    try:
        model = _get_embeddings_model()

        # Embed user transcript
        user_emb = np.array(model.embed_query(transcript))

        # Embed reference texts
        high_embs = [np.array(e) for e in model.embed_documents(HIGH_EXTRAVERSION_REFS)]
        low_embs = [np.array(e) for e in model.embed_documents(LOW_EXTRAVERSION_REFS)]

        # Mean cosine similarity to each pole
        high_sim = float(np.mean([_cosine_similarity(user_emb, h) for h in high_embs]))
        low_sim = float(np.mean([_cosine_similarity(user_emb, l) for l in low_embs]))

        # Convert relative similarity to 1–5 score
        # balance ∈ [-1, 1]: positive → closer to high-E
        denom = high_sim + low_sim
        if denom < 1e-8:
            balance = 0.0
        else:
            balance = (high_sim - low_sim) / denom

        # Map to [1, 5] with amplification
        # Empirically, cosine similarities are often close so we amplify
        amplified = balance * 3.0  # expand the effective range
        raw_score = 3.0 + amplified
        score = max(1.0, min(5.0, raw_score))

        classification = _classify(score)
        confidence = min(1.0, abs(amplified) / 1.5)

        return {
            "method": "embedding",
            "score": round(score, 2),
            "classification": classification,
            "confidence": round(confidence, 3),
            "high_similarity": round(high_sim, 4),
            "low_similarity": round(low_sim, 4),
            "balance": round(balance, 4),
        }

    except Exception as e:
        return {
            "method": "embedding",
            "score": 3.0,
            "classification": "Medium",
            "confidence": 0.0,
            "error": f"Embedding scorer failed: {str(e)}",
        }


def explain_score(result: dict) -> str:
    """Human-readable explanation."""
    if "error" in result:
        return f"Embedding Scorer: FAILED — {result['error']}"
    if "warning" in result:
        return f"Embedding Scorer: {result['warning']}"

    lines = [
        f"Embedding-Based Score: {result['score']:.2f}/5.0 → {result['classification']}",
        f"Confidence: {result['confidence']:.1%}",
        f"Similarity to High-E references: {result['high_similarity']:.4f}",
        f"Similarity to Low-E references:  {result['low_similarity']:.4f}",
        f"Balance (positive = extraverted):  {result.get('balance', 0):.4f}",
    ]
    return "\n".join(lines)
