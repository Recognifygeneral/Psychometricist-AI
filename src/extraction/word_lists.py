"""Curated word lists for linguistic feature extraction.

Sources:
  - LIWC 2015 categories (Pennebaker et al., 2015)
  - NRC Emotion Lexicon (Mohammad & Turney, 2013)
  - Mairesse et al. (2007) — personality-language markers
  - Extraversion-specific linguistic correlates from:
      Pennebaker & King (1999), Yarkoni (2010), Schwartz et al. (2013)

These lists are intentionally broad. False positives are acceptable
because the *ratio* is what matters, and noise averages out over a
full interview transcript.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# POSITIVE EMOTION WORDS
# High-E speakers use significantly more positive emotion language
# (Pennebaker & King, 1999; Mairesse et al., 2007)
# ═══════════════════════════════════════════════════════════════════════════
POSITIVE_EMOTION: frozenset[str] = frozenset({
    # Joy / happiness
    "happy", "happiness", "glad", "joyful", "joy", "delighted", "pleased",
    "cheerful", "merry", "blissful", "ecstatic", "elated", "euphoric",
    "thrilled", "overjoyed", "content", "satisfied", "fulfilled",
    # Enthusiasm / energy
    "excited", "exciting", "excitement", "enthusiastic", "passionate",
    "eager", "energetic", "energized", "pumped", "stoked", "hyped",
    "inspired", "motivated", "driven", "alive", "vibrant", "dynamic",
    # Positive evaluations
    "great", "amazing", "awesome", "fantastic", "wonderful", "brilliant",
    "excellent", "superb", "marvelous", "terrific", "incredible",
    "outstanding", "magnificent", "phenomenal", "spectacular", "glorious",
    "perfect", "beautiful", "gorgeous", "lovely", "nice", "good",
    "cool", "sweet", "neat", "rad", "sick",  # colloquial positives
    # Affection / warmth
    "love", "loved", "loving", "adore", "cherish", "treasure",
    "care", "caring", "warm", "warmth", "kind", "tender", "fond",
    "affectionate", "compassionate", "generous", "gentle",
    # Gratitude / optimism
    "grateful", "thankful", "blessed", "fortunate", "lucky",
    "optimistic", "hopeful", "positive", "upbeat", "bright",
    "promising", "encouraged", "confident", "proud",
    # Fun / enjoyment
    "fun", "funny", "hilarious", "enjoy", "enjoying", "enjoyable",
    "entertaining", "amusing", "playful", "laugh", "laughing", "laughter",
    "humor", "humorous", "wit", "witty", "giggle", "smile", "smiling",
    "grin", "beam", "beaming", "celebrate", "celebration", "party",
})

# ═══════════════════════════════════════════════════════════════════════════
# NEGATIVE EMOTION WORDS
# Low-E / high-N speakers use more negative emotion language
# ═══════════════════════════════════════════════════════════════════════════
NEGATIVE_EMOTION: frozenset[str] = frozenset({
    # Sadness
    "sad", "sadness", "unhappy", "miserable", "depressed", "depressing",
    "gloomy", "melancholy", "somber", "bleak", "desolate", "despair",
    "hopeless", "helpless", "heartbroken", "grief", "grieving",
    "mourning", "sorrowful", "dejected", "disheartened", "downcast",
    # Anger / frustration
    "angry", "anger", "furious", "enraged", "irritated", "annoyed",
    "frustrated", "frustrating", "frustration", "mad", "livid",
    "outraged", "resentful", "bitter", "hostile", "aggravated",
    # Anxiety / fear
    "anxious", "anxiety", "worried", "worry", "nervous", "scared",
    "afraid", "fearful", "terrified", "panicked", "dread", "dreading",
    "uneasy", "tense", "stressed", "stress", "stressful", "overwhelmed",
    "apprehensive", "insecure",
    # General negative
    "terrible", "awful", "horrible", "dreadful", "disgusting",
    "hate", "hated", "hatred", "loathe", "detest", "despise",
    "bored", "boring", "tedious", "monotonous", "dull",
    "lonely", "loneliness", "isolated", "alone",
    "tired", "exhausted", "drained", "fatigued", "weary",
    "disappointed", "disappointing", "disappointment",
    "embarrassed", "ashamed", "guilty", "regret", "regretful",
    "disgusted", "repulsed", "bothered", "troubled", "disturbed",
    "pessimistic", "cynical", "doubtful",
})

# ═══════════════════════════════════════════════════════════════════════════
# SOCIAL REFERENCE WORDS
# Extraverts reference social contexts, people, and group activities
# significantly more often (Pennebaker & King, 1999; Schwartz et al., 2013)
# ═══════════════════════════════════════════════════════════════════════════
SOCIAL_REFERENCES: frozenset[str] = frozenset({
    # People
    "people", "person", "someone", "everyone", "everybody", "anyone",
    "somebody", "nobody", "crowd", "audience", "public",
    # Relationships
    "friend", "friends", "friendship", "buddy", "buddies", "pal", "pals",
    "mate", "mates", "companion", "companions", "acquaintance",
    "neighbor", "neighbors", "neighbour", "neighbours",
    "colleague", "colleagues", "coworker", "coworkers",
    "classmate", "classmates", "roommate", "roommates",
    "partner", "boyfriend", "girlfriend", "spouse", "husband", "wife",
    "family", "mom", "dad", "mother", "father", "brother", "sister",
    "son", "daughter", "kids", "children", "parents",
    # Social activities
    "party", "parties", "gathering", "gatherings", "event", "events",
    "meetup", "hangout", "outing", "reunion", "barbecue", "dinner",
    "lunch", "brunch", "drinks", "clubbing", "dancing",
    "festival", "concert", "game", "games",
    # Social verbs
    "meet", "meeting", "met", "socialize", "socializing", "mingle",
    "mingling", "chat", "chatting", "talk", "talking", "talked",
    "conversation", "conversations", "discuss", "discussing",
    "hang", "hanging", "invite", "invited", "join", "joined",
    "visit", "visiting", "visited", "host", "hosting", "hosted",
    "share", "sharing", "shared", "connect", "connecting",
    # Group references
    "group", "groups", "team", "teams", "club", "clubs",
    "community", "communities", "organization", "society",
    "together", "company",  # as in "enjoy company"
})

# ═══════════════════════════════════════════════════════════════════════════
# FIRST PERSON SINGULAR PRONOUNS
# Research is mixed: slightly associated with introversion/neuroticism
# (Pennebaker & King, 1999). Included for completeness.
# ═══════════════════════════════════════════════════════════════════════════
FIRST_PERSON_SINGULAR: frozenset[str] = frozenset({
    "i", "me", "my", "mine", "myself",
    # Common contractions
    "i'm", "i've", "i'll", "i'd",
})

# ═══════════════════════════════════════════════════════════════════════════
# FIRST PERSON PLURAL PRONOUNS
# Extraverts use more "we/us" — reflects group orientation
# (Pennebaker & King, 1999)
# ═══════════════════════════════════════════════════════════════════════════
FIRST_PERSON_PLURAL: frozenset[str] = frozenset({
    "we", "us", "our", "ours", "ourselves",
    # Common contractions
    "we're", "we've", "we'll", "we'd",
})

# ═══════════════════════════════════════════════════════════════════════════
# ASSERTIVE / DOMINANT LANGUAGE
# Extraverts (especially high-E3 Assertiveness) use more confident,
# decisive language (Mairesse et al., 2007)
# ═══════════════════════════════════════════════════════════════════════════
ASSERTIVE_LANGUAGE: frozenset[str] = frozenset({
    # Certainty markers
    "definitely", "certainly", "absolutely", "obviously", "clearly",
    "undoubtedly", "undeniably", "unquestionably", "surely", "indeed",
    "always", "never", "must", "shall", "will",
    # Confidence
    "confident", "confidence", "sure", "certain", "convinced",
    "know", "believe", "assert", "insist", "declare", "state",
    "demand", "require", "decide", "decided", "determined",
    # Leadership / initiative
    "lead", "leading", "leader", "initiative", "organize", "organized",
    "manage", "managed", "direct", "directed", "command", "charge",
    "responsibility", "responsible", "accountable",
    # Strength
    "strong", "bold", "brave", "courageous", "fearless", "powerful",
    "capable", "competent", "skilled", "accomplished", "successful",
    "achieve", "achieved", "achievement", "accomplish", "win", "won", "victory", "triumph", "excel", "excelled",
})

# ═══════════════════════════════════════════════════════════════════════════
# HEDGING / TENTATIVE LANGUAGE
# Introverts and neurotic individuals hedge more (Pennebaker & King, 1999)
# ═══════════════════════════════════════════════════════════════════════════
HEDGING_LANGUAGE: frozenset[str] = frozenset({
    # Possibility / uncertainty
    "maybe", "perhaps", "possibly", "probably", "might", "could",
    "somewhat", "fairly", "rather", "quite",
    # Qualifiers
    "sometimes", "occasionally", "rarely", "seldom", "hardly",
    "slightly", "barely", "almost", "nearly", "approximately",
    # Softeners
    "guess", "suppose", "wonder", "seem", "seems", "seemed",
    "appear", "appears", "appeared", "tend", "tends", "tended",
    "likely", "unlikely", "uncertain", "unsure",
    # Verbal hedges (multi-word handled separately)
    "kinda", "sorta",
    # Tentativeness
    "hesitant", "tentative", "cautious", "careful", "wary",
    "reluctant", "doubtful", "skeptical", "sceptical",
    "honestly", "actually", "basically", "literally",  # discourse markers
})

# ═══════════════════════════════════════════════════════════════════════════
# EXCITEMENT / STIMULATION WORDS
# High-E5 (Excitement-Seeking) speakers reference adventure,
# risk, novelty, and stimulation
# ═══════════════════════════════════════════════════════════════════════════
EXCITEMENT_WORDS: frozenset[str] = frozenset({
    "adventure", "adventurous", "thrill", "thrilling", "exciting",
    "excitement", "adrenaline", "rush", "exhilarating",
    "risk", "risky", "dare", "daring", "bold", "wild",
    "spontaneous", "spontaneity", "impulsive", "impulse",
    "explore", "exploring", "exploration", "discover", "discovery",
    "travel", "traveling", "travelling", "trip", "journey",
    "new", "novel", "novelty", "different", "unique", "unusual",
    "extreme", "intense", "intensity", "fast", "speed", "racing",
    "challenge", "challenging", "compete", "competition", "competitive",
})

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-WORD HEDGE PHRASES
# Checked via substring matching on the lowercased text
# ═══════════════════════════════════════════════════════════════════════════
HEDGE_PHRASES: tuple[str, ...] = (
    "kind of", "sort of", "a little", "a bit",
    "i guess", "i think", "i suppose", "i wonder",
    "not sure", "not certain", "don't know", "don't really",
    "to be honest", "i mean", "you know",
    "more or less", "in a way", "so to speak",
)

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-WORD ASSERTIVE PHRASES
# ═══════════════════════════════════════════════════════════════════════════
ASSERTIVE_PHRASES: tuple[str, ...] = (
    "i know", "i believe", "without a doubt", "no question",
    "for sure", "of course", "no doubt", "make sure",
    "take charge", "step up", "speak up", "stand up",
    "right away", "let's go", "let's do",
)
