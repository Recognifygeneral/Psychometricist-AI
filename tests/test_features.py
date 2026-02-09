"""Tests for linguistic feature extraction module.

Validates that the feature extraction pipeline correctly identifies
word categories, computes ratios, and handles edge cases.
"""

from src.extraction.features import extract_features, extract_features_multi


class TestBasicExtraction:
    """Test core feature extraction on known texts."""

    def test_positive_emotion_detection(self):
        text = "I'm really happy and excited about this amazing opportunity!"
        features = extract_features(text)
        assert features.positive_emotion_count >= 3  # happy, excited, amazing
        assert features.positive_emotion_ratio > 0.0

    def test_negative_emotion_detection(self):
        text = "I feel sad and anxious about the terrible situation."
        features = extract_features(text)
        assert features.negative_emotion_count >= 3  # sad, anxious, terrible
        assert features.negative_emotion_ratio > 0.0

    def test_social_references(self):
        text = "I went to a party with my friends and met a lot of new people."
        features = extract_features(text)
        assert features.social_reference_count >= 3  # party, friends, people
        assert features.social_reference_ratio > 0.0

    def test_first_person_singular(self):
        text = "I think I should focus on my own goals and do it myself."
        features = extract_features(text)
        assert features.first_person_singular_count >= 3  # I, I, my, myself
        assert features.first_person_singular_ratio > 0.0

    def test_first_person_plural(self):
        text = "We decided to go together because our team works well as a unit."
        features = extract_features(text)
        assert features.first_person_plural_count >= 2  # we, our
        assert features.first_person_plural_ratio > 0.0

    def test_assertive_language(self):
        text = "I definitely know what I want and I will achieve my goals."
        features = extract_features(text)
        assert features.assertive_count >= 3  # definitely, know, will, achieve
        assert features.assertive_ratio > 0.0

    def test_hedging_language(self):
        text = "Maybe I could possibly look into that, I suppose it might work."
        features = extract_features(text)
        assert features.hedging_count >= 3  # maybe, could, possibly, suppose, might
        assert features.hedging_ratio > 0.0

    def test_exclamation_count(self):
        text = "Wow! That's amazing! I can't believe it!"
        features = extract_features(text)
        assert features.exclamation_count == 3

    def test_question_count(self):
        text = "What do you think? Should we go? Is it safe?"
        features = extract_features(text)
        assert features.question_count == 3


class TestDerivedMetrics:
    """Test computed metrics."""

    def test_word_count(self):
        text = "one two three four five"
        features = extract_features(text)
        assert features.word_count == 5

    def test_sentence_count(self):
        text = "First sentence. Second sentence. Third sentence."
        features = extract_features(text)
        assert features.sentence_count >= 3

    def test_lexical_diversity(self):
        # All unique words → TTR = 1.0
        text = "apple banana cherry durian elderberry"
        features = extract_features(text)
        assert features.lexical_diversity == 1.0

        # Repeated words → lower TTR
        text2 = "the the the the the cat"
        features2 = extract_features(text2)
        assert features2.lexical_diversity < 1.0

    def test_avg_sentence_length(self):
        text = "One two. Three four five six."
        features = extract_features(text)
        assert features.avg_sentence_length > 0

    def test_scoring_vector_keys(self):
        text = "Hello world, this is a test."
        features = extract_features(text)
        vec = features.scoring_vector()
        expected_keys = {
            "positive_emotion_ratio", "negative_emotion_ratio",
            "social_reference_ratio", "first_person_singular_ratio",
            "first_person_plural_ratio", "assertive_ratio", "hedging_ratio",
            "excitement_ratio", "exclamation_ratio", "avg_sentence_length",
            "lexical_diversity", "word_count",
        }
        assert set(vec.keys()) == expected_keys


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_string(self):
        features = extract_features("")
        assert features.word_count == 0
        assert features.positive_emotion_ratio == 0.0

    def test_none_input(self):
        features = extract_features(None)
        assert features.word_count == 0

    def test_whitespace_only(self):
        features = extract_features("   \n  \t  ")
        assert features.word_count == 0

    def test_single_word(self):
        features = extract_features("happy")
        assert features.word_count == 1
        assert features.positive_emotion_count == 1
        assert features.positive_emotion_ratio == 1.0

    def test_contractions(self):
        text = "I'm very happy and I've been doing well."
        features = extract_features(text)
        assert features.first_person_singular_count >= 2  # I'm, I've

    def test_to_dict(self):
        features = extract_features("Hello world")
        d = features.to_dict()
        assert isinstance(d, dict)
        assert "word_count" in d
        assert "positive_emotion_ratio" in d


class TestMultiTurn:
    """Test multi-turn extraction."""

    def test_extract_features_multi(self):
        turns = [
            "I love going out with friends!",
            "We always have such a great time together.",
        ]
        features = extract_features_multi(turns)
        assert features.word_count > 0
        assert features.positive_emotion_count > 0
        assert features.social_reference_count > 0

    def test_extract_empty_turns(self):
        features = extract_features_multi([])
        assert features.word_count == 0

    def test_extract_mixed_turns(self):
        turns = ["Hello!", "", "  ", "World!"]
        features = extract_features_multi(turns)
        assert features.word_count == 2


class TestHighVsLowExtraversion:
    """Validate that the features correctly differentiate
    prototypical high-E vs low-E text patterns."""

    HIGH_E_TEXT = (
        "I absolutely love going to parties and meeting new people! "
        "My friends always say I'm the life of the party. We go out "
        "every weekend and have the most amazing adventures. I'm "
        "always excited to try new things and I definitely don't "
        "mind being the center of attention!"
    )

    LOW_E_TEXT = (
        "I tend to keep to myself mostly. I suppose I might enjoy "
        "the occasional quiet evening at home with a book. I'm "
        "not really sure about large gatherings, they seem somewhat "
        "overwhelming perhaps. I usually prefer solitary activities."
    )

    def test_high_e_has_more_positive_emotion(self):
        high = extract_features(self.HIGH_E_TEXT)
        low = extract_features(self.LOW_E_TEXT)
        assert high.positive_emotion_ratio > low.positive_emotion_ratio

    def test_high_e_has_more_social_references(self):
        high = extract_features(self.HIGH_E_TEXT)
        low = extract_features(self.LOW_E_TEXT)
        assert high.social_reference_ratio > low.social_reference_ratio

    def test_low_e_has_more_hedging(self):
        high = extract_features(self.HIGH_E_TEXT)
        low = extract_features(self.LOW_E_TEXT)
        assert low.hedging_ratio > high.hedging_ratio

    def test_high_e_has_more_exclamations(self):
        high = extract_features(self.HIGH_E_TEXT)
        low = extract_features(self.LOW_E_TEXT)
        assert high.exclamation_count > low.exclamation_count
