"""Smoke tests — verify local graph, feature extraction, and import chain.

All tests are local (no API key required).
"""

from __future__ import annotations

from src.extraction.features import extract_features
from src.graph.local_graph import (
    get_all_data_for_scoring,
    get_all_probes,
    get_facets_for_trait,
    get_items_for_facet,
    get_linguistic_features,
    get_probes_for_facet,
    get_unused_probe,
)
from src.scoring.feature_scorer import score_with_features
from src.session.logger import SessionLogger
from src.workflow import MAX_TURNS, build_graph


class TestLocalGraph:
    """Verify the JSON-backed graph can load psychometric data."""

    def test_facets_loaded(self):
        facets = get_facets_for_trait()
        assert len(facets) == 6
        codes = {f["code"] for f in facets}
        assert codes == {"E1", "E2", "E3", "E4", "E5", "E6"}

    def test_probes_per_facet(self):
        for facet in get_facets_for_trait():
            probes = get_probes_for_facet(facet["code"])
            assert len(probes) >= 1

    def test_all_probes_flat_pool(self):
        all_probes = get_all_probes()
        assert len(all_probes) >= 10

    def test_unused_probe_logic(self):
        probe1 = get_unused_probe("E1", [])
        assert probe1 is not None
        probe2 = get_unused_probe("E1", [probe1["id"]])
        assert probe2 is None or probe2["id"] != probe1["id"]

    def test_items_per_facet(self):
        for facet in get_facets_for_trait():
            items = get_items_for_facet(facet["code"])
            assert isinstance(items, list)

    def test_linguistic_features_per_facet(self):
        for facet in get_facets_for_trait():
            feats = get_linguistic_features(facet["code"])
            assert isinstance(feats, list)

    def test_scoring_data_bundle(self):
        scoring = get_all_data_for_scoring()
        assert scoring["trait"] == "Extraversion"
        assert len(scoring["facets"]) == 6
        total_items = sum(len(f["items"]) for f in scoring["facets"])
        assert total_items >= 10


class TestImportChain:
    """Verify that key modules import without errors."""

    def test_feature_extraction(self):
        feats = extract_features(
            "I love meeting new people! We always have a great time together."
        )
        assert feats.word_count > 0
        assert feats.positive_emotion_count > 0
        assert feats.social_reference_count > 0

    def test_feature_scorer(self):
        feats = extract_features(
            "I love parties, friends, and exciting adventures! "
            "We go out every weekend."
        )
        result = score_with_features(feats)
        assert 1.0 <= result["score"] <= 5.0
        assert result["classification"] in ("Low", "Medium", "High")

    def test_workflow_compiles(self):
        assert MAX_TURNS == 10
        graph = build_graph()
        assert graph is not None

    def test_session_logger_instantiates(self):
        sl = SessionLogger("smoke-test")
        sl.log_turn(
            1,
            "How are you?",
            "I'm great, love people!",
            probe_id="smoke-probe",
            features=extract_features("I'm great, love people!"),
        )
        assert len(sl.turns) == 1

