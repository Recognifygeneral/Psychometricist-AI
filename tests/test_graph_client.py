"""Tests for the unified graph client dispatch and reset functions."""

from __future__ import annotations

from src.graph import graph_client, local_graph


class TestReset:
    """Verify reset functions clear cached state."""

    def test_graph_client_reset_clears_neo4j_flag(self):
        graph_client._USE_NEO4J = False
        graph_client.reset()
        assert graph_client._USE_NEO4J is None

    def test_local_graph_reset_clears_cache(self):
        # Populate the cache first
        local_graph._load()
        assert local_graph._CACHE is not None
        local_graph.reset()
        assert local_graph._CACHE is None


class TestDispatch:
    """Verify dispatch routes to local fallback when Neo4j is unavailable."""

    def test_get_facets_returns_six(self):
        graph_client.reset()
        facets = graph_client.get_facets_for_trait()
        assert len(facets) == 6

    def test_get_all_probes_returns_list(self):
        graph_client.reset()
        probes = graph_client.get_all_probes()
        assert len(probes) >= 10
        assert all("id" in p for p in probes)

    def test_get_probes_for_facet(self):
        graph_client.reset()
        probes = graph_client.get_probes_for_facet("E1")
        assert len(probes) >= 1

    def test_get_unused_probe(self):
        graph_client.reset()
        probe = graph_client.get_unused_probe("E1", [])
        assert probe is not None
        assert "id" in probe

    def test_get_items_for_facet(self):
        graph_client.reset()
        items = graph_client.get_items_for_facet("E1")
        assert isinstance(items, list)

    def test_get_linguistic_features(self):
        graph_client.reset()
        feats = graph_client.get_linguistic_features("E1")
        assert isinstance(feats, list)
