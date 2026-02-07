"""Unified graph interface — auto-selects Neo4j or local JSON fallback.

Usage:
    from src.graph import graph_client
    facets = graph_client.get_facets_for_trait()
    probe  = graph_client.get_unused_probe("E1", used_ids=[])

If NEO4J_URI is set in .env and the connection succeeds, uses Neo4j.
Otherwise, falls back to the local JSON file (zero infrastructure).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()

_USE_NEO4J: bool | None = None  # lazy-init


def _check_neo4j() -> bool:
    """Try to connect to Neo4j. Returns True if successful."""
    global _USE_NEO4J
    if _USE_NEO4J is not None:
        return _USE_NEO4J

    uri = os.getenv("NEO4J_URI", "")
    if not uri or uri.startswith("neo4j+s://xxxxxxxx"):
        _USE_NEO4J = False
        return False

    try:
        from src.graph.neo4j_client import get_driver

        with get_driver() as driver:
            driver.verify_connectivity()
        _USE_NEO4J = True
        print("[graph] Connected to Neo4j ✓")
    except Exception as e:
        print(f"[graph] Neo4j unavailable ({e}), using local JSON fallback.")
        _USE_NEO4J = False

    return _USE_NEO4J


# ── Unified API ───────────────────────────────────────────────────────────


def get_facets_for_trait(trait_name: str = "Extraversion") -> list[dict]:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_facets_for_trait as _neo

        with get_driver() as d:
            return _neo(d, trait_name)
    else:
        from src.graph.local_graph import get_facets_for_trait as _local

        return _local(trait_name)


def get_probes_for_facet(facet_code: str) -> list[dict]:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_probes_for_facet as _neo

        with get_driver() as d:
            return _neo(d, facet_code)
    else:
        from src.graph.local_graph import get_probes_for_facet as _local

        return _local(facet_code)


def get_unused_probe(facet_code: str, used_ids: list[str]) -> dict | None:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_unused_probe as _neo

        with get_driver() as d:
            return _neo(d, facet_code, used_ids)
    else:
        from src.graph.local_graph import get_unused_probe as _local

        return _local(facet_code, used_ids)


def get_items_for_facet(facet_code: str) -> list[dict]:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_items_for_facet as _neo

        with get_driver() as d:
            return _neo(d, facet_code)
    else:
        from src.graph.local_graph import get_items_for_facet as _local

        return _local(facet_code)


def get_linguistic_features(facet_code: str) -> list[dict]:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_linguistic_features as _neo

        with get_driver() as d:
            return _neo(d, facet_code)
    else:
        from src.graph.local_graph import get_linguistic_features as _local

        return _local(facet_code)


def get_all_probes() -> list[dict]:
    """Return ALL probes as a flat list (for simplified probe selection)."""
    if _check_neo4j():
        # Flatten all facets' probes
        from src.graph.neo4j_client import get_driver, get_facets_for_trait as _neo_facets
        from src.graph.neo4j_client import get_probes_for_facet as _neo_probes

        with get_driver() as d:
            facets = _neo_facets(d)
            all_probes = []
            for f in facets:
                all_probes.extend(_neo_probes(d, f["code"]))
            return all_probes
    else:
        from src.graph.local_graph import get_all_probes as _local

        return _local()


def get_all_data_for_scoring(trait_name: str = "Extraversion") -> dict:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_all_data_for_scoring as _neo

        with get_driver() as d:
            return _neo(d, trait_name)
    else:
        from src.graph.local_graph import get_all_data_for_scoring as _local

        return _local(trait_name)
