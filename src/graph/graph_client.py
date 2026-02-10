"""Unified graph interface selecting Neo4j or local JSON fallback."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_USE_NEO4J: bool | None = None


def reset() -> None:
    """Clear the cached backend decision — call from tests."""
    global _USE_NEO4J  # noqa: PLW0603
    _USE_NEO4J = None


def _check_neo4j() -> bool:
    """Try Neo4j connectivity once and cache the decision."""
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
        logger.info("[graph] Connected to Neo4j")
    except Exception as e:  # noqa: BLE001 — any failure → fall back to local JSON
        logger.warning("[graph] Neo4j unavailable (%s), using local JSON fallback.", e)
        _USE_NEO4J = False

    return _USE_NEO4J


def _dispatch(fn_name: str, *args: Any) -> Any:
    """Route a query to Neo4j (with driver) or the local JSON fallback.

    Both backends expose functions with the same name; the Neo4j variant
    takes ``driver`` as its first positional argument.
    """
    if _check_neo4j():
        from src.graph import neo4j_client

        with neo4j_client.get_driver() as driver:
            return getattr(neo4j_client, fn_name)(driver, *args)

    from src.graph import local_graph

    return getattr(local_graph, fn_name)(*args)


# ── Public API (each delegates to Neo4j or local JSON) ────────────────────


def get_facets_for_trait(trait_name: str = "Extraversion") -> list[dict]:
    """Return all facets for a personality trait."""
    return _dispatch("get_facets_for_trait", trait_name)


def get_probes_for_facet(facet_code: str) -> list[dict]:
    """Return all probes linked to a facet."""
    return _dispatch("get_probes_for_facet", facet_code)


def get_unused_probe(facet_code: str, used_ids: list[str]) -> dict | None:
    """Return a single unused probe for a facet, or None."""
    return _dispatch("get_unused_probe", facet_code, used_ids)


def get_items_for_facet(facet_code: str) -> list[dict]:
    """Return all IPIP items for a facet."""
    return _dispatch("get_items_for_facet", facet_code)


def get_linguistic_features(facet_code: str) -> list[dict]:
    """Return linguistic features associated with a facet."""
    return _dispatch("get_linguistic_features", facet_code)


def get_all_probes() -> list[dict]:
    """Return all probes as a flat list."""
    return _dispatch("get_all_probes")


def get_all_data_for_scoring(trait_name: str = "Extraversion") -> dict:
    if _check_neo4j():
        from src.graph.neo4j_client import get_all_data_for_scoring as _neo
        from src.graph.neo4j_client import get_driver

        with get_driver() as d:
            return _neo(d, trait_name)

    from src.graph.local_graph import get_all_data_for_scoring as _local

    return _local(trait_name)
