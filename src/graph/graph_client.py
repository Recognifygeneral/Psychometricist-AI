"""Unified graph interface selecting Neo4j or local JSON fallback."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_USE_NEO4J: bool | None = None


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
    except Exception as e:
        logger.warning("[graph] Neo4j unavailable (%s), using local JSON fallback.", e)
        _USE_NEO4J = False

    return _USE_NEO4J


def get_facets_for_trait(trait_name: str = "Extraversion") -> list[dict]:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_facets_for_trait as _neo

        with get_driver() as d:
            return _neo(d, trait_name)

    from src.graph.local_graph import get_facets_for_trait as _local

    return _local(trait_name)


def get_probes_for_facet(facet_code: str) -> list[dict]:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_probes_for_facet as _neo

        with get_driver() as d:
            return _neo(d, facet_code)

    from src.graph.local_graph import get_probes_for_facet as _local

    return _local(facet_code)


def get_unused_probe(facet_code: str, used_ids: list[str]) -> dict | None:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_unused_probe as _neo

        with get_driver() as d:
            return _neo(d, facet_code, used_ids)

    from src.graph.local_graph import get_unused_probe as _local

    return _local(facet_code, used_ids)


def get_items_for_facet(facet_code: str) -> list[dict]:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_items_for_facet as _neo

        with get_driver() as d:
            return _neo(d, facet_code)

    from src.graph.local_graph import get_items_for_facet as _local

    return _local(facet_code)


def get_linguistic_features(facet_code: str) -> list[dict]:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_linguistic_features as _neo

        with get_driver() as d:
            return _neo(d, facet_code)

    from src.graph.local_graph import get_linguistic_features as _local

    return _local(facet_code)


def get_all_probes() -> list[dict]:
    """Return all probes as a flat list."""
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_facets_for_trait as _neo_facets
        from src.graph.neo4j_client import get_probes_for_facet as _neo_probes

        with get_driver() as d:
            facets = _neo_facets(d)
            all_probes: list[dict] = []
            for facet in facets:
                all_probes.extend(_neo_probes(d, facet["code"]))
            return all_probes

    from src.graph.local_graph import get_all_probes as _local

    return _local()


def get_all_data_for_scoring(trait_name: str = "Extraversion") -> dict:
    if _check_neo4j():
        from src.graph.neo4j_client import get_driver, get_all_data_for_scoring as _neo

        with get_driver() as d:
            return _neo(d, trait_name)

    from src.graph.local_graph import get_all_data_for_scoring as _local

    return _local(trait_name)
