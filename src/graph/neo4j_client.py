"""Neo4j graph operations: connection helper and query functions."""

from __future__ import annotations

import os
from typing import Any

from neo4j import GraphDatabase


def get_driver():
    """Return a Neo4j driver instance (caller must close it or use ``with``).

    Connection parameters are read lazily from environment variables
    on each call — never cached at module level.
    """
    uri = os.getenv("NEO4J_URI", "neo4j+s://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(user, pwd))


# ── Query helpers used by agents ──────────────────────────────────────────


def get_facets_for_trait(driver, trait_name: str = "Extraversion") -> list[dict]:
    """Return all facets for a trait, ordered by code."""
    records, _, _ = driver.execute_query(
        """
        MATCH (t:Trait {name: $trait})-[:HAS_FACET]->(f:Facet)
        RETURN f.code AS code, f.name AS name, f.description AS description
        ORDER BY f.code
        """,
        trait=trait_name,
        database_="neo4j",
    )
    return [dict(r) for r in records]


def get_probes_for_facet(driver, facet_code: str) -> list[dict]:
    """Return all probes linked to a facet."""
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Facet {code: $code})<-[:EXPLORES]-(p:Probe)
        RETURN p.id AS id, p.text AS text, p.target_behavior AS target_behavior
        """,
        code=facet_code,
        database_="neo4j",
    )
    return [dict(r) for r in records]


def get_unused_probe(driver, facet_code: str, used_ids: list[str]) -> dict | None:
    """Return a single unused probe for a facet, or None if all used."""
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Facet {code: $code})<-[:EXPLORES]-(p:Probe)
        WHERE NOT p.id IN $used
        RETURN p.id AS id, p.text AS text, p.target_behavior AS target_behavior
        LIMIT 1
        """,
        code=facet_code,
        used=used_ids,
        database_="neo4j",
    )
    return dict(records[0]) if records else None


def get_items_for_facet(driver, facet_code: str) -> list[dict]:
    """Return all IPIP items for a facet."""
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Facet {code: $code})<-[:MEASURES]-(i:Item)
        RETURN i.text AS text, i.keying AS keying, i.position AS position
        ORDER BY i.position
        """,
        code=facet_code,
        database_="neo4j",
    )
    return [dict(r) for r in records]


def get_linguistic_features(driver, facet_code: str) -> list[dict]:
    """Return linguistic features associated with a facet."""
    records, _, _ = driver.execute_query(
        """
        MATCH (f:Facet {code: $code})<-[:INDICATES]-(lf:LinguisticFeature)
        RETURN lf.name AS name, lf.description AS description, lf.direction AS direction
        """,
        code=facet_code,
        database_="neo4j",
    )
    return [dict(r) for r in records]


def get_all_data_for_scoring(driver, trait_name: str = "Extraversion") -> dict:
    """Fetch everything the Scorer needs: facets, items, features."""
    facets = get_facets_for_trait(driver, trait_name)
    result: dict[str, Any] = {"trait": trait_name, "facets": []}
    for f in facets:
        items = get_items_for_facet(driver, f["code"])
        features = get_linguistic_features(driver, f["code"])
        result["facets"].append(
            {**f, "items": items, "linguistic_features": features}
        )
    return result


def get_all_probes(driver, trait_name: str = "Extraversion") -> list[dict]:
    """Return all probes as a flat list across all facets."""
    facets = get_facets_for_trait(driver, trait_name)
    all_probes: list[dict] = []
    for facet in facets:
        all_probes.extend(get_probes_for_facet(driver, facet["code"]))
    return all_probes
