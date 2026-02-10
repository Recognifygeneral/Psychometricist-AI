"""Seed the Neo4j graph with the IPIP Extraversion psychometric structure.

Usage:
    python -m src.graph.seed          # or: seed-graph  (via pyproject entry-point)

The script is idempotent: it uses MERGE so re-running won't create duplicates.
"""

from __future__ import annotations

import json

from src.graph.neo4j_client import get_driver
from src.paths import IPIP_DATA_PATH as DATA_PATH


def _load_data() -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Cypher seed statements ────────────────────────────────────────────────

_CREATE_TRAIT = """
MERGE (t:Trait {name: $name})
SET t.code        = $code,
    t.description = $description,
    t.source      = $source,
    t.alpha       = $alpha
"""

_CREATE_FACET = """
MATCH (t:Trait {name: $trait})
MERGE (f:Facet {code: $code})
SET f.name        = $name,
    f.neo_name    = $neo_name,
    f.description = $description
MERGE (t)-[:HAS_FACET]->(f)
"""

_CREATE_ITEM = """
MATCH (f:Facet {code: $primary_facet})
MERGE (i:Item {position: $position})
SET i.text   = $text,
    i.keying = $keying
MERGE (i)-[:MEASURES]->(f)
"""

_LINK_ITEM_SECONDARY = """
MATCH (i:Item {position: $position})
MATCH (f:Facet {code: $facet_code})
MERGE (i)-[:MEASURES]->(f)
"""

_CREATE_PROBE = """
MATCH (f:Facet {code: $facet})
MERGE (p:Probe {id: $id})
SET p.text            = $text,
    p.target_behavior = $target_behavior,
    p.item_position   = $item_position
MERGE (p)-[:EXPLORES]->(f)
"""

_CREATE_LINGUISTIC_FEATURE = """
MERGE (lf:LinguisticFeature {name: $name})
SET lf.description = $description,
    lf.direction   = $direction
"""

_LINK_FEATURE_FACET = """
MATCH (lf:LinguisticFeature {name: $name})
MATCH (f:Facet {code: $facet_code})
MERGE (lf)-[:INDICATES]->(f)
"""


def seed(driver) -> dict[str, int]:
    """Populate the graph. Returns counts of created entities."""
    data = _load_data()
    counts: dict[str, int] = {}

    # 1. Trait
    t = data["trait"]
    driver.execute_query(
        _CREATE_TRAIT,
        name=t["name"],
        code=t["code"],
        description=t["description"],
        source=t["source"],
        alpha=t["reliability_alpha"],
        database_="neo4j",
    )
    counts["traits"] = 1

    # 2. Facets
    for f in data["facets"]:
        driver.execute_query(
            _CREATE_FACET,
            trait=t["name"],
            code=f["code"],
            name=f["name"],
            neo_name=f["neo_name"],
            description=f["description"],
            database_="neo4j",
        )
    counts["facets"] = len(data["facets"])

    # 3. Items
    for item in data["items"]:
        driver.execute_query(
            _CREATE_ITEM,
            position=item["position"],
            text=item["text"],
            keying=item["keying"],
            primary_facet=item["primary_facet"],
            database_="neo4j",
        )
        for sec in item.get("secondary_facets", []):
            driver.execute_query(
                _LINK_ITEM_SECONDARY,
                position=item["position"],
                facet_code=sec,
                database_="neo4j",
            )
    counts["items"] = len(data["items"])

    # 4. Probes
    for p in data["probes"]:
        driver.execute_query(
            _CREATE_PROBE,
            id=p["id"],
            facet=p["facet"],
            text=p["text"],
            target_behavior=p["target_behavior"],
            item_position=p.get("item_position"),
            database_="neo4j",
        )
    counts["probes"] = len(data["probes"])

    # 5. Linguistic features
    for lf in data["linguistic_features"]:
        driver.execute_query(
            _CREATE_LINGUISTIC_FEATURE,
            name=lf["name"],
            description=lf["description"],
            direction=lf["direction"],
            database_="neo4j",
        )
        for facet_code in lf["facets"]:
            driver.execute_query(
                _LINK_FEATURE_FACET,
                name=lf["name"],
                facet_code=facet_code,
                database_="neo4j",
            )
    counts["linguistic_features"] = len(data["linguistic_features"])

    return counts


def main() -> None:
    """CLI entry-point."""
    print("Connecting to Neo4j …")
    with get_driver() as driver:
        driver.verify_connectivity()
        print("Connected ✓")
        counts = seed(driver)
        print("Seed complete:")
        for entity, n in counts.items():
            print(f"  {entity}: {n}")


if __name__ == "__main__":
    main()
