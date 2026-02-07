"""Local JSON-based graph client — fallback when Neo4j is unavailable.

Provides the same query interface as neo4j_client.py but reads directly
from data/ipip_extraversion.json.  This lets the system run with zero
infrastructure (just an OpenAI key).
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ipip_extraversion.json"

_CACHE: dict | None = None


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        with open(DATA_PATH, encoding="utf-8") as f:
            _CACHE = json.load(f)
    return _CACHE


# ── Public API (mirrors neo4j_client functions) ──────────────────────────


def get_facets_for_trait(trait_name: str = "Extraversion") -> list[dict]:
    data = _load()
    return [
        {"code": f["code"], "name": f["name"], "description": f["description"]}
        for f in data["facets"]
    ]


def get_probes_for_facet(facet_code: str) -> list[dict]:
    data = _load()
    return [
        {"id": p["id"], "text": p["text"], "target_behavior": p["target_behavior"]}
        for p in data["probes"]
        if p["facet"] == facet_code
    ]


def get_unused_probe(facet_code: str, used_ids: list[str]) -> dict | None:
    probes = get_probes_for_facet(facet_code)
    for p in probes:
        if p["id"] not in used_ids:
            return p
    return None


def get_items_for_facet(facet_code: str) -> list[dict]:
    data = _load()
    return [
        {"text": it["text"], "keying": it["keying"], "position": it["position"]}
        for it in data["items"]
        if it["primary_facet"] == facet_code
    ]


def get_linguistic_features(facet_code: str) -> list[dict]:
    data = _load()
    return [
        {"name": lf["name"], "description": lf["description"], "direction": lf["direction"]}
        for lf in data["linguistic_features"]
        if facet_code in lf["facets"]
    ]


def get_all_data_for_scoring(trait_name: str = "Extraversion") -> dict:
    facets = get_facets_for_trait(trait_name)
    result = {"trait": trait_name, "facets": []}
    for f in facets:
        items = get_items_for_facet(f["code"])
        features = get_linguistic_features(f["code"])
        result["facets"].append({**f, "items": items, "linguistic_features": features})
    return result
