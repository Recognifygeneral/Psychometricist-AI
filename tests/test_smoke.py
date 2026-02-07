"""Smoke test — verify the local graph fallback and import chain."""

from src.graph.local_graph import (
    get_facets_for_trait,
    get_unused_probe,
    get_all_data_for_scoring,
    get_probes_for_facet,
    get_items_for_facet,
    get_linguistic_features,
)

# 1. Facets
facets = get_facets_for_trait()
print(f"Facets loaded: {len(facets)}")
for f in facets:
    print(f"  {f['code']} {f['name']}")

# 2. Probes per facet
print()
for f in facets:
    probes = get_probes_for_facet(f["code"])
    print(f"  {f['code']}: {len(probes)} probes")

# 3. Unused probe logic
probe = get_unused_probe("E1", [])
print(f"\nFirst E1 probe: {probe['text'][:60]}...")
probe2 = get_unused_probe("E1", [probe["id"]])
print(f"Second E1 probe: {probe2['text'][:60]}...")
probe3 = get_unused_probe("E1", [probe["id"], probe2["id"]])
print(f"Third E1 probe (should be None): {probe3}")

# 4. Items per facet
print()
for f in facets:
    items = get_items_for_facet(f["code"])
    print(f"  {f['code']}: {len(items)} items")

# 5. Linguistic features
print()
for f in facets:
    feats = get_linguistic_features(f["code"])
    print(f"  {f['code']}: {len(feats)} linguistic features")

# 6. Full scoring data
scoring = get_all_data_for_scoring()
total_items = sum(len(f["items"]) for f in scoring["facets"])
total_feats = sum(len(f["linguistic_features"]) for f in scoring["facets"])
print(f"\nScoring bundle: {len(scoring['facets'])} facets, {total_items} items, {total_feats} feature links")

# 7. Import chain: agents + workflow
from src.models.state import AssessmentState
from src.agents.interviewer import interviewer_node
from src.agents.scorer import scorer_node
from src.workflow import build_graph, FACET_ORDER, MAX_TURNS

print(f"\nWorkflow: {len(FACET_ORDER)} facets, max {MAX_TURNS} turns")

graph = build_graph()
print(f"Graph compiled: {type(graph).__name__}")

print("\n✓ All smoke tests passed!")
