"""Smoke test — verify the local graph fallback, new modules, and import chain."""

from src.graph.local_graph import (
    get_facets_for_trait,
    get_unused_probe,
    get_all_data_for_scoring,
    get_probes_for_facet,
    get_items_for_facet,
    get_linguistic_features,
    get_all_probes,
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

# 3. All probes (flat pool)
all_probes = get_all_probes()
print(f"\nAll probes (flat pool): {len(all_probes)}")

# 4. Unused probe logic
probe = get_unused_probe("E1", [])
print(f"\nFirst E1 probe: {probe['text'][:60]}...")
probe2 = get_unused_probe("E1", [probe["id"]])
print(f"Second E1 probe: {probe2['text'][:60]}...")
probe3 = get_unused_probe("E1", [probe["id"], probe2["id"]])
print(f"Third E1 probe (should be None): {probe3}")

# 5. Items per facet
print()
for f in facets:
    items = get_items_for_facet(f["code"])
    print(f"  {f['code']}: {len(items)} items")

# 6. Linguistic features
print()
for f in facets:
    feats = get_linguistic_features(f["code"])
    print(f"  {f['code']}: {len(feats)} linguistic features")

# 7. Full scoring data
scoring = get_all_data_for_scoring()
total_items = sum(len(f["items"]) for f in scoring["facets"])
total_feats = sum(len(f["linguistic_features"]) for f in scoring["facets"])
print(f"\nScoring bundle: {len(scoring['facets'])} facets, {total_items} items, {total_feats} feature links")

# 8. Feature extraction
from src.extraction.features import extract_features
feats = extract_features("I love meeting new people! We always have a great time together.")
print(f"\nExtracted features: {feats.word_count} words, {feats.positive_emotion_count} positive, "
      f"{feats.social_reference_count} social refs")

# 9. Feature-based scorer
from src.scoring.feature_scorer import score_with_features
test_feats = extract_features("I love parties, friends, and exciting adventures! We go out every weekend.")
feat_result = score_with_features(test_feats)
print(f"Feature scorer: score={feat_result['score']:.2f}, cls={feat_result['classification']}, "
      f"conf={feat_result['confidence']:.2f}")

# 10. Import chain: agents + workflow
from src.models.state import AssessmentState
from src.agents.interviewer import interviewer_node
from src.agents.scorer import scorer_node
from src.workflow import build_graph, MAX_TURNS
from src.session.logger import SessionLogger
from src.scoring.ensemble import score_ensemble

print(f"\nWorkflow: max {MAX_TURNS} turns")

graph = build_graph()
print(f"Graph compiled: {type(graph).__name__}")

# 11. Session logger
logger = SessionLogger("smoke-test")
logger.log_turn(
    1,
    "How are you?",
    "I'm great, love people!",
    probe_id="smoke-probe",
    features=extract_features("I'm great, love people!"),
)
print(f"Session logger: {len(logger.turns)} turns logged")

print("\n✓ All smoke tests passed!")
