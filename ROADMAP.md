# AI Psychometricist — Roadmap

## Current State: MVP (v0.1)

**Trait**: Extraversion only (IPIP 50-item Big Five Markers, 10 items)
**Agents**: Interviewer + Scorer
**Graph**: Neo4j Aura with Trait → Facet → Item → Probe + LinguisticFeature
**Evaluation**: Self-report IPIP questionnaire + Pearson r correlation

---

## Phase 2 — Observer Agent

- [ ] Add **Observer** agent that runs after each user message
- [ ] Extract linguistic features in real time using spaCy:
  - Positive emotion word frequency (NRC / LIWC word lists)
  - Social reference word count
  - First-person singular pronoun ratio
  - Response length (word count)
  - Assertive language markers
  - Hedging / uncertainty markers
- [ ] Store extracted features as `(:Observation)` nodes in Neo4j linked to the session
- [ ] Feed Observer output to the Scorer as supplementary evidence

## Phase 3 — Navigator Agent

- [ ] Add **Navigator** agent that queries Neo4j between turns
- [ ] Compute per-facet coverage score (how much evidence collected)
- [ ] Implement uncertainty estimation:
  - Track variance of linguistic feature signals per facet
  - Use Bayesian credible intervals on trait estimates
  - Stop when 95% CI width < configurable threshold
- [ ] Adaptive routing: Navigator suggests which facet to explore next
  (instead of sequential E1→E2→…→E6)
- [ ] "Go deeper" logic: if signals are contradictory for a facet,
  Navigator requests more probes before moving on

## Phase 4 — Expand to All Big Five

- [ ] Add Openness, Conscientiousness, Agreeableness, Neuroticism
- [ ] Upgrade to **IPIP-NEO-120** (4 items × 30 facets = 120 items)
- [ ] Facet-level scoring (not just domain-level)
- [ ] Generate probes for all 30 facets
- [ ] Add facet-specific linguistic markers from literature
- [ ] Dynamic interview length based on Navigator uncertainty

## Phase 5 — Advanced Scoring

- [ ] Hybrid scoring: combine LLM judgment + NLP feature extraction
- [ ] Item Response Theory (IRT) model:
  - Fit 2PL model to calibrate item difficulty / discrimination
  - Computerized Adaptive Testing (CAT) — select most informative probe next
- [ ] Scorer confidence intervals on final trait estimates
- [ ] Profile validity checks (acquiescence, infrequency detection)

## Phase 6 — Large-Scale Validation

- [ ] Run on existing personality + text datasets:
  - **myPersonality** dataset (Facebook data + Big Five scores)
  - **Essays dataset** (Pennebaker — essays + Big Five)
  - **PANDORA** dataset (Reddit posts + personality scores)
- [ ] Target: Pearson r > 0.5 (moderate convergent validity)
- [ ] Stretch goal: r > 0.7 (strong convergent validity)
- [ ] Cross-validation: k-fold on dataset samples
- [ ] Discriminant validity: verify Extraversion score doesn't correlate
  too highly with other Big Five dimensions

## Phase 7 — Clinical / Research Deployment

- [ ] Web interface (Streamlit or React frontend)
- [ ] Multi-session support (user can return and continue)
- [ ] Therapist / researcher dashboard showing graph-based profiles
- [ ] Export to standard psychometric report format (PDF)
- [ ] Multilingual support (IPIP items are translated into 25+ languages)
- [ ] GDPR-compliant data handling and consent flow
- [ ] IRB protocol template for research use

## Phase 8 — Beyond Big Five

- [ ] HEXACO model (6 factors, adds Honesty-Humility)
- [ ] VIA Character Strengths (24 strengths)
- [ ] Clinical screening instruments (PHQ-9, GAD-7 conversational adaptations)
- [ ] Pluggable graph schema: any psychometric instrument → graph → agents

---

## Architecture Evolution

```text
MVP (now)           Full System (future)
─────────           ────────────────────
Interviewer ←→ User    Interviewer ←→ User
    ↓                      ↓         ↑
  Scorer               Observer → Navigator
                           ↓         ↓
                         Scorer ← Neo4j Graph
```
