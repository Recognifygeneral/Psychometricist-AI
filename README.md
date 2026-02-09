# AI Psychometricist

A **scientific feasibility study** exploring whether conversational AI can reliably estimate psychological traits from semi-structured dialogue. Built with **LangGraph**, **GPT-5.2**, and multi-method scoring (linguistic features, embedding similarity, LLM classification).

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Tests](https://img.shields.io/badge/tests-57%20passing-brightgreen.svg)](tests/)

## Research Question

> Can a conversational AI system, through 10 turns of semi-structured dialogue,
> produce Extraversion estimates that correlate meaningfully (r ≥ .40) with
> validated self-report measures?

## Approach

The system conducts a natural interview (10 turns) and scores responses with **three independent methods**, then fuses them:

| Method | Basis | Literature |
|--------|-------|------------|
| **Feature-based** | Counts positive/negative emotion words, social references, pronouns, assertive/hedging language | Pennebaker & King 1999; Mairesse et al. 2007; Yarkoni 2010 |
| **Embedding similarity** | Cosine similarity of response embeddings against high-E / low-E reference vignettes | Schwartz et al. 2013 |
| **LLM classification** | GPT-5.2 zero-shot prompt: classify Low/Medium/High with confidence | — |
| **Ensemble** | Confidence-weighted mean + majority-vote classification | — |

Each method independently produces: **score** (1–5), **classification** (Low/Medium/High), **confidence** (0–1).

## Current Target Construct

**Extraversion** (Big Five) measured across 6 facets using public-domain **IPIP** items:

| Facet | Code | Example Probe |
|-------|------|---------------|
| Friendliness | E1 | _Tell me about a time you met someone new…_ |
| Gregariousness | E2 | _How do you usually feel at large social gatherings?_ |
| Assertiveness | E3 | _When working in a group, how do you handle disagreements?_ |
| Activity Level | E4 | _Walk me through a typical day…_ |
| Excitement-Seeking | E5 | _What's the most exciting thing you've done recently?_ |
| Cheerfulness | E6 | _How would your friends describe your general mood?_ |

## Architecture

```text
User ◀──────────── Interviewer Agent (GPT-5.2) ◀── Probe Pool (10 probes)
  │                       │
  │  10 turns             │ per-turn feature extraction
  ▼                       ▼
Transcript ───────▶ Scoring Pipeline
                    ├── Feature Scorer  (word lists, ratios → weighted sum)
                    ├── Embedding Scorer (text-embedding-3-small → cosine sim)
                    ├── LLM Scorer      (GPT-5.2 → classification + confidence)
                    └── Ensemble        (confidence-weighted fusion)
                           │
                           ▼
                    Session Logger ──▶ data/sessions/{id}_{ts}.json
```

**Two LangGraph agents:**
- **Interviewer** — generates warm, open-ended questions guided by graph-stored probes; extracts linguistic features from each turn
- **Scorer** — delegates to the ensemble scoring pipeline; logs structured session data

**Data backend:** JSON fallback (default) or Neo4j Aura (optional, for graph exploration).

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Recognifygeneral/Psychometricist-AI.git
cd Psychometricist-AI

python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -e ".[web]"
# Or manually:
pip install langgraph langchain-openai langchain-core pydantic python-dotenv scipy numpy
pip install fastapi uvicorn  # optional, for web UI
```

### 2. Configuration

```bash
cp .env.example .env
# Add your OpenAI API key:
OPENAI_API_KEY=sk-...
```

Neo4j is **optional** — the system runs fully with the local JSON fallback.

### 3. Run the Interview

```bash
# CLI mode (terminal interview)
python -m src.main

# Web mode (browser-based chat UI)
python -m web.app
# → Open http://localhost:8080
```

## Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

Quick cloud deployment:

1. Push this repo to your GitHub
2. Go to [railway.app/new](https://railway.app/new) → Deploy from GitHub
3. Select your repository
4. Add environment variable: `OPENAI_API_KEY=sk-...`
5. Deploy → Railway provides a public URL

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions, Neo4j setup, and troubleshooting.

The system conducts a 10-turn interview and displays multi-method scoring results.

### 4. After the Interview

Results are automatically saved to `data/sessions/`. Each session file contains:
- Per-turn transcripts with timestamps
- Extracted linguistic features per turn
- Scoring results from all methods
- Ensemble classification and confidence

## Evaluation Workflow

```bash
# 1. Participant takes standardized IPIP self-report (5 min)
python -m src.evaluation.self_report

# 2. Participant completes AI interview (8–12 min)
python -m src.main

# 3. After N ≥ 5 participants, analyze agreement:
python -m src.evaluation.compare
```

Outputs per-method: Pearson r, Spearman ρ, MAE, classification agreement rate.

## Testing

```bash
# Full suite (57 tests)
pytest tests/ -v

# Individual suites
pytest tests/test_features.py     # 27 tests — linguistic feature extraction
pytest tests/test_scoring.py      # 18 tests — scoring modules + ensemble
pytest tests/test_e2e_mock.py     # 1 test  — full interview loop (mocked LLM)
pytest tests/test_evaluation.py   # self-report/session link plumbing
pytest tests/test_web_api.py      # FastAPI endpoint validation + flow
pytest tests/test_settings.py     # settings and threshold fallback
python tests/test_smoke.py        # Import chain + data loading + feature scoring
```

## Project Structure

```
├── data/
│   ├── ipip_extraversion.json       # IPIP items, facets, probes, linguistic features
│   └── sessions/                    # Structured JSON session logs (auto-created)
├── src/
│   ├── main.py                      # CLI entry-point
│   ├── workflow.py                  # LangGraph StateGraph (10-turn loop)
│   ├── agents/
│   │   ├── interviewer.py           # Open-ended questioning agent
│   │   └── scorer.py               # Delegates to ensemble scoring pipeline
│   ├── extraction/
│   │   ├── word_lists.py            # ~200 curated words (LIWC/NRC-inspired)
│   │   └── features.py             # LinguisticFeatures dataclass + extraction
│   ├── scoring/
│   │   ├── feature_scorer.py        # Rule-based weighted-sum scorer
│   │   ├── embedding_scorer.py      # Cosine similarity (OpenAI embeddings)
│   │   ├── llm_scorer.py           # GPT-5.2 domain + facet classification
│   │   └── ensemble.py             # Confidence-weighted fusion
│   ├── session/
│   │   └── logger.py               # JSON session logging
│   ├── graph/
│   │   ├── graph_client.py          # Unified interface (Neo4j or local)
│   │   ├── neo4j_client.py          # Neo4j query functions
│   │   ├── local_graph.py           # JSON fallback implementation
│   │   └── seed.py                  # Neo4j seeding script
│   ├── models/
│   │   └── state.py                 # AssessmentState TypedDict
│   └── evaluation/
│       ├── self_report.py           # Standard IPIP questionnaire CLI
│       └── compare.py              # Multi-method correlation analysis
├── web/
│   ├── app.py                       # FastAPI server
│   └── static/index.html           # Chat UI (dark theme)
├── tests/
│   ├── test_features.py             # Feature extraction tests (27)
│   ├── test_scoring.py              # Scoring module tests (18)
│   ├── test_e2e_mock.py             # Full workflow with mocked LLMs
│   └── test_smoke.py               # Import chain + integration
├── pyproject.toml
├── .gitignore
└── ROADMAP.md
```

## Scoring Detail

### Feature-Based Scorer

Extracts linguistic features and applies empirically-grounded weights:

| Feature | Direction | Weight | Source |
|---------|-----------|--------|--------|
| Positive emotion ratio | + | 8.0 | Pennebaker & King 1999 |
| Negative emotion ratio | − | 5.0 | Pennebaker & King 1999 |
| Social reference ratio | + | 10.0 | Mairesse et al. 2007 |
| First-person plural ratio | + | 6.0 | Schwartz et al. 2013 |
| Words per turn | + | 0.03 | Mehl et al. 2006 |
| Exclamation ratio | + | 15.0 | Yarkoni 2010 |
| Assertive language ratio | + | 8.0 | Mairesse et al. 2007 |
| Hedging ratio | − | 6.0 | Pennebaker & King 1999 |

Formula: `score = 3.0 + Σ(weight × direction × (value − baseline))`, clipped to [1, 5].

### Embedding Scorer

- Encodes the full user transcript with `text-embedding-3-small`
- Compares cosine similarity against 7 high-E and 7 low-E reference vignettes
- Maps relative similarity balance to a 1–5 score

### LLM Scorer

- Single GPT-5.2 prompt: classify overall Extraversion (Low/Medium/High) with score, confidence, and textual evidence
- Optional facet-level analysis as secondary output

### Ensemble

- Confidence-weighted mean of all available method scores
- Majority-vote classification
- Reports per-method agreement and overall confidence

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph 0.4+ (StateGraph, interrupt/resume) |
| LLM | OpenAI GPT-5.2 (chat) + text-embedding-3-small (embeddings) |
| Feature extraction | Custom word lists (~200 words, LIWC/NRC-inspired) |
| Graph DB | Neo4j Aura (optional) / JSON fallback |
| Psychometric items | IPIP public domain (Goldberg 1992) |
| Statistics | scipy, numpy (Pearson r, Spearman ρ, MAE) |
| Web UI | FastAPI + vanilla HTML/CSS/JS |
| Tests | pytest (57 tests) |

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan.

**Key next steps:**
1. **Pilot validation** — Collect N ≥ 30 sessions, compute test-retest reliability
2. **Adaptive stopping** — End early if confidence > threshold
3. **Expand to Big Five** — Add Openness, Conscientiousness, Agreeableness, Neuroticism
4. **spaCy integration** — POS tags, dependency parsing, named entities for richer features
5. **Pre-trained classifier** — Train a lightweight model on myPersonality / Essays datasets

## References

- Goldberg, L. R. (1992). The development of markers for the Big-Five factor structure. *Psychological Assessment*, 4(1), 26–42.
- Mairesse, F., et al. (2007). Using linguistic cues for the automatic recognition of personality in conversation and text. *JAIR*, 30, 457–500.
- Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an individual difference. *JPSP*, 77(6), 1296–1312.
- Schwartz, H. A., et al. (2013). Personality, gender, and age in the language of social media. *PLoS ONE*, 8(9).
- Yarkoni, T. (2010). Personality in 100,000 words. *Social Psychological and Personality Science*, 1(4), 363–373.

## License

MIT License — see [LICENSE](LICENSE) for details.
Uses **IPIP public-domain items** — no licensing restrictions.

## Contact

Marco @ Recognifygeneral — [GitHub](https://github.com/Recognifygeneral)

---

**Status**: v0.2.0 | 57 tests passing | Multi-method scoring | Session logging | Ready for pilot
