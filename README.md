# AI Psychometricist

Multi-agent conversational psychometric assessment system built with **LangGraph**, **Neo4j**, and **GPT-5.2**. Instead of traditional questionnaires, this system conducts natural, open-ended interviews to measure personality traits.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

## 🎯 MVP: Extraversion Assessment

The current MVP measures **Extraversion** (Big Five model) through a 12-turn conversational interview covering 6 facets:
- E1 Friendliness (Warmth)
- E2 Gregariousness  
- E3 Assertiveness
- E4 Activity Level
- E5 Excitement-Seeking
- E6 Cheerfulness (Positive Emotions)

Uses public-domain **IPIP** (International Personality Item Pool) items for validation.

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────┐
│         LangGraph Workflow                  │
│  ┌─────────┐    ┌──────────┐    ┌────────┐ │
│  │ Router  │───▶│Interview │───▶│ Scorer │ │
│  │  Node   │◀───│   Agent  │    │ Agent  │ │
│  └─────────┘    └──────────┘    └────────┘ │
└───────────┬─────────────────────────────────┘
            │
            ▼
    ┌──────────────┐         ┌───────────┐
    │   Neo4j      │   OR    │ Local     │
    │   Graph      │◀────────│ JSON      │
    │   (optional) │         │ Fallback  │
    └──────────────┘         └───────────┘
         Traits, Facets, Items, Probes
```

**Two Agents:**
- **Interviewer**: Generates warm, open-ended questions guided by graph-stored probes
- **Scorer**: Analyzes full transcript, rates 6 facets (1-5), produces overall score + Low/Medium/High classification

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repo
git clone https://github.com/Recognifygeneral/Psychometricist-AI.git
cd Psychometricist-AI

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install langgraph langchain-openai langchain-core neo4j pydantic python-dotenv scipy numpy
```

### 2. Configuration

```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-...

# Neo4j is OPTIONAL — the system runs with local JSON fallback
# To use Neo4j Aura (free tier):
# 1. Sign up at aura.neo4j.io
# 2. Create a free instance
# 3. Add credentials to .env:
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

### 3. (Optional) Seed Neo4j

If using Neo4j, populate the graph:

```bash
python -m src.graph.seed
```

### 4. Run the Interview

```bash
python -m src.main
```

A 12-turn conversational interview will begin. Answer naturally — there are no right/wrong answers. Type `quit` to end early.

## 📊 Evaluation Workflow

### Compare AI scores with self-report questionnaire:

```bash
# 1. Take the standard IPIP self-report (5 min)
python -m src.evaluation.self_report

# 2. Run the AI interview (8-12 min)
python -m src.main

# 3. After collecting N≥5 participants, compute correlation:
python -m src.evaluation.compare
```

Outputs Pearson r, Spearman ρ, MAE, and classification agreement.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Specific tests
pytest tests/test_smoke.py        # Import chain + data loading
pytest tests/test_e2e_mock.py     # Full interview with mocked LLM
```

## 📁 Project Structure

```
AI psychometricist/
├── data/
│   └── ipip_extraversion.json      # IPIP items, facets, probes, linguistic features
├── src/
│   ├── main.py                     # CLI entry-point
│   ├── workflow.py                 # LangGraph StateGraph
│   ├── agents/
│   │   ├── interviewer.py          # Open-ended questioning agent
│   │   └── scorer.py               # Transcript analysis & scoring agent
│   ├── graph/
│   │   ├── graph_client.py         # Unified interface (Neo4j or local)
│   │   ├── neo4j_client.py         # Neo4j query functions
│   │   ├── local_graph.py          # JSON fallback implementation
│   │   └── seed.py                 # Neo4j seeding script
│   ├── models/
│   │   └── state.py                # AssessmentState TypedDict
│   └── evaluation/
│       ├── self_report.py          # Standard IPIP questionnaire CLI
│       └── compare.py              # Correlation analysis
├── tests/
│   ├── test_smoke.py               # Import & data integrity tests
│   └── test_e2e_mock.py            # Full workflow with mocked LLM
├── pyproject.toml
├── .gitignore
└── ROADMAP.md                      # Phases 2-8 (Observer, Navigator, full Big Five)
```

## 🛠️ Tech Stack

- **LangGraph** 0.4+ — Multi-agent orchestration with state management
- **LangChain + OpenAI** — LLM interface (GPT-5.2)
- **Neo4j** (optional) — Graph database for psychometric structures
- **IPIP** — Public-domain personality items (Goldberg, 1992)
- **Python 3.11+** — Core language
- **pytest** — Test framework

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan (v0.1 → clinical deployment).

**Next phases:**
- **Phase 2**: Observer agent (real-time linguistic feature extraction with spaCy)
- **Phase 3**: Navigator agent (adaptive facet routing, uncertainty-based stopping)
- **Phase 4**: Expand to all Big Five traits (IPIP-NEO-120)
- **Phase 5**: Hybrid scoring (LLM + IRT/CAT)
- **Phase 6**: Large-scale validation on myPersonality / Essays datasets
- **Phase 7**: Web interface, therapist dashboard
- **Phase 8**: HEXACO, VIA Strengths, clinical screening instruments

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

Uses **IPIP public-domain items** — no licensing restrictions.

## 🙏 Acknowledgments

- **International Personality Item Pool (IPIP)** — Lewis R. Goldberg
- **NEO-PI-R facets** — Costa & McCrae (1992)
- **LangGraph framework** — LangChain AI
- **Neo4j Graph Database** — Neo4j, Inc.

## 📬 Contact

Marco @ Recognifygeneral — [GitHub](https://github.com/Recognifygeneral)

---

**⚡ Status**: MVP functional | Tests passing | Ready for pilot testing
