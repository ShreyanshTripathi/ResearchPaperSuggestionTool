# Research Agent Framework

A multi-agent system for academic paper discovery that learns your research interests over time. Three specialised agents work together: one finds papers precisely on your topic, one surfaces foundational and adjacent work, and one builds a persistent map of how your interests connect and evolve across sessions.

---

## How it works

```
User query: "RL-based unlearning"
        │
        ▼
  ┌─────────────────────────────────┐
  │         Orchestrator            │
  └──────┬──────────┬──────────────┘
         │          │
         ▼          ▼
   ┌──────────┐ ┌───────────────┐    ┌──────────────────┐
   │ Agent 1  │ │   Agent 2     │───▶│     Agent 3      │
   │ Focused  │ │   Broader     │    │  Interest Map    │
   │  Search  │ │   Context     │    │  (RAG + memory)  │
   └────┬─────┘ └──────┬────────┘    └──────────────────┘
        │              │
        ▼              ▼
   arXiv API    Semantic Scholar
                + LLM topic expansion
```

**Agent 1 — Focused Search** queries arXiv with your exact topic, then uses an LLM to re-rank results by strict topical relevance. Only papers whose *main contribution* matches your query pass through.

**Agent 2 — Broader Context** asks the LLM to identify the parent fields and adjacent research areas (e.g. "RL-based unlearning" → "machine unlearning", "AI safety", "continual learning"), then fetches foundational and highly-cited papers from each using Semantic Scholar's influential citation counts.

**Agent 3 — Interest Map** queries a local vector database of your past searches, applies exponential temporal decay so recent sessions dominate, and surfaces papers from previous sessions that connect to today's topic. After each session it saves the new papers and query to the store, building your interest map over time.

---

## Quickstart

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Add your OpenAI API key**

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

**3. Run**

```bash
# Interactive mode — run multiple searches to build up your interest map
python main.py

# Single query
python main.py --query "RL-based machine unlearning"

# View your accumulated research interest profile
python main.py --show-history

# Debug mode — see each agent's reasoning steps
python main.py --verbose

# Wipe search history and start fresh
python main.py --clear-history
```

---

## Project structure

```
research_agent/
├── main.py                          # CLI entry point
├── orchestrator.py                  # Coordinates agents, merges results
├── config.py                        # All settings (env-var driven)
├── models.py                        # Shared Pydantic data models
│
├── agents/
│   ├── focused_search_agent.py      # Agent 1: strict topic search + LLM re-ranking
│   ├── broader_context_agent.py     # Agent 2: topic expansion + foundational papers
│   └── interest_map_agent.py        # Agent 3: RAG over search history
│
├── tools/
│   ├── arxiv_tool.py                # arXiv API wrapper (LangChain Tool)
│   └── semantic_scholar_tool.py     # Semantic Scholar API wrapper
│
├── memory/
│   └── vector_store.py              # ChromaDB persistence + temporal discounting
│
└── tests/
    └── test_vector_store.py         # Unit tests for the memory layer
```

---

## Configuration

All settings are controlled via environment variables (or a `.env` file). No values are hardcoded.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model used by all agents |
| `LLM_TEMPERATURE` | `0.0` | Deterministic output for structured tasks |
| `ARXIV_MAX_RESULTS` | `10` | Papers fetched per Agent 1 query |
| `SEMANTIC_SCHOLAR_MAX_RESULTS` | `8` | Papers fetched per Agent 2 topic |
| `CHROMA_PERSIST_DIR` | `./chroma_store` | Where ChromaDB stores its data |
| `TEMPORAL_DECAY_LAMBDA` | `0.05` | Decay rate λ — half-life ≈ 14 days |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `BROAD_TOPIC_EXPANSION_COUNT` | `3` | Number of related topics Agent 2 expands to |
| `AGENT1_FINAL_PAPERS` | `5` | Max papers returned by Agent 1 |
| `AGENT2_FINAL_PAPERS` | `6` | Max papers returned by Agent 2 |
| `AGENT3_FINAL_PAPERS` | `4` | Max historically-related papers from Agent 3 |

---

## Temporal decay model

Agent 3 weights past searches using exponential decay:

```
weight = exp(−λ × days_since_search)
```

With the default λ = 0.05:

| Time ago | Weight |
|---|---|
| Today | 1.000 |
| 1 week | 0.704 |
| 2 weeks (half-life) | 0.500 |
| 1 month | 0.223 |
| 3 months | 0.011 |

The final ranking score for a past search is `cosine_similarity × temporal_weight`. This means a search from two months ago needs to be nearly twice as similar to today's query to rank alongside one from last week — recent interests dominate, but old ones are never fully erased.

---

## Key technology choices

| Component | Technology | Reason |
|---|---|---|
| Agent loop | LangChain ReAct | Self-correcting: the agent can refine its query if first results are poor |
| RAG pipeline | LlamaIndex | Purpose-built for retrieval + LLM synthesis over document stores |
| Vector database | ChromaDB | In-process, no server, persists to disk, native LangChain/LlamaIndex support |
| Embeddings | sentence-transformers | Runs locally — no API cost or latency for embedding historical searches |
| Paper sources | arXiv + Semantic Scholar | arXiv for recent pre-prints; S2 for citation influence scores and cross-domain coverage |
| Data models | Pydantic v2 | Runtime type validation, automatic JSON serialisation for ChromaDB metadata |
| LLM re-ranking | GPT-4o-mini | Semantic relevance judgments TF-IDF can't make ("is this paper *about* this topic?") |

---

## Running tests

```bash
pytest tests/ -v
```

Tests cover: temporal weight correctness, save/retrieve roundtrip, decay ordering (newer searches rank higher), interest map normalisation, and paper-level deduplication.

---

## Extending the framework

**Swap the LLM** — change `LLM_MODEL` in `.env` or in `config.py`. Any model supported by LangChain's `ChatOpenAI` (or swap the class for `ChatAnthropic`, `ChatOllama`, etc.) works without touching agent code.

**Add a new paper source** — create a new file in `tools/`, implement a `fetch_*` function returning `list[Paper]`, and optionally wrap it with `@tool` for LangChain. Call it from the relevant agent.

**Add a web UI** — `orchestrator.py` returns a `ResearchSession` Pydantic object. Wrap `run_research_session()` in a FastAPI endpoint or a Streamlit app; no agent code changes needed.

**Enable parallel agent execution** — Agents 1 and 2 are independent. Replace the sequential calls in `orchestrator.py` with `asyncio.gather()` and convert the agent functions to `async def` for ~40% faster wall-clock time.

**Use a cloud vector store** — replace `ResearchVectorStore` in `memory/vector_store.py` with a Pinecone or Weaviate client. The interface (`save_search`, `get_similar_searches`, `build_interest_map`) stays the same; only the backend changes.

---

## Limitations

- Requires an OpenAI API key. All LLM calls (re-ranking, topic expansion, narrative synthesis) use your account.
- arXiv and Semantic Scholar are queried without authentication; rate limits apply (~3 s between arXiv requests, ~1 req/s for Semantic Scholar).
- The embedding model (~80 MB) is downloaded from Hugging Face on first run.
- Agent 3's interest map is only as good as your search history — it needs several sessions to build meaningful connections.
