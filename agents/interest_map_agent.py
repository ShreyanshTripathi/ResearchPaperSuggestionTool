import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from memory.vector_store import ResearchVectorStore
from models import AgentResult, Paper, SearchRecord

logger = logging.getLogger(__name__)


def _build_llamaindex_retriever(vector_store: ResearchVectorStore):
    # Access the underlying ChromaDB collection from our store
    chroma_papers_col = vector_store._papers_col   # noqa: private access is intentional

    llama_vector_store = ChromaVectorStore(chroma_collection=chroma_papers_col)
    storage_context = StorageContext.from_defaults(vector_store=llama_vector_store)

    index = VectorStoreIndex.from_vector_store(
        llama_vector_store,
        storage_context=storage_context,
    )
    return index.as_retriever(similarity_top_k=settings.chroma_top_k)


def _synthesise_interest_map(
    interest_map: dict[str, float],
    similar_searches: list[dict],
    current_query: str,
) -> str:
    if not interest_map and not similar_searches:
        return "No research history yet. This is your first search session."

    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0.2,   # Slightly creative for narrative generation
        api_key=settings.api_key,
    )

    top_interests = list(interest_map.items())[:10]
    interest_str = "\n".join(
        f"  - '{q}' (weight: {w:.3f})"
        for q, w in top_interests
    )

    sim_str = "\n".join(
        f"  - '{s['query']}' "
        f"(searched {s['days_ago']:.0f} days ago, "
        f"similarity: {s['raw_similarity']:.2f}, "
        f"time-weighted score: {s['weighted_score']:.3f})"
        for s in similar_searches[:5]
    )

    prompt = f"""
You are summarising a researcher's academic interests based on their search history.

TODAY'S QUERY: "{current_query}"

OVERALL RESEARCH INTEREST MAP (higher weight = more persistent interest):
{interest_str or "  (no history yet)"}

MOST RELATED PAST SEARCHES (sorted by weighted relevance to today's query):
{sim_str or "  (no closely related past searches)"}

Write a 2–4 sentence research interest profile that:
1. Identifies the researcher's core research themes from the history.
2. Explains how today's query connects to (or represents a shift from) past interests.
3. Notes any emerging trends (recent searches getting more weight).

Be specific and academic in tone. Do not use bullet points — write flowing prose.
"""

    resp = llm.invoke(prompt)
    return resp.content.strip()


def run_interest_map_agent(
    query: str,
    current_session_papers: list[Paper],
    vector_store: ResearchVectorStore,
) -> AgentResult:
    logger.info("[Agent 3] Running interest map for: '%s'", query)
    similar_searches = vector_store.get_similar_searches(query, top_k=settings.chroma_top_k)
    historical_paper_metas = vector_store.get_related_papers(query, top_k=settings.agent3_final_papers)
    historical_papers: list[Paper] = []
    for meta in historical_paper_metas:
        try:
            historical_papers.append(Paper(
                paper_id=meta.get("paper_id", "unknown"),
                title=meta.get("title", "Untitled"),
                authors=meta.get("authors", "").split(", "),
                abstract="(Seen in previous session — abstract not stored)",
                url=meta.get("url", ""),
                source=meta.get("source", "history"),
                published_date=meta.get("published_date"),
                relevance_score=meta.get("weighted_score", 0.0),
            ))
        except Exception as exc:
            logger.debug("Skipping malformed historical paper meta: %s", exc)

    interest_map = vector_store.build_interest_map()

    interest_narrative = _synthesise_interest_map(interest_map, similar_searches, query)
    logger.info("[Agent 3] Interest narrative: %s", interest_narrative[:100])
    all_current_papers = current_session_papers  # Agents 1 + 2 combined

   
    search_record = SearchRecord(
        query=query,
        paper_ids=[p.paper_id for p in all_current_papers],
    )
    vector_store.save_search(search_record)
    vector_store.save_papers(all_current_papers, source_query=query)

    return AgentResult(
        agent_name="InterestMapAgent",
        papers=historical_papers,
        reasoning=interest_narrative,
        metadata={
            "interest_map": interest_map,
            "similar_past_searches": [
                {
                    "query": s["query"],
                    "days_ago": s["days_ago"],
                    "weighted_score": s["weighted_score"],
                }
                for s in similar_searches[:5]
            ],
        },
    )
