import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from config import settings
from models import AgentResult, Paper
from tools.arxiv_tool import fetch_arxiv_papers
from tools.semantic_scholar_tool import get_influential_papers, search_semantic_scholar

logger = logging.getLogger(__name__)


def _expand_to_related_topics(query: str) -> list[str]:
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )

    n = settings.broad_topic_expansion_count
    prompt = f"""
You are a research domain expert.

Given this specific research topic: "{query}"

Identify exactly {n} broader or adjacent research areas that:
1. Provide theoretical or methodological foundations for this topic.
2. Address the same high-level problem from a different angle.
3. The current topic is applied within or derived from.

Return ONLY a JSON object with this exact format:
{{"topics": ["topic 1", "topic 2", "topic 3"]}}

Example: if the input is "RL-based machine unlearning", a good response is:
{{"topics": ["machine unlearning", "AI safety and privacy", "continual learning and catastrophic forgetting"]}}

No explanation, no markdown — just the JSON object.
"""

    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        topics = data.get("topics", [])
        logger.info("[Agent 2] Expanded '%s' to topics: %s", query, topics)
        return topics[:n]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("[Agent 2] Topic expansion parse failed: %s", exc)
        return [query.split()[-1], query, f"AI safety {query.split()[-1]}"][:n]


def _fetch_papers_for_topic(topic: str, max_results: int) -> list[Paper]:
    arxiv_papers = fetch_arxiv_papers(topic, max_results=max_results // 2)
    s2_papers = get_influential_papers(topic, limit=max_results // 2)
    return arxiv_papers + s2_papers


def run_broader_context_agent(
    query: str,
    exclude_paper_ids: Optional[set[str]] = None,
) -> AgentResult:
    exclude_ids = exclude_paper_ids or set()
    logger.info("[Agent 2] Running broader context search for: '%s'", query)
    broader_topics = _expand_to_related_topics(query)
    all_papers: list[Paper] = []
    seen_ids: set[str] = set(exclude_ids)

    for topic in broader_topics:
        papers = _fetch_papers_for_topic(
            topic,
            max_results=settings.semantic_scholar_max_results,
        )
        for paper in papers:
            if paper.paper_id not in seen_ids:
                seen_ids.add(paper.paper_id)
                all_papers.append(paper)
    scored = _score_broader_relevance(all_papers, query, broader_topics)
    top_papers = scored[:settings.agent2_final_papers]

    return AgentResult(
        agent_name="BroaderContextAgent",
        papers=top_papers,
        reasoning=(
            f"Expanded '{query}' into {len(broader_topics)} related areas: "
            f"{', '.join(broader_topics)}. "
            f"Fetched {len(all_papers)} unique papers, returning top {len(top_papers)}."
        ),
        metadata={"expanded_topics": broader_topics},
    )


def _score_broader_relevance(
    papers: list[Paper],
    original_query: str,
    broader_topics: list[str],
) -> list[Paper]:
    if not papers:
        return []

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )

    topics_str = ", ".join(broader_topics)
    paper_list = "\n\n".join(
        f"[{i}] Title: {p.title}\nAbstract: {p.abstract[:350]}"
        for i, p in enumerate(papers[:20])   # Cap at 20 to avoid token limits
    )

    prompt = f"""
You are evaluating papers for a researcher working on: "{original_query}"

These papers come from the broader research areas: {topics_str}

Score each paper from 0–10 based on:
- Is it a foundational or widely-cited paper in the broader area? (higher score)
- Does it directly inform research on "{original_query}"? (higher score)
- Is it recent and still actively referenced? (bonus)
- Is it clearly tangential or only keyword-adjacent? (lower score)

Papers:
{paper_list}

Return ONLY valid JSON: {{"scores": [score_0, score_1, ...]}}
"""

    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        scores = data.get("scores", [])

        for i, paper in enumerate(papers[:20]):
            if i < len(scores):
                paper.relevance_score = float(scores[i]) / 10.0

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("[Agent 2] Broader scoring failed: %s", exc)

    return sorted(papers, key=lambda p: p.relevance_score, reverse=True)
