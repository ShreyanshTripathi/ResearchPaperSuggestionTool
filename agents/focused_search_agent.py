import json
import logging
from typing import Any

from langchain_classic.agents import AgentExecutor, create_openai_tools_agent, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from config import settings
from models import AgentResult, Paper
from tools.arxiv_tool import arxiv_search_tool, fetch_arxiv_papers

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a focused academic literature search agent.

Your task: find papers that are DIRECTLY and STRICTLY about the user's query topic.
Do NOT include papers that merely cite or mention the topic in passing.

Steps:
1. Formulate 1-2 precise arXiv search queries using the arxiv_search_tool.
   Use field specifiers when helpful: ti: (title), abs: (abstract), cat: (category).
2. Review the results and identify papers that exactly match the topic.
3. If the initial results are too broad, run a more specific follow-up query.
4. Return your final assessment of the most relevant papers.

Be strict: a paper about "reinforcement learning for robotics" is NOT relevant
to a query about "RL-based machine unlearning" unless it discusses unlearning.
"""


def build_focused_agent() -> AgentExecutor:
    llm = ChatGroq(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = [arxiv_search_tool]

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=4,
        handle_parsing_errors=True,
        verbose=False,  # Set True for debugging to see each reasoning step
    )


def _llm_rerank_papers(papers: list[Paper], query: str) -> list[Paper]:
    if not papers:
        return []

    llm = ChatGroq(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.api_key,
    )

    paper_list = "\n\n".join(
        f"[{i}] Title: {p.title}\nAbstract: {p.abstract[:400]}"
        for i, p in enumerate(papers)
    )

    prompt = f"""
Rate each paper's relevance to the query: "{query}"

Score from 0 (irrelevant) to 10 (perfectly on-topic).
A score of 8+ means the paper's MAIN CONTRIBUTION is this exact topic.
A score of 5-7 means the paper addresses the topic but as a secondary theme.
A score below 5 means the paper only mentions the topic in passing.

Papers:
{paper_list}

Return ONLY valid JSON in this format:
{{"scores": [score_for_paper_0, score_for_paper_1, ...]}}
No explanation, no markdown, just the JSON object.
"""

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        scores = data.get("scores", [])

        for i, paper in enumerate(papers):
            if i < len(scores):
                paper.relevance_score = float(scores[i]) / 10.0   # normalise to 0–1

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Re-ranking parse failed: %s — using original order.", exc)

    # Sort descending by relevance score
    return sorted(papers, key=lambda p: p.relevance_score, reverse=True)


def run_focused_search_agent(query: str) -> AgentResult:
    logger.info("[Agent 1] Running focused search for: '%s'", query)
    raw_papers = fetch_arxiv_papers(query, max_results=settings.arxiv_max_results)
    reranked = _llm_rerank_papers(raw_papers, query)

    if any(p.relevance_score > 0.0 for p in reranked):
        filtered = [p for p in reranked if p.relevance_score >= 0.5]
    else:
        filtered = reranked

    if len(filtered) < 3:
        logger.info("[Agent 1] Fewer than 3 relevant papers found; invoking ReAct agent for query refinement.")
        agent = build_focused_agent()
        agent_output = agent.invoke({"input": f"Find papers strictly about: {query}"})
        fallback = fetch_arxiv_papers(f"{query} cs.LG", max_results=settings.arxiv_max_results)
        reranked2 = _llm_rerank_papers(fallback, query)
        filtered = reranked2[:settings.agent1_final_papers]

    top_papers = filtered[:settings.agent1_final_papers]

    return AgentResult(
        agent_name="FocusedSearchAgent",
        papers=top_papers,
        reasoning=f"Retrieved {len(raw_papers)} papers from arXiv, re-ranked by LLM, kept top {len(top_papers)} with relevance ≥ 0.5.",
    )
