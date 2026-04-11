import logging
from typing import Optional

import arxiv
from langchain_core.tools import tool

from models import Paper
from config import settings

logger = logging.getLogger(__name__)


def _parse_arxiv_result(result: arxiv.Result) -> Paper:
    """
    Convert an arxiv.Result into our Paper model.

    WHY a separate parser?
      Keeps the mapping logic in one place.  If arXiv changes its API,
      we update only this function.
    """
    return Paper(
        paper_id=result.entry_id.split("/")[-1],   # e.g. "2310.12345v2" → "2310.12345v2"
        title=result.title.strip().replace("\n", " "),
        authors=[a.name for a in result.authors],
        abstract=result.summary.strip().replace("\n", " "),
        url=result.entry_id,
        source="arxiv",
        published_date=result.published.date().isoformat() if result.published else None,
        categories=result.categories,
        relevance_score=0.0,  # filled in by LLM re-ranker later
    )


def fetch_arxiv_papers(
    query: str,
    max_results: int = 10,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
) -> list[Paper]:
    """
    Fetch papers from arXiv for a given query string.

    Args:
        query:       arXiv-syntax query string. Supports field tags like
                     "ti:unlearning AND cat:cs.LG" but plain text also works.
        max_results: Upper bound on returned papers.
        sort_by:     arxiv.SortCriterion.Relevance is best for semantic search;
                     SortCriterion.SubmittedDate is best for "latest papers".

    WHY arxiv.Client with custom retry?
      The default Client handles HTTP 429 (rate limit) transparently with
      exponential back-off.  Without it, burst requests fail silently.
    """
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3.0,     # Polite: arXiv asks for ≥3 s between requests
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
    )

    papers: list[Paper] = []
    try:
        for result in client.results(search):
            papers.append(_parse_arxiv_result(result))
    except Exception as exc:
        logger.warning("arXiv fetch failed for query '%s': %s", query, exc)

    logger.info("arXiv returned %d papers for '%s'", len(papers), query)
    return papers


# ── LangChain Tool wrapper ────────────────────────────────────────────────────
# WHY @tool decorator?
#   It introspects the docstring + type hints to build a JSON schema that the
#   LLM (via function calling) can read to decide when and how to invoke this
#   tool.  No manual schema writing needed.

@tool
def arxiv_search_tool(query: str, max_results: Optional[int] = None) -> str:
    """
    Search arXiv for academic papers matching the query.

    Returns a formatted string of paper titles, authors, and abstracts
    that the agent can read and reason about.

    Args:
        query:       Natural-language or arXiv-syntax search query.
        max_results: Maximum number of papers to return (default from settings).
    """
    n = max_results or settings.arxiv_max_results
    papers = fetch_arxiv_papers(query, max_results=n)

    if not papers:
        return f"No papers found on arXiv for query: '{query}'"

    lines = [f"Found {len(papers)} papers on arXiv for '{query}':\n"]
    for i, p in enumerate(papers, 1):
        lines.append(
            f"{i}. [{p.paper_id}] {p.title}\n"
            f"   Authors: {', '.join(p.authors[:3])}\n"
            f"   Date: {p.published_date}\n"
            f"   Abstract: {p.abstract[:300]}...\n"
            f"   URL: {p.url}\n"
        )
    return "\n".join(lines)
