import logging
import time
from typing import Optional

import httpx
from langchain_core.tools import tool

from models import Paper
from config import settings

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"

PAPER_FIELDS = "paperId,title,authors,abstract,year,externalIds,fieldsOfStudy,influentialCitationCount,url"


def _s2_result_to_paper(data: dict) -> Optional[Paper]:
    """
    Map a Semantic Scholar API response dict to our Paper model.

    WHY return Optional?  S2 sometimes returns records with missing titles
    or abstracts (conference proceedings, book chapters).  We skip those
    rather than crashing.
    """
    title = data.get("title") or ""
    abstract = data.get("abstract") or ""
    if not title or not abstract:
        return None

    authors = [a.get("name", "") for a in data.get("authors", [])]
    arxiv_id = (data.get("externalIds") or {}).get("ArXiv", "")
    s2_url = (
        data.get("url")
        or f"https://www.semanticscholar.org/paper/{data.get('paperId','')}"
    )

    return Paper(
        paper_id=arxiv_id or data.get("paperId", ""),
        title=title.strip(),
        authors=authors,
        abstract=abstract.strip(),
        url=s2_url,
        source="semantic_scholar",
        published_date=str(data.get("year")) if data.get("year") else None,
        categories=data.get("fieldsOfStudy") or [],
        relevance_score=0.0,
    )


def search_semantic_scholar(
    query: str,
    max_results: int = 8,
    fields_of_study: Optional[list[str]] = None,
) -> list[Paper]:
    """
    Search Semantic Scholar for papers matching a query.

    Args:
        query:           Free-text search query.
        max_results:     Max papers to return.
        fields_of_study: Optional filter, e.g. ["Computer Science"].

    WHY httpx over requests?
      httpx supports both sync and async, has a cleaner API, and is the
      direction the Python ecosystem is heading.  requests is sync-only.
    """
    params: dict = {
        "query": query,
        "limit": min(max_results, 100),  # S2 API cap is 100
        "fields": PAPER_FIELDS,
    }
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    papers: list[Paper] = []
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(f"{S2_BASE}/paper/search", params=params)
            resp.raise_for_status()
            results = resp.json().get("data", [])
            for item in results:
                p = _s2_result_to_paper(item)
                if p:
                    papers.append(p)
        time.sleep(1.0)  # Polite rate-limiting — 1 req/s without an API key
    except httpx.HTTPStatusError as exc:
        logger.warning("S2 HTTP error for '%s': %s", query, exc)
    except Exception as exc:
        logger.warning("S2 fetch failed for '%s': %s", query, exc)

    logger.info("Semantic Scholar returned %d papers for '%s'", len(papers), query)
    return papers


def get_influential_papers(topic: str, limit: int = 5) -> list[Paper]:
    """
    Retrieve highly-cited (influential) papers in a topic area.

    WHY influential citation count?
      Influential citations (S2's metric) track papers cited specifically
      because they introduced a key method or result — a better proxy for
      foundational importance than raw citation counts.
    """
    params = {
        "query": topic,
        "limit": limit * 3,  # Fetch extra so we can filter by influence
        "fields": PAPER_FIELDS,
    }
    papers: list[Paper] = []
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(f"{S2_BASE}/paper/search", params=params)
            resp.raise_for_status()
            results = resp.json().get("data", [])

        # Sort by influential citation count descending
        results.sort(
            key=lambda x: x.get("influentialCitationCount") or 0,
            reverse=True,
        )
        for item in results[:limit]:
            p = _s2_result_to_paper(item)
            if p:
                papers.append(p)
        time.sleep(1.0)
    except Exception as exc:
        logger.warning("S2 influential fetch failed for '%s': %s", topic, exc)

    return papers


# ── LangChain Tool wrapper ────────────────────────────────────────────────────


@tool
def semantic_scholar_search_tool(query: str, max_results: Optional[int] = None) -> str:
    """
    Search Semantic Scholar for academic papers matching the query.

    Particularly useful for finding foundational or highly-cited papers
    in a broader research area.  Returns titles, authors, abstracts, and
    publication years.

    Args:
        query:       Natural-language search query.
        max_results: Maximum number of papers to return.
    """
    n = max_results or settings.semantic_scholar_max_results
    papers = search_semantic_scholar(query, max_results=n)

    if not papers:
        return f"No papers found on Semantic Scholar for: '{query}'"

    lines = [f"Found {len(papers)} papers on Semantic Scholar for '{query}':\n"]
    for i, p in enumerate(papers, 1):
        lines.append(
            f"{i}. [{p.paper_id}] {p.title}\n"
            f"   Authors: {', '.join(p.authors[:3])}\n"
            f"   Year: {p.published_date}\n"
            f"   Abstract: {p.abstract[:300]}...\n"
            f"   URL: {p.url}\n"
        )
    return "\n".join(lines)
