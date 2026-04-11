import logging
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import settings
from memory.vector_store import ResearchVectorStore
from models import AgentResult, Paper, ResearchSession
from agents.focused_search_agent import run_focused_search_agent
from agents.broader_context_agent import run_broader_context_agent
from agents.interest_map_agent import run_interest_map_agent

logger = logging.getLogger(__name__)
console = Console()


def _deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    seen: set[str] = set()
    unique: list[Paper] = []
    for p in papers:
        if p.paper_id not in seen and p.paper_id:
            seen.add(p.paper_id)
            unique.append(p)
    return unique


def run_research_session(
    query: str, vector_store: ResearchVectorStore
) -> ResearchSession:
    logger.info("=== Starting research session for: '%s' ===", query)
    session_start = time.time()

    session = ResearchSession(query=query)
    console.print("\n[bold cyan]▶ Agent 1: Focused Search[/bold cyan]")
    try:
        a1_result: AgentResult = run_focused_search_agent(query)
        session.focused_papers = a1_result.papers
        console.print(
            f"  [green]✓[/green] Found {len(a1_result.papers)} focused papers."
        )
        console.print(f"  [dim]{a1_result.reasoning}[/dim]")
    except Exception as exc:
        logger.error("[Agent 1] Failed: %s", exc, exc_info=True)
        console.print(f"  [red]✗ Agent 1 failed: {exc}[/red]")
        a1_result = AgentResult(agent_name="FocusedSearchAgent", papers=[])
    console.print("\n[bold yellow]▶ Agent 2: Broader Context[/bold yellow]")
    a1_paper_ids = {p.paper_id for p in a1_result.papers}
    try:
        a2_result: AgentResult = run_broader_context_agent(
            query,
            exclude_paper_ids=a1_paper_ids,
        )
        session.broader_papers = a2_result.papers
        console.print(
            f"  [green]✓[/green] Found {len(a2_result.papers)} broader papers."
        )
        console.print(f"  [dim]{a2_result.reasoning}[/dim]")
    except Exception as exc:
        logger.error("[Agent 2] Failed: %s", exc, exc_info=True)
        console.print(f"  [red]✗ Agent 2 failed: {exc}[/red]")
        a2_result = AgentResult(agent_name="BroaderContextAgent", papers=[])

    console.print("\n[bold magenta]▶ Agent 3: Interest Map[/bold magenta]")
    all_current_papers = a1_result.papers + a2_result.papers
    try:
        a3_result: AgentResult = run_interest_map_agent(
            query,
            current_session_papers=all_current_papers,
            vector_store=vector_store,
        )
        session.interest_papers = a3_result.papers
        session.interest_map_summary = a3_result.reasoning
        console.print(
            f"  [green]✓[/green] Found {len(a3_result.papers)} historically-related papers."
        )
    except Exception as exc:
        logger.error("[Agent 3] Failed: %s", exc, exc_info=True)
        console.print(f"  [red]✗ Agent 3 failed: {exc}[/red]")

    elapsed = time.time() - session_start
    logger.info("Session completed in %.1f seconds.", elapsed)

    return session


def display_session_results(session: ResearchSession) -> None:
    console.print()
    console.print(
        Panel(
            f"[bold]Research Session Results[/bold]\nQuery: [cyan]{session.query}[/cyan]",
            box=box.ROUNDED,
        )
    )

    def _paper_table(title: str, papers: list[Paper], color: str) -> None:
        if not papers:
            console.print(f"\n[{color}]{title}[/{color}]: No papers found.")
            return
        table = Table(title=title, box=box.SIMPLE_HEAVY, title_style=color)
        table.add_column("Title", style="bold", max_width=55)
        table.add_column("Authors", max_width=25)
        table.add_column("Date", max_width=12)
        table.add_column("Score", justify="right", max_width=6)
        table.add_column("Source", max_width=12)

        for p in papers:
            authors_str = ", ".join(p.authors[:2])
            if len(p.authors) > 2:
                authors_str += " et al."
            table.add_row(
                p.title[:55],
                authors_str[:25],
                str(p.published_date) or "—",
                f"{p.relevance_score:.2f}",
                p.source,
            )
        console.print(table)

    _paper_table("Focused Papers (Agent 1)", session.focused_papers, "cyan")
    _paper_table(
        "Broader Context Papers (Agent 2)", session.broader_papers, "yellow"
    )
    _paper_table(
        "From Your Research History (Agent 3)", session.interest_papers, "magenta"
    )

    if session.interest_map_summary:
        console.print(
            Panel(
                session.interest_map_summary,
                title="[magenta]Your Research Interest Profile[/magenta]",
                box=box.ROUNDED,
            )
        )
