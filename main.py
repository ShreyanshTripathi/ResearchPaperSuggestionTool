
import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

load_dotenv()

from config import settings
from memory.vector_store import ResearchVectorStore
from orchestrator import display_session_results, run_research_session

console = Console()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("llama_index").setLevel(logging.WARNING)


def check_prerequisites() -> bool:
    if not settings.openai_api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY is not set.\n"
            "Create a .env file with OPENAI_API_KEY=sk-... or export it in your shell.",
        )
        return False
    return True


def interactive_loop(vector_store: ResearchVectorStore) -> None:
    console.print(
        "\n[bold cyan]Research Agent Framework[/bold cyan]\n"
        "Type your research topic and press Enter. "
        "Commands: [bold]quit[/bold], [bold]history[/bold], [bold]clear[/bold]\n"
    )

    while True:
        try:
            query = console.input("[bold green]Topic > [/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if query.lower() in ("history", "h"):
            _display_interest_map(vector_store)
            continue

        if query.lower() in ("clear", "reset"):
            _clear_history(vector_store)
            continue

        session = run_research_session(query, vector_store)
        display_session_results(session)


def _display_interest_map(vector_store: ResearchVectorStore) -> None:
    interest_map = vector_store.build_interest_map()
    if not interest_map:
        console.print("[dim]No search history yet.[/dim]")
        return

    console.print("\n[bold magenta]Your Research Interest Map[/bold magenta]")
    console.print("[dim](topics weighted by recency and frequency)[/dim]\n")
    for rank, (topic, weight) in enumerate(list(interest_map.items())[:15], 1):
        bar_len = int(weight * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        console.print(f"  {rank:2}. [cyan]{topic:<40}[/cyan] {bar} {weight:.3f}")


def _clear_history(vector_store: ResearchVectorStore) -> None:
    try:
        vector_store._client.delete_collection("search_history")
        vector_store._client.delete_collection("papers_seen")
        console.print("[green]History cleared.[/green]")
    except Exception as exc:
        console.print(f"[red]Failed to clear history: {exc}[/red]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research Agent Framework — multi-agent academic paper discovery",
    )
    parser.add_argument("--query", "-q", help="Research topic query (skips interactive prompt)")
    parser.add_argument("--show-history", action="store_true", help="Display the interest map and exit")
    parser.add_argument("--clear-history", action="store_true", help="Wipe search history and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    if not check_prerequisites():
        sys.exit(1)

    vector_store = ResearchVectorStore()

    if args.show_history:
        _display_interest_map(vector_store)
        return

    if args.clear_history:
        _clear_history(vector_store)
        return

    if args.query:
        session = run_research_session(args.query, vector_store)
        display_session_results(session)
    else:
        interactive_loop(vector_store)


if __name__ == "__main__":
    main()
