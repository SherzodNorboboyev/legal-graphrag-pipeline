"""Executable CLI search client for the Legal GraphRAG pipeline.

Usage:
    python -m src.search_client "What are the laws about taxation?"

The client displays:
- vector-matched chunks
- keyword matches
- parent document metadata
- linked topics
- final summarized answer
"""

from __future__ import annotations

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import configure_logging, get_settings
from src.retrieval.hybrid_search import HybridSearch, RetrievalCandidate
from src.vector_ops.embeddings import get_embedding_provider

console = Console()


def render_candidate_table(
    title: str,
    candidates: list[RetrievalCandidate],
    *,
    score_field: str,
    max_rows: int = 8,
) -> Table:
    """Render candidates as a Rich table."""

    table = Table(title=title, show_lines=True)
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Document")
    table.add_column("Topics")
    table.add_column("Evidence")

    for rank, candidate in enumerate(candidates[:max_rows], start=1):
        score = getattr(candidate, score_field, 0.0) or 0.0
        topics = ", ".join(candidate.topics[:6]) if candidate.topics else "No linked topics"

        document = (
            f"{candidate.document_title or 'Unknown document'}\n"
            f"number: {candidate.document_number or 'N/A'} | "
            f"type: {candidate.document_type or 'N/A'} | "
            f"date: {candidate.document_date or 'N/A'}\n"
            f"url: {candidate.source_url or 'N/A'}"
        )

        table.add_row(
            str(rank),
            f"{float(score):.4f}",
            document,
            topics,
            candidate.evidence_preview(),
        )

    return table


def search_cli(
    query: str = typer.Argument(..., help="Natural-language legal research query."),
    top_k: int = typer.Option(50, "--top-k", min=1, help="Candidate pool size before reranking."),
    top_n: int = typer.Option(5, "--top-n", min=1, help="Final number of contexts for answer synthesis."),
    show_debug: bool = typer.Option(False, "--debug", help="Enable debug logging and timing output."),
) -> None:
    """Run hybrid GraphRAG search from the terminal."""

    settings = get_settings()
    configure_logging("DEBUG" if show_debug else settings.log_level)

    try:
        embedder = get_embedding_provider(settings)

        with HybridSearch(settings, embedder=embedder) as search:
            response = search.search(query, top_k=top_k, top_n=top_n)

        console.print(
            Panel(
                f"[bold]Query:[/bold] {query}\n"
                f"[bold]Top K:[/bold] {top_k}\n"
                f"[bold]Top N:[/bold] {top_n}",
                title="Legal GraphRAG Search",
                border_style="blue",
            )
        )

        console.print(
            render_candidate_table(
                "Vector Matched Chunks",
                response.vector_candidates,
                score_field="vector_score",
                max_rows=min(top_n, 8),
            )
        )

        console.print(
            render_candidate_table(
                "Keyword Matches",
                response.keyword_candidates,
                score_field="keyword_score",
                max_rows=min(top_n, 8),
            )
        )

        console.print(
            render_candidate_table(
                "Final Hybrid / Reranked Results",
                response.final_candidates,
                score_field="combined_score",
                max_rows=top_n,
            )
        )

        console.print(
            Panel(
                response.answer,
                title="Final Answer",
                border_style="green",
            )
        )

        if show_debug:
            timing_table = Table(title="Timings", show_header=True)
            timing_table.add_column("Stage")
            timing_table.add_column("Milliseconds", justify="right")

            for stage, milliseconds in response.timings_ms.items():
                timing_table.add_row(stage, f"{milliseconds:.3f}")

            console.print(timing_table)

    except Exception as exc:  # noqa: BLE001 - CLI should show clear error and exit non-zero
        logger.exception("Search failed: {}", exc)
        console.print(
            Panel(
                str(exc),
                title="Search failed",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    typer.run(search_cli)