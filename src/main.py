"""Main CLI entrypoint for the Legal GraphRAG pipeline.

Run with:

    python -m src.main --help

Part 1 intentionally exposes the final command surface as stubs. Later parts
will replace each stub with the real implementation while preserving the same
CLI interface.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import Settings, configure_logging, get_settings, log_startup_summary

app = typer.Typer(
    name="legal-graphrag-pipeline",
    help="CLI for the Oman legal documents GraphRAG pipeline.",
    add_completion=False,
)

console = Console()


class OutputFormat(str, Enum):
    """Supported output formats for future commands."""

    text = "text"
    json = "json"


@app.callback()
def cli_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """Initialize global CLI behavior."""

    settings = get_settings()
    configure_logging("DEBUG" if verbose else settings.log_level)

    if verbose:
        log_startup_summary(settings)


@app.command("scrape")
def scrape(
    max_pages: Optional[int] = typer.Option(
        None,
        "--max-pages",
        help="Maximum number of pages to crawl in this run. Defaults to MAX_PAGES.",
    ),
    reset_checkpoint: bool = typer.Option(
        False,
        "--reset-checkpoint",
        help="Reset crawler checkpoint before scraping.",
    ),
) -> None:
    """Scrape qanoon.om and save raw/Markdown-ready artifacts.

    Part 1 stub:
    The real crawler will be implemented in later parts.
    """

    settings = get_settings()
    effective_max_pages = max_pages or settings.max_pages

    logger.info("Scrape command invoked.")
    logger.info("Source: {}", settings.qanoon_base_url)
    logger.info("English source: {}", settings.decree_base_url)
    logger.info("Max pages: {}", effective_max_pages)
    logger.info("Reset checkpoint: {}", reset_checkpoint)

    console.print(
        Panel(
            "\n".join(
                [
                    "[bold yellow]Part 1 stub[/bold yellow]",
                    "Scraper is not implemented yet.",
                    "",
                    f"Configured source: {settings.qanoon_base_url}",
                    f"English source: {settings.decree_base_url}",
                    f"Raw output directory: {settings.raw_dir}",
                    f"Markdown output directory: {settings.markdown_dir}",
                    f"Effective max pages: {effective_max_pages}",
                    f"Reset checkpoint requested: {reset_checkpoint}",
                ]
            ),
            title="scrape",
            border_style="yellow",
        )
    )


@app.command("ingest")
def ingest(
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Maximum number of local parsed documents to ingest.",
    ),
) -> None:
    """Ingest parsed documents into Neo4j.

    Part 1 stub:
    Neo4j schema and ingestion will be implemented in later parts.
    """

    settings = get_settings()

    logger.info("Ingest command invoked.")
    logger.info("Neo4j URI: {}", settings.neo4j_uri)
    logger.info("Neo4j database: {}", settings.neo4j_database)
    logger.info("Limit: {}", limit)

    console.print(
        Panel(
            "\n".join(
                [
                    "[bold yellow]Part 1 stub[/bold yellow]",
                    "Graph ingestion is not implemented yet.",
                    "",
                    f"Neo4j URI: {settings.neo4j_uri}",
                    f"Neo4j database: {settings.neo4j_database}",
                    f"Markdown input directory: {settings.markdown_dir}",
                    f"Limit: {limit}",
                ]
            ),
            title="ingest",
            border_style="yellow",
        )
    )


@app.command("extract-topics")
def extract_topics(
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Maximum number of documents to process.",
    ),
    only_without_topics: bool = typer.Option(
        True,
        "--only-without-topics/--all-documents",
        help="Process only documents that do not already have topic links.",
    ),
) -> None:
    """Extract legal topics and link Topic nodes to Document nodes.

    Part 1 stub:
    LLM and fallback topic extraction will be implemented in later parts.
    """

    settings = get_settings()

    logger.info("Extract-topics command invoked.")
    logger.info("Topic LLM provider: {}", settings.topic_llm_provider)
    logger.info("OpenAI configured: {}", settings.is_openai_available)
    logger.info("Limit: {}", limit)
    logger.info("Only without topics: {}", only_without_topics)

    console.print(
        Panel(
            "\n".join(
                [
                    "[bold yellow]Part 1 stub[/bold yellow]",
                    "Topic extraction is not implemented yet.",
                    "",
                    f"Topic provider: {settings.topic_llm_provider}",
                    f"OpenAI configured: {settings.is_openai_available}",
                    f"Limit: {limit}",
                    f"Only without topics: {only_without_topics}",
                ]
            ),
            title="extract-topics",
            border_style="yellow",
        )
    )


@app.command("embed")
def embed(
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Maximum number of documents/topics to embed.",
    ),
) -> None:
    """Generate embeddings for chunks and topics.

    Part 1 stub:
    Chunking, embedding provider, and embedding cache will be implemented in
    later parts.
    """

    settings = get_settings()

    logger.info("Embed command invoked.")
    logger.info("Embedding provider: {}", settings.embedding_provider)
    logger.info("Embedding dimensions: {}", settings.active_embedding_dimensions)
    logger.info("Embedding cache path: {}", settings.embedding_cache_path)
    logger.info("Limit: {}", limit)

    console.print(
        Panel(
            "\n".join(
                [
                    "[bold yellow]Part 1 stub[/bold yellow]",
                    "Embedding generation is not implemented yet.",
                    "",
                    f"Embedding provider: {settings.embedding_provider}",
                    f"Embedding dimensions: {settings.active_embedding_dimensions}",
                    f"SentenceTransformer model: {settings.sentence_transformer_model}",
                    f"Embedding cache path: {settings.embedding_cache_path}",
                    f"Limit: {limit}",
                ]
            ),
            title="embed",
            border_style="yellow",
        )
    )


@app.command("merge-topics")
def merge_topics(
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Cosine similarity threshold for duplicate topic merging.",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--apply",
        help="Show merge plan without mutating the graph. Use --apply to mutate.",
    ),
) -> None:
    """Merge duplicate Topic nodes by embedding similarity.

    Part 1 stub:
    Topic merging will be implemented in later parts.
    """

    settings = get_settings()
    effective_threshold = threshold if threshold is not None else settings.topic_merge_similarity

    logger.info("Merge-topics command invoked.")
    logger.info("Threshold: {}", effective_threshold)
    logger.info("Dry run: {}", dry_run)

    console.print(
        Panel(
            "\n".join(
                [
                    "[bold yellow]Part 1 stub[/bold yellow]",
                    "Topic merging is not implemented yet.",
                    "",
                    f"Similarity threshold: {effective_threshold}",
                    f"Dry run: {dry_run}",
                ]
            ),
            title="merge-topics",
            border_style="yellow",
        )
    )


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Natural-language legal research query."),
    top_k: int = typer.Option(
        50,
        "--top-k",
        min=1,
        help="Candidate pool size before graph expansion and reranking.",
    ),
    final_k: int = typer.Option(
        5,
        "--final-k",
        min=1,
        help="Number of final contexts to use for answer synthesis.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--format",
        help="Output format.",
    ),
) -> None:
    """Run hybrid GraphRAG search.

    Part 1 stub:
    Hybrid search will be implemented in later parts.
    """

    settings = get_settings()

    logger.info("Search command invoked.")
    logger.info("Query: {}", query)
    logger.info("Top K: {}", top_k)
    logger.info("Final K: {}", final_k)
    logger.info("Output format: {}", output_format.value)
    logger.info("Vector weight: {}", settings.hybrid_vector_weight)
    logger.info("Keyword weight: {}", settings.hybrid_keyword_weight)
    logger.info("Reranker enabled: {}", settings.reranker_enabled)

    table = Table(title="Search Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Field")
    table.add_column("Value")

    table.add_row("Status", "Part 1 stub")
    table.add_row("Query", query)
    table.add_row("Top K", str(top_k))
    table.add_row("Final K", str(final_k))
    table.add_row("Output format", output_format.value)
    table.add_row("Neo4j URI", settings.neo4j_uri)
    table.add_row("Embedding provider", settings.embedding_provider)
    table.add_row("Vector weight", str(settings.hybrid_vector_weight))
    table.add_row("Keyword weight", str(settings.hybrid_keyword_weight))
    table.add_row("Reranker enabled", str(settings.reranker_enabled))

    console.print(table)
    console.print(
        Panel(
            "Hybrid search is not implemented yet. Later parts will add vector search, keyword fallback, graph traversal, reranking, and answer synthesis.",
            title="search",
            border_style="yellow",
        )
    )


@app.command("doctor")
def doctor() -> None:
    """Print configuration diagnostics for local development.

    This helper command is useful even in Part 1 because it verifies that the
    settings layer, environment file, and local data directories are working.
    """

    settings = get_settings()
    log_startup_summary(settings)

    table = Table(title="Legal GraphRAG Configuration", show_header=True, header_style="bold green")
    table.add_column("Setting")
    table.add_column("Value")

    safe_values = {
        "APP_ENV": settings.app_env,
        "LOG_LEVEL": settings.log_level,
        "DATA_DIR": str(settings.data_dir),
        "RAW_DIR": str(settings.raw_dir),
        "MARKDOWN_DIR": str(settings.markdown_dir),
        "SAMPLE_OUTPUT_DIR": str(settings.sample_output_dir),
        "CHECKPOINT_FILE": str(settings.checkpoint_file),
        "QANOON_BASE_URL": settings.qanoon_base_url,
        "DECREE_BASE_URL": settings.decree_base_url,
        "SCRAPE_ENGLISH": str(settings.scrape_english),
        "USE_PLAYWRIGHT": str(settings.use_playwright),
        "MAX_PAGES": str(settings.max_pages),
        "REQUEST_TIMEOUT_SECONDS": str(settings.request_timeout_seconds),
        "REQUEST_RETRIES": str(settings.request_retries),
        "THROTTLE_MIN_SECONDS": str(settings.throttle_min_seconds),
        "THROTTLE_MAX_SECONDS": str(settings.throttle_max_seconds),
        "NEO4J_URI": settings.neo4j_uri,
        "NEO4J_USER": settings.neo4j_user,
        "NEO4J_DATABASE": settings.neo4j_database,
        "NEO4J_VECTOR_DIMENSIONS": str(settings.neo4j_vector_dimensions),
        "EMBEDDING_PROVIDER": settings.embedding_provider,
        "SENTENCE_TRANSFORMER_MODEL": settings.sentence_transformer_model,
        "TOPIC_LLM_PROVIDER": settings.topic_llm_provider,
        "SYNTHESIS_PROVIDER": settings.synthesis_provider,
        "CHUNK_MIN_TOKENS": str(settings.chunk_min_tokens),
        "CHUNK_MAX_TOKENS": str(settings.chunk_max_tokens),
        "CHUNK_OVERLAP_TOKENS": str(settings.chunk_overlap_tokens),
        "HYBRID_VECTOR_WEIGHT": str(settings.hybrid_vector_weight),
        "HYBRID_KEYWORD_WEIGHT": str(settings.hybrid_keyword_weight),
        "RERANKER_ENABLED": str(settings.reranker_enabled),
        "TOPIC_MERGE_SIMILARITY": str(settings.topic_merge_similarity),
        "OPENAI_CONFIGURED": str(settings.is_openai_available),
    }

    for key, value in safe_values.items():
        table.add_row(key, value)

    console.print(table)


def main() -> None:
    """Entrypoint wrapper used by `python -m src.main`."""

    app()


if __name__ == "__main__":
    main()