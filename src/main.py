"""Main Typer CLI entrypoint for the Legal GraphRAG pipeline.

Run examples:

    python -m src.main setup-schema
    python -m src.main scrape --max-pages 25
    python -m src.main ingest
    python -m src.main extract-topics
    python -m src.main embed
    python -m src.main merge-topics --dry-run
    python -m src.main search "What are the laws about taxation?"

This module intentionally keeps imports for heavy components inside commands so
`python -m src.main --help` remains fast and does not require model loading.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

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
    """Supported CLI output formats."""

    text = "text"
    json = "json"


@app.callback()
def cli_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Explicit log level: DEBUG, INFO, WARNING, ERROR.",
    ),
) -> None:
    """Initialize logging for all CLI commands."""

    settings = get_settings()
    effective_level = "DEBUG" if verbose else (log_level or settings.log_level)
    configure_logging(effective_level)

    if verbose:
        log_startup_summary(settings)


@app.command("doctor")
def doctor() -> None:
    """Print sanitized runtime configuration diagnostics."""

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
        "EMBEDDING_CACHE_PATH": str(settings.embedding_cache_path),
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
        "CROSS_ENCODER_MODEL": settings.cross_encoder_model,
        "TOPIC_MERGE_SIMILARITY": str(settings.topic_merge_similarity),
        "OPENAI_CONFIGURED": str(settings.is_openai_available),
    }

    for key, value in safe_values.items():
        table.add_row(key, value)

    console.print(table)


@app.command("setup-schema")
def setup_schema(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show schema setup plan without connecting to Neo4j."),
) -> None:
    """Create Neo4j constraints, lookup indexes, full-text indexes, and vector indexes."""

    from src.ingestion.graph_ingest import GraphIngestor

    settings = get_settings()

    if dry_run:
        console.print(
            Panel(
                "\n".join(
                    [
                        "Schema setup dry-run:",
                        "- Unique constraint: Document.id",
                        "- Unique constraint: Topic.normalized_name",
                        "- Unique constraint: Chunk.id",
                        "- Lookup indexes: document number/type/date/source, chunk document/language, topic name",
                        "- Full-text indexes: Document title/content, Chunk text, Topic name",
                        "- Vector indexes: Chunk.embedding and Topic.embedding",
                        f"- Vector dimensions: {settings.active_embedding_dimensions}",
                    ]
                ),
                title="setup-schema",
                border_style="yellow",
            )
        )
        return

    try:
        with GraphIngestor(settings) as ingestor:
            ingestor.verify_connection()
            result = ingestor.ensure_schema()

        console.print(f"[green]Neo4j schema setup complete[/green]: applied={len(result.applied)}, skipped_optional={len(result.skipped_optional)}")

    except Exception as exc:  # noqa: BLE001 - CLI should fail clearly
        logger.exception("Schema setup failed: {}", exc)
        raise typer.Exit(code=1) from exc


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
    dry_run: bool = typer.Option(False, "--dry-run", help="Show crawl configuration without making HTTP requests."),
) -> None:
    """Scrape qanoon.om/decree.om and save raw HTML/PDF plus parsed Markdown JSON artifacts."""

    from src.scraper.checkpoint import CheckpointManager
    from src.scraper.crawler import QanoonCrawler

    settings = get_settings()
    effective_max_pages = max_pages or settings.max_pages

    if dry_run:
        console.print(
            Panel(
                "\n".join(
                    [
                        "Scrape dry-run:",
                        f"- Qanoon source: {settings.qanoon_base_url}",
                        f"- English source: {settings.decree_base_url}",
                        f"- Scrape English: {settings.scrape_english}",
                        f"- Max pages: {effective_max_pages}",
                        f"- Raw output: {settings.raw_dir}",
                        f"- Markdown output: {settings.markdown_dir}",
                        f"- Checkpoint file: {settings.checkpoint_file}",
                        f"- Reset checkpoint: {reset_checkpoint}",
                    ]
                ),
                title="scrape",
                border_style="yellow",
            )
        )
        return

    try:
        if reset_checkpoint:
            CheckpointManager(settings.checkpoint_file).reset()

        crawler = QanoonCrawler(settings)
        checkpoint = asyncio.run(crawler.run(max_pages=effective_max_pages))

        console.print(
            f"[green]Scrape complete[/green]: "
            f"visited={len(checkpoint.visited_urls)}, "
            f"queued={len(checkpoint.queued_urls)}, "
            f"failed={len(checkpoint.failed_urls)}, "
            f"documents={len(checkpoint.documents)}"
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("Scrape failed: {}", exc)
        raise typer.Exit(code=1) from exc


@app.command("ingest")
def ingest(
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Maximum number of local parsed documents to ingest.",
    ),
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        min=1,
        help="Number of documents per Neo4j transaction.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate local files but do not write to Neo4j."),
) -> None:
    """Create schema and ingest parsed document JSON files from data/markdown into Neo4j."""

    from src.ingestion.graph_ingest import GraphIngestor

    settings = get_settings()
    markdown_paths = sorted(settings.markdown_dir.glob("*.json"))

    if limit is not None:
        markdown_paths = markdown_paths[:limit]

    if dry_run:
        console.print(
            Panel(
                "\n".join(
                    [
                        "Ingest dry-run:",
                        f"- Markdown directory: {settings.markdown_dir}",
                        f"- JSON files selected: {len(markdown_paths)}",
                        f"- Batch size: {batch_size}",
                        f"- Neo4j URI: {settings.neo4j_uri}",
                        f"- Neo4j database: {settings.neo4j_database}",
                    ]
                ),
                title="ingest",
                border_style="yellow",
            )
        )
        return

    try:
        with GraphIngestor(settings) as ingestor:
            ingestor.verify_connection()
            ingestor.ensure_schema()
            count = ingestor.ingest_markdown_dir(limit=limit, batch_size=batch_size)

        console.print(f"[green]Ingestion complete[/green]: {count} documents upserted into Neo4j.")

    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion failed: {}", exc)
        raise typer.Exit(code=1) from exc


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
    dry_run: bool = typer.Option(False, "--dry-run", help="Extract topics but do not write HAS_TOPIC links."),
) -> None:
    """Extract legal topics and link Topic nodes to parent Document nodes."""

    from src.ingestion.graph_ingest import GraphIngestor
    from src.llm_agents.topic_extractor import TopicExtractor

    settings = get_settings()

    try:
        extractor = TopicExtractor(settings)
        processed_count = 0
        linked_topic_count = 0

        preview_table = Table(title="Topic Extraction Preview", show_header=True)
        preview_table.add_column("Document")
        preview_table.add_column("Topics")

        with GraphIngestor(settings) as ingestor:
            ingestor.verify_connection()
            ingestor.ensure_schema()

            documents = ingestor.fetch_documents(
                limit=limit,
                only_without_topics=only_without_topics,
            )

            for document in documents:
                topics = extractor.extract(
                    title=document.get("title") or document["id"],
                    content_en=document.get("contentEn"),
                    content_ar=document.get("contentAr"),
                )

                processed_count += 1

                if dry_run:
                    preview_table.add_row(
                        document.get("title") or document["id"],
                        ", ".join(topic.name for topic in topics) or "No topics extracted",
                    )
                    continue

                linked_topic_count += ingestor.link_document_topics(
                    document_id=document["id"],
                    topics=topics,
                )

        if dry_run:
            console.print(preview_table)
            console.print(f"[yellow]Topic extraction dry-run complete[/yellow]: documents={processed_count}")
        else:
            console.print(
                f"[green]Topic extraction complete[/green]: "
                f"documents={processed_count}, topic_links={linked_topic_count}"
            )

    except Exception as exc:  # noqa: BLE001
        logger.exception("Topic extraction failed: {}", exc)
        raise typer.Exit(code=1) from exc


@app.command("embed")
def embed(
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Maximum number of documents/topics to process.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Chunk documents and count work without embedding or writing."),
) -> None:
    """Create Chunk nodes and embeddings for document chunks and Topic nodes."""

    from src.ingestion.graph_ingest import GraphIngestor
    from src.models import Chunk
    from src.vector_ops.chunking import MarkdownChunker
    from src.vector_ops.embeddings import get_embedding_provider

    settings = get_settings()

    try:
        chunker = MarkdownChunker(
            min_tokens=settings.chunk_min_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )

        embedder = None if dry_run else get_embedding_provider(settings)

        document_count = 0
        chunk_count = 0
        topic_embedding_count = 0

        with GraphIngestor(settings) as ingestor:
            ingestor.verify_connection()
            ingestor.ensure_schema()

            documents = ingestor.fetch_documents(limit=limit)

            for document in documents:
                document_id = document["id"]

                for language, content_key in [
                    ("en", "contentEn"),
                    ("ar", "contentAr"),
                    ("fr", "contentFr"),
                ]:
                    markdown = document.get(content_key)
                    if not markdown:
                        continue

                    chunks = chunker.chunk_markdown(
                        markdown,
                        document_id=document_id,
                        language=language,
                    )

                    if not chunks:
                        continue

                    if dry_run:
                        chunk_count += len(chunks)
                        continue

                    assert embedder is not None
                    vectors = embedder.embed_texts([chunk.text for chunk in chunks])

                    embedded_chunks: list[Chunk] = []
                    for chunk, vector in zip(chunks, vectors, strict=True):
                        embedded_chunks.append(
                            Chunk(
                                id=chunk.id,
                                document_id=chunk.document_id,
                                language=chunk.language,
                                text=chunk.text,
                                chunk_index=chunk.chunk_index,
                                embedding=vector,
                                heading_context=chunk.heading_context,
                                token_count=chunk.token_count,
                                metadata=chunk.metadata,
                            )
                        )

                    chunk_count += ingestor.batch_upsert_chunks(embedded_chunks)

                document_count += 1

            topics = ingestor.fetch_topics_without_embeddings(limit=limit)

            if dry_run:
                topic_embedding_count = len(topics)
            else:
                assert embedder is not None
                for topic in topics:
                    topic_text = topic.get("name") or topic.get("normalized_name")
                    normalized_name = topic.get("normalized_name")

                    if not topic_text or not normalized_name:
                        continue

                    vector = embedder.embed_text(topic_text)
                    _upsert_topic_embedding(ingestor, normalized_name, vector)
                    topic_embedding_count += 1

        if dry_run:
            console.print(
                f"[yellow]Embedding dry-run complete[/yellow]: "
                f"documents={document_count}, chunks_to_create={chunk_count}, "
                f"topics_to_embed={topic_embedding_count}"
            )
        else:
            console.print(
                f"[green]Embedding complete[/green]: "
                f"documents={document_count}, chunks={chunk_count}, "
                f"topic_embeddings={topic_embedding_count}"
            )

    except Exception as exc:  # noqa: BLE001
        logger.exception("Embedding failed: {}", exc)
        raise typer.Exit(code=1) from exc


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
    """Merge duplicate Topic nodes by embedding similarity."""

    from src.vector_ops.topic_merging import TopicMerger

    settings = get_settings()
    effective_threshold = threshold if threshold is not None else settings.topic_merge_similarity

    try:
        with TopicMerger(settings) as merger:
            plan = merger.merge_duplicate_topics(
                threshold=effective_threshold,
                dry_run=dry_run,
            )

        console.print(f"[green]Topic merge completed[/green]: {plan.summary()}")

    except Exception as exc:  # noqa: BLE001
        logger.exception("Topic merge failed: {}", exc)
        raise typer.Exit(code=1) from exc


@app.command("communities")
def communities() -> None:
    """Run Louvain community detection using Neo4j GDS if available."""

    from src.vector_ops.topic_merging import TopicMerger

    settings = get_settings()

    try:
        with TopicMerger(settings) as merger:
            result = merger.run_louvain_community_detection()

        if result.get("ok"):
            console.print(f"[green]Community detection complete[/green]: {result}")
        else:
            console.print(f"[yellow]Community detection skipped or failed[/yellow]: {result}")

    except Exception as exc:  # noqa: BLE001
        logger.exception("Community detection failed: {}", exc)
        raise typer.Exit(code=1) from exc


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Natural-language legal research query."),
    top_k: int = typer.Option(50, "--top-k", min=1, help="Candidate pool size before reranking."),
    top_n: int = typer.Option(5, "--top-n", min=1, help="Final number of contexts for answer synthesis."),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format", help="Output format."),
) -> None:
    """Run hybrid GraphRAG search from the main CLI."""

    from src.retrieval.hybrid_search import HybridSearch
    from src.vector_ops.embeddings import get_embedding_provider

    settings = get_settings()

    try:
        embedder = get_embedding_provider(settings)

        with HybridSearch(settings, embedder=embedder) as searcher:
            response = searcher.search(query, top_k=top_k, top_n=top_n)

        if output_format == OutputFormat.json:
            console.print_json(
                data={
                    "query": response.query,
                    "answer": response.answer,
                    "timings_ms": response.timings_ms,
                    "final_candidates": [asdict(candidate) for candidate in response.final_candidates],
                }
            )
            return

        console.print(
            Panel(
                f"[bold]Query:[/bold] {query}\n"
                f"[bold]Candidates:[/bold] top_k={top_k}, top_n={top_n}",
                title="Legal GraphRAG Search",
                border_style="blue",
            )
        )
        console.print(_candidate_table("Final Hybrid Results", response.final_candidates))
        console.print(Panel(response.answer, title="Final Answer", border_style="green"))

    except Exception as exc:  # noqa: BLE001
        logger.exception("Search failed: {}", exc)
        raise typer.Exit(code=1) from exc


def _candidate_table(title: str, candidates: list[Any]) -> Table:
    """Render retrieval candidates as a Rich table."""

    table = Table(title=title, show_lines=True)
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Document")
    table.add_column("Topics")
    table.add_column("Evidence")

    for rank, candidate in enumerate(candidates, start=1):
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
            f"{candidate.combined_score:.4f}",
            document,
            topics,
            candidate.evidence_preview(),
        )

    return table


def _upsert_topic_embedding(ingestor: Any, normalized_name: str, embedding: list[float]) -> None:
    """Update Topic.embedding while staying compatible with older GraphIngestor versions."""

    if hasattr(ingestor, "upsert_topic_embedding"):
        ingestor.upsert_topic_embedding(normalized_name=normalized_name, embedding=embedding)
        return

    def _tx(tx, topic_name: str, vector: list[float]) -> None:
        query = """
        MATCH (t:Topic {normalized_name: $normalized_name})
        SET t.embedding = $embedding,
            t.updated_at = datetime()
        """
        tx.run(query, normalized_name=topic_name, embedding=vector).consume()

    ingestor._execute_write(_tx, normalized_name, [float(value) for value in embedding])


def main() -> None:
    """Entrypoint wrapper used by `python -m src.main`."""

    app()


if __name__ == "__main__":
    main()