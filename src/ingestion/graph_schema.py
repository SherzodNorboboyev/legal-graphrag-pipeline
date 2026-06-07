"""Neo4j schema setup for the Legal GraphRAG pipeline.

The schema follows the assignment's simplified GraphRAG principle:
- Document stores language-specific Markdown properties directly.
- Topic and Chunk are separate vector-searchable nodes.
- Document-level legal relations are represented through AMENDS and REPEALS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger
from neo4j import Driver
from neo4j.exceptions import Neo4jError


@dataclass(frozen=True)
class SchemaStatement:
    """One schema statement and its failure policy."""

    name: str
    cypher: str
    critical: bool = True


@dataclass
class SchemaSetupResult:
    """Summary returned after schema setup."""

    applied: list[str] = field(default_factory=list)
    skipped_optional: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True when no critical schema statement failed."""

        return not self.failed


class Neo4jGraphSchema:
    """Create idempotent constraints, regular indexes, full-text indexes, and vector indexes."""

    def __init__(
        self,
        driver: Driver,
        database: str = "neo4j",
        vector_dimensions: int = 384,
    ):
        self.driver = driver
        self.database = database
        self.vector_dimensions = int(vector_dimensions)

    def setup(self, wait_for_indexes: bool = True) -> SchemaSetupResult:
        """Apply all schema objects safely.

        Vector indexes are marked optional because Neo4j version and edition
        differences can affect availability. If vector index creation fails, the
        rest of the graph ingestion layer remains usable, and retrieval can later
        fall back to Python-side cosine similarity.
        """

        result = SchemaSetupResult()

        statements = [
            *self.constraint_statements(),
            *self.lookup_index_statements(),
            *self.fulltext_index_statements(),
            *self.vector_index_statements(),
        ]

        with self.driver.session(database=self.database) as session:
            for statement in statements:
                try:
                    logger.debug("Applying Neo4j schema statement '{}'.", statement.name)
                    session.run(statement.cypher).consume()
                    result.applied.append(statement.name)
                except Neo4jError as exc:
                    if statement.critical:
                        result.failed.append(statement.name)
                        logger.exception(
                            "Critical Neo4j schema statement '{}' failed: {}",
                            statement.name,
                            exc,
                        )
                        raise
                    result.skipped_optional.append(statement.name)
                    logger.warning(
                        "Optional Neo4j schema statement '{}' was skipped: {}",
                        statement.name,
                        exc,
                    )

            if wait_for_indexes:
                self.await_indexes(session)

        logger.info(
            "Neo4j schema setup completed. applied={}, skipped_optional={}, failed={}",
            len(result.applied),
            len(result.skipped_optional),
            len(result.failed),
        )
        return result

    def constraint_statements(self) -> list[SchemaStatement]:
        """Return idempotent unique constraints."""

        return [
            SchemaStatement(
                name="document_id_unique",
                cypher="CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            ),
            SchemaStatement(
                name="topic_normalized_name_unique",
                cypher=(
                    "CREATE CONSTRAINT topic_normalized_name_unique IF NOT EXISTS "
                    "FOR (t:Topic) REQUIRE t.normalized_name IS UNIQUE"
                ),
            ),
            SchemaStatement(
                name="chunk_id_unique",
                cypher="CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            ),
        ]

    def lookup_index_statements(self) -> list[SchemaStatement]:
        """Return idempotent lookup indexes used by ingestion and retrieval."""

        return [
            SchemaStatement(
                name="document_number_index",
                cypher="CREATE INDEX document_number_index IF NOT EXISTS FOR (d:Document) ON (d.number)",
            ),
            SchemaStatement(
                name="document_type_index",
                cypher="CREATE INDEX document_type_index IF NOT EXISTS FOR (d:Document) ON (d.document_type)",
            ),
            SchemaStatement(
                name="document_date_index",
                cypher="CREATE INDEX document_date_index IF NOT EXISTS FOR (d:Document) ON (d.date)",
            ),
            SchemaStatement(
                name="document_source_url_index",
                cypher="CREATE INDEX document_source_url_index IF NOT EXISTS FOR (d:Document) ON (d.source_url)",
            ),
            SchemaStatement(
                name="chunk_document_id_index",
                cypher="CREATE INDEX chunk_document_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
            ),
            SchemaStatement(
                name="chunk_language_index",
                cypher="CREATE INDEX chunk_language_index IF NOT EXISTS FOR (c:Chunk) ON (c.language)",
            ),
            SchemaStatement(
                name="topic_name_index",
                cypher="CREATE INDEX topic_name_index IF NOT EXISTS FOR (t:Topic) ON (t.name)",
            ),
        ]

    def fulltext_index_statements(self) -> list[SchemaStatement]:
        """Return full-text indexes for BM25-style retrieval fallback."""

        return [
            SchemaStatement(
                name="document_title_content_fulltext",
                cypher=(
                    "CREATE FULLTEXT INDEX document_title_content_fulltext IF NOT EXISTS "
                    "FOR (d:Document) ON EACH [d.title, d.title_ar, d.title_en, d.title_fr, "
                    "d.contentAr, d.contentEn, d.contentFr]"
                ),
            ),
            SchemaStatement(
                name="chunk_text_fulltext",
                cypher="CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
            ),
            SchemaStatement(
                name="topic_name_fulltext",
                cypher="CREATE FULLTEXT INDEX topic_name_fulltext IF NOT EXISTS FOR (t:Topic) ON EACH [t.name, t.normalized_name]",
            ),
        ]

    def vector_index_statements(self) -> list[SchemaStatement]:
        """Return Neo4j 5 vector index statements.

        Syntax target: Neo4j 5.x vector indexes.

        These are optional to keep the ingestion path production-safe on Neo4j
        installations where vector indexes are unavailable or disabled.
        """

        dimensions = self.vector_dimensions

        return [
            SchemaStatement(
                name="chunk_embedding_vector_index",
                critical=False,
                cypher=f"""
                CREATE VECTOR INDEX chunk_embedding_vector_index IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """,
            ),
            SchemaStatement(
                name="topic_embedding_vector_index",
                critical=False,
                cypher=f"""
                CREATE VECTOR INDEX topic_embedding_vector_index IF NOT EXISTS
                FOR (t:Topic) ON (t.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """,
            ),
        ]

    def await_indexes(self, session) -> None:
        """Ask Neo4j to await index population.

        This call is best-effort because procedure signatures vary slightly
        across Neo4j versions. Failure here should not invalidate schema setup.
        """

        try:
            session.run("CALL db.awaitIndexes(300)").consume()
            logger.debug("Neo4j indexes are online or await timeout completed.")
        except Neo4jError as exc:
            logger.warning("Could not await Neo4j indexes. Continuing. Error: {}", exc)