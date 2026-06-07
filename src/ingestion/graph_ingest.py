"""Neo4j ingestion layer for Legal GraphRAG.

CLI integration notes:
- `python -m src.main ingest` should call `GraphIngestor.ensure_schema()`
  and then `GraphIngestor.ingest_markdown_dir()`.
- Later commands can use `link_document_topics()` after topic extraction and
  `batch_upsert_chunks()` after semantic chunking and embedding generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from loguru import logger
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired, TransientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from src.config import Settings, get_settings
from src.ingestion.graph_schema import Neo4jGraphSchema, SchemaSetupResult
from src.models import Chunk, CrossReference, LegalDocument, ParsedDocument, Topic, utc_now_iso


class GraphIngestionError(RuntimeError):
    """Raised when graph ingestion cannot continue safely."""


class GraphIngestor:
    """Connection manager and repository for Neo4j graph ingestion.

    The class owns the Neo4j driver by default and supports context-manager use:

    ```python
    from src.config import get_settings
    from src.ingestion.graph_ingest import GraphIngestor

    settings = get_settings()

    with GraphIngestor(settings) as ingestor:
        ingestor.verify_connection()
        ingestor.ensure_schema()
        ingestor.ingest_markdown_dir()
    ```
    """

    def __init__(
        self,
        settings: Settings | None = None,
        driver: Driver | None = None,
    ):
        self.settings = settings or get_settings()
        self.database = self.settings.neo4j_database

        if driver is None:
            self.driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
                max_connection_pool_size=25,
                connection_timeout=30,
            )
            self._owns_driver = True
        else:
            self.driver = driver
            self._owns_driver = False

    def __enter__(self) -> "GraphIngestor":
        """Return context-managed ingestor."""

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close owned driver on context exit."""

        self.close()

    def close(self) -> None:
        """Close Neo4j driver if this instance owns it."""

        if self._owns_driver:
            self.driver.close()

    def verify_connection(self) -> None:
        """Validate Neo4j connectivity and raise a clear error if unavailable."""

        try:
            self.driver.verify_connectivity()
            logger.info("Neo4j connection verified: {}", self.settings.neo4j_uri)
        except Exception as exc:  # noqa: BLE001 - convert to project-specific message
            raise GraphIngestionError(
                f"Could not connect to Neo4j at {self.settings.neo4j_uri}. "
                "Check docker compose, NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD."
            ) from exc

    def ensure_schema(self) -> SchemaSetupResult:
        """Create Neo4j constraints and indexes."""

        vector_dimensions = getattr(
            self.settings,
            "active_embedding_dimensions",
            getattr(self.settings, "neo4j_vector_dimensions", 384),
        )

        schema = Neo4jGraphSchema(
            driver=self.driver,
            database=self.database,
            vector_dimensions=vector_dimensions,
        )
        return schema.setup(wait_for_indexes=True)

    def ingest_markdown_dir(
        self,
        directory: Path | None = None,
        limit: int | None = None,
        batch_size: int = 100,
    ) -> int:
        """Ingest document JSON files from the markdown data directory.

        Parameters
        ----------
        directory:
            Directory containing parsed document JSON files. Defaults to
            `settings.markdown_dir`.
        limit:
            Optional maximum number of files to ingest.
        batch_size:
            Number of documents per write transaction.
        """

        directory = directory or self.settings.markdown_dir

        if not directory.exists():
            raise GraphIngestionError(f"Markdown directory does not exist: {directory}")

        paths = sorted(directory.glob("*.json"))

        if limit is not None:
            paths = paths[:limit]

        documents: list[LegalDocument] = []

        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                document = self.document_from_payload(payload)
                documents.append(document)
            except Exception as exc:  # noqa: BLE001 - continue with other files
                logger.exception("Failed to load parsed document JSON {}: {}", path, exc)

        if not documents:
            logger.warning("No documents found to ingest from {}", directory)
            return 0

        return self.ingest_documents(documents, batch_size=batch_size)

    def document_from_payload(self, payload: dict[str, Any]) -> LegalDocument:
        """Convert scraper JSON payload into a canonical LegalDocument model."""

        try:
            return LegalDocument.model_validate(payload)
        except Exception as legal_error:  # noqa: BLE001 - try parsed compatibility model
            try:
                return ParsedDocument.model_validate(payload).to_legal_document()
            except Exception as parsed_error:  # noqa: BLE001
                raise GraphIngestionError(
                    "Payload cannot be validated as LegalDocument or ParsedDocument."
                ) from parsed_error

    def ingest_documents(
        self,
        documents: Iterable[LegalDocument | ParsedDocument | dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Batch-ingest documents and then create cross-reference relationships.

        Documents are upserted first so that AMENDS/REPEALS edges can resolve
        targets inside the same batch or corpus.
        """

        normalized_documents = [self.normalize_document_input(document) for document in documents]

        if not normalized_documents:
            return 0

        for batch in batched(normalized_documents, batch_size):
            self.batch_upsert_documents(batch)

        linked_count = 0
        for document in normalized_documents:
            linked_count += self.link_cross_references_for_document(document)

        logger.info(
            "Ingested {} documents into Neo4j. legal_cross_reference_links_created_or_matched={}",
            len(normalized_documents),
            linked_count,
        )
        return len(normalized_documents)

    def normalize_document_input(self, document: LegalDocument | ParsedDocument | dict[str, Any]) -> LegalDocument:
        """Normalize supported document inputs to LegalDocument."""

        if isinstance(document, LegalDocument):
            return document

        if isinstance(document, ParsedDocument):
            return document.to_legal_document()

        if isinstance(document, dict):
            return self.document_from_payload(document)

        raise TypeError(f"Unsupported document input type: {type(document)!r}")

    def upsert_document(self, document: LegalDocument | ParsedDocument | dict[str, Any]) -> None:
        """Upsert one Document node."""

        normalized = self.normalize_document_input(document)
        self.batch_upsert_documents([normalized])

    def batch_upsert_documents(self, documents: Sequence[LegalDocument]) -> int:
        """Upsert a batch of Document nodes."""

        if not documents:
            return 0

        rows = [self.document_to_row(document) for document in documents]
        self._execute_write(self._tx_batch_upsert_documents, rows)
        logger.debug("Upserted {} Document nodes.", len(rows))
        return len(rows)

    @staticmethod
    def _tx_batch_upsert_documents(tx, rows: list[dict[str, Any]]) -> None:
        """Neo4j transaction function for batch document upsert."""

        query = """
        UNWIND $documents AS row
        MERGE (d:Document {id: row.id})
        SET d.title = coalesce(row.title, d.title),
            d.title_ar = coalesce(row.title_ar, d.title_ar),
            d.title_en = coalesce(row.title_en, d.title_en),
            d.title_fr = coalesce(row.title_fr, d.title_fr),
            d.date = coalesce(row.date, d.date),
            d.document_type = coalesce(row.document_type, d.document_type),
            d.number = coalesce(row.number, d.number),
            d.issuer = coalesce(row.issuer, d.issuer),
            d.source_url = coalesce(row.source_url, d.source_url),
            d.language = coalesce(row.language, d.language),
            d.contentAr = coalesce(row.contentAr, d.contentAr),
            d.contentEn = coalesce(row.contentEn, d.contentEn),
            d.contentFr = coalesce(row.contentFr, d.contentFr),
            d.language_urls = coalesce(row.language_urls, d.language_urls),
            d.pdf_urls = coalesce(row.pdf_urls, d.pdf_urls),
            d.raw_paths = coalesce(row.raw_paths, d.raw_paths),
            d.cross_references = coalesce(row.cross_references, d.cross_references),
            d.metadata = coalesce(row.metadata, d.metadata),
            d.created_at = coalesce(d.created_at, row.created_at),
            d.updated_at = row.updated_at
        """
        tx.run(query, documents=rows).consume()

    def document_to_row(self, document: LegalDocument) -> dict[str, Any]:
        """Convert LegalDocument into Neo4j-safe row properties."""

        return {
            "id": document.id,
            "title": document.title,
            "title_ar": document.title_ar,
            "title_en": document.title_en,
            "title_fr": document.title_fr,
            "date": document.date,
            "document_type": document.document_type,
            "number": document.number,
            "issuer": document.issuer,
            "source_url": document.source_url,
            "language": document.language,
            "contentAr": document.content_ar,
            "contentEn": document.content_en,
            "contentFr": document.content_fr,
            # Neo4j supports primitive arrays, but dictionaries/lists of maps are
            # safer as JSON strings for portability across versions.
            "language_urls": json.dumps(document.language_urls, ensure_ascii=False, sort_keys=True),
            "pdf_urls": document.pdf_urls,
            "raw_paths": json.dumps(document.raw_paths, ensure_ascii=False, sort_keys=True),
            "cross_references": json.dumps(
                [reference.model_dump(mode="json") for reference in document.cross_references],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "metadata": json.dumps(document.metadata, ensure_ascii=False, sort_keys=True),
            "created_at": document.created_at,
            "updated_at": utc_now_iso(),
        }

    def upsert_topic(self, topic: Topic | dict[str, Any]) -> None:
        """Upsert one Topic node."""

        normalized_topic = topic if isinstance(topic, Topic) else Topic.model_validate(topic)
        self.batch_upsert_topics([normalized_topic])

    def batch_upsert_topics(self, topics: Sequence[Topic]) -> int:
        """Upsert Topic nodes without linking them to documents."""

        if not topics:
            return 0

        rows = [self.topic_to_row(topic) for topic in topics]
        self._execute_write(self._tx_batch_upsert_topics, rows)
        logger.debug("Upserted {} Topic nodes.", len(rows))
        return len(rows)

    @staticmethod
    def _tx_batch_upsert_topics(tx, rows: list[dict[str, Any]]) -> None:
        """Neo4j transaction function for Topic upsert."""

        query = """
        UNWIND $topics AS row
        MERGE (t:Topic {normalized_name: row.normalized_name})
        SET t.name = coalesce(row.name, t.name),
            t.confidence = coalesce(row.confidence, t.confidence),
            t.evidence = coalesce(row.evidence, t.evidence),
            t.source = coalesce(row.source, t.source),
            t.metadata = coalesce(row.metadata, t.metadata),
            t.embedding = coalesce(row.embedding, t.embedding),
            t.created_at = coalesce(t.created_at, row.created_at),
            t.updated_at = row.updated_at
        """
        tx.run(query, topics=rows).consume()

    def link_document_topics(
        self,
        document_id: str,
        topics: Sequence[Topic | dict[str, Any]],
    ) -> int:
        """Upsert topics and link them to one Document via HAS_TOPIC."""

        if not topics:
            return 0

        normalized_topics = [topic if isinstance(topic, Topic) else Topic.model_validate(topic) for topic in topics]
        rows = [self.topic_to_row(topic) for topic in normalized_topics]

        self._execute_write(self._tx_link_document_topics, document_id, rows)
        logger.debug("Linked {} Topic nodes to Document {}", len(rows), document_id)
        return len(rows)

    @staticmethod
    def _tx_link_document_topics(tx, document_id: str, topic_rows: list[dict[str, Any]]) -> None:
        """Neo4j transaction function for HAS_TOPIC relationships."""

        query = """
        MATCH (d:Document {id: $document_id})
        UNWIND $topics AS row
        MERGE (t:Topic {normalized_name: row.normalized_name})
        SET t.name = coalesce(row.name, t.name),
            t.confidence = coalesce(row.confidence, t.confidence),
            t.evidence = coalesce(row.evidence, t.evidence),
            t.source = coalesce(row.source, t.source),
            t.metadata = coalesce(row.metadata, t.metadata),
            t.embedding = coalesce(row.embedding, t.embedding),
            t.created_at = coalesce(t.created_at, row.created_at),
            t.updated_at = row.updated_at
        MERGE (d)-[r:HAS_TOPIC]->(t)
        SET r.confidence = row.confidence,
            r.evidence = row.evidence,
            r.source = row.source,
            r.updated_at = row.updated_at
        """
        tx.run(query, document_id=document_id, topics=topic_rows).consume()

    def topic_to_row(self, topic: Topic) -> dict[str, Any]:
        """Convert Topic model into Neo4j-safe row."""

        return {
            "name": topic.name,
            "normalized_name": topic.normalized_name,
            "embedding": topic.embedding,
            "confidence": topic.confidence,
            "evidence": topic.evidence,
            "source": topic.source,
            "metadata": json.dumps(topic.metadata, ensure_ascii=False, sort_keys=True),
            "created_at": topic.created_at,
            "updated_at": utc_now_iso(),
        }

    def upsert_chunk(self, chunk: Chunk | dict[str, Any]) -> None:
        """Upsert one Chunk node and its HAS_CHUNK relationship."""

        normalized_chunk = chunk if isinstance(chunk, Chunk) else Chunk.model_validate(chunk)
        self.batch_upsert_chunks([normalized_chunk])

    def batch_upsert_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Upsert Chunk nodes and link them to parent Documents."""

        if not chunks:
            return 0

        rows = [self.chunk_to_row(chunk) for chunk in chunks]
        self._execute_write(self._tx_batch_upsert_chunks, rows)
        logger.debug("Upserted {} Chunk nodes and HAS_CHUNK relationships.", len(rows))
        return len(rows)

    @staticmethod
    def _tx_batch_upsert_chunks(tx, rows: list[dict[str, Any]]) -> None:
        """Neo4j transaction function for Chunk upsert."""

        query = """
        UNWIND $chunks AS row
        MATCH (d:Document {id: row.document_id})
        MERGE (c:Chunk {id: row.id})
        SET c.document_id = row.document_id,
            c.language = row.language,
            c.text = row.text,
            c.chunk_index = row.chunk_index,
            c.heading_context = row.heading_context,
            c.token_count = row.token_count,
            c.metadata = row.metadata,
            c.embedding = coalesce(row.embedding, c.embedding),
            c.created_at = coalesce(c.created_at, row.created_at),
            c.updated_at = row.updated_at
        MERGE (d)-[r:HAS_CHUNK]->(c)
        SET r.language = row.language,
            r.chunk_index = row.chunk_index,
            r.updated_at = row.updated_at
        """
        tx.run(query, chunks=rows).consume()

    def chunk_to_row(self, chunk: Chunk) -> dict[str, Any]:
        """Convert Chunk model into Neo4j-safe row."""

        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "language": chunk.language,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "embedding": chunk.embedding,
            "heading_context": chunk.heading_context,
            "token_count": chunk.token_count,
            "metadata": json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
            "created_at": chunk.created_at,
            "updated_at": utc_now_iso(),
        }

    def link_cross_references_for_document(self, document: LegalDocument) -> int:
        """Create AMENDS/REPEALS relationships for a document's cross references."""

        linked_count = 0

        for reference in document.cross_references:
            if not reference.is_document_level_relationship:
                continue

            try:
                linked_count += int(self.link_cross_reference(document.id or "", reference))
            except Exception as exc:  # noqa: BLE001 - continue with other references
                logger.warning(
                    "Failed to link cross-reference for document {} target_number={} relation={}: {}",
                    document.id,
                    reference.target_number,
                    reference.relation_type,
                    exc,
                )

        return linked_count

    def link_cross_reference(self, source_document_id: str, reference: CrossReference | dict[str, Any]) -> int:
        """Create one AMENDS or REPEALS edge if the target Document exists."""

        if not source_document_id:
            raise ValueError("source_document_id is required.")

        normalized_reference = reference if isinstance(reference, CrossReference) else CrossReference.model_validate(reference)

        if normalized_reference.relation_type not in {"AMENDS", "REPEALS"}:
            logger.debug("Skipping non-document-level reference type: {}", normalized_reference.relation_type)
            return 0

        relationship_type = normalized_reference.relation_type

        if normalized_reference.target_document_id:
            return self._execute_write(
                self._tx_link_cross_reference_by_id,
                source_document_id,
                normalized_reference.target_document_id,
                relationship_type,
                normalized_reference.raw_text,
                normalized_reference.context,
            )

        if normalized_reference.target_number:
            return self._execute_write(
                self._tx_link_cross_reference_by_number,
                source_document_id,
                normalized_reference.target_number,
                normalized_reference.target_document_type,
                relationship_type,
                normalized_reference.raw_text,
                normalized_reference.context,
            )

        logger.debug("Skipping cross-reference with no target id/number: {}", normalized_reference)
        return 0

    @staticmethod
    def _tx_link_cross_reference_by_id(
        tx,
        source_document_id: str,
        target_document_id: str,
        relationship_type: str,
        raw_text: str,
        context: str | None,
    ) -> int:
        """Neo4j transaction function to link AMENDS/REPEALS by target ID."""

        query = f"""
        MATCH (source:Document {{id: $source_document_id}})
        MATCH (target:Document {{id: $target_document_id}})
        WHERE source.id <> target.id
        MERGE (source)-[r:{relationship_type}]->(target)
        SET r.raw_text = $raw_text,
            r.context = $context,
            r.updated_at = datetime()
        RETURN count(r) AS linked_count
        """
        record = tx.run(
            query,
            source_document_id=source_document_id,
            target_document_id=target_document_id,
            raw_text=raw_text,
            context=context,
        ).single()

        return int(record["linked_count"] if record else 0)

    @staticmethod
    def _tx_link_cross_reference_by_number(
        tx,
        source_document_id: str,
        target_number: str,
        target_document_type: str | None,
        relationship_type: str,
        raw_text: str,
        context: str | None,
    ) -> int:
        """Neo4j transaction function to link AMENDS/REPEALS by target number."""

        query = f"""
        MATCH (source:Document {{id: $source_document_id}})
        MATCH (target:Document)
        WHERE target.number = $target_number
          AND source.id <> target.id
          AND ($target_document_type IS NULL OR target.document_type = $target_document_type)
        MERGE (source)-[r:{relationship_type}]->(target)
        SET r.raw_text = $raw_text,
            r.context = $context,
            r.updated_at = datetime()
        RETURN count(r) AS linked_count
        """
        record = tx.run(
            query,
            source_document_id=source_document_id,
            target_number=target_number,
            target_document_type=target_document_type,
            raw_text=raw_text,
            context=context,
        ).single()

        return int(record["linked_count"] if record else 0)

    def fetch_documents(self, limit: int | None = None, only_without_topics: bool = False) -> list[dict[str, Any]]:
        """Fetch Document nodes for later topic extraction/chunking commands."""

        where_clause = "WHERE NOT (d)-[:HAS_TOPIC]->(:Topic)" if only_without_topics else ""
        limit_clause = "LIMIT $limit" if limit is not None else ""

        query = f"""
        MATCH (d:Document)
        {where_clause}
        RETURN d.id AS id,
               d.title AS title,
               d.date AS date,
               d.document_type AS document_type,
               d.number AS number,
               d.issuer AS issuer,
               d.source_url AS source_url,
               d.contentAr AS contentAr,
               d.contentEn AS contentEn,
               d.contentFr AS contentFr
        ORDER BY d.date DESC, d.id
        {limit_clause}
        """

        with self.driver.session(database=self.database) as session:
            records = session.run(query, limit=limit)
            return [dict(record) for record in records]

    def fetch_topics_without_embeddings(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Fetch Topic nodes that do not yet have embeddings."""

        limit_clause = "LIMIT $limit" if limit is not None else ""

        query = f"""
        MATCH (t:Topic)
        WHERE t.embedding IS NULL
        RETURN t.name AS name,
               t.normalized_name AS normalized_name,
               t.confidence AS confidence,
               t.evidence AS evidence,
               t.source AS source
        ORDER BY t.normalized_name
        {limit_clause}
        """

        with self.driver.session(database=self.database) as session:
            records = session.run(query, limit=limit)
            return [dict(record) for record in records]
        
    def upsert_topic_embedding(self, normalized_name: str, embedding: list[float]) -> None:
        """Set or update embedding vector for one Topic node."""

        if not normalized_name:
            raise ValueError("normalized_name is required.")

        if not embedding:
            raise ValueError("embedding must not be empty.")

        self._execute_write(
            self._tx_upsert_topic_embedding,
            normalized_name,
            [float(value) for value in embedding],
        )

    @staticmethod
    def _tx_upsert_topic_embedding(tx, normalized_name: str, embedding: list[float]) -> None:
        """Neo4j transaction function for updating Topic.embedding."""

        query = """
        MATCH (t:Topic {normalized_name: $normalized_name})
        SET t.embedding = $embedding,
            t.updated_at = datetime()
        """

        tx.run(
            query,
            normalized_name=normalized_name,
            embedding=embedding,
        ).consume()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=8.0),
        retry=retry_if_exception_type((ServiceUnavailable, SessionExpired, TransientError, TimeoutError)),
    )
    def _execute_write(self, transaction_function, *args, **kwargs):
        """Execute a write transaction with retry for transient failures."""

        try:
            with self.driver.session(database=self.database) as session:
                return session.execute_write(transaction_function, *args, **kwargs)
        except (ServiceUnavailable, SessionExpired, TransientError, TimeoutError):
            logger.warning("Transient Neo4j write failure. Retrying transaction.")
            raise
        except Neo4jError as exc:
            raise GraphIngestionError(f"Neo4j write transaction failed: {exc}") from exc

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=8.0),
        retry=retry_if_exception_type((ServiceUnavailable, SessionExpired, TransientError, TimeoutError)),
    )
    def _execute_read(self, transaction_function, *args, **kwargs):
        """Execute a read transaction with retry for transient failures."""

        try:
            with self.driver.session(database=self.database) as session:
                return session.execute_read(transaction_function, *args, **kwargs)
        except (ServiceUnavailable, SessionExpired, TransientError, TimeoutError):
            logger.warning("Transient Neo4j read failure. Retrying transaction.")
            raise
        except Neo4jError as exc:
            raise GraphIngestionError(f"Neo4j read transaction failed: {exc}") from exc


def batched(items: Sequence[Any], batch_size: int) -> Iterable[list[Any]]:
    """Yield a sequence in fixed-size batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])