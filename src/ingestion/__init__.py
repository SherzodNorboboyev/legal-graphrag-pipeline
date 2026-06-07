"""Neo4j ingestion package for the Legal GraphRAG pipeline.

This package contains graph schema initialization and ingestion helpers for
Document, Topic, Chunk, AMENDS, REPEALS, HAS_TOPIC, and HAS_CHUNK graph data.
"""

from src.ingestion.graph_ingest import GraphIngestionError, GraphIngestor
from src.ingestion.graph_schema import Neo4jGraphSchema, SchemaSetupResult

__all__ = [
    "GraphIngestionError",
    "GraphIngestor",
    "Neo4jGraphSchema",
    "SchemaSetupResult",
]