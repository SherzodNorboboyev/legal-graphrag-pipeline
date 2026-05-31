"""Application configuration and logging setup.

This module centralizes all runtime configuration for the Legal GraphRAG
pipeline. Values are loaded from environment variables and, during local
development, from a `.env` file.

No secrets should be hardcoded in source code. Use `.env` locally and a proper
secret manager in production.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables.

    The settings object is intentionally broad in Part 1 so that later pipeline
    modules can use the same stable configuration surface without repeatedly
    changing environment variable names.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_env: str = Field(default="local", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # -------------------------------------------------------------------------
    # Data directories
    # -------------------------------------------------------------------------
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    raw_dir: Path = Field(default=Path("data/raw"), alias="RAW_DIR")
    markdown_dir: Path = Field(default=Path("data/markdown"), alias="MARKDOWN_DIR")
    sample_output_dir: Path = Field(default=Path("data/sample_output"), alias="SAMPLE_OUTPUT_DIR")
    checkpoint_file: Path = Field(default=Path("data/crawl_checkpoint.json"), alias="CHECKPOINT_FILE")
    embedding_cache_path: Path = Field(default=Path("data/embedding_cache.sqlite3"), alias="EMBEDDING_CACHE_PATH")

    # -------------------------------------------------------------------------
    # Source websites
    # -------------------------------------------------------------------------
    qanoon_base_url: str = Field(default="https://qanoon.om/", alias="QANOON_BASE_URL")
    decree_base_url: str = Field(default="https://decree.om/", alias="DECREE_BASE_URL")
    scrape_english: bool = Field(default=True, alias="SCRAPE_ENGLISH")

    # -------------------------------------------------------------------------
    # Scraper behavior
    # -------------------------------------------------------------------------
    use_playwright: bool = Field(default=False, alias="USE_PLAYWRIGHT")
    max_pages: int = Field(default=1000, ge=1, alias="MAX_PAGES")
    request_timeout_seconds: float = Field(default=30.0, gt=0, alias="REQUEST_TIMEOUT_SECONDS")
    request_retries: int = Field(default=3, ge=0, alias="REQUEST_RETRIES")
    throttle_min_seconds: float = Field(default=1.0, ge=0, alias="THROTTLE_MIN_SECONDS")
    throttle_max_seconds: float = Field(default=3.5, ge=0, alias="THROTTLE_MAX_SECONDS")

    # -------------------------------------------------------------------------
    # Neo4j
    # -------------------------------------------------------------------------
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="please-change-me", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    neo4j_vector_dimensions: int = Field(default=384, gt=0, alias="NEO4J_VECTOR_DIMENSIONS")

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    embedding_provider: Literal["sentence_transformers", "openai"] = Field(
        default="sentence_transformers",
        alias="EMBEDDING_PROVIDER",
    )
    sentence_transformer_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="SENTENCE_TRANSFORMER_MODEL",
    )
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    openai_embedding_dimensions: int = Field(default=1536, gt=0, alias="OPENAI_EMBEDDING_DIMENSIONS")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # -------------------------------------------------------------------------
    # LLM topic extraction and answer synthesis
    # -------------------------------------------------------------------------
    topic_llm_provider: Literal["fallback", "openai"] = Field(default="fallback", alias="TOPIC_LLM_PROVIDER")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    topic_extraction_max_retries: int = Field(default=2, ge=0, alias="TOPIC_EXTRACTION_MAX_RETRIES")
    synthesis_provider: Literal["fallback", "openai"] = Field(default="fallback", alias="SYNTHESIS_PROVIDER")

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------
    chunk_min_tokens: int = Field(default=500, ge=50, alias="CHUNK_MIN_TOKENS")
    chunk_max_tokens: int = Field(default=900, ge=100, alias="CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = Field(default=120, ge=0, alias="CHUNK_OVERLAP_TOKENS")

    # -------------------------------------------------------------------------
    # Hybrid retrieval
    # -------------------------------------------------------------------------
    hybrid_vector_weight: float = Field(default=0.65, ge=0.0, le=1.0, alias="HYBRID_VECTOR_WEIGHT")
    hybrid_keyword_weight: float = Field(default=0.35, ge=0.0, le=1.0, alias="HYBRID_KEYWORD_WEIGHT")
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="CROSS_ENCODER_MODEL",
    )

    # -------------------------------------------------------------------------
    # Topic merging
    # -------------------------------------------------------------------------
    topic_merge_similarity: float = Field(default=0.88, ge=0.0, le=1.0, alias="TOPIC_MERGE_SIMILARITY")

    @field_validator("qanoon_base_url", "decree_base_url")
    @classmethod
    def ensure_trailing_slash(cls, value: str) -> str:
        """Normalize base URLs to always end with a slash.

        This prevents subtle URL joining bugs later in the scraper layer.
        """

        value = value.strip()
        return value if value.endswith("/") else f"{value}/"

    @field_validator("throttle_max_seconds")
    @classmethod
    def validate_throttle_range(cls, value: float, info) -> float:
        """Validate that max throttle is not lower than min throttle."""

        min_value = info.data.get("throttle_min_seconds", 0.0)
        if value < min_value:
            raise ValueError("THROTTLE_MAX_SECONDS must be greater than or equal to THROTTLE_MIN_SECONDS")
        return value

    @field_validator("chunk_max_tokens")
    @classmethod
    def validate_chunk_range(cls, value: int, info) -> int:
        """Validate chunk size boundaries early during application startup."""

        min_value = info.data.get("chunk_min_tokens", 500)
        if value < min_value:
            raise ValueError("CHUNK_MAX_TOKENS must be greater than or equal to CHUNK_MIN_TOKENS")
        return value

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info) -> int:
        """Validate that overlap is smaller than the max chunk size."""

        max_value = info.data.get("chunk_max_tokens", 900)
        if value >= max_value:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_MAX_TOKENS")
        return value

    @property
    def active_embedding_dimensions(self) -> int:
        """Return the expected vector dimension for the active embedding provider."""

        if self.embedding_provider == "openai":
            return self.openai_embedding_dimensions
        return self.neo4j_vector_dimensions

    @property
    def is_openai_available(self) -> bool:
        """Return whether an OpenAI API key is configured."""

        return bool(self.openai_api_key and self.openai_api_key.strip())

    def ensure_directories(self) -> None:
        """Create local runtime directories required by the pipeline.

        Directory creation is safe and idempotent. This method is called when
        settings are loaded so CLI commands can assume the data layout exists.
        """

        directories = [
            self.data_dir,
            self.raw_dir,
            self.markdown_dir,
            self.sample_output_dir,
            self.checkpoint_file.parent,
            self.embedding_cache_path.parent,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def configure_logging(level: str | None = None) -> None:
    """Configure Loguru logging for CLI commands.

    Parameters
    ----------
    level:
        Optional explicit logging level. If not provided, `LOG_LEVEL` from the
        environment is used, falling back to `INFO`.
    """

    log_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        backtrace=False,
        diagnose=False,
        enqueue=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load, validate, cache, and return application settings."""

    settings = Settings()
    settings.ensure_directories()
    return settings


def log_startup_summary(settings: Settings) -> None:
    """Log a concise startup summary without exposing secrets."""

    logger.info("Application environment: {}", settings.app_env)
    logger.info("Data directory: {}", settings.data_dir)
    logger.info("Raw directory: {}", settings.raw_dir)
    logger.info("Markdown directory: {}", settings.markdown_dir)
    logger.info("Qanoon base URL: {}", settings.qanoon_base_url)
    logger.info("Decree base URL: {}", settings.decree_base_url)
    logger.info("Neo4j URI: {}", settings.neo4j_uri)
    logger.info("Neo4j database: {}", settings.neo4j_database)
    logger.info("Embedding provider: {}", settings.embedding_provider)
    logger.info("Embedding dimensions: {}", settings.active_embedding_dimensions)
    logger.info("OpenAI configured: {}", settings.is_openai_available)