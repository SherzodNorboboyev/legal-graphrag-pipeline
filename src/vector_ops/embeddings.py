"""Embedding providers, cache, and vector similarity helpers.

Default provider:
- SentenceTransformers

Optional provider:
- OpenAI embeddings

Graceful fallback:
- Deterministic hashing embeddings if local model loading fails or OpenAI is
  not configured. This keeps tests and offline development functional.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from loguru import logger

from src.config import Settings, get_settings


class EmbeddingDimensionError(ValueError):
    """Raised when an embedding vector does not match expected dimensions."""


class EmbeddingCache:
    """SQLite-backed embedding cache.

    Cache key includes provider, model, dimensions, and text hash. This prevents
    mixing vectors from different providers or incompatible dimensions.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create cache table if it does not exist."""

        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    def make_key(self, provider: str, model: str, dimensions: int, text: str) -> tuple[str, str]:
        """Create cache key and text digest."""

        text_sha256 = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        key = f"{provider}:{model}:{dimensions}:{text_sha256}"
        return key, text_sha256

    def get(self, provider: str, model: str, dimensions: int, text: str) -> list[float] | None:
        """Return cached vector if present."""

        key, _ = self.make_key(provider, model, dimensions, text)

        with self._lock, sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT vector_json FROM embeddings WHERE cache_key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None

        return [float(value) for value in json.loads(row[0])]

    def set(self, provider: str, model: str, dimensions: int, text: str, vector: list[float]) -> None:
        """Store one vector in cache."""

        key, text_sha256 = self.make_key(provider, model, dimensions, text)

        with self._lock, sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO embeddings (
                    cache_key,
                    provider,
                    model,
                    dimensions,
                    text_sha256,
                    vector_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    provider,
                    model,
                    dimensions,
                    text_sha256,
                    json.dumps([float(value) for value in vector]),
                ),
            )
            connection.commit()


class BaseEmbeddingProvider(ABC):
    """Base class for cache-aware embedding providers."""

    provider_name: str = "base"

    def __init__(
        self,
        *,
        model_name: str,
        dimensions: int,
        cache: EmbeddingCache | None = None,
    ):
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")

        self.model_name = model_name
        self.dimensions = int(dimensions)
        self.cache = cache

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""

        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with batch support and caching."""

        normalized_texts = [text if text is not None else "" for text in texts]
        outputs: list[list[float] | None] = []
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, text in enumerate(normalized_texts):
            cached_vector = None

            if self.cache is not None:
                cached_vector = self.cache.get(
                    self.provider_name,
                    self.model_name,
                    self.dimensions,
                    text,
                )

            if cached_vector is not None:
                validate_vector_dimension(cached_vector, self.dimensions)
                outputs.append(cached_vector)
            else:
                outputs.append(None)
                missing_indices.append(index)
                missing_texts.append(text)

        if missing_texts:
            generated_vectors = self._embed_uncached(missing_texts)

            if len(generated_vectors) != len(missing_texts):
                raise RuntimeError(
                    f"Embedding provider returned {len(generated_vectors)} vectors "
                    f"for {len(missing_texts)} input texts."
                )

            for original_index, vector in zip(missing_indices, generated_vectors, strict=True):
                normalized_vector = normalize_vector(validate_vector_dimension(vector, self.dimensions))
                outputs[original_index] = normalized_vector

                if self.cache is not None:
                    self.cache.set(
                        self.provider_name,
                        self.model_name,
                        self.dimensions,
                        normalized_texts[original_index],
                        normalized_vector,
                    )

        return [vector if vector is not None else [] for vector in outputs]

    @abstractmethod
    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts without consulting cache."""


class HashingEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic local embedding fallback based on feature hashing.

    This is not semantically equivalent to transformer embeddings, but it is
    stable, fast, dependency-free, and useful for tests or degraded operation.
    """

    provider_name = "hashing_fallback"

    def __init__(
        self,
        *,
        model_name: str = "hashing-fallback-v1",
        dimensions: int = 384,
        cache: EmbeddingCache | None = None,
    ):
        super().__init__(model_name=model_name, dimensions=dimensions, cache=cache)

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic hashed vectors."""

        return [self._hash_text(text) for text in texts]

    def _hash_text(self, text: str) -> list[float]:
        """Hash lexical features into a fixed-size vector."""

        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w\u0600-\u06FF]+", (text or "").lower(), flags=re.UNICODE)

        if not tokens:
            tokens = [text or "empty"]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        return normalize_vector(vector)


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """SentenceTransformers embedding provider with graceful model-load fallback."""

    provider_name = "sentence_transformers"

    def __init__(
        self,
        *,
        model_name: str,
        dimensions: int = 384,
        cache: EmbeddingCache | None = None,
    ):
        super().__init__(model_name=model_name, dimensions=dimensions, cache=cache)
        self._model = None
        self._model_unavailable = False
        self._fallback = HashingEmbeddingProvider(dimensions=dimensions, cache=None)

    def _load_model(self):
        """Lazy-load SentenceTransformer model."""

        if self._model is not None:
            return self._model

        if self._model_unavailable:
            return None

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model: {}", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception as exc:  # noqa: BLE001 - fallback is required for degraded mode
            self._model_unavailable = True
            logger.warning(
                "Could not load SentenceTransformer model '{}'. "
                "Using deterministic hashing fallback. Error: {}",
                self.model_name,
                exc,
            )
            return None

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with SentenceTransformers or hashing fallback."""

        model = self._load_model()

        if model is None:
            return self._fallback._embed_uncached(texts)

        vectors = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        result = [[float(value) for value in vector] for vector in vectors]
        for vector in result:
            validate_vector_dimension(vector, self.dimensions)

        return result


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider with optional hashing fallback on API errors."""

    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        dimensions: int,
        cache: EmbeddingCache | None = None,
        fallback_on_error: bool = True,
    ):
        super().__init__(model_name=model_name, dimensions=dimensions, cache=cache)
        self.api_key = api_key
        self.fallback_on_error = fallback_on_error
        self._client = None
        self._fallback = HashingEmbeddingProvider(dimensions=dimensions, cache=None)

    def _load_client(self):
        """Lazy-load OpenAI client."""

        if self._client is not None:
            return self._client

        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI embeddings API."""

        try:
            client = self._load_client()
            response = client.embeddings.create(
                model=self.model_name,
                input=texts,
                dimensions=self.dimensions,
            )
            vectors = [[float(value) for value in item.embedding] for item in response.data]

            for vector in vectors:
                validate_vector_dimension(vector, self.dimensions)

            return vectors
        except Exception as exc:  # noqa: BLE001 - fallback keeps pipeline usable
            if not self.fallback_on_error:
                raise

            logger.warning(
                "OpenAI embedding request failed. Using deterministic hashing fallback. Error: {}",
                exc,
            )
            return self._fallback._embed_uncached(texts)


def get_embedding_provider(settings: Settings | None = None) -> BaseEmbeddingProvider:
    """Factory for the configured embedding provider."""

    settings = settings or get_settings()
    cache = EmbeddingCache(settings.embedding_cache_path)

    provider = getattr(settings, "embedding_provider", "sentence_transformers")

    if provider == "openai":
        api_key = getattr(settings, "openai_api_key", None)

        if not api_key:
            logger.warning("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is empty. Using hashing fallback.")
            return HashingEmbeddingProvider(
                dimensions=getattr(settings, "openai_embedding_dimensions", 1536),
                cache=cache,
            )

        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model_name=getattr(settings, "openai_embedding_model", "text-embedding-3-small"),
            dimensions=getattr(settings, "openai_embedding_dimensions", 1536),
            cache=cache,
        )

    return SentenceTransformerEmbeddingProvider(
        model_name=getattr(settings, "sentence_transformer_model", "sentence-transformers/all-MiniLM-L6-v2"),
        dimensions=getattr(settings, "neo4j_vector_dimensions", 384),
        cache=cache,
    )


def validate_vector_dimension(vector: Iterable[float], expected_dimensions: int) -> list[float]:
    """Validate vector dimension and finite numeric values."""

    values = [float(value) for value in vector]

    if len(values) != expected_dimensions:
        raise EmbeddingDimensionError(
            f"Embedding dimension mismatch: expected {expected_dimensions}, got {len(values)}."
        )

    for value in values:
        if not math.isfinite(value):
            raise ValueError("Embedding vector contains non-finite values.")

    return values


def normalize_vector(vector: Iterable[float]) -> list[float]:
    """Return an L2-normalized vector."""

    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))

    if norm == 0:
        return values

    return [value / norm for value in values]


def cosine_similarity(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    """Compute cosine similarity between two vectors."""

    a = [float(value) for value in vector_a]
    b = [float(value) for value in vector_b]

    if not a or not b or len(a) != len(b):
        return 0.0

    denominator = math.sqrt(sum(value * value for value in a)) * math.sqrt(sum(value * value for value in b))

    if denominator == 0:
        return 0.0

    return sum(x * y for x, y in zip(a, b, strict=True)) / denominator