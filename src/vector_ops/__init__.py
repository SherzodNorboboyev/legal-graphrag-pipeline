"""Vector operation modules for Legal GraphRAG.

This package contains Markdown chunking, embedding providers, embedding cache,
and similarity helpers.
"""

from src.vector_ops.chunking import MarkdownChunker, count_tokens, detokenize_tokens
from src.vector_ops.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingCache,
    EmbeddingDimensionError,
    HashingEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    cosine_similarity,
    get_embedding_provider,
    normalize_vector,
    validate_vector_dimension,
)

__all__ = [
    "MarkdownChunker",
    "count_tokens",
    "detokenize_tokens",
    "BaseEmbeddingProvider",
    "EmbeddingCache",
    "EmbeddingDimensionError",
    "HashingEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "cosine_similarity",
    "get_embedding_provider",
    "normalize_vector",
    "validate_vector_dimension",
]