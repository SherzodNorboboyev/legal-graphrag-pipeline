"""LLM agent modules for the Legal GraphRAG pipeline.

This package contains prompt templates and topic extraction logic. The topic
extractor supports optional OpenAI integration and a deterministic offline
fallback so the pipeline remains usable without external LLM credentials.
"""

from src.llm_agents.topic_extractor import TopicExtractionError, TopicExtractionResult, TopicExtractor

__all__ = [
    "TopicExtractionError",
    "TopicExtractionResult",
    "TopicExtractor",
]