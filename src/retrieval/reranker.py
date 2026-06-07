"""Cross-encoder reranking utilities.

The reranker is optional. Hybrid retrieval should remain functional even when
the model cannot be loaded, the dependency is not installed, or the machine is
offline. In those cases, candidates are sorted by their existing hybrid score.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence, TypeVar

from loguru import logger


class Rerankable(Protocol):
    """Protocol expected by `CrossEncoderReranker`.

    Candidate objects may be dataclasses, Pydantic models, or simple objects.
    The reranker is defensive and also supports dictionaries.
    """

    combined_score: float
    rerank_score: float | None


T = TypeVar("T")


class CrossEncoderReranker:
    """Optional CrossEncoder reranker for query-candidate alignment.

    Parameters
    ----------
    model_name:
        SentenceTransformers CrossEncoder model name. `BAAI/bge-reranker-base`
        is a strong default. Smaller MS MARCO MiniLM models can be used for
        lower latency.
    enabled:
        If False, reranking is skipped and candidates are sorted by existing
        hybrid score.
    batch_size:
        Number of query-candidate pairs evaluated per model batch.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        enabled: bool = True,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.batch_size = batch_size
        self._model: Any | None = None
        self._model_unavailable = False

    def rerank(
        self,
        query: str,
        candidates: Sequence[T],
        *,
        top_n: int | None = None,
        text_getter: Callable[[T], str] | None = None,
    ) -> list[T]:
        """Rerank candidates and return the best items.

        If the model cannot be loaded or prediction fails, this method returns a
        fallback sort by existing score. This is important for production CLI
        reliability because retrieval should not fail due to an optional model.
        """

        if not candidates:
            return []

        if not self.enabled:
            return self.fallback_sort(candidates, top_n=top_n)

        model = self._load_model()
        if model is None:
            return self.fallback_sort(candidates, top_n=top_n)

        getter = text_getter or self.default_text_getter
        pairs = [(query, getter(candidate)) for candidate in candidates]

        try:
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            scores = [float(score) for score in raw_scores]
            normalized_scores = self.minmax_normalize(scores)

            for candidate, raw_score, normalized_score in zip(candidates, scores, normalized_scores, strict=True):
                base_score = self.get_candidate_score(candidate)
                blended_score = 0.7 * normalized_score + 0.3 * base_score
                self.set_candidate_rerank_score(candidate, raw_score)
                self.set_candidate_score(candidate, blended_score)

            sorted_candidates = sorted(candidates, key=self.get_candidate_score, reverse=True)
            return list(sorted_candidates[:top_n]) if top_n is not None else list(sorted_candidates)

        except Exception as exc:  # noqa: BLE001 - optional reranker must not break retrieval
            logger.warning("CrossEncoder reranking failed. Falling back to hybrid scores. Error: {}", exc)
            return self.fallback_sort(candidates, top_n=top_n)

    def fallback_sort(self, candidates: Sequence[T], *, top_n: int | None = None) -> list[T]:
        """Sort candidates by their existing score without model inference."""

        sorted_candidates = sorted(candidates, key=self.get_candidate_score, reverse=True)
        return list(sorted_candidates[:top_n]) if top_n is not None else list(sorted_candidates)

    def _load_model(self) -> Any | None:
        """Lazy-load the SentenceTransformers CrossEncoder model."""

        if self._model is not None:
            return self._model

        if self._model_unavailable:
            return None

        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder reranker model: {}", self.model_name)
            self._model = CrossEncoder(self.model_name)
            return self._model

        except Exception as exc:  # noqa: BLE001 - model loading failure is recoverable
            self._model_unavailable = True
            logger.warning(
                "Could not load CrossEncoder model '{}'. Existing hybrid scores will be used. Error: {}",
                self.model_name,
                exc,
            )
            return None

    def default_text_getter(self, candidate: T) -> str:
        """Extract reranking text from a candidate object or dictionary."""

        if isinstance(candidate, dict):
            return str(
                candidate.get("expanded_text")
                or candidate.get("expanded_context")
                or candidate.get("text")
                or ""
            )

        return str(
            getattr(candidate, "expanded_text", None)
            or getattr(candidate, "expanded_context", None)
            or getattr(candidate, "text", "")
            or ""
        )

    def get_candidate_score(self, candidate: T) -> float:
        """Read candidate score from common score attributes."""

        if isinstance(candidate, dict):
            for key in ("combined_score", "score", "vector_score", "keyword_score"):
                if key in candidate and candidate[key] is not None:
                    return float(candidate[key])
            return 0.0

        for attr in ("combined_score", "score", "vector_score", "keyword_score"):
            if hasattr(candidate, attr):
                value = getattr(candidate, attr)
                if value is not None:
                    return float(value)

        return 0.0

    def set_candidate_score(self, candidate: T, score: float) -> None:
        """Set candidate score on a dictionary or object."""

        if isinstance(candidate, dict):
            candidate["combined_score"] = float(score)
            candidate["score"] = float(score)
            return

        if hasattr(candidate, "combined_score"):
            setattr(candidate, "combined_score", float(score))
        elif hasattr(candidate, "score"):
            setattr(candidate, "score", float(score))

    def set_candidate_rerank_score(self, candidate: T, score: float) -> None:
        """Set raw reranker score on a dictionary or object."""

        if isinstance(candidate, dict):
            candidate["rerank_score"] = float(score)
            return

        if hasattr(candidate, "rerank_score"):
            setattr(candidate, "rerank_score", float(score))

    @staticmethod
    def minmax_normalize(scores: Sequence[float]) -> list[float]:
        """Normalize arbitrary scores into the 0-1 interval."""

        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0 if max_score > 0 else 0.0 for _ in scores]

        return [(score - min_score) / (max_score - min_score) for score in scores]