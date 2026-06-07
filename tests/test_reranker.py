from src.retrieval.hybrid_search import HybridSearch, RetrievalCandidate
from src.retrieval.reranker import CrossEncoderReranker


def test_score_normalization() -> None:
    scores = [10.0, 20.0, 30.0]

    normalized = HybridSearch.normalize_scores(scores)

    assert normalized == [0.0, 0.5, 1.0]


def test_score_normalization_equal_positive_values() -> None:
    scores = [5.0, 5.0, 5.0]

    normalized = HybridSearch.normalize_scores(scores)

    assert normalized == [1.0, 1.0, 1.0]


def test_candidate_score_normalization_in_place() -> None:
    candidates = [
        RetrievalCandidate(chunk_id="c1", text="one", vector_score=2.0),
        RetrievalCandidate(chunk_id="c2", text="two", vector_score=4.0),
    ]

    HybridSearch.normalize_candidate_scores(candidates, score_attr="vector_score")

    assert candidates[0].vector_score == 0.0
    assert candidates[1].vector_score == 1.0


def test_fallback_reranker_sorts_by_existing_score_when_disabled() -> None:
    candidates = [
        RetrievalCandidate(chunk_id="low", text="low relevance", combined_score=0.2),
        RetrievalCandidate(chunk_id="high", text="high relevance", combined_score=0.9),
        RetrievalCandidate(chunk_id="mid", text="medium relevance", combined_score=0.5),
    ]

    reranker = CrossEncoderReranker(enabled=False)
    ranked = reranker.rerank("query", candidates, top_n=2)

    assert [candidate.chunk_id for candidate in ranked] == ["high", "mid"]


def test_model_load_failure_falls_back_to_existing_score(monkeypatch) -> None:
    candidates = [
        RetrievalCandidate(chunk_id="a", text="candidate a", combined_score=0.3),
        RetrievalCandidate(chunk_id="b", text="candidate b", combined_score=0.7),
    ]

    reranker = CrossEncoderReranker(enabled=True)
    monkeypatch.setattr(reranker, "_load_model", lambda: None)

    ranked = reranker.rerank("query", candidates)

    assert [candidate.chunk_id for candidate in ranked] == ["b", "a"]
    assert all(candidate.rerank_score is None for candidate in ranked)


def test_reranker_supports_dictionary_candidates_when_disabled() -> None:
    candidates = [
        {"chunk_id": "a", "text": "candidate a", "combined_score": 0.1},
        {"chunk_id": "b", "text": "candidate b", "combined_score": 0.8},
    ]

    reranker = CrossEncoderReranker(enabled=False)
    ranked = reranker.rerank("query", candidates)

    assert ranked[0]["chunk_id"] == "b"