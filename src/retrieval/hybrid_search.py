"""Neo4j-based hybrid search for Legal GraphRAG.

Retrieval stages:
1. Dense vector search over Chunk embeddings.
2. Sparse keyword/full-text search over Chunk text.
3. Score normalization and weighted merge.
4. Graph context expansion with parent Document metadata and linked Topics.
5. Optional CrossEncoder reranking.
6. Optional LLM answer synthesis or deterministic extractive fallback.

Latency-oriented design notes:
- Candidate generation uses Neo4j indexes and limits early with top_k.
- Graph expansion is batched by chunk IDs instead of one query per candidate.
- CrossEncoder reranking is applied only to the merged candidate pool.
- LLM synthesis is optional and receives only final top_n contexts.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from loguru import logger
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

from src.config import Settings, get_settings
from src.retrieval.reranker import CrossEncoderReranker
from src.vector_ops.embeddings import BaseEmbeddingProvider, cosine_similarity, get_embedding_provider


@dataclass
class RetrievalCandidate:
    """One chunk candidate returned by hybrid retrieval."""

    chunk_id: str
    text: str
    document_id: str | None = None
    document_title: str | None = None
    document_number: str | None = None
    document_type: str | None = None
    document_date: str | None = None
    issuer: str | None = None
    source_url: str | None = None
    language: str | None = None
    chunk_index: int | None = None
    topics: list[str] = field(default_factory=list)
    vector_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float | None = None
    match_sources: set[str] = field(default_factory=set)
    expanded_text: str = ""

    @property
    def score(self) -> float:
        """Compatibility property used by generic rerankers."""

        return self.combined_score

    @score.setter
    def score(self, value: float) -> None:
        """Set combined score through compatibility property."""

        self.combined_score = float(value)

    def evidence_preview(self, max_chars: int = 420) -> str:
        """Return a compact evidence snippet for terminal display."""

        compact = re.sub(r"\s+", " ", self.text or "").strip()
        return compact[:max_chars] + ("..." if len(compact) > max_chars else "")

    def document_label(self) -> str:
        """Return a readable document label."""

        pieces = [self.document_title or "Unknown document"]

        if self.document_number:
            pieces.append(f"No. {self.document_number}")

        if self.document_date:
            pieces.append(str(self.document_date))

        return " | ".join(pieces)


@dataclass
class HybridSearchResponse:
    """Full response returned by HybridSearch."""

    query: str
    vector_candidates: list[RetrievalCandidate]
    keyword_candidates: list[RetrievalCandidate]
    merged_candidates: list[RetrievalCandidate]
    final_candidates: list[RetrievalCandidate]
    answer: str
    timings_ms: dict[str, float] = field(default_factory=dict)


class HybridSearch:
    """Hybrid vector, keyword, graph-context retrieval client."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: BaseEmbeddingProvider | None = None,
        driver: Driver | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedding_provider(self.settings)

        if driver is None:
            self.driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
                max_connection_pool_size=15,
            )
            self._owns_driver = True
        else:
            self.driver = driver
            self._owns_driver = False

        self.database = self.settings.neo4j_database
        self.reranker = reranker or CrossEncoderReranker(
            model_name=getattr(self.settings, "cross_encoder_model", "BAAI/bge-reranker-base"),
            enabled=getattr(self.settings, "reranker_enabled", False),
        )
        self._openai_client: Any | None = None

    def __enter__(self) -> "HybridSearch":
        """Return context-managed search client."""

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close owned Neo4j driver."""

        self.close()

    def close(self) -> None:
        """Close Neo4j driver if this class created it."""

        if self._owns_driver:
            self.driver.close()

    def search(
        self,
        query: str,
        *,
        top_k: int = 50,
        top_n: int = 5,
    ) -> HybridSearchResponse:
        """Run the full hybrid retrieval pipeline."""

        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        timings: dict[str, float] = {}

        start = time.perf_counter()
        query_embedding = self.embedder.embed_text(query)
        timings["embed_query"] = self.elapsed_ms(start)

        start = time.perf_counter()
        vector_candidates = self.vector_search(query_embedding, top_k=top_k)
        timings["vector_search"] = self.elapsed_ms(start)

        start = time.perf_counter()
        keyword_candidates = self.keyword_search(query, top_k=top_k)
        timings["keyword_search"] = self.elapsed_ms(start)

        start = time.perf_counter()
        merged_candidates = self.merge_candidates(
            vector_candidates=vector_candidates,
            keyword_candidates=keyword_candidates,
            vector_weight=getattr(self.settings, "hybrid_vector_weight", 0.65),
            keyword_weight=getattr(self.settings, "hybrid_keyword_weight", 0.35),
            top_k=top_k,
        )
        timings["merge"] = self.elapsed_ms(start)

        start = time.perf_counter()
        expanded_candidates = self.expand_graph_context(merged_candidates)
        timings["graph_expansion"] = self.elapsed_ms(start)

        start = time.perf_counter()
        final_candidates = self.reranker.rerank(
            query,
            expanded_candidates,
            top_n=top_n,
            text_getter=lambda candidate: candidate.expanded_text or candidate.text,
        )
        timings["rerank"] = self.elapsed_ms(start)

        start = time.perf_counter()
        answer = self.synthesize_answer(query, final_candidates)
        timings["answer_synthesis"] = self.elapsed_ms(start)

        return HybridSearchResponse(
            query=query,
            vector_candidates=vector_candidates,
            keyword_candidates=keyword_candidates,
            merged_candidates=expanded_candidates,
            final_candidates=final_candidates,
            answer=answer,
            timings_ms=timings,
        )

    def vector_search(self, query_embedding: list[float], *, top_k: int = 50) -> list[RetrievalCandidate]:
        """Retrieve top_k chunks by Neo4j vector index."""

        cypher = """
        CALL db.index.vector.queryNodes('chunk_embedding_vector_index', $top_k, $embedding)
        YIELD node, score
        MATCH (d:Document)-[:HAS_CHUNK]->(node)
        OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)
        RETURN
            node.id AS chunk_id,
            node.text AS text,
            node.language AS language,
            node.chunk_index AS chunk_index,
            score AS vector_score,
            d.id AS document_id,
            d.title AS document_title,
            d.number AS document_number,
            d.document_type AS document_type,
            d.date AS document_date,
            d.issuer AS issuer,
            d.source_url AS source_url,
            collect(DISTINCT coalesce(t.name, t.normalized_name)) AS topics
        ORDER BY vector_score DESC
        """

        try:
            with self.driver.session(database=self.database) as session:
                records = list(session.run(cypher, top_k=top_k, embedding=query_embedding))

            candidates = [self.record_to_candidate(record, source="vector") for record in records]
            self.normalize_candidate_scores(candidates, score_attr="vector_score")
            return candidates

        except Neo4jError as exc:
            logger.warning("Neo4j vector index search failed. Falling back to Python vector scan. Error: {}", exc)
            return self.python_vector_scan_fallback(query_embedding, top_k=top_k)

    def python_vector_scan_fallback(self, query_embedding: list[float], *, top_k: int = 50) -> list[RetrievalCandidate]:
        """Fallback vector retrieval by scanning Chunk embeddings in Python.

        This is suitable for small local datasets. At scale, Neo4j vector indexes
        or a dedicated vector store should be used.
        """

        cypher = """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.embedding IS NOT NULL
        OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)
        RETURN
            c.id AS chunk_id,
            c.text AS text,
            c.language AS language,
            c.chunk_index AS chunk_index,
            c.embedding AS embedding,
            d.id AS document_id,
            d.title AS document_title,
            d.number AS document_number,
            d.document_type AS document_type,
            d.date AS document_date,
            d.issuer AS issuer,
            d.source_url AS source_url,
            collect(DISTINCT coalesce(t.name, t.normalized_name)) AS topics
        """

        try:
            with self.driver.session(database=self.database) as session:
                records = list(session.run(cypher))

            candidates: list[RetrievalCandidate] = []

            for record in records:
                candidate = self.record_to_candidate(record, source="vector_fallback")
                candidate.vector_score = cosine_similarity(query_embedding, record.get("embedding") or [])
                candidates.append(candidate)

            candidates.sort(key=lambda candidate: candidate.vector_score, reverse=True)
            candidates = candidates[:top_k]
            self.normalize_candidate_scores(candidates, score_attr="vector_score")
            return candidates

        except Neo4jError as exc:
            logger.error("Python vector fallback failed: {}", exc)
            return []

    def keyword_search(self, query: str, *, top_k: int = 50) -> list[RetrievalCandidate]:
        """Retrieve keyword candidates via Neo4j full-text search."""

        fulltext_query = self.to_lucene_query(query)

        cypher = """
        CALL db.index.fulltext.queryNodes('chunk_text_fulltext', $search_query, {limit: $top_k})
        YIELD node, score
        MATCH (d:Document)-[:HAS_CHUNK]->(node)
        OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)
        RETURN
            node.id AS chunk_id,
            node.text AS text,
            node.language AS language,
            node.chunk_index AS chunk_index,
            score AS keyword_score,
            d.id AS document_id,
            d.title AS document_title,
            d.number AS document_number,
            d.document_type AS document_type,
            d.date AS document_date,
            d.issuer AS issuer,
            d.source_url AS source_url,
            collect(DISTINCT coalesce(t.name, t.normalized_name)) AS topics
        ORDER BY keyword_score DESC
        """

        try:
            with self.driver.session(database=self.database) as session:
                records = list(session.run(cypher, search_query=fulltext_query, top_k=top_k))

            candidates = [self.record_to_candidate(record, source="keyword") for record in records]
            self.normalize_candidate_scores(candidates, score_attr="keyword_score")
            return candidates

        except Neo4jError as exc:
            logger.warning("Neo4j full-text search failed. Falling back to CONTAINS keyword scan. Error: {}", exc)
            return self.contains_keyword_fallback(query, top_k=top_k)

    def contains_keyword_fallback(self, query: str, *, top_k: int = 50) -> list[RetrievalCandidate]:
        """Fallback keyword search using simple CONTAINS matching."""

        terms = self.keyword_terms(query)
        if not terms:
            return []

        cypher = """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE ANY(term IN $terms WHERE toLower(c.text) CONTAINS term)
        OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)
        WITH c, d, collect(DISTINCT coalesce(t.name, t.normalized_name)) AS topics,
             size([term IN $terms WHERE toLower(c.text) CONTAINS term]) AS match_count
        RETURN
            c.id AS chunk_id,
            c.text AS text,
            c.language AS language,
            c.chunk_index AS chunk_index,
            toFloat(match_count) AS keyword_score,
            d.id AS document_id,
            d.title AS document_title,
            d.number AS document_number,
            d.document_type AS document_type,
            d.date AS document_date,
            d.issuer AS issuer,
            d.source_url AS source_url,
            topics AS topics
        ORDER BY keyword_score DESC
        LIMIT $top_k
        """

        try:
            with self.driver.session(database=self.database) as session:
                records = list(session.run(cypher, terms=terms, top_k=top_k))

            candidates = [self.record_to_candidate(record, source="keyword_fallback") for record in records]
            self.normalize_candidate_scores(candidates, score_attr="keyword_score")
            return candidates

        except Neo4jError as exc:
            logger.error("Keyword fallback failed: {}", exc)
            return []

    def merge_candidates(
        self,
        *,
        vector_candidates: list[RetrievalCandidate],
        keyword_candidates: list[RetrievalCandidate],
        vector_weight: float,
        keyword_weight: float,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        """Merge vector and keyword results by chunk ID."""

        merged: dict[str, RetrievalCandidate] = {}

        for candidate in vector_candidates:
            candidate.match_sources.add("vector")
            candidate.combined_score = candidate.vector_score * vector_weight
            merged[candidate.chunk_id] = candidate

        for candidate in keyword_candidates:
            candidate.match_sources.add("keyword")
            weighted_keyword_score = candidate.keyword_score * keyword_weight

            if candidate.chunk_id in merged:
                existing = merged[candidate.chunk_id]
                existing.keyword_score = max(existing.keyword_score, candidate.keyword_score)
                existing.combined_score += weighted_keyword_score
                existing.match_sources.add("keyword")
                existing.topics = sorted(set(existing.topics + candidate.topics))
            else:
                candidate.combined_score = weighted_keyword_score
                merged[candidate.chunk_id] = candidate

        ranked = sorted(merged.values(), key=lambda candidate: candidate.combined_score, reverse=True)
        return ranked[:top_k]

    def expand_graph_context(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """Attach parent Document metadata and linked Topic context to candidates."""

        if not candidates:
            return []

        chunk_ids = [candidate.chunk_id for candidate in candidates]
        candidate_by_id = {candidate.chunk_id: candidate for candidate in candidates}

        cypher = """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.id IN $chunk_ids
        OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)
        RETURN
            c.id AS chunk_id,
            c.text AS text,
            c.language AS language,
            c.chunk_index AS chunk_index,
            d.id AS document_id,
            d.title AS document_title,
            d.number AS document_number,
            d.document_type AS document_type,
            d.date AS document_date,
            d.issuer AS issuer,
            d.source_url AS source_url,
            collect(DISTINCT coalesce(t.name, t.normalized_name)) AS topics
        """

        try:
            with self.driver.session(database=self.database) as session:
                records = list(session.run(cypher, chunk_ids=chunk_ids))

            for record in records:
                chunk_id = record["chunk_id"]
                candidate = candidate_by_id.get(chunk_id)

                if candidate is None:
                    continue

                self.update_candidate_from_record(candidate, record)
                candidate.expanded_text = self.build_expanded_context(candidate)

        except Neo4jError as exc:
            logger.warning("Graph context expansion failed. Using existing candidate metadata. Error: {}", exc)
            for candidate in candidates:
                candidate.expanded_text = self.build_expanded_context(candidate)

        return candidates

    def build_expanded_context(self, candidate: RetrievalCandidate) -> str:
        """Build text used for reranking and answer synthesis."""

        topics = ", ".join(candidate.topics) if candidate.topics else "No linked topics"

        return (
            f"Document title: {candidate.document_title or 'Unknown'}\n"
            f"Document number: {candidate.document_number or 'N/A'}\n"
            f"Document type: {candidate.document_type or 'N/A'}\n"
            f"Date: {candidate.document_date or 'N/A'}\n"
            f"Issuer: {candidate.issuer or 'N/A'}\n"
            f"Source URL: {candidate.source_url or 'N/A'}\n"
            f"Language: {candidate.language or 'unknown'}\n"
            f"Linked topics: {topics}\n\n"
            f"Chunk text:\n{candidate.text}"
        )

    def synthesize_answer(self, query: str, candidates: Sequence[RetrievalCandidate]) -> str:
        """Generate final answer using configured LLM or extractive fallback."""

        if not candidates:
            return (
                "No matching legal contexts were found. Run scrape, ingest, "
                "extract-topics, and embed before searching."
            )

        if self.should_use_openai_synthesis():
            try:
                return self.synthesize_with_openai(query, candidates)
            except Exception as exc:  # noqa: BLE001 - fallback answer is required
                logger.warning("OpenAI answer synthesis failed. Using extractive answer. Error: {}", exc)

        return self.extractive_answer(query, candidates)

    def synthesize_with_openai(self, query: str, candidates: Sequence[RetrievalCandidate]) -> str:
        """Synthesize a grounded answer with OpenAI."""

        from openai import OpenAI

        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)

        contexts = "\n\n---\n\n".join(
            (candidate.expanded_text or candidate.text)[:3500]
            for candidate in candidates
        )

        prompt = (
            "You are a careful legal research assistant. Answer only from the "
            "retrieved Oman legal document contexts. Do not provide legal advice. "
            "Mention document titles or numbers when available. If the contexts "
            "are insufficient, say so.\n\n"
            f"User question:\n{query}\n\n"
            f"Retrieved contexts:\n{contexts}\n\n"
            "Write a concise grounded answer."
        )

        response = self._openai_client.chat.completions.create(
            model=getattr(self.settings, "openai_chat_model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.choices[0].message.content or self.extractive_answer(query, candidates)

    def extractive_answer(self, query: str, candidates: Sequence[RetrievalCandidate]) -> str:
        """Create deterministic answer from retrieved contexts."""

        lines = ["Based on the retrieved graph and chunk context:"]

        for index, candidate in enumerate(candidates, start=1):
            topics = ", ".join(candidate.topics[:6]) if candidate.topics else "no linked topics"
            evidence = self.best_evidence_sentence(candidate.text, query)

            lines.append(
                f"{index}. {candidate.document_label()} appears relevant. "
                f"Linked topics: {topics}. Evidence: {evidence}"
            )

        lines.append(
            "This is a retrieval summary, not legal advice. Verify the cited "
            "source documents before relying on the result."
        )

        return "\n".join(lines)

    def best_evidence_sentence(self, text: str, query: str) -> str:
        """Select a compact sentence with lexical overlap against the query."""

        query_terms = set(self.keyword_terms(query))
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?؟])\s+|\n+", text or "") if sentence.strip()]

        if not sentences:
            return ""

        best_sentence = sentences[0]
        best_overlap = -1

        for sentence in sentences:
            sentence_terms = set(self.keyword_terms(sentence))
            overlap = len(query_terms & sentence_terms)

            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence

        compact = re.sub(r"\s+", " ", best_sentence).strip()
        return compact[:500] + ("..." if len(compact) > 500 else "")

    def should_use_openai_synthesis(self) -> bool:
        """Return True if LLM answer synthesis is configured."""

        return (
            getattr(self.settings, "synthesis_provider", "fallback") == "openai"
            and bool(getattr(self.settings, "openai_api_key", None))
        )

    def record_to_candidate(self, record: Any, *, source: str) -> RetrievalCandidate:
        """Convert a Neo4j record into a RetrievalCandidate."""

        topics = [topic for topic in list(record.get("topics") or []) if topic]
        candidate = RetrievalCandidate(
            chunk_id=str(record.get("chunk_id") or ""),
            text=str(record.get("text") or ""),
            document_id=record.get("document_id"),
            document_title=record.get("document_title"),
            document_number=record.get("document_number"),
            document_type=record.get("document_type"),
            document_date=record.get("document_date"),
            issuer=record.get("issuer"),
            source_url=record.get("source_url"),
            language=record.get("language"),
            chunk_index=record.get("chunk_index"),
            topics=topics,
            vector_score=float(record.get("vector_score") or 0.0),
            keyword_score=float(record.get("keyword_score") or 0.0),
            match_sources={source},
        )
        candidate.expanded_text = self.build_expanded_context(candidate)
        return candidate

    def update_candidate_from_record(self, candidate: RetrievalCandidate, record: Any) -> None:
        """Update candidate metadata from a graph expansion record."""

        candidate.text = str(record.get("text") or candidate.text)
        candidate.document_id = record.get("document_id") or candidate.document_id
        candidate.document_title = record.get("document_title") or candidate.document_title
        candidate.document_number = record.get("document_number") or candidate.document_number
        candidate.document_type = record.get("document_type") or candidate.document_type
        candidate.document_date = record.get("document_date") or candidate.document_date
        candidate.issuer = record.get("issuer") or candidate.issuer
        candidate.source_url = record.get("source_url") or candidate.source_url
        candidate.language = record.get("language") or candidate.language
        candidate.chunk_index = record.get("chunk_index") if record.get("chunk_index") is not None else candidate.chunk_index
        candidate.topics = sorted(set(candidate.topics + [topic for topic in list(record.get("topics") or []) if topic]))

    def to_lucene_query(self, query: str) -> str:
        """Convert user query into a simple Lucene OR query."""

        terms = self.keyword_terms(query)

        if not terms:
            return query

        escaped_terms = [self.escape_lucene_term(term) for term in terms[:12]]
        return " OR ".join(term for term in escaped_terms if term)

    def keyword_terms(self, text: str) -> list[str]:
        """Extract normalized keyword terms for fallback search."""

        terms = re.findall(r"[\w\u0600-\u06FF]{3,}", (text or "").lower(), flags=re.UNICODE)
        stopwords = {
            "the",
            "and",
            "are",
            "about",
            "what",
            "with",
            "from",
            "this",
            "that",
            "laws",
            "law",
            "ما",
            "هي",
            "في",
            "من",
            "على",
            "عن",
            "هذا",
            "هذه",
            "قانون",
        }
        return [term for term in terms if term not in stopwords]

    def escape_lucene_term(self, term: str) -> str:
        """Escape Lucene special characters conservatively."""

        return re.sub(r'([+\-!(){}\[\]^"~*?:\\/])', r"\\\1", term)

    @staticmethod
    def normalize_candidate_scores(candidates: list[RetrievalCandidate], *, score_attr: str) -> None:
        """Normalize a candidate score field in place to 0-1."""

        if not candidates:
            return

        scores = [float(getattr(candidate, score_attr, 0.0) or 0.0) for candidate in candidates]
        normalized = HybridSearch.normalize_scores(scores)

        for candidate, score in zip(candidates, normalized, strict=True):
            setattr(candidate, score_attr, score)

    @staticmethod
    def normalize_scores(scores: Sequence[float]) -> list[float]:
        """Normalize numeric scores into 0-1 range."""

        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0 if max_score > 0 else 0.0 for _ in scores]

        return [(score - min_score) / (max_score - min_score) for score in scores]

    @staticmethod
    def elapsed_ms(start_time: float) -> float:
        """Return elapsed milliseconds from perf_counter start time."""

        return round((time.perf_counter() - start_time) * 1000, 3)