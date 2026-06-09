# Architecture Report: Legal GraphRAG Pipeline for Oman Legal Documents

## 1. Executive summary

This project implements a production-oriented GraphRAG pipeline for Oman legal documents from `https://qanoon.om/`.

The system converts public legal documents into a structured, searchable graph representation. It combines web scraping, Markdown serialization, Neo4j graph storage, topic extraction, semantic chunking, embeddings, hybrid retrieval, optional CrossEncoder reranking, and topic/community optimization.

The architecture is designed to be practical for a take-home engineering assessment while preserving production principles:

- deterministic environment configuration
- modular package structure
- resumable ingestion
- explicit provenance storage
- safe fallback paths
- graph-native schema design
- hybrid retrieval instead of vector-only retrieval

---

## 2. Architecture overview

```text
qanoon.om / decree.om
        |
        v
Async Crawler
        |
        +--> checkpoint JSON
        +--> raw HTML/PDF + sidecar metadata
        |
        v
Parser + Markdown Converter
        |
        +--> data/markdown/*.json
        |
        v
Neo4j Ingestion
        |
        +--> Document nodes
        +--> AMENDS / REPEALS relationships
        |
        v
Topic Extraction
        |
        +--> Topic nodes
        +--> HAS_TOPIC relationships
        |
        v
Semantic Chunking + Embeddings
        |
        +--> Chunk nodes
        +--> HAS_CHUNK relationships
        +--> Topic and Chunk embeddings
        |
        v
Hybrid Retrieval
        |
        +--> vector search
        +--> keyword search
        +--> graph expansion
        +--> optional reranking
        +--> answer synthesis
```

---

## 3. Why GraphRAG

Flat vector search is useful but incomplete for legal corpora. Laws, decrees, and ministerial decisions are not isolated pieces of text. They include:

- document identity
- document type
- legal number
- issuer
- publication date
- language versions
- amendment relationships
- repeal relationships
- legal topics
- topic communities

GraphRAG keeps these relationships explicit. A user query can retrieve a semantically relevant chunk, then traverse to:

- parent document metadata
- linked legal topics
- amended or repealed documents
- related topic communities

This makes retrieval more explainable and better aligned with legal research workflows.

---

## 4. Scraping strategy

The scraper is implemented with async `httpx`. It starts from configured source URLs, discovers listing/document/PDF/language links, and processes pages safely.

Main scraper capabilities:

- custom browser-like headers
- random user-agent rotation
- random throttling
- retries with exponential backoff
- timeout handling
- HTML/PDF content-type detection
- Arabic/English version discovery
- raw response persistence
- sidecar metadata persistence
- parser error isolation
- checkpoint/resume

The crawler is intentionally conservative. It does not attempt to bypass hard access controls. If the website serves CAPTCHAs or persistent blocks, the correct behavior is to pause, reduce crawl rate, or request authorized access.

---

## 5. Anti-bot and checkpointing

The crawler includes polite anti-bot resilience:

```text
random user-agent
custom Accept/Accept-Language headers
random throttle delay
exponential retry with jitter
checkpointed URL queue
failed URL registry
```

Checkpoint state includes:

- visited URLs
- queued URLs
- failed URLs
- parsed document metadata
- output paths

The checkpoint is saved atomically to avoid corruption if the process is interrupted.

---

## 6. Markdown serialization

All HTML/PDF content is normalized into Markdown because Markdown is:

- readable
- easy to inspect
- LLM-friendly
- suitable for semantic chunking
- capable of preserving headings and tables

The converter removes:

- scripts
- styles
- navigation
- headers
- footers
- tracking-like blocks
- social/share widgets

The converter preserves:

- H1/H2/H3 hierarchy
- legal article labels
- tables as Markdown tables
- Arabic text order
- source provenance

PDF extraction uses `pypdf`. If a PDF has no text layer, the pipeline stores raw PDF provenance and creates a warning Markdown wrapper. OCR is not part of the MVP because it increases operational complexity and quality risk.

---

## 7. Graph schema design

The graph uses a simplified legal identity principle: one legal instrument maps to one `Document` node.

### Document

```text
(:Document {
  id,
  title,
  title_ar,
  title_en,
  title_fr,
  date,
  document_type,
  number,
  issuer,
  source_url,
  language,
  contentAr,
  contentEn,
  contentFr,
  language_urls,
  pdf_urls,
  raw_paths,
  metadata,
  created_at,
  updated_at
})
```

Language-specific Markdown is stored directly on the `Document` node:

```text
contentAr
contentEn
contentFr
```

This avoids splitting translations into separate graph nodes and keeps the legal identity consolidated.

### Topic

```text
(:Topic {
  name,
  normalized_name,
  embedding,
  confidence,
  evidence,
  source,
  aliases,
  metadata
})
```

### Chunk

```text
(:Chunk {
  id,
  document_id,
  language,
  text,
  chunk_index,
  heading_context,
  token_count,
  embedding,
  metadata
})
```

### Relationships

```text
(Document)-[:HAS_TOPIC]->(Topic)
(Document)-[:HAS_CHUNK {language, chunk_index}]->(Chunk)
(Document)-[:AMENDS]->(Document)
(Document)-[:REPEALS]->(Document)
```

Indexes and constraints:

- unique `Document.id`
- unique `Topic.normalized_name`
- unique `Chunk.id`
- document lookup indexes
- full-text indexes for document/chunk/topic search
- vector indexes for chunk/topic embeddings where Neo4j supports them

---

## 8. LLM topic extraction

Topic extraction is implemented with two modes:

### OpenAI mode

When configured, the system sends English content first, otherwise Arabic content, to an LLM with a JSON-only prompt.

Expected schema:

```json
{
  "topics": [
    {
      "name": "Taxation",
      "normalized_name": "taxation",
      "confidence": 0.92,
      "evidence": "taxable income"
    }
  ]
}
```

The extractor includes:

- JSON-only system prompt
- bounded content length
- repair prompt on invalid JSON
- retry logic
- Pydantic validation
- confidence filtering
- max topic limit
- deduplication by normalized name

### Fallback mode

If OpenAI is not configured or fails, deterministic keyword extraction is used. It recognizes common legal domains such as:

- Taxation
- Customs
- Labour Regulation
- Omanization
- Public Health
- Education
- Environmental Protection
- Real Estate
- Maritime Law
- Banking Regulation
- Competition Law
- Judicial Administration

Fallback extraction also uses title terms, with stopword and numeric filtering to avoid noisy year-like or generic topics.

---

## 9. Chunking and embeddings

The chunker is Markdown-aware. It parses headings, preserves heading context, and chunks documents into target sizes.

Defaults:

```text
CHUNK_MIN_TOKENS=500
CHUNK_MAX_TOKENS=900
CHUNK_OVERLAP_TOKENS=120
```

Chunk metadata includes:

- document ID
- language
- chunk index
- text
- heading context
- token count
- deterministic chunk ID

Embeddings:

- SentenceTransformers default
- OpenAI optional
- deterministic hashing fallback if model loading fails
- SQLite cache to avoid recomputation
- batch embedding support
- dimension validation
- cosine similarity helper

Embeddings are generated for:

- every `Chunk`
- every `Topic`

---

## 10. Hybrid retrieval design

The retrieval system uses multi-stage search.

### Stage 1: Candidate generation

Dense vector search:

```text
Neo4j vector index over Chunk.embedding
```

Sparse keyword search:

```text
Neo4j full-text index over Chunk.text
```

Fallbacks:

- Python cosine scan for small datasets if vector index is unavailable
- Neo4j `CONTAINS` keyword fallback if full-text index fails

### Stage 2: Score normalization and merge

Vector and keyword scores are normalized to 0-1 and combined:

```text
combined_score = vector_score * HYBRID_VECTOR_WEIGHT
               + keyword_score * HYBRID_KEYWORD_WEIGHT
```

Default weights:

```text
HYBRID_VECTOR_WEIGHT=0.65
HYBRID_KEYWORD_WEIGHT=0.35
```

### Stage 3: Graph context expansion

For each chunk candidate, the retriever appends:

- parent document title
- document number
- document type
- date
- issuer
- source URL
- linked topics
- language
- chunk text

This expanded context improves reranking and answer synthesis.

---

## 11. CrossEncoder reranking

CrossEncoder reranking is optional. It is more accurate than bi-encoder similarity because it scores query-candidate pairs jointly.

Default model:

```text
BAAI/bge-reranker-base
```

The implementation supports:

- lazy model loading
- batch reranking
- fallback to hybrid scores if loading fails
- top-N selection after reranking

Latency trade-off:

- Vector search is fast and scalable.
- CrossEncoder reranking is slower but improves precision.
- The system reranks only the top candidate pool to control latency.

---

## 12. Topic merging

LLMs and fallback extractors can produce duplicate topics such as:

```text
Labour Policies
Labour Regulation
Labor Regulation
```

The topic merger:

1. Fetches topics with embeddings.
2. Computes cosine similarity.
3. Groups topics above a threshold.
4. Chooses a canonical topic name.
5. Consolidates incoming `HAS_TOPIC` relationships.
6. Deletes duplicate topic nodes.
7. Preserves aliases.

Default threshold:

```text
TOPIC_MERGE_SIMILARITY=0.88
```

The command supports dry-run mode and should be reviewed before applying.

---

## 13. Community detection

Community detection is implemented as a Neo4j GDS Louvain function.

Graph projection:

```text
Document -- HAS_TOPIC -- Topic
```

Writeback property:

```text
community_id
```

If Neo4j GDS is unavailable, the command returns a safe warning instead of failing the pipeline.

Community detection is useful for discovering legal subfields, such as:

- taxation and fees
- labour and workforce policy
- transport and infrastructure
- education and public institutions
- banking and financial regulation

---

## 14. Scaling to 1,000,000 legal articles

At large scale, the main bottlenecks are:

1. Crawl throughput and politeness constraints.
2. PDF extraction and OCR requirements.
3. LLM topic extraction cost.
4. Embedding generation throughput.
5. Neo4j write throughput.
6. Vector index build and update latency.
7. CrossEncoder reranking latency.
8. Graph traversal fan-out.

Recommended scaling architecture:

```text
Crawler workers
    |
Message queue
    |
Parser workers
    |
Object storage raw data
    |
Embedding workers
    |
Batch graph writer
    |
Neo4j cluster / read replicas
    |
Search service
```

Optimizations:

- batch graph writes
- cache embeddings
- precompute topic embeddings
- separate ingestion from query workloads
- use worker queues
- use retryable idempotent writes
- limit CrossEncoder reranking to top 20-50 candidates
- optionally use a dedicated vector database if Neo4j vector search becomes limiting

---

## 15. Latency trade-offs

### Vector search

Fast and scalable, but may retrieve semantically close but legally incomplete chunks.

### Keyword search

Precise for legal numbers, names, and exact terms, but misses paraphrases.

### Graph expansion

Adds explainability and metadata but requires graph traversal.

### CrossEncoder reranking

Improves ranking quality but adds model inference latency.

### LLM synthesis

Produces a readable answer but adds cost and latency. The fallback extractive answer remains deterministic and offline.

Latency strategy:

```text
retrieve top_k fast
batch graph expansion
rerank only top candidates
synthesize from top_n only
cache frequent embeddings
keep fallback paths available
```

---

## 16. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Website HTML changes | Defensive parser and TODO selector points |
| Network interruption | Checkpoint/resume |
| Rate limiting | Random throttle, retry, checkpoint |
| CAPTCHA | Pause/reduce rate/request access |
| Invalid LLM JSON | Safe parser and repair prompt |
| No LLM key | Deterministic fallback extraction |
| Model download failure | Hashing embedding fallback |
| Neo4j vector index unavailable | Python cosine fallback |
| Full-text index unavailable | CONTAINS fallback |
| GDS unavailable | Safe community detection warning |
| Scanned PDF | Store raw PDF and warning wrapper |

---

## 17. Evaluation readiness

The repository demonstrates:

- modular architecture
- Pydantic validation
- deterministic IDs
- configurable environment
- Dockerized Neo4j
- real CLI orchestration
- tests for parser, models, chunking, topic extraction, reranking, and topic merging
- fallback logic for optional external dependencies
- clear documentation and troubleshooting
- sample data for offline demo

---

## 18. Final operational sequence

```bash
docker compose up -d neo4j

python -m src.main doctor
python -m src.main setup-schema

python -m src.main scrape --max-pages 25
python -m src.main ingest
python -m src.main extract-topics
python -m src.main embed

python -m src.main merge-topics --dry-run
python -m src.main search "What are the laws about taxation?"
```

For the standalone search client:

```bash
python -m src.search_client "What are the laws about taxation?"
```