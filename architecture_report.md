# Architecture Report Draft

## Project title

Legal GraphRAG Pipeline for Oman Legal Documents

## Current phase

Part 1: repository skeleton, environment configuration, Docker setup, CLI entrypoint, and documentation draft.

---

## 1. Executive summary

This project implements a GraphRAG pipeline for Oman legal documents from `https://qanoon.om/`.

The goal is to transform public legal documents into a searchable graph-based retrieval system. The final pipeline will crawl legal documents, normalize HTML/PDF sources into Markdown, store document-level properties in Neo4j, extract topics, create semantic chunks, generate embeddings, and provide hybrid search over graph and vector representations.

The project is designed for staged implementation. Part 1 establishes the production foundation:

- deterministic configuration through Pydantic Settings
- Dockerized Neo4j
- local data directory structure
- CLI command surface
- documentation and architecture draft

---

## 2. Target data flow

```text
qanoon.om
   |
   v
Scraper
   |
   +--> raw HTML/PDF storage
   |
   v
Markdown converter
   |
   v
Document JSON / Markdown artifacts
   |
   v
Neo4j graph ingestion
   |
   +--> Document nodes
   +--> Topic nodes
   +--> Chunk nodes
   |
   v
Embeddings and graph relationships
   |
   v
Hybrid retrieval client
```

---

## 3. Why GraphRAG

A legal corpus is not just unstructured text. Legal documents contain:

- document numbers
- dates
- issuers
- document types
- Arabic and English language versions
- amendments
- repeals
- references to related legal instruments
- topics and legal domains

Flat vector search is useful for semantic similarity, but it loses explicit legal relationships. GraphRAG improves explainability and retrieval quality by combining:

1. semantic similarity from embeddings
2. graph traversal from legal relationships
3. metadata filtering on document properties
4. topic-level retrieval and clustering

---

## 4. Planned graph model

### Document node

```text
(:Document {
  id,
  title,
  date,
  document_type,
  number,
  issuer,
  source_url,
  contentAr,
  contentEn,
  created_at,
  updated_at
})
```

The design stores Arabic and English Markdown content directly on the same `Document` node. This avoids splitting translations into separate nodes and keeps legal identity centralized.

### Topic node

```text
(:Topic {
  name,
  normalized_name,
  embedding
})
```

### Chunk node

```text
(:Chunk {
  id,
  document_id,
  language,
  text,
  chunk_index,
  embedding
})
```

### Relationships

```text
(Document)-[:HAS_TOPIC]->(Topic)
(Document)-[:HAS_CHUNK {language: "en"|"ar"}]->(Chunk)
(Document)-[:AMENDS]->(Document)
(Document)-[:REPEALS]->(Document)
```

---

## 5. Part 1 repository design

The current repository contains:

```text
src/config.py
```

Central application settings using Pydantic Settings.

```text
src/main.py
```

Typer-based CLI with pipeline command stubs.

```text
docker-compose.yml
```

Neo4j Community Edition container with persistent volumes.

```text
data/
```

Local runtime data directories.

```text
README.md
```

Setup and usage documentation.

---

## 6. Configuration strategy

All runtime values are controlled through environment variables. The repository includes `.env.example` as the template.

Key configuration categories:

- application environment
- data paths
- source websites
- scraper behavior
- Neo4j connection
- embedding provider
- LLM provider
- chunking settings
- hybrid retrieval settings
- topic merging threshold

This avoids hardcoded secrets and makes the project portable across local, staging, and production environments.

---

## 7. Logging strategy

The project uses Loguru for structured CLI-friendly logging.

Logging goals:

- consistent format across commands
- configurable log level
- clear startup diagnostics
- readable error messages
- no accidental secret logging

Part 1 configures logging centrally in `src/config.py`, and each CLI command initializes logging before execution.

---

## 8. Docker and Neo4j design

Neo4j is deployed through Docker Compose with:

- HTTP browser exposed on port `7474`
- Bolt driver exposed on port `7687`
- persistent data volume
- persistent logs volume
- persistent plugins volume
- APOC plugin enabled for future graph utilities

The initial setup uses Neo4j Community Edition. Advanced community detection such as Louvain may require Neo4j Graph Data Science in later stages.

---

## 9. Planned scraper architecture

Future scraper modules will include:

```text
src/scraper/crawler.py
src/scraper/parser.py
src/scraper/markdown_converter.py
src/scraper/checkpoint.py
```

The scraper should support:

- retry with exponential backoff
- random user-agent headers
- polite throttling
- checkpoint/resume
- raw HTML/PDF storage
- Arabic/English version linking
- robust parser fallbacks if selectors change
- logging and continuation on page-level errors

---

## 10. Planned Markdown serialization strategy

The Markdown conversion layer should:

- remove scripts and tracking markup
- remove navigation and layout noise
- preserve H1/H2/H3 hierarchy
- convert tables to Markdown tables
- preserve Arabic text without direction corruption
- support PDF text extraction when a text layer exists

Markdown is chosen because it is:

- readable
- LLM-friendly
- easy to chunk semantically
- suitable for provenance review

---

## 11. Planned topic extraction strategy

Topic extraction will use `contentEn` first where available. If no English text exists, it will use `contentAr`.

The LLM prompt should request structured JSON:

```json
{
  "topics": [
    {
      "name": "Taxation",
      "normalized_name": "taxation",
      "confidence": 0.93,
      "evidence": "short supporting phrase"
    }
  ]
}
```

The implementation should include:

- JSON schema expectations in the prompt
- retry on invalid JSON
- safe parsing
- deterministic keyword fallback
- normalized topic names for deduplication

---

## 12. Planned chunking and embedding strategy

Chunking target:

- 500-1000 approximate tokens
- overlap support
- Markdown heading context preservation
- chunk metadata persistence

Embeddings:

- SentenceTransformers by default
- OpenAI embeddings optionally
- local embedding cache
- embeddings for both `Topic` and `Chunk` nodes

---

## 13. Planned hybrid retrieval strategy

The search client will combine:

1. vector similarity over chunk embeddings
2. keyword or BM25-style matching
3. graph traversal for document and topic context
4. optional cross-encoder reranking
5. final answer synthesis

The CLI should display:

- top matching chunks
- parent document metadata
- linked topics
- summarized final answer

---

## 14. Scaling considerations

Potential bottlenecks:

- crawling throughput and website throttling
- PDF extraction quality and latency
- LLM topic extraction cost
- embedding generation throughput
- Neo4j write throughput
- vector index performance
- cross-encoder reranking latency

Planned mitigations:

- resumable crawling
- batched graph writes
- embedding cache
- queue-based ingestion stages
- limiting reranking to top candidates
- optional local or hosted model providers

---

## 15. Risk and fallback plan

| Risk | Fallback |
|---|---|
| qanoon.om HTML changes | defensive parser and selector TODOs |
| network errors | retry and checkpoint resume |
| anti-bot throttling | random pacing and user-agent rotation |
| invalid LLM output | safe JSON parser and retry |
| no LLM key | keyword-based fallback extraction |
| Neo4j vector index unavailable | Python-side cosine fallback |
| scanned PDF | store raw file and log extraction limitation |

---

## 16. Implementation roadmap

### Part 1

- repository skeleton
- dependencies
- environment configuration
- Docker Compose
- Pydantic Settings
- Typer CLI stubs
- README draft
- architecture draft

### Part 2

- scraper
- retry logic
- checkpointing
- raw HTML/PDF persistence

### Part 3

- parser
- Markdown converter
- tests for HTML and table conversion

### Part 4

- Neo4j schema
- graph ingestion
- document relationship extraction

### Part 5

- topic extraction
- chunking
- embeddings

### Part 6

- hybrid retrieval
- reranker
- topic merging
- final tests and submission hardening

---

## 17. Current limitations

This is the initial architecture and repository foundation only. The CLI commands are stubs and do not yet perform scraping, ingestion, extraction, embedding, merging, or search. Later parts will replace each stub with real implementation while preserving the same command surface.