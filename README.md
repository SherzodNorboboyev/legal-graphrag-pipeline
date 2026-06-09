# legal-graphrag-pipeline

Production-oriented Legal GraphRAG pipeline for Oman legal documents from [qanoon.om](https://qanoon.om/).

The pipeline crawls public legal documents, converts HTML/PDF content to Markdown, stores consolidated multilingual legal documents in Neo4j, extracts topics, creates semantic chunks and embeddings, and exposes hybrid graph/vector search through a CLI.

---

## 1. Project overview

This repository implements an end-to-end GraphRAG system for Oman legislation.

Pipeline stages:

1. **Scrape** qanoon.om and linked English translation pages.
2. **Persist raw provenance** as HTML/PDF plus metadata.
3. **Convert content to Markdown** while preserving legal hierarchy and tables.
4. **Ingest into Neo4j** as `Document`, `Topic`, and `Chunk` nodes.
5. **Extract legal topics** using OpenAI when configured or deterministic fallback logic.
6. **Chunk Markdown semantically** with heading context and overlap.
7. **Generate embeddings** with SentenceTransformers by default or OpenAI optionally.
8. **Search with hybrid retrieval**: dense vector search, sparse keyword search, graph expansion, optional cross-encoder reranking, and answer synthesis.

---

## 2. Architecture diagram

```text
+-----------------------------+
|        qanoon.om            |
| Arabic legal HTML/PDF       |
+--------------+--------------+
               |
               | English links where available
               v
+-----------------------------+
|        decree.om            |
| English translations        |
+--------------+--------------+
               |
               v
+-----------------------------+
| Scraper                     |
| - async HTTP                |
| - retry/backoff             |
| - random user-agent         |
| - throttling                |
| - checkpoint/resume         |
+--------------+--------------+
               |
               v
+-----------------------------+
| Raw Store                   |
| data/raw/*.html             |
| data/raw/*.pdf              |
| data/raw/*.json metadata    |
+--------------+--------------+
               |
               v
+-----------------------------+
| Markdown Serialization      |
| - clean noise               |
| - preserve headings         |
| - preserve tables           |
| - Arabic-safe whitespace    |
+--------------+--------------+
               |
               v
+-----------------------------+
| Neo4j Graph                 |
| Document                    |
| Topic                       |
| Chunk                       |
| AMENDS / REPEALS            |
| HAS_TOPIC / HAS_CHUNK       |
+--------------+--------------+
               |
               v
+-----------------------------+
| Vector and Graph Retrieval  |
| - dense vector search       |
| - full-text keyword search  |
| - graph context expansion   |
| - optional cross-encoder    |
| - final answer synthesis    |
+-----------------------------+
```

---

## 3. Repository structure

```text
legal-graphrag-pipeline/
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── architecture_report.md
├── Makefile
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── models.py
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── crawler.py
│   │   ├── parser.py
│   │   ├── markdown_converter.py
│   │   └── checkpoint.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── graph_schema.py
│   │   └── graph_ingest.py
│   ├── llm_agents/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── topic_extractor.py
│   ├── vector_ops/
│   │   ├── __init__.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   └── topic_merging.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   └── search_client.py
├── data/
│   ├── raw/
│   ├── markdown/
│   └── sample_output/
│       ├── sample_document_en.md
│       ├── sample_document_ar.md
│       └── sample_metadata.json
└── tests/
```

---

## 4. Technology stack

- Python 3.11+
- Typer CLI
- Pydantic Settings
- Loguru logging
- httpx async crawler
- BeautifulSoup / lxml parsing
- markdownify and custom table conversion
- Neo4j 5 Community Edition
- SentenceTransformers default embeddings
- OpenAI optional embeddings, topic extraction, and synthesis
- SQLite embedding cache
- CrossEncoder reranker optional
- pytest test suite

---

## 5. Setup

```bash
git clone <your-repo-url> legal-graphrag-pipeline
cd legal-graphrag-pipeline

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Optional Playwright browser install:

```bash
playwright install chromium
```

Playwright is disabled by default. Enable only if HTML rendering fallback is needed.

---

## 6. Docker Neo4j

Start Neo4j:

```bash
docker compose up -d neo4j
```

Open Neo4j Browser:

```text
http://localhost:7474
```

Default local credentials from `.env.example`:

```text
username: neo4j
password: please-change-me
```

Stop Neo4j:

```bash
docker compose down
```

Remove local Neo4j volumes:

```bash
docker compose down -v
```

Change `NEO4J_PASSWORD` before any non-local deployment.

---

## 7. Environment variables

| Variable | Description |
|---|---|
| `APP_ENV` | Runtime environment label. |
| `LOG_LEVEL` | Default logging level. |
| `DATA_DIR` | Root data directory. |
| `RAW_DIR` | Raw HTML/PDF output directory. |
| `MARKDOWN_DIR` | Parsed document JSON output directory. |
| `SAMPLE_OUTPUT_DIR` | Sample output directory. |
| `CHECKPOINT_FILE` | Crawl checkpoint path. |
| `EMBEDDING_CACHE_PATH` | SQLite embedding cache path. |
| `QANOON_BASE_URL` | Arabic source website. |
| `DECREE_BASE_URL` | English translation source. |
| `SCRAPE_ENGLISH` | Whether to fetch linked English pages. |
| `USE_PLAYWRIGHT` | Optional browser rendering fallback. |
| `MAX_PAGES` | Default crawl budget. |
| `REQUEST_TIMEOUT_SECONDS` | HTTP timeout. |
| `REQUEST_RETRIES` | Retry count per request. |
| `THROTTLE_MIN_SECONDS` | Minimum random delay between requests. |
| `THROTTLE_MAX_SECONDS` | Maximum random delay between requests. |
| `NEO4J_URI` | Neo4j Bolt URI. |
| `NEO4J_USER` | Neo4j username. |
| `NEO4J_PASSWORD` | Neo4j password. |
| `NEO4J_DATABASE` | Neo4j database name. |
| `NEO4J_VECTOR_DIMENSIONS` | Default vector dimensions for local embeddings. |
| `EMBEDDING_PROVIDER` | `sentence_transformers` or `openai`. |
| `SENTENCE_TRANSFORMER_MODEL` | Local embedding model. |
| `OPENAI_API_KEY` | Optional OpenAI key. |
| `TOPIC_LLM_PROVIDER` | `fallback` or `openai`. |
| `OPENAI_CHAT_MODEL` | OpenAI chat model for topic extraction/synthesis. |
| `SYNTHESIS_PROVIDER` | `fallback` or `openai`. |
| `CHUNK_MIN_TOKENS` | Minimum chunk size target. |
| `CHUNK_MAX_TOKENS` | Maximum chunk size target. |
| `CHUNK_OVERLAP_TOKENS` | Chunk overlap. |
| `HYBRID_VECTOR_WEIGHT` | Dense search score weight. |
| `HYBRID_KEYWORD_WEIGHT` | Keyword score weight. |
| `RERANKER_ENABLED` | Enable CrossEncoder reranking. |
| `CROSS_ENCODER_MODEL` | CrossEncoder reranker model. |
| `TOPIC_MERGE_SIMILARITY` | Default duplicate topic merge threshold. |

---

## 8. Full pipeline commands

Recommended local flow:

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

Full crawl:

```bash
python -m src.main scrape
```

Resume crawl automatically:

```bash
python -m src.main scrape
```

Restart crawl from scratch:

```bash
python -m src.main scrape --reset-checkpoint
```

Dry-run examples:

```bash
python -m src.main scrape --max-pages 25 --dry-run
python -m src.main ingest --dry-run
python -m src.main extract-topics --dry-run
python -m src.main embed --dry-run
python -m src.main merge-topics --dry-run
```

---

## 9. CLI search usage

Main CLI:

```bash
python -m src.main search "What are the laws about taxation?"
```

Standalone search client:

```bash
python -m src.search_client "What are the laws about taxation?"
python -m src.search_client "What are the rules about labour regulation?" --top-k 50 --top-n 5 --debug
```

The search output shows:

- vector-matched chunks
- keyword matches
- parent document metadata
- linked topics
- final summarized answer

---

## 10. Testing

Run all tests:

```bash
pytest
```

Run selected tests:

```bash
python -m pytest tests/test_markdown_converter.py tests/test_parser.py
python -m pytest tests/test_graph_models.py
python -m pytest tests/test_chunking.py tests/test_topic_extractor.py
python -m pytest tests/test_reranker.py tests/test_topic_merging.py
```

---

## 11. Sample data

Sample files are included under:

```text
data/sample_output/
```

To ingest the sample document:

```bash
mkdir -p data/markdown
cp data/sample_output/sample_metadata.json data/markdown/sample_metadata.json

python -m src.main setup-schema
python -m src.main ingest --limit 1
python -m src.main extract-topics --limit 1
python -m src.main embed --limit 1
python -m src.main search "tax registration fees"
```

Or use Makefile:

```bash
make ingest-sample
make search QUERY="tax registration fees"
```

---

## 12. Troubleshooting

### Neo4j authentication fails

Ensure `.env` and `docker-compose.yml` use the same password. If Neo4j was already initialized with a different password, recreate the volume:

```bash
docker compose down -v
docker compose up -d neo4j
```

### Vector index creation fails

Neo4j vector index support depends on Neo4j version. The schema setup treats vector indexes as optional. Retrieval can fall back to Python cosine scanning for small datasets.

### Full-text search fails

Run:

```bash
python -m src.main setup-schema
```

Then retry search.

### No search results

Run the pipeline in order:

```bash
python -m src.main scrape --max-pages 25
python -m src.main ingest
python -m src.main extract-topics
python -m src.main embed
```

### SentenceTransformer model download is slow

The first run downloads the model. For production, pre-warm the model cache or use a deployment image with the model already available.

### OpenAI is not used

By default the project runs offline with fallback extraction and local embeddings. To use OpenAI:

```dotenv
OPENAI_API_KEY=...
TOPIC_LLM_PROVIDER=openai
SYNTHESIS_PROVIDER=openai
EMBEDDING_PROVIDER=openai
```

### Topic merging produced strange merges

Automatic topic merging is powerful but should be reviewed. Use dry-run first:

```bash
python -m src.main merge-topics --dry-run
```

Only apply after inspecting candidate merges:

```bash
python -m src.main merge-topics --apply
```

---

## 13. Known limitations

- Public website selectors may change. The parser is defensive and includes TODO points for future selector updates.
- PDF extraction works only for PDFs with a text layer. Scanned PDFs require OCR, which is intentionally excluded from this MVP.
- Fallback topic extraction can create generic topic terms when LLM extraction is disabled. Stopword filtering and dry-run merge review reduce this risk.
- Graph relationship extraction is heuristic and links AMENDS/REPEALS only when target documents are present in the graph.
- CrossEncoder reranking improves precision but increases latency and model load time.
- Louvain community detection requires Neo4j Graph Data Science plugin, not included in the default Neo4j Community Docker image.

---

## 14. Six-day implementation roadmap

| Day | Deliverable |
|---|---|
| Day 1 | Repository skeleton, config, Docker Compose, CLI stubs, README draft |
| Day 2 | Async scraper, parser, checkpoint manager, Markdown converter |
| Day 3 | Pydantic models, Neo4j schema, graph ingestion |
| Day 4 | LLM topic extraction, semantic chunking, embedding providers |
| Day 5 | Hybrid search, CrossEncoder reranker, topic merging, community detection |
| Day 6 | Final CLI integration, README, architecture report, sample data, final checklist |

---

## 15. Interview preparation answers

### How does the scraper behave when throttling increases or CAPTCHAs appear?

The scraper uses randomized user-agent headers, custom request headers, random throttling, retry with exponential backoff, timeout handling, and checkpoint/resume. If the website starts returning rate-limit or block responses, the crawler preserves progress and continues later. If a real CAPTCHA is served, the correct production behavior is not to bypass it; the pipeline should reduce rate, pause, or request authorized data access.

### Why use GraphRAG instead of flat vector search?

Legal documents contain explicit structure: document type, number, issuer, dates, amendments, repeals, topics, and multilingual versions. GraphRAG preserves those relationships and allows retrieval to combine semantic similarity with legal topology. This produces more explainable answers than flat chunk retrieval alone.

### Why store languages as properties on one Document node?

The legal identity is the document, not the translation. Storing `contentAr`, `contentEn`, and `contentFr` on one node avoids duplicate graph identities and keeps topics, chunks, amendments, and repeals attached to the same legal instrument.

### Why merge topics at threshold 0.88?

A high cosine threshold reduces false merges. Topic labels are short, so overly low thresholds can merge legally distinct concepts. `0.88` is a conservative default that should be validated against a labeled sample. Production usage should dry-run merge plans before applying.

### Where are bottlenecks at 1,000,000 articles?

The main bottlenecks are crawling throughput, PDF parsing, LLM topic extraction cost, embedding generation, Neo4j write throughput, vector index build time, and CrossEncoder latency. Scale by separating crawl/parse/embed/ingest queues, batching writes, caching embeddings, using worker pools, and limiting reranking to top candidates.

### What are the latency trade-offs of CrossEncoder reranking?

Bi-encoder vector search is fast because query and documents are embedded independently. CrossEncoders are more accurate because they jointly score query-candidate pairs, but they are slower. This project applies CrossEncoder reranking only after top-k candidate generation and keeps it optional.

---

## Development Notes

This project was developed using AI-assisted engineering workflows for rapid prototyping and code generation.

All architectural decisions, integration work, testing, debugging, and final validation were performed manually.