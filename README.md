# legal-graphrag-pipeline

Legal GraphRAG pipeline for Oman legal documents from [qanoon.om](https://qanoon.om/).

This repository is designed as a production-oriented take-home assignment project. The final pipeline will support:

1. Crawling Oman legal documents from `qanoon.om`
2. Discovering Arabic and English versions where available
3. Persisting raw HTML/PDF assets
4. Converting legal content into structured Markdown
5. Ingesting legal documents into Neo4j
6. Extracting legal topics with LLM or deterministic fallback logic
7. Creating semantic chunks and embeddings
8. Performing hybrid retrieval using vector search, keyword search, and graph traversal

> Current status: **Part 1 skeleton**. The CLI commands are intentionally present as stubs and will be implemented in later parts.

---

## Project structure

```text
legal-graphrag-pipeline/
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── architecture_report.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   └── main.py
└── data/
    ├── raw/
    │   └── .gitkeep
    ├── markdown/
    │   └── .gitkeep
    └── sample_output/
        └── .gitkeep
```

Planned final structure:

```text
legal-graphrag-pipeline/
├── src/
│   ├── scraper/
│   │   ├── crawler.py
│   │   ├── parser.py
│   │   ├── markdown_converter.py
│   │   └── checkpoint.py
│   ├── ingestion/
│   │   ├── graph_schema.py
│   │   └── graph_ingest.py
│   ├── llm_agents/
│   │   ├── topic_extractor.py
│   │   └── prompts.py
│   ├── vector_ops/
│   │   ├── embeddings.py
│   │   ├── chunking.py
│   │   └── topic_merging.py
│   ├── retrieval/
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   └── search_client.py
└── tests/
```

---

## Architecture overview

```text
+---------------------+
|     qanoon.om       |
| HTML / PDF sources  |
+----------+----------+
           |
           v
+-----------------------------+
| Scraper                     |
| - retries                   |
| - throttling                |
| - checkpointing             |
| - Arabic/English discovery  |
+-------------+---------------+
              |
              v
+-----------------------------+
| Raw Data Store              |
| data/raw/*.html             |
| data/raw/*.pdf              |
| metadata sidecars           |
+-------------+---------------+
              |
              v
+-----------------------------+
| Markdown Transformer        |
| - headings                  |
| - tables                    |
| - body cleanup              |
+-------------+---------------+
              |
              v
+-----------------------------+
| Neo4j Graph                 |
| Document / Topic / Chunk    |
| AMENDS / REPEALS / HAS_*    |
+-------------+---------------+
              |
              v
+-----------------------------+
| GraphRAG Retrieval          |
| vector + keyword + graph    |
| optional cross-encoder      |
+-----------------------------+
```

---

## Technology stack

- Python 3.11+
- Typer CLI
- Pydantic Settings
- Loguru logging
- Neo4j graph database
- Docker Compose
- BeautifulSoup / lxml for parsing
- Markdownify or custom Markdown conversion
- SentenceTransformers or OpenAI embeddings
- Optional Playwright crawler fallback
- Pytest for tests in later parts

---

## Requirements

Install Python dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Environment setup

Copy the example environment file:

```bash
cp .env.example .env
```

Review and update values:

```dotenv
NEO4J_PASSWORD=please-change-me
QANOON_BASE_URL=https://qanoon.om/
DECREE_BASE_URL=https://decree.om/
```

---

## Docker Compose setup

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

For any real environment, change `NEO4J_PASSWORD` before first startup.

Stop services:

```bash
docker compose down
```

Remove local Neo4j volumes:

```bash
docker compose down -v
```

---

## CLI usage

The Part 1 CLI is available through:

```bash
python -m src.main --help
```

Available commands:

```bash
python -m src.main scrape
python -m src.main ingest
python -m src.main extract-topics
python -m src.main embed
python -m src.main merge-topics
python -m src.main search "What are the laws about taxation?"
```

Current commands are stubs. They validate configuration, initialize logging, and show the intended pipeline stage.

---

## Data directories

```text
data/raw/
```

Raw HTML/PDF files and metadata will be stored here.

```text
data/markdown/
```

Normalized Markdown document JSON or Markdown files will be stored here.

```text
data/sample_output/
```

Sample outputs for submission and review will be stored here.

---

## Planned pipeline stages

### 1. Scraping

The scraper will:

- Crawl `qanoon.om`
- Discover legal document links
- Discover Arabic and English versions
- Handle HTML and PDF sources
- Save raw files and metadata
- Support retry, throttling, random user-agent headers, and checkpoint resume

### 2. Markdown serialization

The Markdown converter will:

- Remove scripts, styles, tracking blocks, and layout noise
- Preserve legal headings
- Preserve tables as Markdown tables
- Keep Arabic text intact
- Normalize HTML/PDF content into one format

### 3. Graph ingestion

Neo4j will store:

```text
(:Document)
(:Topic)
(:Chunk)
```

Relationships:

```text
(Document)-[:HAS_TOPIC]->(Topic)
(Document)-[:HAS_CHUNK]->(Chunk)
(Document)-[:AMENDS]->(Document)
(Document)-[:REPEALS]->(Document)
```

### 4. Topic extraction

Topic extraction will use:

- English content first if available
- Arabic content if English is unavailable
- Structured JSON prompts
- Safe JSON parsing and retry logic
- Deterministic keyword fallback

### 5. Chunking and embeddings

The vector pipeline will:

- Chunk Markdown into semantic 500-1000 token chunks
- Preserve heading context
- Generate chunk and topic embeddings
- Cache embeddings locally
- Store vectors in Neo4j

### 6. Hybrid retrieval

Search will combine:

- Dense vector similarity
- Keyword/BM25-style retrieval
- Parent document metadata
- Linked topic traversal
- Optional cross-encoder reranking
- Final grounded answer synthesis

---

## Local development commands

Format and lint tools are not enforced in Part 1, but recommended later:

```bash
pytest -q
python -m src.main --help
python -m src.main scrape
```

---

## Known limitations in Part 1

- Scraping is not implemented yet.
- Neo4j schema creation is not implemented yet.
- Topic extraction is not implemented yet.
- Embedding generation is not implemented yet.
- Search is a CLI stub only.
- Tests will be added in later parts.
