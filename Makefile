.PHONY: install test neo4j-up neo4j-down setup-schema scrape-sample ingest-sample search doctor clean-cache

PYTHON ?= python
QUERY ?= What are the laws about taxation?

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

neo4j-up:
	docker compose up -d neo4j

neo4j-down:
	docker compose down

setup-schema:
	$(PYTHON) -m src.main setup-schema

scrape-sample:
	$(PYTHON) -m src.main scrape --max-pages 25

ingest-sample:
	mkdir -p data/markdown
	cp data/sample_output/sample_metadata.json data/markdown/sample_metadata.json
	$(PYTHON) -m src.main setup-schema
	$(PYTHON) -m src.main ingest --limit 1
	$(PYTHON) -m src.main extract-topics --limit 1
	$(PYTHON) -m src.main embed --limit 1

search:
	$(PYTHON) -m src.search_client "$(QUERY)"

doctor:
	$(PYTHON) -m src.main doctor

clean-cache:
	rm -f data/embedding_cache.sqlite3
	rm -f data/crawl_checkpoint.json