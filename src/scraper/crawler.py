"""Asynchronous crawler for qanoon.om legal documents.

The crawler is robust by design:
- random user-agent rotation
- custom request headers
- random throttling
- retries with exponential backoff
- timeout handling
- HTML/PDF content detection
- defensive link discovery
- Arabic/English linking
- checkpoint/resume
- raw response and metadata persistence
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
from loguru import logger

from src.config import Settings
from src.scraper.checkpoint import CheckpointDocumentRecord, CheckpointManager, CrawlCheckpoint
from src.scraper.markdown_converter import HtmlToMarkdownConverter, MarkdownConversionResult
from src.scraper.parser import ParsedLegalDocument, QanoonParser


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]


@dataclass(frozen=True)
class FetchResult:
    """HTTP response data required by the crawler."""

    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    headers: dict[str, str]
    body: bytes

    @property
    def text(self) -> str:
        """Decode body as UTF-8 text with replacement for malformed bytes."""

        return self.body.decode("utf-8", errors="replace")

    @property
    def is_html(self) -> bool:
        """Return True if the response looks like HTML."""

        content_type = self.content_type.lower()
        return "text/html" in content_type or "application/xhtml+xml" in content_type

    @property
    def is_pdf(self) -> bool:
        """Return True if the response looks like PDF."""

        content_type = self.content_type.lower()
        final_url = self.final_url.lower()
        return "application/pdf" in content_type or final_url.endswith(".pdf") or "data.qanoon.om" in final_url

    @property
    def sha256(self) -> str:
        """Return SHA-256 digest of response body."""

        return hashlib.sha256(self.body).hexdigest()


class QanoonCrawler:
    """Resumable async crawler for Oman legal documents."""

    def __init__(
        self,
        settings: Settings,
        parser: QanoonParser | None = None,
        converter: HtmlToMarkdownConverter | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ):
        self.settings = settings
        self.settings.ensure_directories()

        self.parser = parser or QanoonParser(
            qanoon_base_url=settings.qanoon_base_url,
            decree_base_url=settings.decree_base_url,
        )
        self.converter = converter or HtmlToMarkdownConverter()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(settings.checkpoint_file)

    async def run(
        self,
        max_pages: int | None = None,
        start_urls: Iterable[str] | None = None,
    ) -> CrawlCheckpoint:
        """Run the crawler until queue is exhausted or page budget is reached."""

        checkpoint = self.checkpoint_manager.load()
        page_budget = max_pages or self.settings.max_pages

        if not checkpoint.queued_urls and not checkpoint.visited_urls:
            seed_urls = list(start_urls or self.default_start_urls())
            inserted = checkpoint.enqueue_many([self.parser.normalize_url(url) for url in seed_urls])
            logger.info("Initialized crawl queue with {} seed URLs.", inserted)
            self.checkpoint_manager.save(checkpoint)

        queue: deque[str] = deque(checkpoint.queued_urls)
        processed_count = 0

        logger.info(
            "Starting crawl run. budget={}, queued={}, visited={}, failed={}",
            page_budget,
            len(queue),
            len(checkpoint.visited_urls),
            len(checkpoint.failed_urls),
        )

        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        limits = httpx.Limits(max_connections=6, max_keepalive_connections=3)

        async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
            while queue and processed_count < page_budget:
                current_url = self.parser.normalize_url(queue.popleft())

                if current_url in checkpoint.visited_urls:
                    continue

                if not self.parser.is_allowed_crawl_url(current_url):
                    logger.debug("Skipping out-of-scope URL: {}", current_url)
                    checkpoint.mark_visited(current_url)
                    checkpoint.queued_urls = list(queue)
                    self.checkpoint_manager.save(checkpoint)
                    continue

                try:
                    fetch_result = await self.fetch_with_retries(client, current_url)

                    if fetch_result is None:
                        checkpoint.mark_failed(current_url, "fetch_failed_after_retries")
                        checkpoint.queued_urls = list(queue)
                        self.checkpoint_manager.save(checkpoint)
                        continue

                    raw_path = self.save_raw_response(fetch_result)
                    processed_count += 1

                    if fetch_result.is_pdf:
                        await self.process_pdf_response(fetch_result, raw_path, checkpoint)
                    elif fetch_result.is_html:
                        newly_discovered = await self.process_html_response(
                            client=client,
                            fetch_result=fetch_result,
                            raw_path=raw_path,
                            checkpoint=checkpoint,
                        )

                        for discovered_url in newly_discovered:
                            normalized = self.parser.normalize_url(discovered_url)
                            if normalized not in checkpoint.visited_urls and normalized not in queue:
                                queue.append(normalized)
                    else:
                        logger.info(
                            "Skipping unsupported content type url={} content_type={}",
                            fetch_result.final_url,
                            fetch_result.content_type,
                        )

                    checkpoint.mark_visited(current_url)
                except Exception as exc:  # noqa: BLE001 - page-level errors must not kill whole crawl
                    logger.exception("Unexpected processing failure for {}: {}", current_url, exc)
                    checkpoint.mark_failed(current_url, f"{type(exc).__name__}: {exc}")
                finally:
                    checkpoint.queued_urls = list(queue)
                    self.checkpoint_manager.save(checkpoint)
                    await self.random_throttle()

        logger.info(
            "Crawl finished. processed={}, queued_remaining={}, visited={}, failed={}, documents={}",
            processed_count,
            len(checkpoint.queued_urls),
            len(checkpoint.visited_urls),
            len(checkpoint.failed_urls),
            len(checkpoint.documents),
        )
        return checkpoint

    def default_start_urls(self) -> list[str]:
        """Return conservative crawl seeds.

        The parser will discover category, listing, document, PDF, and language
        URLs from these entry points.
        """

        return [
            self.settings.qanoon_base_url,
            self.settings.decree_base_url,
        ]

    async def fetch_with_retries(self, client: httpx.AsyncClient, url: str) -> FetchResult | None:
        """Fetch a URL with retry, timeout handling, and exponential backoff."""

        last_error: Exception | None = None
        max_attempts = self.settings.request_retries + 1

        for attempt in range(max_attempts):
            try:
                response = await client.get(url, headers=self.build_headers())

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"Retryable status code: {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                response.raise_for_status()

                return FetchResult(
                    requested_url=url,
                    final_url=self.parser.normalize_url(str(response.url)),
                    status_code=response.status_code,
                    content_type=response.headers.get("content-type", ""),
                    headers=self.safe_response_headers(response.headers),
                    body=response.content,
                )

            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning("Timeout fetching {} attempt {}/{}", url, attempt + 1, max_attempts)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response else "unknown"
                logger.warning(
                    "HTTP error fetching {} attempt {}/{} status={}",
                    url,
                    attempt + 1,
                    max_attempts,
                    status_code,
                )
                if exc.response is not None and 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                    break
            except httpx.RequestError as exc:
                last_error = exc
                logger.warning("Request error fetching {} attempt {}/{}: {}", url, attempt + 1, max_attempts, exc)

            if attempt < max_attempts - 1:
                sleep_seconds = self.retry_sleep_seconds(attempt)
                logger.debug("Sleeping {:.2f}s before retrying {}", sleep_seconds, url)
                await asyncio.sleep(sleep_seconds)

        logger.error("Failed to fetch {} after {} attempts. Last error: {}", url, max_attempts, last_error)
        return None

    async def process_html_response(
        self,
        client: httpx.AsyncClient,
        fetch_result: FetchResult,
        raw_path: Path,
        checkpoint: CrawlCheckpoint,
    ) -> list[str]:
        """Process HTML response, save document if applicable, and return discovered URLs."""

        html = fetch_result.text
        final_url = fetch_result.final_url

        discovered = self.parser.discover_links(html, final_url)
        crawlable_urls = discovered.all_crawlable_urls(include_pdfs=True)

        if not self.parser.is_document_url(final_url):
            logger.debug(
                "Processed listing/non-document page {}. discovered={}",
                final_url,
                len(crawlable_urls),
            )
            return crawlable_urls

        try:
            parsed = self.parser.parse_document(html, final_url)
            markdown_result = self.converter.convert(
                parsed.content_html or html,
                source_url=final_url,
                title=parsed.title,
            )

            document_payload = self.build_document_payload(
                parsed=parsed,
                markdown_result=markdown_result,
                raw_paths={parsed.language: str(raw_path)},
            )

            if self.settings.scrape_english:
                await self.merge_linked_language_content(
                    client=client,
                    base_payload=document_payload,
                    parsed=parsed,
                    checkpoint=checkpoint,
                )

            output_path = self.save_document_payload(document_payload)
            self.record_checkpoint_document(checkpoint, parsed, output_path, raw_paths=document_payload.get("raw_paths", {}))

            logger.info("Saved parsed document id={} url={}", parsed.id, final_url)
        except Exception as exc:  # noqa: BLE001 - parser errors should not stop discovery
            logger.exception("Failed to parse document page {}: {}", final_url, exc)
            checkpoint.mark_failed(final_url, f"parse_failed:{type(exc).__name__}:{exc}")

        return crawlable_urls

    async def process_pdf_response(
        self,
        fetch_result: FetchResult,
        raw_path: Path,
        checkpoint: CrawlCheckpoint,
    ) -> None:
        """Process a PDF response and save a Markdown-wrapped document payload."""

        try:
            parsed = self.parser.parse_pdf_metadata(fetch_result.final_url)
            markdown_result = self.converter.convert_pdf_bytes(
                fetch_result.body,
                source_url=fetch_result.final_url,
                title=parsed.title,
            )

            document_payload = self.build_document_payload(
                parsed=parsed,
                markdown_result=markdown_result,
                raw_paths={f"{parsed.language}_pdf": str(raw_path)},
            )

            output_path = self.save_document_payload(document_payload)
            self.record_checkpoint_document(checkpoint, parsed, output_path, raw_paths=document_payload.get("raw_paths", {}))

            logger.info("Saved PDF-derived document id={} url={}", parsed.id, fetch_result.final_url)
        except Exception as exc:  # noqa: BLE001 - PDF extraction can fail on malformed/scanned PDFs
            logger.exception("Failed to process PDF {}: {}", fetch_result.final_url, exc)
            checkpoint.mark_failed(fetch_result.final_url, f"pdf_parse_failed:{type(exc).__name__}:{exc}")

    async def merge_linked_language_content(
        self,
        client: httpx.AsyncClient,
        base_payload: dict[str, Any],
        parsed: ParsedLegalDocument,
        checkpoint: CrawlCheckpoint,
    ) -> None:
        """Fetch and merge linked Arabic/English version when discovered.

        This method is defensive: if translation fetch or parse fails, the base
        document is still saved. The translation URL also remains crawlable via
        the main queue, so failures can be retried in later runs.
        """

        desired_language = "en" if parsed.language == "ar" else "ar"
        language_url = parsed.language_urls.get(desired_language)

        if not language_url:
            return

        if language_url in checkpoint.visited_urls:
            logger.debug("Linked language URL already visited: {}", language_url)
            return

        if not self.parser.is_allowed_crawl_url(language_url):
            logger.debug("Linked language URL is out of scope: {}", language_url)
            return

        fetch_result = await self.fetch_with_retries(client, language_url)

        if fetch_result is None:
            checkpoint.mark_failed(language_url, "linked_language_fetch_failed")
            return

        raw_path = self.save_raw_response(fetch_result)

        try:
            if fetch_result.is_pdf:
                linked_parsed = self.parser.parse_pdf_metadata(
                    fetch_result.final_url,
                    title=parsed.title,
                    fallback_language=desired_language,
                )
                markdown_result = self.converter.convert_pdf_bytes(
                    fetch_result.body,
                    source_url=fetch_result.final_url,
                    title=linked_parsed.title,
                )
            else:
                linked_parsed = self.parser.parse_document(
                    fetch_result.text,
                    fetch_result.final_url,
                    fallback_language=desired_language,
                )
                markdown_result = self.converter.convert(
                    linked_parsed.content_html or fetch_result.text,
                    source_url=fetch_result.final_url,
                    title=linked_parsed.title,
                )

            content_key = "contentEn" if desired_language == "en" else "contentAr"
            title_key = "title_en" if desired_language == "en" else "title_ar"

            base_payload[content_key] = markdown_result.markdown
            base_payload[title_key] = linked_parsed.title
            base_payload.setdefault("language_urls", {})[desired_language] = fetch_result.final_url
            base_payload.setdefault("raw_paths", {})[desired_language] = str(raw_path)
            base_payload.setdefault("metadata", {})["linked_language_merged"] = True

            checkpoint.mark_visited(language_url)
            logger.info("Merged linked {} content from {}", desired_language, fetch_result.final_url)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to merge linked language page {}: {}", language_url, exc)
            checkpoint.mark_failed(language_url, f"linked_language_parse_failed:{type(exc).__name__}:{exc}")

    def build_document_payload(
        self,
        parsed: ParsedLegalDocument,
        markdown_result: MarkdownConversionResult,
        raw_paths: dict[str, str],
    ) -> dict[str, Any]:
        """Build serialized document payload for data/markdown."""

        content_ar = markdown_result.markdown if parsed.language == "ar" else None
        content_en = markdown_result.markdown if parsed.language == "en" else None

        payload: dict[str, Any] = {
            "id": parsed.id,
            "title": parsed.title,
            "title_ar": parsed.title if parsed.language == "ar" else None,
            "title_en": parsed.title if parsed.language == "en" else None,
            "date": parsed.date,
            "document_type": parsed.document_type,
            "number": parsed.number,
            "issuer": parsed.issuer,
            "source_url": parsed.source_url,
            "language": parsed.language,
            "language_urls": parsed.language_urls,
            "pdf_urls": parsed.pdf_urls,
            "contentAr": content_ar,
            "contentEn": content_en,
            "cross_references": [reference.model_dump(mode="json") for reference in parsed.cross_references],
            "raw_paths": raw_paths,
            "metadata": {
                **parsed.metadata,
                "markdown_warnings": markdown_result.warnings or [],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
        }

        return payload

    def save_document_payload(self, payload: dict[str, Any]) -> Path:
        """Save parsed Markdown document JSON, merging with existing language data."""

        document_id = payload["id"]
        output_path = self.settings.markdown_dir / f"{document_id}.json"

        if output_path.exists():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                payload = self.merge_document_payload(existing, payload)
            except Exception as exc:  # noqa: BLE001 - overwrite if existing file is invalid
                logger.warning("Failed to merge existing document payload {}: {}", output_path, exc)

        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path

    def merge_document_payload(self, existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        """Merge new document data into an existing payload without losing languages."""

        merged = dict(existing)

        for key, value in incoming.items():
            if value in (None, "", [], {}):
                continue

            if key in {"language_urls", "raw_paths", "metadata"}:
                merged.setdefault(key, {})
                merged[key].update(value)
            elif key == "pdf_urls":
                merged[key] = sorted(set(merged.get(key, []) + value))
            elif key == "cross_references":
                existing_refs = merged.get(key, [])
                serialized_existing = {json.dumps(ref, ensure_ascii=False, sort_keys=True) for ref in existing_refs}
                for ref in value:
                    serialized = json.dumps(ref, ensure_ascii=False, sort_keys=True)
                    if serialized not in serialized_existing:
                        existing_refs.append(ref)
                merged[key] = existing_refs
            else:
                merged[key] = value

        merged.setdefault("metadata", {})["updated_at"] = datetime.now(timezone.utc).isoformat()
        return merged

    def record_checkpoint_document(
        self,
        checkpoint: CrawlCheckpoint,
        parsed: ParsedLegalDocument,
        output_path: Path,
        raw_paths: dict[str, str],
    ) -> None:
        """Record saved document metadata in checkpoint."""

        checkpoint.record_document(
            CheckpointDocumentRecord(
                document_id=parsed.id,
                source_url=parsed.source_url,
                title=parsed.title,
                language=parsed.language,
                language_urls=parsed.language_urls,
                raw_paths=raw_paths,
                output_path=str(output_path),
                metadata={
                    "document_type": parsed.document_type,
                    "number": parsed.number,
                    "issuer": parsed.issuer,
                },
            )
        )

    def save_raw_response(self, fetch_result: FetchResult) -> Path:
        """Save raw response body and JSON metadata sidecar under data/raw."""

        extension = self.raw_extension(fetch_result)
        file_stem = self.raw_file_stem(fetch_result.final_url)
        raw_path = self.settings.raw_dir / f"{file_stem}{extension}"
        metadata_path = self.settings.raw_dir / f"{file_stem}{extension}.json"

        raw_path.write_bytes(fetch_result.body)

        metadata = {
            "requested_url": fetch_result.requested_url,
            "final_url": fetch_result.final_url,
            "status_code": fetch_result.status_code,
            "content_type": fetch_result.content_type,
            "headers": fetch_result.headers,
            "sha256": fetch_result.sha256,
            "bytes": len(fetch_result.body),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "raw_path": str(raw_path),
        }

        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        logger.debug("Saved raw response {} and metadata {}", raw_path, metadata_path)
        return raw_path

    def raw_file_stem(self, url: str) -> str:
        """Create a deterministic filesystem-safe file stem from URL."""

        parsed = httpx.URL(url)
        path_slug = re.sub(r"[^A-Za-z0-9]+", "-", parsed.path.strip("/")).strip("-")
        path_slug = path_slug[:90] if path_slug else "root"
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        return f"{path_slug}-{digest}"

    def raw_extension(self, fetch_result: FetchResult) -> str:
        """Infer raw file extension from content type and URL."""

        if fetch_result.is_pdf:
            return ".pdf"

        if fetch_result.is_html:
            return ".html"

        content_type = fetch_result.content_type.lower()

        if "json" in content_type:
            return ".json"

        return ".bin"

    def safe_response_headers(self, headers: httpx.Headers) -> dict[str, str]:
        """Return response headers safe for local metadata storage."""

        excluded = {"set-cookie", "cookie", "authorization", "proxy-authorization"}
        return {key: value for key, value in headers.items() if key.lower() not in excluded}

    def build_headers(self) -> dict[str, str]:
        """Build browser-like custom headers with a random user-agent."""

        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
            "Accept-Language": random.choice(
                [
                    "ar,en-US;q=0.9,en;q=0.8",
                    "en-US,en;q=0.9,ar;q=0.8",
                ]
            ),
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def retry_sleep_seconds(self, attempt: int) -> float:
        """Return exponential backoff delay with jitter."""

        base_delay = min(2**attempt, 30)
        jitter = random.uniform(0.25, 1.75)
        return base_delay + jitter

    async def random_throttle(self) -> None:
        """Sleep for a random polite delay between requests."""

        delay = random.uniform(
            self.settings.throttle_min_seconds,
            self.settings.throttle_max_seconds,
        )

        if delay > 0:
            await asyncio.sleep(delay)