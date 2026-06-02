"""JSON checkpoint manager for resumable scraping.

The crawler is expected to run for a long time when attempting full coverage of
qanoon.om. A local checkpoint file allows the process to resume after network
errors, machine restarts, rate limits, or parser failures.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


class CheckpointDocumentRecord(BaseModel):
    """Metadata recorded for a parsed/saved document during crawling."""

    document_id: str
    source_url: str
    title: str | None = None
    language: str | None = None
    language_urls: dict[str, str] = Field(default_factory=dict)
    raw_paths: dict[str, str] = Field(default_factory=dict)
    output_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utc_now_iso)


class CrawlCheckpoint(BaseModel):
    """Serializable crawler state.

    Attributes
    ----------
    visited_urls:
        URLs already processed by the crawler.
    queued_urls:
        URLs discovered but not processed yet.
    failed_urls:
        Mapping of failed URL to failure reason.
    documents:
        Parsed document metadata keyed by document ID.
    updated_at:
        Last checkpoint save timestamp.
    """

    visited_urls: set[str] = Field(default_factory=set)
    queued_urls: list[str] = Field(default_factory=list)
    failed_urls: dict[str, str] = Field(default_factory=dict)
    documents: dict[str, CheckpointDocumentRecord] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utc_now_iso)

    def enqueue(self, url: str) -> bool:
        """Add a URL to the queue if it is not already visited or queued.

        Returns
        -------
        bool
            True if the URL was added, False otherwise.
        """

        if not url:
            return False
        if url in self.visited_urls or url in self.queued_urls:
            return False
        self.queued_urls.append(url)
        return True

    def enqueue_many(self, urls: list[str]) -> int:
        """Add multiple URLs to the queue and return the number inserted."""

        inserted = 0
        for url in urls:
            if self.enqueue(url):
                inserted += 1
        return inserted

    def mark_visited(self, url: str) -> None:
        """Mark a URL as visited and remove it from the pending queue."""

        self.visited_urls.add(url)
        self.queued_urls = [queued_url for queued_url in self.queued_urls if queued_url != url]

    def mark_failed(self, url: str, reason: str) -> None:
        """Record a failed URL without aborting the whole crawl."""

        self.failed_urls[url] = reason
        self.visited_urls.add(url)
        self.queued_urls = [queued_url for queued_url in self.queued_urls if queued_url != url]

    def record_document(self, record: CheckpointDocumentRecord) -> None:
        """Attach saved document metadata to the checkpoint."""

        record.updated_at = utc_now_iso()
        self.documents[record.document_id] = record


class CheckpointManager:
    """Atomic JSON checkpoint manager.

    The manager writes to a temporary file first and then replaces the target
    checkpoint path atomically. This prevents corrupted checkpoint files when the
    Python process is killed during save.
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> CrawlCheckpoint:
        """Load checkpoint from disk or return an empty checkpoint."""

        if not self.checkpoint_path.exists():
            logger.info("Checkpoint file does not exist. Starting new crawl state: {}", self.checkpoint_path)
            return CrawlCheckpoint()

        try:
            raw_text = self.checkpoint_path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
            checkpoint = CrawlCheckpoint.model_validate(payload)
            logger.info(
                "Loaded checkpoint: visited={}, queued={}, failed={}, documents={}",
                len(checkpoint.visited_urls),
                len(checkpoint.queued_urls),
                len(checkpoint.failed_urls),
                len(checkpoint.documents),
            )
            return checkpoint
        except Exception as exc:  # noqa: BLE001 - corrupted checkpoints should not kill startup
            logger.exception("Failed to load checkpoint {}. Starting fresh. Error: {}", self.checkpoint_path, exc)
            return CrawlCheckpoint()

    def save(self, checkpoint: CrawlCheckpoint) -> None:
        """Persist checkpoint to disk atomically."""

        checkpoint.updated_at = utc_now_iso()
        payload = checkpoint.model_dump(mode="json")

        fd, temp_path = tempfile.mkstemp(
            prefix=f"{self.checkpoint_path.name}.",
            suffix=".tmp",
            dir=str(self.checkpoint_path.parent),
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2, sort_keys=True)
                fp.write("\n")

            os.replace(temp_path, self.checkpoint_path)
            logger.debug("Checkpoint saved to {}", self.checkpoint_path)
        except Exception:
            logger.exception("Failed to save checkpoint to {}", self.checkpoint_path)
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def reset(self) -> None:
        """Delete checkpoint file if it exists."""

        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.warning("Deleted checkpoint file: {}", self.checkpoint_path)

    def update_queue(self, checkpoint: CrawlCheckpoint, queued_urls: list[str]) -> None:
        """Replace pending queue and immediately persist checkpoint."""

        checkpoint.queued_urls = queued_urls
        self.save(checkpoint)