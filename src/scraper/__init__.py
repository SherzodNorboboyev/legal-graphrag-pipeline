"""Scraper package for the Legal GraphRAG pipeline.

This package contains the crawl, checkpoint, parsing, and Markdown conversion
layers for Oman legal documents from qanoon.om and linked translation sources.
"""

from src.scraper.checkpoint import CheckpointManager, CrawlCheckpoint
from src.scraper.crawler import QanoonCrawler
from src.scraper.markdown_converter import HtmlToMarkdownConverter, MarkdownConversionResult
from src.scraper.parser import DiscoveredLinks, LegalCrossReference, ParsedLegalDocument, QanoonParser

__all__ = [
    "CheckpointManager",
    "CrawlCheckpoint",
    "QanoonCrawler",
    "HtmlToMarkdownConverter",
    "MarkdownConversionResult",
    "DiscoveredLinks",
    "LegalCrossReference",
    "ParsedLegalDocument",
    "QanoonParser",
]