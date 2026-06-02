"""HTML/PDF to Markdown conversion utilities.

The converter is intentionally conservative: it removes layout/noise elements,
keeps legal hierarchy, converts tables to GitHub-flavored Markdown, and does not
transform Arabic text direction or character order.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from html import escape
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag
from loguru import logger
from markdownify import markdownify as markdownify_html


NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "nav",
    "header",
    "footer",
    "aside",
    ".site-header",
    ".site-footer",
    ".main-navigation",
    ".navigation",
    ".nav-links",
    ".sidebar",
    ".widget",
    ".menu",
    ".search-form",
    ".sharedaddy",
    ".cookie-notice",
    ".adsbygoogle",
    ".advertisement",
    ".social-navigation",
]

CONTENT_SELECTORS = [
    "article .entry-content",
    "article .post-content",
    "article",
    "main .entry-content",
    "main .post-content",
    "main",
    ".entry-content",
    ".post-content",
    ".content",
    "#content",
    "body",
]


@dataclass(frozen=True)
class MarkdownConversionResult:
    """Result returned by the Markdown converter."""

    markdown: str
    title: str | None = None
    source_url: str | None = None
    warnings: list[str] | None = None


class HtmlToMarkdownConverter:
    """Convert HTML legal document bodies and PDF text into Markdown."""

    def convert(self, html: str, source_url: str | None = None, title: str | None = None) -> MarkdownConversionResult:
        """Convert an HTML document or fragment to Markdown.

        Parameters
        ----------
        html:
            Raw HTML document or HTML fragment.
        source_url:
            Optional source URL. Used to resolve relative links.
        title:
            Optional title override. If not provided, the converter tries to
            extract the title from the HTML.
        """

        warnings: list[str] = []

        if not html or not html.strip():
            return MarkdownConversionResult(markdown="", title=title, source_url=source_url, warnings=["empty_html"])

        soup = BeautifulSoup(html, "lxml")
        self._remove_noise(soup)
        self._make_links_absolute(soup, source_url)

        extracted_title = title or self._extract_title(soup)
        content = self._select_content(soup)

        if content is None:
            warnings.append("no_content_container_found")
            content = soup

        self._promote_legal_article_labels(soup, content)
        self._replace_tables_with_markdown(content)

        markdown = markdownify_html(
            str(content),
            heading_style="ATX",
            bullets="-",
            strip=["span"],
        )

        markdown = self._postprocess_markdown(markdown)

        if extracted_title and not markdown.lstrip().startswith("#"):
            markdown = f"# {extracted_title}\n\n{markdown}".strip() + "\n"

        return MarkdownConversionResult(
            markdown=markdown,
            title=extracted_title,
            source_url=source_url,
            warnings=warnings,
        )

    def convert_pdf_text(
        self,
        text: str,
        source_url: str | None = None,
        title: str | None = None,
        extraction_warning: str | None = None,
    ) -> MarkdownConversionResult:
        """Wrap extracted PDF text in a Markdown document.

        This method is the safe fallback for PDF content. It does not assume that
        PDF extraction is perfect. If no text is available, it creates a small
        Markdown stub that preserves provenance and warns downstream stages.
        """

        warnings: list[str] = []
        if extraction_warning:
            warnings.append(extraction_warning)

        cleaned_text = self._postprocess_markdown(text or "")
        heading = title or "PDF Document"

        parts = [f"# {heading}"]

        if source_url:
            parts.append(f"Source: {source_url}")

        if cleaned_text:
            parts.append(cleaned_text)
        else:
            warnings.append("empty_pdf_text")
            parts.append(
                "> PDF text extraction produced no readable text. "
                "The raw PDF should be kept for manual inspection or OCR processing."
            )

        markdown = "\n\n".join(parts).strip() + "\n"
        return MarkdownConversionResult(markdown=markdown, title=heading, source_url=source_url, warnings=warnings)

    def convert_pdf_bytes(
        self,
        pdf_bytes: bytes,
        source_url: str | None = None,
        title: str | None = None,
    ) -> MarkdownConversionResult:
        """Extract text from PDF bytes and wrap it as Markdown.

        The implementation uses `pypdf`, which works when the PDF contains a text
        layer. Scanned PDFs are not OCRed in this project part; the method returns
        a Markdown warning instead of failing the whole crawl.
        """

        if not pdf_bytes:
            return self.convert_pdf_text(
                "",
                source_url=source_url,
                title=title,
                extraction_warning="empty_pdf_bytes",
            )

        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning("pypdf is not installed; PDF text extraction is unavailable.")
            return self.convert_pdf_text(
                "",
                source_url=source_url,
                title=title,
                extraction_warning="pypdf_not_installed",
            )

        pages: list[str] = []
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page_index, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                    page_text = self._postprocess_markdown(page_text)
                    if page_text:
                        pages.append(f"## Page {page_index}\n\n{page_text}")
                except Exception as exc:  # noqa: BLE001 - continue through bad PDF pages
                    logger.warning("Failed to extract PDF page {} from {}: {}", page_index, source_url, exc)
                    pages.append(f"## Page {page_index}\n\n> Text extraction failed for this page.")
        except Exception as exc:  # noqa: BLE001 - PDF parsing should not kill crawler
            logger.exception("Failed to parse PDF bytes from {}: {}", source_url, exc)
            return self.convert_pdf_text(
                "",
                source_url=source_url,
                title=title,
                extraction_warning=f"pdf_parse_failed:{type(exc).__name__}",
            )

        return self.convert_pdf_text(
            "\n\n".join(pages),
            source_url=source_url,
            title=title,
            extraction_warning=None if pages else "pdf_text_layer_not_found",
        )

    def table_to_markdown(self, table: Tag) -> str:
        """Convert an HTML table element into Markdown table syntax."""

        rows: list[list[str]] = []

        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue

            row: list[str] = []
            for cell in cells:
                colspan = self._safe_int(cell.get("colspan"), default=1)
                text = self._escape_table_cell(cell.get_text(" ", strip=True))
                row.append(text)
                for _ in range(max(colspan - 1, 0)):
                    row.append("")
            rows.append(row)

        if not rows:
            return ""

        max_cols = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]

        header = normalized_rows[0]
        separator = ["---"] * max_cols
        body = normalized_rows[1:]

        markdown_rows = [
            self._format_table_row(header),
            self._format_table_row(separator),
        ]
        markdown_rows.extend(self._format_table_row(row) for row in body)

        return "\n".join(markdown_rows)

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        """Remove scripts, styles, navigation, headers, footers, and tracking blocks."""

        for selector in NOISE_SELECTORS:
            for tag in soup.select(selector):
                tag.decompose()

        for tag in list(soup.find_all(True)):
            class_text = " ".join(tag.get("class", []))
            id_text = tag.get("id", "")
            marker = f"{class_text} {id_text}".lower()

            if any(token in marker for token in ["advert", "tracking", "analytics", "cookie", "share", "social"]):
                tag.decompose()

    def _make_links_absolute(self, soup: BeautifulSoup, source_url: str | None) -> None:
        """Resolve relative `href` and `src` attributes using the page URL."""

        if not source_url:
            return

        for tag in soup.find_all(["a", "img"]):
            attribute = "href" if tag.name == "a" else "src"
            value = tag.get(attribute)

            if value:
                tag[attribute] = urljoin(source_url, value)

    def _select_content(self, soup: BeautifulSoup) -> Tag | BeautifulSoup | None:
        """Return the most likely legal content container.

        TODO:
            If qanoon.om changes its WordPress theme, add the new theme-specific
            body selector to `CONTENT_SELECTORS`. The current list deliberately
            prefers semantic containers before falling back to `body`.
        """

        for selector in CONTENT_SELECTORS:
            candidate = soup.select_one(selector)
            if candidate and candidate.get_text(" ", strip=True):
                return candidate

        return soup.body or soup

    def _extract_title(self, soup: BeautifulSoup) -> str | None:
        """Extract page title from semantic and WordPress-style locations."""

        selectors = [
            "h1.entry-title",
            "article h1",
            "main h1",
            "h1",
            "h2.entry-title",
            "article h2",
            "main h2",
            "title",
        ]

        for selector in selectors:
            tag = soup.select_one(selector)
            if tag:
                text = self._clean_text(tag.get_text(" ", strip=True))
                if text:
                    return text

        for attr_name, attr_value in [
            ("property", "og:title"),
            ("name", "twitter:title"),
            ("name", "title"),
        ]:
            meta = soup.find("meta", attrs={attr_name: attr_value})
            if meta and meta.get("content"):
                return self._clean_text(meta["content"])

        return None

    def _promote_legal_article_labels(self, soup: BeautifulSoup, root: Tag | BeautifulSoup) -> None:
        """Promote plain legal article labels into H3 headings.

        Some legal sites render article labels as normal paragraphs or bold text.
        Promoting short labels such as "Article One" or "المادة الأولى" preserves
        the legal structure for later semantic chunking.
        """

        article_label_pattern = re.compile(
            r"^(article\s+[\w\d]+|chapter\s+[\w\d]+|section\s+[\w\d]+|"
            r"المادة\s+\S+|الفصل\s+\S+|الباب\s+\S+)$",
            flags=re.IGNORECASE,
        )

        for paragraph in list(root.find_all(["p", "div"])):
            text = self._clean_text(paragraph.get_text(" ", strip=True))
            if not text or len(text) > 90:
                continue

            if article_label_pattern.match(text):
                h3 = soup.new_tag("h3")
                h3.string = text
                paragraph.replace_with(h3)

    def _replace_tables_with_markdown(self, root: Tag | BeautifulSoup) -> None:
        """Replace each HTML table node with a Markdown table text node."""

        for table in list(root.find_all("table")):
            markdown_table = self.table_to_markdown(table)
            replacement = NavigableString(f"\n\n{markdown_table}\n\n")
            table.replace_with(replacement)

    def _format_table_row(self, cells: Iterable[str]) -> str:
        """Format a Markdown table row."""

        return "| " + " | ".join(cells) + " |"

    def _escape_table_cell(self, text: str) -> str:
        """Escape Markdown table pipes and normalize whitespace inside cells."""

        cleaned = self._clean_text(text)
        return cleaned.replace("|", "\\|")

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace without modifying Arabic character order."""

        text = (text or "").replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _postprocess_markdown(self, markdown: str) -> str:
        """Normalize Markdown whitespace while preserving content order."""

        markdown = (markdown or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
        markdown = re.sub(r"[ \t]+\n", "\n", markdown)
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        lines = [line.rstrip() for line in markdown.split("\n")]
        markdown = "\n".join(lines).strip()

        if not markdown:
            return ""

        return markdown + "\n"

    def _safe_int(self, value: object, default: int) -> int:
        """Safely parse an integer HTML attribute."""

        try:
            return int(str(value))
        except (TypeError, ValueError):
            return default


def escape_html_text(text: str) -> str:
    """Escape text for safe insertion into generated HTML snippets."""

    return escape(text or "", quote=False)