"""Defensive parser for qanoon.om and related English translation pages."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup, Tag
from loguru import logger
from pydantic import BaseModel, Field


ARABIC_DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

DOCUMENT_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"مرسوم\s+سلطاني|royal\s+decree", flags=re.IGNORECASE), "Royal Decree"),
    (re.compile(r"قرار\s+وزاري|ministerial\s+decision", flags=re.IGNORECASE), "Ministerial Decision"),
    (re.compile(r"أمر\s+سامي|الأوامر\s+السامية|royal\s+order", flags=re.IGNORECASE), "Royal Order"),
    (re.compile(r"اتفاقية|معاهدة|agreement|treaty|convention", flags=re.IGNORECASE), "Treaty / Agreement"),
    (re.compile(r"الجريدة\s+الرسمية|official\s+gazette", flags=re.IGNORECASE), "Official Gazette"),
    (re.compile(r"فتوى|legal\s+opinion", flags=re.IGNORECASE), "Legal Opinion"),
]

LISTING_PATH_HINTS = [
    "/category/",
    "/tag/",
    "/author/",
    "/page/",
    "/search/",
    "/feed/",
    "/wp-json/",
]

NON_DOCUMENT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".css",
    ".js",
    ".ico",
    ".xml",
    ".rss",
    ".zip",
)


class LegalCrossReference(BaseModel):
    """Candidate legal cross-reference discovered inside a legal document."""

    raw_text: str
    normalized_number: str
    document_type: str | None = None
    relation_hint: str = "REFERENCES"
    context: str | None = None


class DiscoveredLinks(BaseModel):
    """Links discovered on a page and classified for the crawler."""

    document_urls: list[str] = Field(default_factory=list)
    listing_urls: list[str] = Field(default_factory=list)
    pdf_urls: list[str] = Field(default_factory=list)
    language_urls: dict[str, str] = Field(default_factory=dict)

    def all_crawlable_urls(self, include_pdfs: bool = True) -> list[str]:
        """Return deduplicated URLs suitable for crawler queue insertion."""

        urls: list[str] = []
        urls.extend(self.document_urls)
        urls.extend(self.listing_urls)
        urls.extend(self.language_urls.values())

        if include_pdfs:
            urls.extend(self.pdf_urls)

        return sorted(dict.fromkeys(urls))


class ParsedLegalDocument(BaseModel):
    """Parsed legal document metadata and content block."""

    id: str
    title: str
    date: str | None = None
    document_type: str | None = None
    number: str | None = None
    issuer: str | None = None
    source_url: str
    language: str
    language_urls: dict[str, str] = Field(default_factory=dict)
    pdf_urls: list[str] = Field(default_factory=list)
    content_html: str | None = None
    content_text: str | None = None
    content_blocks: dict[str, str] = Field(default_factory=dict)
    cross_references: list[LegalCrossReference] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parsed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class QanoonParser:
    """Robust parser for qanoon.om listing and document pages.

    The website structure is treated as WordPress-like but not guaranteed stable.
    Selectors are intentionally defensive and are concentrated in a few methods
    so future changes can be patched without rewriting crawler logic.
    """

    def __init__(
        self,
        qanoon_base_url: str = "https://qanoon.om/",
        decree_base_url: str = "https://decree.om/",
    ):
        self.qanoon_base_url = self.normalize_url(qanoon_base_url)
        self.decree_base_url = self.normalize_url(decree_base_url)

        self.qanoon_host = urlparse(self.qanoon_base_url).netloc
        self.decree_host = urlparse(self.decree_base_url).netloc

        self.allowed_document_hosts = {
            self.qanoon_host,
            self.decree_host,
        }
        self.allowed_pdf_hosts = {
            self.qanoon_host,
            self.decree_host,
            "data.qanoon.om",
        }

    def parse_document(
        self,
        html: str,
        source_url: str,
        fallback_language: str | None = None,
    ) -> ParsedLegalDocument:
        """Parse a document-like HTML page into canonical metadata.

        Parameters
        ----------
        html:
            Raw HTML body.
        source_url:
            Original or final response URL.
        fallback_language:
            Optional language override from the crawler. Supported values are
            commonly `ar` and `en`.
        """

        normalized_url = self.normalize_url(source_url)
        soup = BeautifulSoup(html or "", "lxml")

        title = self.extract_title(soup) or self.title_from_url(normalized_url)
        page_text = self.clean_text(soup.get_text(" ", strip=True))

        language = fallback_language or self.detect_language(soup, normalized_url, page_text)
        discovered = self.discover_links(html, normalized_url)

        document_type = self.extract_document_type(soup, title, page_text)
        number = self.extract_document_number(f"{title} {page_text[:1500]}", document_type=document_type)
        issuer = self.extract_issuer(soup, title, page_text)
        date = self.extract_date(soup, page_text)

        content_blocks = self.extract_content_blocks(soup, language)
        content_html = content_blocks.get(language) or self.select_main_content_html(soup)
        content_text = self.clean_text(BeautifulSoup(content_html or "", "lxml").get_text(" ", strip=True))

        cross_references = self.extract_cross_references(page_text)
        document_id = self.build_document_id(
            source_url=normalized_url,
            document_type=document_type,
            number=number,
            title=title,
        )

        language_urls = dict(discovered.language_urls)
        language_urls.setdefault(language, normalized_url)

        parsed = ParsedLegalDocument(
            id=document_id,
            title=title,
            date=date,
            document_type=document_type,
            number=number,
            issuer=issuer,
            source_url=normalized_url,
            language=language,
            language_urls=language_urls,
            pdf_urls=discovered.pdf_urls,
            content_html=content_html,
            content_text=content_text,
            content_blocks=content_blocks,
            cross_references=cross_references,
            metadata={
                "parser": self.__class__.__name__,
                "has_document_signals": bool(document_type or number),
                "text_length": len(content_text or ""),
            },
        )

        logger.debug(
            "Parsed document id={} language={} type={} number={} url={}",
            parsed.id,
            parsed.language,
            parsed.document_type,
            parsed.number,
            parsed.source_url,
        )
        return parsed

    def parse_pdf_metadata(
        self,
        source_url: str,
        title: str | None = None,
        fallback_language: str | None = None,
    ) -> ParsedLegalDocument:
        """Create minimal metadata for a PDF-only source."""

        normalized_url = self.normalize_url(source_url)
        inferred_title = title or self.title_from_url(normalized_url)
        language = fallback_language or self.detect_language_from_url(normalized_url)

        document_type = self.extract_document_type(None, inferred_title, inferred_title)
        number = self.extract_document_number(inferred_title, document_type=document_type)

        return ParsedLegalDocument(
            id=self.build_document_id(
                source_url=normalized_url,
                document_type=document_type,
                number=number,
                title=inferred_title,
            ),
            title=inferred_title,
            document_type=document_type,
            number=number,
            source_url=normalized_url,
            language=language,
            pdf_urls=[normalized_url],
            metadata={
                "parser": self.__class__.__name__,
                "source_format": "pdf",
                "has_document_signals": bool(document_type or number),
            },
        )

    def discover_links(self, html: str, base_url: str) -> DiscoveredLinks:
        """Discover and classify document, listing, PDF, and language links."""

        soup = BeautifulSoup(html or "", "lxml")
        base_url = self.normalize_url(base_url)

        document_urls: set[str] = set()
        listing_urls: set[str] = set()
        pdf_urls: set[str] = set()
        language_urls: dict[str, str] = {}

        for anchor in soup.find_all("a", href=True):
            raw_href = anchor.get("href")
            if not raw_href:
                continue

            absolute_url = self.normalize_url(urljoin(base_url, raw_href))
            anchor_text = self.clean_text(anchor.get_text(" ", strip=True)).lower()

            if self.is_pdf_url(absolute_url):
                pdf_urls.add(absolute_url)
                continue

            detected_language = self.detect_language_link(anchor_text, absolute_url)
            if detected_language:
                language_urls[detected_language] = absolute_url

            if self.is_listing_url(absolute_url):
                listing_urls.add(absolute_url)
            elif self.is_document_url(absolute_url):
                document_urls.add(absolute_url)

        return DiscoveredLinks(
            document_urls=sorted(document_urls),
            listing_urls=sorted(listing_urls),
            pdf_urls=sorted(pdf_urls),
            language_urls=dict(sorted(language_urls.items())),
        )

    def extract_title(self, soup: BeautifulSoup) -> str | None:
        """Extract a legal page title from common semantic and WordPress selectors."""

        # TODO:
        # Add any future qanoon.om theme-specific title selector here. Current
        # selectors intentionally cover semantic HTML and WordPress defaults.
        selectors = [
            "h1.entry-title",
            "article h1",
            "main h1",
            "h1",
            "h2.entry-title",
            "article h2",
            "main h2",
            "h2",
        ]

        for selector in selectors:
            tag = soup.select_one(selector)
            if tag:
                text = self.clean_text(tag.get_text(" ", strip=True))
                if text and text.lower() not in {"search", "menu", "browse"}:
                    return text

        for attr_name, attr_value in [
            ("property", "og:title"),
            ("name", "twitter:title"),
            ("name", "title"),
        ]:
            meta = soup.find("meta", attrs={attr_name: attr_value})
            if meta and meta.get("content"):
                text = self.clean_text(meta["content"])
                if text:
                    return text

        title_tag = soup.find("title")
        if title_tag:
            text = self.clean_text(title_tag.get_text(" ", strip=True))
            if text:
                return text

        return None

    def extract_date(self, soup: BeautifulSoup, page_text: str) -> str | None:
        """Extract publication or issuance date from HTML/time/text."""

        for time_tag in soup.find_all("time"):
            datetime_value = time_tag.get("datetime")
            if datetime_value:
                return self.clean_text(datetime_value)

            text = self.clean_text(time_tag.get_text(" ", strip=True))
            if text:
                return text

        for attr_name, attr_value in [
            ("property", "article:published_time"),
            ("name", "date"),
            ("name", "pubdate"),
            ("itemprop", "datePublished"),
        ]:
            meta = soup.find("meta", attrs={attr_name: attr_value})
            if meta and meta.get("content"):
                return self.clean_text(meta["content"])

        normalized = self.normalize_digits(page_text)

        date_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
            r"\b\d{1,2}\s+من\s+\S+\s+سنة\s+\d{4}\s*هـ?\b",
            r"\b\d{1,2}\s+من\s+\S+\s+سنة\s+\d{4}\s*م\b",
            r"\b\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}\b",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return self.clean_text(match.group(0))

        return None

    def extract_document_type(
        self,
        soup: BeautifulSoup | None,
        title: str,
        page_text: str,
    ) -> str | None:
        """Infer document type from title, categories, and early body text."""

        category_text = ""
        if soup is not None:
            category_candidates = soup.select("a[rel='category tag'], .cat-links a, .entry-categories a")
            category_text = " ".join(tag.get_text(" ", strip=True) for tag in category_candidates)

        haystack = f"{title} {category_text} {page_text[:2000]}"

        for pattern, label in DOCUMENT_TYPE_PATTERNS:
            if pattern.search(haystack):
                return label

        return None

    def extract_document_number(self, text: str, document_type: str | None = None) -> str | None:
        """Extract legal document number such as `50/2026` or gazette issue number."""

        normalized = self.normalize_digits(text)

        slash_patterns = [
            r"(?:رقم|number|no\.?)?\s*(\d{1,4})\s*/\s*(\d{2,4})",
            r"(?:royal\s+decree|ministerial\s+decision|decision|decree)\s+(\d{1,4})\s+of\s+(\d{4})",
        ]

        for pattern in slash_patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return f"{match.group(1)}/{match.group(2)}"

        if document_type == "Official Gazette":
            issue_match = re.search(r"(?:العدد|issue)\s+(\d{1,5})", normalized, flags=re.IGNORECASE)
            if issue_match:
                return issue_match.group(1)

        return None

    def extract_issuer(self, soup: BeautifulSoup, title: str, page_text: str) -> str | None:
        """Extract likely legal issuer from title, tags, or signature lines."""

        title = self.clean_text(title)

        if ":" in title:
            prefix = self.clean_text(title.split(":", 1)[0])
            if 3 <= len(prefix) <= 140:
                return prefix

        if "royal decree" in title.lower() or "مرسوم سلطاني" in title:
            return "Sultan of Oman"

        tag_texts = [
            self.clean_text(tag.get_text(" ", strip=True))
            for tag in soup.select("a[rel='tag'], .tags-links a, .entry-tags a")
        ]

        for tag_text in tag_texts:
            if self._looks_like_issuer(tag_text):
                return tag_text

        lines = [self.clean_text(line) for line in page_text.splitlines() if self.clean_text(line)]
        if not lines:
            lines = re.split(r"(?<=[.؟!])\s+", page_text)

        for line in reversed(lines[-50:]):
            if self._looks_like_issuer(line):
                return self.clean_text(line)

        return None

    def extract_content_blocks(self, soup: BeautifulSoup, fallback_language: str) -> dict[str, str]:
        """Extract possible language-specific content blocks.

        The current assignment collapses translations into properties on one
        Document node. This method returns detected content blocks keyed by
        language. Most pages contain one language, but the structure supports
        pages that render both Arabic and English blocks.

        TODO:
            Add exact qanoon.om/decree.om selectors here if future site pages
            expose stable language-specific containers.
        """

        blocks: dict[str, str] = {}

        language_selectors = {
            "ar": [
                "[lang='ar']",
                "[lang='ar-OM']",
                "[dir='rtl']",
                ".arabic",
                ".lang-ar",
                ".content-ar",
                ".ar",
            ],
            "en": [
                "[lang='en']",
                "[lang='en-US']",
                "[dir='ltr']",
                ".english",
                ".lang-en",
                ".content-en",
                ".en",
            ],
        }

        for language, selectors in language_selectors.items():
            for selector in selectors:
                candidate = soup.select_one(selector)
                if candidate and self.clean_text(candidate.get_text(" ", strip=True)):
                    blocks[language] = str(candidate)
                    break

        main_html = self.select_main_content_html(soup)
        blocks.setdefault(fallback_language, main_html)

        return blocks

    def select_main_content_html(self, soup: BeautifulSoup) -> str:
        """Select the most likely legal content HTML block."""

        # TODO:
        # If a future qanoon.om theme introduces a specific legal body class,
        # add it at the top of this selector list.
        selectors = [
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

        for selector in selectors:
            candidate = soup.select_one(selector)
            if candidate and self.clean_text(candidate.get_text(" ", strip=True)):
                return str(candidate)

        return str(soup.body or soup)

    def extract_cross_references(self, text: str) -> list[LegalCrossReference]:
        """Extract candidate legal cross-references from Arabic/English text."""

        normalized = self.normalize_digits(text)
        sentences = self._split_sentences(normalized)

        references: dict[tuple[str, str, str | None], LegalCrossReference] = {}

        type_hint_pattern = (
            r"(مرسوم\s+سلطاني|قرار\s+وزاري|قانون|royal\s+decree|ministerial\s+decision|law|decree)"
        )
        number_pattern = r"(?:رقم|number|no\.?)?\s*(\d{1,4})\s*/\s*(\d{2,4})"

        combined_pattern = re.compile(
            rf"({type_hint_pattern}[^.؛\n]{{0,120}}?{number_pattern})",
            flags=re.IGNORECASE,
        )

        for sentence in sentences:
            relation_hint = self._relation_hint(sentence)

            for match in combined_pattern.finditer(sentence):
                raw = self.clean_text(match.group(1))
                number = f"{match.group(3)}/{match.group(4)}"
                document_type = self._normalize_reference_type(match.group(2))
                key = (number, relation_hint, document_type)

                references[key] = LegalCrossReference(
                    raw_text=raw,
                    normalized_number=number,
                    document_type=document_type,
                    relation_hint=relation_hint,
                    context=self.clean_text(sentence[:500]),
                )

            for match in re.finditer(number_pattern, sentence, flags=re.IGNORECASE):
                number = f"{match.group(1)}/{match.group(2)}"
                key = (number, relation_hint, None)

                references.setdefault(
                    key,
                    LegalCrossReference(
                        raw_text=self.clean_text(match.group(0)),
                        normalized_number=number,
                        document_type=None,
                        relation_hint=relation_hint,
                        context=self.clean_text(sentence[:500]),
                    ),
                )

        return list(references.values())

    def detect_language(self, soup: BeautifulSoup, source_url: str, page_text: str) -> str:
        """Detect source language from URL host, html attributes, and text."""

        html_tag = soup.find("html")
        if html_tag:
            lang_attr = self.clean_text(str(html_tag.get("lang", ""))).lower()
            if lang_attr.startswith("ar"):
                return "ar"
            if lang_attr.startswith("en"):
                return "en"

        return self.detect_language_from_url(source_url) or self.detect_language_from_text(page_text)

    def detect_language_from_url(self, source_url: str) -> str:
        """Detect language from known hostnames."""

        host = urlparse(source_url).netloc.lower()

        if host == self.decree_host or host.endswith(".decree.om"):
            return "en"

        if host == self.qanoon_host or host.endswith(".qanoon.om"):
            return "ar"

        return "ar"

    def detect_language_from_text(self, text: str) -> str:
        """Detect language by simple Arabic character ratio."""

        if not text:
            return "ar"

        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
        latin_chars = len(re.findall(r"[A-Za-z]", text))

        return "ar" if arabic_chars >= latin_chars else "en"

    def detect_language_link(self, anchor_text: str, url: str) -> str | None:
        """Detect whether an anchor points to another language version."""

        host = urlparse(url).netloc.lower()
        text = anchor_text.lower()

        if "english" in text or host == self.decree_host or host.endswith(".decree.om"):
            return "en"

        if "arabic" in text or "العربية" in text:
            return "ar"

        return None

    def is_document_url(self, url: str) -> bool:
        """Return True when a URL looks like a document detail page."""

        normalized = self.normalize_url(url)
        parsed = urlparse(normalized)
        host = parsed.netloc.lower()
        path = parsed.path.lower()

        if host not in self.allowed_document_hosts:
            return False

        if self.is_pdf_url(normalized):
            return False

        if path in {"", "/", "/p/"}:
            return False

        if path.endswith(NON_DOCUMENT_EXTENSIONS):
            return False

        if any(hint in path for hint in LISTING_PATH_HINTS):
            return False

        if "wp-content" in path or "wp-admin" in path or "wp-includes" in path:
            return False

        return True

    def is_listing_url(self, url: str) -> bool:
        """Return True when a URL looks like a listing/category/page URL."""

        normalized = self.normalize_url(url)
        parsed = urlparse(normalized)
        host = parsed.netloc.lower()
        path = parsed.path.lower()

        if host not in self.allowed_document_hosts:
            return False

        if path in {"", "/", "/p/"}:
            return True

        return any(hint in path for hint in LISTING_PATH_HINTS)

    def is_pdf_url(self, url: str) -> bool:
        """Return True when a URL points to a likely PDF/download resource."""

        normalized = self.normalize_url(url)
        parsed = urlparse(normalized)
        host = parsed.netloc.lower()
        path = parsed.path.lower()

        if host not in self.allowed_pdf_hosts:
            return False

        return path.endswith(".pdf") or host == "data.qanoon.om" or "/pdf/" in path

    def is_allowed_crawl_url(self, url: str) -> bool:
        """Return True if the crawler should be allowed to request the URL."""

        parsed = urlparse(self.normalize_url(url))
        if parsed.scheme not in {"http", "https"}:
            return False

        return parsed.netloc.lower() in (self.allowed_document_hosts | self.allowed_pdf_hosts)

    def build_document_id(
        self,
        source_url: str,
        document_type: str | None,
        number: str | None,
        title: str,
    ) -> str:
        """Build a stable document ID.

        If a legal type and number are available, Arabic and English versions can
        collapse to the same document ID. Otherwise, the canonical URL is used.
        """

        if document_type and number:
            basis = f"{document_type}:{number}".lower()
        else:
            basis = f"{self.normalize_url(source_url)}:{title}".lower()

        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
        return f"doc-{digest}"

    def title_from_url(self, url: str) -> str:
        """Create a readable fallback title from a URL slug."""

        parsed = urlparse(url)
        slug = parsed.path.rstrip("/").split("/")[-1] or parsed.netloc
        slug = re.sub(r"[-_]+", " ", slug).strip()

        return slug or url

    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and tracking query parameters."""

        if not url:
            return url

        parsed = urlparse(url.strip())
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc.lower()

        path = re.sub(r"/{2,}", "/", parsed.path or "/")

        if path != "/" and not re.search(r"\.[A-Za-z0-9]{2,6}$", path):
            path = path.rstrip("/") + "/"

        query_pairs = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith(("utm_", "fbclid", "gclid"))
        ]
        query = urlencode(query_pairs)

        return urlunparse((scheme, netloc, path, "", query, ""))

    def normalize_digits(self, text: str) -> str:
        """Convert Arabic-Indic digits to ASCII digits."""

        return (text or "").translate(ARABIC_DIGIT_TRANSLATION)

    def clean_text(self, text: str) -> str:
        """Normalize whitespace while preserving Arabic text order."""

        cleaned = (text or "").replace("\u00a0", " ")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
        return cleaned.strip()

    def _looks_like_issuer(self, text: str) -> bool:
        """Heuristic issuer detector for Arabic and English legal signatures."""

        text = self.clean_text(text)

        if not (3 <= len(text) <= 180):
            return False

        issuer_markers = [
            "وزارة",
            "وزير",
            "سلطان عمان",
            "هيئة",
            "شرطة",
            "البنك المركزي",
            "Ministry",
            "Minister",
            "Sultan of Oman",
            "Authority",
            "Central Bank",
            "Royal Oman Police",
        ]

        return any(marker.lower() in text.lower() for marker in issuer_markers)

    def _split_sentences(self, text: str) -> list[str]:
        """Split Arabic/English legal text into coarse sentence-like units."""

        return [
            self.clean_text(part)
            for part in re.split(r"[\n.;؛]+", text or "")
            if self.clean_text(part)
        ]

    def _relation_hint(self, sentence: str) -> str:
      """Infer whether a sentence amends, repeals, or merely references another law."""

      if re.search(
          r"\b(amend|amends|amending|amended|modify|modifies|modifying|modified)\b|"
          r"تعديل|يعدل|تعدل|تستبدل",
          sentence,
          re.IGNORECASE,
      ):
          return "AMENDS"

      if re.search(
          r"\b(repeal|repeals|repealed|cancel|cancels|abolish|abolishes|abolished)\b|"
          r"يلغى|تلغى|إلغاء|إلغاؤه",
          sentence,
          re.IGNORECASE,
      ):
          return "REPEALS"

      return "REFERENCES"

    def _normalize_reference_type(self, raw_type: str | None) -> str | None:
        """Normalize legal type text found near a cross-reference."""

        if not raw_type:
            return None

        for pattern, label in DOCUMENT_TYPE_PATTERNS:
            if pattern.search(raw_type):
                return label

        if re.search(r"قانون|law", raw_type, flags=re.IGNORECASE):
            return "Law"

        return self.clean_text(raw_type)