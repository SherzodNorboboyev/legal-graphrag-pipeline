"""Canonical Pydantic models for the Legal GraphRAG pipeline.

These models sit between scraping/parsing and graph ingestion. They normalize
input produced by the crawler, validate important fields, and generate stable
identifiers for graph nodes.

Design notes:
- A legal instrument is represented by one consolidated Document node.
- Arabic, English, and French Markdown are stored as contentAr/contentEn/contentFr.
- Translations are not represented as separate nodes.
- Chunk and Topic nodes are separate because they are vector-searchable units.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


LanguageCode = Literal["ar", "en", "fr", "unknown"]
LegalRelationshipType = Literal["AMENDS", "REPEALS", "REFERENCES"]

ARABIC_DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def normalize_digits(value: str | None) -> str | None:
    """Convert Arabic-Indic digits into ASCII digits."""

    if value is None:
        return None

    return value.translate(ARABIC_DIGIT_TRANSLATION)


def normalize_whitespace(value: str | None) -> str | None:
    """Normalize whitespace while preserving Arabic character order."""

    if value is None:
        return None

    value = value.replace("\u00a0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    return value.strip()


def normalize_document_number(value: str | None) -> str | None:
    """Normalize legal document numbers such as Arabic `٥٠ / ٢٠٢٦` to `50/2026`."""

    if value is None:
        return None

    value = normalize_digits(value) or ""
    value = normalize_whitespace(value) or ""
    value = re.sub(r"\s*/\s*", "/", value)

    if not value:
        return None

    return value


def normalize_identity_component(value: str | None) -> str:
    """Normalize a value for deterministic hashing."""

    if not value:
        return ""

    value = normalize_digits(value) or ""
    value = value.lower().strip()
    value = re.sub(r"[\u064B-\u065F]", "", value)
    value = re.sub(r"[^a-z0-9\u0600-\u06FF/]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def canonicalize_url(url: str | None) -> str:
    """Canonicalize a URL for stable hashing and provenance storage."""

    if not url:
        return ""

    parsed = urlparse(url.strip())
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = re.sub(r"/{2,}", "/", parsed.path or "/")

    if path != "/" and not re.search(r"\.[A-Za-z0-9]{2,8}$", path):
        path = path.rstrip("/") + "/"

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith(("utm_", "fbclid", "gclid"))
    ]
    query = urlencode(query_pairs)

    return urlunparse((scheme, netloc, path, "", query, ""))


def stable_hash(value: str, length: int = 16) -> str:
    """Return a deterministic SHA-1 hash prefix."""

    if not value:
        raise ValueError("Cannot hash an empty value.")

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def generate_document_id(
    *,
    source_url: str | None = None,
    title: str | None = None,
    document_type: str | None = None,
    number: str | None = None,
) -> str:
    """Generate a deterministic Document node ID.

    If `document_type` and `number` are available, the ID intentionally ignores
    source URL so Arabic and English pages for the same legal instrument collapse
    into one Document node. If legal number metadata is unavailable, the canonical
    source URL is used as the fallback identity.
    """

    normalized_number = normalize_document_number(number)
    normalized_type = normalize_identity_component(document_type)

    if normalized_type and normalized_number:
        basis = f"legal-document:{normalized_type}:{normalized_number}"
    elif source_url:
        basis = f"legal-document:url:{canonicalize_url(source_url)}"
    elif title:
        basis = f"legal-document:title:{normalize_identity_component(title)}"
    else:
        raise ValueError("At least one of source_url, title, or document_type+number is required.")

    return f"doc-{stable_hash(basis)}"


def normalize_topic_name(value: str | None) -> str:
    """Normalize a topic label for stable Topic node merging."""

    if not value:
        return ""

    value = normalize_digits(value) or ""
    value = value.lower().strip()
    value = re.sub(r"[\u064B-\u065F]", "", value)
    value = re.sub(r"[^\w\s\u0600-\u06FF]", " ", value)
    value = re.sub(r"_+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def generate_chunk_id(
    *,
    document_id: str,
    language: LanguageCode,
    chunk_index: int,
    text: str,
) -> str:
    """Generate a deterministic Chunk node ID."""

    if not document_id:
        raise ValueError("document_id is required to generate a chunk ID.")

    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative.")

    if not text or not text.strip():
        raise ValueError("text is required to generate a chunk ID.")

    basis = f"chunk:{document_id}:{language}:{chunk_index}:{stable_hash(text.strip(), length=20)}"
    return f"chunk-{stable_hash(basis, length=24)}"


def validate_embedding_values(values: list[float] | None) -> list[float] | None:
    """Validate and cast embedding vectors to finite floats."""

    if values is None:
        return None

    if not values:
        raise ValueError("embedding must not be an empty list when provided.")

    normalized_values: list[float] = []
    for value in values:
        float_value = float(value)

        if not math.isfinite(float_value):
            raise ValueError("embedding values must be finite numbers.")

        normalized_values.append(float_value)

    return normalized_values


class CrossReference(BaseModel):
    """Candidate legal cross-reference from one document to another.

    This model accepts both canonical field names and parser-produced names from
    Part 2. For example, parser output may use `relation_hint` and
    `normalized_number`; this model maps those into `relation_type` and
    `target_number`.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    relation_type: LegalRelationshipType = Field(
        default="REFERENCES",
        validation_alias=AliasChoices("relation_type", "relation_hint", "type"),
        description="Relationship hint: AMENDS, REPEALS, or REFERENCES.",
    )
    source_document_id: str | None = Field(default=None)
    target_document_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("target_document_id", "document_id", "target_id"),
    )
    target_number: str | None = Field(
        default=None,
        validation_alias=AliasChoices("target_number", "normalized_number", "number"),
    )
    target_document_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("target_document_type", "document_type"),
    )
    raw_text: str = Field(default="")
    context: str | None = None

    @field_validator("relation_type", mode="before")
    @classmethod
    def normalize_relation_type(cls, value: object) -> str:
        """Normalize relationship type to uppercase enum values."""

        if value is None:
            return "REFERENCES"

        normalized = str(value).upper().strip()

        if normalized not in {"AMENDS", "REPEALS", "REFERENCES"}:
            raise ValueError("relation_type must be one of AMENDS, REPEALS, REFERENCES.")

        return normalized

    @field_validator("target_number", mode="before")
    @classmethod
    def normalize_target_number(cls, value: object) -> str | None:
        """Normalize target legal number."""

        if value is None:
            return None

        return normalize_document_number(str(value))

    @model_validator(mode="after")
    def validate_reference_target(self) -> "CrossReference":
        """Ensure the reference contains at least some useful target signal."""

        has_target_signal = bool(self.target_document_id or self.target_number or self.raw_text)

        if not has_target_signal:
            raise ValueError("CrossReference requires target_document_id, target_number, or raw_text.")

        return self

    @property
    def is_document_level_relationship(self) -> bool:
        """Return True if this reference should become a Neo4j edge."""

        return self.relation_type in {"AMENDS", "REPEALS"}


class Topic(BaseModel):
    """Legal topic/entity node extracted from a document."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str = Field(..., min_length=1)
    normalized_name: str | None = Field(default=None)
    embedding: list[float] | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str | None = None
    source: str = Field(default="unknown")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate and normalize topic display name."""

        value = normalize_whitespace(value) or ""

        if not value:
            raise ValueError("Topic name cannot be empty.")

        return value

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding(cls, value: object) -> list[float] | None:
        """Validate embedding vector."""

        if value is None:
            return None

        if not isinstance(value, list):
            raise ValueError("embedding must be a list of floats.")

        return validate_embedding_values(value)

    @model_validator(mode="after")
    def fill_normalized_name(self) -> "Topic":
        """Generate normalized topic name if missing."""

        normalized = normalize_topic_name(self.normalized_name or self.name)

        if not normalized:
            raise ValueError("Topic normalized_name cannot be empty.")

        self.normalized_name = normalized
        return self


class Chunk(BaseModel):
    """Semantic text chunk node linked to a parent Document."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str | None = None
    document_id: str = Field(..., min_length=1)
    language: LanguageCode = Field(default="unknown")
    text: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    embedding: list[float] | None = None
    heading_context: list[str] = Field(default_factory=list)
    token_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Validate chunk text."""

        value = normalize_whitespace(value) or ""

        if not value:
            raise ValueError("Chunk text cannot be empty.")

        return value

    @field_validator("heading_context")
    @classmethod
    def normalize_heading_context(cls, value: list[str]) -> list[str]:
        """Normalize heading context values."""

        return [clean for item in value if (clean := normalize_whitespace(item))]

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding(cls, value: object) -> list[float] | None:
        """Validate embedding vector."""

        if value is None:
            return None

        if not isinstance(value, list):
            raise ValueError("embedding must be a list of floats.")

        return validate_embedding_values(value)

    @model_validator(mode="after")
    def fill_computed_fields(self) -> "Chunk":
        """Fill deterministic ID and approximate token count."""

        if not self.id:
            self.id = generate_chunk_id(
                document_id=self.document_id,
                language=self.language,
                chunk_index=self.chunk_index,
                text=self.text,
            )

        if self.token_count is None:
            self.token_count = approximate_token_count(self.text)

        return self


class LegalDocument(BaseModel):
    """Canonical consolidated legal document model.

    This model maps directly to the Neo4j Document node. Language-specific
    Markdown fields use Pythonic snake_case internally but serialize to the exact
    graph property names `contentAr`, `contentEn`, and `contentFr`.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str | None = None
    title: str = Field(..., min_length=1)
    date: str | None = None
    document_type: str | None = None
    number: str | None = None
    issuer: str | None = None
    source_url: str = Field(..., min_length=1)
    language: LanguageCode = Field(default="unknown")

    title_ar: str | None = None
    title_en: str | None = None
    title_fr: str | None = None

    content_ar: str | None = Field(default=None, alias="contentAr")
    content_en: str | None = Field(default=None, alias="contentEn")
    content_fr: str | None = Field(default=None, alias="contentFr")

    language_urls: dict[str, str] = Field(default_factory=dict)
    pdf_urls: list[str] = Field(default_factory=list)
    raw_paths: dict[str, str] = Field(default_factory=dict)
    cross_references: list[CrossReference] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        """Validate document title."""

        value = normalize_whitespace(value) or ""

        if not value:
            raise ValueError("LegalDocument title cannot be empty.")

        return value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        """Validate source URL and canonicalize it."""

        parsed = urlparse(value.strip())

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source_url must be an absolute HTTP(S) URL.")

        return canonicalize_url(value)

    @field_validator("number", mode="before")
    @classmethod
    def normalize_number(cls, value: object) -> str | None:
        """Normalize legal document number."""

        if value is None:
            return None

        return normalize_document_number(str(value))

    @field_validator("language_urls")
    @classmethod
    def normalize_language_urls(cls, value: dict[str, str]) -> dict[str, str]:
        """Canonicalize language URL mappings."""

        normalized: dict[str, str] = {}

        for language, url in value.items():
            language_key = language.strip().lower()

            if language_key not in {"ar", "en", "fr", "unknown"}:
                language_key = "unknown"

            normalized[language_key] = canonicalize_url(url)

        return normalized

    @field_validator("pdf_urls")
    @classmethod
    def normalize_pdf_urls(cls, value: list[str]) -> list[str]:
        """Canonicalize and deduplicate PDF URLs."""

        urls = [canonicalize_url(url) for url in value if url]
        return sorted(dict.fromkeys(urls))

    @model_validator(mode="after")
    def fill_document_id_and_titles(self) -> "LegalDocument":
        """Generate missing ID and fill language-specific title when possible."""

        if not self.id:
            self.id = generate_document_id(
                source_url=self.source_url,
                title=self.title,
                document_type=self.document_type,
                number=self.number,
            )

        if self.language == "ar" and not self.title_ar:
            self.title_ar = self.title
        elif self.language == "en" and not self.title_en:
            self.title_en = self.title
        elif self.language == "fr" and not self.title_fr:
            self.title_fr = self.title

        return self

    @property
    def best_content(self) -> str | None:
        """Return best available Markdown content for topic extraction."""

        return self.content_en or self.content_ar or self.content_fr

    def content_by_language(self) -> dict[str, str]:
        """Return available content keyed by language code."""

        result: dict[str, str] = {}

        if self.content_ar:
            result["ar"] = self.content_ar

        if self.content_en:
            result["en"] = self.content_en

        if self.content_fr:
            result["fr"] = self.content_fr

        return result


class ParsedDocument(BaseModel):
    """Intermediary parsed document model produced by the scraper/parser layer.

    This model is intentionally compatible with Part 2 parser output and can be
    converted into the canonical `LegalDocument` model.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str | None = None
    title: str = Field(..., min_length=1)
    date: str | None = None
    document_type: str | None = None
    number: str | None = None
    issuer: str | None = None
    source_url: str = Field(..., min_length=1)
    language: LanguageCode = Field(default="unknown")

    title_ar: str | None = None
    title_en: str | None = None
    title_fr: str | None = None

    markdown: str | None = None
    content_html: str | None = None
    content_text: str | None = None

    content_ar: str | None = Field(default=None, alias="contentAr")
    content_en: str | None = Field(default=None, alias="contentEn")
    content_fr: str | None = Field(default=None, alias="contentFr")

    language_urls: dict[str, str] = Field(default_factory=dict)
    pdf_urls: list[str] = Field(default_factory=list)
    raw_paths: dict[str, str] = Field(default_factory=dict)
    cross_references: list[CrossReference] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parsed_at: str = Field(default_factory=utc_now_iso)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        """Validate parsed document title."""

        value = normalize_whitespace(value) or ""

        if not value:
            raise ValueError("ParsedDocument title cannot be empty.")

        return value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        """Validate and canonicalize source URL."""

        parsed = urlparse(value.strip())

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source_url must be an absolute HTTP(S) URL.")

        return canonicalize_url(value)

    @field_validator("number", mode="before")
    @classmethod
    def normalize_number(cls, value: object) -> str | None:
        """Normalize parsed legal number."""

        if value is None:
            return None

        return normalize_document_number(str(value))

    @model_validator(mode="after")
    def fill_id(self) -> "ParsedDocument":
        """Generate deterministic parsed document ID if missing."""

        if not self.id:
            self.id = generate_document_id(
                source_url=self.source_url,
                title=self.title,
                document_type=self.document_type,
                number=self.number,
            )

        return self

    def to_legal_document(self) -> LegalDocument:
        """Convert parsed document into the canonical LegalDocument model."""

        content_ar = self.content_ar
        content_en = self.content_en
        content_fr = self.content_fr

        # If parser produced a single Markdown field, map it to the detected language.
        if self.markdown:
            if self.language == "ar" and not content_ar:
                content_ar = self.markdown
            elif self.language == "en" and not content_en:
                content_en = self.markdown
            elif self.language == "fr" and not content_fr:
                content_fr = self.markdown

        return LegalDocument(
            id=self.id,
            title=self.title,
            date=self.date,
            document_type=self.document_type,
            number=self.number,
            issuer=self.issuer,
            source_url=self.source_url,
            language=self.language,
            title_ar=self.title_ar,
            title_en=self.title_en,
            title_fr=self.title_fr,
            contentAr=content_ar,
            contentEn=content_en,
            contentFr=content_fr,
            language_urls=self.language_urls,
            pdf_urls=self.pdf_urls,
            raw_paths=self.raw_paths,
            cross_references=self.cross_references,
            metadata={
                **self.metadata,
                "parsed_at": self.parsed_at,
                "converted_from": "ParsedDocument",
            },
        )


def approximate_token_count(text: str) -> int:
    """Approximate token count for metadata and validation.

    This avoids requiring a tokenizer dependency inside the model layer.
    """

    return len(re.findall(r"[\w\u0600-\u06FF]+|[^\s]", text or ""))