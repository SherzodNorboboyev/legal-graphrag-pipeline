"""Topic extraction for Legal GraphRAG documents.

The extractor uses OpenAI only when explicitly configured. Otherwise it uses a
deterministic keyword fallback that works offline and is suitable for tests,
demos, and low-cost ingestion runs.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from src.config import Settings, get_settings
from src.llm_agents.prompts import (
    TOPIC_EXTRACTION_SYSTEM_PROMPT,
    build_json_repair_prompt,
    build_topic_extraction_prompt,
)
from src.models import LegalDocument, Topic, normalize_topic_name, normalize_whitespace


class TopicExtractionError(RuntimeError):
    """Raised when topic extraction fails unexpectedly."""


class TopicExtractionResult(BaseModel):
    """Validated topic extraction result."""

    topics: list[Topic] = Field(default_factory=list)


@dataclass(frozen=True)
class KeywordRule:
    """One deterministic fallback keyword rule."""

    pattern: re.Pattern[str]
    topic_name: str
    confidence: float
    evidence_label: str


KEYWORD_RULES: list[KeywordRule] = [
    KeywordRule(re.compile(r"\b(tax|taxation|taxable|vat|excise)\b|ضريبة|ضرائب|الدخل", re.I), "Taxation", 0.78, "taxation terms"),
    KeywordRule(re.compile(r"\b(customs|tariff|import|export)\b|جمارك|تعرفة|استيراد|تصدير", re.I), "Customs", 0.74, "customs terms"),
    KeywordRule(re.compile(r"\b(labou?r|employment|worker|wage|salary)\b|العمل|عامل|أجور|راتب", re.I), "Labour Regulation", 0.76, "labour terms"),
    KeywordRule(re.compile(r"\b(omanization|national workforce)\b|التعمين|القوى العاملة الوطنية", re.I), "Omanization", 0.80, "omanization terms"),
    KeywordRule(re.compile(r"\b(education|school|university|student)\b|تعليم|مدرسة|جامعة|طالب", re.I), "Education", 0.72, "education terms"),
    KeywordRule(re.compile(r"\b(health|medical|hospital|medicine|pharmaceutical)\b|صحة|طبي|مستشفى|دواء", re.I), "Public Health", 0.72, "health terms"),
    KeywordRule(re.compile(r"\b(environment|pollution|waste|climate)\b|البيئة|تلوث|نفايات|مناخ", re.I), "Environmental Protection", 0.73, "environment terms"),
    KeywordRule(re.compile(r"\b(real estate|land|property|registry)\b|عقار|أراض|الأراضي|السجل العقاري", re.I), "Real Estate", 0.73, "real estate terms"),
    KeywordRule(re.compile(r"\b(maritime|ship|vessel|port|seafarer)\b|بحري|سفينة|ميناء|موانئ", re.I), "Maritime Law", 0.73, "maritime terms"),
    KeywordRule(re.compile(r"\b(bank|banking|central bank|finance|financial)\b|بنك|مصرف|المركزي|مالي", re.I), "Banking Regulation", 0.74, "banking terms"),
    KeywordRule(re.compile(r"\b(competition|monopoly|antitrust)\b|منافسة|احتكار", re.I), "Competition Law", 0.76, "competition terms"),
    KeywordRule(re.compile(r"\b(judicial|court|judge|prosecution|litigation)\b|قضاء|محكمة|قاض|الادعاء|دعوى", re.I), "Judicial Administration", 0.71, "judicial terms"),
    KeywordRule(re.compile(r"\b(fee|fees|charge|charges)\b|رسوم|رسم", re.I), "Fees", 0.68, "fees terms"),
    KeywordRule(re.compile(r"\b(investment|investor|capital market)\b|استثمار|مستثمر|سوق المال", re.I), "Investment", 0.70, "investment terms"),
    KeywordRule(re.compile(r"\b(transport|traffic|road|vehicle)\b|نقل|مرور|طريق|مركبة", re.I), "Transport", 0.69, "transport terms"),
    KeywordRule(re.compile(r"\b(criminal|penal|crime|procedure)\b|جزائي|جنائي|جريمة|الإجراءات الجزائية", re.I), "Criminal Procedure", 0.72, "criminal procedure terms"),
    KeywordRule(re.compile(r"\b(public procurement|tender|tenders|contract)\b|مناقصات|المشتريات|عقد", re.I), "Public Procurement", 0.70, "procurement terms"),
    KeywordRule(re.compile(r"\b(data protection|privacy|personal data)\b|حماية البيانات|الخصوصية|البيانات الشخصية", re.I), "Data Protection", 0.70, "data protection terms"),
]


TITLE_STOPWORDS = {
    "royal",
    "decree",
    "ministerial",
    "decision",
    "law",
    "laws",
    "issuing",
    "promulgating",
    "regarding",
    "amending",
    "oman",
    "sultanate",
    "number",
    "no",
    "article",
    "مرسوم",
    "سلطاني",
    "قرار",
    "وزاري",
    "قانون",
    "إصدار",
    "بإصدار",
    "تعديل",
    "شأن",
    "رقم",
    "المادة",
}


class TopicExtractor:
    """Extract legal topics from consolidated document Markdown.

    Extraction order:
    1. Use `contentEn` if available.
    2. Otherwise use `contentAr`.
    3. Use OpenAI if configured.
    4. Fall back to deterministic local keyword extraction.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        max_topics: int = 12,
        min_confidence: float = 0.25,
        max_content_chars: int = 12000,
    ):
        self.settings = settings or get_settings()
        self.max_topics = max_topics
        self.min_confidence = min_confidence
        self.max_content_chars = max_content_chars
        self._openai_client: Any | None = None

    def extract_for_document(self, document: LegalDocument) -> list[Topic]:
        """Extract topics from a `LegalDocument` model."""

        return self.extract(
            title=document.title,
            content_en=document.content_en,
            content_ar=document.content_ar,
            max_topics=self.max_topics,
            min_confidence=self.min_confidence,
        )

    def extract(
        self,
        *,
        title: str,
        content_en: str | None = None,
        content_ar: str | None = None,
        max_topics: int | None = None,
        min_confidence: float | None = None,
    ) -> list[Topic]:
        """Extract topics from English content first, then Arabic content.

        Parameters
        ----------
        title:
            Document title.
        content_en:
            English Markdown content if available.
        content_ar:
            Arabic Markdown content if available.
        max_topics:
            Optional override for max number of topics.
        min_confidence:
            Optional override for minimum accepted confidence.
        """

        effective_max_topics = max_topics or self.max_topics
        effective_min_confidence = self.min_confidence if min_confidence is None else min_confidence

        content, language = self.select_content(content_en=content_en, content_ar=content_ar)

        if not content:
            logger.warning("Topic extraction skipped for '{}': no content available.", title)
            return []

        if self.should_use_openai():
            try:
                topics = self.extract_with_openai(
                    title=title,
                    content=content,
                    language=language,
                    max_topics=effective_max_topics,
                    min_confidence=effective_min_confidence,
                )
                if topics:
                    return topics
            except Exception as exc:  # noqa: BLE001 - required fallback path
                logger.exception("OpenAI topic extraction failed for '{}'. Falling back. Error: {}", title, exc)

        return self.extract_with_keywords(
            title=title,
            content=content,
            language=language,
            max_topics=effective_max_topics,
            min_confidence=effective_min_confidence,
        )

    def select_content(self, *, content_en: str | None, content_ar: str | None) -> tuple[str, str]:
        """Return preferred content and language.

        English is preferred because topic normalization is usually more stable.
        Arabic remains fully supported when English is unavailable.
        """

        if content_en and content_en.strip():
            return content_en.strip(), "en"

        if content_ar and content_ar.strip():
            return content_ar.strip(), "ar"

        return "", "unknown"

    def should_use_openai(self) -> bool:
        """Return True when OpenAI topic extraction is configured."""

        provider = getattr(self.settings, "topic_llm_provider", "fallback")
        api_key = getattr(self.settings, "openai_api_key", None)
        return provider == "openai" and bool(api_key)

    def extract_with_openai(
        self,
        *,
        title: str,
        content: str,
        language: str,
        max_topics: int,
        min_confidence: float,
    ) -> list[Topic]:
        """Extract topics using OpenAI Chat Completions with JSON retry logic."""

        from openai import OpenAI

        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)

        prompt = build_topic_extraction_prompt(
            title=title,
            content=content,
            language=language,
            max_topics=max_topics,
            max_content_chars=self.max_content_chars,
        )

        last_response_text = ""
        attempts = int(getattr(self.settings, "topic_extraction_max_retries", 2)) + 1

        for attempt in range(attempts):
            user_prompt = prompt if attempt == 0 else build_json_repair_prompt(last_response_text)

            response = self._openai_client.chat.completions.create(
                model=getattr(self.settings, "openai_chat_model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": TOPIC_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            last_response_text = response.choices[0].message.content or ""
            topics = self.safe_parse_topics(
                last_response_text,
                source="openai",
                max_topics=max_topics,
                min_confidence=min_confidence,
            )

            if topics:
                return topics

            logger.warning("OpenAI returned invalid or empty topic JSON for '{}', attempt {}/{}.", title, attempt + 1, attempts)
            time.sleep(min(2**attempt, 5))

        logger.warning("OpenAI topic extraction exhausted retries for '{}'.", title)
        return []

    def extract_with_keywords(
        self,
        *,
        title: str,
        content: str,
        language: str,
        max_topics: int,
        min_confidence: float,
    ) -> list[Topic]:
        """Extract topics with deterministic keyword rules.

        This fallback is intentionally transparent. It favors common legal
        domains and adds title terms only when dictionary matches are sparse.
        """

        haystack = f"{title}\n{content[: self.max_content_chars]}"
        topics: list[Topic] = []

        for rule in KEYWORD_RULES:
            match = rule.pattern.search(haystack)
            if not match:
                continue

            evidence = self._short_evidence(haystack, match.start(), match.end(), fallback=rule.evidence_label)
            topics.append(
                self._build_topic(
                    name=rule.topic_name,
                    normalized_name=normalize_topic_name(rule.topic_name),
                    confidence=rule.confidence,
                    evidence=evidence,
                    source=f"fallback_keyword:{language}",
                )
            )

        if len(topics) < min(3, max_topics):
            topics.extend(
                self.extract_title_term_topics(
                    title=title,
                    existing_topics=topics,
                    max_additional=max_topics - len(topics),
                    language=language,
                )
            )

        return self.filter_and_deduplicate_topics(
            topics,
            max_topics=max_topics,
            min_confidence=min_confidence,
        )

    def extract_title_term_topics(
        self,
        *,
        title: str,
        existing_topics: list[Topic],
        max_additional: int,
        language: str,
    ) -> list[Topic]:
        """Extract fallback topics from high-signal title terms."""

        if max_additional <= 0:
            return []

        existing_names = {topic.normalized_name for topic in existing_topics}
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}|[\u0600-\u06FF]{4,}", title or "")
        normalized_words = [
            word.strip("-_").lower()
            for word in words
            if word.strip("-_").lower() not in TITLE_STOPWORDS
        ]

        counts = Counter(normalized_words)
        topics: list[Topic] = []

        for word, _ in counts.most_common(max_additional * 2):
            if len(topics) >= max_additional:
                break

            display_name = word.title() if re.match(r"[A-Za-z]", word) else word
            normalized = normalize_topic_name(display_name)

            if not normalized or normalized in existing_names:
                continue

            topics.append(
                self._build_topic(
                    name=display_name,
                    normalized_name=normalized,
                    confidence=0.42,
                    evidence="title term",
                    source=f"fallback_title_terms:{language}",
                )
            )
            existing_names.add(normalized)

        return topics

    def safe_parse_topics(
        self,
        raw_text: str,
        *,
        source: str = "llm",
        max_topics: int | None = None,
        min_confidence: float | None = None,
    ) -> list[Topic]:
        """Safely parse topic JSON returned by an LLM.

        The method accepts:
        - a clean JSON object with `topics`
        - a raw JSON list
        - JSON wrapped inside markdown fences
        - text with a JSON object embedded in it

        Invalid JSON returns an empty list instead of raising.
        """

        if not raw_text or not raw_text.strip():
            return []

        effective_max_topics = max_topics or self.max_topics
        effective_min_confidence = self.min_confidence if min_confidence is None else min_confidence

        for candidate in self.json_candidates(raw_text):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            raw_topics: list[Any]

            if isinstance(payload, dict):
                if "topics" in payload and isinstance(payload["topics"], list):
                    raw_topics = payload["topics"]
                elif {"name", "confidence"}.issubset(set(payload.keys())):
                    raw_topics = [payload]
                else:
                    continue
            elif isinstance(payload, list):
                raw_topics = payload
            else:
                continue

            topics: list[Topic] = []

            for item in raw_topics:
                if not isinstance(item, dict):
                    continue

                topic = self.topic_from_mapping(item, source=source)
                if topic is not None:
                    topics.append(topic)

            return self.filter_and_deduplicate_topics(
                topics,
                max_topics=effective_max_topics,
                min_confidence=effective_min_confidence,
            )

        return []

    def json_candidates(self, raw_text: str) -> list[str]:
        """Return possible JSON substrings from noisy model output."""

        text = raw_text.strip()
        candidates = [text]

        fenced_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        candidates.extend(match.strip() for match in fenced_matches if match.strip())

        object_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if object_match:
            candidates.append(object_match.group(1).strip())

        array_match = re.search(r"(\[.*\])", text, flags=re.DOTALL)
        if array_match:
            candidates.append(array_match.group(1).strip())

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(candidates))

    def topic_from_mapping(self, payload: dict[str, Any], *, source: str) -> Topic | None:
        """Convert one JSON object into the canonical Pydantic Topic model."""

        name = normalize_whitespace(str(payload.get("name") or "")) or ""
        normalized_name = normalize_whitespace(str(payload.get("normalized_name") or "")) or None

        if not name and normalized_name:
            name = normalized_name

        if not name:
            return None

        try:
            confidence = float(payload.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        evidence = normalize_whitespace(str(payload.get("evidence") or "")) or ""

        try:
            return Topic(
                name=name,
                normalized_name=normalized_name or normalize_topic_name(name),
                confidence=max(0.0, min(1.0, confidence)),
                evidence=evidence,
                source=payload.get("source") or source,
            )
        except ValidationError as exc:
            logger.debug("Skipping invalid topic payload {}: {}", payload, exc)
            return None

    def filter_and_deduplicate_topics(
        self,
        topics: list[Topic],
        *,
        max_topics: int,
        min_confidence: float,
    ) -> list[Topic]:
        """Filter by confidence and keep highest-confidence topic per normalized name."""

        deduped: dict[str, Topic] = {}

        for topic in topics:
            if topic.confidence < min_confidence:
                continue

            normalized_name = topic.normalized_name or normalize_topic_name(topic.name)
            if not normalized_name:
                continue

            existing = deduped.get(normalized_name)
            if existing is None or topic.confidence > existing.confidence:
                deduped[normalized_name] = topic

        return sorted(deduped.values(), key=lambda item: item.confidence, reverse=True)[:max_topics]

    def _build_topic(
        self,
        *,
        name: str,
        normalized_name: str,
        confidence: float,
        evidence: str,
        source: str,
    ) -> Topic:
        """Build a validated Topic object for fallback extraction."""

        return Topic(
            name=name,
            normalized_name=normalized_name,
            confidence=confidence,
            evidence=evidence,
            source=source,
        )

    def _short_evidence(self, text: str, start: int, end: int, *, fallback: str) -> str:
        """Extract a compact evidence phrase around a keyword match."""

        left = max(0, start - 70)
        right = min(len(text), end + 70)
        evidence = normalize_whitespace(text[left:right]) or fallback
        return evidence[:220]