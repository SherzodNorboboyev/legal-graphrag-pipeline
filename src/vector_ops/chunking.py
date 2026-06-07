"""Markdown-aware semantic chunking for Legal GraphRAG.

The chunker preserves heading context, supports token overlap, and returns
canonical `Chunk` Pydantic models ready for Neo4j ingestion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from src.models import Chunk, LanguageCode


TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06FF]+|[^\s]", re.UNICODE)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class MarkdownSection:
    """Internal representation of a Markdown section."""

    heading_context: list[str] = field(default_factory=list)
    text: str = ""
    token_count: int = 0


class MarkdownChunker:
    """Split Markdown into semantically coherent chunks.

    Defaults are chosen for the assignment target of 500-1000 tokens. Smaller
    values can be injected in tests.
    """

    def __init__(
        self,
        min_tokens: int = 500,
        max_tokens: int = 900,
        overlap_tokens: int = 120,
    ):
        if min_tokens <= 0:
            raise ValueError("min_tokens must be greater than zero.")

        if max_tokens < min_tokens:
            raise ValueError("max_tokens must be greater than or equal to min_tokens.")

        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative.")

        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens.")

        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_markdown(
        self,
        markdown: str,
        *,
        document_id: str,
        language: LanguageCode = "unknown",
    ) -> list[Chunk]:
        """Chunk a Markdown document into canonical Chunk models."""

        cleaned_markdown = self._normalize_text(markdown)

        if not cleaned_markdown:
            return []

        sections = self.parse_sections(cleaned_markdown)
        chunk_payloads: list[tuple[str, list[str]]] = []

        current_parts: list[str] = []
        current_context: list[str] = []
        current_tokens = 0
        previous_chunk_text = ""

        def flush_current() -> None:
            """Flush current buffer into chunk payloads."""

            nonlocal current_parts, current_context, current_tokens, previous_chunk_text

            text = self._normalize_text("\n\n".join(current_parts))

            if text:
                chunk_payloads.append((text, current_context.copy()))
                previous_chunk_text = text

            current_parts = []
            current_context = []
            current_tokens = 0

        for section in sections:
            wrapped_section_text = self.with_heading_context(section.text, section.heading_context)
            section_tokens = count_tokens(wrapped_section_text)

            if section_tokens > self.max_tokens:
                if current_parts:
                    flush_current()

                chunk_payloads.extend(self.split_oversized_section(section))
                previous_chunk_text = chunk_payloads[-1][0] if chunk_payloads else previous_chunk_text
                continue

            if current_parts and current_tokens + section_tokens > self.max_tokens:
                flush_current()

                overlap = self.tail_tokens_text(previous_chunk_text, self.overlap_tokens)
                overlap_tokens = count_tokens(overlap)

                # Keep overlap only when it does not cause the next chunk to exceed max_tokens.
                if overlap and overlap_tokens + section_tokens <= self.max_tokens:
                    current_parts = [overlap]
                    current_tokens = overlap_tokens
                    current_context = section.heading_context.copy()

            current_parts.append(wrapped_section_text)
            current_context = section.heading_context.copy()
            current_tokens += section_tokens

        if current_parts:
            flush_current()

        chunks: list[Chunk] = []

        for chunk_index, (chunk_text, heading_context) in enumerate(chunk_payloads):
            normalized_text = self._normalize_text(chunk_text)

            if not normalized_text:
                continue

            chunks.append(
                Chunk(
                    document_id=document_id,
                    language=language,
                    text=normalized_text,
                    chunk_index=chunk_index,
                    heading_context=heading_context,
                    token_count=count_tokens(normalized_text),
                )
            )

        logger.debug("Chunked document {} into {} chunks.", document_id, len(chunks))
        return chunks

    def chunk(
        self,
        markdown: str,
        *,
        document_id: str,
        language: LanguageCode = "unknown",
    ) -> list[Chunk]:
        """Alias for `chunk_markdown` used by pipeline commands."""

        return self.chunk_markdown(markdown, document_id=document_id, language=language)

    def parse_sections(self, markdown: str) -> list[MarkdownSection]:
        """Parse Markdown into heading-aware sections."""

        lines = markdown.splitlines()
        sections: list[MarkdownSection] = []
        heading_stack: list[tuple[int, str]] = []
        current_context: list[str] = []
        buffer: list[str] = []

        def flush_buffer() -> None:
            text = self._normalize_text("\n".join(buffer))
            if text:
                sections.append(
                    MarkdownSection(
                        heading_context=current_context.copy(),
                        text=text,
                        token_count=count_tokens(text),
                    )
                )
            buffer.clear()

        for line in lines:
            heading_match = HEADING_PATTERN.match(line.strip())

            if heading_match:
                flush_buffer()

                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                heading_stack[:] = [(existing_level, text) for existing_level, text in heading_stack if existing_level < level]
                heading_stack.append((level, heading_text))

                current_context = [text for _, text in heading_stack]
                buffer.append(line)
            else:
                buffer.append(line)

        flush_buffer()

        if not sections:
            sections.append(
                MarkdownSection(
                    heading_context=[],
                    text=markdown,
                    token_count=count_tokens(markdown),
                )
            )

        return sections

    def split_oversized_section(self, section: MarkdownSection) -> list[tuple[str, list[str]]]:
        """Split a single section that exceeds max token size."""

        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", section.text) if paragraph.strip()]
        if not paragraphs:
            return []

        output: list[tuple[str, list[str]]] = []
        current_paragraphs: list[str] = []
        current_tokens = 0

        context_prefix = self.context_prefix(section.heading_context)
        context_tokens = count_tokens(context_prefix)
        available_body_tokens = max(50, self.max_tokens - context_tokens)

        def flush_paragraphs() -> None:
            nonlocal current_paragraphs, current_tokens

            if current_paragraphs:
                text = self.with_heading_context("\n\n".join(current_paragraphs), section.heading_context)
                output.append((self._normalize_text(text), section.heading_context.copy()))
                current_paragraphs = []
                current_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = count_tokens(paragraph)

            if paragraph_tokens > available_body_tokens:
                flush_paragraphs()

                for window_text in self.token_windows(paragraph, available_body_tokens):
                    text = self.with_heading_context(window_text, section.heading_context)
                    output.append((self._normalize_text(text), section.heading_context.copy()))

                continue

            if current_paragraphs and current_tokens + paragraph_tokens > available_body_tokens:
                flush_paragraphs()

            current_paragraphs.append(paragraph)
            current_tokens += paragraph_tokens

        flush_paragraphs()
        return output

    def token_windows(self, text: str, window_size: int) -> list[str]:
        """Split text into overlapping token windows."""

        tokens = tokenize(text)

        if not tokens:
            return []

        window_size = max(1, min(window_size, self.max_tokens))
        step = max(1, window_size - self.overlap_tokens)
        windows: list[str] = []

        for start in range(0, len(tokens), step):
            window_tokens = tokens[start : start + window_size]
            if not window_tokens:
                break

            windows.append(detokenize_tokens(window_tokens))

            if start + window_size >= len(tokens):
                break

        return windows

    def with_heading_context(self, text: str, heading_context: list[str]) -> str:
        """Prefix text with heading context unless already present."""

        text = self._normalize_text(text)

        if not heading_context:
            return text

        prefix = self.context_prefix(heading_context)

        if text.startswith(prefix):
            return text

        return self._normalize_text(f"{prefix}\n\n{text}")

    def context_prefix(self, heading_context: list[str]) -> str:
        """Build a compact context prefix from Markdown headings."""

        cleaned = [heading.strip() for heading in heading_context if heading and heading.strip()]

        if not cleaned:
            return ""

        return "Context: " + " > ".join(cleaned)

    def tail_tokens_text(self, text: str, token_count: int) -> str:
        """Return the last `token_count` tokens of a text for overlap."""

        if token_count <= 0:
            return ""

        tokens = tokenize(text)

        if not tokens:
            return ""

        return detokenize_tokens(tokens[-token_count:])

    def _normalize_text(self, text: str) -> str:
        """Normalize chunk text whitespace without changing Arabic order."""

        text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text using a lightweight regex suitable for English and Arabic."""

    return TOKEN_PATTERN.findall(text or "")


def count_tokens(text: str) -> int:
    """Approximate token count for chunking without external tokenizer dependency."""

    return len(tokenize(text))


def detokenize_tokens(tokens: list[str]) -> str:
    """Best-effort detokenization for regex token windows."""

    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?،؛؟\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()