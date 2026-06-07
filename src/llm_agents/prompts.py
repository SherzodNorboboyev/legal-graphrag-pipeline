"""Prompt templates for legal topic extraction.

The prompts are designed for Oman legal documents that may be written in Arabic
or English. The model is instructed to return JSON only because downstream graph
ingestion expects structured topic objects.
"""

from __future__ import annotations

from textwrap import dedent


TOPIC_EXTRACTION_SYSTEM_PROMPT = dedent(
    """
    You are a legal knowledge engineering assistant building a GraphRAG index
    for Oman legal documents.

    Your job is to extract reusable legal topics, legal domains, regulated
    sectors, public authorities, and important legal concepts from a legal
    document.

    You must obey these rules:
    - Return JSON only.
    - Do not wrap the JSON in markdown fences.
    - Do not include explanations outside JSON.
    - Prefer concise English topic names when possible.
    - If the document is Arabic and no safe English translation is obvious,
      return a concise Arabic topic name.
    - Topics should be reusable across many documents, not overly specific
      one-off phrases.
    - Evidence must be a short phrase copied or closely paraphrased from the
      document.
    - Confidence must be a number between 0.0 and 1.0.
    """
).strip()


TOPIC_EXTRACTION_SCHEMA_EXAMPLE = {
    "topics": [
        {
            "name": "Taxation",
            "normalized_name": "taxation",
            "confidence": 0.92,
            "evidence": "taxable income and tax exemptions",
        }
    ]
}


def build_topic_extraction_prompt(
    *,
    title: str,
    content: str,
    language: str,
    max_topics: int = 12,
    max_content_chars: int = 12000,
) -> str:
    """Build a JSON-only topic extraction prompt.

    Parameters
    ----------
    title:
        Legal document title.
    content:
        Markdown content selected from contentEn or contentAr.
    language:
        Detected content language. Usually `en`, `ar`, or `unknown`.
    max_topics:
        Maximum number of topics requested from the LLM.
    max_content_chars:
        Safety bound to avoid sending very large documents to the LLM.
    """

    trimmed_content = (content or "").strip()[:max_content_chars]
    normalized_language = language or "unknown"

    return dedent(
        f"""
        Extract up to {max_topics} core legal topics/entities from the legal
        document below.

        Expected output schema:

        {{
          "topics": [
            {{
              "name": "...",
              "normalized_name": "...",
              "confidence": 0.0,
              "evidence": "..."
            }}
          ]
        }}

        Field requirements:
        - name: human-readable legal topic, preferably English.
        - normalized_name: lowercase, stable, deduplicated label. Use spaces,
          not punctuation. Example: "labour regulation".
        - confidence: number between 0.0 and 1.0.
        - evidence: short supporting phrase from the document.

        Good topic examples:
        - Taxation
        - Customs
        - Labour Regulation
        - Omanization
        - Maritime Law
        - Judicial Fees
        - Public Health
        - Education
        - Real Estate
        - Environmental Protection
        - Competition Law
        - Banking Regulation

        Avoid:
        - full document titles as topics
        - generic words such as "law", "article", "decision", "decree"
        - duplicate synonyms in the same response

        Document language: {normalized_language}

        Document title:
        {title}

        Document markdown:
        {trimmed_content}

        Return JSON only.
        """
    ).strip()


def build_json_repair_prompt(invalid_response: str) -> str:
    """Build a small prompt asking the model to repair invalid topic JSON.

    This prompt is used only after the first model response could not be parsed.
    It intentionally does not resend the full legal document, reducing cost.
    """

    return dedent(
        f"""
        The previous response was not valid JSON for the required schema.

        Convert the following text into valid JSON only:

        {invalid_response[:4000]}

        Required schema:

        {{
          "topics": [
            {{
              "name": "...",
              "normalized_name": "...",
              "confidence": 0.0,
              "evidence": "..."
            }}
          ]
        }}

        Return JSON only. Do not include markdown fences or commentary.
        """
    ).strip()