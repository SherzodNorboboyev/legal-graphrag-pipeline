import pytest
from pydantic import ValidationError

from src.models import (
    Chunk,
    CrossReference,
    LegalDocument,
    ParsedDocument,
    Topic,
    generate_chunk_id,
    generate_document_id,
    normalize_document_number,
    normalize_topic_name,
)


def test_generate_document_id_is_deterministic_and_translation_safe() -> None:
    arabic_id = generate_document_id(
        source_url="https://qanoon.om/p/2026/rd2026058/",
        title="مرسوم سلطاني رقم ٥٨ / ٢٠٢٦",
        document_type="Royal Decree",
        number="٥٨ / ٢٠٢٦",
    )
    english_id = generate_document_id(
        source_url="https://decree.om/2026/rd2026058/",
        title="Royal Decree 58/2026",
        document_type="Royal Decree",
        number="58/2026",
    )

    assert arabic_id == english_id
    assert arabic_id.startswith("doc-")


def test_generate_document_id_falls_back_to_canonical_url() -> None:
    id_one = generate_document_id(source_url="https://qanoon.om/p/2026/example#comments", title="Example")
    id_two = generate_document_id(source_url="https://qanoon.om/p/2026/example/", title="Different title")

    assert id_one == id_two


def test_normalize_document_number_converts_arabic_digits() -> None:
    assert normalize_document_number("٥٠ / ٢٠٢٦") == "50/2026"
    assert normalize_document_number("  12 / 1999 ") == "12/1999"


def test_legal_document_validation_and_id_generation() -> None:
    document = LegalDocument(
        title="وزارة العدل والشؤون القانونية: قرار وزاري رقم ٥٠ / ٢٠٢٦",
        document_type="Ministerial Decision",
        number="٥٠ / ٢٠٢٦",
        issuer="وزارة العدل والشؤون القانونية",
        source_url="https://qanoon.om/p/2026/md2026050/",
        language="ar",
        contentAr="# قرار وزاري\n\nنص قانوني.",
    )

    expected_id = generate_document_id(
        source_url="https://qanoon.om/p/2026/md2026050/",
        title=document.title,
        document_type="Ministerial Decision",
        number="50/2026",
    )

    assert document.id == expected_id
    assert document.number == "50/2026"
    assert document.content_ar == "# قرار وزاري\n\nنص قانوني."
    assert document.title_ar == document.title


def test_legal_document_rejects_invalid_language() -> None:
    with pytest.raises(ValidationError):
        LegalDocument(
            title="Invalid language document",
            source_url="https://qanoon.om/p/example/",
            language="de",
        )


def test_legal_document_rejects_relative_source_url() -> None:
    with pytest.raises(ValidationError):
        LegalDocument(
            title="Relative URL",
            source_url="/p/example/",
            language="en",
        )


def test_topic_normalized_name_generation() -> None:
    topic = Topic(name=" Labour Regulation! ")

    assert topic.name == "Labour Regulation!"
    assert topic.normalized_name == "labour regulation"
    assert normalize_topic_name("Taxation & Fees") == "taxation fees"


def test_topic_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        Topic(name="   ")


def test_cross_reference_accepts_parser_aliases() -> None:
    reference = CrossReference.model_validate(
        {
            "relation_hint": "repeals",
            "normalized_number": "٥٧ / ٢٠١٢",
            "document_type": "Royal Decree",
            "raw_text": "يلغى المرسوم السلطاني رقم ٥٧ / ٢٠١٢",
        }
    )

    assert reference.relation_type == "REPEALS"
    assert reference.target_number == "57/2012"
    assert reference.target_document_type == "Royal Decree"
    assert reference.is_document_level_relationship


def test_cross_reference_rejects_unknown_relation_type() -> None:
    with pytest.raises(ValidationError):
        CrossReference(
            relation_type="MENTIONS",
            target_number="1/2020",
            raw_text="bad relation",
        )


def test_chunk_generates_deterministic_id_and_token_count() -> None:
    text = "Article One. This law applies to all taxable entities."
    chunk = Chunk(
        document_id="doc-example",
        language="en",
        text=text,
        chunk_index=0,
        heading_context=["Tax Law", "Article One"],
    )

    expected_id = generate_chunk_id(
        document_id="doc-example",
        language="en",
        chunk_index=0,
        text=text,
    )

    assert chunk.id == expected_id
    assert chunk.token_count is not None
    assert chunk.token_count > 0
    assert chunk.heading_context == ["Tax Law", "Article One"]


def test_chunk_rejects_invalid_language() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            document_id="doc-example",
            language="uz",
            text="Some legal text.",
            chunk_index=0,
        )


def test_chunk_rejects_empty_embedding() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            document_id="doc-example",
            language="en",
            text="Some legal text.",
            chunk_index=0,
            embedding=[],
        )


def test_parsed_document_converts_to_legal_document_with_language_markdown() -> None:
    parsed = ParsedDocument(
        title="Royal Decree 58/2026",
        document_type="Royal Decree",
        number="58/2026",
        source_url="https://decree.om/2026/rd2026058/",
        language="en",
        markdown="# Royal Decree 58/2026\n\nEnglish content.",
        language_urls={"en": "https://decree.om/2026/rd2026058/"},
        cross_references=[
            {
                "relation_hint": "AMENDS",
                "normalized_number": "33/2013",
                "raw_text": "amends Royal Decree 33/2013",
            }
        ],
    )

    document = parsed.to_legal_document()

    assert isinstance(document, LegalDocument)
    assert document.id == parsed.id
    assert document.content_en == "# Royal Decree 58/2026\n\nEnglish content."
    assert document.language_urls["en"] == "https://decree.om/2026/rd2026058/"
    assert document.cross_references[0].relation_type == "AMENDS"
    assert document.cross_references[0].target_number == "33/2013"