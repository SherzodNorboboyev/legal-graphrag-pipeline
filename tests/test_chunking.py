from src.vector_ops.chunking import MarkdownChunker, count_tokens


def test_short_document_returns_single_chunk_with_heading_context() -> None:
    markdown = """
# Tax Law

## Chapter One

Article One. This law applies to taxable entities in Oman.
"""

    chunker = MarkdownChunker(min_tokens=20, max_tokens=120, overlap_tokens=10)
    chunks = chunker.chunk_markdown(markdown, document_id="doc-tax", language="en")

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-tax"
    assert chunks[0].language == "en"
    assert "Tax Law" in chunks[0].heading_context
    assert "Chapter One" in chunks[0].heading_context
    assert "Context: Tax Law > Chapter One" in chunks[0].text


def test_chunk_size_for_large_markdown() -> None:
    repeated = "This article regulates taxable income and tax obligations. " * 120
    markdown = f"""
# Tax Law

## Chapter One

{repeated}

## Chapter Two

{repeated}
"""

    chunker = MarkdownChunker(min_tokens=50, max_tokens=130, overlap_tokens=20)
    chunks = chunker.chunk_markdown(markdown, document_id="doc-large", language="en")

    assert len(chunks) > 1
    assert all(chunk.token_count <= 130 for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


def test_deterministic_chunk_ids() -> None:
    markdown = """
# Labour Law

## Wages

Workers must receive wages according to the applicable rules.
"""

    chunker = MarkdownChunker(min_tokens=10, max_tokens=100, overlap_tokens=5)

    chunks_one = chunker.chunk_markdown(markdown, document_id="doc-labour", language="en")
    chunks_two = chunker.chunk_markdown(markdown, document_id="doc-labour", language="en")

    assert [chunk.id for chunk in chunks_one] == [chunk.id for chunk in chunks_two]


def test_empty_document_returns_no_chunks() -> None:
    chunker = MarkdownChunker(min_tokens=10, max_tokens=100, overlap_tokens=5)

    assert chunker.chunk_markdown("", document_id="doc-empty", language="en") == []
    assert chunker.chunk_markdown("   \n\n   ", document_id="doc-empty", language="en") == []


def test_count_tokens_supports_arabic_and_english() -> None:
    text = "Article One المادة الأولى applies to taxation ضريبة."

    assert count_tokens(text) >= 8