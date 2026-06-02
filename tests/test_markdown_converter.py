from bs4 import BeautifulSoup

from src.scraper.markdown_converter import HtmlToMarkdownConverter


def test_script_style_navigation_and_footer_are_removed() -> None:
    html = """
    <html>
      <head>
        <style>body { color: red; }</style>
        <script>window.tracker = true;</script>
      </head>
      <body>
        <header>Site header</header>
        <nav>Navigation</nav>
        <article>
          <h1>Royal Decree 1/2026</h1>
          <p>This is the legal body.</p>
        </article>
        <footer>Footer text</footer>
      </body>
    </html>
    """

    result = HtmlToMarkdownConverter().convert(html)

    assert "window.tracker" not in result.markdown
    assert "color: red" not in result.markdown
    assert "Site header" not in result.markdown
    assert "Navigation" not in result.markdown
    assert "Footer text" not in result.markdown
    assert "# Royal Decree 1/2026" in result.markdown
    assert "This is the legal body." in result.markdown


def test_heading_conversion_preserves_hierarchy() -> None:
    html = """
    <article>
      <h1>Main Law</h1>
      <h2>Chapter One</h2>
      <h3>Article One</h3>
      <p>Legal text.</p>
    </article>
    """

    result = HtmlToMarkdownConverter().convert(html)

    assert "# Main Law" in result.markdown
    assert "## Chapter One" in result.markdown
    assert "### Article One" in result.markdown
    assert "Legal text." in result.markdown


def test_table_is_converted_to_markdown_table() -> None:
    html = """
    <article>
      <h1>Fees Schedule</h1>
      <table>
        <tr><th>Service</th><th>Fee</th></tr>
        <tr><td>Registration</td><td>10 OMR</td></tr>
        <tr><td>Renewal</td><td>5 OMR</td></tr>
      </table>
    </article>
    """

    result = HtmlToMarkdownConverter().convert(html)

    assert "| Service | Fee |" in result.markdown
    assert "| --- | --- |" in result.markdown
    assert "| Registration | 10 OMR |" in result.markdown
    assert "| Renewal | 5 OMR |" in result.markdown


def test_arabic_text_is_not_reordered_or_removed() -> None:
    html = """
    <article>
      <h1>المادة الأولى</h1>
      <p>يعمل بهذا القانون من تاريخ نشره في الجريدة الرسمية.</p>
    </article>
    """

    result = HtmlToMarkdownConverter().convert(html)

    assert "المادة الأولى" in result.markdown
    assert "يعمل بهذا القانون من تاريخ نشره في الجريدة الرسمية." in result.markdown


def test_pdf_text_wrapper_creates_markdown_with_source() -> None:
    result = HtmlToMarkdownConverter().convert_pdf_text(
        "First page text.\nSecond line.",
        source_url="https://data.qanoon.om/example.pdf",
        title="Example PDF",
    )

    assert "# Example PDF" in result.markdown
    assert "Source: https://data.qanoon.om/example.pdf" in result.markdown
    assert "First page text." in result.markdown


def test_table_to_markdown_escapes_pipe_characters() -> None:
    soup = BeautifulSoup(
        """
        <table>
          <tr><th>Name</th><th>Value</th></tr>
          <tr><td>A | B</td><td>10</td></tr>
        </table>
        """,
        "lxml",
    )

    table = soup.find("table")
    assert table is not None

    markdown = HtmlToMarkdownConverter().table_to_markdown(table)

    assert "A \\| B" in markdown