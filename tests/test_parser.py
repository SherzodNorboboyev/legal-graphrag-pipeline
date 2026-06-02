from src.scraper.parser import QanoonParser


SAMPLE_ARABIC_HTML = """
<html lang="ar" dir="rtl">
  <head>
    <title>وزارة العدل والشؤون القانونية: قرار وزاري رقم ٥٠ / ٢٠٢٦</title>
    <meta property="article:published_time" content="2026-05-20" />
  </head>
  <body>
    <header>header noise</header>
    <article>
      <h1>وزارة العدل والشؤون القانونية: قرار وزاري رقم ٥٠ / ٢٠٢٦ بتخويل صفة الضبطية القضائية</h1>
      <time datetime="2026-05-20">٢٠ مايو ٢٠٢٦</time>
      <a href="https://data.qanoon.om/pdf/example.pdf">تحميل</a>
      <a href="https://decree.om/2026/md2026050/">English</a>
      <p>استنادا إلى قانون الإجراءات الجزائية الصادر بالمرسوم السلطاني رقم ٩٧ / ٩٩.</p>
      <p>وعلى قانون حماية المنافسة ومنع الاحتكار الصادر بالمرسوم السلطاني رقم ٦٧ / ٢٠١٤.</p>
      <h3>المادة الأولى</h3>
      <p>يخول شاغلو الوظائف الآتية صفة الضبطية القضائية.</p>
    </article>
  </body>
</html>
"""


SAMPLE_LISTING_HTML = """
<html>
  <body>
    <a href="/p/category/royal-decrees/">Royal Decrees</a>
    <a href="/p/2026/rd2026058/">المرسوم السلطاني ٥٨ / ٢٠٢٦</a>
    <a href="https://data.qanoon.om/pdf/rd2026058.pdf">تحميل</a>
    <a href="https://decree.om/2026/rd2026058/">English</a>
  </body>
</html>
"""


def test_parser_extracts_document_metadata_from_sample_html() -> None:
    parser = QanoonParser("https://qanoon.om/", "https://decree.om/")

    parsed = parser.parse_document(
        SAMPLE_ARABIC_HTML,
        "https://qanoon.om/p/2026/md2026050/",
    )

    assert parsed.title.startswith("وزارة العدل والشؤون القانونية")
    assert parsed.date == "2026-05-20"
    assert parsed.document_type == "Ministerial Decision"
    assert parsed.number == "50/2026"
    assert parsed.issuer == "وزارة العدل والشؤون القانونية"
    assert parsed.source_url == "https://qanoon.om/p/2026/md2026050/"
    assert parsed.language == "ar"
    assert parsed.language_urls["en"] == "https://decree.om/2026/md2026050/"
    assert "https://data.qanoon.om/pdf/example.pdf" in parsed.pdf_urls
    assert "المادة الأولى" in (parsed.content_text or "")


def test_parser_discovers_links_defensively() -> None:
    parser = QanoonParser("https://qanoon.om/", "https://decree.om/")

    discovered = parser.discover_links(SAMPLE_LISTING_HTML, "https://qanoon.om/")

    assert "https://qanoon.om/p/category/royal-decrees/" in discovered.listing_urls
    assert "https://qanoon.om/p/2026/rd2026058/" in discovered.document_urls
    assert "https://data.qanoon.om/pdf/rd2026058.pdf" in discovered.pdf_urls
    assert discovered.language_urls["en"] == "https://decree.om/2026/rd2026058/"


def test_parser_extracts_cross_reference_candidates() -> None:
    parser = QanoonParser("https://qanoon.om/", "https://decree.om/")

    text = """
    يلغى المرسوم السلطاني رقم ٥٧ / ٢٠١٢ المشار إليه.
    This decision amends Royal Decree Number 33/2013.
    """

    references = parser.extract_cross_references(text)
    normalized_numbers = {reference.normalized_number for reference in references}
    relation_hints = {(reference.normalized_number, reference.relation_hint) for reference in references}

    assert "57/2012" in normalized_numbers
    assert "33/2013" in normalized_numbers
    assert ("57/2012", "REPEALS") in relation_hints
    assert ("33/2013", "AMENDS") in relation_hints


def test_parser_detects_english_language_from_decree_host() -> None:
    parser = QanoonParser("https://qanoon.om/", "https://decree.om/")

    html = """
    <html lang="en">
      <body>
        <article>
          <h1>Royal Decree 58/2026 Promulgating the Urban Planning Law</h1>
          <p>We, Haitham bin Tarik, Sultan of Oman.</p>
        </article>
      </body>
    </html>
    """

    parsed = parser.parse_document(html, "https://decree.om/2026/rd2026058/")

    assert parsed.language == "en"
    assert parsed.document_type == "Royal Decree"
    assert parsed.number == "58/2026"


def test_parser_normalizes_urls_and_filters_non_documents() -> None:
    parser = QanoonParser("https://qanoon.om/", "https://decree.om/")

    assert parser.normalize_url("https://qanoon.om/p/2026/rd2026058#comments") == "https://qanoon.om/p/2026/rd2026058/"
    assert parser.is_document_url("https://qanoon.om/p/2026/rd2026058/")
    assert not parser.is_document_url("https://qanoon.om/p/category/royal-decrees/")
    assert not parser.is_document_url("https://qanoon.om/")