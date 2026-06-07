from src.llm_agents.topic_extractor import TopicExtractor


def test_topic_extractor_fallback_uses_english_content_first() -> None:
    extractor = TopicExtractor(max_topics=5, min_confidence=0.2)

    topics = extractor.extract(
        title="Royal Decree issuing Tax Law",
        content_en="This law regulates taxation, taxable income, and tax exemptions.",
        content_ar="ينظم هذا القانون ضريبة الدخل.",
    )

    normalized_names = {topic.normalized_name for topic in topics}

    assert "taxation" in normalized_names
    assert all(topic.confidence >= 0.2 for topic in topics)


def test_topic_extractor_fallback_supports_arabic_content() -> None:
    extractor = TopicExtractor(max_topics=5, min_confidence=0.2)

    topics = extractor.extract(
        title="قرار وزاري بشأن الرسوم",
        content_en=None,
        content_ar="يحدد هذا القرار رسوم التسجيل ورسوم التجديد.",
    )

    normalized_names = {topic.normalized_name for topic in topics}

    assert "fees" in normalized_names


def test_safe_parse_topics_valid_json_object() -> None:
    extractor = TopicExtractor(max_topics=5, min_confidence=0.2)

    raw = """
    {
      "topics": [
        {
          "name": "Taxation",
          "normalized_name": "taxation",
          "confidence": 0.91,
          "evidence": "taxable income"
        },
        {
          "name": "Fees",
          "normalized_name": "fees",
          "confidence": 0.75,
          "evidence": "registration fees"
        }
      ]
    }
    """

    topics = extractor.safe_parse_topics(raw, source="test", max_topics=5, min_confidence=0.2)

    assert len(topics) == 2
    assert topics[0].normalized_name == "taxation"
    assert topics[0].source == "test"


def test_safe_parse_topics_handles_markdown_fenced_json() -> None:
    extractor = TopicExtractor(max_topics=5, min_confidence=0.2)

    raw = """
    Here is the output:

    ```json
    {
      "topics": [
        {
          "name": "Labour Regulation",
          "normalized_name": "labour regulation",
          "confidence": 0.88,
          "evidence": "employment and wages"
        }
      ]
    }
    ```
    """

    topics = extractor.safe_parse_topics(raw, source="test", max_topics=5, min_confidence=0.2)

    assert len(topics) == 1
    assert topics[0].normalized_name == "labour regulation"


def test_safe_parse_topics_invalid_json_returns_empty_list() -> None:
    extractor = TopicExtractor(max_topics=5, min_confidence=0.2)

    topics = extractor.safe_parse_topics(
        "not json at all",
        source="test",
        max_topics=5,
        min_confidence=0.2,
    )

    assert topics == []


def test_safe_parse_topics_filters_low_confidence_and_deduplicates() -> None:
    extractor = TopicExtractor(max_topics=5, min_confidence=0.5)

    raw = """
    {
      "topics": [
        {
          "name": "Taxation",
          "normalized_name": "taxation",
          "confidence": 0.3,
          "evidence": "weak evidence"
        },
        {
          "name": "Tax",
          "normalized_name": "taxation",
          "confidence": 0.9,
          "evidence": "strong evidence"
        }
      ]
    }
    """

    topics = extractor.safe_parse_topics(raw, source="test", max_topics=5, min_confidence=0.5)

    assert len(topics) == 1
    assert topics[0].normalized_name == "taxation"
    assert topics[0].confidence == 0.9