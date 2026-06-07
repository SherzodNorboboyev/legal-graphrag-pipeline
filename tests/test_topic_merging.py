from src.vector_ops.topic_merging import TopicMerger, TopicRecord


def test_cosine_threshold_grouping_detects_duplicates() -> None:
    merger = TopicMerger(driver=None)

    topics = [
        TopicRecord(name="Taxation", normalized_name="taxation", embedding=[1.0, 0.0, 0.0]),
        TopicRecord(name="Tax Law", normalized_name="tax law", embedding=[0.99, 0.01, 0.0]),
        TopicRecord(name="Maritime Law", normalized_name="maritime law", embedding=[0.0, 1.0, 0.0]),
    ]

    groups = merger.find_duplicate_groups(topics, threshold=0.95)

    assert len(groups) == 1
    assert set(groups[0].members) == {"taxation", "tax law"}


def test_no_group_when_similarity_below_threshold() -> None:
    merger = TopicMerger(driver=None)

    topics = [
        TopicRecord(name="Taxation", normalized_name="taxation", embedding=[1.0, 0.0]),
        TopicRecord(name="Maritime Law", normalized_name="maritime law", embedding=[0.0, 1.0]),
    ]

    groups = merger.find_duplicate_groups(topics, threshold=0.95)

    assert groups == []


def test_dry_run_merge_plan_contains_candidates_without_applying() -> None:
    merger = TopicMerger(driver=None)

    topics = [
        TopicRecord(name="Labour Regulation", normalized_name="labour regulation", embedding=[1.0, 0.0, 0.0]),
        TopicRecord(name="Labor Policies", normalized_name="labor policies", embedding=[0.98, 0.02, 0.0]),
        TopicRecord(name="Banking", normalized_name="banking", embedding=[0.0, 1.0, 0.0]),
    ]

    plan = merger.merge_duplicate_topics(
        threshold=0.95,
        dry_run=True,
        topics=topics,
    )

    assert plan.dry_run is True
    assert plan.applied_count == 0
    assert len(plan.groups) == 1
    assert len(plan.candidates) == 1
    assert plan.candidates[0].drop_normalized_name in {"labour regulation", "labor policies"}
    assert "dry_run=True" in plan.summary()


def test_choose_canonical_topic_prefers_shorter_name() -> None:
    merger = TopicMerger(driver=None)

    canonical = merger.choose_canonical_topic(["labour regulation", "labour"])

    assert canonical == "labour"