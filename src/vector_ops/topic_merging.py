"""Topic merging and community detection utilities.

This module implements bonus optimization features:
- topic duplicate detection by embedding cosine similarity
- dry-run merge planning
- Neo4j relationship consolidation for duplicate Topic nodes
- optional Neo4j GDS Louvain community detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from loguru import logger
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

from src.config import Settings, get_settings
from src.vector_ops.embeddings import cosine_similarity


@dataclass(frozen=True)
class TopicRecord:
    """In-memory topic record used for similarity grouping."""

    name: str
    normalized_name: str
    embedding: list[float]


@dataclass(frozen=True)
class TopicMergeCandidate:
    """One duplicate topic merge candidate."""

    keep_normalized_name: str
    drop_normalized_name: str
    similarity: float


@dataclass
class TopicMergeGroup:
    """Group of duplicate topics above threshold."""

    canonical_normalized_name: str
    members: list[str]
    pair_similarities: dict[tuple[str, str], float] = field(default_factory=dict)


@dataclass
class TopicMergePlan:
    """Result of a topic merge dry-run or applied merge."""

    threshold: float
    dry_run: bool
    groups: list[TopicMergeGroup] = field(default_factory=list)
    candidates: list[TopicMergeCandidate] = field(default_factory=list)
    applied_count: int = 0

    def summary(self) -> str:
        """Return a compact human-readable merge summary."""

        return (
            f"TopicMergePlan(threshold={self.threshold}, dry_run={self.dry_run}, "
            f"groups={len(self.groups)}, candidates={len(self.candidates)}, "
            f"applied_count={self.applied_count})"
        )


class TopicMerger:
    """Merge duplicate Topic nodes and run optional community detection."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        driver: Driver | None = None,
    ):
        self.settings = settings or get_settings()
        self.database = getattr(self.settings, "neo4j_database", "neo4j")

        if driver is None:
            self.driver: Driver | None = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
                max_connection_pool_size=10,
            )
            self._owns_driver = True
        else:
            self.driver = driver
            self._owns_driver = False

    def __enter__(self) -> "TopicMerger":
        """Return context-managed merger."""

        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close owned Neo4j driver."""

        self.close()

    def close(self) -> None:
        """Close owned Neo4j driver."""

        if self._owns_driver and self.driver is not None:
            self.driver.close()

    def fetch_topic_records(self) -> list[TopicRecord]:
        """Fetch all Topic nodes that have embeddings."""

        if self.driver is None:
            return []

        query = """
        MATCH (t:Topic)
        WHERE t.embedding IS NOT NULL
        RETURN t.name AS name,
               t.normalized_name AS normalized_name,
               t.embedding AS embedding
        ORDER BY t.normalized_name
        """

        with self.driver.session(database=self.database) as session:
            records = list(session.run(query))

        topics: list[TopicRecord] = []

        for record in records:
            normalized_name = record.get("normalized_name")
            embedding = record.get("embedding")

            if not normalized_name or not embedding:
                continue

            topics.append(
                TopicRecord(
                    name=record.get("name") or normalized_name,
                    normalized_name=normalized_name,
                    embedding=[float(value) for value in embedding],
                )
            )

        return topics

    def find_duplicate_groups(
        self,
        topics: Sequence[TopicRecord] | None = None,
        *,
        threshold: float | None = None,
    ) -> list[TopicMergeGroup]:
        """Detect duplicate topic groups by cosine similarity."""

        effective_threshold = threshold if threshold is not None else getattr(self.settings, "topic_merge_similarity", 0.88)
        topic_records = list(topics) if topics is not None else self.fetch_topic_records()

        if not topic_records:
            return []

        parent: dict[str, str] = {topic.normalized_name: topic.normalized_name for topic in topic_records}
        topic_by_name = {topic.normalized_name: topic for topic in topic_records}
        pair_similarities: dict[tuple[str, str], float] = {}

        def find(name: str) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        def union(left: str, right: str) -> None:
            root_left = find(left)
            root_right = find(right)

            if root_left == root_right:
                return

            canonical = self.choose_canonical_topic([root_left, root_right])
            other = root_right if canonical == root_left else root_left
            parent[other] = canonical

        for index, left_topic in enumerate(topic_records):
            for right_topic in topic_records[index + 1 :]:
                similarity = cosine_similarity(left_topic.embedding, right_topic.embedding)

                if similarity >= effective_threshold:
                    key = tuple(sorted((left_topic.normalized_name, right_topic.normalized_name)))
                    pair_similarities[key] = similarity
                    union(left_topic.normalized_name, right_topic.normalized_name)

        grouped_names: dict[str, list[str]] = {}

        for topic in topic_records:
            root = find(topic.normalized_name)
            grouped_names.setdefault(root, []).append(topic.normalized_name)

        groups: list[TopicMergeGroup] = []

        for members in grouped_names.values():
            unique_members = sorted(set(members))

            if len(unique_members) < 2:
                continue

            canonical = self.choose_canonical_topic(unique_members)
            group_pair_scores = {
                key: value
                for key, value in pair_similarities.items()
                if key[0] in unique_members and key[1] in unique_members
            }

            # Keep members ordered with canonical first for deterministic plans.
            ordered_members = [canonical] + [member for member in unique_members if member != canonical]
            groups.append(
                TopicMergeGroup(
                    canonical_normalized_name=canonical,
                    members=ordered_members,
                    pair_similarities=group_pair_scores,
                )
            )

        logger.info(
            "Detected {} duplicate topic groups at threshold {} from {} topics.",
            len(groups),
            effective_threshold,
            len(topic_records),
        )
        return groups

    def build_merge_candidates(
        self,
        groups: Sequence[TopicMergeGroup],
        topics: Sequence[TopicRecord],
    ) -> list[TopicMergeCandidate]:
        """Build pairwise merge candidates from duplicate groups."""

        topic_by_name = {topic.normalized_name: topic for topic in topics}
        candidates: list[TopicMergeCandidate] = []

        for group in groups:
            keep = group.canonical_normalized_name
            keep_topic = topic_by_name.get(keep)

            if keep_topic is None:
                continue

            for drop in group.members:
                if drop == keep:
                    continue

                drop_topic = topic_by_name.get(drop)
                if drop_topic is None:
                    continue

                similarity = cosine_similarity(keep_topic.embedding, drop_topic.embedding)
                candidates.append(
                    TopicMergeCandidate(
                        keep_normalized_name=keep,
                        drop_normalized_name=drop,
                        similarity=similarity,
                    )
                )

        candidates.sort(key=lambda candidate: candidate.similarity, reverse=True)
        return candidates

    def merge_duplicate_topics(
        self,
        *,
        threshold: float | None = None,
        dry_run: bool = True,
        topics: Sequence[TopicRecord] | None = None,
    ) -> TopicMergePlan:
        """Create or apply a duplicate topic merge plan."""

        effective_threshold = threshold if threshold is not None else getattr(self.settings, "topic_merge_similarity", 0.88)
        topic_records = list(topics) if topics is not None else self.fetch_topic_records()
        groups = self.find_duplicate_groups(topic_records, threshold=effective_threshold)
        candidates = self.build_merge_candidates(groups, topic_records)

        plan = TopicMergePlan(
            threshold=effective_threshold,
            dry_run=dry_run,
            groups=groups,
            candidates=candidates,
            applied_count=0,
        )

        if dry_run:
            for candidate in candidates:
                logger.info(
                    "Dry-run topic merge: keep='{}', drop='{}', similarity={:.4f}",
                    candidate.keep_normalized_name,
                    candidate.drop_normalized_name,
                    candidate.similarity,
                )
            return plan

        if self.driver is None:
            raise RuntimeError("Cannot apply topic merges without a Neo4j driver.")

        applied_count = 0
        for candidate in candidates:
            applied_count += self.merge_topic_pair(
                keep_normalized_name=candidate.keep_normalized_name,
                drop_normalized_name=candidate.drop_normalized_name,
            )

        plan.applied_count = applied_count
        return plan

    def merge_topic_pair(self, *, keep_normalized_name: str, drop_normalized_name: str) -> int:
        """Merge one duplicate Topic node into its canonical Topic node.

        The Cypher consolidates incoming HAS_TOPIC relationships from Documents,
        appends the dropped topic name into `aliases`, and then deletes the
        duplicate Topic node.
        """

        if self.driver is None:
            raise RuntimeError("Cannot merge topics without a Neo4j driver.")

        if keep_normalized_name == drop_normalized_name:
            return 0

        query = """
        MATCH (keep:Topic {normalized_name: $keep})
        MATCH (drop:Topic {normalized_name: $drop})
        OPTIONAL MATCH (d:Document)-[oldRel:HAS_TOPIC]->(drop)
        WITH keep, drop, collect({
            document_id: d.id,
            confidence: oldRel.confidence,
            evidence: oldRel.evidence,
            source: oldRel.source
        }) AS rels
        UNWIND rels AS rel
        WITH keep, drop, rel
        WHERE rel.document_id IS NOT NULL
        MATCH (doc:Document {id: rel.document_id})
        MERGE (doc)-[newRel:HAS_TOPIC]->(keep)
        SET newRel.confidence = coalesce(newRel.confidence, rel.confidence),
            newRel.evidence = coalesce(newRel.evidence, rel.evidence),
            newRel.source = coalesce(newRel.source, rel.source),
            newRel.updated_at = datetime()
        WITH keep, drop
        SET keep.aliases = coalesce(keep.aliases, []) +
            CASE WHEN drop.name IS NULL THEN [] ELSE [drop.name] END,
            keep.updated_at = datetime()
        DETACH DELETE drop
        RETURN 1 AS merged
        """

        try:
            with self.driver.session(database=self.database) as session:
                record = session.run(query, keep=keep_normalized_name, drop=drop_normalized_name).single()

            merged = int(record["merged"] if record else 0)
            if merged:
                logger.info("Merged Topic '{}' into '{}'.", drop_normalized_name, keep_normalized_name)
            return merged

        except Neo4jError as exc:
            logger.error("Failed to merge Topic '{}' into '{}': {}", drop_normalized_name, keep_normalized_name, exc)
            return 0

    def run_louvain_community_detection(
        self,
        *,
        graph_name: str = "legal_topic_document_graph",
        write_property: str = "community_id",
    ) -> dict[str, object]:
        """Run Neo4j GDS Louvain community detection if available.

        Returns a structured status dictionary instead of raising when GDS is not
        installed. This keeps the bonus command safe in local Community Edition.
        """

        if self.driver is None:
            return {
                "ok": False,
                "reason": "No Neo4j driver available.",
            }

        try:
            with self.driver.session(database=self.database) as session:
                # Drop existing projection if present. This may fail if GDS is not installed.
                try:
                    session.run("CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName", graph_name=graph_name).consume()
                except Neo4jError:
                    pass

                session.run(
                    """
                    CALL gds.graph.project(
                        $graph_name,
                        ['Document', 'Topic'],
                        {
                            HAS_TOPIC: {
                                orientation: 'UNDIRECTED'
                            }
                        }
                    )
                    YIELD graphName, nodeCount, relationshipCount
                    RETURN graphName, nodeCount, relationshipCount
                    """,
                    graph_name=graph_name,
                ).consume()

                record = session.run(
                    """
                    CALL gds.louvain.write($graph_name, {writeProperty: $write_property})
                    YIELD communityCount, modularity, ranLevels
                    RETURN communityCount, modularity, ranLevels
                    """,
                    graph_name=graph_name,
                    write_property=write_property,
                ).single()

                session.run("CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName", graph_name=graph_name).consume()

            return {
                "ok": True,
                "community_count": record["communityCount"] if record else None,
                "modularity": record["modularity"] if record else None,
                "ran_levels": record["ranLevels"] if record else None,
                "write_property": write_property,
            }

        except Neo4jError as exc:
            logger.warning("Neo4j GDS Louvain is unavailable or failed: {}", exc)
            return {
                "ok": False,
                "reason": str(exc),
                "hint": "Install Neo4j Graph Data Science plugin to enable Louvain community detection.",
            }

    def choose_canonical_topic(self, names: Iterable[str]) -> str:
        """Choose a deterministic canonical topic name.

        Shorter normalized names are usually broader and better as canonical
        labels. Lexicographic order breaks ties deterministically.
        """

        clean_names = [name for name in names if name]

        if not clean_names:
            raise ValueError("Cannot choose canonical topic from empty names.")

        return sorted(clean_names, key=lambda item: (len(item), item))[0]