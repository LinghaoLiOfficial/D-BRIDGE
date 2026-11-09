from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError


@dataclass(frozen=True)
class Concept:
    concept_id: str
    name: str
    embedding: list[float]


class GraphStore(Protocol):
    def reset(self) -> None: ...
    def candidates(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]: ...
    def initialize_posts(self, posts: list[dict[str, Any]], batch_size: int) -> None: ...
    def upsert_concept(self, concept: Concept, source_post_ids: list[str], batch_size: int) -> None: ...
    def merge_concept(self, source_id: str, target_id: str, batch_size: int) -> None: ...
    def relate(self, source_id: str, target_id: str, relation: str, reason: str, batch_size: int) -> None: ...
    def hierarchy(self) -> tuple[list[str], list[tuple[str, str]]]: ...
    def set_granularities(self, granularities: dict[str, int], batch_size: int) -> None: ...
    def export(self) -> dict[str, list[dict[str, Any]]]: ...
    def close(self) -> None: ...


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str, database: str, embedding_dimensions: int):
        self.driver: Driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.embedding_dimensions = embedding_dimensions
        try:
            self.driver.verify_connectivity()
        except AuthError as exc:
            self.driver.close()
            raise RuntimeError(
                "Neo4j authentication failed. Ensure NEO4J_USERNAME and NEO4J_PASSWORD "
                "in the selected env file match the credentials used to initialize the "
                "persistent Neo4j volume. For this local project, run `make neo4j-reset` "
                "only if discarding the existing Neo4j data is acceptable."
            ) from exc
        except Exception:
            self.driver.close()
            raise
        with self.driver.session(database=database) as session:
            session.run("CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.post_id IS UNIQUE").consume()
            session.run(
                "CREATE VECTOR INDEX concept_embedding IF NOT EXISTS FOR (c:Concept) ON (c.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dimensions}, `vector.similarity_function`: 'cosine'}}}}"
            ).consume()

    def reset(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) WHERE n:Post OR n:Concept DETACH DELETE n").consume()

    def candidates(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if len(embedding) != self.embedding_dimensions:
            raise ValueError(f"Embedding has {len(embedding)} dimensions; expected {self.embedding_dimensions}")
        # Use the explicit vector index when available, with a property scan fallback for small or warming indexes.
        with self.driver.session(database=self.database) as session:
            try:
                rows = session.run(
                    "CALL db.index.vector.queryNodes('concept_embedding', $top_k, $embedding) "
                    "YIELD node, score RETURN node.concept_id AS concept_id, node.name AS name, score AS similarity "
                    "ORDER BY similarity DESC",
                    top_k=top_k,
                    embedding=embedding,
                )
                return [dict(row) for row in rows]
            except Exception:
                rows = session.run(
                    "MATCH (c:Concept) WHERE c.embedding IS NOT NULL "
                    "RETURN c.concept_id AS concept_id, c.name AS name, c.embedding AS embedding"
                )
                scored = []
                for row in rows:
                    similarity = _cosine(embedding, row["embedding"])
                    scored.append({"concept_id": row["concept_id"], "name": row["name"], "similarity": similarity})
                return sorted(scored, key=lambda item: item["similarity"], reverse=True)[:top_k]

    def initialize_posts(self, posts: list[dict[str, Any]], batch_size: int) -> None:
        batches = [posts[index:index + batch_size] for index in range(0, len(posts), batch_size)]
        with self.driver.session(database=self.database) as session:
            for batch in batches:
                records = [
                    {
                        "post_id": str(post["post_id"]),
                        "properties": {
                            key: _neo4j_property(value)
                            for key, value in post.items()
                            if key != "post_id" and value is not None
                        },
                    }
                    for post in batch
                ]
                session.run(
                    "UNWIND $posts AS post "
                    "MERGE (p:Post {post_id: post.post_id}) "
                    "SET p += post.properties",
                    posts=records,
                ).consume()

    def upsert_concept(self, concept: Concept, source_post_ids: list[str], batch_size: int) -> None:
        if len(concept.embedding) != self.embedding_dimensions:
            raise ValueError(f"Embedding has {len(concept.embedding)} dimensions; expected {self.embedding_dimensions}")
        batches = [source_post_ids[index:index + batch_size] for index in range(0, len(source_post_ids), batch_size)] or [[]]
        with self.driver.session(database=self.database) as session:
            for batch in batches:
                session.run(
                    "MERGE (c:Concept {concept_id: $concept_id}) "
                    "ON CREATE SET c.name = $name, c.embedding = $embedding, c.created_at = datetime() "
                    "ON MATCH SET c.name = $name, c.embedding = $embedding, c.updated_at = datetime() "
                    "WITH c UNWIND $post_ids AS post_id "
                    "MATCH (p:Post {post_id: post_id}) "
                    "MERGE (p)-[:MENTIONS]->(c)",
                    concept_id=concept.concept_id,
                    name=concept.name,
                    embedding=concept.embedding,
                    post_ids=batch,
                ).consume()

    def merge_concept(self, source_id: str, target_id: str, batch_size: int) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MATCH (source:Concept {concept_id: $source_id}), (target:Concept {concept_id: $target_id}) "
                "MERGE (source)-[:EQUIVALENT_TO]->(target) "
                "SET target.aliases = CASE WHEN source.name IN coalesce(target.aliases, []) "
                "THEN coalesce(target.aliases, []) ELSE coalesce(target.aliases, []) + source.name END "
                "WITH source, target MATCH (source)-[r:IS_A]->(other:Concept) WHERE other <> target "
                "MERGE (target)-[copy:IS_A {relation: r.relation}]->(other) "
                "SET copy.reason = coalesce(r.reason, 'transferred from equivalent concept')",
                source_id=source_id,
                target_id=target_id,
            ).consume()
            session.run(
                "MATCH (source:Concept {concept_id: $source_id}), (target:Concept {concept_id: $target_id}) "
                "MATCH (source)-[r:RELATED_TO]->(other:Concept) WHERE other <> target "
                "MERGE (target)-[copy:RELATED_TO {relation: r.relation}]->(other) "
                "SET copy.reason = coalesce(r.reason, 'transferred from equivalent concept')",
                source_id=source_id,
                target_id=target_id,
            ).consume()
            session.run(
                "MATCH (source:Concept {concept_id: $source_id}), (target:Concept {concept_id: $target_id}) "
                "MATCH (source)<-[r:IS_A]-(other:Concept) WHERE other <> target "
                "MERGE (target)<-[copy:IS_A {relation: r.relation}]-(other) "
                "SET copy.reason = coalesce(r.reason, 'transferred from equivalent concept')",
                source_id=source_id,
                target_id=target_id,
            ).consume()
            session.run(
                "MATCH (source:Concept {concept_id: $source_id}), (target:Concept {concept_id: $target_id}) "
                "MATCH (source)<-[r:RELATED_TO]-(other:Concept) WHERE other <> target "
                "MERGE (target)<-[copy:RELATED_TO {relation: r.relation}]-(other) "
                "SET copy.reason = coalesce(r.reason, 'transferred from equivalent concept')",
                source_id=source_id,
                target_id=target_id,
            ).consume()
            session.run(
                "MATCH (source:Concept {concept_id: $source_id}), (target:Concept {concept_id: $target_id}) "
                "MATCH (post:Post)-[:MENTIONS]->(source) MERGE (post)-[:MENTIONS]->(target)",
                source_id=source_id,
                target_id=target_id,
            ).consume()
            session.run(
                "MATCH (source:Concept {concept_id: $source_id}) DETACH DELETE source",
                source_id=source_id,
            ).consume()

    def relate(self, source_id: str, target_id: str, relation: str, reason: str, batch_size: int) -> None:
        if relation not in {"上位", "下位", "相似"}:
            return
        relation_type = "IS_A" if relation in {"上位", "下位"} else "RELATED_TO"
        stored_relation = relation
        if relation == "上位":
            source_id, target_id = target_id, source_id
        with self.driver.session(database=self.database) as session:
            session.run(
                f"MATCH (source:Concept {{concept_id: $source_id}}), (target:Concept {{concept_id: $target_id}}) "
                f"MERGE (source)-[r:{relation_type} {{relation: $relation}}]->(target) "
                "SET r.reason = $reason, r.updated_at = datetime()",
                source_id=source_id,
                target_id=target_id,
                relation=stored_relation,
                reason=reason,
            ).consume()

    def hierarchy(self) -> tuple[list[str], list[tuple[str, str]]]:
        with self.driver.session(database=self.database) as session:
            concept_ids = [row["concept_id"] for row in session.run(
                "MATCH (c:Concept) RETURN c.concept_id AS concept_id ORDER BY concept_id"
            )]
            edges = [
                (row["source_id"], row["target_id"])
                for row in session.run(
                    "MATCH (child:Concept)-[:IS_A]->(parent:Concept) "
                    "RETURN child.concept_id AS source_id, parent.concept_id AS target_id "
                    "ORDER BY source_id, target_id"
                )
            ]
            return concept_ids, edges

    def set_granularities(self, granularities: dict[str, int], batch_size: int) -> None:
        records = [
            {"concept_id": concept_id, "granularity": granularity}
            for concept_id, granularity in sorted(granularities.items())
        ]
        with self.driver.session(database=self.database) as session:
            for index in range(0, len(records), batch_size):
                session.run(
                    "UNWIND $concepts AS concept "
                    "MATCH (c:Concept {concept_id: concept.concept_id}) "
                    "SET c.granularity = concept.granularity, c.updated_at = datetime()",
                    concepts=records[index:index + batch_size],
                ).consume()

    def export(self) -> dict[str, list[dict[str, Any]]]:
        with self.driver.session(database=self.database) as session:
            nodes = [dict(row) for row in session.run(
                "MATCH (c:Concept) RETURN c.concept_id AS concept_id, c.name AS name, c.aliases AS aliases, "
                "c.granularity AS granularity, "
                "c.embedding AS embedding, toString(c.created_at) AS created_at, toString(c.updated_at) AS updated_at "
                "ORDER BY concept_id"
            )]
            edges = [dict(row) for row in session.run(
                "MATCH (a:Concept)-[r]->(b:Concept) "
                "RETURN a.concept_id AS source_id, type(r) AS type, r.relation AS relation, "
                "b.concept_id AS target_id, r.reason AS reason ORDER BY source_id, target_id"
            )]
            posts = [dict(row) for row in session.run(
                "MATCH (p:Post) RETURN properties(p) AS properties ORDER BY p.post_id"
            )]
            return {
                "concepts": nodes,
                "posts": [row["properties"] for row in posts],
                "relationships": edges,
            }

    def close(self) -> None:
        self.driver.close()


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _neo4j_property(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list) and _is_neo4j_scalar_list(value):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _is_neo4j_scalar_list(value: list[Any]) -> bool:
    if not value:
        return True
    if any(item is None or not isinstance(item, (str, bool, int, float)) for item in value):
        return False
    types = {type(item) for item in value}
    return len(types) == 1 or types <= {int, float}
