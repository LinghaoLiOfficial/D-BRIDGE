from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from neo4j import Driver, GraphDatabase


@dataclass(frozen=True)
class Concept:
    concept_id: str
    name: str
    embedding: list[float]


class GraphStore(Protocol):
    def candidates(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]: ...
    def upsert_concept(self, concept: Concept, source_post_ids: list[str], batch_size: int) -> None: ...
    def merge_concept(self, source_id: str, target_id: str, batch_size: int) -> None: ...
    def relate(self, source_id: str, target_id: str, relation: str, reason: str, batch_size: int) -> None: ...
    def export(self) -> dict[str, list[dict[str, Any]]]: ...
    def close(self) -> None: ...


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str, database: str, embedding_dimensions: int):
        self.driver: Driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.embedding_dimensions = embedding_dimensions
        self.driver.verify_connectivity()
        with self.driver.session(database=database) as session:
            session.run("CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.post_id IS UNIQUE").consume()
            session.run(
                "CREATE VECTOR INDEX concept_embedding IF NOT EXISTS FOR (c:Concept) ON (c.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dimensions}, `vector.similarity_function`: 'cosine'}}}}"
            ).consume()

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
                    "MERGE (p:Post {post_id: post_id}) "
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

    def export(self) -> dict[str, list[dict[str, Any]]]:
        with self.driver.session(database=self.database) as session:
            nodes = [dict(row) for row in session.run(
                "MATCH (c:Concept) RETURN c.concept_id AS concept_id, c.name AS name, c.aliases AS aliases, "
                "c.embedding AS embedding, toString(c.created_at) AS created_at, toString(c.updated_at) AS updated_at "
                "ORDER BY concept_id"
            )]
            edges = [dict(row) for row in session.run(
                "MATCH (a:Concept)-[r]->(b:Concept) "
                "RETURN a.concept_id AS source_id, type(r) AS type, r.relation AS relation, "
                "b.concept_id AS target_id, r.reason AS reason ORDER BY source_id, target_id"
            )]
            return {"concepts": nodes, "relationships": edges}

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
