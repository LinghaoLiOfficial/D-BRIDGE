from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from tqdm import tqdm

from .config import Settings
from .data import Post
from .prompts import DecisionOutput, ConceptsOutput, decision_messages, extraction_messages, normalize_concepts, normalize_decision
from .sampling import stratified_sample
from .storage import Concept, GraphStore


class ExperimentState(TypedDict, total=False):
    posts: list[dict[str, Any]]
    sampled_posts: list[dict[str, Any]]
    concepts: list[dict[str, Any]]
    concept_index: int
    current_concept: dict[str, Any]
    current_embedding: list[float]
    candidates: list[dict[str, Any]]
    decision: dict[str, Any]
    labels: list[dict[str, Any]]
    decisions: list[dict[str, Any]]


def _message_objects(messages: list[tuple[str, str]]) -> list[Any]:
    return [SystemMessage(content=content) if role == "system" else HumanMessage(content=content) for role, content in messages]


def build_experiment_graph(
    settings: Settings,
    chat_model: Any,
    embedding_model: Any,
    graph_store: GraphStore,
    logger: Any,
):
    progress: dict[str, tqdm] = {}

    def sample_node(state: ExperimentState) -> dict[str, Any]:
        posts = state["posts"]
        post_objects = [Post(
            post_id=item["post_id"],
            created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
            hotness=float(item["hotness"]),
            text=item["text"],
        ) for item in posts]
        sampled = [post.to_dict() for post in stratified_sample(
            post_objects, settings.sample_size, settings.time_bins, settings.hotness_bins
        )]
        logger.bind(task="分层数据抽样").info("业务输入={} 业务输出={}", json.dumps(posts, ensure_ascii=False), json.dumps(sampled, ensure_ascii=False))
        return {"sampled_posts": sampled, "labels": [], "decisions": []}

    def extract_node(state: ExperimentState) -> dict[str, Any]:
        messages = extraction_messages(state["sampled_posts"])
        started = time.perf_counter()
        logger.bind(task="原子语义概念提取").info("LLM输入={}", json.dumps(messages, ensure_ascii=False))
        try:
            structured = chat_model.with_structured_output(ConceptsOutput)
            response = structured.invoke(_message_objects(messages))
            concepts = normalize_concepts(response)
            progress["bar"] = tqdm(total=len(concepts), desc="概念处理", unit="概念")
            known_posts = {post["post_id"] for post in state["sampled_posts"]}
            for concept in concepts:
                concept["source_post_ids"] = [post_id for post_id in concept["source_post_ids"] if post_id in known_posts]
                if not concept["source_post_ids"]:
                    concept["source_post_ids"] = sorted(known_posts)
            logger.bind(task="原子语义概念提取").info(
                "LLM输出={} 耗时秒={:.3f}", json.dumps(concepts, ensure_ascii=False), time.perf_counter() - started
            )
            return {"concepts": concepts, "concept_index": 0}
        except Exception:
            logger.bind(task="原子语义概念提取").exception("LLM任务失败，耗时秒={:.3f}", time.perf_counter() - started)
            raise

    def prepare_node(state: ExperimentState) -> dict[str, Any]:
        concept = state["concepts"][state["concept_index"]]
        started = time.perf_counter()
        try:
            embedding = embedding_model.embed_query(concept["name"])
            if len(embedding) != settings.embedding_dimensions:
                raise ValueError(
                    f"Embedding has {len(embedding)} dimensions; configured EMBEDDING_DIMENSIONS={settings.embedding_dimensions}"
                )
            candidates = graph_store.candidates(embedding, settings.similarity_top_k)
            candidates = [
                item for item in candidates
                if float(item["similarity"]) >= settings.similarity_threshold
            ]
            logger.bind(task="语义相似节点检索").info(
                "业务输入={} 业务输出={} 耗时秒={:.3f}",
                json.dumps({"concept": concept["name"], "top_k": settings.similarity_top_k,
                            "threshold": settings.similarity_threshold}, ensure_ascii=False),
                json.dumps(candidates, ensure_ascii=False),
                time.perf_counter() - started,
            )
            return {"current_concept": concept, "current_embedding": embedding, "candidates": candidates}
        except Exception:
            logger.bind(task="语义相似节点检索").exception("业务步骤失败，耗时秒={:.3f}", time.perf_counter() - started)
            raise

    def reconcile_node(state: ExperimentState) -> dict[str, Any]:
        concept = state["current_concept"]
        messages = decision_messages(concept["name"], state["candidates"])
        started = time.perf_counter()
        logger.bind(task="等价判定与层级关系识别").info("LLM输入={}", json.dumps(messages, ensure_ascii=False))
        try:
            structured = chat_model.with_structured_output(DecisionOutput)
            response = structured.invoke(_message_objects(messages))
            decision = normalize_decision(response)
            candidate_ids = {item["concept_id"] for item in state["candidates"]}
            equivalent = decision["equivalent_to"]
            target = decision["target_concept_id"]
            if equivalent is not None and equivalent not in candidate_ids:
                raise ValueError(f"LLM returned unknown equivalent candidate: {equivalent}")
            if target is not None and target not in candidate_ids:
                raise ValueError(f"LLM returned unknown relationship candidate: {target}")
            if equivalent is not None:
                decision["target_concept_id"] = equivalent
            elif decision["relation"] in {"上位", "下位", "相似"} and target is None and state["candidates"]:
                target = state["candidates"][0]["concept_id"]
                decision["target_concept_id"] = target
            logger.bind(task="等价判定与层级关系识别").info(
                "LLM输出={} 耗时秒={:.3f}", json.dumps(decision, ensure_ascii=False), time.perf_counter() - started
            )
            return {"decision": decision}
        except Exception:
            logger.bind(task="等价判定与层级关系识别").exception("LLM任务失败，耗时秒={:.3f}", time.perf_counter() - started)
            raise

    def update_node(state: ExperimentState) -> dict[str, Any]:
        current = state["current_concept"]
        embedding = state["current_embedding"]
        decision = state["decision"]
        target = decision["target_concept_id"]
        concept_id = "concept_" + uuid.uuid5(uuid.NAMESPACE_URL, current["name"].casefold()).hex
        label = {"concept_id": concept_id, "name": current["name"], "source_post_ids": current["source_post_ids"]}
        started = time.perf_counter()
        try:
            if decision["equivalent_to"] and decision["equivalent_to"] != concept_id:
                target_candidate = next(item for item in state["candidates"] if item["concept_id"] == decision["equivalent_to"])
                graph_store.upsert_concept(
                    Concept(concept_id, current["name"], embedding),
                    current["source_post_ids"],
                    settings.neo4j_batch_size,
                )
                graph_store.merge_concept(concept_id, decision["equivalent_to"], settings.neo4j_batch_size)
                label = {"concept_id": decision["equivalent_to"], "name": target_candidate["name"],
                         "source_post_ids": current["source_post_ids"], "merged": True}
            else:
                concept_record = Concept(concept_id, current["name"], embedding)
                graph_store.upsert_concept(concept_record, current["source_post_ids"], settings.neo4j_batch_size)
                if target and target != concept_id and decision["relation"] in {"上位", "下位", "相似"}:
                    graph_store.relate(concept_id, target, decision["relation"], decision["reason"], settings.neo4j_batch_size)
            logger.bind(task="语义网络增量更新").info(
                "业务输入={} 业务输出={} 耗时秒={:.3f}",
                json.dumps({"concept": current, "decision": decision}, ensure_ascii=False),
                json.dumps(label, ensure_ascii=False), time.perf_counter() - started,
            )
            progress["bar"].update(1)
            return {
                "labels": [*state["labels"], label],
                "decisions": [*state["decisions"], {"concept": current["name"], **decision}],
                "concept_index": state["concept_index"] + 1,
            }
        except Exception:
            logger.bind(task="语义网络增量更新").exception("业务步骤失败，耗时秒={:.3f}", time.perf_counter() - started)
            raise

    def route(state: ExperimentState) -> str:
        return "prepare_concept" if state["concept_index"] < len(state["concepts"]) else "finish"

    def finish_node(state: ExperimentState) -> dict[str, Any]:
        if "bar" in progress:
            progress["bar"].close()
        return {}

    builder = StateGraph(ExperimentState)
    builder.add_node("stratified_sampling", sample_node)
    builder.add_node("extract_atomic_concepts", extract_node)
    builder.add_node("prepare_concept", prepare_node)
    builder.add_node("classify_equivalence_and_relation", reconcile_node)
    builder.add_node("incremental_graph_update", update_node)
    builder.add_node("finish", finish_node)
    builder.add_edge(START, "stratified_sampling")
    builder.add_edge("stratified_sampling", "extract_atomic_concepts")
    builder.add_conditional_edges("extract_atomic_concepts", route, {
        "prepare_concept": "prepare_concept", "finish": "finish",
    })
    builder.add_edge("prepare_concept", "classify_equivalence_and_relation")
    builder.add_edge("classify_equivalence_and_relation", "incremental_graph_update")
    builder.add_conditional_edges("incremental_graph_update", route, {
        "prepare_concept": "prepare_concept", "finish": "finish",
    })
    builder.add_edge("finish", END)
    return builder.compile()
