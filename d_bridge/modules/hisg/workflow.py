from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from tqdm import tqdm

from .config import Settings
from .data import Post
from .granularity import compute_granularities
from .prompts import (
    DecisionOutput,
    ConceptsOutput,
    PromptCatalog,
    decision_messages,
    extraction_messages,
    normalize_concepts,
    normalize_decision,
    resolve_decision,
)
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
    target_labels: list[dict[str, Any]]
    granularity: dict[str, Any]
    decisions: list[dict[str, Any]]
    llm_usage: list[dict[str, Any]]
    sampling_report: dict[str, Any]


def _message_objects(messages: list[tuple[str, str]]) -> list[Any]:
    return [SystemMessage(content=content) if role == "system" else HumanMessage(content=content) for role, content in messages]


def _token_value(primary: dict[str, Any], fallback: dict[str, Any], *names: str) -> int | None:
    for source in (primary, fallback):
        for name in names:
            value = source.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _response_usage(raw: Any) -> dict[str, Any]:
    usage_metadata = getattr(raw, "usage_metadata", None)
    usage_metadata = usage_metadata if isinstance(usage_metadata, dict) else {}
    response_metadata = getattr(raw, "response_metadata", None)
    response_metadata = response_metadata if isinstance(response_metadata, dict) else {}
    token_usage = response_metadata.get("token_usage")
    token_usage = token_usage if isinstance(token_usage, dict) else {}
    input_tokens = _token_value(usage_metadata, token_usage, "input_tokens", "prompt_tokens")
    output_tokens = _token_value(usage_metadata, token_usage, "output_tokens", "completion_tokens")
    total_tokens = _token_value(usage_metadata, token_usage, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "available": any(value is not None for value in (input_tokens, output_tokens, total_tokens)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "model": response_metadata.get("model_name") or response_metadata.get("model"),
        "response_id": getattr(raw, "id", None),
        "system_fingerprint": response_metadata.get("system_fingerprint"),
    }


def _invoke_structured(chat_model: Any, schema: Any, messages: list[tuple[str, str]]) -> tuple[Any, dict[str, Any]]:
    structured = chat_model.with_structured_output(schema, include_raw=True)
    response = structured.invoke(_message_objects(messages))
    if isinstance(response, dict) and "parsed" in response:
        parsing_error = response.get("parsing_error")
        if parsing_error is not None:
            raise ValueError(f"Unable to parse structured LLM response: {parsing_error}")
        return response["parsed"], _response_usage(response.get("raw"))
    return response, _response_usage(response)


def _merge_post_concepts(
    batches: list[tuple[int, str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for _, post_id, concepts in sorted(batches, key=lambda item: item[0]):
        for concept in concepts:
            name = concept["name"].strip()
            key = "".join(name.split()).casefold()
            if not key:
                continue
            if key not in merged:
                merged[key] = {"name": name, "source_post_ids": set()}
            merged[key]["source_post_ids"].add(post_id)
    return [
        {"name": item["name"], "source_post_ids": sorted(item["source_post_ids"])}
        for item in merged.values()
    ]


def build_experiment_graph(
    settings: Settings,
    chat_model: Any,
    embedding_model: Any,
    graph_store: GraphStore,
    logger: Any,
    prompt_catalog: PromptCatalog,
    trial_index: int = 1,
    seed: int = 42,
    usage_collector: list[dict[str, Any]] | None = None,
    sampling_collector: dict[str, Any] | None = None,
):
    progress: dict[str, tqdm] = {}
    collected_usage = usage_collector if usage_collector is not None else []

    def sample_node(state: ExperimentState) -> dict[str, Any]:
        posts = state["posts"]
        post_objects = [Post(
            post_id=item["post_id"],
            created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
            hotness=float(item["hotness"]),
            text=item["text"],
            extra={
                key: value for key, value in item.items()
                if key not in {"post_id", "created_at", "hotness", "text"}
            },
        ) for item in posts]
        sampling = stratified_sample(
            post_objects,
            settings.sample_size,
            settings.time_bins,
            settings.hotness_bins,
            seed=seed,
        )
        sampled = [post.to_dict() for post in sampling.posts]
        if sampling_collector is not None:
            sampling_collector.clear()
            sampling_collector.update(sampling.report)
        graph_store.initialize_posts(sampled, settings.neo4j_batch_size)
        logger.bind(task="分层数据抽样", sampling_report=sampling.report, trial_index=trial_index, seed=seed).info(
            "业务输入={} 业务输出={}",
            json.dumps(posts, ensure_ascii=False),
            json.dumps(sampled, ensure_ascii=False),
        )
        return {
            "sampled_posts": sampled,
            "sampling_report": sampling.report,
            "labels": [],
            "decisions": [],
            "llm_usage": [],
        }

    def extract_node(state: ExperimentState) -> dict[str, Any]:
        posts = state["sampled_posts"]
        started = time.perf_counter()
        extraction_progress = tqdm(total=len(posts), desc="帖子风险概念提取", unit="帖子")
        results: list[tuple[int, str, list[dict[str, Any]], dict[str, Any]]] = []

        def extract_post(post_index: int, post: dict[str, Any]):
            post_id = post["post_id"]
            messages = extraction_messages(prompt_catalog, [post])
            post_started = time.perf_counter()
            bound = logger.bind(
                task="原子舆情风险概念提取",
                trial_index=trial_index,
                seed=seed,
                post_id=post_id,
                post_index=post_index,
            )
            bound.info("LLM输入={}", json.dumps(messages, ensure_ascii=False))
            try:
                response, usage = _invoke_structured(chat_model, ConceptsOutput, messages)
                concepts = normalize_concepts(response)
                for concept in concepts:
                    concept["source_post_ids"] = [post_id]
                usage.update({
                    "task": "原子舆情风险概念提取",
                    "trial_index": trial_index,
                    "seed": seed,
                    "post_id": post_id,
                    "post_index": post_index,
                    "duration_seconds": round(time.perf_counter() - post_started, 6),
                })
                logger.bind(
                    task="原子舆情风险概念提取",
                    trial_index=trial_index,
                    seed=seed,
                    post_id=post_id,
                    post_index=post_index,
                    llm_usage=usage,
                ).info(
                    "LLM输出={} 耗时秒={:.3f}",
                    json.dumps(concepts, ensure_ascii=False),
                    time.perf_counter() - post_started,
                )
                return post_index, post_id, concepts, usage
            except Exception:
                bound.exception("LLM任务失败，耗时秒={:.3f}", time.perf_counter() - post_started)
                raise

        workers = max(1, min(settings.concept_extraction_concurrency, len(posts)))
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="concept-extraction")
        futures: dict[Future, tuple[int, str]] = {}
        try:
            futures = {
                executor.submit(extract_post, post_index, post): (post_index, post["post_id"])
                for post_index, post in enumerate(posts)
            }
            for future in as_completed(futures):
                results.append(future.result())
                extraction_progress.update(1)
        except Exception:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            completed_indexes = {item[0] for item in results}
            for future in futures:
                if future.cancelled() or not future.done() or future.exception() is not None:
                    continue
                result = future.result()
                if result[0] not in completed_indexes:
                    results.append(result)
                    completed_indexes.add(result[0])
            results.sort(key=lambda item: item[0])
            collected_usage.extend(item[3] for item in results)
            logger.bind(task="原子舆情风险概念提取", trial_index=trial_index, seed=seed).exception(
                "逐帖风险概念提取失败，耗时秒={:.3f}", time.perf_counter() - started
            )
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            extraction_progress.close()

        results.sort(key=lambda item: item[0])
        collected_usage.extend(item[3] for item in results)
        concepts = _merge_post_concepts([
            (post_index, post_id, post_concepts)
            for post_index, post_id, post_concepts, _ in results
        ])
        progress["bar"] = tqdm(total=len(concepts), desc="风险概念处理", unit="风险概念")
        logger.bind(task="原子舆情风险概念提取", trial_index=trial_index, seed=seed).info(
            "逐帖风险概念提取完成，帖子数={} 合并后风险概念数={} 总耗时秒={:.3f}",
            len(posts),
            len(concepts),
            time.perf_counter() - started,
        )
        return {"concepts": concepts, "concept_index": 0, "llm_usage": list(collected_usage)}

    def process_batch_node(state: ExperimentState) -> dict[str, Any]:
        concepts = state["concepts"]
        start_index = state["concept_index"]
        batch_end = min(start_index + settings.concept_processing_concurrency, len(concepts))
        prepared: list[dict[str, Any]] = []

        # Candidate snapshots for a batch must all observe the same graph state.
        for concept_index in range(start_index, batch_end):
            concept = concepts[concept_index]
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
                logger.bind(task="风险语义相似节点检索").info(
                    "业务输入={} 业务输出={} 耗时秒={:.3f}",
                    json.dumps({"concept": concept["name"], "top_k": settings.similarity_top_k,
                                "threshold": settings.similarity_threshold}, ensure_ascii=False),
                    json.dumps(candidates, ensure_ascii=False),
                    time.perf_counter() - started,
                )
                prepared.append({
                    "index": concept_index,
                    "concept": concept,
                    "embedding": embedding,
                    "candidates": candidates,
                })
            except Exception:
                logger.bind(task="风险语义相似节点检索").exception(
                    "业务步骤失败，耗时秒={:.3f}", time.perf_counter() - started
                )
                raise

        def classify(item: dict[str, Any]) -> dict[str, Any]:
            concept = item["concept"]
            candidates = item["candidates"]
            messages = decision_messages(prompt_catalog, concept["name"], candidates)
            started = time.perf_counter()
            bound = logger.bind(
                task="风险等价与类别层级关系识别",
                trial_index=trial_index,
                seed=seed,
                concept=concept["name"],
            )
            bound.info("LLM输入={}", json.dumps(messages, ensure_ascii=False))
            usage = None
            try:
                response, usage = _invoke_structured(chat_model, DecisionOutput, messages)
                usage.update({
                    "task": "风险等价与类别层级关系识别",
                    "trial_index": trial_index,
                    "seed": seed,
                    "concept": concept["name"],
                    "concept_index": item["index"],
                    "duration_seconds": round(time.perf_counter() - started, 6),
                })
                semantic_decision = normalize_decision(response)
                decision = resolve_decision(semantic_decision, candidates)
                logger.bind(task="风险等价与类别层级关系识别", llm_usage=usage, trial_index=trial_index, seed=seed).info(
                    "LLM语义输出={} 映射决策={} 耗时秒={:.3f}",
                    json.dumps(semantic_decision, ensure_ascii=False),
                    json.dumps(decision, ensure_ascii=False),
                    time.perf_counter() - started,
                )
                return {**item, "decision": decision, "usage": usage, "error": None}
            except Exception as exc:
                bound.exception("LLM任务失败，耗时秒={:.3f}", time.perf_counter() - started)
                if usage is not None:
                    usage["duration_seconds"] = round(time.perf_counter() - started, 6)
                return {**item, "usage": usage, "error": exc}

        classified: dict[int, dict[str, Any]] = {}
        workers = max(1, min(settings.concept_processing_concurrency, len(prepared)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="concept-processing") as executor:
            futures = {executor.submit(classify, item): item["index"] for item in prepared}
            for future in as_completed(futures):
                index = futures[future]
                classified[index] = future.result()

        ordered = [classified[index] for index in range(start_index, batch_end)]
        batch_usage = [item["usage"] for item in ordered if item["usage"] is not None]
        collected_usage.extend(batch_usage)
        failed = next((item["error"] for item in ordered if item["error"] is not None), None)
        if failed is not None:
            raise failed

        labels = list(state["labels"])
        decisions = list(state["decisions"])
        for item in ordered:
            current = item["concept"]
            embedding = item["embedding"]
            candidates = item["candidates"]
            decision = item["decision"]
            target = decision["target_concept_id"]
            concept_id = "concept_" + uuid.uuid5(uuid.NAMESPACE_URL, current["name"].casefold()).hex
            label = {"concept_id": concept_id, "name": current["name"], "source_post_ids": current["source_post_ids"]}
            started = time.perf_counter()
            try:
                if decision["equivalent_to"] and decision["equivalent_to"] != concept_id:
                    target_candidate = next(candidate for candidate in candidates if candidate["concept_id"] == decision["equivalent_to"])
                    graph_store.upsert_concept(
                        Concept(concept_id, current["name"], embedding),
                        current["source_post_ids"],
                        settings.neo4j_batch_size,
                    )
                    graph_store.merge_concept(concept_id, decision["equivalent_to"], settings.neo4j_batch_size)
                    label = {"concept_id": decision["equivalent_to"], "name": target_candidate["name"],
                             "source_post_ids": current["source_post_ids"], "merged": True}
                else:
                    graph_store.upsert_concept(
                        Concept(concept_id, current["name"], embedding),
                        current["source_post_ids"],
                        settings.neo4j_batch_size,
                    )
                    if target and target != concept_id and decision["relation"] in {"上位", "下位", "相似"}:
                        graph_store.relate(concept_id, target, decision["relation"], decision["reason"], settings.neo4j_batch_size)
                logger.bind(task="风险语义网络增量更新").info(
                    "业务输入={} 业务输出={} 耗时秒={:.3f}",
                    json.dumps({"concept": current, "decision": decision}, ensure_ascii=False),
                    json.dumps(label, ensure_ascii=False), time.perf_counter() - started,
                )
                labels.append(label)
                decisions.append({"concept": current["name"], **decision})
                progress["bar"].update(1)
            except Exception:
                logger.bind(task="风险语义网络增量更新").exception(
                    "业务步骤失败，耗时秒={:.3f}", time.perf_counter() - started
                )
                raise

        return {
            "labels": labels,
            "decisions": decisions,
            "concept_index": batch_end,
            "llm_usage": list(collected_usage),
        }

    def route(state: ExperimentState) -> str:
        return "process_batch" if state["concept_index"] < len(state["concepts"]) else "assign_granularity"

    def assign_granularity_node(state: ExperimentState) -> dict[str, Any]:
        started = time.perf_counter()
        concept_ids, hierarchy_edges = graph_store.hierarchy()
        result = compute_granularities(concept_ids, hierarchy_edges)
        graph_store.set_granularities(result.granularities, settings.neo4j_batch_size)

        labels = [
            {**label, "granularity": result.granularities[label["concept_id"]]}
            for label in state["labels"]
        ]
        canonical: dict[str, dict[str, Any]] = {}
        for label in labels:
            concept_id = label["concept_id"]
            if concept_id not in canonical:
                canonical[concept_id] = {
                    "concept_id": concept_id,
                    "name": label["name"],
                    "granularity": label["granularity"],
                    "source_post_ids": set(),
                }
            canonical[concept_id]["source_post_ids"].update(label.get("source_post_ids", []))
        target_granularities = set(settings.target_granularities)
        target_labels = [
            {
                **{key: value for key, value in canonical[concept_id].items() if key != "source_post_ids"},
                "source_post_ids": sorted(canonical[concept_id]["source_post_ids"]),
            }
            for concept_id in sorted(canonical)
            if canonical[concept_id]["granularity"] in target_granularities
        ]
        report = {
            "target_granularities": list(settings.target_granularities),
            "node_count": len(result.granularities),
            "distribution": result.distribution,
            "ignored_cycle_edge_count": len(result.ignored_cycle_edges),
            "ignored_cycle_edges": result.ignored_cycle_edges,
        }
        logger.bind(task="风险标签颗粒度标注", granularity_report=report).info(
            "业务输入={} 业务输出={} 耗时秒={:.3f}",
            json.dumps({"concept_count": len(concept_ids), "hierarchy_edge_count": len(hierarchy_edges)}, ensure_ascii=False),
            json.dumps(report, ensure_ascii=False),
            time.perf_counter() - started,
        )
        return {"labels": labels, "target_labels": target_labels, "granularity": report}

    def finish_node(state: ExperimentState) -> dict[str, Any]:
        if "bar" in progress:
            progress["bar"].close()
        return {}

    builder = StateGraph(ExperimentState)
    builder.add_node("stratified_sampling", sample_node)
    builder.add_node("extract_atomic_concepts", extract_node)
    builder.add_node("classify_equivalence_and_relation", process_batch_node)
    builder.add_node("assign_granularity", assign_granularity_node)
    builder.add_node("finish", finish_node)
    builder.add_edge(START, "stratified_sampling")
    builder.add_edge("stratified_sampling", "extract_atomic_concepts")
    builder.add_conditional_edges("extract_atomic_concepts", route, {
        "process_batch": "classify_equivalence_and_relation", "assign_granularity": "assign_granularity",
    })
    builder.add_conditional_edges("classify_equivalence_and_relation", route, {
        "process_batch": "classify_equivalence_and_relation", "assign_granularity": "assign_granularity",
    })
    builder.add_edge("assign_granularity", "finish")
    builder.add_edge("finish", END)
    return builder.compile()
