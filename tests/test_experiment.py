from __future__ import annotations

import json
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import pandas as pd
from loguru import logger
from neo4j.exceptions import AuthError

from d_bridge.config import Settings, load_settings, parse_experiment_seeds, validate_http_url
from d_bridge.data import Post, load_posts
from d_bridge.prompts import (
    ConceptsOutput,
    DecisionOutput,
    decision_messages,
    extraction_messages,
    load_prompt_catalog,
    normalize_concepts,
    normalize_decision,
)
from d_bridge.randomness import initialize_random_state
from d_bridge.reporting import stability_report
from d_bridge.runtime import create_chat_model, create_embedding_model, create_run_directory, run_experiment
from d_bridge.sampling import stratified_sample
from d_bridge.storage import Concept, Neo4jStore
from d_bridge.workflow import build_experiment_graph


class FakeChatModel:
    def __init__(self, extraction, decisions, *, include_usage=True):
        self.extraction = extraction
        self.decisions = iter(decisions)
        self.calls = 0
        self.messages = []
        self.include_raw = False
        self.include_usage = include_usage
        self.lock = threading.Lock()

    def with_structured_output(self, _schema, *, include_raw=False):
        self.include_raw = include_raw
        return self

    def invoke(self, messages):
        is_extraction = any("post_id（来源标识）" in str(message.content) for message in messages)
        with self.lock:
            self.calls += 1
            call_number = self.calls
            self.messages.append(messages)
            parsed = self.extraction if is_extraction else next(self.decisions)
        if not self.include_raw:
            return parsed
        raw = SimpleNamespace(
            id=f"response-{call_number}",
            usage_metadata=(
                {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                if self.include_usage else {}
            ),
            response_metadata={"model_name": "fake-model", "system_fingerprint": "fake-fingerprint"},
        )
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


class ConcurrentChatModel:
    def __init__(self, *, fail_post_id=None):
        self.fail_post_id = fail_post_id
        self.calls = 0
        self.active_extractions = 0
        self.max_active_extractions = 0
        self.extraction_messages = []
        self.lock = threading.Lock()

    def with_structured_output(self, schema, *, include_raw=False):
        parent = self

        class StructuredCall:
            def invoke(self, messages):
                return parent._invoke(schema, messages, include_raw)

        return StructuredCall()

    def _invoke(self, schema, messages, include_raw):
        with self.lock:
            self.calls += 1
            call_number = self.calls
        if schema is ConceptsOutput:
            human = messages[-1].content
            post_id = next(post_id for post_id in ("p1", "p2", "p3") if f'"{post_id}"' in human)
            with self.lock:
                self.active_extractions += 1
                self.max_active_extractions = max(self.max_active_extractions, self.active_extractions)
                self.extraction_messages.append(human)
            try:
                time.sleep({"p1": 0.06, "p2": 0.01, "p3": 0.02}[post_id])
                if post_id == self.fail_post_id:
                    raise RuntimeError(f"failed extraction for {post_id}")
                name = {"p1": "EV 充电", "p2": " ev充电 ", "p3": "Ev充 电"}[post_id]
                parsed = {"concepts": [{"name": name, "source_post_ids": ["wrong-post"]}]}
            finally:
                with self.lock:
                    self.active_extractions -= 1
        else:
            parsed = {
                "equivalent_to": None,
                "target_concept_id": None,
                "relation": "无关",
                "reason": "无候选",
            }
        if not include_raw:
            return parsed
        raw = SimpleNamespace(
            id=f"concurrent-response-{call_number}",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            response_metadata={"model_name": "fake-model", "system_fingerprint": None},
        )
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


class FakeEmbeddings:
    def embed_query(self, _text):
        return [1.0, 0.0]


class MemoryGraphStore:
    def __init__(self):
        self.nodes = {}
        self.posts = {}
        self.relationships = []
        self.closed = False
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        self.nodes.clear()
        self.posts.clear()
        self.relationships.clear()

    def candidates(self, _embedding, _top_k):
        return [
            {"concept_id": concept.concept_id, "name": concept.name, "similarity": 0.9}
            for concept in self.nodes.values()
        ]

    def initialize_posts(self, posts, batch_size):
        assert batch_size > 0
        self.posts.update((post["post_id"], dict(post)) for post in posts)

    def upsert_concept(self, concept: Concept, source_post_ids, batch_size):
        assert batch_size > 0
        assert all(post_id in self.posts for post_id in source_post_ids)
        self.nodes.setdefault(concept.concept_id, concept)
        self.relationships.extend((post_id, "MENTIONS", concept.concept_id) for post_id in source_post_ids)

    def merge_concept(self, source_id, target_id, _batch_size):
        if source_id in self.nodes:
            del self.nodes[source_id]
        self.relationships.append((source_id, "EQUIVALENT_TO", target_id))

    def relate(self, source_id, target_id, relation, reason, _batch_size):
        self.relationships.append((source_id, relation, target_id, reason))

    def export(self):
        return {
            "concepts": list(self.nodes),
            "posts": list(self.posts.values()),
            "relationships": self.relationships,
        }

    def close(self):
        self.closed = True


class CaptureLogger:
    def __init__(self):
        self.events = []

    def bind(self, **context):
        return BoundLogger(self, context)


class BoundLogger:
    def __init__(self, parent, context):
        self.parent = parent
        self.context = context

    def info(self, message, *args):
        self.parent.events.append(("info", self.context, message.format(*args)))

    def exception(self, message, *args):
        self.parent.events.append(("exception", self.context, message.format(*args)))


def settings(**overrides):
    values = dict(
        api_key="secret-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        temperature=0.0,
        enable_thinking=False,
        timeout=1.0,
        max_retries=0,
        concept_extraction_concurrency=3,
        embedding_base_url="https://example.invalid/v1",
        embedding_api_key="secret-embedding-key",
        embedding_model="test-embedding",
        embedding_dimensions=2,
        data_dir=Path("data"),
        result_dir=Path("result"),
        prompt_template_path=Path(__file__).parents[1] / "config" / "prompt_templates.json",
        experiment_seeds=(42,),
        sample_size=5,
        time_bins=2,
        hotness_bins=2,
        similarity_top_k=5,
        similarity_threshold=0.35,
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="secret-password",
        neo4j_database="neo4j",
        neo4j_batch_size=10,
    )
    values.update(overrides)
    return Settings(**values)


def test_load_sample_json_and_validate_records(tmp_path):
    posts = load_posts(Path(__file__).parents[1] / "data")
    assert len(posts) == 20
    assert posts[0].post_id == "post-001"
    assert posts[0].created_at.tzinfo is not None

    bad_dir = tmp_path / "invalid"
    bad_dir.mkdir()
    (bad_dir / "bad.json").write_text(json.dumps({"post_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_posts(bad_dir)


def test_load_posts_preserves_extra_properties(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "posts.json").write_text(json.dumps({
        "post_id": "p1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "hotness": 4,
        "text": "城市骑行",
        "author": "用户A",
        "tags": ["出行", "城市"],
    }), encoding="utf-8")

    post = load_posts(data_dir)[0].to_dict()

    assert post["author"] == "用户A"
    assert post["tags"] == ["出行", "城市"]


def test_stratified_sample_is_deterministic_and_capped():
    posts = [
        Post(str(index), datetime(2026, 1, index + 1, tzinfo=timezone.utc), index, str(index))
        for index in range(8)
    ]
    first = stratified_sample(posts, 5, 2, 2)
    assert first == stratified_sample(posts, 5, 2, 2)
    assert len(first.posts) == 5
    capped = stratified_sample(posts[:2], 5, 2, 2)
    assert capped.posts == posts[:2]
    assert capped.report["actual_sample_size"] == 2
    assert capped.report["overall_coverage_rate"] == 1.0


def test_tied_values_are_randomized_into_balanced_quantile_bins():
    posts = [
        Post(str(index), datetime(2026, 1, 1, tzinfo=timezone.utc), 10, str(index))
        for index in range(20)
    ]

    first = stratified_sample(posts, 8, 3, 4, seed=7)
    repeated = stratified_sample(posts, 8, 3, 4, seed=7)
    other_seed = stratified_sample(posts, 8, 3, 4, seed=8)

    assert first == repeated
    assert [post.post_id for post in first.posts] != [post.post_id for post in other_seed.posts]
    time_counts = {}
    hotness_counts = {}
    for item in first.report["strata"]:
        time_counts[item["time_bin"]] = time_counts.get(item["time_bin"], 0) + item["population_count"]
        hotness_counts[item["hotness_bin"]] = hotness_counts.get(item["hotness_bin"], 0) + item["population_count"]
    assert max(time_counts.values()) - min(time_counts.values()) <= 1
    assert max(hotness_counts.values()) - min(hotness_counts.values()) <= 1


def test_proportional_allocation_covers_all_strata_before_largest_remainder():
    posts = load_posts(Path(__file__).parents[1] / "data")

    result = stratified_sample(posts, 5, 2, 2, seed=42)
    strata = result.report["strata"]

    assert sorted(item["population_count"] for item in strata) == [4, 4, 6, 6]
    assert sum(item["sampled_count"] for item in strata) == 5
    assert all(item["sampled_count"] >= 1 for item in strata)
    assert next(item for item in strata if item["sampled_count"] == 2)["population_count"] == 6
    assert result.report["minimum_one_per_stratum_applied"] is True
    assert result.report["stratum_coverage_rate"] == 1.0
    assert result.report["overall_coverage_rate"] == 0.25


def test_sample_smaller_than_strata_uses_proportional_allocation_without_minimum():
    posts = load_posts(Path(__file__).parents[1] / "data")

    result = stratified_sample(posts, 2, 2, 2, seed=42)

    assert result.report["minimum_one_per_stratum_applied"] is False
    assert result.report["actual_sample_size"] == 2
    assert result.report["covered_strata"] == 2
    assert result.report["stratum_coverage_rate"] == 0.5
    assert all(item["sampled_count"] <= item["population_count"] for item in result.report["strata"])


def test_sampling_handles_empty_input_and_more_bins_than_posts():
    empty = stratified_sample([], 5, 3, 4, seed=42)
    assert empty.posts == []
    assert empty.report["effective_time_bins"] == 0
    assert empty.report["effective_hotness_bins"] == 0
    assert empty.report["strata"] == []

    posts = [
        Post(str(index), datetime(2026, 1, index + 1, tzinfo=timezone.utc), index, str(index))
        for index in range(2)
    ]
    full = stratified_sample(posts, 10, 5, 6, seed=42)
    assert full.posts == posts
    assert full.report["effective_time_bins"] == 2
    assert full.report["effective_hotness_bins"] == 2
    assert full.report["overall_coverage_rate"] == 1.0
    assert all(item["coverage_rate"] == 1.0 for item in full.report["strata"])


@pytest.mark.parametrize("value", ["[]", "not-json", "[true]", "[1.5]", "[9223372036854775808]"])
def test_experiment_seed_validation_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="EXPERIMENT_SEEDS"):
        parse_experiment_seeds(value)


def test_experiment_seed_list_preserves_order_and_duplicates():
    assert parse_experiment_seeds("[42, -1, 42]") == (42, -1, 42)


def test_runtime_seed_reproduces_python_numpy_and_pandas_randomness():
    frame = pd.DataFrame({"value": range(10)})
    first_runtime = initialize_random_state(-7)
    first = (random.random(), np.random.random(), frame.sample(4).index.tolist())
    second_runtime = initialize_random_state(-7)
    second = (random.random(), np.random.random(), frame.sample(4).index.tolist())

    assert first == second
    assert first_runtime == second_runtime
    assert first_runtime["numpy_seed"] == (-7) % (2 ** 32)
    initialize_random_state(8)
    assert random.random() != first[0]


def test_prompt_response_normalization_and_validation():
    concepts = normalize_concepts({"concepts": [
        {"name": "共享单车", "source_post_ids": ["p1"]},
        {"name": "共享单车", "source_post_ids": ["p2"]},
    ]})
    assert concepts == [{"name": "共享单车", "source_post_ids": ["p1", "p2"]}]
    assert normalize_decision({
        "equivalent_to": "c1", "target_concept_id": "c1", "relation": "无关", "reason": "同义词"
    })["equivalent_to"] == "c1"
    with pytest.raises(ValueError, match="Unsupported relationship"):
        normalize_decision({"equivalent_to": None, "relation": "近义", "reason": ""})


def test_default_prompt_catalog_loads_and_renders_json_inputs():
    catalog = load_prompt_catalog(Path(__file__).parents[1] / "config" / "prompt_templates.json")

    extraction = extraction_messages(catalog, [{"post_id": "p1", "text": "城市骑行"}])
    decision = decision_messages(catalog, "城市骑行", [{"concept_id": "c1", "name": "骑行"}])

    assert catalog.templates["concept_extraction"].version == "1.1.0"
    assert '"post_id": "p1"' in extraction[1][1]
    assert '"new_concept": "城市骑行"' in decision[1][1]
    assert '"concept_id": "c1"' in decision[1][1]


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda value: value["templates"].pop("relation_decision"), "missing required template"),
        (lambda value: value.update({"schema_version": "2.0"}), "Unsupported prompt template schema_version"),
        (lambda value: value["templates"]["concept_extraction"].pop("version"), "version must be"),
        (lambda value: value["templates"]["concept_extraction"].pop("messages"), "messages must be"),
        (
            lambda value: value["templates"]["concept_extraction"]["messages"].update({"human": "no input"}),
            "missing required variables",
        ),
        (
            lambda value: value["templates"]["concept_extraction"]["messages"].update(
                {"human": "{posts_json} {unknown}"}
            ),
            "undeclared variables",
        ),
    ],
)
def test_prompt_catalog_rejects_invalid_templates(tmp_path, mutate, error):
    source = Path(__file__).parents[1] / "config" / "prompt_templates.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    mutate(value)
    path = tmp_path / "invalid-prompts.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_prompt_catalog(path)


def test_langgraph_runs_real_model_nodes_and_relationship_branches():
    config = settings()
    store = MemoryGraphStore()
    store.upsert_concept(Concept("existing", "城市出行", [0.0, 1.0]), [], 10)
    chat = FakeChatModel(
        {"concepts": [
            {"name": "公共交通", "source_post_ids": ["p1"]},
            {"name": "公交接驳", "source_post_ids": ["p2"]},
        ]},
        [
            {"equivalent_to": None, "target_concept_id": "existing", "relation": "相似", "reason": "公共交通相关"},
            {"equivalent_to": None, "target_concept_id": "existing", "relation": "下位", "reason": "公交接驳更具体"},
        ],
    )
    log = CaptureLogger()
    catalog = load_prompt_catalog(config.prompt_template_path)
    graph = build_experiment_graph(config, chat, FakeEmbeddings(), store, log, catalog)
    node_names = set(graph.get_graph().nodes)
    assert "extract_atomic_concepts" in node_names
    assert "classify_equivalence_and_relation" in node_names

    state = graph.invoke({"posts": [
        {"post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 4, "text": "公共交通"},
        {"post_id": "p2", "created_at": "2026-01-02T00:00:00+00:00", "hotness": 8, "text": "公交接驳"},
    ]})
    assert chat.calls == 4  # one extraction per post plus one decision per merged concept
    assert state["llm_usage"][0]["total_tokens"] == 15
    assert len(state["labels"]) == 2
    assert store.posts["p1"]["text"] == "公共交通"
    assert store.posts["p1"]["hotness"] == 4.0
    assert any(item[1] == "下位" for item in store.relationships)
    assert len([event for event in log.events if event[0] == "info"]) >= 5


def test_concurrent_per_post_extraction_merges_deterministically():
    config = settings(sample_size=3, concept_extraction_concurrency=2)
    store = MemoryGraphStore()
    chat = ConcurrentChatModel()
    catalog = load_prompt_catalog(config.prompt_template_path)
    graph = build_experiment_graph(config, chat, FakeEmbeddings(), store, CaptureLogger(), catalog)

    state = graph.invoke({"posts": [
        {"post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 1, "text": "第一篇"},
        {"post_id": "p2", "created_at": "2026-01-02T00:00:00+00:00", "hotness": 2, "text": "第二篇"},
        {"post_id": "p3", "created_at": "2026-01-03T00:00:00+00:00", "hotness": 3, "text": "第三篇"},
    ]})

    assert chat.calls == 4
    assert chat.max_active_extractions == 2
    assert len(chat.extraction_messages) == 3
    assert all(sum(f'"{post_id}"' in message for post_id in ("p1", "p2", "p3")) == 1
               for message in chat.extraction_messages)
    assert state["concepts"] == [{
        "name": "EV 充电",
        "source_post_ids": ["p1", "p2", "p3"],
    }]
    extraction_usage = [item for item in state["llm_usage"] if item["task"] == "原子语义概念提取"]
    assert [item["post_index"] for item in extraction_usage] == [0, 1, 2]
    assert [item["post_id"] for item in extraction_usage] == ["p1", "p2", "p3"]


def test_failed_per_post_extraction_preserves_completed_usage():
    config = settings(sample_size=2, concept_extraction_concurrency=1)
    store = MemoryGraphStore()
    chat = ConcurrentChatModel(fail_post_id="p2")
    usage = []
    catalog = load_prompt_catalog(config.prompt_template_path)
    graph = build_experiment_graph(
        config,
        chat,
        FakeEmbeddings(),
        store,
        CaptureLogger(),
        catalog,
        usage_collector=usage,
    )

    with pytest.raises(RuntimeError, match="failed extraction for p2"):
        graph.invoke({"posts": [
            {"post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 1, "text": "第一篇"},
            {"post_id": "p2", "created_at": "2026-01-02T00:00:00+00:00", "hotness": 2, "text": "第二篇"},
        ]})

    assert len(usage) == 1
    assert usage[0]["post_id"] == "p1"


def test_equivalent_decision_merges_into_canonical_candidate():
    config = settings()
    store = MemoryGraphStore()
    canonical = Concept("canonical", "共享单车", [1.0, 0.0])
    store.upsert_concept(canonical, [], 10)
    chat = FakeChatModel(
        {"concepts": [{"name": "共享自行车", "source_post_ids": ["p1"]}]},
        [{"equivalent_to": "canonical", "target_concept_id": "canonical", "relation": "相似", "reason": "同义"}],
    )
    catalog = load_prompt_catalog(config.prompt_template_path)
    graph = build_experiment_graph(config, chat, FakeEmbeddings(), store, CaptureLogger(), catalog)
    state = graph.invoke({"posts": [
        {"post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 4, "text": "共享自行车"}
    ]})
    assert list(store.nodes) == ["canonical"]
    assert state["labels"][0]["merged"] is True
    assert any(item[1] == "EQUIVALENT_TO" for item in store.relationships)


def test_missing_provider_usage_is_explicitly_marked_unavailable():
    config = settings()
    store = MemoryGraphStore()
    chat = FakeChatModel(
        {"concepts": [{"name": "骑行", "source_post_ids": ["p1"]}]},
        [{"equivalent_to": None, "target_concept_id": None, "relation": "无关", "reason": "无候选"}],
        include_usage=False,
    )
    catalog = load_prompt_catalog(config.prompt_template_path)
    graph = build_experiment_graph(config, chat, FakeEmbeddings(), store, CaptureLogger(), catalog)

    state = graph.invoke({"posts": [
        {"post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 4, "text": "骑行"}
    ]})

    assert state["llm_usage"][0]["available"] is False
    assert state["llm_usage"][0]["total_tokens"] is None


def test_config_snapshot_excludes_credentials_and_run_directories_never_overwrite(tmp_path):
    config = settings()
    snapshot = config.public_dict()
    assert not {"api_key", "embedding_api_key", "neo4j_password"} & snapshot.keys()
    representation = repr(config)
    assert "secret-key" not in representation
    assert "secret-embedding-key" not in representation
    assert "secret-password" not in representation
    first = create_run_directory(tmp_path, datetime(2026, 11, 6, tzinfo=timezone.utc))
    second = create_run_directory(tmp_path, datetime(2026, 11, 6, tzinfo=timezone.utc))
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_explicit_env_file_overrides_inherited_credentials(tmp_path, monkeypatch):
    configured = {
        "API_KEY": "file-chat-key",
        "BASE_URL": "https://chat.example/v1",
        "MODEL": "chat-model",
        "EMBEDDING_API_KEY": "file-embedding-key",
        "EMBEDDING_BASE_URL": "https://embedding.example/v1",
        "EMBEDDING_MODEL": "embedding-model",
        "NEO4J_PASSWORD": "file-neo4j-password",
        "ENABLE_THINKING": "TrUe",
        "CONCEPT_EXTRACTION_CONCURRENCY": "4",
    }
    for name in configured:
        monkeypatch.setenv(name, f"inherited-{name.lower()}")
    env_file = tmp_path / "experiment.env"
    env_file.write_text(
        "\n".join(f"{name}={value}" for name, value in configured.items()),
        encoding="utf-8",
    )

    loaded = load_settings(env_file)

    assert loaded.api_key == "file-chat-key"
    assert loaded.embedding_api_key == "file-embedding-key"
    assert loaded.neo4j_password == "file-neo4j-password"
    assert loaded.enable_thinking is True
    assert loaded.concept_extraction_concurrency == 4


def test_thinking_and_extraction_concurrency_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("ENABLE_THINKING", raising=False)
    monkeypatch.delenv("CONCEPT_EXTRACTION_CONCURRENCY", raising=False)
    env_file = tmp_path / "defaults.env"
    env_file.write_text(
        "\n".join([
            "API_KEY=chat-key",
            "BASE_URL=https://chat.example/v1",
            "MODEL=qwen-test",
            "EMBEDDING_API_KEY=embedding-key",
            "EMBEDDING_BASE_URL=https://embedding.example/v1",
            "EMBEDDING_MODEL=embedding-model",
        ]),
        encoding="utf-8",
    )

    loaded = load_settings(env_file)

    assert loaded.enable_thinking is False
    assert loaded.concept_extraction_concurrency == 3


@pytest.mark.parametrize("value", ["1", "yes", "", "disabled"])
def test_enable_thinking_rejects_non_boolean_values(tmp_path, value):
    env_file = tmp_path / "invalid-thinking.env"
    env_file.write_text(
        "\n".join([
            "API_KEY=chat-key",
            "BASE_URL=https://chat.example/v1",
            "MODEL=qwen-test",
            "EMBEDDING_API_KEY=embedding-key",
            "EMBEDDING_BASE_URL=https://embedding.example/v1",
            "EMBEDDING_MODEL=embedding-model",
            f"ENABLE_THINKING={value}",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ENABLE_THINKING must be true or false"):
        load_settings(env_file)


def test_concept_extraction_concurrency_must_be_positive(tmp_path):
    env_file = tmp_path / "invalid-concurrency.env"
    env_file.write_text(
        "\n".join([
            "API_KEY=chat-key",
            "BASE_URL=https://chat.example/v1",
            "MODEL=qwen-test",
            "EMBEDDING_API_KEY=embedding-key",
            "EMBEDDING_BASE_URL=https://embedding.example/v1",
            "EMBEDDING_MODEL=embedding-model",
            "ENABLE_THINKING=false",
            "CONCEPT_EXTRACTION_CONCURRENCY=0",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CONCEPT_EXTRACTION_CONCURRENCY must be >= 1"):
        load_settings(env_file)


def test_neo4j_auth_failure_has_actionable_error_and_closes_driver(monkeypatch):
    class RejectingDriver:
        closed = False

        def verify_connectivity(self):
            raise AuthError("authentication failure")

        def close(self):
            self.closed = True

    driver = RejectingDriver()
    monkeypatch.setattr("d_bridge.storage.GraphDatabase.driver", lambda *_args, **_kwargs: driver)

    with pytest.raises(RuntimeError, match="make neo4j-reset"):
        Neo4jStore("bolt://localhost:7687", "neo4j", "wrong", "neo4j", 2)

    assert driver.closed is True


def test_model_endpoint_validation_reports_swapped_or_incomplete_values():
    assert validate_http_url("BASE_URL", "https://example.com/v1") == "https://example.com/v1"
    assert validate_http_url("BASE_URL", "https://example.com/v1/") == "https://example.com/v1"
    with pytest.raises(ValueError, match="looks like an API key"):
        validate_http_url("BASE_URL", "sk-example-secret")
    with pytest.raises(ValueError, match="complete http"):
        validate_http_url("EMBEDDING_BASE_URL", "example.com/v1")


def test_embedding_client_sends_raw_strings_to_compatible_endpoint():
    model = create_embedding_model(settings())
    assert model.check_embedding_ctx_length is False


def test_chat_client_does_not_receive_python_random_seed():
    model = create_chat_model(settings(experiment_seeds=(11, 12)))
    assert model.seed is None


@pytest.mark.parametrize("enabled", [False, True])
def test_qwen_chat_client_receives_configured_thinking_mode(enabled):
    model = create_chat_model(settings(model="Qwen3.7-Flash", enable_thinking=enabled))

    assert model.extra_body == {"enable_thinking": enabled}


def test_non_qwen_client_omits_thinking_extension_and_warns_when_enabled():
    warnings = []
    sink_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        model = create_chat_model(settings(model="gpt-compatible", enable_thinking=True))
    finally:
        logger.remove(sink_id)

    assert model.extra_body is None
    assert any("was not applied" in message for message in warnings)


def test_full_run_writes_artifacts_without_credentials(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "posts.json").write_text(json.dumps([
        {"post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 4, "text": "城市骑行"}
    ]), encoding="utf-8")
    config = settings(data_dir=data_dir, result_dir=tmp_path / "results")
    store = MemoryGraphStore()
    chat = FakeChatModel(
        {"concepts": [{"name": "城市骑行", "source_post_ids": ["p1"]}]},
        [{"equivalent_to": None, "target_concept_id": None, "relation": "无关", "reason": "无候选"}],
    )

    run_dir = run_experiment(config, chat_model=chat, embedding_model=FakeEmbeddings(), graph_store=store)

    generated_files = {path.name for path in run_dir.iterdir()}
    assert {"config.json", "prompts.json", "input_posts.json", "experiment.log", "summary.json", "trials"} <= generated_files
    saved_config = (run_dir / "config.json").read_text(encoding="utf-8")
    assert "secret-key" not in saved_config
    assert "secret-embedding-key" not in saved_config
    assert "secret-password" not in saved_config
    saved_prompts = json.loads((run_dir / "prompts.json").read_text(encoding="utf-8"))
    assert saved_prompts["templates"]["concept_extraction"]["version"] == "1.1.0"
    assert "secret-key" not in json.dumps(saved_prompts)
    trial_dir = next((run_dir / "trials").iterdir())
    result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    sampling = json.loads((trial_dir / "sampling.json").read_text(encoding="utf-8"))
    assert result["sampled_posts"][0]["post_id"] == "p1"
    assert result["sampling"] == sampling
    assert sampling["population_size"] == 1
    assert sampling["actual_sample_size"] == 1
    assert sampling["overall_coverage_rate"] == 1.0
    assert sampling["strata"][0]["population_count"] == 1
    assert sampling["strata"][0]["sampled_count"] == 1
    assert len(result["labels"]) == 1
    assert result["graph"]["posts"][0]["text"] == "城市骑行"
    assert result["prompt_templates"]["snapshot_file"] == "prompts.json"
    assert result["prompt_templates"]["versions"] == {
        "concept_extraction": "1.1.0",
        "relation_decision": "1.0.0",
    }
    config_snapshot = json.loads(saved_config)
    assert config_snapshot["prompt_templates"]["source_path"] == str(config.prompt_template_path)
    assert config_snapshot["prompt_templates"]["versions"] == result["prompt_templates"]["versions"]
    assert result["trial"]["seed"] == 42
    assert result["usage"] == {
        "calls": 2,
        "available_calls": 2,
        "unavailable_calls": 0,
        "input_tokens": 20,
        "output_tokens": 10,
        "total_tokens": 30,
    }
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["stability"]["status"] == "insufficient_samples"
    assert summary["sampling"]["overall_coverage_rate"] == {
        "average": 1.0, "minimum": 1.0, "maximum": 1.0,
    }
    assert store.reset_calls == 1
    assert chat.calls == 2


def test_custom_prompt_path_is_used_by_both_model_nodes_and_snapshotted(tmp_path):
    source = Path(__file__).parents[1] / "config" / "prompt_templates.json"
    templates = json.loads(source.read_text(encoding="utf-8"))
    templates["templates"]["concept_extraction"]["version"] = "2.1.0"
    templates["templates"]["concept_extraction"]["messages"]["system"] = "CUSTOM EXTRACTION"
    templates["templates"]["relation_decision"]["version"] = "3.2.0"
    templates["templates"]["relation_decision"]["messages"]["system"] = "CUSTOM DECISION"
    prompt_path = tmp_path / "custom-prompts.json"
    prompt_path.write_text(json.dumps(templates, ensure_ascii=False), encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "posts.json").write_text(json.dumps({
        "post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 1, "text": "骑行"
    }), encoding="utf-8")
    config = settings(
        data_dir=data_dir,
        result_dir=tmp_path / "results",
        prompt_template_path=prompt_path,
    )
    chat = FakeChatModel(
        {"concepts": [{"name": "骑行", "source_post_ids": ["p1"]}]},
        [{"equivalent_to": None, "target_concept_id": None, "relation": "无关", "reason": "无候选"}],
    )

    run_dir = run_experiment(
        config,
        chat_model=chat,
        embedding_model=FakeEmbeddings(),
        graph_store=MemoryGraphStore(),
    )

    assert chat.messages[0][0].content == "CUSTOM EXTRACTION"
    assert chat.messages[1][0].content == "CUSTOM DECISION"
    snapshot = json.loads((run_dir / "prompts.json").read_text(encoding="utf-8"))
    assert snapshot == templates
    trial_dir = next((run_dir / "trials").iterdir())
    result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    assert result["prompt_templates"]["versions"] == {
        "concept_extraction": "2.1.0",
        "relation_decision": "3.2.0",
    }


def test_prompt_snapshot_survives_a_later_model_failure(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "posts.json").write_text(json.dumps({
        "post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 1, "text": "骑行"
    }), encoding="utf-8")
    config = settings(data_dir=data_dir, result_dir=tmp_path / "results")
    chat = FakeChatModel({"unexpected": []}, [])

    with pytest.raises(ValueError, match="concepts"):
        run_experiment(
            config,
            chat_model=chat,
            embedding_model=FakeEmbeddings(),
            graph_store=MemoryGraphStore(),
        )

    run_dirs = list(config.result_dir.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "prompts.json").is_file()
    assert (run_dirs[0] / "config.json").is_file()
    assert (run_dirs[0] / "summary.json").is_file()
    failed_trial = next((run_dirs[0] / "trials").iterdir())
    assert (failed_trial / "sampling.json").is_file()
    failed_sampling = json.loads((failed_trial / "sampling.json").read_text(encoding="utf-8"))
    assert failed_sampling["actual_sample_size"] == 1
    summary = json.loads((run_dirs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["sampling"]["status"] == "available"


def test_multiple_seed_trials_reset_graph_and_report_stability(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "posts.json").write_text(json.dumps({
        "post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 1, "text": "骑行"
    }), encoding="utf-8")
    config = settings(data_dir=data_dir, result_dir=tmp_path / "results", experiment_seeds=(3, 5, 3))
    store = MemoryGraphStore()
    created_seeds = []

    def chat_factory(seed):
        created_seeds.append(seed)
        return FakeChatModel(
            {"concepts": [{"name": "骑行", "source_post_ids": ["p1"]}]},
            [{"equivalent_to": None, "target_concept_id": None, "relation": "无关", "reason": "无候选"}],
        )

    run_dir = run_experiment(
        config,
        chat_model_factory=chat_factory,
        embedding_model=FakeEmbeddings(),
        graph_store=store,
    )

    assert created_seeds == [3, 5, 3]
    assert store.reset_calls == 3
    trial_dirs = sorted((run_dir / "trials").iterdir())
    assert len(trial_dirs) == 3
    sampled = [json.loads((path / "result.json").read_text(encoding="utf-8"))["sampled_posts"] for path in trial_dirs]
    assert sampled[0] == sampled[1] == sampled[2]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["stability"]["comparison_pairs"] == 3
    assert summary["stability"]["concepts"]["exact_match_rate"] == 1.0
    assert summary["usage"]["total_tokens"] == 90
    assert summary["sampling"]["overall_coverage_rate"]["average"] == 1.0
    assert summary["sampling"]["stratum_coverage_rate"]["minimum"] == 1.0
    assert all((path / "sampling.json").is_file() for path in trial_dirs)
    assert len(store.nodes) == 1


def test_failed_trial_stops_remaining_seeds_and_clears_graph(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "posts.json").write_text(json.dumps({
        "post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 1, "text": "骑行"
    }), encoding="utf-8")
    config = settings(data_dir=data_dir, result_dir=tmp_path / "results", experiment_seeds=(1, 2, 3))
    store = MemoryGraphStore()
    created_seeds = []

    def chat_factory(seed):
        created_seeds.append(seed)
        if seed == 2:
            return FakeChatModel({"unexpected": []}, [])
        return FakeChatModel(
            {"concepts": [{"name": "骑行", "source_post_ids": ["p1"]}]},
            [{"equivalent_to": None, "target_concept_id": None, "relation": "无关", "reason": "无候选"}],
        )

    with pytest.raises(ValueError, match="concepts"):
        run_experiment(
            config,
            chat_model_factory=chat_factory,
            embedding_model=FakeEmbeddings(),
            graph_store=store,
        )

    assert created_seeds == [1, 2]
    assert store.reset_calls == 3  # two trial starts plus failed-trial cleanup
    assert not store.nodes and not store.posts
    run_dir = next(config.result_dir.iterdir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_trials"] == 1
    assert summary["failed_trials"] == 1
    assert summary["sampling"]["status"] == "available"
    assert len(summary["sampling"]["trials"]) == 2
    assert len(list((run_dir / "trials").iterdir())) == 2


def test_stability_report_ignores_relationship_reason():
    left = {
        "concepts": [{"name": "骑行"}],
        "labels": [{"concept_id": "c1", "name": "骑行"}],
        "graph": {"relationships": [{
            "source_id": "c1", "type": "RELATED_TO", "relation": "相似", "target_id": "c2", "reason": "A",
        }]},
    }
    right = json.loads(json.dumps(left, ensure_ascii=False))
    right["graph"]["relationships"][0]["reason"] = "B"

    report = stability_report([(1, left), (2, right)])

    assert report["relationships"]["exact_match_rate"] == 1.0


def test_graph_store_concept_mapping_keeps_source_ids():
    concept = Concept("id-1", "概念", [0.1, 0.2])
    assert concept.concept_id == "id-1"
    assert concept.embedding == [0.1, 0.2]
