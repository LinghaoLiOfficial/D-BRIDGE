from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from d_bridge.config import Settings
from d_bridge.data import Post, load_posts
from d_bridge.prompts import normalize_concepts, normalize_decision
from d_bridge.runtime import create_run_directory, run_experiment
from d_bridge.sampling import stratified_sample
from d_bridge.storage import Concept
from d_bridge.workflow import build_experiment_graph


class FakeChatModel:
    def __init__(self, extraction, decisions):
        self.extraction = extraction
        self.decisions = iter(decisions)
        self.calls = 0

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return self.extraction
        return next(self.decisions)


class FakeEmbeddings:
    def embed_query(self, _text):
        return [1.0, 0.0]


class MemoryGraphStore:
    def __init__(self):
        self.nodes = {}
        self.relationships = []
        self.closed = False

    def candidates(self, _embedding, _top_k):
        return [
            {"concept_id": concept.concept_id, "name": concept.name, "similarity": 0.9}
            for concept in self.nodes.values()
        ]

    def upsert_concept(self, concept: Concept, source_post_ids, batch_size):
        assert batch_size > 0
        self.nodes.setdefault(concept.concept_id, concept)
        self.relationships.extend((post_id, "MENTIONS", concept.concept_id) for post_id in source_post_ids)

    def merge_concept(self, source_id, target_id, _batch_size):
        if source_id in self.nodes:
            del self.nodes[source_id]
        self.relationships.append((source_id, "EQUIVALENT_TO", target_id))

    def relate(self, source_id, target_id, relation, reason, _batch_size):
        self.relationships.append((source_id, relation, target_id, reason))

    def export(self):
        return {"concepts": list(self.nodes), "relationships": self.relationships}

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
        timeout=1.0,
        max_retries=0,
        embedding_base_url="https://example.invalid/v1",
        embedding_api_key="secret-embedding-key",
        embedding_model="test-embedding",
        embedding_dimensions=2,
        data_dir=Path("data"),
        result_dir=Path("result"),
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
    assert len(posts) == 5
    assert posts[0].post_id == "post-001"
    assert posts[0].created_at.tzinfo is not None

    bad_dir = tmp_path / "invalid"
    bad_dir.mkdir()
    (bad_dir / "bad.json").write_text(json.dumps({"post_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_posts(bad_dir)


def test_stratified_sample_is_deterministic_and_capped():
    posts = [
        Post(str(index), datetime(2026, 1, index + 1, tzinfo=timezone.utc), index, str(index))
        for index in range(8)
    ]
    first = stratified_sample(posts, 5, 2, 2)
    assert first == stratified_sample(posts, 5, 2, 2)
    assert len(first) == 5
    assert stratified_sample(posts[:2], 5, 2, 2) == posts[:2]


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
    graph = build_experiment_graph(config, chat, FakeEmbeddings(), store, log)
    node_names = set(graph.get_graph().nodes)
    assert "extract_atomic_concepts" in node_names
    assert "classify_equivalence_and_relation" in node_names

    state = graph.invoke({"posts": [
        {"post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 4, "text": "公共交通"},
        {"post_id": "p2", "created_at": "2026-01-02T00:00:00+00:00", "hotness": 8, "text": "公交接驳"},
    ]})
    assert chat.calls == 3  # extraction plus one real chat-model call per concept
    assert len(state["labels"]) == 2
    assert any(item[1] == "下位" for item in store.relationships)
    assert len([event for event in log.events if event[0] == "info"]) >= 5


def test_equivalent_decision_merges_into_canonical_candidate():
    config = settings()
    store = MemoryGraphStore()
    canonical = Concept("canonical", "共享单车", [1.0, 0.0])
    store.upsert_concept(canonical, [], 10)
    chat = FakeChatModel(
        {"concepts": [{"name": "共享自行车", "source_post_ids": ["p1"]}]},
        [{"equivalent_to": "canonical", "target_concept_id": "canonical", "relation": "相似", "reason": "同义"}],
    )
    graph = build_experiment_graph(config, chat, FakeEmbeddings(), store, CaptureLogger())
    state = graph.invoke({"posts": [
        {"post_id": "p1", "created_at": "2026-01-01T00:00:00+00:00", "hotness": 4, "text": "共享自行车"}
    ]})
    assert list(store.nodes) == ["canonical"]
    assert state["labels"][0]["merged"] is True
    assert any(item[1] == "EQUIVALENT_TO" for item in store.relationships)


def test_config_snapshot_excludes_credentials_and_run_directories_never_overwrite(tmp_path):
    config = settings()
    snapshot = config.public_dict()
    assert not {"api_key", "embedding_api_key", "neo4j_password"} & snapshot.keys()
    first = create_run_directory(tmp_path, datetime(2026, 11, 6, tzinfo=timezone.utc))
    second = create_run_directory(tmp_path, datetime(2026, 11, 6, tzinfo=timezone.utc))
    assert first != second
    assert first.is_dir() and second.is_dir()


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

    assert {"config.json", "input_posts.json", "experiment.log", "result.json", "graph.json"} <= {
        path.name for path in run_dir.iterdir()
    }
    saved_config = (run_dir / "config.json").read_text(encoding="utf-8")
    assert "secret-key" not in saved_config
    assert "secret-embedding-key" not in saved_config
    assert "secret-password" not in saved_config
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["sampled_posts"][0]["post_id"] == "p1"
    assert len(result["labels"]) == 1
    assert chat.calls == 2


def test_graph_store_concept_mapping_keeps_source_ids():
    concept = Concept("id-1", "概念", [0.1, 0.2])
    assert concept.concept_id == "id-1"
    assert concept.embedding == [0.1, 0.2]
