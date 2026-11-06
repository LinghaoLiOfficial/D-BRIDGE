from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

from .config import Settings
from .data import load_posts
from .storage import Neo4jStore
from .workflow import build_experiment_graph


def create_run_directory(result_root: Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S_%f")
    root = result_root / timestamp
    candidate = root
    suffix = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = result_root / f"{timestamp}_{suffix}"
            suffix += 1


def setup_run_logging(run_dir: Path) -> list[int]:
    file_sink = logger.add(
        run_dir / "experiment.log",
        level="INFO",
        encoding="utf-8",
        enqueue=True,
        serialize=True,
        backtrace=False,
        diagnose=False,
    )
    return [file_sink]


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items() if not re.search(r"(api.?key|password|secret|token)", key, re.I)}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if hasattr(value, "model_dump"):
        return _safe_json(value.model_dump())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run_experiment(
    settings: Settings,
    data_dir: Path | None = None,
    *,
    chat_model: Any | None = None,
    embedding_model: Any | None = None,
    graph_store: Any | None = None,
) -> Path:
    run_dir = create_run_directory(settings.result_dir)
    sink_ids = setup_run_logging(run_dir)
    logger.info("Run directory: {}", run_dir)
    config_snapshot = settings.public_dict()
    config_snapshot["data_dir"] = str(data_dir or settings.data_dir)
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    store = None
    try:
        posts = load_posts(data_dir or settings.data_dir)
        (run_dir / "input_posts.json").write_text(
            json.dumps([post.to_dict() for post in posts], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.bind(task="加载帖子数据").info("业务输入目录={} 业务输出帖子数={}", data_dir or settings.data_dir, len(posts))
        chat_model = chat_model or ChatOpenAI(
            api_key=settings.api_key, base_url=settings.base_url, model=settings.model,
            temperature=settings.temperature, timeout=settings.timeout, max_retries=settings.max_retries,
        )
        embedding_model = embedding_model or OpenAIEmbeddings(
            api_key=settings.embedding_api_key, base_url=settings.embedding_base_url, model=settings.embedding_model,
            dimensions=settings.embedding_dimensions, request_timeout=settings.timeout, max_retries=settings.max_retries,
        )
        store = graph_store or Neo4jStore(
            settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password,
            settings.neo4j_database, settings.embedding_dimensions,
        )
        graph = build_experiment_graph(settings, chat_model, embedding_model, store, logger)
        logger.info("LangGraph nodes: {}", ", ".join(graph.get_graph().nodes))
        final_state = graph.invoke({"posts": [post.to_dict() for post in posts]})
        graph_export = store.export()
        outputs = {
            "sampled_posts": final_state.get("sampled_posts", []),
            "concepts": final_state.get("concepts", []),
            "labels": final_state.get("labels", []),
            "decisions": final_state.get("decisions", []),
            "graph": graph_export,
        }
        (run_dir / "result.json").write_text(
            json.dumps(_safe_json(outputs), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "graph.json").write_text(
            json.dumps(_safe_json(graph_export), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Experiment completed: concepts={}, labels={}", len(outputs["concepts"]), len(outputs["labels"]))
        return run_dir
    except Exception:
        logger.exception("Experiment failed")
        raise
    finally:
        if store is not None:
            store.close()
        for sink_id in sink_ids:
            logger.remove(sink_id)
