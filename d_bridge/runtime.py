from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

from .config import Settings
from .data import load_posts
from .prompts import load_prompt_catalog
from .randomness import initialize_random_state
from .reporting import aggregate_usage, sampling_summary, stability_report
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


def _is_sensitive_key(key: str) -> bool:
    return bool(re.search(r"api.?key|password|secret|access.?token|auth.?token|bearer", key, re.I))


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items() if not _is_sensitive_key(str(key))}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if hasattr(value, "model_dump"):
        return _safe_json(value.model_dump())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_safe_json(value), ensure_ascii=False, indent=2), encoding="utf-8")


def create_embedding_model(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        request_timeout=settings.timeout,
        max_retries=settings.max_retries,
        # Alibaba's OpenAI-compatible endpoint accepts string arrays, not token ID arrays.
        check_embedding_ctx_length=False,
    )


def create_chat_model(settings: Settings) -> ChatOpenAI:
    options: dict[str, Any] = {}
    if "qwen" in settings.model.casefold():
        options["extra_body"] = {"enable_thinking": settings.enable_thinking}
    elif settings.enable_thinking:
        logger.warning(
            "ENABLE_THINKING=true was not applied because model {!r} is not recognized as Qwen",
            settings.model,
        )
    return ChatOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        temperature=settings.temperature,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        **options,
    )


def _trial_directory(parent: Path, trial_index: int, seed: int) -> Path:
    seed_label = str(seed) if seed >= 0 else f"neg{abs(seed)}"
    path = parent / "trials" / f"trial-{trial_index:03d}-seed-{seed_label}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _summary(
    status: str,
    seeds: tuple[int, ...],
    trials: list[dict[str, Any]],
    completed_outputs: list[tuple[int, dict[str, Any]]],
    usage_events: list[dict[str, Any]],
    sampling_reports: list[tuple[int, dict[str, Any]]],
    started: float,
) -> dict[str, Any]:
    return {
        "status": status,
        "requested_trials": len(seeds),
        "completed_trials": sum(item["status"] == "completed" for item in trials),
        "failed_trials": sum(item["status"] == "failed" for item in trials),
        "experiment_seeds": list(seeds),
        "duration_seconds": round(time.perf_counter() - started, 6),
        "usage": aggregate_usage(usage_events),
        "sampling": sampling_summary(sampling_reports),
        "stability": stability_report(completed_outputs),
        "trials": trials,
    }


def run_experiment(
    settings: Settings,
    data_dir: Path | None = None,
    *,
    chat_model: Any | None = None,
    embedding_model: Any | None = None,
    graph_store: Any | None = None,
    chat_model_factory: Callable[[int], Any] | None = None,
    embedding_model_factory: Callable[[], Any] | None = None,
    graph_store_factory: Callable[[], Any] | None = None,
) -> Path:
    if chat_model is not None and chat_model_factory is not None:
        raise ValueError("Provide chat_model or chat_model_factory, not both")
    if embedding_model is not None and embedding_model_factory is not None:
        raise ValueError("Provide embedding_model or embedding_model_factory, not both")
    if graph_store is not None and graph_store_factory is not None:
        raise ValueError("Provide graph_store or graph_store_factory, not both")

    batch_started = time.perf_counter()
    run_dir = create_run_directory(settings.result_dir)
    root_sink_ids = setup_run_logging(run_dir)
    logger.info("Batch directory: {}", run_dir)
    store = None
    trials: list[dict[str, Any]] = []
    completed_outputs: list[tuple[int, dict[str, Any]]] = []
    batch_usage_events: list[dict[str, Any]] = []
    batch_sampling_reports: list[tuple[int, dict[str, Any]]] = []

    try:
        prompt_catalog = load_prompt_catalog(settings.prompt_template_path)
        prompt_summary = prompt_catalog.version_summary()
        posts = load_posts(data_dir or settings.data_dir)
        serialized_posts = [post.to_dict() for post in posts]
        _write_json(run_dir / "prompts.json", prompt_catalog.to_dict())
        _write_json(run_dir / "input_posts.json", serialized_posts)

        config_snapshot = settings.public_dict()
        config_snapshot["data_dir"] = str(data_dir or settings.data_dir)
        config_snapshot["prompt_templates"] = {
            "source_path": str(settings.prompt_template_path),
            **prompt_summary,
        }
        config_snapshot["input_snapshot"] = {
            "file": "input_posts.json",
            "post_count": len(serialized_posts),
        }
        _write_json(run_dir / "config.json", config_snapshot)
        logger.info("Prompt templates loaded: {}", json.dumps(prompt_summary, ensure_ascii=False))
        logger.bind(task="加载帖子数据").info(
            "业务输入目录={} 业务输出帖子数={}", data_dir or settings.data_dir, len(posts)
        )

        embedding_model = embedding_model or (
            embedding_model_factory() if embedding_model_factory else create_embedding_model(settings)
        )
        store = graph_store or (
            graph_store_factory() if graph_store_factory else Neo4jStore(
                settings.neo4j_uri,
                settings.neo4j_username,
                settings.neo4j_password,
                settings.neo4j_database,
                settings.embedding_dimensions,
            )
        )

        for trial_index, seed in enumerate(settings.experiment_seeds, start=1):
            trial_started = time.perf_counter()
            trial_dir = _trial_directory(run_dir, trial_index, seed)
            trial_sink_ids = setup_run_logging(trial_dir)
            runtime_random = initialize_random_state(seed)
            trial_config = {
                **config_snapshot,
                "trial": {
                    "index": trial_index,
                    "seed": seed,
                    "temperature": settings.temperature,
                    "runtime_random": runtime_random,
                },
            }
            _write_json(trial_dir / "config.json", trial_config)
            logger.bind(trial_index=trial_index, seed=seed).info("Trial started")
            trial_usage_events: list[dict[str, Any]] = []
            trial_sampling_report: dict[str, Any] = {}
            sampling_recorded = False
            try:
                store.reset()
                current_chat_model = chat_model or (
                    chat_model_factory(seed) if chat_model_factory else create_chat_model(settings)
                )
                graph = build_experiment_graph(
                    settings,
                    current_chat_model,
                    embedding_model,
                    store,
                    logger,
                    prompt_catalog,
                    trial_index=trial_index,
                    seed=seed,
                    usage_collector=trial_usage_events,
                    sampling_collector=trial_sampling_report,
                )
                logger.bind(trial_index=trial_index, seed=seed).info(
                    "LangGraph nodes: {}", ", ".join(graph.get_graph().nodes)
                )
                final_state = graph.invoke({"posts": serialized_posts})
                if trial_sampling_report:
                    batch_sampling_reports.append((trial_index, dict(trial_sampling_report)))
                    sampling_recorded = True
                graph_export = store.export()
                usage_events = trial_usage_events
                output = {
                    "trial": {
                        "index": trial_index,
                        "seed": seed,
                        "temperature": settings.temperature,
                        "runtime_random": runtime_random,
                    },
                    "sampled_posts": final_state.get("sampled_posts", []),
                    "sampling": trial_sampling_report,
                    "concepts": final_state.get("concepts", []),
                    "labels": final_state.get("labels", []),
                    "target_labels": final_state.get("target_labels", []),
                    "granularity": final_state.get("granularity", {}),
                    "decisions": final_state.get("decisions", []),
                    "graph": graph_export,
                    "prompt_templates": prompt_summary,
                    "llm_usage": usage_events,
                    "usage": aggregate_usage(usage_events),
                }
                _write_json(trial_dir / "result.json", output)
                _write_json(trial_dir / "graph.json", graph_export)
                _write_json(trial_dir / "sampling.json", trial_sampling_report)
                batch_usage_events.extend(usage_events)
                completed_outputs.append((trial_index, output))
                trials.append({
                    "index": trial_index,
                    "seed": seed,
                    "status": "completed",
                    "directory": str(trial_dir.relative_to(run_dir)),
                    "duration_seconds": round(time.perf_counter() - trial_started, 6),
                    "usage": output["usage"],
                    "sampling": {
                        "overall_coverage_rate": trial_sampling_report["overall_coverage_rate"],
                        "stratum_coverage_rate": trial_sampling_report["stratum_coverage_rate"],
                    },
                })
                logger.bind(trial_index=trial_index, seed=seed).info(
                    "Trial completed: concepts={}, labels={}, target_labels={}",
                    len(output["concepts"]), len(output["labels"]), len(output["target_labels"])
                )
            except Exception as exc:
                batch_usage_events.extend(trial_usage_events)
                if trial_sampling_report and not sampling_recorded:
                    batch_sampling_reports.append((trial_index, dict(trial_sampling_report)))
                if trial_sampling_report:
                    _write_json(trial_dir / "sampling.json", trial_sampling_report)
                logger.bind(trial_index=trial_index, seed=seed).exception("Trial failed")
                try:
                    _write_json(trial_dir / "graph.json", store.export())
                except Exception as export_exc:
                    logger.bind(trial_index=trial_index, seed=seed).warning(
                        "Unable to export partial graph: {}", export_exc
                    )
                error = {"type": type(exc).__name__, "message": str(exc)}
                _write_json(trial_dir / "error.json", {
                    "error": error,
                    "llm_usage": trial_usage_events,
                    "usage": aggregate_usage(trial_usage_events),
                    "sampling": trial_sampling_report or None,
                })
                trials.append({
                    "index": trial_index,
                    "seed": seed,
                    "status": "failed",
                    "directory": str(trial_dir.relative_to(run_dir)),
                    "duration_seconds": round(time.perf_counter() - trial_started, 6),
                    "error": error,
                    "usage": aggregate_usage(trial_usage_events),
                    "sampling": ({
                        "overall_coverage_rate": trial_sampling_report["overall_coverage_rate"],
                        "stratum_coverage_rate": trial_sampling_report["stratum_coverage_rate"],
                    } if trial_sampling_report else None),
                })
                try:
                    store.reset()
                except Exception:
                    logger.bind(trial_index=trial_index, seed=seed).exception(
                        "Unable to clear graph after failed trial"
                    )
                _write_json(
                    run_dir / "summary.json",
                    _summary(
                        "failed", settings.experiment_seeds, trials, completed_outputs,
                        batch_usage_events, batch_sampling_reports, batch_started,
                    ),
                )
                raise
            finally:
                for sink_id in trial_sink_ids:
                    logger.remove(sink_id)

        _write_json(
            run_dir / "summary.json",
            _summary(
                "completed", settings.experiment_seeds, trials, completed_outputs,
                batch_usage_events, batch_sampling_reports, batch_started,
            ),
        )
        logger.info("Experiment batch completed: trials={}", len(trials))
        return run_dir
    except Exception:
        logger.exception("Experiment batch failed")
        raise
    finally:
        if store is not None:
            store.close()
        for sink_id in root_sink_ids:
            logger.remove(sink_id)
