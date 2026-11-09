from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


def validate_http_url(name: str, value: str) -> str:
    parsed = urlparse(value)
    if value.startswith(("sk-", "sk_")):
        raise ValueError(
            f"{name} looks like an API key; check whether API_KEY and {name} were swapped"
        )
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be a complete http:// or https:// URL")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    base_url: str
    model: str
    temperature: float
    enable_thinking: bool
    timeout: float
    max_retries: int
    concept_extraction_concurrency: int
    concept_processing_concurrency: int
    embedding_base_url: str
    embedding_api_key: str = field(repr=False)
    embedding_model: str
    embedding_dimensions: int
    data_dir: Path
    result_dir: Path
    prompt_template_path: Path
    experiment_seeds: tuple[int, ...]
    sample_size: int
    time_bins: int
    hotness_bins: int
    similarity_top_k: int
    similarity_threshold: float
    target_granularities: tuple[int, ...]
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str = field(repr=False)
    neo4j_database: str
    neo4j_batch_size: int

    def public_dict(self) -> dict[str, object]:
        values = asdict(self)
        values.pop("api_key")
        values.pop("embedding_api_key")
        values.pop("neo4j_password")
        values["data_dir"] = str(self.data_dir)
        values["result_dir"] = str(self.result_dir)
        values["prompt_template_path"] = str(self.prompt_template_path)
        return values


def load_settings(env_file: Path | None = None) -> Settings:
    # An explicitly selected env file is the experiment configuration source.
    # Override inherited shell values so Compose and Python do not silently use
    # different credentials after a terminal variable was exported earlier.
    load_dotenv(env_file or Path(".env"), override=True)

    def required(name: str) -> str:
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(f"Missing required configuration: {name}")
        return value

    def integer(name: str, default: int, minimum: int = 1) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return value

    def number(name: str, default: float, minimum: float | None = None) -> float:
        try:
            value = float(os.getenv(name, str(default)))
        except ValueError as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return value

    def boolean(name: str, default: bool) -> bool:
        value = os.getenv(name, str(default).lower()).strip().lower()
        if value == "true":
            return True
        if value == "false":
            return False
        raise ValueError(f"{name} must be true or false")

    threshold = number("SIMILARITY_THRESHOLD", 0.35)
    if not -1 <= threshold <= 1:
        raise ValueError("SIMILARITY_THRESHOLD must be between -1 and 1")
    temperature = number("TEMPERATURE", 0.0, 0.0)
    if temperature > 2:
        raise ValueError("TEMPERATURE must be between 0 and 2")

    api_key = required("API_KEY")
    base_url = validate_http_url("BASE_URL", required("BASE_URL"))
    embedding_api_key = required("EMBEDDING_API_KEY")
    embedding_base_url = validate_http_url("EMBEDDING_BASE_URL", required("EMBEDDING_BASE_URL"))
    if api_key.startswith(("http://", "https://")):
        raise ValueError("API_KEY looks like a URL; check whether API_KEY and BASE_URL were swapped")
    if embedding_api_key.startswith(("http://", "https://")):
        raise ValueError(
            "EMBEDDING_API_KEY looks like a URL; check whether EMBEDDING_API_KEY and EMBEDDING_BASE_URL were swapped"
        )

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=required("MODEL"),
        temperature=temperature,
        enable_thinking=boolean("ENABLE_THINKING", False),
        timeout=number("TIMEOUT", 60.0, 0.1),
        max_retries=integer("MAX_RETRIES", 2, 0),
        concept_extraction_concurrency=integer("CONCEPT_EXTRACTION_CONCURRENCY", 3),
        concept_processing_concurrency=integer("CONCEPT_PROCESSING_CONCURRENCY", 3),
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        embedding_model=required("EMBEDDING_MODEL"),
        embedding_dimensions=integer("EMBEDDING_DIMENSIONS", 1536),
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        result_dir=Path(os.getenv("RESULT_DIR", "result")),
        prompt_template_path=Path(os.getenv("PROMPT_TEMPLATE_PATH", "config/prompt_templates.json")),
        experiment_seeds=parse_experiment_seeds(os.getenv("EXPERIMENT_SEEDS", "[42]")),
        sample_size=integer("SAMPLE_SIZE", 5),
        time_bins=integer("TIME_BINS", 2),
        hotness_bins=integer("HOTNESS_BINS", 2),
        similarity_top_k=integer("SIMILARITY_TOP_K", 5),
        similarity_threshold=threshold,
        target_granularities=parse_target_granularities(os.getenv("TARGET_GRANULARITIES", "[0]")),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687").strip(),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j").strip(),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "change-this-password"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j").strip(),
        neo4j_batch_size=integer("NEO4J_BATCH_SIZE", 100),
    )


def parse_experiment_seeds(value: str) -> tuple[int, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("EXPERIMENT_SEEDS must be a JSON array of integers") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("EXPERIMENT_SEEDS must be a non-empty JSON array")
    minimum = -(2 ** 63)
    maximum = 2 ** 63 - 1
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in parsed):
        raise ValueError("EXPERIMENT_SEEDS must contain only integers")
    if any(seed < minimum or seed > maximum for seed in parsed):
        raise ValueError("EXPERIMENT_SEEDS values must fit in a signed 64-bit integer")
    return tuple(parsed)


def parse_target_granularities(value: str) -> tuple[int, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("TARGET_GRANULARITIES must be a JSON array of non-negative integers") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("TARGET_GRANULARITIES must be a non-empty JSON array")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise ValueError("TARGET_GRANULARITIES must contain only integers")
    if any(item < 0 for item in parsed):
        raise ValueError("TARGET_GRANULARITIES values must be >= 0")
    return tuple(sorted(set(parsed)))
