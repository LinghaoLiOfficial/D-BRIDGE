from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    temperature: float
    timeout: float
    max_retries: int
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dimensions: int
    data_dir: Path
    result_dir: Path
    sample_size: int
    time_bins: int
    hotness_bins: int
    similarity_top_k: int
    similarity_threshold: float
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    neo4j_batch_size: int

    def public_dict(self) -> dict[str, object]:
        values = asdict(self)
        values.pop("api_key")
        values.pop("embedding_api_key")
        values.pop("neo4j_password")
        values["data_dir"] = str(self.data_dir)
        values["result_dir"] = str(self.result_dir)
        return values


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or Path(".env"), override=False)

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

    threshold = number("SIMILARITY_THRESHOLD", 0.35)
    if not -1 <= threshold <= 1:
        raise ValueError("SIMILARITY_THRESHOLD must be between -1 and 1")

    return Settings(
        api_key=required("API_KEY"),
        base_url=required("BASE_URL"),
        model=required("MODEL"),
        temperature=number("TEMPERATURE", 0.0, 0.0),
        timeout=number("TIMEOUT", 60.0, 0.1),
        max_retries=integer("MAX_RETRIES", 2, 0),
        embedding_base_url=required("EMBEDDING_BASE_URL"),
        embedding_api_key=required("EMBEDDING_API_KEY"),
        embedding_model=required("EMBEDDING_MODEL"),
        embedding_dimensions=integer("EMBEDDING_DIMENSIONS", 1536),
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        result_dir=Path(os.getenv("RESULT_DIR", "result")),
        sample_size=integer("SAMPLE_SIZE", 5),
        time_bins=integer("TIME_BINS", 2),
        hotness_bins=integer("HOTNESS_BINS", 2),
        similarity_top_k=integer("SIMILARITY_TOP_K", 5),
        similarity_threshold=threshold,
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687").strip(),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j").strip(),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "change-this-password"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j").strip(),
        neo4j_batch_size=integer("NEO4J_BATCH_SIZE", 100),
    )
