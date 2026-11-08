from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Post:
    post_id: str
    created_at: datetime
    hotness: float
    text: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "post_id": self.post_id,
            "created_at": self.created_at.isoformat(),
            "hotness": self.hotness,
            "text": self.text,
        }


def load_posts(data_dir: Path) -> list[Post]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    posts: list[Post] = []
    seen: set[str] = set()
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON data files found in {data_dir}")
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read JSON data file {path}: {exc}") from exc
        entries = raw if isinstance(raw, list) else [raw]
        for index, item in enumerate(entries):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: item {index} must be an object")
            missing = {"post_id", "created_at", "hotness", "text"} - item.keys()
            if missing:
                raise ValueError(f"{path}: item {index} missing {', '.join(sorted(missing))}")
            post_id = str(item["post_id"]).strip()
            if not post_id or post_id in seen:
                raise ValueError(f"{path}: post_id must be non-empty and unique: {post_id!r}")
            try:
                created_at = datetime.fromisoformat(str(item["created_at"]).replace("Z", "+00:00"))
                hotness = float(item["hotness"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid created_at or hotness for {post_id}") from exc
            if not str(item["text"]).strip():
                raise ValueError(f"{path}: text must be non-empty for {post_id}")
            extra = {
                key: value for key, value in item.items()
                if key not in {"post_id", "created_at", "hotness", "text"}
            }
            seen.add(post_id)
            posts.append(Post(post_id, created_at, hotness, str(item["text"]).strip(), extra))
    return posts
