from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class ConceptMention(BaseModel):
    name: str = Field(description="简洁规范的中文名词短语")
    source_post_ids: list[str] = Field(description="直接表达该概念的来源 post_id")


class ConceptsOutput(BaseModel):
    concepts: list[ConceptMention] = Field(description="去重后的原子语义概念及来源帖子")


class DecisionOutput(BaseModel):
    equivalent_to: str | None = Field(description="完全等价时填写候选 concept_id，否则为 null")
    target_concept_id: str | None = Field(description="关系指向的候选 concept_id；无候选时为 null")
    relation: str = Field(description="仅可为上位、下位、相似、无关")
    reason: str = Field(description="简短判断依据")


def extraction_messages(posts: list[dict[str, Any]]) -> list[tuple[str, str]]:
    payload = json.dumps(posts, ensure_ascii=False, indent=2)
    return [
        ("system", "你是中文社交媒体语义标注专家。任务目标：从帖子中提取彼此独立、可复用的原子语义主题概念。仅输出符合 schema 的 JSON。保留具体主题，不把情绪、句子或整帖当概念。"),
        ("human", f"输入字段：post_id（来源标识）、created_at（时间）、hotness（热度）、text（帖子正文）。输出字段 concepts：概念对象数组；每项包含 name（简短规范中文名词短语）和 source_post_ids（直接表达该概念的帖子 ID 数组），合并重复概念并合并来源。\n帖子数据：\n{payload}"),
    ]


def decision_messages(concept: str, candidates: list[dict[str, Any]]) -> list[tuple[str, str]]:
    payload = json.dumps(candidates, ensure_ascii=False, indent=2)
    return [
        ("system", "你是语义标签本体工程专家。严格区分完全等价与上下位/相关关系。仅输出符合 schema 的 JSON。"),
        ("human", f"判断新概念与候选概念的关系。输入字段：new_concept（新概念）；candidates（候选 concept_id、name、similarity）。输出字段：equivalent_to（仅完全同义时填候选 ID，否则 null）；target_concept_id（选最相关候选作为关系目标，无候选时 null）；relation（仅可为上位、下位、相似、无关）；reason。关系视角固定为‘新概念相对目标候选’：新概念更宽泛选上位，新概念更具体选下位，语义关联但不构成上下位选相似，无关选无关。没有候选时 equivalent_to=null、target_concept_id=null 且 relation=无关。\n输入数据：\n{json.dumps({'new_concept': concept, 'candidates': candidates}, ensure_ascii=False, indent=2)}"),
    ]


def normalize_concepts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, ConceptsOutput):
        items = value.concepts
    elif isinstance(value, dict):
        items = value.get("concepts")
    elif isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Concept extraction response must be a JSON object")
        items = parsed.get("concepts")
    else:
        content = getattr(value, "content", None)
        if content is None:
            raise ValueError("Concept extraction response has an unsupported shape")
        return normalize_concepts(content)
    if not isinstance(items, list):
        raise ValueError("LLM response field 'concepts' must be a list")
    normalized: dict[str, set[str]] = {}
    for item in items:
        if isinstance(item, str):
            name, sources = item.strip(), []
        elif isinstance(item, ConceptMention):
            name, sources = item.name.strip(), item.source_post_ids
        elif isinstance(item, dict):
            name, sources = str(item.get("name", "")).strip(), item.get("source_post_ids", [])
        else:
            raise ValueError("Each concept must be a string or an object with name and source_post_ids")
        if not isinstance(sources, list) or any(not isinstance(source, str) for source in sources):
            raise ValueError("source_post_ids must be a list of strings")
        if name:
            normalized.setdefault(name, set()).update(sources)
    return [{"name": name, "source_post_ids": sorted(sources)} for name, sources in normalized.items()]


def normalize_decision(value: Any) -> dict[str, Any]:
    if isinstance(value, DecisionOutput):
        result = value.model_dump()
    elif isinstance(value, dict):
        result = value
    elif isinstance(value, str):
        result = json.loads(value)
    else:
        content = getattr(value, "content", None)
        if content is None:
            raise ValueError("Relationship response has an unsupported shape")
        return normalize_decision(content)
    if not isinstance(result, dict):
        raise ValueError("Relationship response must be a JSON object")
    relation = result.get("relation")
    if relation not in {"上位", "下位", "相似", "无关"}:
        raise ValueError(f"Unsupported relationship: {relation!r}")
    equivalent_to = result.get("equivalent_to")
    if equivalent_to is not None and not isinstance(equivalent_to, str):
        raise ValueError("equivalent_to must be a candidate ID or null")
    return {
        "equivalent_to": equivalent_to,
        "target_concept_id": result.get("target_concept_id"),
        "relation": relation,
        "reason": str(result.get("reason", "")),
    }
