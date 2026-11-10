from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

from pydantic import BaseModel, Field


class ConceptMention(BaseModel):
    name: str = Field(description="简洁规范、由文本直接支持的中文舆情风险概念")
    source_post_ids: list[str] = Field(description="直接支持该风险概念的来源 post_id")


class ConceptsOutput(BaseModel):
    concepts: list[ConceptMention] = Field(description="去重后的原子舆情风险概念及来源帖子")


class DecisionOutput(BaseModel):
    equivalent_to: str | None = Field(description="风险含义完全等价时填写候选 concept_id，否则为 null")
    target_concept_id: str | None = Field(
        description="唯一的关系目标候选 concept_id；优先选择可形成上下位的候选，无候选时为 null"
    )
    relation: str = Field(description="仅可为上位、下位、相似、无关；不等价时优先识别上下位")
    reason: str = Field(description="简短判断依据")


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    allowed_variables: tuple[str, ...]
    system: str
    human: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "allowed_variables": list(self.allowed_variables),
            "messages": {"system": self.system, "human": self.human},
        }


@dataclass(frozen=True)
class PromptCatalog:
    schema_version: str
    templates: dict[str, PromptTemplate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "templates": {
                name: template.to_dict()
                for name, template in self.templates.items()
            },
        }

    def version_summary(self, snapshot_file: str = "prompts.json") -> dict[str, Any]:
        return {
            "snapshot_file": snapshot_file,
            "schema_version": self.schema_version,
            "versions": {
                name: template.version
                for name, template in self.templates.items()
            },
        }


REQUIRED_TEMPLATE_VARIABLES = {
    "concept_extraction": {"posts_json"},
    "relation_decision": {"input_json"},
}
SUPPORTED_SCHEMA_VERSION = "1.0"


def _template_fields(value: str, template_name: str, message_name: str) -> set[str]:
    fields: set[str] = set()
    try:
        parsed = Formatter().parse(value)
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if not field_name or format_spec or conversion or any(char in field_name for char in (".", "[", "]")):
                raise ValueError(
                    f"Prompt template {template_name}.{message_name} contains an unsupported placeholder: {field_name!r}"
                )
            fields.add(field_name)
    except ValueError as exc:
        if str(exc).startswith("Prompt template"):
            raise
        raise ValueError(
            f"Prompt template {template_name}.{message_name} has invalid placeholder syntax"
        ) from exc
    return fields


def load_prompt_catalog(path: Path) -> PromptCatalog:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Prompt template file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prompt template file is not valid JSON: {path}: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Prompt template file root must be a JSON object")
    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("Prompt template schema_version must be a non-empty string")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported prompt template schema_version: {schema_version!r}; expected {SUPPORTED_SCHEMA_VERSION!r}"
        )
    raw_templates = raw.get("templates")
    if not isinstance(raw_templates, dict):
        raise ValueError("Prompt template file must contain a templates object")

    templates: dict[str, PromptTemplate] = {}
    for name, required_variables in REQUIRED_TEMPLATE_VARIABLES.items():
        raw_template = raw_templates.get(name)
        if not isinstance(raw_template, dict):
            raise ValueError(f"Prompt template is missing required template: {name}")
        version = raw_template.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"Prompt template {name}.version must be a non-empty string")
        allowed_variables = raw_template.get("allowed_variables")
        if (
            not isinstance(allowed_variables, list)
            or any(not isinstance(item, str) or not item for item in allowed_variables)
            or len(set(allowed_variables)) != len(allowed_variables)
        ):
            raise ValueError(f"Prompt template {name}.allowed_variables must be a unique list of strings")
        messages = raw_template.get("messages")
        if not isinstance(messages, dict):
            raise ValueError(f"Prompt template {name}.messages must be an object")
        system = messages.get("system")
        human = messages.get("human")
        if not isinstance(system, str) or not system.strip():
            raise ValueError(f"Prompt template {name}.messages.system must be a non-empty string")
        if not isinstance(human, str) or not human.strip():
            raise ValueError(f"Prompt template {name}.messages.human must be a non-empty string")

        declared = set(allowed_variables)
        used = _template_fields(system, name, "system") | _template_fields(human, name, "human")
        unknown = used - declared
        if unknown:
            raise ValueError(f"Prompt template {name} uses undeclared variables: {sorted(unknown)}")
        missing = required_variables - used
        if missing:
            raise ValueError(f"Prompt template {name} is missing required variables: {sorted(missing)}")
        unsupported = declared - required_variables
        if unsupported:
            raise ValueError(f"Prompt template {name} declares unsupported variables: {sorted(unsupported)}")
        unused = declared - used
        if unused:
            raise ValueError(f"Prompt template {name} declares unused variables: {sorted(unused)}")
        templates[name] = PromptTemplate(version.strip(), tuple(allowed_variables), system, human)

    return PromptCatalog(schema_version.strip(), templates)


def _render_template(template_name: str, template: PromptTemplate, values: dict[str, str]) -> list[tuple[str, str]]:
    expected = set(template.allowed_variables)
    received = set(values)
    if received != expected:
        raise ValueError(
            f"Prompt template {template_name} expected variables {sorted(expected)}, got {sorted(received)}"
        )
    return [
        ("system", template.system.format_map(values)),
        ("human", template.human.format_map(values)),
    ]


def extraction_messages(catalog: PromptCatalog, posts: list[dict[str, Any]]) -> list[tuple[str, str]]:
    payload = json.dumps(posts, ensure_ascii=False, indent=2)
    return _render_template("concept_extraction", catalog.templates["concept_extraction"], {"posts_json": payload})


def decision_messages(catalog: PromptCatalog, concept: str, candidates: list[dict[str, Any]]) -> list[tuple[str, str]]:
    payload = json.dumps({"new_concept": concept, "candidates": candidates}, ensure_ascii=False, indent=2)
    return _render_template("relation_decision", catalog.templates["relation_decision"], {"input_json": payload})


def normalize_concepts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, ConceptsOutput):
        items = value.concepts
    elif isinstance(value, dict):
        items = value.get("concepts")
    elif isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Risk concept extraction response must be a JSON object")
        items = parsed.get("concepts")
    else:
        content = getattr(value, "content", None)
        if content is None:
            raise ValueError("Risk concept extraction response has an unsupported shape")
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
            raise ValueError("Each risk concept must be a string or an object with name and source_post_ids")
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
            raise ValueError("Risk relationship response has an unsupported shape")
        return normalize_decision(content)
    if not isinstance(result, dict):
        raise ValueError("Risk relationship response must be a JSON object")
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
