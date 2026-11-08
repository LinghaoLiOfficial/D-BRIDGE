from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Hashable


def aggregate_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    available = [event for event in events if event.get("available")]

    def total(name: str) -> int | None:
        values = [event[name] for event in available if isinstance(event.get(name), int)]
        return sum(values) if values else None

    return {
        "calls": len(events),
        "available_calls": len(available),
        "unavailable_calls": len(events) - len(available),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "total_tokens": total("total_tokens"),
    }


def _concepts(output: dict[str, Any]) -> set[Hashable]:
    return {item["name"] for item in output.get("concepts", []) if item.get("name")}


def _labels(output: dict[str, Any]) -> set[Hashable]:
    return {
        (item.get("concept_id"), item.get("name"))
        for item in output.get("labels", [])
        if item.get("concept_id") or item.get("name")
    }


def _relationships(output: dict[str, Any]) -> set[Hashable]:
    relationships = output.get("graph", {}).get("relationships", [])
    normalized: set[Hashable] = set()
    for item in relationships:
        if isinstance(item, dict):
            normalized.add((
                item.get("source_id"), item.get("type"),
                item.get("relation"), item.get("target_id"),
            ))
        elif isinstance(item, (list, tuple)):
            normalized.add(tuple(item[:3]))
    return normalized


def _metric(
    trials: list[tuple[int, dict[str, Any]]],
    extractor: Callable[[dict[str, Any]], set[Hashable]],
) -> dict[str, Any]:
    pairs = []
    for (left_index, left), (right_index, right) in combinations(trials, 2):
        left_items = extractor(left)
        right_items = extractor(right)
        union = left_items | right_items
        score = len(left_items & right_items) / len(union) if union else 1.0
        pairs.append({
            "left_trial": left_index,
            "right_trial": right_index,
            "jaccard": round(score, 6),
            "exact_match": left_items == right_items,
        })
    if not pairs:
        return {
            "average_jaccard": None,
            "minimum_jaccard": None,
            "exact_match_rate": None,
            "pairs": [],
        }
    return {
        "average_jaccard": round(sum(pair["jaccard"] for pair in pairs) / len(pairs), 6),
        "minimum_jaccard": min(pair["jaccard"] for pair in pairs),
        "exact_match_rate": round(sum(pair["exact_match"] for pair in pairs) / len(pairs), 6),
        "pairs": pairs,
    }


def stability_report(trials: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    comparison_pairs = len(trials) * (len(trials) - 1) // 2
    return {
        "status": "available" if comparison_pairs else "insufficient_samples",
        "comparison_pairs": comparison_pairs,
        "concepts": _metric(trials, _concepts),
        "labels": _metric(trials, _labels),
        "relationships": _metric(trials, _relationships),
    }


def sampling_summary(reports: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    if not reports:
        return {"status": "unavailable", "trials": []}

    def statistics(name: str) -> dict[str, float]:
        values = [float(report[name]) for _, report in reports]
        return {
            "average": round(sum(values) / len(values), 6),
            "minimum": min(values),
            "maximum": max(values),
        }

    return {
        "status": "available",
        "overall_coverage_rate": statistics("overall_coverage_rate"),
        "stratum_coverage_rate": statistics("stratum_coverage_rate"),
        "trials": [
            {
                "trial_index": trial_index,
                "seed": report["seed"],
                "population_size": report["population_size"],
                "actual_sample_size": report["actual_sample_size"],
                "overall_coverage_rate": report["overall_coverage_rate"],
                "nonempty_strata": report["nonempty_strata"],
                "covered_strata": report["covered_strata"],
                "stratum_coverage_rate": report["stratum_coverage_rate"],
            }
            for trial_index, report in reports
        ],
    }
