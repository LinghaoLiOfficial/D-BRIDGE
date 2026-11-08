from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from .data import Post


@dataclass(frozen=True)
class SamplingResult:
    posts: list[Post]
    report: dict[str, Any]


def _balanced_bins(
    posts: list[Post],
    bins: int,
    value: Callable[[Post], float],
    rng: random.Random,
) -> tuple[dict[str, int], int]:
    effective_bins = min(bins, len(posts))
    if not posts:
        return {}, 0

    tied: dict[float, list[Post]] = defaultdict(list)
    for post in posts:
        tied[value(post)].append(post)

    ranked: list[Post] = []
    for item_value in sorted(tied):
        group = sorted(tied[item_value], key=lambda post: post.post_id)
        rng.shuffle(group)
        ranked.extend(group)

    assignments = {
        post.post_id: min(effective_bins - 1, index * effective_bins // len(ranked))
        for index, post in enumerate(ranked)
    }
    return assignments, effective_bins


def _allocate(
    capacities: dict[tuple[int, int], int],
    sample_size: int,
    rng: random.Random,
) -> tuple[dict[tuple[int, int], int], bool]:
    keys = sorted(capacities)
    minimum_applied = sample_size >= len(keys) and bool(keys)
    allocation = {key: 1 if minimum_applied else 0 for key in keys}
    remaining = sample_size - sum(allocation.values())
    if remaining <= 0:
        return allocation, minimum_applied

    available = {key: capacities[key] - allocation[key] for key in keys}
    total_available = sum(available.values())
    quotas = {key: remaining * available[key] / total_available for key in keys}
    floors = {key: min(available[key], math.floor(quotas[key])) for key in keys}
    for key, count in floors.items():
        allocation[key] += count

    leftover = remaining - sum(floors.values())
    tie_breakers = {key: rng.random() for key in keys}
    priority = sorted(
        keys,
        key=lambda key: (-(quotas[key] - floors[key]), tie_breakers[key], key),
    )
    for key in priority:
        if leftover == 0:
            break
        if allocation[key] < capacities[key]:
            allocation[key] += 1
            leftover -= 1
    if leftover:
        raise RuntimeError("Unable to allocate the requested sample across strata")
    return allocation, minimum_applied


def stratified_sample(
    posts: list[Post],
    sample_size: int,
    time_bins: int,
    hotness_bins: int,
    seed: int = 42,
) -> SamplingResult:
    if sample_size < 1 or time_bins < 1 or hotness_bins < 1:
        raise ValueError("sample_size, time_bins, and hotness_bins must be positive")

    rng = random.Random(seed)
    target_size = min(sample_size, len(posts))
    time_groups, effective_time_bins = _balanced_bins(
        posts, time_bins, lambda post: post.created_at.timestamp(), rng
    )
    hotness_groups, effective_hotness_bins = _balanced_bins(
        posts, hotness_bins, lambda post: post.hotness, rng
    )
    strata: dict[tuple[int, int], list[Post]] = defaultdict(list)
    for post in posts:
        strata[(time_groups[post.post_id], hotness_groups[post.post_id])].append(post)
    for group in strata.values():
        group.sort(key=lambda post: (post.created_at, post.post_id))

    capacities = {key: len(group) for key, group in strata.items()}
    allocation, minimum_applied = _allocate(capacities, target_size, rng)
    selected: list[Post] = []
    strata_report: list[dict[str, Any]] = []
    for key in sorted(strata):
        group = strata[key]
        count = allocation[key]
        sampled = rng.sample(group, count) if count else []
        sampled.sort(key=lambda post: (post.created_at, post.post_id))
        selected.extend(sampled)
        strata_report.append({
            "time_bin": key[0],
            "hotness_bin": key[1],
            "population_count": len(group),
            "population_share": round(len(group) / len(posts), 6) if posts else 0.0,
            "allocated_count": count,
            "sampled_count": len(sampled),
            "coverage_rate": round(len(sampled) / len(group), 6),
            "sample_share": round(len(sampled) / target_size, 6) if target_size else 0.0,
            "time_range": {
                "min": min(post.created_at for post in group).isoformat(),
                "max": max(post.created_at for post in group).isoformat(),
            },
            "hotness_range": {
                "min": min(post.hotness for post in group),
                "max": max(post.hotness for post in group),
            },
            "sampled_post_ids": [post.post_id for post in sampled],
        })

    selected.sort(key=lambda post: (post.created_at, post.post_id))
    covered_strata = sum(item["sampled_count"] > 0 for item in strata_report)
    report = {
        "seed": seed,
        "population_size": len(posts),
        "requested_sample_size": sample_size,
        "actual_sample_size": len(selected),
        "overall_coverage_rate": round(len(selected) / len(posts), 6) if posts else 0.0,
        "requested_time_bins": time_bins,
        "effective_time_bins": effective_time_bins,
        "requested_hotness_bins": hotness_bins,
        "effective_hotness_bins": effective_hotness_bins,
        "nonempty_strata": len(strata_report),
        "covered_strata": covered_strata,
        "stratum_coverage_rate": round(covered_strata / len(strata_report), 6) if strata_report else 0.0,
        "minimum_one_per_stratum_applied": minimum_applied,
        "strata": strata_report,
    }
    return SamplingResult(selected, report)
