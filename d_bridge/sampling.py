from __future__ import annotations

from collections import defaultdict

from .data import Post


def _rank_bins(values: list[float], bins: int) -> dict[float, int]:
    ordered = sorted(values)
    result: dict[float, int] = {}
    for value in set(values):
        rank = sum(item < value for item in ordered)
        result[value] = min(bins - 1, rank * bins // max(1, len(ordered)))
    return result


def stratified_sample(posts: list[Post], sample_size: int, time_bins: int, hotness_bins: int) -> list[Post]:
    if sample_size < 1 or time_bins < 1 or hotness_bins < 1:
        raise ValueError("sample_size, time_bins, and hotness_bins must be positive")
    if len(posts) <= sample_size:
        return sorted(posts, key=lambda post: (post.created_at, post.post_id))

    timestamps = [post.created_at.timestamp() for post in posts]
    hotness = [post.hotness for post in posts]
    time_groups = _rank_bins(timestamps, time_bins)
    hotness_groups = _rank_bins(hotness, hotness_bins)
    strata: dict[tuple[int, int], list[Post]] = defaultdict(list)
    for post in posts:
        key = (time_groups[post.created_at.timestamp()], hotness_groups[post.hotness])
        strata[key].append(post)
    for group in strata.values():
        group.sort(key=lambda post: (post.created_at, post.post_id))

    selected: list[Post] = []
    keys = sorted(strata)
    while len(selected) < sample_size:
        progressed = False
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop(0))
                progressed = True
                if len(selected) == sample_size:
                    break
        if not progressed:
            break
    return sorted(selected, key=lambda post: (post.created_at, post.post_id))
