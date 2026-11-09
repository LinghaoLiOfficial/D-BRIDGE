from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GranularityResult:
    granularities: dict[str, int]
    distribution: dict[str, int]
    ignored_cycle_edges: list[dict[str, str]]


def compute_granularities(
    concept_ids: Iterable[str],
    hierarchy_edges: Iterable[tuple[str, str]],
) -> GranularityResult:
    """Compute longest root-to-node depths for child-to-parent IS_A edges."""
    nodes = {str(concept_id) for concept_id in concept_ids}
    edges = {(str(child), str(parent)) for child, parent in hierarchy_edges}
    for child, parent in edges:
        nodes.add(child)
        nodes.add(parent)

    adjacency = {node: [] for node in nodes}
    for child, parent in sorted(edges):
        adjacency[child].append(parent)

    components = _strongly_connected_components(adjacency)
    component_by_node = {
        node: component_index
        for component_index, component in enumerate(components)
        for node in component
    }
    cyclic_components = {
        index for index, component in enumerate(components)
        if len(component) > 1 or any((node, node) in edges for node in component)
    }
    ignored = sorted(
        (child, parent)
        for child, parent in edges
        if component_by_node[child] == component_by_node[parent]
        and component_by_node[child] in cyclic_components
    )
    ignored_set = set(ignored)
    parents = {
        node: sorted(parent for parent in adjacency[node] if (node, parent) not in ignored_set)
        for node in nodes
    }

    granularities: dict[str, int] = {}

    def depth(node: str) -> int:
        if node not in granularities:
            granularities[node] = 1 + max((depth(parent) for parent in parents[node]), default=-1)
        return granularities[node]

    for node in sorted(nodes):
        depth(node)

    distribution: dict[str, int] = {}
    for value in granularities.values():
        key = str(value)
        distribution[key] = distribution.get(key, 0) + 1

    return GranularityResult(
        granularities=dict(sorted(granularities.items())),
        distribution={key: distribution[key] for key in sorted(distribution, key=int)},
        ignored_cycle_edges=[
            {"source_id": child, "target_id": parent}
            for child, parent in ignored
        ],
    )


def _strongly_connected_components(adjacency: dict[str, list[str]]) -> list[tuple[str, ...]]:
    index = 0
    indexes: dict[str, int] = {}
    low_links: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = index
        low_links[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in adjacency[node]:
            if target not in indexes:
                visit(target)
                low_links[node] = min(low_links[node], low_links[target])
            elif target in on_stack:
                low_links[node] = min(low_links[node], indexes[target])

        if low_links[node] != indexes[node]:
            return
        component: list[str] = []
        while True:
            target = stack.pop()
            on_stack.remove(target)
            component.append(target)
            if target == node:
                break
        components.append(tuple(sorted(component)))

    for node in sorted(adjacency):
        if node not in indexes:
            visit(node)
    return sorted(components)
