from d_bridge.granularity import compute_granularities


def test_granularity_uses_longest_path_across_roots_and_parents():
    result = compute_granularities(
        ["root-a", "root-b", "middle", "leaf", "isolated"],
        [
            ("middle", "root-a"),
            ("leaf", "middle"),
            ("leaf", "root-a"),
            ("leaf", "root-b"),
        ],
    )

    assert result.granularities == {
        "isolated": 0,
        "leaf": 2,
        "middle": 1,
        "root-a": 0,
        "root-b": 0,
    }
    assert result.distribution == {"0": 3, "1": 1, "2": 1}
    assert result.ignored_cycle_edges == []


def test_cycle_edges_are_ignored_deterministically_without_hiding_external_hierarchy():
    result = compute_granularities(
        ["root", "a", "b", "child", "self"],
        [
            ("a", "b"),
            ("b", "a"),
            ("a", "root"),
            ("child", "a"),
            ("self", "self"),
        ],
    )

    assert result.granularities == {
        "a": 1,
        "b": 0,
        "child": 2,
        "root": 0,
        "self": 0,
    }
    assert result.ignored_cycle_edges == [
        {"source_id": "a", "target_id": "b"},
        {"source_id": "b", "target_id": "a"},
        {"source_id": "self", "target_id": "self"},
    ]


def test_empty_graph_has_empty_distribution():
    result = compute_granularities([], [])

    assert result.granularities == {}
    assert result.distribution == {}
    assert result.ignored_cycle_edges == []
