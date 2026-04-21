# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for select_episodes_per_task.

Uses a minimal fake meta whose `episodes` is just a list of row dicts with
`tasks` and `episode_index` — matching how the real LeRobotDatasetMetadata
exposes per-episode rows.
"""

from smolvla_goal import select_episodes_per_task


class _FakeMeta:
    def __init__(self, rows):
        self.episodes = rows


def _rows(task_to_episodes: dict[str, list[int]]) -> list[dict]:
    rows = []
    for task, eps in task_to_episodes.items():
        for ep in eps:
            rows.append({"tasks": [task], "episode_index": ep})
    return rows


def test_caps_at_n_per_task():
    meta = _FakeMeta(
        _rows(
            {
                "pick_red": [0, 1, 2, 3, 4],
                "place_blue": [5, 6, 7],
                "fold_cloth": [8, 9, 10, 11],
            }
        )
    )
    selected = select_episodes_per_task(meta, n=2, seed=0)
    assert len(selected) == 2 + 2 + 2
    # Result sorted ascending.
    assert selected == sorted(selected)


def test_fewer_than_n_available_returns_all():
    meta = _FakeMeta(_rows({"only_task": [10, 11]}))
    assert select_episodes_per_task(meta, n=5, seed=0) == [10, 11]


def test_deterministic_across_seed():
    meta = _FakeMeta(_rows({"task_a": list(range(20))}))
    a = select_episodes_per_task(meta, n=3, seed=42)
    b = select_episodes_per_task(meta, n=3, seed=42)
    c = select_episodes_per_task(meta, n=3, seed=43)
    assert a == b
    assert a != c


def test_multi_task_tuple_grouping():
    # Episodes that share the same multi-task tuple group together.
    rows = [
        {"tasks": ["pick", "place"], "episode_index": 0},
        {"tasks": ["pick", "place"], "episode_index": 1},
        {"tasks": ["pick", "place"], "episode_index": 2},
        {"tasks": ["pick"], "episode_index": 3},  # different task key
    ]
    meta = _FakeMeta(rows)
    selected = select_episodes_per_task(meta, n=1, seed=0)
    # Two distinct task tuples → two episodes selected, and ep 3 must be one
    # since it's the only member of its group.
    assert len(selected) == 2
    assert 3 in selected


if __name__ == "__main__":
    test_caps_at_n_per_task()
    test_fewer_than_n_available_returns_all()
    test_deterministic_across_seed()
    test_multi_task_tuple_grouping()
    print("\nAll episode-selection tests passed.")
