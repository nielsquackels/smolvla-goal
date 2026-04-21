# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Deterministic episode sub-selection by task.

Given a `LeRobotDatasetMetadata`, pick `n` episodes per unique task using
`random.Random(seed)` so the selection is reproducible across runs.
"""

import random
from collections import defaultdict


def select_episodes_per_task(meta, n: int, seed: int = 0) -> list[int]:
    """Return episode indices with at most `n` episodes per unique task.

    The `meta.episodes` dataset is expected to expose `tasks` (a list[str]
    per row) and `episode_index`. Grouping key is the tuple of task strings,
    so multi-task episodes stay grouped together.
    """
    by_task: dict[tuple, list[int]] = defaultdict(list)
    for row in meta.episodes:
        task_key = tuple(row["tasks"])
        by_task[task_key].append(int(row["episode_index"]))

    rng = random.Random(seed)
    selected: list[int] = []
    for task_key in sorted(by_task):  # sort for determinism across Python runs
        episodes = by_task[task_key]
        rng.shuffle(episodes)
        selected.extend(episodes[:n])
    return sorted(selected)
