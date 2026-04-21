#!/usr/bin/env python
# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Training entrypoint for SmolVLA-Goal.

Thin wrapper around LeRobot's `lerobot-train`:

1. Importing `smolvla_goal` registers `SmolVLAGoalConfig` as `smolvla_goal`.
2. A YAML recipe (default `configs/training_data.yaml`) declares the
   multi-dataset mixture with per-dataset camera maps and episodes-per-task.
3. We monkeypatch `make_dataset` inside `lerobot.scripts.lerobot_train` so
   the training loop sees our concat-of-goal-conditioned-normalized
   LeRobotDatasets. Optimizer, accelerate, wandb, and checkpointing are
   reused verbatim from upstream.

Example:
    python train.py \\
        --policy.type=smolvla_goal \\
        --policy.path=lerobot/smolvla_base \\
        --output_dir=outputs/goal_run_0 \\
        --steps=30000 \\
        --batch_size=16

The training recipe path defaults to `configs/training_data.yaml`; override
with `--smolvla-goal-config=/other/path.yaml`.
"""

import sys
from pathlib import Path

import yaml

import lerobot.scripts.lerobot_train as _lerobot_train
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import smolvla_goal  # noqa: F401 — side effect: registers SmolVLAGoalConfig
from smolvla_goal import (
    ConcatLeRobotDataset,
    GoalConditionedDataset,
    NormalizedCameraDataset,
    select_episodes_per_task,
)

_DEFAULT_RECIPE_PATH = Path(__file__).resolve().parent / "configs" / "training_data.yaml"
_RECIPE_ARG_PREFIX = "--smolvla-goal-config="


def _pop_recipe_path() -> Path:
    """Pop `--smolvla-goal-config=...` from sys.argv; return its value or default."""
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg.startswith(_RECIPE_ARG_PREFIX):
            path = Path(arg[len(_RECIPE_ARG_PREFIX) :])
            del sys.argv[i]
            return path
    return _DEFAULT_RECIPE_PATH


def _load_recipe(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Training recipe not found: {path}")
    with path.open() as f:
        recipe = yaml.safe_load(f)
    if "datasets" not in recipe or not recipe["datasets"]:
        raise ValueError(f"Recipe {path} must declare at least one dataset under `datasets`.")
    return recipe


def _build_sub_dataset(entry: dict, episodes_per_task: int, seed: int):
    """Build one LeRobotDataset → NormalizedCameraDataset → GoalConditionedDataset."""
    repo_id = entry["repo_id"]
    camera_map = entry["cameras"]

    # First load meta-only to pick episodes, then load the filtered dataset.
    # A lazy alternative is to load full then filter, but loading full decodes
    # everything — explicit filtering keeps per-dataset load cheap.
    meta_only = LeRobotDataset(repo_id)
    selected = select_episodes_per_task(meta_only.meta, episodes_per_task, seed)
    del meta_only  # release file handles before reopening with episodes filter

    base = LeRobotDataset(repo_id, episodes=selected)
    normalized = NormalizedCameraDataset(base, camera_map=camera_map)
    return GoalConditionedDataset(normalized)


_original_make_dataset = _lerobot_train.make_dataset


def _make_multi_goal_dataset(cfg):
    recipe = _make_multi_goal_dataset.recipe
    seed = recipe.get("seed", 0)
    n_per_task = recipe.get("episodes_per_task", 3)
    subs = [_build_sub_dataset(entry, n_per_task, seed) for entry in recipe["datasets"]]
    return ConcatLeRobotDataset(subs)


# Replaces `lerobot.datasets.make_dataset` (bound as
# `lerobot.scripts.lerobot_train.make_dataset` via
# `from lerobot.datasets import make_dataset` in that module). If LeRobot
# renames or reorganizes this symbol, update the target below.
_lerobot_train.make_dataset = _make_multi_goal_dataset


def _ensure_dummy_dataset_cli():
    """Draccus validates `cfg.dataset.repo_id` exists; we ignore it but must set one."""
    if not any(a.startswith("--dataset.repo_id") for a in sys.argv[1:]):
        first_repo = _make_multi_goal_dataset.recipe["datasets"][0]["repo_id"]
        sys.argv.append(f"--dataset.repo_id={first_repo}")


if __name__ == "__main__":
    recipe_path = _pop_recipe_path()
    _make_multi_goal_dataset.recipe = _load_recipe(recipe_path)
    _ensure_dummy_dataset_cli()
    _lerobot_train.main()
