#!/usr/bin/env python
# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Training entrypoint for SmolVLA-Goal.

Thin wrapper around LeRobot's `lerobot-train`:
1. Importing `smolvla_goal` registers `SmolVLAGoalConfig` as `smolvla_goal`.
2. We monkeypatch the `make_dataset` symbol inside
   `lerobot.scripts.lerobot_train` so the training loop sees a
   `GoalConditionedDataset` wrapper (which injects `observation.goal_image.0`
   and augments `.meta`). Everything else — optimizer, accelerate, wandb,
   checkpointing, sampler — is reused verbatim from upstream.

Example:
    python train.py \\
        --policy.type=smolvla_goal \\
        --policy.path=lerobot/smolvla_base \\
        --dataset.repo_id=lerobot/svla_so101_pickplace \\
        --output_dir=outputs/goal_run_0

If `--dataset.repo_id` is not provided, defaults to `lerobot/svla_so101_pickplace`.
"""

import sys

import lerobot.scripts.lerobot_train as _lerobot_train

import smolvla_goal  # noqa: F401 — side effect: registers SmolVLAGoalConfig
from smolvla_goal import GoalConditionedDataset

_DEFAULT_DATASET_REPO_ID = "lerobot/svla_so101_pickplace"

_original_make_dataset = _lerobot_train.make_dataset


def _make_goal_dataset(cfg):
    base = _original_make_dataset(cfg)
    return GoalConditionedDataset(base)


# Replaces lerobot.datasets.make_dataset (bound as lerobot.scripts.lerobot_train.make_dataset
# via `from lerobot.datasets import make_dataset` in that module). If a LeRobot update
# breaks this — e.g. renames the symbol or moves the import — update the target below.
_lerobot_train.make_dataset = _make_goal_dataset


def _ensure_default_dataset():
    if not any(a.startswith("--dataset.repo_id") for a in sys.argv[1:]):
        sys.argv.append(f"--dataset.repo_id={_DEFAULT_DATASET_REPO_ID}")


if __name__ == "__main__":
    _ensure_default_dataset()
    _lerobot_train.main()
