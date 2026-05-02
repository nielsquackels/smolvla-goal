#!/usr/bin/env python
# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Training entrypoint for SmolVLA-Goal.

Thin wrapper around LeRobot's `lerobot-train`:

1. Importing `smolvla_goal` registers `SmolVLAGoalConfig` as `smolvla_goal`.
2. A YAML recipe (default `configs/training_data.yaml`) declares the
   multi-dataset mixture with per-dataset camera maps and episodes-per-task.
3. `smolvla_goal._train_patches.install` monkeypatches `make_dataset` and
   `update_policy` inside `lerobot.scripts.lerobot_train`. Optimizer,
   accelerate, wandb, and checkpointing are reused verbatim from upstream.

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

import smolvla_goal  # noqa: F401 — side effect: registers SmolVLAGoalConfig
from smolvla_goal._train_patches import install as _install_patches

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


def _ensure_dummy_dataset_cli(recipe: dict) -> None:
    """Draccus validates `cfg.dataset.repo_id` exists; we ignore it but must set one."""
    if not any(a.startswith("--dataset.repo_id") for a in sys.argv[1:]):
        sys.argv.append(f"--dataset.repo_id={recipe['datasets'][0]['repo_id']}")


if __name__ == "__main__":
    recipe = _load_recipe(_pop_recipe_path())
    _install_patches(recipe)
    _ensure_dummy_dataset_cli(recipe)
    _lerobot_train.main()
