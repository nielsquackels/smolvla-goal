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


def _check_hf_write_token() -> None:
    """Warn early if the HF token won't be able to push the final checkpoint.

    Catches two failure modes that otherwise only surface after training ends:
      1. Token is read-only.
      2. Token's namespace doesn't match `--policy.repo_id` (and isn't an org the
         user belongs to).
    """
    repo_id = next(
        (a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--policy.repo_id=")),
        None,
    )
    try:
        from huggingface_hub import HfApi
        info = HfApi().whoami()
    except Exception:
        return  # Not logged in or network unavailable — lerobot will surface it at push time.

    role = info.get("auth", {}).get("accessToken", {}).get("role", "")
    if role and role not in ("write", "admin"):
        print(
            f"\nWARNING: HF token is read-only (role={role!r})."
            " The final checkpoint push will fail.\n"
            "Run `hf auth login` with a write-scope token before starting training.\n",
            file=sys.stderr,
        )

    if repo_id and "/" in repo_id:
        target_ns = repo_id.split("/", 1)[0]
        owned = {info.get("name")} | {o.get("name") for o in info.get("orgs", [])}
        owned.discard(None)
        if target_ns not in owned:
            print(
                f"\nWARNING: --policy.repo_id namespace {target_ns!r} is not owned by"
                f" the logged-in HF user {info.get('name')!r} (orgs: {sorted(owned - {info.get('name')})})."
                " The final checkpoint push will 403 after training completes.\n",
                file=sys.stderr,
            )


if __name__ == "__main__":
    recipe = _load_recipe(_pop_recipe_path())
    _install_patches(recipe)
    _ensure_dummy_dataset_cli(recipe)
    _check_hf_write_token()
    _lerobot_train.main()
