# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkeypatches applied to `lerobot.scripts.lerobot_train`.

Two patches:
- `make_dataset` is replaced with a builder that constructs our concat-of-
  goal-conditioned-normalized LeRobotDatasets from a YAML recipe.
- `update_policy` is wrapped to log `goal_type_embedding` weight/grad norms
  to WandB.
"""

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import lerobot.scripts.lerobot_train as _lerobot_train

from .camera_normalize import NormalizedCameraDataset
from .concat_dataset import ConcatLeRobotDataset
from .episode_selection import select_episodes_per_task
from .goal_dataset import GoalConditionedDataset


class _EpisodeByValueLookup:
    """Proxy that makes `episodes[ep_idx]` look up by episode_index VALUE.

    LeRobot's DatasetReader and DatasetMetadata do `meta.episodes[ep_idx]`,
    which is positional indexing into the underlying HF Dataset. That
    silently assumes episode rows are stored in `[0, 1, ..., N-1]` order;
    several community datasets and v2.1→v3.0 conversions break that
    assumption, returning the wrong episode's `dataset_from/to_index` →
    `KeyError` deep inside the dataloader.

    Negative ints, slices, and column-name strings pass through unchanged.
    """

    def __init__(self, hf_dataset):
        self._inner = hf_dataset
        self._pos_by_value = {int(v): i for i, v in enumerate(hf_dataset["episode_index"])}

    def __getitem__(self, key):
        if isinstance(key, int) and key >= 0 and key in self._pos_by_value:
            return self._inner[self._pos_by_value[key]]
        return self._inner[key]

    def __len__(self):
        return len(self._inner)

    def __iter__(self):
        return iter(self._inner)

    def __contains__(self, item):
        return item in self._inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _build_sub_dataset(entry: dict, episodes_per_task: int, seed: int, policy_cfg):
    """Build one LeRobotDataset → NormalizedCameraDataset → GoalConditionedDataset."""
    repo_id = entry["repo_id"]
    camera_map = entry["cameras"]

    # First load meta-only to pick episodes, then load the filtered dataset.
    # A lazy alternative is to load full then filter, but loading full decodes
    # everything — explicit filtering keeps per-dataset load cheap.
    meta_only = LeRobotDataset(repo_id)
    selected = select_episodes_per_task(meta_only.meta, episodes_per_task, seed)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta_only.meta)
    del meta_only  # release file handles before reopening with episodes filter

    base = LeRobotDataset(repo_id, episodes=selected, delta_timestamps=delta_timestamps)
    base.meta.episodes = _EpisodeByValueLookup(base.meta.episodes)
    normalized = NormalizedCameraDataset(base, camera_map=camera_map)
    return GoalConditionedDataset(normalized)


def _build_make_dataset(recipe: dict):
    """Return a `make_dataset(cfg)` closure bound to a specific recipe."""
    seed = recipe.get("seed", 0)
    n_per_task = recipe.get("episodes_per_task", 3)

    def make_dataset(cfg):
        subs = [
            _build_sub_dataset(entry, n_per_task, seed, cfg.policy)
            for entry in recipe["datasets"]
        ]
        return ConcatLeRobotDataset(subs)

    return make_dataset


def _find_goal_embedding(policy):
    """Traverse accelerate/DDP wrappers to find goal_type_embedding, or return None."""
    for candidate in (policy, getattr(policy, "module", None)):
        if candidate is None:
            continue
        gem = getattr(getattr(candidate, "model", None), "goal_type_embedding", None)
        if gem is not None:
            return gem
    return None


def _wrap_update_policy(original):
    def _update_policy_with_goal_logging(train_metrics, policy, batch, *args, **kwargs):
        gem = _find_goal_embedding(policy)

        # Capture grad norm during backward (zero_grad is called inside update_policy,
        # so we cannot read .grad after it returns).
        captured_grad_norm = [None]
        hook = (
            gem.register_hook(lambda g: captured_grad_norm.__setitem__(0, g.detach().norm().item()))
            if gem is not None
            else None
        )

        train_metrics, output_dict = original(train_metrics, policy, batch, *args, **kwargs)

        if gem is not None:
            hook.remove()
            output_dict["goal_emb/weight_norm"] = gem.data.norm().item()
            if captured_grad_norm[0] is not None:
                output_dict["goal_emb/grad_norm"] = captured_grad_norm[0]

        return train_metrics, output_dict

    return _update_policy_with_goal_logging


def install(recipe: dict) -> None:
    """Install both monkeypatches against `lerobot.scripts.lerobot_train`."""
    _lerobot_train.make_dataset = _build_make_dataset(recipe)
    _lerobot_train.update_policy = _wrap_update_policy(_lerobot_train.update_policy)
