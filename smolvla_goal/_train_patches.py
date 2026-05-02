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


class _EpisodeProxy:
    """Episode metadata proxy that fixes two classes of community dataset bugs:

    1. Out-of-order rows: looks up by episode_index VALUE, not row position.
       LeRobot's DatasetReader does `meta.episodes[ep_idx]` expecting row N to
       hold episode N; several community datasets and v2.1→v3.0 conversions
       break that assumption.

    2. Incorrect dataset_from/to_index: some datasets' episodes parquet has
       wrong frame-range bounds.  We override them with ranges recomputed from
       the actual loaded hf_dataset, so delta-timestamp clamping stays within
       the frames that are really present in _absolute_to_relative_idx.

    Negative ints, slices, and column-name strings pass through unchanged.
    """

    def __init__(
        self,
        hf_episodes,
        pos_by_value: dict[int, int],
        ep_ranges: dict[int, tuple[int, int]],
    ):
        self._inner = hf_episodes
        self._pos_by_value = pos_by_value
        self._ep_ranges = ep_ranges

    def __getitem__(self, key):
        if isinstance(key, int) and key >= 0:
            if key in self._pos_by_value:
                row = dict(self._inner[self._pos_by_value[key]])
            else:
                row = dict(self._inner[key])
            if key in self._ep_ranges:
                row["dataset_from_index"] = self._ep_ranges[key][0]
                row["dataset_to_index"] = self._ep_ranges[key][1]
            return row
        return self._inner[key]

    def __len__(self):
        return len(self._inner)

    def __iter__(self):
        return iter(self._inner)

    def __contains__(self, item):
        return item in self._inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _make_episode_proxy(base_dataset) -> _EpisodeProxy:
    """Build an _EpisodeProxy from the already-loaded dataset.

    Computes:
    - pos_by_value: episode_index value → row position in meta.episodes
    - ep_ranges:    episode_index value → (actual_from, actual_to) derived
                    from the loaded frame parquet, not from the metadata
    """
    raw_episodes = base_dataset.meta.episodes
    pos_by_value = {int(v): i for i, v in enumerate(raw_episodes["episode_index"])}

    ep_ranges: dict[int, tuple[int, int]] = {}
    hf = base_dataset.reader.hf_dataset
    if hf is not None and hasattr(hf, "data"):
        ep_col = hf.data.column("episode_index").to_pylist()
        idx_col = hf.data.column("index").to_pylist()
        lo: dict[int, int] = {}
        hi: dict[int, int] = {}
        for ep, abs_i in zip(ep_col, idx_col):
            ep_int = int(ep)
            if ep_int not in lo:
                lo[ep_int] = abs_i
                hi[ep_int] = abs_i + 1
            else:
                if abs_i < lo[ep_int]:
                    lo[ep_int] = abs_i
                if abs_i + 1 > hi[ep_int]:
                    hi[ep_int] = abs_i + 1
        ep_ranges = {ep: (lo[ep], hi[ep]) for ep in lo}

    return _EpisodeProxy(raw_episodes, pos_by_value, ep_ranges)


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
    base.meta.episodes = _make_episode_proxy(base)
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
