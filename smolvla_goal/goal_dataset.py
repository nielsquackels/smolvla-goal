# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Goal-image conditioning wrapper around LeRobotDataset.

For each item, injects `observation.goal_image.0` — the last frame of the same
episode, pulled from a deterministically-chosen non-wrist camera (picked per
episode via `random.Random(episode_index)`).
"""

import random
from copy import deepcopy


class _AugmentedMeta:
    """Proxy around `LeRobotDatasetMetadata` that advertises the goal image as
    an extra camera feature.

    All attribute reads forward to the base meta except `features`, `stats`,
    `camera_keys`, `image_keys`, `video_keys` — those return augmented dicts
    that include the goal key, cloned from a representative real camera.

    The goal key IS included in `camera_keys` so the training loop's
    uint8→float32 conversion applies to it. Note: any downstream code that
    iterates `camera_keys` assuming "live camera" (e.g. robot control logic)
    would see the goal key; this wrapper is intended for training only.
    """

    def __init__(self, base_meta, goal_key: str, source_camera: str):
        self._base = base_meta
        self._goal_key = goal_key
        # Shallow-copy the dicts so we don't mutate the underlying meta.
        # Sub-dicts (e.g. stats[key]) are shared by reference so any in-place
        # mutations like IMAGENET_STATS injection still propagate.
        self._features = {**base_meta.features}
        self._features[goal_key] = deepcopy(base_meta.features[source_camera])

        base_stats = base_meta.stats or {}
        self._stats = {**base_stats}
        if source_camera in base_stats:
            self._stats[goal_key] = deepcopy(base_stats[source_camera])

    @property
    def features(self):
        return self._features

    @property
    def stats(self):
        return self._stats

    @property
    def camera_keys(self):
        return [k for k, ft in self._features.items() if ft["dtype"] in ("video", "image")]

    @property
    def image_keys(self):
        return [k for k, ft in self._features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self):
        return [k for k, ft in self._features.items() if ft["dtype"] == "video"]

    def __getattr__(self, name):
        return getattr(self._base, name)


class GoalConditionedDataset:
    """Wraps a LeRobotDataset, emitting `{goal_key}` = last frame of the same episode."""

    def __init__(self, base_dataset, goal_key: str = "observation.goal_image.0"):
        self._base = base_dataset
        self._goal_key = goal_key

        cameras = list(base_dataset.meta.camera_keys)
        if not cameras:
            raise ValueError("Underlying dataset has no camera keys; cannot source a goal image.")
        non_wrist = [c for c in cameras if "wrist" not in c.lower()]
        representative = (non_wrist or cameras)[0]

        self.meta = _AugmentedMeta(base_dataset.meta, goal_key, representative)

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _pick_camera_for_episode(self, episode_index: int) -> str:
        cameras = list(self._base.meta.camera_keys)
        non_wrist = [c for c in cameras if "wrist" not in c.lower()]
        pool = non_wrist if non_wrist else cameras
        return random.Random(episode_index).choice(pool)

    def _last_frame_index(self, episode_index: int) -> int:
        return int(self._base.meta.episodes[episode_index]["dataset_to_index"]) - 1

    def __getitem__(self, idx):
        item = self._base[idx]
        episode_index = int(item["episode_index"])
        source_cam = self._pick_camera_for_episode(episode_index)
        last_idx = self._last_frame_index(episode_index)
        last_item = self._base[last_idx]
        item[self._goal_key] = last_item[source_cam]
        return item
