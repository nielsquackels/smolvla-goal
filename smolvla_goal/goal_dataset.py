# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Goal-image conditioning wrapper around LeRobotDataset.

For each item, injects `observation.goal_image.0` — the last frame of the same
episode, pulled from a deterministically-chosen non-wrist camera (picked per
episode via `random.Random(episode_index)`).
"""

import random
import re
from copy import deepcopy

# Matches wrist / gripper / end-effector / eye-in-hand style names used across
# community datasets. Used to exclude these cameras from goal-image sourcing —
# we want the goal to reflect the full scene, not a close-up.
_WRIST_RE = re.compile(
    r"(wrist|gripper|endeffector|end[_-]?effector|hand[_-]?eye|eye[_-]?in[_-]?hand|eih)",
    re.IGNORECASE,
)


def _is_wrist(cam_key: str) -> bool:
    return bool(_WRIST_RE.search(cam_key))


class _AugmentedMeta:
    """Proxy around `LeRobotDatasetMetadata` that advertises the goal image as
    an extra camera feature.

    All attribute reads forward to the base meta except `features`, `stats`,
    `camera_keys`, `image_keys`, `video_keys` — those return augmented dicts
    that include the goal key, cloned from a representative real camera.
    """

    def __init__(self, base_meta, goal_key: str, source_camera: str):
        self._base = base_meta
        self._goal_key = goal_key
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
    """Wraps a LeRobotDataset, emitting `{goal_key}` = last frame of the same episode.

    Handles episode-filtered LeRobotDatasets correctly: we build a
    `{episode_index → last relative index}` map at init. For real
    LeRobotDatasets the map is built from `hf_dataset`'s columns (no video
    decode); for test doubles we fall back to iterating `__getitem__`.
    """

    def __init__(self, base_dataset, goal_key: str = "observation.goal_image.0"):
        self._base = base_dataset
        self._goal_key = goal_key

        cameras = list(base_dataset.meta.camera_keys)
        if not cameras:
            raise ValueError("Underlying dataset has no camera keys; cannot source a goal image.")
        non_wrist = [c for c in cameras if not _is_wrist(c)]
        representative = (non_wrist or cameras)[0]

        self.meta = _AugmentedMeta(base_dataset.meta, goal_key, representative)
        self._last_rel_idx = self._build_last_rel_idx_map()

    def _build_last_rel_idx_map(self) -> dict[int, int]:
        """Return {episode_index: last_relative_index} for frames in the (possibly
        filtered) base dataset.
        """
        hf = getattr(self._base, "hf_dataset", None)
        if hf is not None and hasattr(hf, "data"):
            ep_col = hf.data.column("episode_index").to_pylist()
            idx_col = hf.data.column("index").to_pylist()
            last: dict[int, tuple[int, int]] = {}
            for rel_idx, (ep, abs_i) in enumerate(zip(ep_col, idx_col, strict=True)):
                prev = last.get(ep)
                if prev is None or abs_i > prev[1]:
                    last[ep] = (rel_idx, abs_i)
            return {ep: rel for ep, (rel, _) in last.items()}
        # Fallback for test doubles: iterate __getitem__. OK on small fakes; not
        # suitable for real LeRobotDatasets (would decode videos per frame).
        last_map: dict[int, int] = {}
        for rel_idx in range(len(self._base)):
            item = self._base[rel_idx]
            last_map[int(item["episode_index"])] = rel_idx
        return last_map

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _pick_camera_for_episode(self, episode_index: int) -> str:
        cameras = list(self._base.meta.camera_keys)
        non_wrist = [c for c in cameras if not _is_wrist(c)]
        pool = non_wrist if non_wrist else cameras
        return random.Random(episode_index).choice(pool)

    def __getitem__(self, idx):
        item = self._base[idx]
        episode_index = int(item["episode_index"])
        source_cam = self._pick_camera_for_episode(episode_index)
        last_rel = self._last_rel_idx[episode_index]
        last_item = self._base[last_rel]
        item[self._goal_key] = last_item[source_cam]
        return item
