# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Goal-image conditioning wrapper around LeRobotDataset.

For each item, injects `observation.goal_image.0` — a random frame at least
`min_goal_steps_ahead` steps into the future within the same episode, pulled
from a deterministically-chosen non-wrist camera (picked per episode via
`random.Random(episode_index)`). Falls back to the last frame of the episode
when fewer than `min_goal_steps_ahead` future frames remain.
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
    """Wraps a LeRobotDataset, emitting `{goal_key}` = a random future frame.

    For each item, the goal image is drawn uniformly at random from the frames
    that are at least `min_goal_steps_ahead` steps ahead (within the same
    episode). When fewer than `min_goal_steps_ahead` future frames remain, falls
    back to the last frame of the episode.

    Camera selection is deterministic per episode (seed = episode_index); goal
    frame selection is deterministic per (episode, within-episode position)
    (seed = (episode_index, within_ep_pos)). Both are reproducible across
    wrapper instances and training runs.

    Handles episode-filtered LeRobotDatasets correctly: frame index structures
    are built from `hf_dataset`'s columns at init (no video decode). For test
    doubles without `hf_dataset`, falls back to iterating `__getitem__`.
    """

    def __init__(
        self,
        base_dataset,
        goal_key: str = "observation.goal_image.0",
        min_goal_steps_ahead: int = 8,
    ):
        self._base = base_dataset
        self._goal_key = goal_key
        self._min_goal_steps_ahead = min_goal_steps_ahead

        cameras = list(base_dataset.meta.camera_keys)
        if not cameras:
            raise ValueError("Underlying dataset has no camera keys; cannot source a goal image.")
        non_wrist = [c for c in cameras if not _is_wrist(c)]
        representative = (non_wrist or cameras)[0]

        self.meta = _AugmentedMeta(base_dataset.meta, goal_key, representative)
        self._build_episode_data()

    def _build_episode_data(self) -> None:
        """Build per-episode frame lists and a within-episode position lookup.

        Sets:
          _episode_frames: {episode_index → [rel_idxs in episode order]}
          _frame_within_ep: {rel_idx → within-episode position}
        """
        hf = getattr(self._base, "hf_dataset", None)
        if hf is not None and hasattr(hf, "data"):
            ep_col = hf.data.column("episode_index").to_pylist()
            idx_col = hf.data.column("index").to_pylist()
            ep_frames: dict[int, list[tuple[int, int]]] = {}
            for rel_idx, (ep, abs_i) in enumerate(zip(ep_col, idx_col, strict=True)):
                ep_frames.setdefault(ep, []).append((rel_idx, abs_i))
            self._episode_frames: dict[int, list[int]] = {
                ep: [rel for rel, _ in sorted(pairs, key=lambda x: x[1])]
                for ep, pairs in ep_frames.items()
            }
        else:
            # Fallback for test doubles: iterate __getitem__. OK on small fakes.
            ep_to_rels: dict[int, list[int]] = {}
            for rel_idx in range(len(self._base)):
                item = self._base[rel_idx]
                ep_to_rels.setdefault(int(item["episode_index"]), []).append(rel_idx)
            self._episode_frames = ep_to_rels

        self._frame_within_ep: dict[int, int] = {
            rel_idx: pos
            for frames in self._episode_frames.values()
            for pos, rel_idx in enumerate(frames)
        }

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

        ep_frames = self._episode_frames[episode_index]
        within_ep_pos = self._frame_within_ep[idx]
        future_start = within_ep_pos + self._min_goal_steps_ahead

        if future_start < len(ep_frames):
            future_pool = ep_frames[future_start:]
            seed = episode_index * 100_000 + within_ep_pos
            goal_rel = random.Random(seed).choice(future_pool)
        else:
            goal_rel = ep_frames[-1]

        goal_item = self._base[goal_rel]
        item[self._goal_key] = goal_item[source_cam]
        return item
