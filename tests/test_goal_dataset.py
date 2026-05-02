# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for GoalConditionedDataset.

Uses a small fake dataset to avoid Hub downloads. The fake only needs to
expose enough of the LeRobotDataset surface for the wrapper to work:
`__len__`, `__getitem__`, and a `meta` with `features`, `stats`,
`camera_keys`, `episodes`.
"""

import random

import numpy as np
import torch

from smolvla_goal import GoalConditionedDataset


class _FakeMeta:
    def __init__(self, features, stats, episodes):
        self.features = features
        self.stats = stats
        self.episodes = episodes  # list of row-dicts, indexable by episode idx

    @property
    def camera_keys(self):
        return [k for k, ft in self.features.items() if ft["dtype"] in ("video", "image")]


class _FakeDataset:
    """Minimal stand-in for LeRobotDataset.

    Holds a flat list of frames, each tagged with an episode_index. Each camera
    key gets a deterministic image of shape (3, 4, 4) so we can verify identity
    via exact tensor equality.
    """

    def __init__(self, camera_keys, episode_lengths):
        self._cameras = list(camera_keys)
        self._frames = []
        episodes = []
        cursor = 0
        for ep_idx, ep_len in enumerate(episode_lengths):
            for frame_in_ep in range(ep_len):
                frame = {"episode_index": torch.tensor(ep_idx)}
                for cam in self._cameras:
                    # Unique, reproducible image content per (cam, ep, frame).
                    seed = hash((cam, ep_idx, frame_in_ep)) & 0xFFFFFFFF
                    rng = np.random.default_rng(seed)
                    frame[cam] = torch.from_numpy(
                        rng.integers(0, 256, size=(3, 4, 4), dtype=np.uint8)
                    )
                self._frames.append(frame)
            episodes.append(
                {"dataset_from_index": cursor, "dataset_to_index": cursor + ep_len}
            )
            cursor += ep_len

        features = {
            cam: {"dtype": "video", "shape": [3, 4, 4], "names": ["channels", "height", "width"]}
            for cam in self._cameras
        }
        features["observation.state"] = {"dtype": "float32", "shape": [6]}
        stats = {
            cam: {"mean": torch.zeros(3), "std": torch.ones(3)} for cam in self._cameras
        }
        self.meta = _FakeMeta(features, stats, episodes)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        # Return a shallow copy so wrapper mutations don't pollute the source.
        return dict(self._frames[idx])


def test_goal_image_shape_and_key():
    ds = _FakeDataset(["observation.images.top", "observation.images.wrist"], [3, 5])
    wrapped = GoalConditionedDataset(ds)
    item = wrapped[0]
    assert "observation.goal_image.0" in item
    assert item["observation.goal_image.0"].shape == (3, 4, 4)


def test_goal_is_at_least_8_steps_ahead():
    # Two episodes of length 16. For every frame, compute the expected goal
    # frame deterministically and verify the goal image matches.
    ds = _FakeDataset(["observation.images.top"], [16, 16])
    wrapped = GoalConditionedDataset(ds)

    for frame_idx in range(len(ds)):
        item = wrapped[frame_idx]
        ep = int(item["episode_index"])
        ep_start = ep * 16
        within_ep = frame_idx - ep_start
        future_start = within_ep + 8
        if future_start < 16:
            future_pool = list(range(ep_start + future_start, ep_start + 16))
            expected_rel = random.Random(ep * 100_000 + within_ep).choice(future_pool)
        else:
            expected_rel = ep_start + 15  # fallback: last frame of episode
        expected = ds[expected_rel]["observation.images.top"]
        assert torch.equal(item["observation.goal_image.0"], expected)


def test_fallback_when_fewer_than_8_steps_remain():
    # Episode of length 10. within_ep_pos >= 2 → future_start >= 10 → fallback to last (rel 9).
    ds = _FakeDataset(["observation.images.top"], [10])
    wrapped = GoalConditionedDataset(ds)

    last_frame_goal = ds[9]["observation.images.top"]
    for within_ep in range(2, 10):
        item = wrapped[within_ep]
        assert torch.equal(item["observation.goal_image.0"], last_frame_goal)

    # Frame at pos 0 has a future pool ([8, 9]) and should NOT always be last.
    item = wrapped[0]
    expected_rel = random.Random(0).choice([8, 9])  # seed = 0 * 100_000 + 0
    assert torch.equal(item["observation.goal_image.0"], ds[expected_rel]["observation.images.top"])


def test_goal_selection_reproducible():
    # Multiple non-wrist cameras so camera RNG has a real choice to make.
    # Long episodes so goal frame RNG also has a real choice to make.
    ds = _FakeDataset(
        ["observation.images.top", "observation.images.side", "observation.images.front"],
        [16, 16, 16, 16],
    )
    wrapped = GoalConditionedDataset(ds)
    wrapped2 = GoalConditionedDataset(ds)

    # Two independent wrapper instances must agree on every frame.
    for i in range(len(ds)):
        assert torch.equal(
            wrapped[i]["observation.goal_image.0"],
            wrapped2[i]["observation.goal_image.0"],
        )


def test_wrist_excluded_when_alternatives_exist():
    cameras = ["observation.images.wrist", "observation.images.top"]
    ep_len = 16
    ds = _FakeDataset(cameras, [ep_len] * 4)
    wrapped = GoalConditionedDataset(ds)

    for frame_idx in range(len(ds)):
        item = wrapped[frame_idx]
        ep = int(item["episode_index"])
        ep_start = ep * ep_len
        within_ep = frame_idx - ep_start
        future_start = within_ep + 8
        if future_start < ep_len:
            future_pool = list(range(ep_start + future_start, ep_start + ep_len))
            goal_rel = random.Random(ep * 100_000 + within_ep).choice(future_pool)
        else:
            goal_rel = ep_start + ep_len - 1
        # Goal must come from the top camera, not wrist.
        assert torch.equal(item["observation.goal_image.0"], ds[goal_rel]["observation.images.top"])
        assert not torch.equal(item["observation.goal_image.0"], ds[goal_rel]["observation.images.wrist"])


def test_wrist_used_when_only_option():
    ep_len = 16
    ds = _FakeDataset(["observation.images.wrist_left"], [ep_len, ep_len])
    wrapped = GoalConditionedDataset(ds)

    for frame_idx in range(len(ds)):
        item = wrapped[frame_idx]
        ep = int(item["episode_index"])
        ep_start = ep * ep_len
        within_ep = frame_idx - ep_start
        future_start = within_ep + 8
        if future_start < ep_len:
            future_pool = list(range(ep_start + future_start, ep_start + ep_len))
            goal_rel = random.Random(ep * 100_000 + within_ep).choice(future_pool)
        else:
            goal_rel = ep_start + ep_len - 1
        assert torch.equal(
            item["observation.goal_image.0"], ds[goal_rel]["observation.images.wrist_left"]
        )


def test_meta_augmentation():
    ds = _FakeDataset(["observation.images.top", "observation.images.wrist"], [2, 2])
    wrapped = GoalConditionedDataset(ds)

    # Augmented features/camera_keys include the goal key.
    assert "observation.goal_image.0" in wrapped.meta.features
    assert "observation.goal_image.0" in wrapped.meta.camera_keys
    assert "observation.goal_image.0" in wrapped.meta.stats

    # Feature spec copied from a non-wrist camera.
    assert wrapped.meta.features["observation.goal_image.0"]["shape"] == [3, 4, 4]

    # Underlying meta is not mutated.
    assert "observation.goal_image.0" not in ds.meta.features
    assert "observation.goal_image.0" not in ds.meta.camera_keys


def test_len_and_forwarded_attrs():
    ds = _FakeDataset(["observation.images.top"], [3, 3])
    wrapped = GoalConditionedDataset(ds)
    assert len(wrapped) == len(ds) == 6
    # Unknown attribute reads should forward to the base dataset.
    ds.custom_attr = "hello"
    assert wrapped.custom_attr == "hello"


if __name__ == "__main__":
    test_goal_image_shape_and_key()
    test_goal_is_at_least_8_steps_ahead()
    test_fallback_when_fewer_than_8_steps_remain()
    test_goal_selection_reproducible()
    test_wrist_excluded_when_alternatives_exist()
    test_wrist_used_when_only_option()
    test_meta_augmentation()
    test_len_and_forwarded_attrs()
    print("\nAll goal-dataset tests passed.")
