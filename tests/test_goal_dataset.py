# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for GoalConditionedDataset.

Uses a small fake dataset to avoid Hub downloads. The fake only needs to
expose enough of the LeRobotDataset surface for the wrapper to work:
`__len__`, `__getitem__`, and a `meta` with `features`, `stats`,
`camera_keys`, `episodes`.
"""

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


def test_goal_is_last_frame_of_same_episode():
    ds = _FakeDataset(["observation.images.top"], [3, 4, 2])
    wrapped = GoalConditionedDataset(ds)

    # Episode 0: frames 0..2, last is 2. Episode 1: frames 3..6, last is 6.
    # Episode 2: frames 7..8, last is 8.
    expected_last_frames = {0: 2, 1: 6, 2: 8}
    for frame_idx in range(len(ds)):
        item = wrapped[frame_idx]
        ep = int(item["episode_index"])
        expected = ds[expected_last_frames[ep]]["observation.images.top"]
        assert torch.equal(item["observation.goal_image.0"], expected)


def test_camera_selection_deterministic_per_episode():
    # Multiple non-wrist cameras so RNG actually has a choice to make.
    ds = _FakeDataset(
        ["observation.images.top", "observation.images.side", "observation.images.front"],
        [2, 2, 2, 2],
    )
    wrapped = GoalConditionedDataset(ds)

    # All frames from the same episode must get the same goal image.
    ep0_goals = [wrapped[i]["observation.goal_image.0"] for i in (0, 1)]
    assert torch.equal(ep0_goals[0], ep0_goals[1])

    # Two different wrapper instances should pick the same camera per episode
    # (deterministic via random.Random(episode_index)).
    wrapped2 = GoalConditionedDataset(ds)
    for i in range(len(ds)):
        assert torch.equal(
            wrapped[i]["observation.goal_image.0"],
            wrapped2[i]["observation.goal_image.0"],
        )


def test_wrist_excluded_when_alternatives_exist():
    ds = _FakeDataset(
        ["observation.images.wrist", "observation.images.top"], [4, 4, 4, 4, 4]
    )
    wrapped = GoalConditionedDataset(ds)

    # Across many episodes, goal images must never match the wrist camera of
    # the last frame — because wrist should be excluded from the selection pool.
    for frame_idx in range(0, len(ds), 4):
        item = wrapped[frame_idx]
        ep = int(item["episode_index"])
        last_idx = ds.meta.episodes[ep]["dataset_to_index"] - 1
        wrist_last = ds[last_idx]["observation.images.wrist"]
        top_last = ds[last_idx]["observation.images.top"]
        assert not torch.equal(item["observation.goal_image.0"], wrist_last)
        assert torch.equal(item["observation.goal_image.0"], top_last)


def test_wrist_used_when_only_option():
    ds = _FakeDataset(["observation.images.wrist_left"], [3, 3])
    wrapped = GoalConditionedDataset(ds)
    item = wrapped[0]
    last_idx = ds.meta.episodes[0]["dataset_to_index"] - 1
    assert torch.equal(
        item["observation.goal_image.0"], ds[last_idx]["observation.images.wrist_left"]
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
    test_goal_is_last_frame_of_same_episode()
    test_camera_selection_deterministic_per_episode()
    test_wrist_excluded_when_alternatives_exist()
    test_wrist_used_when_only_option()
    test_meta_augmentation()
    test_len_and_forwarded_attrs()
    print("\nAll goal-dataset tests passed.")
