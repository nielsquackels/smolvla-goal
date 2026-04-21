# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for ConcatLeRobotDataset.

Stats must be numpy arrays with shapes aggregate_stats accepts:
- `count` shape (1,)
- image stats shape (3,1,1); non-image like state can be any ≥1D shape
"""

import numpy as np
import pytest
import torch

from smolvla_goal import ConcatLeRobotDataset


class _FakeMeta:
    def __init__(self, features, stats, episodes, fps=30):
        self.features = features
        self.stats = stats
        self.episodes = episodes
        self.fps = fps

    @property
    def camera_keys(self):
        return [k for k, ft in self.features.items() if ft["dtype"] in ("video", "image")]


class _FakeDataset:
    """Minimal dataset with a fixed length and dict-of-columns episodes meta."""

    def __init__(
        self,
        length: int,
        state_mean: float,
        fps: int = 30,
        n_episodes: int = 2,
        tag: str = "A",
    ):
        self._length = length
        self._tag = tag
        # Split frames roughly evenly across episodes.
        per = length // n_episodes
        from_idx = [i * per for i in range(n_episodes)]
        to_idx = [(i + 1) * per for i in range(n_episodes)]
        to_idx[-1] = length  # absorb remainder
        episodes = {
            "dataset_from_index": from_idx,
            "dataset_to_index": to_idx,
            "episode_index": list(range(n_episodes)),
            "length": [t - f for f, t in zip(from_idx, to_idx, strict=True)],
        }
        features = {
            "observation.state": {"dtype": "float32", "shape": [6]},
            "observation.images.main": {
                "dtype": "video",
                "shape": [3, 32, 32],
                "names": ["channels", "height", "width"],
            },
        }
        stats = {
            "observation.state": {
                "mean": np.full((6,), state_mean, dtype=np.float32),
                "std": np.ones((6,), dtype=np.float32),
                "min": np.full((6,), state_mean - 1, dtype=np.float32),
                "max": np.full((6,), state_mean + 1, dtype=np.float32),
                "count": np.array([length], dtype=np.int64),
            },
        }
        self.meta = _FakeMeta(features, stats, episodes, fps=fps)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if not 0 <= idx < self._length:
            raise IndexError(idx)
        return {
            "observation.state": torch.zeros(6),
            "_source": self._tag,
            "_rel_idx": idx,
        }


def test_len_is_sum_of_subs():
    a = _FakeDataset(10, state_mean=0.0, tag="A")
    b = _FakeDataset(7, state_mean=1.0, tag="B")
    c = _FakeDataset(5, state_mean=2.0, tag="C")
    concat = ConcatLeRobotDataset([a, b, c])
    assert len(concat) == 22
    assert concat.num_frames == 22


def test_routing_via_bisect():
    a = _FakeDataset(4, state_mean=0.0, tag="A")
    b = _FakeDataset(3, state_mean=1.0, tag="B")
    concat = ConcatLeRobotDataset([a, b])

    assert concat[0]["_source"] == "A"
    assert concat[3]["_source"] == "A"
    assert concat[4]["_source"] == "B"
    assert concat[6]["_source"] == "B"

    # Sub-relative indices recovered correctly.
    assert concat[0]["_rel_idx"] == 0
    assert concat[3]["_rel_idx"] == 3
    assert concat[4]["_rel_idx"] == 0
    assert concat[6]["_rel_idx"] == 2


def test_out_of_bounds_raises():
    a = _FakeDataset(3, state_mean=0.0)
    concat = ConcatLeRobotDataset([a])
    with pytest.raises(IndexError):
        concat[3]
    with pytest.raises(IndexError):
        concat[-1]


def test_stats_aggregated():
    a = _FakeDataset(10, state_mean=0.0, tag="A")
    b = _FakeDataset(10, state_mean=2.0, tag="B")
    concat = ConcatLeRobotDataset([a, b])

    state_stats = concat.meta.stats["observation.state"]
    # Equal counts → aggregated mean is simple average of the two.
    np.testing.assert_allclose(state_stats["mean"], np.full(6, 1.0, dtype=np.float32))
    # Count is summed.
    np.testing.assert_array_equal(state_stats["count"], np.array([20], dtype=np.int64))


def test_features_unioned():
    a = _FakeDataset(5, state_mean=0.0)
    b = _FakeDataset(5, state_mean=0.0)
    concat = ConcatLeRobotDataset([a, b])
    assert "observation.state" in concat.meta.features
    assert "observation.images.main" in concat.meta.features
    assert "observation.images.main" in concat.meta.camera_keys


def test_fps_disagreement_raises():
    a = _FakeDataset(5, state_mean=0.0, fps=30)
    b = _FakeDataset(5, state_mean=0.0, fps=15)
    with pytest.raises(ValueError, match="disagree on fps"):
        ConcatLeRobotDataset([a, b])


def test_episode_metadata_concatenated():
    a = _FakeDataset(10, state_mean=0.0, n_episodes=2)
    b = _FakeDataset(6, state_mean=0.0, n_episodes=3)
    concat = ConcatLeRobotDataset([a, b])

    assert concat.num_episodes == 5
    # episodes exposed as a contiguous range 0..N-1 for downstream consumers.
    assert concat.episodes == list(range(5))


def test_empty_rejected():
    with pytest.raises(ValueError, match="at least one"):
        ConcatLeRobotDataset([])


if __name__ == "__main__":
    test_len_is_sum_of_subs()
    test_routing_via_bisect()
    test_out_of_bounds_raises()
    test_stats_aggregated()
    test_features_unioned()
    test_fps_disagreement_raises()
    test_episode_metadata_concatenated()
    test_empty_rejected()
    print("\nAll concat tests passed.")
