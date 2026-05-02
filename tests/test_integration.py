# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Integration test: full data pipeline, one DataLoader batch.

Patches LeRobotDataset with a fake to avoid Hub downloads.
Catches two bugs from the first real training run (2026-05-02):

  Issue 8 — delta_timestamps not threaded through _build_sub_dataset
    → action shape (action_dim,) instead of (chunk_size, action_dim)
    → att/pad mask mismatch inside SmolVLA flow matching at step 0.

  Issue 9 — positional indexing into out-of-order episode tables
    → wrong from/to frame indices → KeyError in _absolute_to_relative_idx.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from smolvla_goal._train_patches import _EpisodeByValueLookup, _build_make_dataset

_CHUNK = 4      # chunk_size — tiny, but distinct from _ACT_DIM
_ACT_DIM = 7
_STATE_DIM = 6
_FPS = 10
_EP_LEN = 12    # frames per episode; > min_goal_steps_ahead (8) + 1
_EP_VALS = (5, 0, 3)  # episode_index VALUES at table positions 0, 1, 2 (out-of-order)


# ── Minimal HF-Dataset column shim for GoalConditionedDataset fast path ──────

class _FakeCol:
    def __init__(self, values): self._v = values
    def to_pylist(self): return list(self._v)

class _FakeTable:
    def __init__(self, cols): self._c = cols
    def column(self, name): return _FakeCol(self._c[name])

class _FakeHFDS:
    def __init__(self, ep_col, idx_col):
        self.data = _FakeTable({"episode_index": ep_col, "index": idx_col})


# ── Fake episode table (positional position ≠ episode_index value) ────────────

class _FakeEpTable:
    """List-of-dicts mimicking the HF Dataset column interface.

    Episode rows are stored out-of-order so that `table[0]["episode_index"] == 5`,
    not 0 — exercising the _EpisodeByValueLookup fix for issue 9.
    """
    def __init__(self):
        self._rows = [
            {
                "episode_index": v,
                "tasks": ["task_a" if i < 2 else "task_b"],
                "dataset_from_index": i * _EP_LEN,
                "dataset_to_index": (i + 1) * _EP_LEN,
            }
            for i, v in enumerate(_EP_VALS)
        ]

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self._rows]
        return self._rows[key]

    def __iter__(self): return iter(self._rows)
    def __len__(self): return len(self._rows)


# ── Fake LeRobotDataset ───────────────────────────────────────────────────────

class _FakeMeta:
    fps = _FPS
    features = {
        "observation.images.top": {"dtype": "video", "shape": [3, 4, 4]},
        "observation.state": {"dtype": "float32", "shape": [_STATE_DIM]},
        "action": {"dtype": "float32", "shape": [_ACT_DIM]},
    }
    stats = {
        "observation.images.top": {
            "mean": np.zeros((3, 1, 1), dtype=np.float32),
            "std": np.ones((3, 1, 1), dtype=np.float32),
            "min": np.zeros((3, 1, 1), dtype=np.float32),
            "max": np.ones((3, 1, 1), dtype=np.float32),
            "count": np.array([_EP_LEN * len(_EP_VALS)], dtype=np.int64),
        }
    }

    def __init__(self):
        self.episodes = _FakeEpTable()
        self.meta = self

    @property
    def camera_keys(self):
        return [k for k, v in self.features.items() if v["dtype"] in ("video", "image")]


class _FakeLRDataset:
    def __init__(self, repo_id, episodes=None, delta_timestamps=None):
        self._dt = delta_timestamps
        n_total = len(_EP_VALS) * _EP_LEN
        ep_col = [v for v in _EP_VALS for _ in range(_EP_LEN)]
        self.hf_dataset = _FakeHFDS(ep_col, list(range(n_total)))
        self.meta = _FakeMeta()

    def __len__(self):
        return len(_EP_VALS) * _EP_LEN

    def __getitem__(self, idx):
        pos = idx // _EP_LEN
        ep_val = _EP_VALS[pos]
        n_act = len(self._dt["action"]) if self._dt and "action" in self._dt else None
        return {
            "episode_index": torch.tensor(ep_val),
            "index": torch.tensor(idx),
            "observation.images.top": torch.randint(0, 256, (3, 4, 4), dtype=torch.uint8),
            "observation.state": torch.zeros(_STATE_DIM),
            "action": torch.zeros(n_act, _ACT_DIM) if n_act else torch.zeros(_ACT_DIM),
            "observation.language_tokens": torch.zeros(8, dtype=torch.long),
            "observation.language_attention_mask": torch.ones(8, dtype=torch.bool),
        }


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _recipe():
    return {
        "seed": 42,
        "episodes_per_task": 10,
        "datasets": [
            {
                "repo_id": "fake/ds",
                "cameras": {"observation.images.top": "observation.images.main"},
            }
        ],
    }


def _policy_cfg():
    return SimpleNamespace(
        observation_delta_indices=[0],
        action_delta_indices=list(range(_CHUNK)),
        reward_delta_indices=None,
    )


def _patch_lerobot():
    return patch(
        "smolvla_goal._train_patches.LeRobotDataset",
        side_effect=lambda repo_id, episodes=None, delta_timestamps=None: _FakeLRDataset(
            repo_id, episodes=episodes, delta_timestamps=delta_timestamps
        ),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_action_has_chunk_dimension():
    """batch['action'] must be (B, chunk_size, action_dim).

    Without delta_timestamps threaded through _build_sub_dataset, actions
    come out flat (action_dim,) and flow matching raises a shape mismatch.
    """
    with _patch_lerobot():
        ds = _build_make_dataset(_recipe())(SimpleNamespace(policy=_policy_cfg()))

    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["action"].shape == (2, _CHUNK, _ACT_DIM), (
        f"Expected (2, {_CHUNK}, {_ACT_DIM}), got {batch['action'].shape!r} — "
        "delta_timestamps likely not passed to LeRobotDataset in _build_sub_dataset"
    )


def test_episode_lookup_is_wrapped():
    """_build_sub_dataset must replace meta.episodes with _EpisodeByValueLookup.

    Without the wrap, positional indexing into an out-of-order episode table
    returns wrong from/to frame indices, causing a KeyError at step 0.
    """
    from smolvla_goal._train_patches import _build_sub_dataset

    entry = {
        "repo_id": "fake/ds",
        "cameras": {"observation.images.top": "observation.images.main"},
    }
    with _patch_lerobot():
        sub = _build_sub_dataset(entry, episodes_per_task=10, seed=42, policy_cfg=_policy_cfg())

    # GoalConditionedDataset._base → NormalizedCameraDataset._base → _FakeLRDataset
    inner = sub._base._base
    assert isinstance(inner.meta.episodes, _EpisodeByValueLookup), (
        "meta.episodes must be wrapped in _EpisodeByValueLookup after _build_sub_dataset; "
        "without it, out-of-order episode tables cause KeyError in the dataloader"
    )


def test_out_of_order_episodes_no_keyerror():
    """Iterating a DataLoader built from out-of-order episodes must not raise KeyError."""
    with _patch_lerobot():
        ds = _build_make_dataset(_recipe())(SimpleNamespace(policy=_policy_cfg()))

    for _batch in DataLoader(ds, batch_size=4, shuffle=False):
        pass


if __name__ == "__main__":
    test_action_has_chunk_dimension()
    test_episode_lookup_is_wrapped()
    test_out_of_order_episodes_no_keyerror()
    print("\nAll integration tests passed.")
