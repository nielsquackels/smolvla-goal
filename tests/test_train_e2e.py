# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""End-to-end training simulation: 2 gradient steps, no Hub downloads.

Uses a randomly-initialized SmolVLAGoalPolicy (load_vlm_weights=False) and a
fake multi-dataset recipe. On a 6 GB GPU (RTX 2060) completes in ~60 s.
Requires CUDA; skipped otherwise.

Verifies the things the unit/integration tests can't:
  - ConcatLeRobotDataset.meta is in a format make_policy can consume
    (feature dicts have the "names" field dataset_to_policy_features expects)
  - policy.forward() accepts batches produced by our full pipeline without
    shape or dtype errors
  - loss is finite and backward() reaches goal_type_embedding
  - _wrap_update_policy captures goal_emb/weight_norm and goal_emb/grad_norm
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lerobot.policies.factory import make_policy
from smolvla_goal import SmolVLAGoalConfig
from smolvla_goal._train_patches import _build_make_dataset, _wrap_update_policy

# ── Constants ─────────────────────────────────────────────────────────────────

_CHUNK = 4
_ACT_DIM = 7
_STATE_DIM = 6
_FPS = 10
_EP_LEN = 12
_EP_VALS = (5, 0, 3)  # out-of-order episode values


# ── Minimal HF-Dataset column shim ───────────────────────────────────────────

class _FakeCol:
    def __init__(self, values): self._v = values
    def to_pylist(self): return list(self._v)

class _FakeTable:
    def __init__(self, cols): self._c = cols
    def column(self, name): return _FakeCol(self._c[name])

class _FakeHFDS:
    def __init__(self, ep_col, idx_col):
        self.data = _FakeTable({"episode_index": ep_col, "index": idx_col})


# ── Fake episode table ────────────────────────────────────────────────────────

class _FakeEpTable:
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


# ── Fake meta — includes "names" so dataset_to_policy_features works ──────────

class _FakeMeta:
    fps = _FPS
    features = {
        "observation.images.top": {
            "dtype": "video",
            "shape": [3, 4, 4],  # small; NormalizedCameraDataset resizes to 512×512
            "names": ["channel", "height", "width"],
        },
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


# ── Fake LeRobotDataset ───────────────────────────────────────────────────────

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
            "observation.language.tokens": torch.zeros(8, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(8, dtype=torch.bool),
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


def _build_dataset():
    with _patch_lerobot():
        return _build_make_dataset(_recipe())(SimpleNamespace(policy=_policy_cfg()))


def _build_policy(ds):
    # chunk_size must equal len(action_delta_indices) used when building the dataset.
    # In production this is 50 (SmolVLA default); here we use _CHUNK=4 to keep
    # fakes small. Real training enforces the match via resolve_delta_timestamps.
    cfg = SmolVLAGoalConfig(
        num_goal_images=1,
        load_vlm_weights=False,
        chunk_size=_CHUNK,
        n_action_steps=_CHUNK,
    )
    return make_policy(cfg, ds_meta=ds.meta)


def _prepare_batch(batch, ds, device):
    """Move to device and convert uint8 cameras to float — mirrors the training loop."""
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    for cam_key in ds.meta.camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].float() / 255.0
    return batch


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA (~60 s on RTX 2060)")
def test_policy_forward_two_steps():
    """make_policy → DataLoader → 2 forward+backward passes.

    Checks:
    - meta.features format is accepted by make_policy / dataset_to_policy_features
    - policy.forward(batch) does not crash on our pipeline's output shape
    - loss is finite on both steps
    - goal_type_embedding receives a gradient (it is in the computation graph)
    """
    device = torch.device("cuda")
    ds = _build_dataset()
    policy = _build_policy(ds).to(device)
    policy.train()

    loader = DataLoader(ds, batch_size=1, shuffle=False)

    for step, raw_batch in enumerate(loader):
        if step >= 2:
            break
        batch = _prepare_batch(raw_batch, ds, device)

        policy.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(batch)

        assert torch.isfinite(loss), f"step {step}: loss is not finite ({loss})"
        loss.backward()

        gem = policy.model.goal_type_embedding
        assert gem.grad is not None, (
            "goal_type_embedding has no gradient after backward — "
            "it is not connected to the computation graph"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA (~60 s on RTX 2060)")
def test_wrap_update_policy_logs_goal_embedding():
    """_wrap_update_policy must add goal_emb/weight_norm and goal_emb/grad_norm to output_dict."""
    device = torch.device("cuda")
    ds = _build_dataset()
    policy = _build_policy(ds).to(device)
    policy.train()

    loader = DataLoader(ds, batch_size=1, shuffle=False)
    batch = _prepare_batch(next(iter(loader)), ds, device)

    def _fake_update_policy(train_metrics, pol, b, optimizer, *args, **kwargs):
        pol.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, out = pol.forward(b)
        loss.backward()
        return train_metrics, out or {}

    wrapped = _wrap_update_policy(_fake_update_policy)
    _, output_dict = wrapped({}, policy, batch, optimizer=None)

    assert "goal_emb/weight_norm" in output_dict, (
        "goal_emb/weight_norm missing from output_dict — "
        "_wrap_update_policy may not be finding goal_type_embedding"
    )
    assert "goal_emb/grad_norm" in output_dict, (
        "goal_emb/grad_norm missing — hook may not have fired during backward"
    )
    assert output_dict["goal_emb/grad_norm"] > 0, "grad_norm is zero — embedding is not training"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA device found — skipping e2e tests.")
    else:
        print(f"Running on {torch.cuda.get_device_name(0)} …")
        test_policy_forward_two_steps()
        print("✓ test_policy_forward_two_steps")
        test_wrap_update_policy_logs_goal_embedding()
        print("✓ test_wrap_update_policy_logs_goal_embedding")
        print("\nAll e2e tests passed.")
