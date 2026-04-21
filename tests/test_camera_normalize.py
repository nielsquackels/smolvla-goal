# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for NormalizedCameraDataset.

Verifies renaming, resize-with-pad to canonical shape, zero-fill + padding
masks for missing canonical slots, dropped extras, and meta augmentation.
"""

import numpy as np
import pytest
import torch

from smolvla_goal import CANONICAL_CAMERA_VOCAB, NormalizedCameraDataset


class _FakeMeta:
    def __init__(self, features, stats):
        self.features = features
        self.stats = stats

    @property
    def camera_keys(self):
        return [k for k, ft in self.features.items() if ft["dtype"] in ("video", "image")]


class _FakeDataset:
    """Fake base dataset with configurable per-camera image shapes."""

    def __init__(self, cam_shapes: dict[str, tuple[int, int, int]], n_frames: int = 2):
        self._cam_shapes = cam_shapes
        self._n = n_frames
        self._frames = []
        for i in range(n_frames):
            frame = {"episode_index": torch.tensor(0), "observation.state": torch.zeros(6)}
            for cam, shape in cam_shapes.items():
                rng = np.random.default_rng(hash((cam, i)) & 0xFFFFFFFF)
                frame[cam] = torch.from_numpy(
                    rng.integers(0, 256, size=shape, dtype=np.uint8)
                )
            self._frames.append(frame)

        features = {
            cam: {"dtype": "video", "shape": list(shape), "names": ["channels", "height", "width"]}
            for cam, shape in cam_shapes.items()
        }
        features["observation.state"] = {"dtype": "float32", "shape": [6]}
        stats = {cam: {"mean": torch.zeros(3), "std": torch.ones(3)} for cam in cam_shapes}
        self.meta = _FakeMeta(features, stats)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return dict(self._frames[idx])


def test_rename_and_resize():
    ds = _FakeDataset({"observation.images.top": (3, 240, 320)})
    wrapped = NormalizedCameraDataset(
        ds,
        camera_map={"observation.images.top": "observation.images.main"},
        target_hw=(128, 128),
    )
    item = wrapped[0]

    # Renamed key present with canonical shape; original key dropped.
    assert item["observation.images.main"].shape == (3, 128, 128)
    assert item["observation.images.main"].dtype == torch.uint8
    assert "observation.images.top" not in item

    # Mask true for present camera.
    assert item["observation.images.main_padding_mask"].item() is True


def test_missing_slots_zero_filled_with_false_mask():
    ds = _FakeDataset({"observation.images.top": (3, 240, 320)})
    wrapped = NormalizedCameraDataset(
        ds,
        camera_map={"observation.images.top": "observation.images.main"},
        target_hw=(64, 64),
    )
    item = wrapped[0]

    for canonical in CANONICAL_CAMERA_VOCAB:
        assert canonical in item
        assert item[canonical].shape == (3, 64, 64)
        assert f"{canonical}_padding_mask" in item

    assert item["observation.images.secondary_padding_mask"].item() is False
    assert item["observation.images.wrist_padding_mask"].item() is False
    assert torch.all(item["observation.images.secondary"] == 0)
    assert torch.all(item["observation.images.wrist"] == 0)


def test_extras_dropped():
    # `observation.images.extra` is not in camera_map → must be dropped.
    ds = _FakeDataset(
        {
            "observation.images.top": (3, 240, 320),
            "observation.images.extra": (3, 240, 320),
        }
    )
    wrapped = NormalizedCameraDataset(
        ds,
        camera_map={"observation.images.top": "observation.images.main"},
        target_hw=(32, 32),
    )
    item = wrapped[0]
    assert "observation.images.extra" not in item
    assert "observation.images.top" not in item


def test_non_camera_fields_preserved():
    ds = _FakeDataset({"observation.images.top": (3, 100, 100)})
    wrapped = NormalizedCameraDataset(
        ds,
        camera_map={"observation.images.top": "observation.images.main"},
        target_hw=(32, 32),
    )
    item = wrapped[0]
    assert "observation.state" in item
    assert "episode_index" in item


def test_meta_exposes_canonical_slots():
    ds = _FakeDataset({"observation.images.top": (3, 100, 100)})
    wrapped = NormalizedCameraDataset(
        ds,
        camera_map={"observation.images.top": "observation.images.main"},
        target_hw=(32, 32),
    )
    for canonical in CANONICAL_CAMERA_VOCAB:
        assert canonical in wrapped.meta.features
        assert wrapped.meta.features[canonical]["shape"] == [3, 32, 32]
        assert wrapped.meta.features[canonical]["dtype"] in ("video", "image")

    # Original camera key dropped from features.
    assert "observation.images.top" not in wrapped.meta.features
    # camera_keys reflects canonical vocab.
    assert set(wrapped.meta.camera_keys) == set(CANONICAL_CAMERA_VOCAB)


def test_unknown_target_rejected():
    ds = _FakeDataset({"observation.images.top": (3, 100, 100)})
    with pytest.raises(ValueError, match="not in canonical vocab"):
        NormalizedCameraDataset(
            ds, camera_map={"observation.images.top": "observation.images.bogus"}
        )


def test_missing_source_rejected():
    ds = _FakeDataset({"observation.images.top": (3, 100, 100)})
    with pytest.raises(ValueError, match="not found in base dataset"):
        NormalizedCameraDataset(
            ds, camera_map={"observation.images.nope": "observation.images.main"}
        )


def test_portrait_and_landscape_both_pad_to_target():
    # Two sub-datasets with opposite aspect ratios both end up (3, 96, 96).
    ds_landscape = _FakeDataset({"observation.images.top": (3, 240, 320)})
    ds_portrait = _FakeDataset({"observation.images.top": (3, 320, 240)})
    for ds in (ds_landscape, ds_portrait):
        wrapped = NormalizedCameraDataset(
            ds, camera_map={"observation.images.top": "observation.images.main"}, target_hw=(96, 96)
        )
        assert wrapped[0]["observation.images.main"].shape == (3, 96, 96)


if __name__ == "__main__":
    test_rename_and_resize()
    test_missing_slots_zero_filled_with_false_mask()
    test_extras_dropped()
    test_non_camera_fields_preserved()
    test_meta_exposes_canonical_slots()
    test_unknown_target_rejected()
    test_missing_source_rejected()
    test_portrait_and_landscape_both_pad_to_target()
    print("\nAll camera-normalize tests passed.")
