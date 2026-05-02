# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Normalize heterogeneous camera keys + resolutions to a fixed canonical vocab.

Community SO-101 datasets use different camera names (`top`, `front`, `gripper`,
`endeffector`, `handeye`, ...) and different resolutions. This wrapper:

1. Renames cameras per a user-supplied map to the canonical 4-slot scheme
   (main / secondary / wrist; goal_image is added separately by
   GoalConditionedDataset).
2. Resize-with-pads every camera image to a fixed (H, W) so items from
   different datasets collate into batches.
3. Fills missing canonical slots with zero tensors + `{key}_padding_mask=False`,
   wiring into SmolVLA's empty-camera handling. Present cameras get
   `{key}_padding_mask=True`.

Extra cameras not in the map are dropped.
"""

from copy import deepcopy

import torch
import torch.nn.functional as F

CANONICAL_CAMERA_VOCAB = (
    "observation.images.main",
    "observation.images.secondary",
    "observation.images.wrist",
)


def _resize_with_pad_uint8(img: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Aspect-preserving resize + top-left pad to `target_hw`. Single (C,H,W) uint8 image.

    Pads with 0 (black), which becomes -1 after SmolVLA's uint8→float / 255 → *2-1
    pipeline — matching its native `resize_with_pad(pad_value=-1)` semantics.
    """
    c, h, w = img.shape
    th, tw = target_hw
    ratio = max(w / tw, h / th)
    rh = max(1, int(h / ratio))
    rw = max(1, int(w / ratio))
    img_f = img.unsqueeze(0).float()
    resized = F.interpolate(img_f, size=(rh, rw), mode="bilinear", align_corners=False)
    ph = max(0, th - rh)
    pw = max(0, tw - rw)
    # Pad (left, right, top, bottom) — pad on left+top to match SmolVLA.
    padded = F.pad(resized, (pw, 0, ph, 0), value=0.0)
    return padded.squeeze(0).clamp(0, 255).to(torch.uint8)


class _NormalizedMeta:
    def __init__(self, base_meta, camera_map: dict[str, str], target_hw: tuple[int, int]):
        self._base = base_meta
        c, h, w = 3, target_hw[0], target_hw[1]

        # Build canonical feature specs. Prefer a real source camera's spec as template
        # (copy dtype + names), but override shape to post-resize.
        base_features = dict(base_meta.features)
        self._features = {k: v for k, v in base_features.items() if k not in base_meta.camera_keys}

        template = None
        for src in camera_map:
            if src in base_features:
                template = base_features[src]
                break
        if template is None:
            # No source camera in map matched the base — fall back to any base camera.
            for k in base_meta.camera_keys:
                template = base_features[k]
                break
        if template is None:
            raise ValueError("Base dataset has no camera features; camera normalization impossible.")

        for canonical in CANONICAL_CAMERA_VOCAB:
            ft = deepcopy(template)
            ft["shape"] = [c, h, w]
            self._features[canonical] = ft

        # Stats: copy from source per map; canonical keys without a mapped source get zeros.
        base_stats = base_meta.stats or {}
        self._stats = {k: v for k, v in base_stats.items() if k not in base_meta.camera_keys}
        reverse_map = {dst: src for src, dst in camera_map.items()}
        for canonical in CANONICAL_CAMERA_VOCAB:
            src = reverse_map.get(canonical)
            if src and src in base_stats:
                self._stats[canonical] = deepcopy(base_stats[src])

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


class NormalizedCameraDataset:
    """Wraps a LeRobotDataset, renames cameras per `camera_map`, pads to canonical vocab."""

    def __init__(
        self,
        base_dataset,
        camera_map: dict[str, str],
        target_hw: tuple[int, int] = (512, 512),
    ):
        self._base = base_dataset
        self._camera_map = dict(camera_map)
        self._target_hw = tuple(target_hw)
        self._canonical = CANONICAL_CAMERA_VOCAB
        self._target_chw = (3, target_hw[0], target_hw[1])

        # Validate: every mapped target must be a canonical slot.
        unknown = set(self._camera_map.values()) - set(self._canonical)
        if unknown:
            raise ValueError(
                f"camera_map targets {unknown} are not in canonical vocab {self._canonical}"
            )
        # Validate: every mapped source must exist in the base camera set.
        base_cams = set(base_dataset.meta.camera_keys)
        missing = set(self._camera_map) - base_cams
        if missing:
            raise ValueError(
                f"camera_map sources {missing} not found in base dataset cameras {sorted(base_cams)}"
            )

        self.meta = _NormalizedMeta(base_dataset.meta, self._camera_map, self._target_hw)

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __getitem__(self, idx):
        item = self._base[idx]
        out = dict(item)
        # Drop all base camera keys AND their _is_pad variants (added by LeRobot
        # for delta_timestamps). Stale source names must not leak into batches.
        for k in self._base.meta.camera_keys:
            out.pop(k, None)
            out.pop(f"{k}_is_pad", None)
        # Insert renamed + resized cameras; carry over _is_pad under the new name.
        filled_is_pad: dict = {}
        for src, dst in self._camera_map.items():
            if src in item:
                out[dst] = _resize_with_pad_uint8(item[src], self._target_hw)
                out[f"{dst}_padding_mask"] = torch.tensor(True)
                if f"{src}_is_pad" in item:
                    out[f"{dst}_is_pad"] = item[f"{src}_is_pad"]
                    filled_is_pad[dst] = item[f"{src}_is_pad"]
        # Fill any canonical slot that wasn't covered by the map.
        for canonical in self._canonical:
            if canonical not in out:
                out[canonical] = torch.zeros(self._target_chw, dtype=torch.uint8)
                out[f"{canonical}_padding_mask"] = torch.tensor(False)
                # Keep key-set uniform across datasets: add zero _is_pad if other
                # cameras in this item have one (same shape, all-False = not padded).
                if filled_is_pad:
                    template = next(iter(filled_is_pad.values()))
                    out[f"{canonical}_is_pad"] = torch.zeros_like(template, dtype=torch.bool)
        return out
