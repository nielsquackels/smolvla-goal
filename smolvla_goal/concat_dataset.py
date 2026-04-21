# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Concatenate multiple (normalized, goal-conditioned) LeRobotDatasets.

Exposes a unified `.meta` with aggregated stats so SmolVLA's MEAN_STD
normalization for state/action is computed across all sub-datasets. Camera
features are taken from the first sub-dataset — assumes upstream
NormalizedCameraDataset has already unified the camera vocab.

We don't subclass MultiLeRobotDataset because it intersects feature keys
across datasets, which silently disables cameras whose names differ between
sub-datasets. Our stack pre-normalizes camera keys, so intersection would
work — but aggregating stats and exposing a clean meta ourselves is simpler.
"""

import bisect

from lerobot.datasets.compute_stats import aggregate_stats


class _ConcatMeta:
    def __init__(self, subs):
        # Features: union across sub-datasets, preferring sub-0 for conflicts.
        # After NormalizedCameraDataset, camera keys should be identical.
        features: dict = {}
        for d in reversed(subs):
            features.update(d.meta.features)
        self._features = features

        # Stats: aggregate via lerobot's helper (weighted by counts).
        stats_list = [d.meta.stats for d in subs if d.meta.stats]
        self._stats = aggregate_stats(stats_list) if stats_list else {}

        # Per-sub-dataset episode metadata, concatenated raw. Values retain
        # their sub-dataset-relative semantics (unoffset), since SmolVLA's
        # training path doesn't read these — drop_n_last_frames isn't in its
        # config. If a future policy uses EpisodeAwareSampler with this
        # concat, sub-dataset offsets need to be applied here.
        from_list, to_list, ep_list, length_list = [], [], [], []
        for d in subs:
            eps = d.meta.episodes
            from_list.extend(eps["dataset_from_index"])
            to_list.extend(eps["dataset_to_index"])
            ep_list.extend(eps["episode_index"])
            if "length" in eps.column_names if hasattr(eps, "column_names") else False:
                length_list.extend(eps["length"])
        self._episodes = {
            "dataset_from_index": from_list,
            "dataset_to_index": to_list,
            "episode_index": ep_list,
        }
        if length_list:
            self._episodes["length"] = length_list

        # FPS: must match across sub-datasets.
        fps_values = {getattr(d.meta, "fps", None) for d in subs}
        fps_values.discard(None)
        if len(fps_values) > 1:
            raise ValueError(f"Sub-datasets disagree on fps: {fps_values}")
        self._fps = fps_values.pop() if fps_values else 30

    @property
    def features(self):
        return self._features

    @property
    def stats(self):
        return self._stats

    @property
    def episodes(self):
        return self._episodes

    @property
    def camera_keys(self):
        return [k for k, ft in self._features.items() if ft["dtype"] in ("video", "image")]

    @property
    def image_keys(self):
        return [k for k, ft in self._features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self):
        return [k for k, ft in self._features.items() if ft["dtype"] == "video"]

    @property
    def fps(self):
        return self._fps

    @property
    def total_episodes(self):
        return len(self._episodes["dataset_from_index"])

    @property
    def total_frames(self):
        return (
            sum(self._episodes["length"])
            if "length" in self._episodes
            else sum(
                t - f
                for f, t in zip(
                    self._episodes["dataset_from_index"],
                    self._episodes["dataset_to_index"],
                    strict=True,
                )
            )
        )


class ConcatLeRobotDataset:
    """Concatenates multiple LeRobotDataset-like datasets for multi-dataset training.

    Sub-datasets should already expose a uniform camera vocabulary — this
    class does not rename or re-shape. Use `NormalizedCameraDataset` upstream.
    """

    def __init__(self, sub_datasets):
        self._subs = list(sub_datasets)
        if not self._subs:
            raise ValueError("ConcatLeRobotDataset requires at least one sub-dataset.")
        self._lengths = [len(d) for d in self._subs]
        self._cumlens = [0]
        for n in self._lengths:
            self._cumlens.append(self._cumlens[-1] + n)

        self.meta = _ConcatMeta(self._subs)
        self.episodes = list(range(self.meta.total_episodes))

    @property
    def num_episodes(self):
        return self.meta.total_episodes

    @property
    def num_frames(self):
        return self._cumlens[-1]

    @property
    def fps(self):
        return self.meta.fps

    def __len__(self):
        return self._cumlens[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._cumlens[-1]:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self._cumlens[-1]}.")
        sub_idx = bisect.bisect_right(self._cumlens, idx) - 1
        offset = self._cumlens[sub_idx]
        return self._subs[sub_idx][idx - offset]
