#!/usr/bin/env python
# Copyright 2026 Niels Quackels. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""One-time dataset preparation for SmolVLA-Goal training.

Reads the training recipe YAML (default `configs/training_data.yaml`), checks
each dataset's codebase version on the Hub, and converts v2.1 datasets to
v3.0 locally (no push). After this runs, `train.py` can consume every
dataset via `LeRobotDataset(repo_id)` without further setup.

Usage:
    python scripts/prepare_datasets.py                            # default recipe
    python scripts/prepare_datasets.py --config=/path/to/file.yaml

If `convert_dataset` hits a network error or the v2.1→v3.0 conversion fails
for a single dataset, we log and move on so one broken repo doesn't block
the rest.
"""

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

import yaml

from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

_DEFAULT_RECIPE_PATH = Path(__file__).resolve().parent.parent / "configs" / "training_data.yaml"


def _fetch_codebase_version(repo_id: str) -> str | None:
    """Read `codebase_version` from the Hub's `meta/info.json`. Returns None on failure."""
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        return info.get("codebase_version")
    except Exception as exc:
        logging.warning("Could not fetch info.json for %s: %s", repo_id, exc)
        return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_RECIPE_PATH)
    args = parser.parse_args()

    with args.config.open() as f:
        recipe = yaml.safe_load(f)

    repo_ids = [entry["repo_id"] for entry in recipe["datasets"]]
    logging.info("Preparing %d dataset(s) from %s", len(repo_ids), args.config)

    for repo_id in repo_ids:
        version = _fetch_codebase_version(repo_id)
        if version == "v3.0":
            logging.info("[%s] already v3.0 — skipping", repo_id)
            continue
        if version != "v2.1":
            logging.warning(
                "[%s] unexpected codebase_version=%r — attempting conversion anyway",
                repo_id,
                version,
            )
        logging.info("[%s] converting v2.1 → v3.0 (local only, no push)", repo_id)
        try:
            convert_dataset(repo_id=repo_id, push_to_hub=False)
        except Exception as exc:
            logging.exception("[%s] conversion failed: %s", repo_id, exc)
            continue
        logging.info("[%s] done", repo_id)

    logging.info("All done.")


if __name__ == "__main__":
    sys.exit(main())
