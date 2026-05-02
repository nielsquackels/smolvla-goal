#!/usr/bin/env bash
# Idempotent installer for smolvla-goal.
#
# Pins torchcodec to a CPU-only build so the install works on rented GPUs
# (e.g. vast.ai images shipping torchcodec+cu130 against missing CUDA 13 NPP
# runtime libs). Video decode runs in dataloader workers — CPU is fine.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [ ! -d lerobot ]; then
    echo "==> Cloning lerobot/ alongside this repo"
    git clone https://github.com/huggingface/lerobot.git
fi

echo "==> Removing any pre-existing torchcodec (likely a +cuXXX build on rented GPUs)"
pip uninstall -y torchcodec || true

echo "==> Installing lerobot with [smolvla,dataset] extras"
pip install -e "lerobot/[smolvla,dataset]"

echo "==> Installing CPU-only torchcodec from PyTorch index"
pip install "torchcodec>=0.3.0,<0.11.0" --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing smolvla-goal"
pip install -e .

echo "==> Installing wandb"
pip install wandb

cat <<'EOF'

==> Done.

Next:
  wandb login        # paste W&B API key
  hf auth login      # paste HF token (write scope, for final checkpoint push)

Smoke test:
  python tests/test_smoke.py
EOF
