# Training on vast.ai — step-by-step guide

Tested on an RTX 5090 instance (32 GB VRAM). Should work on any 24 GB+ GPU 
that can run `vast.ai`'s PyTorch images.
During training, only 9.2GB of the VRAM was usually used.

---

## 1. Rent and connect

- Rent an RTX 5090 (or similar 24 GB+ GPU) on [vast.ai](https://vast.ai).
- Choose a **PyTorch** image (e.g. `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`).
- Connect via SSH:

```bash
ssh -p <port> root@<host>
```

---

## 2. Clone the repo

```bash
cd /workspace
git clone https://github.com/nielsquackels/smolvla-goal.git
cd smolvla-goal
```

If you already cloned on a previous run, just pull:

```bash
cd /workspace/smolvla-goal
git pull
```

---

## 3. Install

```bash
bash setup.sh
```

This clones LeRobot alongside the repo, installs both with the correct extras,
and pins torchcodec to a CPU-only build (avoids CUDA 13 NPP linkage issues that
appear in some vast.ai images). Idempotent — safe to re-run. Takes 2–5 minutes.

---

## 4. Authenticate

```bash
wandb login        # paste your W&B API key
hf auth login      # paste your HF token (write scope required for checkpoint push)
```

The `git-credential` warning from `hf auth login` is harmless — the token is
saved to `/workspace/.hf_home/token` and used directly for all Hub API calls.

---

## 5. Prepare datasets

```bash
python scripts/prepare_datasets.py
```

Converts v2.1 community datasets to v3.0 format. Safe to re-run; "already v3.0"
messages are benign. Takes a few minutes on first run.

---

## 6. Smoke test

```bash
python tests/test_smoke.py
```

Should complete without errors. Runs 2 gradient steps through the full stack
on tiny fake data to catch config and import issues before the real run.

---

## 7. Launch training inside tmux

**Always use tmux** so training survives SSH disconnects.

```bash
# Start a new session (or attach to existing: tmux attach)
tmux new -s train

# Clean up any previous failed run
rm -rf outputs/goal_run_0

# Launch
python train.py \
  --policy.type=smolvla_goal \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.repo_id=Niels2/smolvla-goal-run0 \
  --output_dir=outputs/goal_run_0 \
  --steps=30000 \
  --batch_size=16 \
  --save_freq=5000 \
  --wandb.enable=true \
  --wandb.entity=nielsquackels-personal-projects \
  --wandb.project=smolvla-goal \
  --wandb.notes="first 5090 run"
```

Detach from the session (leaves training running):

```
Ctrl+b  then  d
```

Close the terminal and go do something else. SSH back in tomorrow:

```bash
tmux attach
```

---

## 8. Monitor in W&B

Open the run URL printed at startup. Key signals to watch:

| Metric | Expected behaviour |
|---|---|
| `train/loss` | Starts ~0.4, should trend down over 30k steps |
| `train/grad_norm` | Starts elevated, stabilises |
| `goal_emb/weight_norm` | Starts near 0 (init std=0.02), drifts up as the goal embedding learns |
| `goal_emb/grad_norm` | Confirms the goal embedding is receiving gradient |
| System → GPU Util / VRAM | Should be near 100% util and ~24–28 GB VRAM |

Checkpoints are saved every 5 000 steps locally (`outputs/goal_run_0/checkpoints/`)
and uploaded to W&B as artifacts.

---

## 9. After training

The final checkpoint is pushed automatically to HuggingFace Hub at
`huggingface.co/Niels2/smolvla-goal-run0`.

**Destroy the vast.ai instance.** Billing is per-second — don't forget.
