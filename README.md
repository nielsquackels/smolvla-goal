# smolvla-goal

> *"Put away the groceries."* — sure, but where? Milk in the fridge door or the top shelf? Bananas in the fruit bowl or on the counter?

Goal-conditioned [SmolVLA](https://huggingface.co/blog/smolvla): adds a target image to the input, so your robot knows what "done" should look like.

---

## What this is

SmolVLA is a 450M-parameter vision-language-action model. It takes camera views, a text instruction, and robot state, and outputs a chunk of actions.

This repo adds a **goal image**: a picture of what the scene should look like after the task is done. The hope is that conditioning on a visual target gives you a higher success rate than language alone, and lets the robot match personal preferences (where *you* keep the cereal) without having to describe them in words.

[π-0.7](https://www.physicalintelligence.company/blog/pi07) recently demonstrated that scaling data diversity and conditioning signals is a big unlock for generalist robots. This is a small experiment in that spirit: keep SmolVLA's efficiency, give it a richer way to specify goals.

## How it works

Run the goal image through the same vision tower as the regular cameras, then add a small learned vector (`goal_type_embedding`) to its patch tokens before they enter the prefix. This gives the transformer a tag it can use to tell "live camera view" apart from "goal view."

The tag is initialized near zero, so at the start of finetuning the goal image is just treated as an extra camera view — the model can attend to it but doesn't yet know it's a goal. The embedding drifts away from zero during training as the model learns to use it.

## Installation

```bash
git clone https://github.com/nielsquackels/smolvla-goal.git
cd smolvla-goal

conda create -n smolvla-goal python=3.12
conda activate smolvla-goal
bash setup.sh
```

`setup.sh` clones LeRobot, installs both projects, and pins torchcodec to a
CPU build (avoids CUDA-NPP linkage issues on rented GPUs). Idempotent — safe
to re-run.

Smoke test:

```bash
python tests/test_smoke.py
```

## Usage

```python
from smolvla_goal import SmolVLAGoalConfig, SmolVLAGoalPolicy

config = SmolVLAGoalConfig(num_goal_images=1)
policy = SmolVLAGoalPolicy(config)
```

Your dataset needs to emit goal images under keys like `observation.goal_image.0`. Configurable via `config.goal_image_key_prefix`.

## Training

Finetuning runs on a curated mix of [SO-101](https://huggingface.co/lerobot) community datasets from HuggingFace, on a rented **RTX 5090 (32 GB VRAM)**. Diversity over volume — if the model can succeed across 8 heterogeneous tasks from 7 datasets, it's actually using the goal image rather than memorizing a single task's dynamics.

### Canonical camera vocabulary

Community SO-101 datasets use wildly different camera names (`top`, `front`, `gripper`, `endeffector`, `handeye`, ...) and different counts (2 or 3 cameras per dataset). We normalize to a fixed 4-slot vocabulary:

| Slot | Canonical key | Role |
|---|---|---|
| 1 | `observation.images.main` | Primary external view |
| 2 | `observation.images.secondary` | Secondary external view (optional) |
| 3 | `observation.images.wrist` | Wrist / gripper / eye-in-hand (optional) |
| 4 | `observation.goal_image.0` | Goal image, auto-injected by `GoalConditionedDataset` |

Per-dataset camera mapping lives in [configs/training_data.yaml](configs/training_data.yaml). Missing slots are filled with zero tensors + `{key}_padding_mask=False`, and SmolVLA's existing empty-camera handling masks them out. Datasets with more cameras than slots have the extras dropped.

### Datasets

3 episodes per task per dataset. Selection is deterministic via `random.Random(seed=42)`.

| Repo ID | Tasks | Eps used | Cameras (main / secondary / wrist) |
|---|---|---|---|
| [whosricky/so101-megamix-v1](https://huggingface.co/datasets/whosricky/so101-megamix-v1) | 8 | 24 | top / front / gripper |
| [lerobot/svla_so101_pickplace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) | 1 | 3 | up / side / — |
| [observabot/so101_cloth_folding1](https://huggingface.co/datasets/observabot/so101_cloth_folding1) | 1 | 3 | top / base / endeffector |
| [youliangtan/so101-table-cleanup](https://huggingface.co/datasets/youliangtan/so101-table-cleanup) | 4 | 12 | front / — / wrist |
| [lipsop/so101-block-in-bin-100ep](https://huggingface.co/datasets/lipsop/so101-block-in-bin-100ep) | 1 | 3 | front / — / wrist |
| [seunghoney/so101_test2](https://huggingface.co/datasets/seunghoney/so101_test2) | 1 | 3 | front / side / — |

`xinjiehu76/so101-pick-place-dataset` was skipped — `meta/info.json` failed to fetch. `ud-smart-city/lerobot-so-101-manipulations` was skipped — `meta/info.json` claims v3.0 but the repo has no v3.0 git tag, causing `RevisionNotFoundError`. 4 of the 6 datasets (observabot, youliangtan, lipsop, seunghoney) are codebase version v2.1 and get converted to v3.0 by `scripts/prepare_datasets.py` before training.

### Sampling

Samples are drawn uniformly across the concatenated multi-dataset via `DataLoader(shuffle=True)`. After 3-eps-per-task subsampling the per-dataset frame-count disparity is only ~2–8×, so uniform-by-frame is a reasonable default. If `whosricky/so101-megamix-v1` dominates early gradients, we'll switch to `WeightedRandomSampler` with per-dataset balancing.

### Procedure

```bash
# 1. Install (see Installation above).

# 2. Download + convert v2.1 datasets. Run once; idempotent.
python scripts/prepare_datasets.py

# 3. Launch training.
python train.py \
  --policy.type=smolvla_goal \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.repo_id=<your-hf-username>/smolvla-goal-run0 \
  --output_dir=outputs/goal_run_0 \
  --steps=30000 \
  --batch_size=16 \
  --save_freq=5000
```

Checkpoints are written under `--output_dir` and pushed to the Hub repo at
`--policy.repo_id`. For the full step-by-step procedure including W&B setup
and running on a rented GPU, see [docs/vastai_training_guide.md](docs/vastai_training_guide.md).

### Ablation (planned)

After the first run converges, rerun with each goal image replaced by the last frame of a *randomly chosen other episode*. If the `goal_type_embedding` is actually being used, validation loss should rise. If it doesn't rise, the model isn't conditioning on the goal.

## What actually changes vs. SmolVLA

Two files, three classes, ~150 lines of real code:

| Component | Change |
|---|---|
| `SmolVLAGoalConfig` | Adds `num_goal_images` and `goal_image_key_prefix`. |
| `VLAFlowMatchingGoal` | Adds a learned `goal_type_embedding` of shape `(1, 1, hidden_size)`. Overrides `embed_prefix` to add this to goal-image tokens. |
| `SmolVLAGoalPolicy` | Swaps in the goal-aware model. Overrides `prepare_images` so goal images land at the end of the image list. |

The new parameter is ~960 floats. Genuinely the smallest possible change that can carry a "this is a goal, not a camera" signal.

## Status

- Architecture implemented
- Smoke test passing
- Goal-conditioned dataset wrapper implemented + tested
- Multi-dataset training pipeline: working
- First experiment (goal = final frame of own episode): **in progress** (30k steps on RTX 5090)
- Ablation (random-episode final frame as goal, loss should rise if the signal is being used): not yet run

## License

Apache 2.0. See `LICENSE`.

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot). SmolVLA paper: [Shukor et al. 2025](https://arxiv.org/abs/2506.01844). Vision backbone: [SmolVLM-2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).