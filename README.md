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
git clone https://github.com/huggingface/lerobot.git

conda create -n smolvla-goal python=3.12
conda activate smolvla-goal
pip install -e "lerobot/[smolvla]"
pip install -e .
```

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

Training script: in progress.

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
- Training pipeline: in progress
- First experiment (goal = final frame of own episode): not yet run
- Ablation (random-episode final frame as goal, loss should increase if the model is using the signal): not yet run

Training will use open-source LeRobot datasets from HuggingFace.

## License

Apache 2.0. See `LICENSE`.

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot). SmolVLA paper: [Shukor et al. 2025](https://arxiv.org/abs/2506.01844). Vision backbone: [SmolVLM-2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).