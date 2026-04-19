# smolvla-goal

> *"Pick up the red cube."* — sure, but which one? Where? Shouldn't I just show you a picture?

Goal-conditioned [SmolVLA](https://huggingface.co/blog/smolvla): adds a target image to the prompt, so your robot knows what "done" looks like.

---

## The one-minute version

SmolVLA is a 450M-parameter vision-language-action model. It takes camera views + a text instruction + robot state, and outputs a chunk of actions. Neat, but language is a lossy way to specify goals. *"Put the cup on the left side of the tray"* — which tray? How far left? What orientation?

This repo adds a **goal image** as an extra input: a picture of what the scene should look like after the task is done. Much more precise than text for anything visually specifiable.

**The architectural question:** how do you feed an extra image into a pretrained VLA without breaking it?

**The answer used here:** run the goal image through the same vision tower as the regular cameras, then add a small learned vector (`goal_type_embedding`) to its patch tokens before they enter the prefix. The transformer can then distinguish "live camera view" from "goal view" by the tag rather than by position.

Near-zero init on the new parameter → at training step 0, the model behaves identically to base SmolVLA. It gradually learns to use the goal signal during finetuning.

## Why bother?

Three reasons:

1. **Precision.** A goal image encodes geometric information that would take a paragraph of awkward English.
2. **Demonstration-matching.** Retrieve a frame from a human demonstration video and use it as the goal. Now your robot is conditioned on "make the scene look like this."
3. **Fewer tokens in the instruction.** The language instruction can focus on *what to do* rather than *exactly where things should end up*.

## Installation

Clone both this repo and LeRobot:

```bash
git clone https://github.com/nielsquackels/smolvla-goal.git
cd smolvla-goal
git clone https://github.com/huggingface/lerobot.git
```

Create an env and install:

```bash
conda create -n smolvla-goal python=3.12
conda activate smolvla-goal
pip install -e "lerobot/[smolvla]"
pip install -e .
```

Run the smoke test to verify everything built correctly:

```bash
python tests/test_smoke.py
```

You should see three green checkmarks and no errors.

## Usage

```python
from smolvla_goal import SmolVLAGoalConfig, SmolVLAGoalPolicy

config = SmolVLAGoalConfig(num_goal_images=1)
policy = SmolVLAGoalPolicy(config)
```

Your dataset needs to emit goal images under keys like `observation.goal_image.0`. The key prefix is configurable via `config.goal_image_key_prefix`.

Training script: *coming soon* — still being built.

## What's actually different from SmolVLA

Two files, three classes, ~150 lines of real code:

| Component | What it does |
|---|---|
| `SmolVLAGoalConfig` | Subclass of `SmolVLAConfig`. Adds `num_goal_images` and `goal_image_key_prefix`. |
| `VLAFlowMatchingGoal` | Adds a learned `goal_type_embedding` parameter of shape `(1, 1, hidden_size)`. Overrides `embed_prefix` to add this embedding to goal-image tokens. |
| `SmolVLAGoalPolicy` | Swaps in the goal-aware model. Overrides `prepare_images` to ensure goal images land at the end of the image list (so `embed_prefix` can identify them by position). |

The learned embedding is ~960 floats — genuinely the smallest possible change that can carry the "this is a goal, not a camera" signal.

## Project status

This is an active research project. Current state:

- ✅ Architecture implemented and tested
- ✅ Smoke test passing (policy builds, parameter shapes correct)
- ⏳ Training pipeline
- ⏳ Dataset recording with goal images (SO-101 cubes-and-boxes setup)
- ⏳ First experiment: goal = final frame of own episode
- ⏳ Ablation: random-episode final frame as goal (should make loss worse if the model is actually using the signal)
- ⏳ Retrieval-conditioned experiments

Updates will land in commits. If you're reading this and some of those are still ⏳, the code works but hasn't been trained on a real task yet.

## License

Apache 2.0. Same as LeRobot and SmolVLM. See `LICENSE`.

## Acknowledgements

Built on top of [LeRobot](https://github.com/huggingface/lerobot) by HuggingFace. SmolVLA paper: [Shukor et al. 2025](https://arxiv.org/abs/2506.01844). Vision backbone: [SmolVLM-2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).

## Contact

Niels Quackels · [GitHub](https://github.com/nielsquackels)