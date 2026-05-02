# Training notes — first vast.ai run (2026-05-02)

Context: first attempt to finetune SmolVLA-Goal on a rented RTX 5090 vast.ai
box. Captures the problems we hit, the fixes, and a clean run procedure for
next time. Not a substitute for the README — assumes you've already done
local Installation and Smoke test from there.

## Quick run procedure (vast.ai, RTX 5090)

```bash
# 1. Clone + install (setup.sh pins torchcodec to a CPU build, sidesteps
#    the CUDA 13 NPP rabbit hole entirely; /venv/main is already active)
cd /workspace
git clone https://github.com/nielsquackels/smolvla-goal.git
cd smolvla-goal
bash setup.sh

# 2. Auth
wandb login                                    # paste W&B API key
hf auth login                                  # paste HF token (write scope)

# 3. Prepare datasets (v2.1 → v3.0 conversion)
python scripts/prepare_datasets.py             # safe to re-run; idempotent

# 4. Smoke test
python tests/test_smoke.py

# 5. Launch training (inside the existing tmux so disconnects don't kill it)
rm -rf outputs/goal_run_0
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

Detach from tmux with `Ctrl+b` then `d`. Reattach with `tmux attach`.

After training:
- Final checkpoint pushed to `huggingface.co/Niels2/smolvla-goal-run0`.
- Intermediate checkpoints in `outputs/goal_run_0/checkpoints/`, also
  uploaded as W&B artifacts every `save_freq=5000` steps.
- **Destroy the vast.ai instance.** Billing is per-second.

## What goes to W&B

Standard LeRobot metrics (loss, lr, grad_norm, step time) plus the goal-
embedding stats added in [smolvla_goal/_train_patches.py](../smolvla_goal/_train_patches.py):

- `goal_emb/weight_norm` — L2 norm of `goal_type_embedding`. Starts near
  zero (init std=0.02), should drift up.
- `goal_emb/grad_norm` — pre-clip L2 norm of the embedding's gradient,
  captured via `register_hook` before `optimizer.zero_grad()` runs.

System stats (GPU util, VRAM, temp) auto-logged by wandb.

## Problems hit, in order

### 1. `--policy.type` and `--policy.path` are mutually exclusive
Initial command used both. Drop `--policy.type=smolvla_goal` if using
`--policy.path=lerobot/smolvla_base`, but that loads the **base** SmolVLA
config from the Hub — wrong shape (expects `camera1/2/3`, doesn't know
about goal images). Real answer: use `--policy.type=smolvla_goal` with
`--policy.pretrained_path=lerobot/smolvla_base`. The first instantiates
our `SmolVLAGoalConfig` (so input_features get inferred from the dataset
and `goal_type_embedding` is registered); the second loads the base
weights into it. Missing-key warning for `model.goal_type_embedding` is
expected and harmless — it's our new parameter, not in the checkpoint.

### 2. Output dir already exists
`FileExistsError` if `outputs/goal_run_0/` exists from a prior failed
attempt. `rm -rf outputs/goal_run_0` between attempts. `--resume=true`
expects a real prior checkpoint and will misbehave if the previous run
crashed early.

### 3. `lerobot[smolvla]` doesn't pull `lerobot[dataset]`
`ImportError: 'datasets' is required`. Install both extras:
`pip install -e "lerobot/[smolvla,dataset]"`. If more `require_package`
errors appear, fall back to `lerobot/[all]`.

### 4. `huggingface-cli` no longer ships
The CLI was renamed to `hf` in recent `huggingface_hub`. Use
`hf auth login` (or `pip install -U "huggingface_hub[cli]"` first if even
that's missing). The `git-credential` warning is harmless — `huggingface_hub`
uses the token directly for API pushes.

### 5. `prepare_datasets.py` "conversion failed: already v3.0"
Benign idempotency complaint. `convert_dataset_v21_to_v30` refuses to
re-convert an already-converted local copy. Real run fails only if a
dataset wasn't converted at all, not if it complains about being
already-converted.

### 6. `ud-smart-city` has no v3.0 git tag on Hub
Its `meta/info.json` claims v3.0 (so prepare_datasets skipped it), but
the repo has no `v3.0` git tag, so `LeRobotDataset.get_safe_version`
fails. A `huggingface_hub`/`lerobot` version mismatch made the resulting
`RevisionNotFoundError` blow up with a confusing `TypeError`. Drop
`ud-smart-city/lerobot-so-101-manipulations` from
[configs/training_data.yaml](../configs/training_data.yaml) — the README
already documented one such skip.

### 7. `torchcodec` requires CUDA 13 NPP that isn't in the container
```
OSError: libnppicc.so.13: cannot open shared object file
```
The vast image had `torchcodec==0.10.0+cu130` but no CUDA 13 NPP runtime
libs on `LD_LIBRARY_PATH`.

**Fixed permanently** — `setup.sh` uninstalls the cu130 wheel and reinstalls
torchcodec from the PyTorch CPU index. Video decode runs in dataloader
workers, so CPU is fine at our scale.

`--dataset.video_backend=pyav` does NOT bypass this; torchcodec is
imported at module load time before any backend setting.

### 8. att/pad mask length mismatch (355 vs 321)
```
RuntimeError: The size of tensor a (355) must match the size of tensor b (321)
at non-singleton dimension 2
```
Our `make_dataset` bypassed `lerobot.datasets.factory.resolve_delta_timestamps`,
so each batch row had `actions` of shape `(action_dim,)` instead of
`(chunk_size=50, action_dim)`. Flow matching's `embed_suffix` then built
a malformed suffix where `att_masks` (length=`chunk_size`=50) didn't
match `pad_masks` (length=`action_emb.shape[1]`).

Fix: `_build_sub_dataset` now calls `resolve_delta_timestamps(policy_cfg,
meta.meta)` per-dataset and passes through to `LeRobotDataset(...,
delta_timestamps=...)`.

### 9. `KeyError` in `_absolute_to_relative_idx`
```
KeyError: 62305  (and later: 46962)
```
Hit once delta_timestamps started exercising action-chunk lookups. Real
root cause:

LeRobot's `DatasetReader._get_query_indices`, `_query_videos`, and
`LeRobotDatasetMetadata.get_data_file_path` / `get_video_file_path` all
do `meta.episodes[ep_idx]` — **positional** indexing into the episodes
HF Dataset. That assumes episode rows are stored in `[0, 1, 2, ..., N-1]`
order. Several community datasets (and v2.1→v3.0 conversions) store
them out of order, so the lookup returns the wrong episode's
`dataset_from_index/dataset_to_index`. The action-chunk clamp then
produces an absolute index from the wrong episode's range; that index
isn't in `_absolute_to_relative_idx` (which only contains frames from
*selected* episodes), and the dataloader crashes.

Fix: `_EpisodeByValueLookup` proxy in `_train_patches.py` swaps in for
`meta.episodes` and re-routes non-negative int lookups through a
precomputed `{episode_index_value: row_position}` dict. Slices, negative
ints, and column-name strings pass through. All four LeRobot callers
benefit transparently.

This problem was masked at first because dropping converted v2.1
datasets reduced (but didn't eliminate) the rate of out-of-order episode
tables. The proxy fixes it for natively-v3.0 datasets too.

## Improvements to make next

- **Upstream the `_EpisodeByValueLookup` fix** to LeRobot — it's a real
  bug there, not specific to our pipeline. The four call sites
  (`_get_query_indices`, `_query_videos`, `get_data_file_path`,
  `get_video_file_path`) should look up by value, not position.
- **Patch `prepare_datasets.py` to verify the converted output's episode
  ordering**, or fix the converter to write episodes in
  `episode_index` order so the LeRobot bug doesn't bite.
- ~~**Add a hf_token-write check** in `prepare_datasets.py` or a
  preflight, so users find out *before* a 30k-step run completes that
  the final push will fail.~~ ✓ `_check_hf_write_token` in `train.py`
  validates both write scope and that the `--policy.repo_id` namespace
  is owned by the logged-in user (or one of their orgs).
- **Replace `--dataset.repo_id=<dummy>` shim** in `train.py`. We inject
  a fake `--dataset.repo_id` to satisfy draccus. Cleaner: register a
  custom `cfg.dataset.type=multi_yaml` that draccus accepts directly
  and skip the monkeypatch.
- **Document the canonical command line** somewhere checked-in (this
  doc is a start). The required combination of `--policy.type` +
  `--policy.pretrained_path` + `--policy.repo_id` + W&B flags isn't
  obvious from the README.
- **Memory check before scaling batch_size**. We didn't measure VRAM at
  batch=16 on the 5090; might be able to push to 24 or 32 and reduce
  steps proportionally.
- **Sanity-check `num_episodes=800` printout**. That's the *total* across
  full unfiltered metas summed across sub-datasets, not the count we
  actually train on. Misleading. `_ConcatMeta` should report the
  sampled-episodes count.
- **Add an integration test** ✓ `tests/test_integration.py` covers
  issues 8 and 9 with a tiny fake multi-dataset recipe.
