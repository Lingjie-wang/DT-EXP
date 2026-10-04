# C-only state-matched action preference on the RTX 5090

The user selected only C from the proposed three-arm experiment. Do not launch
A or B. Existing DT and hard-positive runs are historical references, not matched
controls: old hard-positive training resumed at 50k and retained a reference
penalty, confidence weighting and the old strict pair pool. This run cannot
establish a controlled improvement over A/B by comparing their old scores.

## Fixed method

- HalfCheetah-medium-replay-v2; transform each trajectory's rewards to a terminal
  sum, gamma 1. No dense reward labels are used by the learner or pair miner.
- Top/bottom 30% by trajectory return, defined with q70/q30 thresholds.
- For every high-return state, find the exact nearest low-return state at the
  SAME timestep using full-state standardized RMSE. Retain distance <= 0.5.
  At most one pair per positive state; deterministic trajectory-order tie breaks.
- No prefix matching, action-distance/return-gap filter, reference model,
  confidence weights, difficulty mining, online priorities or active normalization.
- Uniform sampling with replacement, auxiliary batch 256. Standard DT retains
  its 20-step context and recorded RTG. The auxiliary branch uses the positive
  context at RTG 12000 (scaled to 12), predicting its current action.
- `loss = DT_masked_MSE + 0.05 * mean(relu(d_pos - stopgrad(d_neg) + 0.05))`.
  The mean includes inactive pairs. Negative actions only control the gate.
  Neither high trajectory return nor Markov dynamics proves a local causal label.
- Train from random initialization for 100,000 completed updates, seed 0 only.
  Ordinary batch 4096; CORL architecture/AdamW/dropout/LR warmup are unchanged.
  There is no checkpoint warm-start or delayed activation of the auxiliary loss.
- Evaluate fixed RTG 12000 every 10k, 100 episodes, seed 42, constant delayed RTG.
  Primary: final 100k score; secondary: mean of the 60k/80k/100k scores. No best
  checkpoint selection. Environment episode dispersion is not training-seed error.

## Independent environment and execution

Use Python 3.10 and PyTorch 2.7.1 with CUDA 12.8 for RTX 5090 support. This differs
from the historical Python 3.9 / Torch 1.11 environment. Keep historical learners,
configs, checkpoints and results unchanged. This is a new experiment, not an
unmodified historical reproduction. Do not install into another project's env.

```bash
uv venv --python 3.10 --seed .runtime/state-only-c-5090-env
uv pip install --python .runtime/state-only-c-5090-env/bin/python \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .runtime/state-only-c-5090-env/bin/python \
  -r scripts/state_only_c/requirements.txt
uv pip install --python .runtime/state-only-c-5090-env/bin/python --no-deps \
  d4rl==1.1 \
  'mjrl @ git+https://github.com/aravindr93/mjrl@3871d93763d3b49c4741e6daeaebbc605fe140dc'
```

Extract MuJoCo 2.1 into `.runtime/state-only-c-5090-install/mujoco210`; place the
HDF5 in `.runtime/state-only-c-5090-install/datasets/halfcheetah_medium_replay-v2.hdf5`.
Dataset SHA256: `48d494a4770c11f48260736dec090b78bdca8375647ed4d24bfa5b6610c7f683`.
MuJoCo archive SHA256: `a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012`.
System OpenGL/OSMesa build dependencies may be needed; record any compatibility
work and verify actual CUDA forward/backward plus a real simulator rollout.

The tested runtime is Python 3.10.21, Torch 2.7.1+cu128 and W&B 0.30.0.
`scripts/state_only_c/requirements-lock.txt` records the installed packages;
the local Torch wheel path is replaced with its public version. Install Torch
from the CUDA 12.8 index above before applying the lock. The official Linux
CPython 3.10 Torch wheel SHA256 is
`d6c3cba198dc93f93422a8545f48a6697890366e4b9701f54351fc27e2304bd3`.
The final runtime passes `pip check`.

Compatibility record: the first smoke failed before training with
`No module named 'six'` during D4RL's MuJoCo registration, followed by
`gym.error.NameNotFound: Environment halfcheetah-medium-replay doesn't exist`.
Installing `six==1.17.0`, imported by MJRL, resolved it. D4RL's declared
dm-control/MuJoCo/PyBullet dependencies are pinned above. No upstream source
patches or monkey patches were applied. Missing optional Flow/CARLA imports
do not affect this task.

The retry `state-only-c-5090-smoke-20261004-v2`, using source `8570b99`,
completed three CUDA updates with the full 4096/256 batches and a real
1000-step HalfCheetah rollout. Losses and gradients were finite; the pair pool
contained 6,768 pairs. Six core tests passed on the server. This is execution
validation only, not an experiment score.

```bash
.runtime/state-only-c-5090-env/bin/python -m unittest discover \
  -s tests -p test_state_only_preference.py -v
bash scripts/state_only_c/run.sh --output_dir results/state-only-c-5090-smoke \
  --update_steps 3 --eval_every 3 --eval_episodes 1 --log_every 1 \
  --wandb_mode offline
bash scripts/state_only_c/run.sh --output_dir results/state-only-c-5090-seed0-20261004
```

Every output directory must be new. Save full model/optimizer/scheduler/RNG
checkpoints every 5k. Worker prefetch queues are not serialized, so exact later
continuation is not claimed. Keep checkpoints and generated data out of Git.
When launching from a frozen Git worktree, set `STATE_ONLY_RUNTIME_BASE` to the
original project directory containing this new runtime and the W&B login, and
pass an absolute `--output_dir`. This leaves the original working tree intact.

The initial offline audit found 6,768 pairs from 61,000 high-return states
(11.095% coverage), covering 61 high-return and 55 low-return trajectories.
Mean state RMSE was 0.41922 and maximum 0.49999. Six unit tests verify same-time
nearest matching, the distance cutoff, prefix independence, context construction
and the exact detached-negative gradient with a whole-batch denominator.

## Observability and publication

W&B: `2820402607-shandong-university/CORL-DDR`, separate C-only group. Log losses,
positive/negative action MSE, active fraction and progress every 100 updates;
log evaluation scores and per-episode returns at every evaluation. Publish only
experiment configuration, metrics and public version/hash metadata, not secrets
or complete private filesystem/environment dumps. Local JSONL, pair statistics,
per-episode results, immutable checkpoints and provenance remain available.

Commit/publish each completed source change promptly; do not rely on continued
server access. Before pushing, run Ruff 0.0.278 on a clean tracked export and
inspect GitHub Actions after publication. Keep unrelated working-tree changes.

## Launched run (2026-10-04)

- W&B: [C seed 0, live metrics](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/xl40z503).
- Execution source: `2e75f26aa388a47ab8d0fa40ccab54d6eb566759`, published to
  GitHub before launch; its [CI passed](https://github.com/Lingjie-wang/DT-EXP/actions/runs/37209976867).
- Server project: `/home/sckd02/workspace/DT-EXP`; frozen source worktree:
  `.runtime/state-only-c-source-2e75f26`. No changes to the original worktree.
- Output: `results/state-only-c-5090-seed0-20261004` under that server project.
  Log: `logs/state-only-c-5090-seed0-20261004.log`.
- Detached tmux session: `state-only-c-seed0-20261004`. Training and W&B use
  direct connections and do not depend on the installation proxy or SSH session.
- Startup audit: full-batch CUDA smoke, one 1000-step rollout, checkpoint state
  and W&B episode table verified. Formal training progressed past 1,300 updates
  with finite metrics. W&B API readback independently verified steps 1/100/200/300,
  6,768 pairs and the exact execution commit. This records launch verification,
  not a completed 100k run or a final policy score.

Historical B reference:
[hard-positive seed 0](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/f3737b2b-6aeb-444f-b9ea-6e60e962f156).
Its saved config confirms `pretrained_checkpoint_path` at 50k,
`preference_start_step=50000` and `reference_weight=0.1`; it is not the proposed
from-scratch, unweighted B with the new state-only pair pool. Only C was launched.

## Interrupted first run and recovery

The initial `xl40z503` run stopped at update 8,043 before applying the optimizer
step: `clip_grad_norm_(error_if_nonfinite=True)` detected a nonfinite gradient
norm. Its final logged loss at 8k was finite (DT 0.06106, preference 0.01030).
The server had ample disk/RAM and the process exited with code 1; W&B finished
uploading the failure run. The latest periodic checkpoint is step 5,000.
Keep that run and its frozen source/output directory intact.

At the user's request, evaluations now run every 10k updates, 100 episodes each.
Final 100k remains primary, and `late_60_80_100k_mean` keeps the original secondary
60k/80k/100k statistic. The diagnostic `last_three_eval_mean` now represents the
last three evaluations (80k/90k/100k under the new cadence).

Recovery uses `--resume_checkpoint` with a NEW output directory and W&B run.
It validates the method, pair pool and normalization, restores model/optimizer/
scheduler and saved RNG state, and retains total update numbering. Worker
prefetch state was not saved, so new worker streams are used: this is a disclosed
recovery branch, not an exact replay. Failure checks stay enabled; a future
backward failure saves its inputs and the last valid optimizer checkpoint before
exiting. Failure details are also uploaded to the W&B summary.

`state_only_gradient_diagnostic.py` runs an isolated replay from a checkpoint,
saves the first bad batch and model, and compares default versus math SDPA
backward passes on the captured inputs. Its outputs are diagnostic artifacts,
not additional training seeds or experimental results.

The isolated replay reproduced the failure at total update 8,156 on Torch
2.7.1+cu128 / RTX 5090. The captured parameters and per-element gradients were
finite, but the default attention backward produced a gradient L2 norm of
1.108e20 (measured in float64); the float32 norm reduction overflowed. Replaying
the SAME model/input with math SDPA yielded norm 1.770. Ordinary DT alone gave
1.108e20 versus 1.714; preference alone gave 933.3 versus 0.0725. Different backend
dropout draws can change exact losses, so these are diagnostic comparisons, not
claims of bitwise equivalence. The issue points to the accelerated attention
backward in this runtime, rather than a nonfinite dataset or the preference loss.

A further controlled replay disabled every dropout ONLY in the diagnostic model:
efficient SDPA still gave norm 7.983e15, while math SDPA gave 1.817. Thus the huge
gradient difference persists without dropout randomness. The captured batch,
model and JSON reports are retained in the ignored diagnostic output directory.

The recovery explicitly sets `attention_backend: math` using public PyTorch
backend controls, disabling accelerated SDPA paths. No upstream DT source is
patched. Architecture, batches, losses, learning rate, clipping and evaluation
protocol remain the same apart from the requested 10k interval. Floating-point
arithmetic and dropout random streams can differ; the recovered run records this
compatibility change and its checkpoint lineage. Do not disable the nonfinite
gradient guard, zero bad gradients, or skip bad updates to hide this failure.

Validation: all eight unit tests passed on the server and locally; the clean
tracked export passes Ruff 0.0.278. Recovery smoke completed updates 5001–5003
with full 4096/256 batches and a real 1000-step rollout. The recovery's starting
model and every optimizer tensor exactly matched the original step-5000 file,
and scheduler state matched as well. This smoke's single episode is not the
formal 100-episode evaluation.
