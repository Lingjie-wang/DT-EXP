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
- Evaluate fixed RTG 12000 every 20k, 100 episodes, seed 42, constant delayed RTG.
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
