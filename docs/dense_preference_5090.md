# Original-reward action preference: matched DT and C

This independent experiment tests whether the existing single-sided C objective
helps DT when labels can use original per-step rewards. Historical delayed code,
configs, results and running jobs are preserved. Do not replace their outputs.
This is our method, not a paper reproduction. No reward model or critic is added.

## Fixed comparison

- HalfCheetah-medium-replay-v2, original rewards, gamma 1; 202 equal-length
  trajectories of 1000 steps. Dataset SHA256:
  `48d494a4770c11f48260736dec090b78bdca8375647ed4d24bfa5b6610c7f683`.
- At EACH timestep separately, rank the observed remaining returns
  `G[t] = sum(rewards[t:])`; take the upper/lower 30% as candidate groups.
  For each upper-group state, find the nearest lower-group state at the SAME
  timestep using standardized full-state RMSE; keep distance <= 0.5.
  Equal horizons are required. Ties that collapse quantile groups are skipped.
  No prefix/action-gap filters, new return-gap threshold or confidence weights.
- The ordinary DT branch uses recorded original-reward RTG tokens.
  Both candidate actions are compared in the positive state's 20-step context.
  Auxiliary RTG at time t is `12000 - sum(positive_rewards[:t])`, scaled by 0.001.
  Thus preceding tokens have consistent decreasing reward budgets, matching
  original-reward evaluation. Do not clamp negative budgets or leak future
  rewards into the desired-return tokens. Prefixes are never matching criteria.
- DT control: ordinary masked action MSE. C: the SAME MSE plus
  `0.05 * mean(relu(d_pos - stopgrad(d_neg) + 0.05))` over all auxiliary samples.
  Negative actions gate positive imitation; there is no negative repulsion
  gradient. Both arms compute the auxiliary branch so RNG draw schedules match;
  its coefficient is zero in DT. Log this diagnostic loss for both arms.
- Both start from seed 0, random initialization, ordinary batch 4096 and
  auxiliary batch 256, 100k updates. Same architecture, optimizer, LR 0.0008,
  warmup 10k, math SDPA, clipping, data workers and pair sampler.
- Evaluate initial desired return 12000 every 10k updates, 100 episodes with
  evaluation seed 42. Subtract the observed reward from RTG after each step.
  Primary endpoint: final 100k score; secondary: mean of 60k/80k/100k scores.
  No best-checkpoint selection. This first pair of runs is a single-seed pilot;
  episode standard deviation is not training-seed uncertainty.
- Checkpoint every 5k, preserving full optimizer/scheduler/RNG and provenance.
  New output directories are mandatory. Explicit recovery may use a new run;
  worker prefetch state is not captured, so exact replay is not claimed.

This comparison estimates the effect of adding C under dense-reward supervision.
It does NOT isolate negative gating from additional positive-action imitation
(that requires a matched B), or isolate reward mode from the changed label pool
by comparing to the historical delayed C. Original RTG removes past rewards but
still includes future behavior-policy effects and imperfect state matching;
the labels are not counterfactual action advantages.

## Initial offline audit

The original HDF5 yields 7,306 pairs from 61,000 upper-group states (11.98%
coverage), spanning 90 positive and 71 negative trajectories. Mean state RMSE
0.41506; mean raw remaining-return gap 1821.45, minimum 34.09. 69.29% of ordered
pairs also occur in the old trajectory-total miner. Every selected pair still
has the same total-return ordering: these data do not supply many dramatic
reversals, and the dense labels remain correlated with trajectory quality.

## Runtime, publication and W&B

Use the independently installed runtime from `docs/state_only_c_5090.md`:
Python 3.10.21, Torch 2.7.1+cu128 on RTX 5090, math SDPA through public controls.
Set `STATE_ONLY_RUNTIME_BASE` to the project containing that runtime, the dataset
and `.dt_runs/wandb.env`; the launcher loads credentials without printing them.
Do not commit secrets, datasets, checkpoints, or generated artifacts.

The entry point is `algorithms/offline/dense_preference_dt.py`; settings are in
`configs/offline/dt/halfcheetah/dense_preference_5090.yaml`. Run from a frozen,
published source worktree. W&B project is `CORL-DDR` under
`2820402607-shandong-university`, group `DensePreference-HCMR-5090-20261005`.
Upload training metrics every 100 updates, evaluation metrics and episode tables
every 10k; upload the pair indices, distances, gaps, statistics and provenance
as a small W&B artifact. Local JSONL, evaluation JSON and checkpoints also remain.

```bash
export STATE_ONLY_RUNTIME_BASE=/home/sckd02/workspace/DT-EXP
bash scripts/dense_preference/run.sh --variant c --preference_weight 0.05 \
  --output_dir /absolute/new/c-output
bash scripts/dense_preference/run.sh --variant dt --preference_weight 0 \
  --output_dir /absolute/new/dt-output
```

## Queue after the current delayed experiment

`scripts/dense_preference/queue.py` runs in detached tmux on the server. It waits
for delayed recovery run `g2256klc` to report successful completion of 100k updates,
checks the final 100-episode evaluation and summary, and waits for the original
training process to exit. A failure or disappeared process halts the queue.
A campaign lock prevents duplicate launches; incomplete output is never reused.

Queue order: full-batch CUDA smoke for C and DT (three updates plus one real
episode each, W&B offline), then C seed 0, then DT seed 0. The smoke must verify
identical initial models, dataset/pair hashes, source and attention backend.
It writes a local final comparison after both formal runs finish. The queue and
training survive loss of the SSH session, but do not auto-resume a server reboot.

```bash
python3 scripts/dense_preference/queue.py \
  --dependency /home/sckd02/workspace/DT-EXP/results/state-only-c-5090-seed0-recovery-20261004 \
  --dependency-run-id g2256klc --dependency-pid 600060 \
  --output-root /home/sckd02/workspace/DT-EXP/results/dense-preference-5090-seed0-20261005
```

`queue_manifest.json` records the source revision and process identity;
`queue_status.json` records waiting, smoke, training, completed or failed state.
Each trial has a separate log and output directory. Explicit restarts skip only
completed outputs with the same source revision, variant and update budget.

## Validation

Run `test_dense_action_preference.py` and `test_state_only_preference.py` with
unittest. Tests cover label reversal when past return is excluded, nearest-state
matching, tied labels, unequal horizons, causal dense RTG budgets, right padding,
detached-negative gradients, and completion/failure/final-evaluation queue gates.
Before publication, run the repository Ruff 0.0.278 check on a clean export of
tracked files. Inspect GitHub Actions after each push.

## Deployed queue (2026-10-05, Asia/Shanghai)

- Execution source: `e6401b2891235ef078354ff4ca73b9db9ad34eaf`, published before
  deployment; [GitHub Actions passed](https://github.com/Lingjie-wang/DT-EXP/actions/runs/37216138132).
- Frozen server source: `/home/sckd02/workspace/DT-EXP/.runtime/dense-preference-source-e6401b2`.
- Detached tmux session: `dense-preference-queue-20261005`; log under the server
  project: `logs/dense-preference-queue-20261005.log`.
- Campaign output: `results/dense-preference-5090-seed0-20261005`. Verified state
  `waiting`; dependency is the existing `g2256klc` recovery. The original training
  was at 49.2k updates during verification and remained the only GPU process.
- Twenty relevant tests passed: nine dense/queue tests, eight historical C
  gradient/mining tests, and three original/delayed reward/evaluation tests.
  Both CPU smokes completed two updates and one real 1000-step rollout. Initial
  model, dataset, pair and source hashes matched, as did the first DT loss.
- [W&B upload validation](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/fugmaxlf)
  is a two-update CPU smoke in `DensePreference-Validation-20261005`, not a formal
  score. API readback verified evaluation at step 2, the episode table and the
  preference-pair artifact. C's CPU smoke used disabled W&B.
- Full-batch GPU smokes and both formal runs are queued after successful current
  completion. They were not started early; no dense experimental result is yet
  established by this deployment record.
