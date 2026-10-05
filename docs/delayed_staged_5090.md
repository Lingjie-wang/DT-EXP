# Corrected study: delayed-reward staged preference on RTX 5090

The user's target is DELAYED rewards. The earlier `LatePreference` and
`PreferenceOnly` campaigns used original dense rewards and did not answer that
question. Retain those records as outside-scope historical experiments; do not
use their scores to accept/reject the delayed method or diagnose its overfitting.

## Reward contract

- HalfCheetah-medium-replay-v2, seed 0, full fixed dataset, gamma 1.
- Move each trajectory's entire reward sum to its last transition. Require every
  intermediate training reward to be zero and every valid recorded RTG to equal
  that trajectory's terminal return. Audit and hash actual transformed arrays.
- Mine the delayed state-matched pool: upper/lower 30% of terminal trajectory
  returns, exact nearest standardized state at the same timestep, RMSE <= 0.5.
  Require 6,768 pairs. No dense RTG ranking, per-step labels or reward gaps.
- Auxiliary desired RTG is constant 12000 (scaled to 12); never subtract dense
  prefix rewards. Check ordinary and auxiliary RTGs in the running trainer.
- Wrap the simulator to expose zero reward before termination and the episode
  sum at termination. Evaluation uses constant RTG; intermediate dense reward
  never reaches the policy. Report total return / D4RL normalized score.
- Reject original-reward or already preference-trained parents before updates.

## Matched protocol

No completed 100k ordinary delayed CORL DT on the 5090 matches this setup.
`StateOnly-C-seed0-recovery-math` already includes preference training; historical
50k CORL and original-DT GPT-2 checkpoints have different budgets or implementations.
Do not substitute them or the dense parent merely to shorten runtime.

1. Train a new ordinary delayed DT for 100k updates from random initialization,
   no preference gradient, batch 4096, auxiliary diagnostic batch 256, context 20,
   embedding 128 / 3 layers / 1 head, dropout 0.1, AdamW LR 8e-4, warmup 10k,
   clipping 0.25 and math SDPA. Evaluate every 10k (100 episodes, seed 42,
   initial RTG 12000); checkpoint every 5k. Auxiliary forwards are diagnostic.
2. Restore the SAME final model, optimizer moments, scheduler counters and RNG
   into four children. Use identical new worker streams because parent prefetch
   queues are not serialized. Lower LR to 1e-4 in all children; no new warmup.
3. Each child adds exactly 5k updates. Evaluate at additional 0/1k/3k/5k, 100
   episodes each. Primary endpoint: final 5k; no best-checkpoint selection.

| Child | Objective |
|---|---|
| DT | ordinary DT MSE |
| B | DT MSE + 0.05 * positive-action MSE |
| C | DT MSE + 0.05 * mean(relu(d_pos - stopgrad(d_neg) + 0.05)) |
| C-only | 0.05 * mean(relu(d_pos - stopgrad(d_neg) + 0.05)) |

All children execute matching ordinary/auxiliary forwards and sampling schedules.
C-only ordinary MSE is computed under no-grad. Historical AdamW moments and
weight decay remain; the negative action only gates positive imitation.

Require exact equality of restored states, first batch, pair indices, dropout
RNG, initial ordinary/positive/negative losses, reward audits and baseline
episode returns across children. Baseline returns must reproduce the delayed
parent's final evaluation within 1e-6 before updates.

## Execution and validation

Independent modules: `delayed_staged.py`, `delayed_staged_parent_dt.py`,
`delayed_staged_dt.py`; configs `delayed_staged_parent_5090.yaml` and
`delayed_staged_5090.yaml`; scripts under `scripts/delayed_staged/`.
Reuse the isolated Python 3.10 / Torch 2.7.1+cu128 runtime. Preserve historical
code, configs, checkpoints and results; this is our ablation, not a reproduction.

The queue refuses dirty source, a busy GPU or existing campaign/trial directories.
First run a full-batch three-update delayed parent smoke, then four three-update
child smokes. Only these smokes use warmup 1 and one-episode evaluations to test
the already-warmed-up child scheduler; their scores are not policy results.
Only after all audits pass, run the 100k parent and automatically the four 5k
children. Failure stops the queue; never skip failed updates or silently resume.

```bash
export STATE_ONLY_RUNTIME_BASE=/home/sckd02/workspace/DT-EXP
python3 scripts/delayed_staged/queue.py \
  --output-root "$STATE_ONLY_RUNTIME_BASE/results/delayed-staged-5090-seed0-20261005"
```

Use tmux so SSH disconnection does not stop training. Save the queue manifest
and status, full checkpoints, reward audits and per-episode evaluations. On
completion write `children_verified.json` and `comparison.json` for all arms
versus continued DT and the frozen delayed parent.

W&B project `2820402607-shandong-university/CORL-DDR`, group
`DelayedStage-HCMR-5090-20261005`, names `DelayedStage-parent-dt-seed0` and
`DelayedStage-{dt,b,c,c_only}-seed0`. Use separate views for parent 0-to-100k
training and children 0-to-5k comparisons. Upload metrics, episode tables and
audit artifacts; keep credentials, datasets and checkpoints out of Git.

Run reward/evaluation/gradient tests and existing DT reward-mode/state-only/
late/matched tests, then Ruff 0.0.278 on a clean tracked export before publication.
Inspect CI and GPU smoke results. This one-seed ablation does not prove
overfitting as a mechanism or isolate activation timing against from-scratch C
at equal total budget. The historical recovered delayed C is contextual only.

## Deployment record (2026-10-05, Asia/Shanghai)

- Frozen training source: `f34c3bf3751fc7a3ce693663dfa471fad6640153`.
  All 34 relevant unit tests and clean-export Ruff passed. GitHub Actions
  [37290091160](https://github.com/Lingjie-wang/DT-EXP/actions/runs/37290091160)
  also passed.
- All five full-batch GPU smoke runs passed, including matching child initial
  model/optimizer/RNG, batch/pair/dropout fingerprints, initial losses and
  baseline episode returns. `smoke_verified.json` records all checks as true.
- The formal ordinary delayed DT started at about 17:31. Its W&B run is
  [a2sn7vr3](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/a2sn7vr3).
  A launch check observed 1,900 completed updates with finite loss. This is
  progress only: the parent and four formal children are not yet completed.
- Actual training data audit: 202 trajectories / 202,000 transitions; every
  nonterminal reward is zero and every valid trajectory RTG is constant.
  Transformed reward/RTG SHA256:
  `13bafcb5b8869f97a87332fb73e9c3600cc80dcdac094ea845fceb6732fb6c3f`.
  The runtime also verified ordinary and auxiliary batch RTGs. The pair pool
  has 6,768 pairs; W&B config confirms `reward_mode=delayed`,
  `variant=parent_dt`, `preference_weight=0`.
- Host campaign directory:
  `/home/sckd02/workspace/DT-EXP/results/delayed-staged-5090-seed0-20261005`.
  Frozen source directory: `.runtime/delayed-staged-source-f34c3bf` under the
  same project. tmux session: `delayed-staged-queue-20261005`.
  Queue log: `logs/delayed-staged-queue-20261005.log`.
  The queue automatically runs the four 5k children after the parent completes.
- Saved W&B views:
  [ordinary delayed DT, 100k](https://forge.coreweave.com/wandb/2820402607-shandong-university/CORL-DDR/workspace?nw=2j7s3f0k3qc)
  and [delayed preference continuations, 5k](https://forge.coreweave.com/wandb/2820402607-shandong-university/CORL-DDR/workspace?nw=2ijs0nhgx97).
  The second view is preconfigured and remains empty until the children start.
- Historical original-reward runs `g9fe4dp1`, `c1amwyrf`, `zie5w3hn`,
  `3mn1pfgo` retain their metrics and names, with appended scope-correction
  notes and tags `reward-original`, `outside-delayed-scope`.
  Their [saved view](https://forge.coreweave.com/wandb/2820402607-shandong-university/CORL-DDR/workspace?nw=8gjbrpa85gn)
  is now explicitly titled `5090 · 原始逐步奖励（非延迟）· 后加偏好 5k`.
