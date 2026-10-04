# Single-step reward C, appended after the dense RTG campaign

This independently named experiment ranks candidate actions by original `r[t]`,
instead of remaining return. It preserves the historical delayed and dense RTG
implementations, configs, output directories and running queue. No V/Q model,
reward decomposition, extra environment data or additional hyperparameters.

## Fixed experiment

- HalfCheetah-medium-replay-v2, original rewards, gamma 1, 202 x 1000 transitions.
  Dataset SHA256: `48d494a4770c11f48260736dec090b78bdca8375647ed4d24bfa5b6610c7f683`.
- At each timestep, select the upper/lower 30% of ORIGINAL SINGLE-STEP REWARDS.
  For every upper-group state, find the nearest lower-group state at that same
  timestep; standardized full-state RMSE <= 0.5. Require equal horizons. Tied
  groups are skipped, not given artificial preferences. Negative rewards are
  ranked relatively. Past/future rewards, RTGs, prefixes and action differences
  cannot change the mining label or distance. RTG disagreement is diagnostic only.
- Keep dense DT's recorded RTGs in ordinary training and its decreasing desired
  RTG at evaluation. The auxiliary positive context uses desired initial 12000
  minus actually observed prefix rewards. Both actions are scored in this same
  positive context. Single-step rewards change ONLY the pair selection criterion,
  not the DT conditioning target or the evaluation objective.
- Keep `DT_MSE + 0.05 * mean(relu(d_pos - stopgrad(d_neg) + 0.05))`, including
  inactive pairs. Negative actions gate positive imitation; they do not receive
  a repulsion gradient. Uniform pair sampling with replacement, batch 256.
- Seed 0 from scratch; ordinary batch 4096, sequence 20, architecture 128/3/1,
  LR 0.0008, warmup 10k, math SDPA, 100k total updates, checkpoints every 5k.
- Evaluate every 10k, 100 episodes, seed 42, initial RTG 12000. Primary endpoint
  is final 100k normalized score; secondary is mean of 60k/80k/100k. No best
  checkpoint selection. This remains a single-training-seed pilot.
- Reuse the already queued dense DT control. Verify its configuration, initial
  model, dataset, runtime versions and shared training source hashes against the
  new run. No second full DT baseline is needed. Only the C label source changes
  relative to dense RTG C, so the resulting pair pool/coverage also changes.

Immediate reward may favor short-term behavior. This experiment tests whether
that local preference improves total policy return; it does not assume immediate
reward identifies long-term action advantage or removes all state-matching bias.

## Pair audit

The fixed miner yields 2,011 pairs out of 61,000 upper-group candidates (3.30%
coverage), spanning 186 positive and 166 negative trajectories. Mean state RMSE
0.43065; mean raw immediate-reward gap 2.27385 (minimum 0.29931).
21.63% of pairs disagree with remaining-return ordering. Only 13.08% of these
ordered pairs also appear in the RTG miner. Do not relax thresholds to match the
RTG pool's 7,306 pairs or tune using policy evaluation outcomes.

## Execution and visibility

Entry point: `algorithms/offline/step_reward_preference_dt.py`.
Config: `configs/offline/dt/halfcheetah/step_reward_preference_5090.yaml`.
Reuse the separate Python 3.10.21 / Torch 2.7.1+cu128 runtime documented in
`docs/state_only_c_5090.md`; set `STATE_ONLY_RUNTIME_BASE` to its original project.
The launcher loads W&B credentials from the existing ignored credential file.
No datasets, checkpoints, run artifacts or secrets belong in Git.

W&B: `2820402607-shandong-university/CORL-DDR`, group
`StepRewardPreference-HCMR-5090-20261005`. Upload training metrics every 100 steps,
evaluations plus per-episode tables every 10k, and a pair artifact containing
indices, state distances, SINGLE-STEP reward gaps, statistics and provenance.

```bash
export STATE_ONLY_RUNTIME_BASE=/home/sckd02/workspace/DT-EXP
bash scripts/step_reward_preference/run.sh --output_dir /absolute/new/output
```

The independent appended queue waits for `dense-preference-5090-seed0-20261005`
to complete BOTH dense RTG C and DT at 100k, including final evaluations and the
comparison file. It also waits for that queue process to exit. Any prior failure
halts the appended queue. An exclusive lock prevents duplicate launches; output
directories are never overwritten or silently restarted.

Server paths under `/home/sckd02/workspace/DT-EXP`:

- Frozen source: `.runtime/step-reward-preference-source-20261005`.
- Campaign: `results/step-reward-preference-5090-seed0-20261005`.
- Tmux session: `step-reward-preference-queue-20261005`.
- Log: `logs/step-reward-preference-queue-20261005.log`.

```bash
python3 scripts/step_reward_preference/queue.py \
  --dependency /home/sckd02/workspace/DT-EXP/results/dense-preference-5090-seed0-20261005 \
  --dependency-commit e6401b2891235ef078354ff4ca73b9db9ad34eaf \
  --dependency-queue-pid 601613 \
  --output-root /home/sckd02/workspace/DT-EXP/results/step-reward-preference-5090-seed0-20261005
```

After the dependency completes, run a full-batch CUDA smoke (three updates and
one real episode, offline W&B), validate baseline compatibility, then run C.
Write a final comparison against both existing dense RTG C and DT. The appended
queue survives SSH disconnects, but a server reboot requires explicit inspection.
`queue_manifest.json` records exact source and dependency process identities;
`queue_status.json` distinguishes waiting, smoke, training, failed and completed.

## Validation

`tests/test_step_reward_preference.py` covers immediate/remaining label reversals,
invariance to other timesteps and RTG fields, nearest-state matching, action and
prefix independence, negative/tied rewards, invalid data, complete-campaign
dependency checks, and baseline configuration/shared-code compatibility. Reuse
the dense-context and detached-gradient tests. Run Ruff 0.0.278 against a clean
tracked export before publishing; verify GitHub Actions after publication.

## Deployment verification (2026-10-05, Asia/Shanghai)

Execution revision `2775c67e5cd15befbc0109bf745295db816f808d` was published before
deployment; [GitHub Actions passed](https://github.com/Lingjie-wang/DT-EXP/actions/runs/37217587785).
All 26 relevant step-reward/dense-context/legacy-gradient tests and the clean
tracked export's Ruff check passed. CPU C and zero-weight DT checks each finished
two updates and one real episode. The zero-weight model after TWO updates was
tensor-for-tensor identical to the previous dense DT CPU check, and the control
compatibility guard passed. These checks do not establish a policy improvement.

[W&B validation run](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/80nbsj0y)
is in `StepRewardPreference-Validation-20261005`, separate from formal results.
API readback confirmed `preference_label=step_reward`, 2,011 pairs, step-2
evaluation, the episode table and the single-step reward pair artifact.
The appended tmux queue was installed to wait for the complete prior campaign;
its manifest records the exact dependency commit and process identity. Full-batch
GPU validation and formal seed-0 C remain behind the prior experiments.
