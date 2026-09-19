# Original v3-high vs paired DT: fresh seeds 3, 4, 5

## Preregistered protocol (2026-09-19)

Do not repeat historical training seeds 0, 1, 2. This campaign accepts only
**3, 4, 5** and never initializes them from another seed's checkpoint.
Dataset: `halfcheetah-medium-replay-v2`, delayed terminal-sum rewards.

For each seed:

1. Train ordinary DT from scratch to exactly 50,000 updates; save complete
   checkpoints at 0, 10k, 20k, 50k. The existing `top_return_weighted_dt.py`
   entry point with `top_weight=1.0` has exactly ordinary DT's action-MSE
   objective and sampling; no high-return weighting is enabled.
2. Generate original strict v3 pairs using that seed and its 50k model.
   Keep 100,000 initial candidates, 64 negative candidates, top/bottom 30%
   trajectories, 25-step buckets and closest-state 25% retention. Apply the
   original five-state/four-action prefix checks (0.75/0.50), time gap <= 5,
   action RMSE >= 0.25 and trajectory-return gap >= 2,000. Use all strict valid
   pairs, not only initially active ones. Pair counts can differ by seed.
3. Fork the SAME seed-specific 50k checkpoint into DT control and original
   v3-high; both continue to exactly 100k using `hard_fork_dt.py` with
   `paired_resume=true`. DT auxiliary weights are both zero; v3 preference
   weight is 0.05 and reference weight is 0.1.

Original v3-high remains unchanged: margin 0.05, batch 256, active-normalization
floor 16, frozen-hardness temperature 0.05, dynamic-priority mix 0.5 and EMA 0.9.
Preference and reference use RTG 12000 only; ordinary DT loss uses recorded RTG.
Reference is frozen at this seed's shared 50k model and is never refreshed.
This is NOT the recent-reference/adaptive method. The strict pair miner and
frozen hardness source are rerun per seed, rather than importing seed-0 pairs.

Use the unchanged CORL config: batch 4096, context 20, hidden 128, 3 layers,
1 head, dropout 0.1, AdamW LR 0.0008, warmup 10k, weight decay 0.0001,
clip 0.25. Formal runs require RTX 4090; both branches match their warmup's
GPU model. Nondeterministic Torch kernels remain enabled as originally.

## Fairness and validation

Both branches restore model, optimizer, scheduler, global RNG and the same
DataLoader generator. Both execute auxiliary forwards, so the DT control has
the exact ordinary-DT gradient while preserving the same dropout consumption.
Training data is independent of preference-pair draws. Worker prefetch queues
are not serialized: paired fresh continuation is matched between arms, but is
not a bitwise reproduction of uninterrupted 0-to-100k DT.

A full-batch GPU smoke covers ALL three new seeds before any long training.
It trains each ordinary DT for two steps, mines pairs, then compares three
continuation steps with intervening evaluation. Each real 50k checkpoint also
passes a separate three-update offline gate before the long branches start.
Audits check source hashes, distinct seed initializations, shared weights,
optimizer LR/scheduler, first three DT minibatch and dropout-RNG hashes,
first DT loss, exact zero control auxiliary terms, finite metrics, update counts
and the unchanged frozen reference. Smoke outputs never initialize production.

The orchestrator only journals metrics and starts existing entry points. SHA256
pins protect all five underlying algorithm files; no old algorithm or launcher
is edited. Results/checkpoints use separate campaign directories. Existing
directories make duplicate launches fail rather than overwrite earlier work.

## Evaluation and records

Delayed evaluation keeps policy RTG constant; normalized scores use the original
environment's total episode return. Evaluate targets 12000 and 6000, each with
100 episodes and eval seed 42, every 5k. Eval RNG is isolated from training RNG.
Predeclare primary outcomes: 100k last and mean over 55k, 60k, ..., 100k;
secondary: best over those same ten points. Also report the common 50k start
and 75k intermediate results. Do not select different evaluation windows by arm.

W&B project: `CORL-DDR`; group:
`V3High-FreshSeeds345-Paired50kTo100k-HCMR-delayed`.

- `V3High-Fresh345-SharedDTWarmup50k-seed{3,4,5}-HCMR-delayed`
- `V3High-Fresh345-DTControl-seed{3,4,5}-To100k-HCMR-delayed`
- `V3High-Fresh345-V3High-seed{3,4,5}-To100k-HCMR-delayed`

Existing trainers append environment and unique IDs to names. There are six
comparison runs plus three shared warmup records, not nine independent methods.
W&B receives metrics, configuration and provenance only; no checkpoints or raw
state/action arrays. Pair diagnosis has W&B disabled; short gates are offline.

Campaign: `v3-high-fresh345-20260919`.
Local-to-server records: `results/<campaign>/production/seedN/`, including
`prepared.json`, run identities, per-update metrics and `final_pair_audit.json`.
Checkpoints: `checkpoints/<campaign>/production/seedN/`.

## Launch

Run lightweight tests with:

```bash
bash scripts/dt_experiments/run_v3_high_fresh345.sbatch tests
```

Submit a GPU `smoke` job, a `prepare` array (indices 3-5) depending on smoke
success, then one `branch SEED` array (indices 0=DT, 1=v3) per seed depending
on its preparation task succeeding. Formal jobs exclude non-4090 nodes.
Each job requests one GPUNorm GPU, 6 CPUs and 48 GB RAM. Account running-job
limits and existing experiments are respected; no existing job is stopped.
Long jobs cannot start after a failed prerequisite. No recurring monitor is set.
