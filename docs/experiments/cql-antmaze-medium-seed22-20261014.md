# Medium AntMaze: six paired CQL runs after the current Umaze campaign

User request, 2026-10-14: run `antmaze-medium-play-v2` and
`antmaze-medium-diverse-v2`, one seed each, with original, uniform and predictive
Shapley rewards. Upload all results to W&B. Finish code and queue the experiments
behind the current experiments; do not start learning ahead of them.

## Fixed design and budget

- Seed **22** for all six runs. Both servers were checked before reservation:
  the 5090 AntMaze plans reserve seed 21 only; Slurm has no active user jobs and
  no AntMaze jobs in its saved plans/protocols. No Medium seed was reserved.
- **1,000,000 updates**, evaluations every **50,000 updates**, **100 episodes**.
  This preserves the official CORL budget. The prior Umaze curves continue to
  improve or fluctuate at 850k-950k, so shortening the budget is not justified.
- Use the unchanged official CQL source at CORL
  `6afec90484bbf47dee05fdf525e26a3ebe028e9b`, each task's unchanged official YAML,
  and D4RL `db6e4b34bb5ce2a51dd3879177c0a0223208a614`.
  Seed, logging/output paths and device are explicit CLI selections.
- Preserve default loader filtering and success terminals. Split raw data at
  success OR timeout, then use exactly official retained transitions.
- Reuse the prior variable-length segment Shapley code without scientific edits:
  state/action return predictor, up to 20 segments, 128 permutations, five folds
  grouped by parent timeout fragment, signed rewards and uniform residual.
  Predictor checks remain report-only. Success timing remains available.
- Change only rewards in independent HDF5 copies. Original rewards use the
  untouched downloaded file. Apply official `10*r-5` inside unchanged CQL.
- Report **best observed evaluation**, alongside every evaluation curve.
  A single seed is exploratory and does not establish statistical significance.

## Durable dependency and resource rules

Predecessor:
`results/cql-antmaze-success-rewards-seed21-1m-5090-20261013`.
The barrier runs before predictor fitting, CPU/GPU training preflights or formal
CQL. Require every predecessor job to be terminal, its trainer process exited,
all expected evaluations present, and its final checkpoint present. A nonzero
W&B shutdown exit is accepted only when these independent training checks pass;
its original exit code stays recorded. Incomplete training blocks the queue.

At reservation, the two prior Shapley runs were approximately 900k and 850k;
both uniform runs had complete 1M artifacts despite W&B shutdown failures.
The shared GPU had six foreign processes plus our two trainers, about 8 GiB
free, and high utilization. No existing process is interrupted.

After the barrier, retain shared-resource admission: at most two own CQL jobs,
eight total GPU processes, 5 GiB GPU + 6 GiB host budget per admitted job,
4 GiB GPU + 8 GiB host reserve, and GPU utilization at most 85% on admission.
Predictors fit sequentially under resource checks. Six jobs run in this order:
both originals, both uniform variants, then both Shapley variants, with up to
two running at once when resource checks permit.

## W&B and provenance

Project: `2820402607-shandong-university/CORL-DDR`.
Six native runs plus a comparison run with one three-curve chart per dataset,
a common evaluation table and a best-score table. Pending jobs have no invented
scores. A separate bounded native-history relay handles the observed old-SDK
upload failures without altering the learner. Logging exit status and scientific
completion are recorded separately.

Entrypoints: `scripts/cql_antmaze_medium_5090/`.
Campaign: `results/cql-antmaze-medium-rewards-seed22-1m-5090-20261014`.
Prepare freezes code, official configs, dependencies and raw-data hashes.
All generated artifacts and credentials stay outside Git. Each output directory
refuses overwrite; a failed manager requires inspection before recovery.

Validation includes the six-arm pairing and official arguments, dependency gating
before all learning, missing checkpoint/incomplete history rejection, handling
of completed training with nonzero exit, and comparison tables selecting maxima.
Existing success-boundary tests cover reward conservation, grouped folds,
padding/masking, real small cross-fitting and shared-GPU admission.
