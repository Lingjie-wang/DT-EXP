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

## Validation receipt

Implementation: `7ddd4b7fd1615ca9ea0a87072b59c9042bb75c0f`, published directly
to `main`. Ruff 0.0.278 passed on a clean tracked-file export. All 32 relevant
unit tests passed in the existing isolated server runtime with CUDA hidden;
these include the Medium campaign, success boundaries, shared Shapley core,
official runner and retry tests. Python compilation and `bash -n` also passed.
GitHub Actions [run 38583424184](https://github.com/Lingjie-wang/DT-EXP/actions/runs/38583424184)
completed successfully.

Pinned upstream configuration SHA256:

- medium-play: `af70ab55185ea22c5a2b2b5cfc3e3b6cb523e7478e5a60fe8469c5f684d26211`
- medium-diverse: `42f0a6b888ce73fea7f24976ca0e8dd70bbcc277421a58f48a23d6d979079fc7`

Official D4RL downloads (HTTPS equivalent of the registered dataset URLs):

- medium-play, 231,110,764 bytes:
  `c9fec1c1daad97c71d61bd47f4ccc9f74faa855b8a2d2e8107d5378018827e5b`
- medium-diverse, 231,095,166 bytes:
  `ad9dcbac05d1f8744a7a216083f513662f43f64f28a4a5eb225e9811902dfae7`

## Queue receipt, 2026-10-14 23:01 Asia/Shanghai

The persistent tmux session `cql-antmaze-medium-20261014` is running on the
5090 server. The campaign records all six jobs as `queued`, with pipeline and
queue stage `waiting_for_predecessor`. No predictor output or training
directory exists yet. The predecessor Umaze Shapley run has completed 1M;
Umaze-Diverse Shapley is still running, so the dependency barrier remains closed.

Both raw files passed server-side structure checks: each contains 1,000,000
records and yields 999,000 official retained transitions. Medium-play has
9,787 raw success flags and 10,752 nonempty success/timeout fragments;
medium-diverse has 1,959 success flags and 2,955 fragments. Fragment returns
are all zero or one, and raw rewards equal the success terminal flags.

[W&B comparison](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/6edaf335-f243-4547-8cd1-aa2ae35ba36d)
was read back through the W&B API: all six jobs are queued, the best-score
table has six pending rows, and the evaluation table has zero rows. Curves
will appear after actual 100-episode evaluations. W&B flattens the per-job
summary fields as `job_states.<job_id>`. Local provenance and readback receipts
are retained inside the ignored campaign directory.
