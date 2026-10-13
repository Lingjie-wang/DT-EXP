# CQL AntMaze: official success boundaries and redistributed rewards

Requested comparison: `antmaze-umaze-v2` and `antmaze-umaze-diverse-v2`, each with
uniform and predictive segment Shapley redistribution. Four new training runs;
reuse the completed original-reward seed 21 official controls. Compare the maximum
observed evaluation score, with the full evaluation curves retained.

The existing controls completed 1M updates. Their best scores were Umaze 94 at
550k and Diverse 55 at 700k. This campaign keeps the official 1M-update budget,
50k evaluation interval, 100 episodes, and seed 21 for a paired comparison.
The 2026-10-13 reservation check found no Slurm jobs or AntMaze plan files;
5090 old campaigns were completed. Other users' GPU processes remain untouched.

## Official fidelity

CORL revision `6afec90484bbf47dee05fdf525e26a3ebe028e9b`; D4RL revision
`db6e4b34bb5ce2a51dd3879177c0a0223208a614`. The official CQL script, YAML files,
loader, optimizer, normalization (`10*r-5`), training and evaluation loops stay
byte-identical. Use the existing isolated Python 3.10 / Torch 2.7.1 cu128 /
W&B 0.12.21 environment. Only the raw HDF5 `rewards` array is replaced in separate
copies. Seed and output/W&B locations identify these independent runs.

Split raw data at success `terminals` OR `timeouts`, then retain exactly the
indices selected by the official default `qlearning_dataset`: indices 0..N-2,
excluding timeout rows. Keep the unfinished tail when the official loader does.
Skip any empty fragment after this filtering. Preserve every success terminal,
next observation, action, and non-reward HDF5 field. The raw files have binary
rewards equal to success terminals; retained fragments therefore return 0 or 1.
These are learning fragments, not necessarily independent collected episodes.

**Success timing is available through segmentation and the learner's terminals.**
This is not a trajectory-return-only information-isolation experiment. One-step
fragments cannot redistribute credit and keep their original scalar reward.
Zero-return fragments can have signed Shapley rewards summing to zero. Only
undiscounted return conservation is claimed; discounting can alter credit.

## Redistribution

Uniform: fragment return divided by its actual retained length, with only a
float32 rounding correction on the last step.

Shapley: reuse the previous state/action return predictor (64-unit segment MLP,
masked pooled embedding, 64-unit head). Up to 20 near-equal contiguous valid
segments; 128 seeded permutations. Padded steps have validity indicators; invalid
segment slots are always masked. Pooling/count retain the previous denominator
20. A one-player game is evaluated exactly once because its 128 permutations
are identical. Fit statistics on training frames/returns only.

Use five contiguous outer folds of parent timeout collection fragments, with
20% of remaining parent groups as inner validation. All success fragments of
one parent stay in the same split. Training/split/validation seeds and attribution
seed base follow the previous implementation. Adam lr 1e-3, weight decay 1e-4,
batch 32, at most 300 epochs, early stopping patience 30, eight fixed validation
masks. The loss averages full-return and random nonempty-coalition MSE. For one
player, the only nonempty coalition is the full set.

Spread each segment contribution over its steps; spread the difference between
true fragment return and summed contributions uniformly over the entire fragment.
Correct only rounding at the last step. Save models, splits, permutations,
contributions, held-out predictions, conservation audits and prediction-vs-mean
MSE diagnostics. The predictive quality gate is **report-only**: a weak predictor
is disclosed and never silently replaced with uniform rewards.

## Execution and W&B

New entry points: `scripts/cql_antmaze_success_5090/`. A prepare command freezes
all source/config/dependency/data inputs, baseline curves and a run plan. The
durable pipeline fits both predictors, materializes reward files, freezes hashes,
checks the real loader, runs 100-step CPU preflights with real W&B, then admits
GPU preflight and training jobs. Every output directory refuses overwrite.
A failed manager requires inspection before recovery; it does not launch duplicates.

At most two own CQL jobs, at most eight total GPU processes, 5 GiB GPU + 6 GiB
host memory budget per new job, 4 GiB GPU + 8 GiB host reserve, admission GPU
utilization at most 85%. Count already running foreign processes, recheck after
GPU preflight, poll every 30 seconds. Predictor fits precede CQL, one at a time,
under the same resource admission constraints.

All native runs use `2820402607-shandong-university/CORL-DDR`. One independent
comparison run has two multi-line charts (one per dataset), overlaying the existing
official original curve and both new curves. It also contains a common evaluation
table and best-score table. Curves contain real completed 100-episode evaluations;
pending arms have no synthetic zero points. The observer updates on new evaluations
and retries network failures independently of training. Preflight runs are in a
separate validation group and excluded from comparison.

```bash
PROJECT=/home/sckd02/workspace/DT-EXP
PYTHON="$PROJECT/.runtime/cql-antmaze-official-wandb012-full-env/bin/python"
ROOT="$PROJECT/results/cql-antmaze-success-rewards-seed21-1m-5090-20261013"
BASELINE="$PROJECT/results/cql-antmaze-official-seed21-1m-5090-20261010-retry2"
PYTHONPATH="$PROJECT" "$PYTHON" -m scripts.cql_antmaze_success_5090.prepare \
  --root "$ROOT" --baseline "$BASELINE" --project "$PROJECT" \
  --revision "$(git -C "$PROJECT" rev-parse HEAD)"
tmux new-session -d -s cql-antmaze-success-20261013 \
  "bash '$ROOT/source/scripts/cql_antmaze_success_5090/launch.sh' '$ROOT'"
```

Inspect `pipeline_status.json`, `predictor/*/fit/status.json`, `queue/status.json`,
per-job `training/{console.log,evaluations.json,completion.json,wandb_manifest.json}`,
and `comparison/{manifest.json,latest.json,status.json}`. Frozen data, checkpoints,
runtime environments and generated artifacts remain outside Git.

## Frozen campaign and initial validation

Campaign: `results/cql-antmaze-success-rewards-seed21-1m-5090-20261013`.
Frozen implementation commit: `3440f6183404e930702d6f91a10d3bc8d6c58a2c`.
All 26 selected tests passed on a clean export using the server runtime; the
[main-branch CI](https://github.com/Lingjie-wang/DT-EXP/actions/runs/38366434782) passed.
All four real-loader audits and all four 100-step online CPU preflights passed.

[W&B comparison charts and tables](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/5f95025b-5d4c-403f-8469-0407555458d9)
were read back from the service: both official 20-point curves, the common
40-row initial evaluation table, and the six-arm best-score table matched the
frozen local records exactly. New arms are appended after completed evaluations.

| Data audit | Umaze | Umaze Diverse |
| --- | ---: | ---: |
| Official retained transitions | 998,573 | 999,000 |
| Nonempty success/timeout fragments | 10,131 | 1,035 |
| Return-one fragments | 8,727 | 36 |
| Zero-return fragments | 1,404 | 999 |
| One-step fragments | 6,616 | 24 |
| Empty fragments skipped after official filtering | 23 | 1 |
| Uniform changed reward positions | 213,681 | 5,441 |

These counts include official retained tail transitions; they are not counts
of independent environment rollouts. Some timeout rows alone form empty raw
fragments and are dropped by the official loader.

| Held-out predictor diagnostic | Umaze | Umaze Diverse |
| --- | ---: | ---: |
| Predictive return MSE | 0.0263074 | 0.0182247 |
| Inner-training-mean MSE | 0.1195215 | 0.0339193 |
| Maximum Shapley fragment return error | 2.98e-8 | 2.33e-10 |
| Minimum redistributed reward | -0.2847183 | -0.0224306 |
| Maximum redistributed reward | 1.0 | 1.0 |

Both diagnostic gates passed. These are predictor diagnostics, not CQL policy
evaluation results. Negative step rewards are allowed and sums are conserved.
The common W&B best-score table is the source for subsequent policy comparisons.
