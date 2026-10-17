# HalfCheetah-medium Shapley on RTX 5090

Add one seed 1 CQL run on HalfCheetah-medium-v2 with predictive Shapley rewards.
Use independent `scripts/cql_shapley_medium_5090` entrypoints and
`results/cql-shapley-hcm-v1-seed1-5090-20261007`. Preserve the three active 5090
runs, Slurm jobs, all historical code and all historical results. No uniform or
dense control is added. This extends the project pilot, not a new paper reproduction.

The original medium HDF5 SHA256 must be
`41324e8487a9556d8cfa4538827f8937f4d63dae3030825ebc9f3a726ed08ca0`.
It contains 1000 complete 1000-step trajectories. All non-reward transitions
match the running medium delayed baseline exactly, including timeout handling.
The predictor receives only state/action sequences and float64 trajectory totals;
no original per-step rewards are stored in predictor inputs.

Fit a new medium predictor using the existing generic fitting source unchanged:
5 contiguous outer folds (640 inner train / 160 validation / 200 held per fold),
20 segments of 50 steps, 128 permutations, hidden width 64, Adam lr .001,
weight decay .0001, batch 32, maximum 300 epochs, patience 30, 8 validation masks.
Keep split/training/validation/attribution seeds from the medium-replay design.
CPU fitting uses two threads in the existing 5090 environment. Restore the best
validation checkpoint and attribute only each fold's held trajectories. Preserve
signed contributions and the existing uniform residual and float32 corrections.
The original protocol and all diagnostics are described in `cql_shapley_pilot.md`.

Prediction MSE must beat the fold-specific training-mean baseline before CQL
can start. A failed gate retains diagnostics and stops this new run; it does not
trigger tuning on CQL evaluations. Preflight requires 100 full-batch CUDA updates
and one complete evaluation episode. Formal CQL starts from scratch, with seed 1,
1M updates, batch 256, gamma .99 and 10 evaluation episodes every 5000 steps.
Only run name/group differ from the medium delayed CQL configuration. The CQL
algorithm, trainer, fitting code and evaluation entrypoint are copied byte for
byte from the frozen existing campaigns; no monkey patches are used.

Prepare on the data host, then transfer the entire new campaign:

```bash
python -m scripts.cql_shapley_medium_5090.prepare \
  --root "$PROJECT/results/cql-shapley-hcm-v1-seed1-5090-20261007" \
  --baseline "$PROJECT/results/cql-corl-delayed-hc-seed1-5090-20261007" \
  --reference "$PROJECT/results/cql-shapley-hcmr-v1-seed1-5090-20261007" \
  --hdf5 "$PROJECT/.runtime/datasets/cql-corl-delayed-20261006/halfcheetah_medium-v2.hdf5"
bash scripts/cql_shapley_medium_5090/launch.sh
```

Deploy a validated published archive into a new server runtime directory and
launch under a separate tmux session. The launcher reuses the existing read-only
5090 environment and ignored W&B credentials, and invokes an independent W&B
observer. Training does not wait for W&B. Logs are in the campaign's `launch/`.
Existing campaigns, fit directories and launch attempts are refused.

W&B: `2820402607-shandong-university/CORL-DDR`, run name
`CQL-CORL-HCM-shapley20x128-v1-seed1-1M-5090`.

Publish scoped commits on top of current GitHub main using fast-forward updates
with a checked expected head. Run Ruff CI and CQL/Shapley tests on a clean export;
inspect GitHub Actions after publishing. Keep datasets, fit artifacts, checkpoints,
runtime state and secrets out of Git, and preserve both servers' working trees.
