# Official CORL CQL AntMaze baselines on the RTX 5090

The two original-reward controls in the earlier reward-redistribution campaign
changed success terminals and timeout handling. They are not the official CORL
data protocol. This independent campaign measures the official baseline on
`antmaze-umaze-v2` and `antmaze-umaze-diverse-v2`, without replacing any old run.

## Protocol

- One run per dataset, seed **21**, paired with the earlier reward comparisons.
- **1,000,000 gradient updates**, **100 evaluation episodes every 50,000 updates**.
  These values are unchanged from the official dataset-specific YAML files.
- CORL revision `6afec90484bbf47dee05fdf525e26a3ebe028e9b`. The CQL script and
  both YAML files are byte-identical to the GitHub upstream, checked with SHA-256.
- D4RL from the dependency repository specified by CORL requirements:
  `tinkoff-ai/d4rl`, revision `db6e4b34bb5ce2a51dd3879177c0a0223208a614`.
  A separate frozen source directory is first on `PYTHONPATH`; neither this source
  nor the shared installed package is edited or monkey-patched.
- Execute the official `algorithms/offline/cql.py` CLI directly. Its dataset
  loading, actor/critic construction, optimization, seeding, evaluation, and
  native W&B logging remain unchanged. No copied training loop is used.
- Default `d4rl.qlearning_dataset(env)`: retain original success terminals,
  drop timeout transitions, and use the official shifted next observations.
  Reward scaling remains `10*r - 5`, discount 0.99, batch size 256.
- Deliberate CLI overrides are seed 21 (official YAML default: 0), W&B labels,
  and checkpoint output path. GPU device is explicitly `cuda`, as in the YAML.
  Evaluation uses seed 21, following the official train/eval seed relationship.
- The available 5090-compatible runtime uses Python 3.10, PyTorch 2.7.1+cu128,
  NumPy 1.23.5 and Gym 0.23.0. These are modern hardware dependencies, not a claim
  to reproduce the exact historical software stack. Exact versions and the
  imported D4RL path are written into the data audit.

The read-only data audit calls the actual official loader on both full datasets
and compares every observation, action, reward, next observation and terminal to
the expected raw HDF5 slices. Expected counts are 998,573 transitions / 8,727
success terminals for Umaze and 999,000 / 36 for Diverse. Separate CPU preflights
exercise the unchanged CLI for 100 updates plus one evaluation. After the old
campaign finishes, GPU preflight repeats that check before each actual run.
Preflight is a separate process and cannot advance the training RNG.

## Durable queue and records

Campaign directory:
`results/cql-antmaze-official-seed21-1m-5090-20261010`.
Predecessor:
`results/cql-antmaze-timeout-four-rewards-seed21-300k-5090-20261009-retry1`.

The queue waits until every predecessor training job has completed. W&B upload
completion is independent of GPU training completion. Then it admits up to two
new jobs, counting all users' GPU processes. It requires at least 5,120 MiB GPU
and 6,144 MiB host memory per new job, plus reserves of 4,096 / 8,192 MiB, at most
eight total GPU processes, and at most 85% utilization at admission. Starting
processes not yet visible to `nvidia-smi` also reserve memory and a process slot.
Resources are rechecked after each GPU preflight. No existing job is stopped.

Seed reservations were checked on 2026-10-10: the 5090 has the known seed-21
timeout comparisons and no official AntMaze baseline; Slurm has no queued jobs
or AntMaze campaign plan files. Seed 21 is intentionally reused for pairing, in
a distinct experiment namespace. The common 1M budget comes from the official
configs; the earlier 300k curves motivate this baseline without proving that
extra updates alone explain the score gap.

Each run has its own console log, native W&B online run in `CORL-DDR`, local
`.wandb` record, and all 20 official checkpoints. The queue mirrors completed
evaluation lines into `evaluations.json`; displayed progress is the last
completed evaluation, not a per-update counter. Completion requires exit code
zero, all 20 evaluations of 100 episodes, and `checkpoint_999999.pt`.
The final/best scores can be read from `training/completion.json`; it does not
claim a separate remote W&B readback verification. Raw native W&B records are
available for sync recovery if the network fails. A failed run is preserved,
not automatically restarted from scratch or silently resumed.

## Reproduction

Create a clean source checkout of this commit. Obtain the pinned D4RL source,
export its `d4rl/` directory, and write `source_manifest.json` with its revision
and SHA-256 for every exported file. The preparation command checks these hashes
and copies the unchanged source, raw datasets and launch harness into the new
campaign directory. Do not reuse an existing output directory.

```bash
python -m scripts.cql_antmaze_official_5090.prepare \
  --root "$CAMPAIGN" --project "$PROJECT" --predecessor "$PREVIOUS" \
  --d4rl-source "$D4RL_SOURCE" --seed 21 --revision "$CODE_REVISION"
export PYTHONPATH="$CAMPAIGN/dependencies/d4rl:$CAMPAIGN/source"
export D4RL_DATASET_DIR="$CAMPAIGN/datasets"
# Set MuJoCo library paths as in launch.sh, using the 5090 runtime Python.
python -m scripts.cql_antmaze_official_5090.audit --root "$CAMPAIGN"
python -m scripts.cql_antmaze_official_5090.queue --root "$CAMPAIGN" --cpu-preflight
tmux new-session -d -s cql-antmaze-official-1m \
  "bash '$CAMPAIGN/source/scripts/cql_antmaze_official_5090/launch.sh' '$CAMPAIGN'"
```

The queue lock and refusal to overwrite an existing status/work directory
prevent accidental duplicate launches. Inspect processes and logs before any
manual recovery. Datasets, environments, checkpoints and runtime logs remain
outside Git; only the reproducible harness, tests and this protocol are committed.

## Launch validation on 2026-10-10

Both full-dataset comparisons passed with the counts above. Both unchanged
official CLI CPU preflights completed 100 updates, one evaluation and a saved
`checkpoint_99.pt`. The W&B API successfully read an existing finished run using
the launch credentials. Seven queue/parser/input-integrity tests passed, and
the repository's Ruff 0.0.278 CI command passed against a clean tracked export.
GPU preflights are deliberately deferred until the predecessor completes and
the resource admission checks pass; the 1M baseline scores are not available yet.
