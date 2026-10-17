# Active RTX 5090 CQL runs: user-requested 500k stop

On 2026-10-07 the user requested that the currently running experiments stop at
500,000 CQL updates. This operational override applies only to these four seed-1
runs in `results/`:

| Campaign | Arm | Existing W&B run ID |
|---|---|---|
| `cql-corl-delayed-hc-seed1-5090-20261007` | `medium` | `3595114cfc74` |
| `cql-corl-delayed-hc-seed1-5090-20261007` | `medium_replay` | `eb9a253fe8b5` |
| `cql-shapley-hcmr-v1-seed1-5090-20261007` | `shapley` | `ce86d7e3d95b` |
| `cql-shapley-hcm-v1-seed1-5090-20261007` | `shapley` | `e069b6be3c8c` |

The 14 pending runs retain their separately prepared 100k budgets. Slurm jobs
are outside this change. Existing source, protocols, rewards, configurations,
logs, evaluations and checkpoints are preserved. The original protocols still
record 1M; the guard plan and per-run receipts record the explicit budget override.

## Stopping and scientific interpretation

The already running Python loops have evaluated their 1M bounds. The independent
`scripts.cql_stop_500k.run` guard waits for the existing 500k evaluation and
checkpoint, with no training source edits or process injection. A progress status
at 500k alone is insufficient: it precedes both evaluation and saving. The guard
requires ten finite episode returns and a CPU-loadable checkpoint whose wrapper
and CQL trainer both record 500,000 updates, including model/optimizer/RNG state.

Each watcher binds signals to the captured command, boot ID and process start
time, and uses Linux pidfds against PID reuse. Only that training PID is stopped.
Independent watchers prevent one slow W&B connection delaying the other stops.
Detection and deserialization can allow a short unsaved tail of updates. Receipts
record the last 100-update progress report; they do not claim to know the exact
number of unsaved updates. The result is the **exact 500k checkpoint and evaluation**;
final/best/last10 summaries consider evaluations only through 500k. Raw history
is retained, including any tail. This is a user-truncated experimental run, not a
claim that the original 1M training program returned normally.

The old W&B observer is stopped before acquiring its lock and resuming the same
run ID. Names change from `-1M-` to `-500k-` at finalization. W&B records original
and effective budgets, stop reason, final metrics and the 500k checkpoint hash.
Telemetry is replayed only through 500k to recover asynchronous uploads; identical
old rows can occur twice and are marked `stop500k_replay`. The checkpoint and
receipt are uploaded; finished state, name, final score, step and uploaded file
are verified through the API. Network failures retry independently after 30s.

Training status becomes `stopped_by_user`. Original launcher logs/status may show
SIGTERM failure because they require a normal 1M exit. Preserve that evidence;
the guard's `stop_receipt.json` explains the intentional stop and is authoritative
for this override. Existing resource queues count live GPU processes and proceed
when slots are available; they do not depend on these old launchers' exit codes.

## Deployment and recovery

Use an immutable Git export on the 5090, not either dirty shared checkout. From
the export, using its configured runtime Python:

```bash
python -m scripts.cql_stop_500k.run prepare \
  --project /home/sckd02/workspace/DT-EXP \
  --directory /home/sckd02/workspace/DT-EXP/.runtime/cql-stop-active-500k-20261007
bash scripts/cql_stop_500k/launch.sh \
  /home/sckd02/workspace/DT-EXP/.runtime/cql-stop-active-500k-20261007/plan.json
```

Run the launcher in persistent tmux session `cql-stop-active-500k-20261007`.
Preparation refuses existing output and validates all four live identities and
unchanged frozen sources before recording anything. `plan.json`, `supervisor.json`
and `run0` through `run3/status.json` record the plan and guard health. Each run has
its own log, lock, stop intent, receipt and W&B verification. No credentials or
generated run files belong in Git.

Real checkpoint loading requires `algorithms.offline.cql.Scalar` and its imported
MuJoCo dependencies. The launcher supplies the existing training runtime's MuJoCo
and library paths; no packages or scientific source are changed. Each watcher
first loads an earlier completed checkpoint on CPU, so missing dependencies fail
before it arms rather than at 500k. This also warms imports before the stop boundary.

After interruption, inspect per-run status and restart the launcher with the
**existing** plan. It never discovers fresh PIDs on recovery. Locks prevent two
writers, receipts permit continuation after termination, and missing training
before a valid checkpoint produces `needs_attention` rather than fake success.
Check `wandb_verified.json` for completed finalization. Historical experiment
implementations and results must remain preserved in future work.
