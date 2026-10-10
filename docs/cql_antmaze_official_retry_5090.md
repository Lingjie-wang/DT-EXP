# Official AntMaze baseline: independent W&B environment retry

The initial `cql-antmaze-official-seed21-1m-5090-20261010` campaign failed
before its first training update on both datasets. The shared environment's
W&B 0.30.0 rejects the unchanged upstream `wandb.run.save()` call with
`TypeError: Run.save() missing 1 required positional argument: 'glob_str'`.
Its disabled-W&B preflights passed, but did not exercise this online code path.
Neither formal run produced an evaluation or checkpoint. All failure records
are retained, including the two native W&B runs.

This retry restores CORL's specified `wandb==0.12.21` in an independent venv.
The numerical stack is copied into that venv using real files (copy-on-write
where supported); the shared environment is never modified. An initial `.pth`
overlay failed the online preflight because D4RL's dependency imports moved
the shared site-packages to the front of `sys.path`. That validation-only
attempt is retained under the `-retry1` output directory and never launched
formal GPU training. The launcher now probes D4RL-before-W&B import order in
a fresh process. Protobuf 3.20.3 and the logger's
missing dependencies are pinned in the retry requirements. PyTorch remains
2.7.1+cu128, NumPy 1.23.5, Gym 0.23.0, and Python 3.10. This remains a modern
5090 hardware stack, not the exact historical CORL environment.

The retry imports the original queue implementation and executes the same
byte-identical official CQL CLI and YAML files. The dataset loader, algorithm,
training/evaluation loops, seed 21, 1M updates, evaluation every 50k updates
with 100 episodes, resource limits and W&B destination are unchanged. See
[the original protocol](cql_antmaze_official_5090.md) for source hashes, data
counts, scheduling and seed rationale. There are no monkey patches.

## Recovery and launch gates

New output: `results/cql-antmaze-official-seed21-1m-5090-20261010-retry2`.
New runtime: `.runtime/cql-antmaze-official-wandb012-full-env`.

Preparation requires the previous manager's lock, a terminal failure state,
the known W&B error, no recorded training progress/checkpoints/evaluations,
and unchanged frozen inputs. It creates a new campaign and records the old
plan hash in `retry_of`. It refuses to overwrite an existing directory.

Before GPU admission, each dataset gets an independent CPU process executing
the unchanged official CLI for 100 updates and one evaluation with **online**
W&B. The check requires a native checkpoint, completed local evaluation,
finished remote run, remote training history through step 100 and matching
remote evaluation score. Validation uses its own W&B group and is not a 1M
result. The launcher requires these verification records and the correct
interpreter/W&B version. The original queue then performs its GPU preflights
and resource checks before each formal online training run.

## Reproduction

Download the packages in `scripts/cql_antmaze_official_retry_5090/requirements.txt`
to a wheelhouse matching CPython 3.10 on Linux x86-64. `promise` and `pathtools`
can be provided as source distributions and built by the target Python 3.10.
The shared stack must provide setuptools and wheel for those builds. Set the
MuJoCo library paths as in the launcher before running the setup's import check.

```bash
bash scripts/cql_antmaze_official_retry_5090/setup_runtime.sh \
  "$SHARED_PYTHON" "$NEW_RUNTIME" "$WHEELHOUSE"
"$NEW_RUNTIME/bin/python" -m scripts.cql_antmaze_official_retry_5090.prepare \
  --previous "$FAILED_CAMPAIGN" --root "$CAMPAIGN" --project "$PROJECT" \
  --d4rl-source "$D4RL_SOURCE" --runtime "$NEW_RUNTIME/bin/python" \
  --revision "$CODE_REVISION"
# Export MuJoCo paths/thread limits and source W&B credentials as in launch.sh.
export PYTHONPATH="$CAMPAIGN/dependencies/d4rl:$CAMPAIGN/source"
export D4RL_DATASET_DIR="$CAMPAIGN/datasets"
"$NEW_RUNTIME/bin/python" -m scripts.cql_antmaze_official_5090.audit \
  --root "$CAMPAIGN"
"$NEW_RUNTIME/bin/python" -m scripts.cql_antmaze_official_retry_5090.online_check \
  --root "$CAMPAIGN"
tmux new-session -d -s cql-antmaze-official-1m-retry2 \
  "bash '$CAMPAIGN/source/scripts/cql_antmaze_official_retry_5090/launch.sh' '$CAMPAIGN'"
```

The fixed host launcher uses the documented runtime path. Runtime versions are
recorded in `retry_runtime.json`; each online preflight writes its native W&B
URL and validation result. Formal progress shown in `queue/status.json` is
the last **completed evaluation**, so it stays zero before the first 50k
evaluation even while W&B records optimizer updates. Inspect both sources
when reporting early progress. No official 1M scores are inferred from preflights.

## Validation on 2026-10-10

Both full-dataset audits and both real online CPU preflights passed in the
fully isolated environment. Each official CLI completed 100 optimizer updates,
one evaluation and `checkpoint_99.pt`; W&B returned a finished run, history
through step 100 and the matching evaluation score. The validation runs are
[Umaze](https://wandb.ai/2820402607-shandong-university/CORL-DDR/runs/9e6c7183-aa7d-49dc-93be-73b00021454d)
and the Diverse run recorded in its `online_cpu_preflight/verification.json`.
Eleven queue/recovery tests passed. Ruff 0.0.278 passed on a clean tracked
export, and both new shell entry points passed `bash -n`.
