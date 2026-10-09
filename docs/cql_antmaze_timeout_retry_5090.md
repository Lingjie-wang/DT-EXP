# AntMaze timeout controls: independent launch retry

This campaign continues the eight experiments in
[the initial protocol](cql_antmaze_timeout_5090.md): two AntMaze datasets,
four reward settings, paired seed 21, and 300,000 CQL updates each. The original
failed launch, fitted models, and diagnostics are preserved. No CQL policy
started in that launch.

## Startup failure and compatibility change

The initial absolute-path invocation of `scripts/cql_antmaze_5090/train.py`
placed its directory first on Python's import path. During dependency import,
`more_itertools` imported `queue.Empty` and `queue.Queue`; the adjacent scheduler
`queue.py` shadowed the standard library and failed with
`ModuleNotFoundError: No module named 'scripts.cql_5090'`.

The independent retry invokes
`python -m scripts.cql_antmaze_retry_5090.train` from the frozen source root.
This changes import resolution, not CQL updates, data, evaluation, or budgets.
Both preflight and policy training use module invocation.

## Explicit decision on failed prediction checks

The user authorized retaining failed diagnostics and running the already fitted
Shapley rewards without tuning, refitting, or replacing rewards. Use the explicit
`--prediction-gate report-only` option. A failed check remains `passed: false`
in each Shapley protocol and audit; it is not reclassified as a passing check.

| Dataset | Held-out model MSE | Training-mean baseline MSE | Relative improvement |
| --- | ---: | ---: | ---: |
| antmaze-umaze-v2 | 674.5281810816856 | 616.219620093383 | -9.4623% |
| antmaze-umaze-diverse-v2 | 0.16765323017702172 | 0.1652730436403758 | -1.4402% |

These predictors did not outperform their mean baselines. Policy experiments
can still measure the effect of their frozen redistribution, but these failed
checks must accompany interpretation of results.

Preparation verifies original input hashes, saved total returns, and exact
equality between saved rewards and rewards reconstructed from saved Shapley
contributions. Other arms reuse their original arrays. All four settings retain
identical non-reward arrays and timeout-only trajectory boundaries. Original
success terminals are not exposed to the predictor. The official `10*r - 5`
reward transform follows redistribution in every arm.

## Reproducible deployment

From the published source root on the 5090, using its existing environment:

```bash
PROJECT=/home/sckd02/workspace/DT-EXP
PYTHON="$PROJECT/.runtime/state-only-c-5090-env/bin/python"
PREVIOUS="$PROJECT/results/cql-antmaze-timeout-four-rewards-seed21-300k-5090-20261009"
"$PYTHON" -m scripts.cql_antmaze_retry_5090.prepare \
  --previous "$PREVIOUS" --root "${PREVIOUS}-retry1" \
  --code-revision "$(git rev-parse HEAD)" --prediction-gate report-only
bash scripts/cql_antmaze_retry_5090/launch.sh
```

For an immutable archive deployment without `.git`, replace `git rev-parse HEAD`
with its published commit SHA. Run the launcher inside persistent tmux. The
manager admits at most two campaign policies and at most eight total GPU
processes, subject to the initial protocol's utilization, GPU memory and host
memory limits. Existing foreign jobs are counted and left running.

The retry has independent output directories, W&B run IDs, and names ending in
`-retry1`. Preparation refuses an active initial queue, an existing retry root,
or any initial policy training directory. An enforced-gate mode exists for
testing but is not the user's selected protocol.

## Validation

Regression tests exercise frozen reward reuse, preservation of failed gate
diagnostics, all eight queued jobs, prevention of replaying started policies,
and module invocation for every preflight and training process. The launcher
requires each job's real CUDA preflight to pass before its full policy run.
The queue independently verifies completed updates and W&B synchronization.
