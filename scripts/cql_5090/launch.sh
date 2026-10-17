#!/usr/bin/env bash
set -euo pipefail
PROJECT=${CQL_5090_PROJECT:-/home/sckd02/workspace/DT-EXP}
CODE=$(cd "$(dirname "$0")/../.." && pwd)
RUNTIME="$PROJECT/.runtime/state-only-c-5090-env"
export MUJOCO_PY_MUJOCO_PATH="$PROJECT/.runtime/state-only-c-5090-install/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export D4RL_DATASET_DIR="$PROJECT/.runtime/state-only-c-5090-install/datasets"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
if [[ -f "$PROJECT/.dt_runs/wandb.env" ]]; then
    set -a
    source "$PROJECT/.dt_runs/wandb.env"
    set +a
fi
cd "$CODE"
export PYTHONPATH="$CODE"
exec "$RUNTIME/bin/python" -m scripts.cql_5090.run --project "$PROJECT"
