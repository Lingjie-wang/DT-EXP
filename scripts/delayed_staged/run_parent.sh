#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT"
# A frozen Git worktree can reuse this experiment's separately installed runtime.
BASE=${STATE_ONLY_RUNTIME_BASE:-"$PROJECT"}
RUNTIME="$BASE/.runtime/state-only-c-5090-env"
export PATH="$RUNTIME/bin:$PATH"
export MUJOCO_PY_MUJOCO_PATH="$BASE/.runtime/state-only-c-5090-install/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export D4RL_DATASET_DIR="$BASE/.runtime/state-only-c-5090-install/datasets"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
if [[ -f "$BASE/.dt_runs/wandb.env" ]]; then
    set -a
    source "$BASE/.dt_runs/wandb.env"
    set +a
fi
exec "$RUNTIME/bin/python" -u algorithms/offline/delayed_staged_parent_dt.py \
    --config_path configs/offline/dt/halfcheetah/delayed_staged_parent_5090.yaml "$@"
