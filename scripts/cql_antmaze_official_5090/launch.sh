#!/usr/bin/env bash
set -euo pipefail
PROJECT=${CQL_5090_PROJECT:-/home/sckd02/workspace/DT-EXP}
ROOT=${1:?Pass the prepared campaign directory}
export MUJOCO_PY_MUJOCO_PATH="$PROJECT/.runtime/state-only-c-5090-install/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
if [[ -f "$PROJECT/.dt_runs/wandb.env" ]]; then
    set -a
    source "$PROJECT/.dt_runs/wandb.env"
    set +a
fi
export PYTHONPATH="$ROOT/dependencies/d4rl:$ROOT/source"
export D4RL_DATASET_DIR="$ROOT/datasets"
cd "$ROOT"
exec "$PROJECT/.runtime/state-only-c-5090-env/bin/python" \
    -m scripts.cql_antmaze_official_5090.queue --root "$ROOT"
