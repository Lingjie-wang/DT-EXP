#!/usr/bin/env bash
set -euo pipefail
PROJECT=${CQL_5090_PROJECT:-/home/sckd02/workspace/DT-EXP}
ROOT=${1:?Pass the separately prepared retry directory}
PYTHON="$PROJECT/.runtime/cql-antmaze-official-wandb012-full-env/bin/python"
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
"$PYTHON" - "$ROOT" <<'PY'
import sys
from pathlib import Path
from scripts.cql_antmaze_official_5090.common import read
from scripts.cql_antmaze_official_retry_5090.online_check import validate_runtime
root = Path(sys.argv[1])
plan = read(root / "plan.json")
validate_runtime(root, plan)
for job in plan["jobs"]:
    verified = read(root / job["id"] / "online_cpu_preflight/verification.json")
    assert verified["wandb_verified"] and verified["completed_updates"] == 100
PY
exec "$PYTHON" -m scripts.cql_antmaze_official_5090.queue --root "$ROOT"
