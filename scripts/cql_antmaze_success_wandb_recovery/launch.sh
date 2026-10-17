#!/usr/bin/env bash
set -euo pipefail
PROJECT=${CQL_5090_PROJECT:-/home/sckd02/workspace/DT-EXP}
ROOT=${1:?Pass the existing campaign directory}
OUT=${2:?Pass an independent recovery output directory}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT"
set -a
source "$PROJECT/.dt_runs/wandb.env"
set +a
mkdir -p "$OUT"
cd "$PROJECT"
exec "$PROJECT/.runtime/cql-antmaze-official-wandb012-full-env/bin/python" \
    -m scripts.cql_antmaze_success_wandb_recovery.relay --root "$ROOT" --out "$OUT" \
    >> "$OUT/console.log" 2>&1
