#!/usr/bin/env bash
set -euo pipefail
PROJECT=${CQL_5090_PROJECT:-/home/sckd02/workspace/DT-EXP}
CODE=$(cd "$(dirname "$0")/../.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export WANDB_CONSOLE=off WANDB_DISABLE_CODE=true CUDA_VISIBLE_DEVICES=""
if [[ -f "$PROJECT/.dt_runs/wandb.env" ]]; then
    set -a
    source "$PROJECT/.dt_runs/wandb.env"
    set +a
fi
cd "$CODE"
export PYTHONPATH="$CODE"
exec "$PROJECT/.runtime/state-only-c-5090-env/bin/python" \
    -m scripts.cql_stop_500k.run launch --plan "$1"
