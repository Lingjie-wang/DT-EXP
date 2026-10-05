#!/usr/bin/env bash
set -euo pipefail
PROJECT="${AUCTIONNET_PROJECT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
if [[ -f "$PROJECT/.dt_runs/wandb.env" ]]; then
    set -a
    source "$PROJECT/.dt_runs/wandb.env"
    set +a
fi
exec "$PROJECT/.runtime/auctionnet-dt-5090-env/bin/python" \
    "$PROJECT/scripts/auctionnet_dt_nonfinal/pipeline.py" --project "$PROJECT"
