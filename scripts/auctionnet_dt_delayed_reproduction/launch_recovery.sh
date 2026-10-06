#!/usr/bin/env bash
set -euo pipefail
PROJECT="${AUCTIONNET_PROJECT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
if [[ -f "$PROJECT/.dt_runs/wandb.env" ]]; then
    set -a
    source "$PROJECT/.dt_runs/wandb.env"
    set +a
fi
cd "$PROJECT"
exec "$PROJECT/.runtime/auctionnet-dt-5090-env/bin/python" \
    -m scripts.auctionnet_dt_delayed_reproduction.recover_repeat3 --project "$PROJECT"
