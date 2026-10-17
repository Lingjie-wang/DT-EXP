#!/usr/bin/env bash
# Prepare new data/run directories and execute the author's unchanged main.py.
set -euo pipefail
PROJECT="${AUCTIONNET_PROJECT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CODE="$PROJECT/scripts/auctionnet_dt"
PYTHON="$PROJECT/.runtime/auctionnet-dt-5090-env/bin/python"
DATA="$PROJECT/.runtime/auctionnet-dt-data-20261005"
UPSTREAM="$PROJECT/.runtime/auctionnet-upstream-20261005"
SMOKE="$PROJECT/results/prgs-dt-auctionnet-5090-smoke-20261005"
FULL="$PROJECT/results/prgs-dt-auctionnet-5090-20261005"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
if [[ -f "$PROJECT/.dt_runs/wandb.env" ]]; then
    set -a
    source "$PROJECT/.dt_runs/wandb.env"
    set +a
fi
wait_for_file() {
    while [[ ! -f "$1" ]]; do sleep 15; done
}
preprocess_period() {
    local period="$1"
    wait_for_file "$DATA/raw/period-$period.csv"
    if [[ ! -f "$DATA/processed/period-$period.json" ]]; then
        "$PYTHON" "$CODE/prepare_data.py" period \
            --raw "$DATA/raw/period-$period.csv" \
            --generator "$UPSTREAM/auctionnet/train_data_generator.py" \
            --output "$DATA/processed"
    fi
}
for period in 7 8; do preprocess_period "$period"; done
"$PYTHON" "$CODE/prepare_data.py" assemble --inputs "$DATA/processed" \
    --output "$DATA/prepared-smoke" --periods 7 8
wait_for_file "$DATA/raw/period-14.csv"
"$PYTHON" "$CODE/prepare_data.py" smoke-csv --raw "$DATA/raw/period-14.csv" \
    --output "$DATA/smoke-period-14.csv"
"$PYTHON" "$CODE/prepare_run.py" --upstream "$UPSTREAM/prgs/AuctionNet" \
    --data "$DATA/prepared-smoke" --csvs "$DATA/smoke-period-14.csv" \
    --root "$SMOKE" --smoke
"$PYTHON" "$CODE/run.py" --root "$SMOKE" --wandb-mode online
for period in 9 10 11 12 13; do preprocess_period "$period"; done
"$PYTHON" "$CODE/prepare_data.py" assemble --inputs "$DATA/processed" \
    --output "$DATA/prepared-full"
csvs=()
for period in 14 15 16 17 18 19 20; do
    wait_for_file "$DATA/raw/period-$period.csv"
    csvs+=("$DATA/raw/period-$period.csv")
done
"$PYTHON" "$CODE/prepare_run.py" --upstream "$UPSTREAM/prgs/AuctionNet" \
    --data "$DATA/prepared-full" --csvs "${csvs[@]}" --root "$FULL"
"$PYTHON" "$CODE/run.py" --root "$FULL" --wandb-mode online
