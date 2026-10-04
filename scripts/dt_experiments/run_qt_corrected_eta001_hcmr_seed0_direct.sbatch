#!/bin/bash
#SBATCH --job-name=qt-fixed-eta001-direct-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/qt-corrected-direct-%j.out

set -euo pipefail
PROJECT=/labmount/users/202615385/code/DT-EXP
QT_CODE="${QT_CORRECTED_SOURCE:-${PROJECT}}"

# Job 12045 inherited a client-local proxy (127.0.0.1:17897), which does not
# exist on a compute node. Clear proxy routing only in this retry's environment.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

# An unauthenticated GET to /graphql returns 405; it still proves HTTPS works.
# Fail on connection/TLS errors before using GPU time for the smoke and training.
curl --noproxy '*' --silent --show-error --connect-timeout 10 --max-time 30 \
  --output /dev/null --write-out 'W&B direct HTTPS status: %{http_code}\n' \
  https://api.wandb.ai/graphql

# Reuse the frozen, previously smoke-tested experiment without changing training.
# The original script verifies its source hashes, runs smoke, then starts 100k.
exec bash "${QT_CODE}/scripts/dt_experiments/run_qt_corrected_eta001_hcmr_seed0.sbatch"
