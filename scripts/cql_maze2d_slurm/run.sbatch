#!/usr/bin/env bash
#SBATCH --job-name=cql-maze2d-s23
#SBATCH --partition=GPULite
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --no-requeue

set -euo pipefail
ROOT=${1:?campaign root required}
source "$ROOT/source/scripts/cql_maze2d_slurm/env.sh" "$ROOT"
cd "$ROOT"
exec "$PYTHON" -m scripts.cql_maze2d_slurm.worker \
    --root "$ROOT" --index "$SLURM_ARRAY_TASK_ID"
