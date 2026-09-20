#!/bin/bash
# Frozen historical learning protocol; user-approved 100k stop and 32G RAM request.
#SBATCH --job-name=v3-legacy345
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/v3-legacy345-%A_%a.out
set -euo pipefail

PROJECT=/labmount/users/202615385/code/DT-EXP
source /labmount/users/202615385/miniforge3/etc/profile.d/conda.sh
conda activate adt-delayed
export MUJOCO_PY_MUJOCO_PATH=/labmount/users/202615385/.mujoco/mujoco210
export C_INCLUDE_PATH="${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/labmount/users/202615385/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export CFLAGS="${CFLAGS:-} -Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types"
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export WANDB_MODE=online
export WANDB_IGNORE_GLOBS='*.pt,*.npz,*.npy'
export PYTHONUNBUFFERED=1

MODE=${1:?Specify smoke, prepare, or branch}
if [[ "$MODE" == smoke ]]; then
  export WANDB_MODE=offline
fi
cd "$PROJECT"
python scripts/dt_experiments/legacy_v3_high.py "$@"
