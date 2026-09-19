#!/bin/bash
#SBATCH --job-name=qt-official-hcmr-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/qt-official-%j.out

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
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_DISABLE_CODE=true
export WANDB_MODE=online
cd "${PROJECT}"
EXTRA=()
if [[ "${1:-}" == smoke ]]; then
  export WANDB_MODE=offline
  EXTRA=(--smoke)
elif [[ -n "${1:-}" ]]; then
  exit 2
fi
"${PROJECT}/third_party/qt-env/bin/python" algorithms/offline/qt_official_runner.py \
  --source "${PROJECT}/third_party/QT" \
  --output "${PROJECT}/checkpoints/qt-official/job-${SLURM_JOB_ID}" \
  --seed 0 --updates 100000 --eval-every 5000 --eval-episodes 100 \
  "${EXTRA[@]}"
