#!/bin/bash
#SBATCH --job-name=action-noise-grad-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/action-noise-gradient-diagnostic-%j.out

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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
cd "${PROJECT}"
git rev-parse HEAD
PYTHONPATH=algorithms/offline python -m unittest discover -s tests \
  -p test_action_noise_gradient_diagnostic.py -v
EXTRA_ARGS=()
if [[ "${1:-}" == smoke ]]; then
  EXTRA_ARGS=(--batches 1 --repeats 2 --main-batch-size 32 --aux-batch-size 16)
elif [[ "${1:-}" == online ]]; then
  # Use only after explicit approval of the diagnostic payload/destination.
  export WANDB_MODE=online
  EXTRA_ARGS=(--wandb)
elif [[ -n "${1:-}" ]]; then
  exit 2
fi
python algorithms/offline/diagnose_action_noise_gradients.py \
  --checkpoint "${PROJECT}/checkpoints/top5-return-fromscratch/DT-Control-FromScratch-HCMR-delayed-seed0/DT-Control-FromScratch-HCMR-delayed-seed0-halfcheetah-medium-replay-v2-61bc26ef/step050000.pt" \
  --output "${PROJECT}/results/action_noise_gradient_diagnostic/job-${SLURM_JOB_ID}" \
  "${EXTRA_ARGS[@]}"
