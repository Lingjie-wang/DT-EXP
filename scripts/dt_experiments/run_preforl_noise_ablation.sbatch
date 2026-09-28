#!/bin/bash
#SBATCH --job-name=preforl-noise-paired-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/preforl-noise-ablation-%A_%a.out

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
export WANDB_MODE=offline
cd "${PROJECT}"
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) NOISE=0 ;;
  1) NOISE=0.01 ;;
  *) exit 2 ;;
esac
EXTRA_ARGS=()
if [[ "${1:-}" == smoke ]]; then
  # Full production architecture/batch size; two updates and two evaluations.
  EXTRA_ARGS=(--gradient-steps 2 --eval-interval 1 --eval-episodes 2
    --wandb-mode disabled)
  PYTHONPATH=algorithms/offline python -m unittest discover -s tests \
    -p test_preforl_noise_ablation.py -v
elif [[ "${1:-}" == online ]]; then
  # Requires explicit approval for the metrics/configuration upload.
  export WANDB_MODE=online
  EXTRA_ARGS=(--wandb-mode online)
elif [[ -n "${1:-}" ]]; then
  exit 2
fi
python algorithms/offline/preforl_noise_ablation.py \
  --shadow-noise "${NOISE}" \
  --output "${PROJECT}/results/preforl_noise_ablation/job-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}/noise-${NOISE}" \
  "${EXTRA_ARGS[@]}"
