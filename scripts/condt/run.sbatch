#!/bin/bash
#SBATCH --job-name=condt-hopper-medium
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/condt-%j.out

set -euo pipefail
PROJECT=/labmount/users/202615385/code/DT-EXP
CAMPAIGN=${1:?campaign required}
ARM=${2:?dt/condt required}
source /labmount/users/202615385/miniforge3/etc/profile.d/conda.sh
conda activate adt-delayed
unset PYTHONPATH
export MUJOCO_PY_MUJOCO_PATH=/labmount/users/202615385/.mujoco/mujoco210
export C_INCLUDE_PATH="${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export CFLAGS="${CFLAGS:-} -Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Author Ray evaluation tasks request CPUs but explicitly run tensors on CUDA.
# Keep Slurm's allocated GPU visible without changing the author's rollout code.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
EXTRA=()
if [[ "${3:-}" == preflight ]]; then
    EXTRA+=(--preflight-only)
elif [[ -n "${3:-}" ]]; then
    exit 2
fi
cd "$PROJECT"
"$PROJECT/.runtime/condt-author-env/bin/python" "$CAMPAIGN/source/run.py" \
    --root "$CAMPAIGN" --arm "$ARM" --device "${CONDT_DEVICE:-cuda}" "${EXTRA[@]}"
