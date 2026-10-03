#!/bin/bash
#SBATCH --job-name=dd-hcmr
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=14-00:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/decision-diffuser-%j.out

set -euo pipefail
PROJECT=/labmount/users/202615385/code/DT-EXP
CAMPAIGN=${1:?campaign required}
ARM=${2:?dense/delayed/cpu-smoke required}
source /labmount/users/202615385/miniforge3/etc/profile.d/conda.sh
conda activate adt-delayed
export PYTHONPATH="$PROJECT/.runtime/decision-diffuser-deps:$PROJECT/.runtime/threshold-cu117:${PYTHONPATH:-}"
export CUDA_MODULE_LOADING=LAZY
export MUJOCO_PY_MUJOCO_PATH=/labmount/users/202615385/.mujoco/mujoco210
export C_INCLUDE_PATH="${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export CFLAGS="${CFLAGS:-} -Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types"
export PYOPENGL_PLATFORM=egl MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled
cd "$PROJECT"
ENTRY="$CAMPAIGN/source/decision_diffuser_repro.py"
if [ "$ARM" = cpu-smoke ]; then
    export CUDA_VISIBLE_DEVICES=""
    python "$ENTRY" smoke --root "$CAMPAIGN" --device cpu
else
    # Do not share mutable smoke files when the two arms start concurrently.
    # Each arm validates actual full-batch loss before its first optimizer update;
    # CPU smoke has already exercised the full diffusion sampler and simulator.
    python "$ENTRY" train --root "$CAMPAIGN" --arm "$ARM" --device cuda:0
    python "$ENTRY" summarize --root "$CAMPAIGN"
fi
