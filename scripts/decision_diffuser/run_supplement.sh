#!/bin/bash
# Run using srun --jobid=TRAIN_JOB --overlap on the existing GPU allocation.
set -euo pipefail
PROJECT=/labmount/users/202615385/code/DT-EXP
CAMPAIGN=${1:?campaign required}
ARM=${2:?arm required}
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled
cd "$PROJECT"
python "$CAMPAIGN/execution/supplement_evaluations.py" run \
    --root "$CAMPAIGN" --arm "$ARM"
