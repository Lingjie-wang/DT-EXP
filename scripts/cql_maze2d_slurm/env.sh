#!/usr/bin/env bash
# Source this file with the frozen campaign path as its first argument.
ROOT=${1:?campaign root required}
BASE=/labmount/users/202615385/miniforge3/envs/adt-delayed
PYTHON=/labmount/users/202615385/code/DT-EXP/.runtime/cql-maze2d-slurm-wandb012-20261017/bin/python
export MUJOCO_PY_MUJOCO_PATH=/labmount/users/202615385/.mujoco/mujoco210
export C_INCLUDE_PATH="$BASE/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="$BASE/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$BASE/lib:$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export CFLAGS="${CFLAGS:-} -Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYOPENGL_PLATFORM=egl MUJOCO_GL=egl WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
export WANDB_MODE=online
export PYTHONPATH="$ROOT/dependencies/d4rl:$ROOT/source"
# CUDA_VISIBLE_DEVICES is assigned by Slurm; never replace it with physical GPU 0.
