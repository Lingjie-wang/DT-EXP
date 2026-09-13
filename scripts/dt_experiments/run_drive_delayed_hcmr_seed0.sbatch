#!/bin/bash
# Paper-faithful DRIVE structure (GMM DT + retrieval + IQL ranking) on the
# project's delayed-reward HalfCheetah medium-replay benchmark.
#SBATCH --job-name=drive-hcmr-d-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/drive-hcmr-delayed-seed0-%j.out

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

cd "${PROJECT}"
python algorithms/offline/drive.py \
  --config_path "${PROJECT}/configs/offline/drive/halfcheetah/medium_replay_v2_delayed.yaml"
