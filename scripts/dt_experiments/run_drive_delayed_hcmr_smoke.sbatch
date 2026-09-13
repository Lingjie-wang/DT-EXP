#!/bin/bash
# One-step end-to-end validation of the DRIVE pipeline.  It builds the actual
# retrieval index and executes both candidate sources in the D4RL environment.
#SBATCH --job-name=drive-smoke-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/drive-hcmr-delayed-smoke-%j.out

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
  --config_path "${PROJECT}/configs/offline/drive/halfcheetah/medium_replay_v2_delayed.yaml" \
  --name DRIVE-HCMR-delayed-smoke-seed0 \
  --policy_update_steps 1 \
  --critic_update_steps 1 \
  --eval_every 1 \
  --eval_episodes 1 \
  --num_workers 0
