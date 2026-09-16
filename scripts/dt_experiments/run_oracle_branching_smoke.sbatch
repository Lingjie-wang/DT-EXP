#!/bin/bash
#SBATCH --job-name=dt-oracle-smoke
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/oracle-branching-smoke-%j.out

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

cd "${PROJECT}"
python algorithms/offline/oracle_branching_dt.py \
  --checkpoint-path "${PROJECT}/checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed0-step50000.pt" \
  --output-dir "${PROJECT}/results/oracle_branching/smoke_seed0_rtg12000" \
  --device cuda:0 \
  --target-return 12000 \
  --reward-mode delayed \
  --eval-seed 42 \
  --eval-episodes 1 \
  --anchors-per-episode 10 \
  --num-pairs 1 \
  --alphas 0.0 \
  --knn-neighbors 64 \
  --bootstrap-reps 200
