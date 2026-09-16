#!/bin/bash
# CPU fallback for the small DT oracle pilot; avoids waiting for a free GPU.
#SBATCH --job-name=dt-oracle-pilot-cpu
#SBATCH --partition=GPUNorm
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/oracle-branching-pilot-cpu-%j.out

set -euo pipefail

PROJECT=/labmount/users/202615385/code/DT-EXP
TARGET_RETURN="${TARGET_RETURN:-12000}"
OUTPUT_TAG="${OUTPUT_TAG:-rtg${TARGET_RETURN}}"

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
  --output-dir "${PROJECT}/results/oracle_branching/pilot_seed0_${OUTPUT_TAG}" \
  --device cpu \
  --target-return "${TARGET_RETURN}" \
  --reward-mode delayed \
  --eval-seed 42 \
  --eval-episodes 5 \
  --anchors-per-episode 10 \
  --num-pairs 8 \
  --alphas 0.05 0.10 0.20 0.30 \
  --knn-neighbors 64 \
  --bootstrap-reps 5000
