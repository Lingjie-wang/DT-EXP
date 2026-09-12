#!/bin/bash
# State-relative outcome-weighted delayed DT.  It preserves CORL's recorded
# delayed-RTG behavior-cloning loss and adds a high-RTG=12000 auxiliary loss
# only for transitions whose observed terminal outcomes are strong relative to
# cross-trajectory nearest states.  It resumes the same seed-0 DT 50k state as
# v3/v4/v5 and evaluates both standard targets for comparability.
#SBATCH --job-name=srdt-hi-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/state-relative-weighted-dt-delayed-hcmr-seed0-%j.out

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
python algorithms/offline/state_relative_dt.py \
  --config_path "${PROJECT}/configs/offline/dt/halfcheetah/medium_replay_v2.yaml" \
  --project corl-ddr \
  --group StateRelative-WeightedDT-delayed-HCMR-seed0 \
  --name StateRelative-WeightedDT-HCMR-delayed-seed0 \
  --reward_mode delayed \
  --train_seed 0 \
  --device cuda:0 \
  --update_steps 75001 \
  --eval_every 5000 \
  --eval_episodes 100 \
  --quality_mode state_relative \
  --quality_fraction 0.30 \
  --quality_aux_weight 0.10 \
  --reference_weight 0.10 \
  --quality_target_return 12000.0 \
  --quality_aux_batch_size 1024 \
  --state_neighbor_count 64 \
  --state_neighbor_candidate_multiplier 4 \
  --state_knn_chunk_size 4096 \
  --pretrained_checkpoint_path "${PROJECT}/checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed0-step50000.pt" \
  --checkpoints_path "${PROJECT}/checkpoints/state-relative-weighted-dt-delayed-hcmr-seed0"
