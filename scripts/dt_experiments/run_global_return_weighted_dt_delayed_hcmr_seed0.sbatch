#!/bin/bash
# Global-return control for State-relative Weighted DT.  Every continuation,
# high-RTG auxiliary, reference anchor, optimizer, and evaluation setting is
# identical to the State-relative seed-0 run.  Only quality_mode changes:
# select the top 30% transitions by trajectory return rather than by local
# cross-trajectory state-relative advantage.
#SBATCH --job-name=grwdt-hi-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/global-return-weighted-dt-delayed-hcmr-seed0-%j.out

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
  --group GlobalReturn-WeightedDT-delayed-HCMR-seed0 \
  --name GlobalReturn-WeightedDT-HCMR-delayed-seed0 \
  --reward_mode delayed \
  --train_seed 0 \
  --device cuda:0 \
  --update_steps 75001 \
  --eval_every 5000 \
  --eval_episodes 100 \
  --quality_mode global_return \
  --quality_fraction 0.30 \
  --quality_aux_weight 0.10 \
  --reference_weight 0.10 \
  --quality_target_return 12000.0 \
  --quality_aux_batch_size 1024 \
  --state_neighbor_count 64 \
  --state_projection_bits 8 \
  --state_neighbor_query_chunk_size 256 \
  --state_neighbor_seed 91731 \
  --pretrained_checkpoint_path "${PROJECT}/checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed0-step50000.pt" \
  --checkpoints_path "${PROJECT}/checkpoints/global-return-weighted-dt-delayed-hcmr-seed0"
