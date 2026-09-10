#!/bin/bash
#SBATCH --job-name=sapdt2-smoke
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/sapdt2-smoke-%j.out

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
export WANDB_MODE=disabled

cd "${PROJECT}"
python algorithms/offline/sap_dt_one_sided.py \
  --config_path "${PROJECT}/configs/offline/dt/halfcheetah/medium_replay_v2.yaml" \
  --reward_mode delayed \
  --train_seed 0 \
  --device cuda:0 \
  --update_steps 2 \
  --eval_every 1 \
  --eval_episodes 1 \
  --batch_size 32 \
  --num_workers 0 \
  --preference_start_step 0 \
  --preference_weight 0.001 \
  --reference_weight 0.1 \
  --preference_margin 0.05 \
  --preference_batch_size 16 \
  --preference_num_pairs 1000 \
  --preference_num_candidates 8 \
  --preference_state_keep_fraction 0.25 \
  --reference_target_return 12000.0
