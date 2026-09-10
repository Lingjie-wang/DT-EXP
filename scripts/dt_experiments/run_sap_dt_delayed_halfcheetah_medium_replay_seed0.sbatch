#!/bin/bash
#SBATCH --job-name=sapdt-d-hcr-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/sapdt-delayed-hc-medium-replay-seed0-%j.out

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
export WANDB_MODE=online

cd "${PROJECT}"
python algorithms/offline/sap_dt.py \
  --config_path "${PROJECT}/configs/offline/dt/halfcheetah/medium_replay_v2.yaml" \
  --project corl-ddr \
  --group DT-vs-SAPDT-delayed-halfcheetah-medium-replay-v2-seed0 \
  --name SAPDTK1DelayedMR \
  --reward_mode delayed \
  --train_seed 0 \
  --device cuda:0 \
  --preference_start_step 50000 \
  --preference_weight 0.01 \
  --preference_temperature 1.0 \
  --preference_batch_size 256 \
  --preference_num_pairs 100000 \
  --preference_num_candidates 64 \
  --preference_target_return 12000.0 \
  --preference_pair_seed 0
