#!/bin/bash
# Fair v3-high continuation using only the frozen-margin-active pairs exported
# by the neighborhood diagnostic.  All DT, reference, RTG, and evaluation
# settings are identical to v3-high seed 0; only the pair source changes.
#SBATCH --job-name=nhf-active-hi-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/neighborhood-active-hard-fork-high-delayed-hcmr-seed0-%j.out

set -euo pipefail

PROJECT=/labmount/users/202615385/code/DT-EXP
PAIR_SOURCE="${PROJECT}/results/neighborhood_pair_diagnostics/neighborhood_hard_fork_diagnostic_hcmr_delayed_seed0_margin_active_pairs.npz"
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

if [[ ! -s "${PAIR_SOURCE}" ]]; then
  echo "Missing frozen-margin-active pair source: ${PAIR_SOURCE}" >&2
  exit 1
fi

cd "${PROJECT}"
python algorithms/offline/hard_fork_dt.py \
  --config_path "${PROJECT}/configs/offline/dt/halfcheetah/medium_replay_v2.yaml" \
  --project corl-ddr \
  --group NeighborhoodActive-HardFork-high-delayed-HCMR-seed0 \
  --name NeighborhoodActive-HardFork-high-HCMR-delayed-seed0 \
  --reward_mode delayed \
  --train_seed 0 \
  --device cuda:0 \
  --update_steps 75001 \
  --eval_every 5000 \
  --eval_episodes 100 \
  --preference_start_step 50000 \
  --preference_mode hard_fork \
  --preference_weight 0.05 \
  --reference_weight 0.1 \
  --preference_margin 0.05 \
  --preference_batch_size 256 \
  --preference_hardness_temperature 0.05 \
  --active_only_normalization true \
  --preference_min_active_pairs 16 \
  --prioritized_pair_sampling true \
  --dynamic_priority_mix 0.5 \
  --dynamic_priority_ema 0.9 \
  --target_aligned_preference true \
  --preference_target_mode high_only \
  --preference_target_return_high 12000.0 \
  --pretrained_checkpoint_path "${PROJECT}/checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed0-step50000.pt" \
  --preference_arrays_path "${PAIR_SOURCE}" \
  --checkpoints_path "${PROJECT}/checkpoints/neighborhood-active-hard-fork-high-hcmr-delayed-seed0"
