#!/bin/bash
# sbatch --array=0-1 this_script [smoke]
# 0: v3-high without reference optimization; 1: matched original reference=0.1.
# Keep reference forward calls at weight zero to preserve RNG consumption and
# log the unweighted deviation. No dropout or preference logic changes.
#SBATCH --job-name=v3-ref-ablate-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/v3-reference-ablation-to100k-seed0-%A_%a.out

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
export PYTHONUNBUFFERED=1

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) ARM=NoRef; REFERENCE_WEIGHT=0.0 ;;
  1) ARM=Ref010; REFERENCE_WEIGHT=0.1 ;;
  *) exit 2 ;;
esac
RUN_NAME="V3High-${ARM}-50kTo100k-HCMR-delayed-seed0"
EXTRA_ARGS=()
if [[ "${1:-}" == smoke ]]; then
  RUN_NAME="Smoke-${RUN_NAME}"
  export WANDB_MODE=offline
  EXTRA_ARGS=(--update_steps 50002 --eval_episodes 1
    --checkpoint_steps '[50001]')
fi
cd "${PROJECT}"
git rev-parse HEAD
python algorithms/offline/hard_fork_dt.py \
  --config_path configs/offline/dt/halfcheetah/medium_replay_v2.yaml \
  --project corl-ddr \
  --group V3High-ReferenceAblation-50kTo100k-HCMR-delayed-seed0 \
  --name "${RUN_NAME}" \
  --reward_mode delayed --train_seed 0 --device cuda:0 \
  --update_steps 100001 --eval_every 5000 --eval_episodes 100 \
  --preference_start_step 50000 --preference_mode hard_fork \
  --preference_weight 0.05 --reference_weight "${REFERENCE_WEIGHT}" \
  --reference_anchor_mode mse --reference_tolerance 0.0 \
  --preference_margin 0.05 --preference_batch_size 256 \
  --preference_hardness_temperature 0.05 \
  --active_only_normalization true --preference_min_active_pairs 16 \
  --prioritized_pair_sampling true \
  --dynamic_priority_mix 0.5 --dynamic_priority_ema 0.9 \
  --target_aligned_preference true --preference_target_mode high_only \
  --preference_target_return_high 12000.0 --preference_pair_seed 0 \
  --pretrained_checkpoint_path "${PROJECT}/checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed0-step50000.pt" \
  --preference_arrays_path "${PROJECT}/results/pair_diagnostics/dt_pair_diagnostic_halfcheetah_medium_replay_delayed_seed0.npz" \
  --checkpoint_steps '[75000,100000]' \
  --checkpoints_path "${PROJECT}/checkpoints/v3-reference-ablation-to100k/${RUN_NAME}" \
  "${EXTRA_ARGS[@]}"
