#!/bin/bash
# Array 0: ordinary DT objective; array 1: original v3-high objective.
# Both execute identical auxiliary forward calls; control weights are BOTH zero.
#SBATCH --job-name=new50k-dt-v3-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/new50k-paired-dt-v3-to75k-seed0-%A_%a.out

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
  0) ARM=DTControl; PREFERENCE_WEIGHT=0.0; REFERENCE_WEIGHT=0.0 ;;
  1) ARM=V3High; PREFERENCE_WEIGHT=0.05; REFERENCE_WEIGHT=0.1 ;;
  *) exit 2 ;;
esac
RUN_NAME="New50k-Paired-${ARM}-To75k-HCMR-delayed-seed0"
EXTRA_ARGS=()
if [[ "${1:-}" == smoke ]]; then
  RUN_NAME="Smoke-${RUN_NAME}"
  export WANDB_MODE=offline
  EXTRA_ARGS=(--update_steps 50003 --eval_every 1 --eval_episodes 1
    --checkpoint_steps '[50003]')
fi
cd "${PROJECT}"
git rev-parse HEAD
python algorithms/offline/hard_fork_dt.py \
  --config_path configs/offline/dt/halfcheetah/medium_replay_v2.yaml \
  --project corl-ddr --group New50k-Paired-DT-v3-50kTo75k-HCMR-delayed-seed0 \
  --name "${RUN_NAME}" \
  --reward_mode delayed --train_seed 0 --device cuda:0 \
  --paired_resume true --pairing_audit_steps 3 \
  --update_steps 75000 --eval_every 5000 --eval_episodes 100 \
  --preference_start_step 50000 --preference_mode hard_fork \
  --preference_weight "${PREFERENCE_WEIGHT}" --reference_weight "${REFERENCE_WEIGHT}" \
  --reference_anchor_mode mse --reference_tolerance 0.0 \
  --preference_margin 0.05 --preference_batch_size 256 \
  --preference_hardness_temperature 0.05 \
  --active_only_normalization true --preference_min_active_pairs 16 \
  --prioritized_pair_sampling true \
  --dynamic_priority_mix 0.5 --dynamic_priority_ema 0.9 \
  --target_aligned_preference true --preference_target_mode high_only \
  --preference_target_return_high 12000.0 --preference_pair_seed 0 \
  --pretrained_checkpoint_path "${PROJECT}/checkpoints/top5-return-fromscratch/DT-Control-FromScratch-HCMR-delayed-seed0/DT-Control-FromScratch-HCMR-delayed-seed0-halfcheetah-medium-replay-v2-61bc26ef/step050000.pt" \
  --preference_arrays_path "${PROJECT}/results/pair_diagnostics/dt_pair_diagnostic_halfcheetah_medium_replay_delayed_seed0.npz" \
  --checkpoint_steps '[55000,60000,65000,70000,75000]' \
  --checkpoints_path "${PROJECT}/checkpoints/new50k-paired-dt-v3-to75k/${RUN_NAME}" \
  "${EXTRA_ARGS[@]}"
