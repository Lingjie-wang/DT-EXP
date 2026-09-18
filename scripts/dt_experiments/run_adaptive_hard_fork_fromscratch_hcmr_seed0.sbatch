#!/bin/bash
# 0=ordinary DT; 1=DT+recent reference; 2=current-scored preference+recent reference.
#SBATCH --job-name=adaptive-hf-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/adaptive-hf-fromscratch-%A_%a.out

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
export PYTHONUNBUFFERED=1
export WANDB_DISABLE_CODE=true
export WANDB_MODE=online
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) ARM=dt; LABEL=DTControl ;;
  1) ARM=recent_ref; LABEL=RecentRef ;;
  2) ARM=adaptive_pref; LABEL=AdaptivePrefRecentRef ;;
  *) exit 2 ;;
esac
RUN_NAME="AdaptiveHF-${LABEL}-FromScratch100k-HCMR-delayed-seed0"
EXTRA_ARGS=()
cd "${PROJECT}"
if [[ "${1:-}" == smoke ]]; then
  export WANDB_MODE=offline
  RUN_NAME="Smoke-${RUN_NAME}"
  EXTRA_ARGS=(--update_steps 9 --auxiliary_start 2 --auxiliary_full 4
    --reference_every 3 --rescore_every 2 --eval_every 3 --eval_episodes 1
    --checkpoint_steps '[0,2,4,6,9]' --log_every 1)
  PYTHONPATH=algorithms/offline python -m unittest discover -s tests \
    -p test_adaptive_hard_fork_dt.py -v
elif [[ -n "${1:-}" ]]; then
  exit 2
fi
python algorithms/offline/adaptive_hard_fork_dt.py \
  --config_path configs/offline/dt/halfcheetah/medium_replay_v2.yaml \
  --project corl-ddr --group AdaptiveHF-FromScratch100k-HCMR-delayed-seed0 \
  --name "${RUN_NAME}" --arm "${ARM}" \
  --reward_mode delayed --train_seed 0 --device cuda:0 \
  --update_steps 100000 --eval_every 5000 --eval_episodes 100 \
  --auxiliary_start 10000 --auxiliary_full 20000 \
  --reference_every 5000 --rescore_every 1000 \
  --preference_weight 0.05 --reference_weight 0.1 --preference_margin 0.05 \
  --preference_batch_size 256 --min_active_pairs 16 --target_return 12000 \
  --coverage_fraction 0.2 --violation_cap 5.0 --pair_seed 0 \
  --pair_arrays_path "${PROJECT}/results/pair_diagnostics/dt_pair_diagnostic_halfcheetah_medium_replay_delayed_seed0.npz" \
  --checkpoints_path "${PROJECT}/checkpoints/adaptive-hf-fromscratch/job-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}/${ARM}" \
  "${EXTRA_ARGS[@]}"
