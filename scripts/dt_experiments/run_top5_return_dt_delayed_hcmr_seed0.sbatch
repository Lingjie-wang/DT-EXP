#!/bin/bash
# Run both arms: sbatch --array=0-1 this_script
# Array 0 is DT control, array 1 doubles the top-5%-trajectory MSE weight.
# Optional smoke: sbatch --array=0-1 this_script smoke
#SBATCH --job-name=top5dt-hcmr-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/top5dt-hcmr-seed0-%A_%a.out

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

case "${SLURM_ARRAY_TASK_ID:-1}" in
  0) ARM=DT-Control; TOP_WEIGHT=1.0 ;;
  1) ARM=Top5Return-Weight2-DT; TOP_WEIGHT=2.0 ;;
  *) exit 2 ;;
esac
RUN_NAME="${ARM}-FromScratch-HCMR-delayed-seed0"
EXTRA_ARGS=()
if [[ "${1:-}" == smoke ]]; then
  RUN_NAME="Smoke-${RUN_NAME}"
  export WANDB_MODE=offline
  EXTRA_ARGS=(--update_steps 2 --eval_every 2 --eval_episodes 1 --log_every 1)
fi
cd "${PROJECT}"
if [[ "${1:-}" == smoke ]]; then
  PYTHONPATH=algorithms/offline python -m unittest discover -s tests \
    -p test_top_return_weighted_dt.py -v
fi
python algorithms/offline/top_return_weighted_dt.py \
  --config_path configs/offline/dt/halfcheetah/medium_replay_v2.yaml \
  --project corl-ddr \
  --group Top5Return-FromScratch-HCMR-delayed-seed0 \
  --name "${RUN_NAME}" \
  --reward_mode delayed --train_seed 0 --device cuda:0 \
  --top_fraction 0.05 --top_weight "${TOP_WEIGHT}" \
  --update_steps 100000 --eval_every 5000 --eval_episodes 100 \
  --checkpoints_path "${PROJECT}/checkpoints/top5-return-fromscratch/${RUN_NAME}" \
  "${EXTRA_ARGS[@]}"
