#!/bin/bash
#SBATCH --job-name=dt-pairdiag-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/dt-pairdiag-hcmr-seed0-%j.out

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
python algorithms/offline/diagnose_sap_pairs.py \
  --checkpoint "${PROJECT}/checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed0-step50000.pt" \
  --env_name halfcheetah-medium-replay-v2 \
  --reward_mode delayed \
  --device cuda:0 \
  --batch_size 512 \
  --seed 0 \
  --num_pairs 100000 \
  --num_candidates 64 \
  --state_keep_fraction 0.25 \
  --prefix_length 5 \
  --min_prefix_steps 5 \
  --prefix_state_threshold 0.75 \
  --prefix_action_threshold 0.50 \
  --max_timestep_gap 5 \
  --action_difference_threshold 0.25 \
  --min_return_gap 2000 \
  --preference_margin 0.05 \
  --result_dir "${PROJECT}/results/pair_diagnostics" \
  --wandb_project corl-ddr \
  --wandb_group SAP-DT-pair-diagnostic-delayed-halfcheetah-medium-replay-v2 \
  --wandb_name DTPairDiagnostic-HCMR-delayed-seed0
