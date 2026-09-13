#!/bin/bash
# Read-only diagnostic for a broader, local variant of v3-high's hard-fork
# pairs. It never updates the DT checkpoint: it only counts independent
# state/time/return/action-filtered pairs that remain margin-active at 50k.
#SBATCH --job-name=nhf-pairdiag-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/neighborhood-hard-fork-pairdiag-hcmr-seed0-%j.out

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
python algorithms/offline/diagnose_neighborhood_hard_fork_pairs.py \
  --checkpoint "${PROJECT}/checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed0-step50000.pt" \
  --env_name halfcheetah-medium-replay-v2 \
  --reward_mode delayed \
  --device cuda:0 \
  --seed 0 \
  --batch_size 512 \
  --neighbor_count 64 \
  --projection_bits 8 \
  --projection_seed 91731 \
  --query_chunk_size 256 \
  --max_timestep_gap 5 \
  --max_state_rmse 0.75 \
  --min_return_gap 2000 \
  --min_action_rmse 0.25 \
  --preference_margin 0.05 \
  --target_return 12000 \
  --result_dir "${PROJECT}/results/neighborhood_pair_diagnostics" \
  --wandb_project corl-ddr \
  --wandb_group Neighborhood-HardFork-pair-diagnostic-HCMR-delayed-seed0 \
  --wandb_name NeighborhoodHardForkDiagnostic-HCMR-delayed-seed0 \
  --wandb_mode online
