#!/bin/bash
#SBATCH --job-name=dt50k-pairprep-s2
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/dt50k-pairprep-hcmr-seed2-%j.out

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
python algorithms/offline/sap_dt_one_sided.py \
  --config_path "${PROJECT}/configs/offline/dt/halfcheetah/medium_replay_v2.yaml" \
  --project corl-ddr \
  --group DT-pair-diagnostic-preparation \
  --name DT50kPairDiagnosticPrep-seed2 \
  --reward_mode delayed \
  --train_seed 2 \
  --device cuda:0 \
  --update_steps 50001 \
  --eval_every 50000 \
  --eval_episodes 10 \
  --preference_start_step 50000 \
  --preference_weight 0.001 \
  --reference_weight 0.1 \
  --preference_margin 0.05 \
  --preference_batch_size 256 \
  --preference_num_pairs 100000 \
  --preference_num_candidates 64 \
  --preference_state_keep_fraction 0.25 \
  --reference_target_return 12000.0 \
  --reference_checkpoint_path "${PROJECT}/checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed2-step50000.pt" \
  --preference_pair_seed 2
