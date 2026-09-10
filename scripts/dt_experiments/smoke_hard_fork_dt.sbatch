#!/bin/bash
#SBATCH --job-name=hfdt-smoke
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/hard-fork-dt-smoke-%j.out

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
export WANDB_MODE=disabled

cd "${PROJECT}"
for MODE in hard_positive hard_fork; do
  python algorithms/offline/hard_fork_dt.py \
    --config_path "${PROJECT}/configs/offline/dt/halfcheetah/medium_replay_v2.yaml" \
    --project corl-ddr \
    --name "Smoke-${MODE}" \
    --reward_mode delayed \
    --train_seed 0 \
    --device cuda:0 \
    --update_steps 50002 \
    --eval_every 100000 \
    --eval_episodes 1 \
    --num_workers 0 \
    --preference_start_step 50000 \
    --preference_mode "${MODE}" \
    --preference_weight 0.05 \
    --reference_weight 0.1 \
    --preference_margin 0.05 \
    --preference_batch_size 256 \
    --preference_hardness_temperature 0.05 \
    --pretrained_checkpoint_path "${PROJECT}/checkpoints/dt-halfcheetah-medium-replay-v2-delayed-seed0-step50000.pt" \
    --preference_arrays_path "${PROJECT}/results/pair_diagnostics/dt_pair_diagnostic_halfcheetah_medium_replay_delayed_seed0.npz"
done
