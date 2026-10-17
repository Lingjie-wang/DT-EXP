#!/bin/bash
#SBATCH --job-name=cql-corl-delayed
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/cql-corl-delayed-%j.out

set -euo pipefail
PROJECT=/labmount/users/202615385/code/DT-EXP
CAMPAIGN=${1:?campaign required}
ARM=${2:?medium/medium_replay required}
source /labmount/users/202615385/miniforge3/etc/profile.d/conda.sh
conda activate adt-delayed
unset PYTHONPATH
export MUJOCO_PY_MUJOCO_PATH=/labmount/users/202615385/.mujoco/mujoco210
export C_INCLUDE_PATH="${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export CFLAGS="${CFLAGS:-} -Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types"
export PYOPENGL_PLATFORM=egl MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$CAMPAIGN/$ARM/hf_cache"
export PYTHONPATH="$CAMPAIGN/source"
cd "$CAMPAIGN/source"
PYTHON="$PROJECT/.runtime/lpt-official-env/bin/python"
ENTRY="$CAMPAIGN/source/scripts/cql_delayed/train.py"
# Independent preflight process leaves formal initialization/RNG untouched.
"$PYTHON" "$ENTRY" --root "$CAMPAIGN" --arm "$ARM" --smoke
"$PYTHON" "$ENTRY" --root "$CAMPAIGN" --arm "$ARM"
