#!/bin/bash
#SBATCH --job-name=cql-env-validation
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/cql-env-validation-%j.out

set -euo pipefail
DELAYED=${1:?delayed retry root required}
SHAPLEY=${2:?Shapley retry root required}
source /labmount/users/202615385/miniforge3/etc/profile.d/conda.sh
conda activate adt-delayed
export MUJOCO_PY_MUJOCO_PATH=/labmount/users/202615385/.mujoco/mujoco210
export C_INCLUDE_PATH="${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export CFLAGS="${CFLAGS:-} -Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types"
export PYOPENGL_PLATFORM=egl MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON=/labmount/users/202615385/code/DT-EXP/.runtime/lpt-official-env/bin/python
for arm in medium medium_replay; do
    export PYTHONPATH="$DELAYED/source"
    "$PYTHON" "$DELAYED/source/scripts/cql_delayed/train.py" \
        --root "$DELAYED/validation" --arm "$arm" --smoke
done
export PYTHONPATH="$SHAPLEY/source"
"$PYTHON" "$SHAPLEY/source/scripts/cql_shapley/train.py" \
    --root "$SHAPLEY/validation" --arm shapley --smoke
