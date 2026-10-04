#!/bin/bash
#SBATCH --job-name=original-dt-hcmr
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/original-dt-%j.out

set -euo pipefail
PROJECT=/labmount/users/202615385/code/DT-EXP
CAMPAIGN=${1:?campaign required}
ARM=${2:?dense/delayed required}
source /labmount/users/202615385/miniforge3/etc/profile.d/conda.sh
conda activate adt-delayed
export MUJOCO_PY_MUJOCO_PATH=/labmount/users/202615385/.mujoco/mujoco210
export C_INCLUDE_PATH="${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export CFLAGS="${CFLAGS:-} -Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types"
export PYOPENGL_PLATFORM=egl MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled
cd "$PROJECT"
EXTRA=()
if [[ "${3:-}" == smoke ]]; then
    export CUDA_VISIBLE_DEVICES=""
    EXTRA=(--smoke --device cpu)
elif [[ -n "${3:-}" ]]; then
    exit 2
else
    # Validate the real simulator and a complete batch on the allocated node
    # before allowing the long run. The login node lacks OSMesa development files.
    "$PROJECT/third_party/qt-env/bin/python" "$CAMPAIGN/source/original_dt_repro.py" \
        train --root "$CAMPAIGN" --arm "$ARM" --smoke --device cuda:0
fi
"$PROJECT/third_party/qt-env/bin/python" "$CAMPAIGN/source/original_dt_repro.py" \
    train --root "$CAMPAIGN" --arm "$ARM" "${EXTRA[@]}"
