#!/bin/bash
#SBATCH --job-name=qt-fixed-eta001-hcmr-s0
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/qt-corrected-%j.out

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
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_DISABLE_CODE=true
cd "${PROJECT}"
QT_CODE="${QT_CORRECTED_SOURCE:-${PROJECT}}"

# A submitted job uses an immutable source copy and verifies its recorded hashes.
if [[ -n "${QT_CORRECTED_MANIFEST:-}" ]]; then
  sha256sum --check "${QT_CORRECTED_MANIFEST}"
fi
QT_PYTHON="${PROJECT}/third_party/qt-env/bin/python"
QT_OUTPUT="${PROJECT}/checkpoints/qt-terminal-corrected-eta001/job-${SLURM_JOB_ID}"
QT_COMMON=(--source "${PROJECT}/third_party/QT" --seed 0 --eta 0.01
  --correct-terminal-rewards --updates 100000 --eval-every 5000 --eval-episodes 100)

PYTHONPATH="${QT_CODE}/algorithms/offline" "${QT_PYTHON}" -m unittest discover \
  -s "${QT_CODE}/tests" -p 'test_qt_*.py' -v

# Use the actual model, batch and all four evaluation protocols before full training.
export WANDB_MODE=offline
"${QT_PYTHON}" "${QT_CODE}/algorithms/offline/qt_official_runner.py" \
  "${QT_COMMON[@]}" --output "${QT_OUTPUT}/smoke" --smoke
"${QT_PYTHON}" "${QT_CODE}/scripts/dt_experiments/verify_qt_corrected_smoke.py" \
  "${QT_OUTPUT}/smoke"

export WANDB_MODE=online
"${QT_PYTHON}" "${QT_CODE}/algorithms/offline/qt_official_runner.py" \
  "${QT_COMMON[@]}" --output "${QT_OUTPUT}/train"
