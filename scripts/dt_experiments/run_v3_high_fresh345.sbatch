#!/bin/bash
# Stage argument: smoke | prepare | branch SEED
# prepare: --array=3-5; branch: --array=0-1 (0=DT, 1=v3-high).
#SBATCH --job-name=v3-high-fresh345
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/v3-high-fresh345-%A_%a.out

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
export WANDB_DISABLE_CODE=true
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "${PROJECT}"

CAMPAIGN=v3-high-fresh345-20260919
case "${1:-}" in
  smoke|tests)
    PYTHONPATH=algorithms/offline:scripts/dt_experiments python -m unittest discover \
      -s tests -p test_paired_hard_fork_resume.py -v
    PYTHONPATH=algorithms/offline:scripts/dt_experiments python -m unittest discover \
      -s tests -p test_top_return_weighted_dt.py -v
    PYTHONPATH=algorithms/offline:scripts/dt_experiments python -m unittest discover \
      -s tests -p test_v3_high_newseeds.py -v
    if [[ "${1}" == tests ]]; then exit 0; fi
    python scripts/dt_experiments/v3_high_newseeds.py smoke --campaign "${CAMPAIGN}"
    ;;
  prepare)
    python scripts/dt_experiments/v3_high_newseeds.py prepare \
      --campaign "${CAMPAIGN}" --seed "${SLURM_ARRAY_TASK_ID}"
    ;;
  branch)
    case "${SLURM_ARRAY_TASK_ID}" in
      0) ARM=dt ;;
      1) ARM=v3 ;;
      *) exit 2 ;;
    esac
    python scripts/dt_experiments/v3_high_newseeds.py branch \
      --campaign "${CAMPAIGN}" --seed "${2}" --arm "${ARM}"
    ;;
  *) exit 2 ;;
esac
