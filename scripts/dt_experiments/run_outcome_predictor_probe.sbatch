#!/bin/bash
# Resource requests are supplied at submission for CPU replay vs GPU training.
#SBATCH --job-name=outcome-probe
#SBATCH --partition=GPUNorm
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/outcome-probe-%j.out
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
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "${PROJECT}"
PROBE_ROOT="${PROJECT}/results/outcome_predictor_probe/v1"
python algorithms/offline/outcome_predictor_probe.py selftest --output-dir "${PROBE_ROOT}/selftest"
case "${PROBE_MODE:-train}" in
  train)
    python algorithms/offline/outcome_predictor_probe.py train --output-dir "${PROBE_ROOT}/models" --device cuda:0
    ;;
  replay)
    python algorithms/offline/outcome_predictor_probe.py replay --output-dir "${PROBE_ROOT}/development_cache" --eval-seed 42
    ;;
  evaluate)
    python algorithms/offline/outcome_predictor_probe.py analyze --output-dir "${PROBE_ROOT}/development_ranking" --model-dir "${PROBE_ROOT}/models" --anchor-cache "${PROBE_ROOT}/development_cache/anchors.pt"
    python algorithms/offline/outcome_predictor_probe.py heldout --output-dir "${PROBE_ROOT}/heldout_ranking" --model-dir "${PROBE_ROOT}/models" --eval-seed 20261001 --eval-episodes 20
    ;;
  shard)
    python algorithms/offline/outcome_predictor_probe.py heldout --output-dir "${PROBE_ROOT}/heldout_shard_${PROBE_START}_${PROBE_STOP}" --model-dir "${PROBE_ROOT}/models" --eval-seed 20261001 --eval-episodes 20 --eval-start "${PROBE_START}" --eval-stop "${PROBE_STOP}"
    ;;
  *) exit 2 ;;
esac
