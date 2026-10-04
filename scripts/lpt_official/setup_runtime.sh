#!/bin/bash
# Separate environment: never update the runtime of an existing experiment.
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")/../.." && pwd)
BASE=/labmount/users/202615385/miniforge3/envs/adt-delayed/bin/python
ENV_DIR="$PROJECT/.runtime/lpt-official-env"
"$BASE" -m venv --system-site-packages "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --no-deps 'torch==2.0.1+cu118' \
    --index-url https://download.pytorch.org/whl/cu118
"$ENV_DIR/bin/python" -m pip install -r "$PROJECT/scripts/lpt_official/requirements.txt"
"$ENV_DIR/bin/python" -m pip install --no-deps \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.6/flash_attn-2.3.6%2Bcu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl'
"$ENV_DIR/bin/python" -m pip freeze > "$ENV_DIR/installed-packages.txt"

