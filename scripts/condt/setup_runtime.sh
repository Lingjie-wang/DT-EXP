#!/bin/bash
# Build a separate runtime; never install into an existing experiment's environment.
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")/../.." && pwd)
BASE=/labmount/users/202615385/miniforge3/envs/adt-delayed/bin/python
ENV_DIR="$PROJECT/.runtime/condt-author-env"
mkdir -p "$ENV_DIR"
if [[ ! -f "$ENV_DIR/pyvenv.cfg" ]]; then
    "$BASE" -m venv --system-site-packages "$ENV_DIR"
fi
PYTHON="$ENV_DIR/bin/python"
"$PYTHON" - <<'PY'
import sys
assert sys.prefix != sys.base_prefix, "Refusing to install outside an isolated venv"
assert sys.prefix.endswith("/.runtime/condt-author-env"), sys.prefix
PY
export PIP_CACHE_DIR="$ENV_DIR/pip-cache"
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
# Save dependency resolution before making any installation changes.
"$PYTHON" -m pip install --dry-run --report "$ENV_DIR/pip-resolution-$RUN_STAMP.json" \
    -r "$PROJECT/scripts/condt/requirements.txt" 2>&1 | tee "$ENV_DIR/pip-resolution-$RUN_STAMP.log"
"$PYTHON" -m pip install -r "$PROJECT/scripts/condt/requirements.txt" \
    2>&1 | tee "$ENV_DIR/pip-install-$RUN_STAMP.log"
# The author entry point imports Adroit classes even for Hopper. Install the
# upstream dependency rather than changing its import behavior. Keep the
# released D4RL package local so the old shared import-order patch is excluded.
"$PYTHON" -m pip install --no-deps --ignore-installed 'd4rl==1.1' \
    2>&1 | tee "$ENV_DIR/pip-d4rl-$RUN_STAMP.log"
"$PYTHON" -m pip install --no-deps \
    'mjrl @ git+https://github.com/aravindr93/mjrl.git@3871d93763d3b49c4741e6daeaebbc605fe140dc' \
    2>&1 | tee "$ENV_DIR/pip-mjrl-$RUN_STAMP.log"
"$PYTHON" - <<'PY' 2>&1 | tee "$ENV_DIR/import-verification-$RUN_STAMP.log"
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

import pandas
import pytorch_metric_learning
import ray
import torch
import transformers

versions = {name: metadata.version(name) for name in (
    "torch", "numpy", "gym", "mujoco-py", "Cython", "transformers", "ray",
    "pytorch-metric-learning", "pandas", "scipy", "scikit-learn", "protobuf", "grpcio",
    "h5py", "wandb", "tensorboardX", "d4rl", "mjrl",
)}
versions["python"] = platform.python_version()
Path(sys.prefix, "verified-versions.json").write_text(json.dumps(versions, indent=2) + "\n")
author_versions = {
    "python": "3.8.5", "torch": "1.10.2", "gym": "0.18.3", "numpy": "1.20.3",
    "transformers": "4.5.1", "wandb": "0.9.1", "h5py": "2.10.0",
}
differences = {name: {"author": original, "runtime": versions[name]}
               for name, original in author_versions.items() if versions[name] != original}
Path(sys.prefix, "environment-differences.json").write_text(json.dumps(differences, indent=2) + "\n")
dependencies = {
    "d4rl": {"version": versions["d4rl"], "source": "PyPI release", "modified": False},
    "mjrl": {
        "version": versions["mjrl"],
        "source": "https://github.com/aravindr93/mjrl",
        "commit": "3871d93763d3b49c4741e6daeaebbc605fe140dc",
        "reason": "Required by author entry point's unconditional Adroit imports",
    },
}
Path(sys.prefix, "dependency-provenance.json").write_text(json.dumps(dependencies, indent=2) + "\n")
print(json.dumps(versions, indent=2))
print("Core imports verified; this check does not import Gym or MuJoCo.")
PY
{
    echo '# Isolated ConDT author-code runtime; inherited base packages are included.'
    echo '# Python 3.9.23, Torch 1.11.0+cu113, and Gym 0.23.0 replace the author runtime.'
    echo '# Existing MuJoCo 2.1 / mujoco-py 2.1.2.14 are inherited without importing or rebuilding.'
    echo '# D4RL 1.1 is an unmodified PyPI release installed locally to exclude the base import-order patch.'
    echo '# MJRL source: https://github.com/aravindr93/mjrl @ 3871d93763d3b49c4741e6daeaebbc605fe140dc.'
    echo '# This is a compatibility environment, not the original unmodified dependency lock.'
    "$PYTHON" - <<'PY'
import json
import sys
from pathlib import Path
for name, change in json.loads(Path(sys.prefix, "environment-differences.json").read_text()).items():
    print(f"# {name}: author={change['author']}; runtime={change['runtime']}")
PY
    "$PYTHON" -m pip freeze
} > "$ENV_DIR/installed-packages.txt"
