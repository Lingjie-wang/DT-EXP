#!/usr/bin/env bash
# Copy the numerical stack, then change logger dependencies only in the copy.
set -euo pipefail
BASE_PYTHON=${1:?Pass the shared numerical-stack Python}
RUNTIME=${2:?Pass a new isolated environment directory}
WHEELS=${3:?Pass a wheelhouse with all requirements (sdists allowed)}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -e "$RUNTIME" ]]; then
    echo "Refusing to overwrite an existing environment: $RUNTIME" >&2
    exit 1
fi
"$BASE_PYTHON" -m venv "$RUNTIME"
"$RUNTIME/bin/python" - "$BASE_PYTHON" <<'PY'
import pathlib
import subprocess
import sys
import sysconfig
shared = subprocess.check_output(
    [sys.argv[1], "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
    text=True).strip()
destination = pathlib.Path(sysconfig.get_path("purelib"))
# Real files keep dependency import-time sys.path edits inside the new runtime.
# Reflinks are copy-on-write when supported; never use hard links or symlinks.
subprocess.run(["cp", "-a", "--reflink=auto", shared + "/.", str(destination)],
               check=True)
PY
"$RUNTIME/bin/python" -m pip uninstall -y wandb protobuf GitPython gitdb smmap \
    psutil setproctitle sentry-sdk docker-pycreds shortuuid promise pathtools
"$RUNTIME/bin/python" -m pip install --ignore-installed --no-deps --no-index \
    --no-build-isolation --find-links "$WHEELS" -r "$SCRIPT_DIR/requirements.txt"
"$RUNTIME/bin/python" - <<'PY'
import inspect
import d4rl
import numpy
import torch
import wandb
assert wandb.__version__ == "0.12.21"
assert numpy.__version__ == "1.23.5"
assert torch.__version__ == "2.7.1+cu128"
assert str(__import__('pathlib').Path(wandb.__file__).resolve()).startswith(
    str(__import__('pathlib').Path(__import__('sys').prefix).resolve()) + "/")
assert inspect.signature(wandb.sdk.wandb_run.Run.save).parameters[
    "glob_str"].default is None
print("Runtime verified:", wandb.__version__, numpy.__version__, torch.__version__)
PY
