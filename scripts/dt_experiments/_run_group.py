"""Shared adapter for grouped DT experiment entry points."""

import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_dt_experiments.py"


def run_group(env_names: Sequence[str]) -> int:
    command = [
        sys.executable,
        str(RUNNER),
        "--datasets",
        *env_names,
        *sys.argv[1:],
    ]
    return subprocess.call(command, cwd=ROOT)
