"""Prepare rewards on CPU, validate unchanged official training, then queue GPUs."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.cql_antmaze_official_5090.common import read, write
from scripts.cql_maze2d_four_5090.materialize import materialize
from scripts.cql_maze2d_four_5090.online_check import check, validate_runtime
from scripts.cql_maze2d_four_5090.queue import environment, execute

def run(root):
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "pipeline_status.json").exists():
            raise FileExistsError("Inspect and preserve the previous pipeline first")
        try:
            plan = read(root / "plan.json")
            write(root / "pipeline_status.json", dict(stage="validating_runtime",
                                                      pid=os.getpid()))
            validate_runtime(root, plan)
            materialize(root, plan)
            write(root / "pipeline_status.json", dict(stage="auditing_official_loader"))
            for job in plan["jobs"]:
                env = environment(root, plan, root / job["id"] / "audit", "audit")
                subprocess.run([sys.executable, "-m",
                                "scripts.cql_maze2d_four_5090.audit",
                                "--root", str(root), "--job", job["id"]],
                               env=env, check=True, timeout=900)
            write(root / "pipeline_status.json", dict(stage="online_cpu_preflight"))
            check(root)
            write(root / "pipeline_status.json", dict(stage="training_queue"))
            execute(root)
            write(root / "pipeline_status.json", dict(stage="queue_finished",
                                                      finished_unix=time.time()))
        except BaseException as error:
            write(root / "pipeline_status.json", dict(stage="failed", error=repr(error),
                                                      failed_unix=time.time()))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve())
