"""Validate all three seed-1 CQL arms, then train them on the RTX 5090."""

import argparse
import fcntl
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.cql_delayed.common import read, verify, write

def require_preflight(work):
    status = read(work / "status.json")
    runtime = read(work / "runtime.json")
    if (status["status"] != "completed" or status["completed_updates"] != 100
            or not math.isfinite(status["final_normalized_score"])
            or not runtime["device"].startswith("cuda")):
        raise ValueError("Require 100 CUDA updates and a complete finite evaluation")


def execute(project, queue):
    delayed = project / "results/cql-corl-delayed-hc-seed1-5090-20261007"
    shapley = project / "results/cql-shapley-hcmr-v1-seed1-5090-20261007"
    arms = [(delayed, "medium", "cql_delayed"),
            (delayed, "medium_replay", "cql_delayed"),
            (shapley, "shapley", "cql_shapley")]
    queue.mkdir(parents=True, exist_ok=True)
    with (queue / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (queue / "status.json").exists():
            raise FileExistsError("Preserve prior launch; create a new recovery attempt")
        processes, observers, logs = [], [], []

        def status(stage, **extra):
            write(queue / "status.json", {"stage": stage, "pid": os.getpid(),
                                          "updated_unix": time.time(), **extra})

        try:
            for root, arm, _ in arms:
                p = verify(root)
                if p["runs"][arm]["seed"] != 1:
                    raise ValueError("Unexpected policy/evaluation seed")
                if (root / arm / "training").exists():
                    raise FileExistsError("Existing training results must be preserved")
            for root, arm, runner in arms:
                status("gpu_preflight", arm=arm)
                env = dict(os.environ, PYTHONPATH=str(root / "source"),
                           WANDB_MODE="disabled")
                log = (queue / f"preflight-{arm}.log").open("x")
                logs.append(log)
                subprocess.run([
                    sys.executable, str(root / "source/scripts" / runner / "train.py"),
                    "--root", str(root / "validation"), "--arm", arm, "--smoke"],
                    cwd=root / "source", env=env, stdout=log,
                    stderr=subprocess.STDOUT, check=True)
                require_preflight(root / "validation" / arm / "preflight")
            status("launching")
            code = Path(__file__).resolve().parents[2]
            for root, arm, runner in arms:
                log = (queue / f"training-{arm}.log").open("x")
                logs.append(log)
                env = dict(os.environ, PYTHONPATH=str(root / "source"),
                           WANDB_MODE="disabled")
                process = subprocess.Popen([
                    sys.executable, str(root / "source/scripts" / runner / "train.py"),
                    "--root", str(root), "--arm", arm], cwd=root / "source",
                    env=env, stdout=log, stderr=subprocess.STDOUT)
                processes.append((arm, process))
                observer_log = (queue / f"wandb-{arm}.log").open("x")
                logs.append(observer_log)
                observer = subprocess.Popen([
                    sys.executable, "-m", "scripts.cql_5090.observe",
                    "--root", str(root), "--arm", arm], cwd=code,
                    env=dict(os.environ, PYTHONPATH=str(code), WANDB_MODE="online"),
                    stdout=observer_log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                observers.append((arm, observer))
            while any(process.poll() is None for _, process in processes):
                status("running", processes={arm: {"pid": p.pid, "exit": p.poll()}
                                             for arm, p in processes},
                       observers={arm: {"pid": p.pid, "exit": p.poll()}
                                  for arm, p in observers})
                time.sleep(30)
            if any(process.returncode for _, process in processes):
                raise RuntimeError("A CQL training arm failed; inspect per-arm logs")
            for root, arm, _ in arms:
                verify(root)
                final = read(root / arm / "training/status.json")
                if (final["status"] != "completed"
                        or final["completed_updates"] != 1000000):
                    raise RuntimeError(f"Incomplete training: {arm}")
            status("training_completed", wandb="Independent bridges finish/read back")
        except BaseException as error:
            status("failed", error=repr(error))
            raise
        finally:
            for stream in logs:
                stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    execute(project, project / ".runtime/cql-5090-seed1-20261007")
