"""Fit medium rewards, require a prediction gate and CUDA preflight, then train."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.cql_5090.run import require_preflight
from scripts.cql_delayed.common import read, verify, write

def validation(root):
    p = verify(root)
    if not p.get("shapley_gate", {}).get("passed"):
        raise ValueError("Medium Shapley prediction gate failed; do not train CQL")
    target = root / "validation"
    (target / "shapley").mkdir(parents=True, exist_ok=False)
    (target / "source").symlink_to("../source", target_is_directory=True)
    for name in p["runs"]["shapley"]["data_sha256"]:
        (target / "shapley" / name).symlink_to(f"../../shapley/{name}")
    write(target / "protocol.json", p)
    return target


def execute(root):
    queue = root / "launch"
    queue.mkdir(exist_ok=True)
    with (queue / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (queue / "status.json").exists():
            raise FileExistsError("Preserve the previous launch attempt")

        def status(stage, **extra):
            write(queue / "status.json", dict(stage=stage, pid=os.getpid(),
                                               updated_unix=time.time(), **extra))

        env = dict(os.environ, PYTHONPATH=str(root / "source"), WANDB_MODE="disabled")

        def command(script, arguments, log):
            with (queue / log).open("x") as stream:
                subprocess.run([sys.executable, str(root / "source/scripts/cql_shapley"
                                                   / script), *arguments],
                               cwd=root / "source", env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)

        try:
            p = verify(root)
            spec = p["runs"]["shapley"]
            if spec["env"] != "halfcheetah-medium-v2" or spec["seed"] != 1:
                raise ValueError("Wrong campaign: require medium seed 1")
            status("fitting_medium_predictor")
            command("fit_redistribute.py", ["--root", str(root), "--device", "cpu"],
                    "fit.log")
            target = validation(root)
            status("gpu_preflight")
            command("train.py", ["--root", str(target), "--arm", "shapley", "--smoke"],
                    "preflight.log")
            require_preflight(target / "shapley/preflight")
            code = Path(__file__).resolve().parents[2]
            with (queue / "wandb.log").open("x") as stream:
                observer = subprocess.Popen([
                    sys.executable, "-m", "scripts.cql_5090.observe",
                    "--root", str(root), "--arm", "shapley"], cwd=code,
                    env=dict(os.environ, PYTHONPATH=str(code), WANDB_MODE="online"),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            status("training", observer_pid=observer.pid)
            command("train.py", ["--root", str(root), "--arm", "shapley"], "train.log")
            verify(root)
            final = read(root / "shapley/training/status.json")
            if final["status"] != "completed" or final["completed_updates"] != 1000000:
                raise ValueError("Incomplete CQL training")
            status("training_completed", wandb="Independent observer finishes upload")
        except BaseException as error:
            status("failed", error=repr(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    execute(parser.parse_args().root.resolve())
