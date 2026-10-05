"""Wait for ordinary non-final DT, then run its terminal-reward comparison."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.auctionnet_dt_nonfinal.pipeline import predecessor_complete, write

def dependency_ready(previous, queue):
    queue_path = queue / "queue_status.json"
    if queue_path.exists():
        queue_state = json.loads(queue_path.read_text())
        if queue_state["stage"] == "failed":
            raise RuntimeError(f"Ordinary-DT queue failed: {queue_state}")
    status_path = previous / "status.json"
    if not status_path.exists():
        return False
    return predecessor_complete(json.loads(status_path.read_text()))


def snapshot_data(source, target, protocol):
    if protocol.get("dataset_version") != "non-final/general":
        raise ValueError("Predecessor must use non-final/general data")
    config = protocol["config"]
    if (config["delayed_reward"] or config["is_stitch"]
            or config["model_type"] != "dt"):
        raise ValueError("Predecessor must be ordinary DT with step rewards")
    audit = json.loads((source / "audit.json").read_text())
    if audit["sha256"] != protocol["data_audit"]["sha256"]:
        raise ValueError("Predecessor data audit changed")
    for name, expected in audit["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Predecessor data changed: {name}")
    shutil.copytree(source, target)


def execute(project, poll_seconds):
    shared = project / ".runtime/auctionnet-dt-nonfinal-data-20261006"
    queue = project / ".runtime/auctionnet-dt-nonfinal-delayed-queue-20261006"
    previous = project / "results/prgs-dt-auctionnet-nonfinal-5090-20261006"
    output = project / "results/prgs-dt-auctionnet-nonfinal-delayed-5090-20261006"
    snapshot = queue / "prepared-data"
    upstream = project / ".runtime/auctionnet-upstream-20261005/prgs/AuctionNet"
    queue.mkdir(parents=True, exist_ok=True)
    lock = (queue / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if output.exists() or snapshot.exists():
        raise FileExistsError("Preserve previous delayed attempt/snapshot")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")

    def update(stage, **extra):
        write(queue / "queue_status.json", {
            "stage": stage, "pid": os.getpid(), "predecessor": str(previous),
            "output": str(output), "updated_unix": time.time(), **extra})

    def command(module, *arguments):
        subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                       cwd=project, env=environment, check=True)

    try:
        while not dependency_ready(previous, shared):
            update("waiting_for_ordinary_nonfinal_dt")
            time.sleep(poll_seconds)
        update("preparing_delayed_run")
        predecessor = json.loads((previous / "protocol.json").read_text())
        snapshot_data(shared / "prepared-full", snapshot, predecessor)
        command("scripts.auctionnet_dt_delayed.prepare_run", "--upstream", upstream,
                "--data", snapshot, "--csvs",
                *[shared / f"raw/period-{p}.csv" for p in range(14, 21)],
                "--root", output)
        protocol_path = output / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["predecessor"] = {
            "root": str(previous),
            "completion": json.loads((previous / "status.json").read_text()),
            "protocol_sha256": hashlib.sha256((previous / "protocol.json").read_bytes())
            .hexdigest(),
        }
        write(protocol_path, protocol)
        update("running")
        command("scripts.auctionnet_dt_delayed.run", "--root", output,
                "--wandb-mode", "online")
        update("completed", run_status=json.loads((output / "status.json").read_text()))
    except BaseException as error:
        update("failed", error=repr(error))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    execute(args.project.resolve(), args.poll_seconds)


if __name__ == "__main__":
    main()
