"""Retry the zero-update W&B initialization failure without changing DT behavior."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.auctionnet_dt.prepare_run import prepare
from scripts.auctionnet_dt_delayed.prepare_run import configure_delayed
from scripts.auctionnet_dt_delayed_reproduction.pipeline import verify_delayed
from scripts.auctionnet_dt_delayed_reproduction.summarize import summarize
from scripts.auctionnet_dt_nonfinal.pipeline import write

def initialization_failure(root):
    path = root / "status.json"
    if not path.exists():
        return False
    status = json.loads(path.read_text())
    error = status.get("error", "").lower()
    return (status.get("status") == "failed" and status.get("completed_updates") == 0
            and "initializing run" in error and "wandb" in error
            and not (root / "console.log").exists()
            and not any((root / "upstream/model").rglob("*.pkl")))


def execute(project):
    stem = "prgs-dt-auctionnet-nonfinal-delayed"
    baseline = project / f"results/{stem}-5090-20261006"
    second = project / f"results/{stem}-repeat2-5090-20261006"
    failed = project / f"results/{stem}-repeat3-5090-20261006"
    queue = project / ".runtime/auctionnet-dt-delayed-repeat3-recovery-20261006"
    queue.mkdir(parents=True, exist_ok=True)
    with (queue / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (queue / "queue_status.json").exists():
            raise FileExistsError("Preserve prior recovery attempt")

        def update(stage, **extra):
            write(queue / "queue_status.json", {
                "stage": stage, "pid": os.getpid(), "updated_unix": time.time(),
                "failed_attempt": str(failed), **extra})

        try:
            prior = verify_delayed(baseline)
            verify_delayed(second)
            if not initialization_failure(failed):
                raise ValueError("Require a zero-training W&B initialization failure")
            ordinary = project / ".runtime/auctionnet-dt-three-run-20261006"
            if json.loads((ordinary / "queue_status.json").read_text())["stage"] != (
                    "completed"):
                raise RuntimeError("Ordinary campaign is not completed")
            data = project / (
                ".runtime/auctionnet-dt-nonfinal-delayed-queue-20261006/prepared-data")
            shared = project / ".runtime/auctionnet-dt-nonfinal-data-20261006"
            upstream = project / ".runtime/auctionnet-upstream-20261005/prgs/AuctionNet"
            environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                               PYTHONUNBUFFERED="1")
            attempts = []
            for mode, suffix in (("online", "retry1"), ("offline", "offline1")):
                output = project / f"results/{stem}-repeat3-{suffix}-5090-20261006"
                update("preparing", output=str(output), wandb_mode=mode)
                prepare(upstream, data,
                        [shared / f"raw/period-{p}.csv" for p in range(14, 21)],
                        output, False)
                configure_delayed(output)
                path = output / "protocol.json"
                protocol = json.loads(path.read_text())
                for key in ("config", "commit", "runtime_sha256", "data_audit",
                            "reward_protocol"):
                    if protocol[key] != prior[key]:
                        raise ValueError(f"Recovery differs from baseline: {key}")
                protocol["repetition"] = 3
                protocol["recovery"] = {
                    "original_failed_attempt": str(failed), "wandb_mode": mode,
                    "earlier_attempts": attempts.copy(), "seed": None,
                    "reason": "W&B init timeout before model creation/training",
                    "selection": "No score selection; retain final100k/target1.0",
                }
                write(path, protocol)
                update("running", output=str(output), wandb_mode=mode)
                result = subprocess.run([
                    sys.executable, "-m", "scripts.auctionnet_dt_delayed.run",
                    "--root", str(output), "--wandb-mode", mode],
                    cwd=project, env=environment, check=False)
                if result.returncode:
                    if mode == "online" and initialization_failure(output):
                        attempts.append(str(output))
                        continue
                    raise RuntimeError(f"Training failed; preserve {output}")
                verify_delayed(output)
                write(queue / "summary.json", summarize([baseline, second, output]))
                sync = "online"
                if mode == "offline":
                    sync = "pending"
                    update("training_completed_syncing", output=str(output))
                    offline = list((output / "wandb").glob("offline-run-*"))
                    if len(offline) != 1:
                        raise ValueError("Expected one offline W&B run")
                    try:
                        result = subprocess.run([
                            str(Path(sys.executable).parent / "wandb"), "sync",
                            str(offline[0])], cwd=project, env=environment,
                            timeout=180, check=False)
                        if result.returncode == 0:
                            sync = "synced"
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                    write(queue / "wandb_sync.json", {
                        "status": sync, "offline_run": str(offline[0])})
                update("completed", output=str(output), wandb_sync=sync,
                       summary=str(queue / "summary.json"))
                break
        except BaseException as error:
            update("failed", error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path,
                        default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    execute(args.project.resolve())


if __name__ == "__main__":
    main()
