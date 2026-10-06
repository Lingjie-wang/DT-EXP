"""Run two more delayed DT trials only after both ordinary repeats finish."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.auctionnet_dt.prepare_run import digest, prepare
from scripts.auctionnet_dt_delayed.prepare_run import configure_delayed
from scripts.auctionnet_dt_delayed_reproduction.summarize import (
    complete,
    summarize,
    validate_protocol,
)
from scripts.auctionnet_dt_nonfinal.pipeline import predecessor_complete, write
from scripts.auctionnet_dt_reproduction.pipeline import verify_run as verify_ordinary

def dependency_ready(campaign, statuses):
    if campaign is None:
        return False
    if campaign["stage"] == "failed":
        raise RuntimeError("Ordinary DT campaign failed")
    ready = [False if status is None else predecessor_complete(status)
             for status in statuses]
    if campaign["stage"] != "completed":
        return False
    if len(ready) != 3 or not all(ready):
        raise RuntimeError("Ordinary campaign completion lacks three verified runs")
    return True


def verify_delayed(root):
    if not complete(json.loads((root / "status.json").read_text())):
        raise RuntimeError("Delayed baseline has not completed")
    protocol = json.loads((root / "protocol.json").read_text())
    validate_protocol(protocol)
    for name, sha in protocol["runtime_sha256"].items():
        if digest(root / "upstream" / name) != sha:
            raise ValueError(f"Frozen delayed source changed: {name}")
    return protocol


def execute(project, poll_seconds):
    baseline = project / "results/prgs-dt-auctionnet-nonfinal-delayed-5090-20261006"
    outputs = [project / (
        f"results/prgs-dt-auctionnet-nonfinal-delayed-repeat{i}-5090-20261006")
        for i in (2, 3)]
    ordinary = [project / "results/prgs-dt-auctionnet-nonfinal-5090-20261006"] + [
        project / f"results/prgs-dt-auctionnet-nonfinal-repeat{i}-5090-20261006"
        for i in (2, 3)]
    predecessor = project / ".runtime/auctionnet-dt-three-run-20261006"
    queue = project / ".runtime/auctionnet-dt-delayed-three-run-20261006"
    queue.mkdir(parents=True, exist_ok=True)
    with (queue / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(p.exists() for p in outputs) or (queue / "summary.json").exists():
            raise FileExistsError("Preserve prior delayed repetition attempts")

        def update(stage, **extra):
            write(queue / "queue_status.json", {
                "stage": stage, "pid": os.getpid(), "updated_unix": time.time(),
                "predecessor": str(predecessor),
                "runs": [str(p) for p in [baseline, *outputs]], **extra})

        def read_optional(path):
            return json.loads(path.read_text()) if path.exists() else None

        try:
            prior = verify_delayed(baseline)
            while not dependency_ready(
                    read_optional(predecessor / "queue_status.json"),
                    [read_optional(p / "status.json") for p in ordinary]):
                update("waiting_for_both_ordinary_repeats")
                time.sleep(poll_seconds)
            for run in ordinary:
                verify_ordinary(run)
            data = project / (
                ".runtime/auctionnet-dt-nonfinal-delayed-queue-20261006/prepared-data")
            if json.loads((data / "audit.json").read_text()) != prior["data_audit"]:
                raise ValueError("Delayed baseline data provenance changed")
            shared = project / ".runtime/auctionnet-dt-nonfinal-data-20261006"
            upstream = project / ".runtime/auctionnet-upstream-20261005/prgs/AuctionNet"
            environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                               PYTHONUNBUFFERED="1")
            for repetition, output in enumerate(outputs, 2):
                update("preparing", repetition=repetition)
                prepare(upstream, data,
                        [shared / f"raw/period-{p}.csv" for p in range(14, 21)],
                        output, False)
                configure_delayed(output)
                path = output / "protocol.json"
                protocol = json.loads(path.read_text())
                for key in ("config", "commit", "runtime_sha256", "data_audit",
                            "reward_protocol"):
                    if protocol[key] != prior[key]:
                        raise ValueError(f"Delayed repeat differs from baseline: {key}")
                protocol["repetition"] = repetition
                protocol["campaign"] = {
                    "baseline": str(baseline), "planned_runs": 3,
                    "ordinary_predecessor": str(predecessor), "seed": None,
                    "seed_policy": "Independent fresh process, no fixed seed",
                    "data": "Reuse original delayed baseline's immutable snapshot",
                    "selection": "Fixed final 100k/target1.0; retain all targets",
                }
                write(path, protocol)
                update("running", repetition=repetition, output=str(output))
                subprocess.run([
                    sys.executable, "-m", "scripts.auctionnet_dt_delayed.run",
                    "--root", str(output), "--wandb-mode", "online"],
                    cwd=project, env=environment, check=True)
                verify_delayed(output)
            update("summarizing")
            write(queue / "summary.json", summarize([baseline, *outputs]))
            update("completed", summary=str(queue / "summary.json"))
        except BaseException as error:
            update("failed", error=repr(error))
            raise


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
