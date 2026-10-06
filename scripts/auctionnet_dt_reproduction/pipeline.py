"""Add two independent official DT runs to the preserved non-final baseline."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.auctionnet_dt.prepare_run import digest, prepare
from scripts.auctionnet_dt_nonfinal.pipeline import predecessor_complete, write
from scripts.auctionnet_dt_reproduction.summarize import summarize, validate_protocol

def verify_run(root):
    status = json.loads((root / "status.json").read_text())
    if not predecessor_complete(status):
        raise RuntimeError("Baseline must have fully completed")
    protocol = json.loads((root / "protocol.json").read_text())
    validate_protocol(protocol)
    for name, sha in protocol["runtime_sha256"].items():
        if digest(root / "upstream" / name) != sha:
            raise ValueError(f"Frozen baseline changed: {name}")
    return protocol


def execute(project):
    baseline = project / "results/prgs-dt-auctionnet-nonfinal-5090-20261006"
    outputs = [project / f"results/prgs-dt-auctionnet-nonfinal-repeat{i}-5090-20261006"
               for i in (2, 3)]
    queue = project / ".runtime/auctionnet-dt-three-run-20261006"
    queue.mkdir(parents=True, exist_ok=True)
    with (queue / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(p.exists() for p in outputs) or (queue / "summary.json").exists():
            raise FileExistsError("Preserve previous repeat attempts")

        def update(stage, **extra):
            write(queue / "queue_status.json", {
                "stage": stage, "pid": os.getpid(), "updated_unix": time.time(),
                "runs": [str(p) for p in [baseline, *outputs]], **extra})

        try:
            prior = verify_run(baseline)
            shared = project / ".runtime/auctionnet-dt-nonfinal-data-20261006"
            data = shared / "prepared-full"
            if json.loads((data / "audit.json").read_text()) != prior["data_audit"]:
                raise ValueError("Baseline data provenance changed")
            # Finish the earlier requested delayed experiment before this new campaign.
            delayed = project / ".runtime/auctionnet-dt-nonfinal-delayed-queue-20261006"
            delayed_state = json.loads((delayed / "queue_status.json").read_text())
            if delayed_state["stage"] != "completed":
                raise RuntimeError("Earlier delayed experiment is not completed")
            upstream = project / ".runtime/auctionnet-upstream-20261005/prgs/AuctionNet"
            environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                               PYTHONUNBUFFERED="1")
            for index, output in enumerate(outputs, 2):
                update("preparing", repetition=index)
                prepare(upstream, data,
                        [shared / f"raw/period-{p}.csv" for p in range(14, 21)],
                        output, False)
                path = output / "protocol.json"
                protocol = json.loads(path.read_text())
                protocol["dataset_version"] = "non-final/general"
                protocol["changes"]["data"] = prior["changes"]["data"]
                protocol["repetition"] = index
                protocol["campaign"] = {
                    "baseline": str(baseline), "planned_runs": 3,
                    "source": "unchanged official PRGS AuctionNet DT",
                    "seed": None, "seed_policy": "Preserve official unseeded CLI",
                    "selection": "Fixed final 100k/target1.0; retain all targets",
                }
                for key in ("config", "runtime_sha256", "data_audit"):
                    if protocol[key] != prior[key]:
                        raise ValueError(f"Repeat differs from baseline: {key}")
                write(path, protocol)
                update("running", repetition=index, output=str(output))
                subprocess.run([
                    sys.executable, str(project / "scripts/auctionnet_dt/run.py"),
                    "--root", str(output), "--wandb-mode", "online"],
                    cwd=project, env=environment, check=True)
                verify_run(output)
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
    args = parser.parse_args()
    execute(args.project.resolve())


if __name__ == "__main__":
    main()
