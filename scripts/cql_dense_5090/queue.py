"""Give the existing repeat queue priority, then fill free GPU training slots."""

import argparse
import fcntl
import os
import time
from pathlib import Path

from scripts.cql_delayed.common import digest, read, write
from scripts.cql_repeats_5090.queue import execute, resources, validate_job

TERMINAL = {"completed", "failed", "preflight_failed"}


def admitted(status, expected_jobs, gpu_pids, now):
    jobs = status.get("jobs", {})
    if set(jobs) != set(expected_jobs):
        return False
    if status.get("stage") in {"training_completed", "finished_with_failures"}:
        return all(j["state"] in TERMINAL for j in jobs.values())
    if (status.get("stage") != "running"
            or now - status.get("updated_unix", 0) > 120):
        return False
    if not all(j["state"] in TERMINAL | {"running"} for j in jobs.values()):
        return False
    # The predecessor will never launch another job once all are admitted.
    # Wait for its last CUDA initializations to appear in the GPU process count.
    running = {j["pid"] for j in jobs.values() if j["state"] == "running"}
    return running.issubset(set(gpu_pids))


def wait_and_run(root):
    plan = read(root / "plan.json")
    dependency = root / "dependency"
    dependency.mkdir(exist_ok=True)
    with (dependency / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dependency / "status.json").exists():
            raise FileExistsError("Preserve prior launch; inspect before recovery")

        def record(stage, **extra):
            write(dependency / "status.json", dict(
                stage=stage, pid=os.getpid(), updated_unix=time.time(), **extra))

        for job in plan["jobs"]:
            validate_job(root, job)
        predecessor = Path(plan["predecessor"])
        try:
            while True:
                if digest(predecessor / "plan.json") != plan["predecessor_plan_sha256"]:
                    raise ValueError("Predecessor plan changed; inspect scheduling")
                try:
                    status = read(predecessor / "queue/status.json")
                    resource = resources()
                except Exception as error:
                    record("waiting_for_predecessor_or_resource_probe",
                           error=repr(error))
                    time.sleep(plan["poll_seconds"])
                    continue
                ready = admitted(status, plan["predecessor_jobs"],
                                 resource["gpu_pids"], time.time())
                record("predecessor_admitted" if ready else "waiting_for_predecessor",
                       predecessor_stage=status["stage"], resources=resource,
                       jobs={j["id"]: {"state": "queued"} for j in plan["jobs"]})
                if ready:
                    break
                time.sleep(plan["poll_seconds"])
            execute(root)
            record("queue_finished", queue_status=read(root / "queue/status.json"))
        except BaseException as error:
            record("manager_failed", error=repr(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    wait_and_run(args.root.resolve())
