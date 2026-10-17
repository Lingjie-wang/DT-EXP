"""Publish aggregate Slurm state while retaining the existing W&B comparison."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.cql_antmaze_official_5090.common import read, write

TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def aggregate(root, plan, accounting):
    states = {}
    for index, job in enumerate(plan["jobs"]):
        path = root / job["id"] / "slurm_status.json"
        state = read(path) if path.exists() else dict(state="queued")
        scheduler_state = accounting.get(index, "UNKNOWN")
        if state["state"] not in {"completed", "failed"}:
            if scheduler_state in TERMINAL:
                state.update(state="failed", error="Slurm ended: " + scheduler_state)
        states[job["id"]] = dict(state, scheduler_state=scheduler_state)
    terminal = all(s["state"] in {"completed", "failed"} for s in states.values())
    stage = ("completed" if all(s["state"] == "completed" for s in states.values())
             else "finished_with_failures") if terminal else "slurm_queue"
    return dict(stage=stage, jobs=states, updated_unix=time.time())


def parse_accounting(report, jobs, submitted):
    indices = {job["id"]: index for index, job in enumerate(jobs)}
    if (len(submitted) != len(indices)
            or set(indices) != {item["id"] for item in submitted}):
        raise ValueError("Each comparison arm needs exactly one submitted task")
    tasks = {item["slurm_id"]: indices[item["id"]] for item in submitted}
    accounting = {}
    for line in report.splitlines():
        identity, state = line.split("|")[:2]
        if identity in tasks:
            accounting[tasks[identity]] = state.split()[0]
    return accounting


def observe(root):
    plan = read(root / "plan.json")
    submission = read(root / "slurm/submission.json")
    arrays = sorted({j["slurm_id"].split("_")[0] for j in submission["jobs"]})
    with (root / "slurm/observer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / "comparison/console.log").open("a") as stream:
            comparison = subprocess.Popen([sys.executable, "-m",
                "scripts.cql_maze2d_four_5090.compare", "--root", str(root)],
                stdout=stream, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    report = subprocess.check_output(["sacct", "--array", "-n", "-P", "-j",
                        ",".join(arrays), "--format=JobID,State"],
                        text=True, timeout=30)
                    accounting = parse_accounting(report, plan["jobs"],
                                                  submission["jobs"])
                    snapshot = aggregate(root, plan, accounting)
                    write(root / "queue/status.json", snapshot)
                    write(root / "slurm/observer_status.json", dict(
                        state="observing", pid=os.getpid(), updated_unix=time.time()))
                    if snapshot["stage"] in {"completed", "finished_with_failures"}:
                        comparison.wait(timeout=300)
                        return
                    if comparison.poll() is not None:
                        raise RuntimeError("Comparison uploader exited unexpectedly")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    write(root / "slurm/observer_status.json", dict(
                        state="retrying", error=repr(error), updated_unix=time.time()))
                time.sleep(30)
        finally:
            if comparison.poll() is None:
                comparison.terminate()
                comparison.wait(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    observe(parser.parse_args().root.resolve())
