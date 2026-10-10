"""Preserve pre-training failures and retry with CORL's pinned W&B dependency."""

import argparse
import fcntl
import shutil
from pathlib import Path

from scripts.cql_antmaze_official_5090.common import digest, read, verify, write
from scripts.cql_antmaze_official_5090.prepare import prepare

def require_unstarted_failure(previous):
    plan = read(previous / "plan.json")
    state = read(previous / "queue/status.json")
    if state["stage"] != "finished_with_failures":
        raise ValueError("Previous manager must have finished with failures")
    for job in plan["jobs"]:
        status = state["jobs"][job["id"]]
        work = previous / job["id"] / "training"
        log = (work / "console.log").read_text(errors="replace")
        if (status["state"] != "failed" or status.get("completed_updates", 0) != 0
                or list(work.glob("checkpoints/*/checkpoint_*.pt"))
                or "Run.save() missing 1 required positional argument" not in log
                or "Time steps: " in log):
            raise ValueError("Retry is restricted to the known pre-training W&B failure")
    verify(previous, plan)
    return plan


def retry(previous, root, project, d4rl_source, revision, runtime):
    with (previous / "queue/lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = require_unstarted_failure(previous)
        prepare(root, project, Path(old["predecessor"]), d4rl_source, 21, revision)
        code = Path(__file__).resolve().parents[2]
        target = root / "source/scripts/cql_antmaze_official_retry_5090"
        shutil.copytree(code / "scripts/cql_antmaze_official_retry_5090", target,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        plan = read(root / "plan.json")
        plan["retry_of"] = dict(root=str(previous), plan_sha256=digest(
            previous / "plan.json"), failure="W&B 0.30.0 requires Run.save(glob_str)")
        plan["runtime_python"] = str(runtime)
        plan["required_wandb_version"] = "0.12.21"
        plan["changes"].append(
            "Restore CORL's W&B dependency in an isolated environment")
        for path in target.rglob("*"):
            if path.is_file():
                plan["frozen_sha256"][str(path.relative_to(root))] = digest(path)
        write(root / "plan.json", plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["previous", "root", "project", "d4rl-source", "runtime"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    retry(args.previous.resolve(), args.root.resolve(), args.project.resolve(),
          args.d4rl_source.resolve(), args.revision, args.runtime.absolute())
