"""Wait for the existing campaign, then invoke unchanged CORL CLIs on shared GPU."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.cql_antmaze_official_5090.common import (
    admissible,
    completed,
    evaluations,
    predecessor_ready,
    read,
    resources,
    verify,
    write,
)

def command(root, plan, job, phase, device="cuda"):
    work = root / job["id"] / phase
    argv = [sys.executable, "-u", str(root / "upstream/algorithms/offline/cql.py"),
            "--config_path", str(root / "upstream" / job["config"]),
            "--seed", str(job["seed"]), "--device", device,
            "--project", plan["wandb_project"], "--group", root.name,
            "--name", f"CQL-official-{job['id']}-1m",
            "--checkpoints_path", str(work / "checkpoints")]
    if phase != "training":
        argv += ["--max_timesteps", "100", "--eval_freq", "100", "--n_episodes", "1"]
    return argv


def environment(root, plan, work, phase):
    return dict(os.environ, PYTHONPATH=f"{root / 'dependencies/d4rl'}:{root / 'source'}",
                PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                D4RL_DATASET_DIR=str(root / "datasets"), WANDB_DIR=str(work),
                WANDB_ENTITY=plan["wandb_entity"],
                WANDB_MODE="online" if phase == "training" else "disabled")


def create_work(root, plan, job, phase, device):
    verify(root, plan)
    work = root / job["id"] / phase
    work.mkdir()  # Refuse to overwrite any earlier attempt.
    argv = command(root, plan, job, phase, device)
    write(work / "command.json", dict(argv=argv, cwd=str(work),
                                      source_sha256=plan["frozen_sha256"][
                                          "upstream/algorithms/offline/cql.py"]))
    return work, argv


def preflight(root, plan, job, device):
    phase = "cpu_preflight" if device == "cpu" else "gpu_preflight"
    work, argv = create_work(root, plan, job, phase, device)
    with (work / "console.log").open("x") as stream:
        result = subprocess.run(argv, cwd=work,
                                env=environment(root, plan, work, phase),
                                stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f"Official preflight exited {result.returncode}: {work}")
    verified = completed(work, 100, 100, 1)
    write(work / "verification.json", dict(device=device, **verified))


def telemetry(work, plan):
    rows = evaluations((work / "console.log").read_text(errors="replace"))
    result = dict(completed_updates=rows[-1]["completed_updates"] if rows else 0,
                  progress_basis="last completed official evaluation")
    if rows:
        result["latest_normalized_score"] = rows[-1]["normalized_score"]
        write(work / "evaluations.json", rows)
    records = list((work / "wandb").glob("run-*/run-*.wandb"))
    if len(records) == 1:
        identity = records[0].stem.removeprefix("run-")
        result["wandb_url"] = (f"https://wandb.ai/{plan['wandb_entity']}/"
                               f"{plan['wandb_project']}/runs/{identity}")
        write(work / "wandb_manifest.json", dict(id=identity,
                                                url=result["wandb_url"]))
    return result


def execute(root):
    plan = read(root / "plan.json")
    verify(root, plan)
    audit = read(root / "data_audit.json")
    if any(not audit["jobs"][j["id"]]["full_array_comparison_passed"]
           for j in plan["jobs"]):
        raise ValueError("Missing official data audit")
    queue = root / "queue"
    with (queue / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (queue / "status.json").exists():
            raise FileExistsError("Inspect existing jobs before any recovery")
        states = {j["id"]: dict(state="queued") for j in plan["jobs"]}
        processes = {}

        def status(stage, **extra):
            write(queue / "status.json", dict(stage=stage, pid=os.getpid(),
                  updated_unix=time.time(), jobs=states, **extra))

        try:
            while not predecessor_ready(plan):
                status("waiting_for_predecessor", predecessor=plan["predecessor"])
                time.sleep(plan["poll_seconds"])
            while True:
                for job in plan["jobs"]:
                    key = job["id"]
                    if states[key]["state"] != "running":
                        continue
                    work = root / key / "training"
                    states[key].update(telemetry(work, plan))
                    result = processes[key].poll()
                    if result is None:
                        continue
                    states[key]["exit"] = result
                    try:
                        if result:
                            raise RuntimeError(f"Official CLI exited {result}")
                        summary = completed(work, job["updates"], job["eval_every"],
                                            job["eval_episodes"])
                        verify(root, plan)
                        write(work / "completion.json", summary)
                        states[key].update(state="completed", completed_updates=summary[
                            "completed_updates"], final_normalized_score=summary[
                                "final_normalized_score"])
                    except Exception as error:
                        states[key].update(state="failed", error=repr(error))
                own = [process.pid for process in processes.values()
                       if process.poll() is None]
                pending = [j for j in plan["jobs"]
                           if states[j["id"]]["state"] in {"queued", "ready"}]
                if not own and not pending:
                    status("completed" if all(s["state"] == "completed"
                                              for s in states.values())
                           else "finished_with_failures")
                    return
                try:
                    resource = resources()
                except Exception as error:
                    status("waiting_for_resource_probe", error=repr(error))
                    time.sleep(plan["poll_seconds"])
                    continue
                status("running" if own else "waiting_for_slots", resources=resource)
                if pending and admissible(plan, resource, own):
                    job = pending[0]
                    key = job["id"]
                    if states[key]["state"] == "queued":
                        states[key]["state"] = "gpu_preflight"
                        status("gpu_preflight", resources=resource)
                        try:
                            preflight(root, plan, job, "cuda")
                            states[key]["state"] = "ready"
                        except Exception as error:
                            states[key].update(state="preflight_failed",
                                               error=repr(error))
                            status("preflight_failed")
                            continue
                    # Other users can consume resources during preflight.
                    try:
                        resource = resources()
                    except Exception as error:
                        status("waiting_for_resource_probe", error=repr(error))
                        time.sleep(plan["poll_seconds"])
                        continue
                    if admissible(plan, resource, own):
                        work, argv = create_work(root, plan, job, "training", "cuda")
                        with (work / "console.log").open("x") as stream:
                            process = subprocess.Popen(
                                argv, cwd=work,
                                env=environment(root, plan, work, "training"),
                                stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
                        processes[key] = process
                        states[key].update(state="running", pid=process.pid,
                                           started_unix=time.time())
                        status("running", resources=resource)
                time.sleep(plan["poll_seconds"])
        except BaseException as error:
            status("manager_failed", error=repr(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.cpu_preflight:
        plan = read(root / "plan.json")
        for job in plan["jobs"]:
            preflight(root, plan, job, "cpu")
    else:
        execute(root)
