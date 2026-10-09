"""Independent retry of unstarted AntMaze policies using safe module invocation."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.cql_5090.run import require_preflight
from scripts.cql_delayed.common import read, write
from scripts.cql_repeats_5090.queue import validate_job
from scripts.cql_uniform_5090.queue import admissible, probe

def execute(root):
    plan = read(root / "plan.json")
    queue = root / "queue"
    queue.mkdir(exist_ok=True)
    code = Path(__file__).resolve().parents[2]
    with (queue / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (queue / "status.json").exists():
            raise FileExistsError("Preserve this launch; inspect before recovery")
        states = {job["id"]: dict(state="queued") for job in plan["jobs"]}
        training, observers = {}, {}

        def status(stage, **extra):
            write(queue / "status.json", dict(
                stage=stage, pid=os.getpid(), updated_unix=time.time(), jobs=states,
                observers={key: dict(pid=p.pid, exit=p.poll())
                           for key, p in observers.items()}, **extra))

        def command(job, smoke=False):
            campaign = root / job["directory"]
            return [sys.executable, "-m", "scripts.cql_antmaze_retry_5090.train",
                    "--root", str(campaign / "validation" if smoke else campaign),
                    "--arm", job["arm"]] + (["--smoke"] if smoke else [])

        def observer(job):
            with (queue / f"{job['id']}-wandb.log").open("a") as stream:
                return subprocess.Popen([
                    sys.executable, "-m", "scripts.cql_antmaze_5090.sync",
                    "--root", str(root / job["directory"]), "--arm", job["arm"]],
                    cwd=code, env=dict(os.environ, PYTHONPATH=str(code),
                                      WANDB_MODE="online", CUDA_VISIBLE_DEVICES=""),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)

        try:
            for job in plan["jobs"]:
                campaign = validate_job(root, job)
                if (campaign / job["arm"] / "training").exists():
                    raise FileExistsError("Historical training must be preserved")
            while True:
                for job in plan["jobs"]:
                    key = job["id"]
                    if (states[key]["state"] == "running"
                            and training[key].poll() is not None):
                        path = root / job["directory"] / job["arm"]
                        path = path / "training/status.json"
                        local = read(path) if path.exists() else {}
                        success = (training[key].returncode == 0
                                   and local.get("status") == "completed"
                                   and local.get("completed_updates") == plan["updates"])
                        states[key].update(state="completed" if success else "failed",
                                           exit=training[key].returncode, training=local)
                    receipt = (root / job["directory"] / "wandb_sync" / job["arm"]
                               / "full_completion_verification.json")
                    if (states[key]["state"] in {"running", "completed"}
                            and not receipt.exists()
                            and (key not in observers
                                 or observers[key].poll() is not None)):
                        observers[key] = observer(job)
                own = [p.pid for p in training.values() if p.poll() is None]
                pending = [j for j in plan["jobs"]
                           if states[j["id"]]["state"] in {"queued", "ready"}]
                if not own and not pending:
                    waiting = [j for j in plan["jobs"]
                               if states[j["id"]]["state"] == "completed" and not
                               (root / j["directory"] / "wandb_sync" / j["arm"]
                                / "full_completion_verification.json").exists()]
                    status("waiting_for_wandb" if waiting else "completed"
                           if all(s["state"] == "completed" for s in states.values())
                           else "finished_with_failures")
                    if not waiting:
                        return
                    time.sleep(plan["poll_seconds"])
                    continue
                try:
                    resource = probe()
                except Exception as error:
                    status("waiting_for_resource_probe", error=repr(error))
                    time.sleep(plan["poll_seconds"])
                    continue
                status("running" if own else "waiting_for_slots", resources=resource)
                if pending and admissible(plan, resource, own):
                    job = pending[0]
                    campaign = validate_job(root, job)
                    env = dict(os.environ, PYTHONPATH=str(campaign / "source"),
                               WANDB_MODE="disabled")
                    if states[job["id"]]["state"] == "queued":
                        states[job["id"]]["state"] = "gpu_preflight"
                        status("gpu_preflight", resources=resource)
                        try:
                            with (queue / f"{job['id']}-preflight.log").open("x") as log:
                                subprocess.run(command(job, smoke=True),
                                               cwd=campaign / "source", env=env,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               check=True, timeout=600)
                            require_preflight(campaign / "validation" / job["arm"]
                                              / "preflight")
                            states[job["id"]]["state"] = "ready"
                        except Exception as error:
                            states[job["id"]].update(state="preflight_failed",
                                                     error=repr(error))
                            status("preflight_failed")
                            continue
                    try:
                        resource = probe()
                    except Exception as error:
                        status("waiting_for_resource_probe", error=repr(error))
                        time.sleep(plan["poll_seconds"])
                        continue
                    if admissible(plan, resource, own):
                        with (queue / f"{job['id']}-training.log").open("x") as log:
                            process = subprocess.Popen(
                                command(job), cwd=campaign / "source", env=env,
                                stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
                        training[job["id"]] = process
                        states[job["id"]].update(state="running", pid=process.pid,
                                                 started_unix=time.time())
                        observers[job["id"]] = observer(job)
                        status("running", resources=resource)
                time.sleep(plan["poll_seconds"])
        except BaseException as error:
            status("manager_failed", error=repr(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    execute(parser.parse_args().root.resolve())
