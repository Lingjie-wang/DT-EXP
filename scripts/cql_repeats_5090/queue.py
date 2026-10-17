"""Resource-limited paired queue; existing GPU jobs count toward the limit."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from scripts.cql_5090.run import require_preflight
from scripts.cql_delayed.common import digest, read, verify, write

def capacity(plan, gpu_pids, own_pids, free_gpu_mib, free_host_mib, pair_size=2):
    return (len(set(gpu_pids) | set(own_pids)) + pair_size <= plan["max_gpu_processes"]
            and free_gpu_mib >= (plan["gpu_memory_reserve_mib"]
                                + pair_size * plan["gpu_memory_per_job_mib"])
            and free_host_mib >= plan["host_memory_reserve_mib"])


def resources():
    def query(value):
        return subprocess.check_output(
            ["nvidia-smi", "-i", "0", value, "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip()
    pids = [int(x.strip()) for x in query("--query-compute-apps=pid").splitlines()
            if x.strip()]
    free = int(query("--query-gpu=memory.free"))
    available = next(int(line.split()[1]) // 1024
                     for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    return dict(gpu_pids=pids, free_gpu_mib=free, free_host_mib=available)


def validate_job(root, job):
    campaign = root / job["directory"]
    if digest(campaign / "protocol.json") != job["protocol_sha256"]:
        raise ValueError("Queue protocol changed")
    p = verify(campaign)
    spec = p["runs"][job["arm"]]
    expected_updates = read(root / "plan.json")["updates"]
    config = yaml.safe_load((campaign / "source" / spec["config"]).read_text())
    if (expected_updates <= 0 or spec["seed"] != job["seed"]
            or spec["updates"] != expected_updates
            or config["seed"] != job["seed"]
            or config["max_timesteps"] != expected_updates):
        raise ValueError("Queue seed/budget differs")
    for name, expected in spec["data_sha256"].items():
        if digest(campaign / job["arm"] / name) != expected:
            raise ValueError("Queue reward data changed")
    return campaign


def execute(root):
    plan = read(root / "plan.json")
    launch = root / "queue"
    launch.mkdir(exist_ok=True)
    code = Path(__file__).resolve().parents[2]
    with (launch / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # No silent replay after a manager failure. Existing attempts are preserved.
        if (launch / "status.json").exists():
            raise FileExistsError("Queue already started; inspect before recovery")
        states = {j["id"]: {"state": "queued"} for j in plan["jobs"]}
        processes = {}
        observers = {}
        pairs = list(dict.fromkeys(j["pair"] for j in plan["jobs"]))

        def status(stage, **extra):
            write(launch / "status.json", dict(
                stage=stage, pid=os.getpid(), updated_unix=time.time(), jobs=states,
                observers={key: {"pid": p.pid, "exit": p.poll()}
                           for key, p in observers.items()}, **extra))

        def training_command(job, smoke=False):
            campaign = root / job["directory"]
            return [sys.executable,
                    str(campaign / "source/scripts" / job["runner"] / "train.py"),
                    "--root", str(campaign / "validation" if smoke else campaign),
                    "--arm", job["arm"]] + (["--smoke"] if smoke else [])

        try:
            for job in plan["jobs"]:
                campaign = validate_job(root, job)
                if (campaign / job["arm"] / "training").exists():
                    raise FileExistsError("Preserve existing training results")
            status("queued")
            while True:
                for job in plan["jobs"]:
                    key = job["id"]
                    if key not in processes or states[key]["state"] != "running":
                        continue
                    process = processes[key]
                    if process.poll() is None:
                        continue
                    path = root / job["directory"] / job["arm"] / "training/status.json"
                    local = read(path) if path.exists() else {}
                    success = (process.returncode == 0 and local.get("status")
                               == "completed" and local.get("completed_updates")
                               == plan["updates"])
                    states[key].update(state="completed" if success else "failed",
                                       exit=process.returncode, training=local)
                pending = [pair for pair in pairs if all(
                    states[j["id"]]["state"] in {"queued", "ready"}
                    for j in plan["jobs"] if j["pair"] == pair)]
                own = [p.pid for p in processes.values() if p.poll() is None]
                if not pending and not own:
                    status("training_completed" if all(s["state"] == "completed"
                           for s in states.values()) else "finished_with_failures")
                    return
                try:
                    resource = resources()
                except Exception as error:
                    status("waiting_for_resource_probe", error=repr(error))
                    time.sleep(plan["poll_seconds"])
                    continue
                status("running" if own else "waiting_for_slots", resources=resource)
                if pending and capacity(plan, own_pids=own, **resource):
                    pair = [j for j in plan["jobs"] if j["pair"] == pending[0]]
                    # Admit both methods together. Smoke processes run sequentially
                    # in the two reserved slots, and never become formal checkpoints.
                    try:
                        for job in pair:
                            if states[job["id"]]["state"] == "ready":
                                continue
                            campaign = validate_job(root, job)
                            states[job["id"]]["state"] = "gpu_preflight"
                            status("gpu_preflight", pair=pending[0])
                            log_path = launch / f"{job['id']}-preflight.log"
                            with log_path.open("x") as log:
                                subprocess.run(
                                    training_command(job, smoke=True),
                                    cwd=campaign / "source", env=dict(
                                        os.environ, PYTHONPATH=str(campaign / "source"),
                                        WANDB_MODE="disabled"), stdout=log,
                                    stderr=subprocess.STDOUT, check=True, timeout=600)
                            require_preflight(campaign / "validation" / job["arm"]
                                              / "preflight")
                            states[job["id"]]["state"] = "ready"
                    except Exception as error:
                        for job in pair:
                            states[job["id"]].update(state="preflight_failed",
                                                     error=repr(error))
                        status("pair_preflight_failed")
                        continue
                    # Recheck after smoke runs: an unrelated GPU user may have started.
                    resource = resources()
                    if not capacity(plan, own_pids=own, **resource):
                        status("waiting_for_slots", resources=resource)
                        time.sleep(plan["poll_seconds"])
                        continue
                    for job in pair:
                        campaign = root / job["directory"]
                        with (launch / f"{job['id']}-training.log").open("x") as log:
                            process = subprocess.Popen(
                                training_command(job), cwd=campaign / "source",
                                env=dict(os.environ, PYTHONPATH=str(campaign / "source"),
                                         WANDB_MODE="disabled"), stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
                        processes[job["id"]] = process
                        states[job["id"]].update(state="running", pid=process.pid,
                                                 started_unix=time.time())
                        with (launch / f"{job['id']}-wandb.log").open("x") as log:
                            observers[job["id"]] = subprocess.Popen(
                                [sys.executable, "-m", "scripts.cql_5090.observe",
                                 "--root", str(campaign), "--arm", job["arm"]],
                                cwd=code, env=dict(os.environ, PYTHONPATH=str(code),
                                                   WANDB_MODE="online"), stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
                    status("running", resources=resource)
                time.sleep(plan["poll_seconds"])
        except BaseException as error:
            status("manager_failed", error=repr(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    execute(args.root.resolve())
