"""External supervision only; the official CQL training process is unchanged."""

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

UPSTREAM_REVISION = "6afec90484bbf47dee05fdf525e26a3ebe028e9b"
D4RL_REVISION = "db6e4b34bb5ce2a51dd3879177c0a0223208a614"
UPSTREAM_FILES = {
    "algorithms/offline/cql.py":
        "76e0db2667c7f1a51c77840fb2aeaa97fcefa165fe1556e54a5981d925ca3b81",
    "configs/offline/cql/antmaze/umaze_v2.yaml":
        "eb158e0fd618511196958320cab5da3e9f2e2a31463a539edf348284f287ee8e",
    "configs/offline/cql/antmaze/umaze_diverse_v2.yaml":
        "6c63993954b1546e847cc9e7984d23005bccb1d475bd5029ec080d4a0b38dd76",
}

def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(root, plan):
    for relative, expected in plan["frozen_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")


def predecessor_ready(plan):
    previous = Path(plan["predecessor"])
    if digest(previous / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("Predecessor plan changed")
    status = read(previous / "queue/status.json")
    expected = {j["id"] for j in read(previous / "plan.json")["jobs"]}
    jobs = status.get("jobs", {})
    return set(jobs) == expected and all(
        job.get("state") == "completed" for job in jobs.values()
    )


def resources():
    gpu = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15).strip().split(",")
    pids = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    available = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    return dict(free_gpu_mib=int(gpu[0]), utilization=int(gpu[1]),
                free_host_mib=int(available.split()[1]) // 1024,
                gpu_pids=[int(pid) for pid in pids.splitlines() if pid.strip()])


def admissible(plan, resource, own):
    visible = set(resource["gpu_pids"])
    pending = len(set(own) - visible)
    return (len(own) < plan["max_parallel"]
            and len(visible | set(own)) + 1 <= plan["max_gpu_processes"]
            and resource["free_gpu_mib"] >= plan["gpu_memory_reserve_mib"]
            + (pending + 1) * plan["gpu_memory_per_job_mib"]
            and resource["free_host_mib"] >= plan["host_memory_reserve_mib"]
            + (pending + 1) * plan["host_memory_per_job_mib"]
            and resource["utilization"] <= plan["max_admission_utilization"])


def evaluations(log):
    """Parse only completed evaluations, never an announced but unfinished step."""
    step, records = None, []
    for line in log.splitlines():
        if line.startswith("Time steps: "):
            step = int(line.split(":", 1)[1])
        match = re.search(
            r"Evaluation over (\d+) episodes: ([^ ,]+) , D4RL score: ([^ ]+)", line
        )
        if match and step is not None:
            score = float(match[3])
            if not math.isfinite(score):
                raise ValueError("Non-finite evaluation")
            records.append(dict(completed_updates=step, episodes=int(match[1]),
                                raw_return_mean=float(match[2]),
                                normalized_score=score))
            step = None
    return records


def completed(work, updates, every, episodes):
    records = evaluations((work / "console.log").read_text(errors="replace"))
    expected_steps = list(range(every, updates + 1, every))
    if ([r["completed_updates"] for r in records] != expected_steps
            or any(r["episodes"] != episodes for r in records)):
        raise ValueError("Incomplete official evaluation history")
    checkpoints = list((work / "checkpoints").glob(f"*/checkpoint_{updates - 1}.pt"))
    if len(checkpoints) != 1 or checkpoints[0].stat().st_size == 0:
        raise ValueError("Missing official final checkpoint")
    return dict(completed_updates=updates, final_normalized_score=records[-1][
        "normalized_score"], best_normalized_score=max(r["normalized_score"]
                                                     for r in records),
                evaluations=records, final_checkpoint=str(checkpoints[0]))
