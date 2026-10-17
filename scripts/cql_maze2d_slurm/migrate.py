"""Adopt a frozen, untrained 5090 campaign in a fresh Slurm directory."""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

from scripts.cql_antmaze_official_5090.common import digest, read, verify, write

def migrate(root, project, runtime):
    plan = read(root / "plan.json")
    stop = read(root / "migration_stop.json")
    if stop["formal_training_started"] or stop["remaining_processes"]:
        raise ValueError("Source campaign must be stopped before any training")
    if any((root / j["id"] / "training").exists() for j in plan["jobs"]):
        raise ValueError("Refuse to migrate an already trained campaign")
    verify(root, plan)
    provenance = root / "provenance_5090"
    provenance.mkdir()
    shutil.copy2(root / "plan.json", provenance / "plan.json")
    shutil.copy2(root / "migration_stop.json", provenance / "migration_stop.json")
    for job in plan["jobs"]:
        audit = root / job["id"] / "data_audit.json"
        shutil.move(str(audit), provenance / (job["id"] + "-data_audit.json"))
    source = root / "source/scripts/cql_maze2d_slurm"
    shutil.copytree(project / "scripts/cql_maze2d_slurm", source,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for file in source.rglob("*"):
        if file.is_file():
            plan["frozen_sha256"][str(file.relative_to(root))] = digest(file)
    plan.update(runtime_python=str(runtime), project=str(project),
                migration_unix=time.time(), scheduler="slurm", max_parallel=2,
                migration_code_revision=subprocess.check_output(
                    ["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
                slurm_resources=dict(partition="GPULite", gpus_per_job=1,
                                     cpus_per_job=2, memory="8G", time="72:00:00"))
    write(root / "plan.json", plan)
    (root / "queue").mkdir()
    (root / "slurm").mkdir()
    write(root / "pipeline_status.json", dict(stage="slurm_training_queue"))
    write(root / "queue/status.json", dict(stage="prepared", jobs={
        j["id"]: dict(state="queued") for j in plan["jobs"]}))
    verify(root, plan)
    print(root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "project", "runtime"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    migrate(args.root.resolve(), args.project.resolve(), args.runtime.absolute())
