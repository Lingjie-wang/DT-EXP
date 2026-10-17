"""Freeze original official inputs and a separate success-boundary experiment."""

import argparse
import shutil
from pathlib import Path

from scripts.cql_antmaze_official_5090.common import (
    digest,
    read,
    UPSTREAM_FILES,
    verify,
    write,
)
from scripts.cql_antmaze_success_5090.fit import SETTINGS

def prepare(root, baseline, project, revision):
    old = read(baseline / "plan.json")
    verify(baseline, old)
    state = read(baseline / "queue/status.json")
    if not all(state["jobs"][j["id"]]["state"] == "completed" for j in old["jobs"]):
        raise ValueError("Require completed official controls")
    for relative, expected in UPSTREAM_FILES.items():
        if digest(baseline / "upstream" / relative) != expected:
            raise ValueError("Official input hash mismatch")
    root.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for folder in ["upstream", "dependencies", "datasets"]:
        shutil.copytree(baseline / folder, root / folder, ignore=ignore)
    for folder in ["scripts/cql_antmaze_success_5090",
                   "scripts/cql_antmaze_official_5090",
                   "scripts/cql_antmaze_official_retry_5090"]:
        shutil.copytree(project / folder, root / "source" / folder, ignore=ignore)
    target = root / "source/algorithms/offline"
    target.mkdir(parents=True)
    shutil.copy2(project / "algorithms/offline/shapley_redistribution.py", target)
    for folder in ["queue", "baseline", "predictor", "comparison"]:
        (root / folder).mkdir()
    jobs, controls = [], {}
    for original in old["jobs"]:
        name = original["id"].split("-official-")[0]
        if original["seed"] != 21:
            raise ValueError("Pair with the existing seed 21 baseline")
        source = baseline / original["id"] / "training"
        records = read(source / "evaluations.json")
        expected_steps = list(range(50000, 1000001, 50000))
        if ([r["completed_updates"] for r in records] != expected_steps
                or any(r["episodes"] != 100 for r in records)):
            raise ValueError("Incomplete official baseline")
        write(root / "baseline" / f"{name}.json", records)
        controls[name] = dict(env=original["env"], evaluations=f"baseline/{name}.json",
                              **read(source / "wandb_manifest.json"))
        for arm in ["uniform", "shapley"]:
            job = dict(original, id=f"{name}-{arm}-seed21", arm=arm, dataset_key=name,
                       dataset_dir=f"reward_datasets/{arm}")
            jobs.append(job)
            (root / job["id"]).mkdir()
    # Interleave datasets so neither waits behind both arms of the other dataset.
    jobs.sort(key=lambda j: (j["arm"] != "uniform", j["dataset_key"]))
    plan = {key: old[key] for key in [
        "upstream_revision", "d4rl_revision", "wandb_entity", "wandb_project",
        "runtime_python", "required_wandb_version", "max_parallel",
        "max_gpu_processes", "gpu_memory_per_job_mib", "gpu_memory_reserve_mib",
        "host_memory_per_job_mib", "host_memory_reserve_mib",
        "max_admission_utilization", "poll_seconds"]}
    plan.update(jobs=jobs, controls=controls, code_revision=revision,
                project=str(project),
                predecessor=str(baseline), predecessor_plan_sha256=digest(
                    baseline / "plan.json"), attribution=SETTINGS,
                protocol="success-or-timeout / official retained transitions v1",
                success_timing_available=True,
                changes=["Only HDF5 rewards change; official source/config unchanged",
                         "Split at success OR timeout, then discard timeout rows",
                         "Keep every default official loader transition, including tail",
                         "Group cross fitting by parent timeout collection fragment",
                         "Up to 20 valid segments, 128 permutations; singleton exact",
                         "Predictor diagnostic gate is report-only; no fallback"],
                seed_audit="2026-10-13: Slurm has no queued jobs or AntMaze plans; "
                           "reuse seed 21 to pair with completed official controls",
                budget_rationale="Official 1M updates / 50k evaluation / 100 episodes. "
                                 "Seed21 best: Umaze94@550k, Diverse55@700k.",
                selection="Compare maximum observed evaluation score, never final only",
                frozen_sha256={})
    for folder in ["upstream", "dependencies", "datasets", "source", "baseline"]:
        for path in (root / folder).rglob("*"):
            if path.is_file():
                plan["frozen_sha256"][str(path.relative_to(root))] = digest(path)
    write(root / "plan.json", plan)
    print(root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "baseline", "project"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.baseline.resolve(), args.project.resolve(),
            args.revision)
