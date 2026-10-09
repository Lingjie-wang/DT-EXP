"""Freeze the unmodified upstream entry point, configs, D4RL and original datasets."""

import argparse
import shutil
from pathlib import Path

import yaml

from scripts.cql_antmaze_official_5090.common import (
    D4RL_REVISION,
    digest,
    read,
    UPSTREAM_FILES,
    UPSTREAM_REVISION,
    write,
)

def prepare(root, project, predecessor, d4rl_source, seed, revision):
    if root.exists():
        raise FileExistsError("Preserve every existing experiment")
    code = Path(__file__).resolve().parents[2]
    for relative, expected in UPSTREAM_FILES.items():
        if digest(code / relative) != expected:
            raise ValueError(f"Not the pinned official CORL file: {relative}")
    dependency = read(d4rl_source / "source_manifest.json")
    if dependency["revision"] != D4RL_REVISION:
        raise ValueError("Wrong upstream D4RL revision")
    for relative, expected in dependency["sha256"].items():
        if digest(d4rl_source / relative) != expected:
            raise ValueError(f"D4RL source changed: {relative}")
    root.mkdir(parents=True)
    for relative in UPSTREAM_FILES:
        target = root / "upstream" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(code / relative, target)
    shutil.copytree(d4rl_source / "d4rl", root / "dependencies/d4rl/d4rl",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(d4rl_source / "source_manifest.json",
                 root / "dependencies/d4rl/source_manifest.json")
    harness = root / "source/scripts/cql_antmaze_official_5090"
    harness.parent.mkdir(parents=True)
    shutil.copytree(code / "scripts/cql_antmaze_official_5090", harness,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (root / "datasets").mkdir()
    (root / "queue").mkdir()
    jobs = []
    for name in ["umaze", "umaze_diverse"]:
        audit = read(predecessor / f"{name}-dense-seed21/dense/audit.json")
        dataset = Path(audit["source_file"])
        if digest(dataset) != audit["original_sha256"]:
            raise ValueError("Original HDF5 changed")
        shutil.copy2(dataset, root / "datasets" / dataset.name)
        config = f"configs/offline/cql/antmaze/{name}_v2.yaml"
        values = yaml.safe_load((root / "upstream" / config).read_text())
        assert values["max_timesteps"] == 1000000
        assert values["eval_freq"] == 50000 and values["n_episodes"] == 100
        job = dict(id=f"{name}-official-seed{seed}", env=values["env"], seed=seed,
                   config=config, dataset=dataset.name, updates=1000000,
                   eval_every=50000, eval_episodes=100)
        jobs.append(job)
        (root / job["id"]).mkdir()
    files = [p for folder in ["upstream", "dependencies", "source", "datasets"]
             for p in (root / folder).rglob("*") if p.is_file()]
    plan = dict(
        jobs=jobs, upstream_repository="https://github.com/tinkoff-ai/CORL",
        upstream_revision=UPSTREAM_REVISION, d4rl_revision=D4RL_REVISION,
        code_revision=revision,
        predecessor=str(predecessor), predecessor_plan_sha256=digest(
            predecessor / "plan.json"), project=str(project),
        frozen_sha256={str(p.relative_to(root)): digest(p) for p in files},
        max_parallel=2, max_gpu_processes=8, gpu_memory_per_job_mib=5120,
        gpu_memory_reserve_mib=4096, host_memory_per_job_mib=6144,
        host_memory_reserve_mib=8192, max_admission_utilization=85,
        poll_seconds=30, wandb_project="CORL-DDR",
        wandb_entity="2820402607-shandong-university",
        official_data="d4rl.qlearning_dataset(env), default terminate_on_end=False",
        changes=["seed 21 pairs with existing reward controls",
                 "W&B project, group and name identify the separate baseline",
                 "Enable checkpoint output; retain original training and eval loops"],
        seed_audit="5090: existing timeout controls use 21; no official baseline. "
                   "Slurm: no queued jobs or AntMaze plan files on 2026-10-10.",
        budget_rationale="1M updates, 50k evaluation interval, 100 episodes: "
                         "the pinned official configs, versus 300k existing controls",
    )
    write(root / "plan.json", plan)
    print(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "project", "predecessor", "d4rl-source"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.project.resolve(), args.predecessor.resolve(),
            args.d4rl_source.resolve(), args.seed, args.revision)
