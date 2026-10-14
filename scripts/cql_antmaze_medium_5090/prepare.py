"""Freeze six paired Medium AntMaze jobs without starting any learning."""

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

from scripts.cql_antmaze_official_5090.common import digest, read, verify, write
from scripts.cql_antmaze_success_5090.fit import SETTINGS

CONFIG_HASHES = {
    "medium_play": "af70ab55185ea22c5a2b2b5cfc3e3b6cb523e7478e5a60fe8469c5f684d26211",
    "medium_diverse": "42f0a6b888ce73fea7f24976ca0e8dd70bbcc277421a58f48a23d6d979079fc7",
}
ARMS = ("original", "uniform", "shapley")


def job_matrix(seed):
    jobs = []
    for arm in ARMS:
        for name, multi in [("medium_play", "False"), ("medium_diverse", "True")]:
            jobs.append(dict(id=f"{name}-{arm}-seed{seed}", seed=seed, arm=arm,
                dataset_key=name, env=f"antmaze-{name.replace('_', '-')}-v2",
                config=f"configs/offline/cql/antmaze/{name}_v2.yaml",
                dataset="Ant_maze_big-maze_noisy_multistart_True_multigoal_"
                        f"{multi}_sparse_fixed.hdf5",
                dataset_dir=("datasets" if arm == "original"
                             else f"reward_datasets/{arm}"),
                updates=1000000, eval_every=50000, eval_episodes=100))
    return jobs


def prepare(root, predecessor, project, inputs, audit, revision, seed=22):
    old = read(predecessor / "plan.json")
    verify(predecessor, old)
    reservation = read(audit)
    if (not reservation.get("both_servers_checked")
            or seed in reservation["reserved_seeds"]):
        raise ValueError("Require a fresh seed checked against both servers")
    for name, expected in CONFIG_HASHES.items():
        if digest(inputs / f"{name}_v2.yaml") != expected:
            raise ValueError("Official Medium config hash mismatch")
    root.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for folder in ["upstream", "dependencies"]:
        shutil.copytree(predecessor / folder, root / folder, ignore=ignore)
    for folder in ["cql_antmaze_medium_5090", "cql_antmaze_success_5090",
                   "cql_antmaze_official_5090", "cql_antmaze_official_retry_5090",
                   "cql_antmaze_success_wandb_recovery"]:
        shutil.copytree(project / "scripts" / folder, root / "source/scripts" / folder,
                        ignore=ignore)
    target = root / "source/algorithms/offline"
    target.mkdir(parents=True)
    shutil.copy2(project / "algorithms/offline/shapley_redistribution.py", target)
    for folder in ["datasets", "predictor", "comparison", "queue", "telemetry_relay"]:
        (root / folder).mkdir()
    jobs = job_matrix(seed)
    datasets = {}
    for name in CONFIG_HASHES:
        job = next(j for j in jobs if j["dataset_key"] == name)
        config = inputs / f"{name}_v2.yaml"
        settings = yaml.safe_load(config.read_text())
        if (settings["env"] != job["env"] or settings["max_timesteps"] != 1000000
                or settings["eval_freq"] != 50000 or settings["n_episodes"] != 100):
            raise ValueError("Unexpected official task or budget")
        shutil.copy2(config, root / "upstream" / job["config"])
        source = inputs / job["dataset"]
        with h5py.File(source, "r") as data:
            rewards, terminals = data["rewards"][:], data["terminals"][:]
            if (not np.array_equal(rewards, terminals)
                    or not np.isin(rewards, [0, 1]).all()):
                raise ValueError("Require the audited binary-success reward structure")
            datasets[name] = dict(rows=len(rewards), successes=int(rewards.sum()),
                                  sha256=digest(source))
        shutil.copy2(source, root / "datasets" / job["dataset"])
    for job in jobs:
        (root / job["id"]).mkdir()
    plan = {key: old[key] for key in [
        "upstream_revision", "d4rl_revision", "wandb_entity", "wandb_project",
        "runtime_python", "required_wandb_version", "max_parallel",
        "max_gpu_processes", "gpu_memory_per_job_mib", "gpu_memory_reserve_mib",
        "host_memory_per_job_mib", "host_memory_reserve_mib",
        "max_admission_utilization", "poll_seconds"]}
    plan.update(jobs=jobs, controls={j["dataset_key"]: dict(env=j["env"])
                for j in jobs}, seed=seed, code_revision=revision,
                project=str(project),
                predecessor=str(predecessor), predecessor_plan_sha256=digest(
                    predecessor / "plan.json"), seed_audit=reservation,
                attribution=SETTINGS, success_timing_available=True,
                protocol="Medium: success-or-timeout / official transitions v1",
                wait_before_predictor=True, raw_data_audit=datasets,
                budget_rationale="Official 1M updates, 50k interval, 100 episodes; "
                "prior Umaze curves still improve/fluctuate at 850k-950k.",
                selection="maximum observed 100-episode evaluation; retain full curves",
                frozen_sha256={})
    for folder in ["upstream", "dependencies", "datasets", "source"]:
        for path in (root / folder).rglob("*"):
            if path.is_file():
                plan["frozen_sha256"][str(path.relative_to(root))] = digest(path)
    write(root / "plan.json", plan)
    write(root / "queue/status.json", dict(stage="prepared", jobs={
        j["id"]: dict(state="queued") for j in jobs}))
    print(root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "predecessor", "project", "inputs", "audit"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()
    prepare(*(getattr(args, name).resolve() for name in
              ["root", "predecessor", "project", "inputs", "audit"]),
            args.revision, args.seed)
