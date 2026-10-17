"""Freeze four reward-only Maze2D variants around an unchanged official CQL CLI."""

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

from scripts.cql_antmaze_official_5090.common import digest, read, verify, write
from scripts.cql_antmaze_success_5090.fit import SETTINGS

CONFIG_SHA256 = "be51a3197b81f40a8858a7e268d74576698dc663f499fd34d3021dc981dac51e"
DATA_SHA256 = "f2d327d95067d53e274ac2fe876b740ed7883d0a9738bd5b447635cdb786bfb7"
CONFIG = "configs/offline/cql/maze2d/umaze_v1.yaml"
DATASET = "maze2d-umaze-sparse-v1.hdf5"
ARMS = ("original", "delayed", "uniform", "shapley")


def job_matrix(seed):
    return [dict(id=f"umaze-{arm}-seed{seed}", seed=seed, arm=arm,
                 dataset_key="umaze", env="maze2d-umaze-v1", config=CONFIG,
                 dataset=DATASET, dataset_dir=("datasets" if arm == "original"
                 else f"reward_datasets/{arm}"), updates=1000000,
                 eval_every=5000, eval_episodes=100) for arm in ARMS]


def prepare(root, reference, project, inputs, audit, revision, seed=23):
    old = read(reference / "plan.json")
    verify(reference, old)
    reservation = read(audit)
    if (not reservation.get("both_servers_checked")
            or seed in reservation["reserved_seeds"]):
        raise ValueError("Require a fresh paired seed checked on both servers")
    if digest(inputs / "umaze_v1.yaml") != CONFIG_SHA256:
        raise ValueError("Official Maze2D YAML hash mismatch")
    if digest(inputs / DATASET) != DATA_SHA256:
        raise ValueError("Official Maze2D HDF5 hash mismatch")
    config = yaml.safe_load((inputs / "umaze_v1.yaml").read_text())
    expected = dict(env="maze2d-umaze-v1", max_timesteps=1000000, eval_freq=5000,
                    n_episodes=100, normalize_reward=False, reward_scale=1.0,
                    reward_bias=0.0)
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("Unexpected official configuration")
    with h5py.File(inputs / DATASET, "r") as data:
        if (data["observations"].shape != (1000000, 4)
                or data["actions"].shape != (1000000, 2)
                or np.any(data["terminals"][:])
                or int(data["timeouts"][:].sum()) != 12459
                or not np.isin(data["rewards"][:], [0, 1]).all()):
            raise ValueError("Dataset differs from the audited sparse v1 data")
    root.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
    for folder in ["upstream", "dependencies"]:
        shutil.copytree(reference / folder, root / folder, ignore=ignore)
    destination = root / "upstream" / CONFIG
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs / "umaze_v1.yaml", destination)
    for folder in ["cql_maze2d_four_5090", "cql_antmaze_success_5090",
                   "cql_antmaze_official_5090"]:
        shutil.copytree(project / "scripts" / folder,
                        root / "source/scripts" / folder, ignore=ignore)
    target = root / "source/algorithms/offline"
    target.mkdir(parents=True)
    shutil.copy2(project / "algorithms/offline/shapley_redistribution.py", target)
    for folder in ["datasets", "predictor", "comparison", "queue"]:
        (root / folder).mkdir()
    shutil.copy2(inputs / DATASET, root / "datasets" / DATASET)
    jobs = job_matrix(seed)
    for job in jobs:
        (root / job["id"]).mkdir()
    plan = {key: old[key] for key in [
        "upstream_revision", "d4rl_revision", "wandb_entity", "wandb_project",
        "runtime_python", "required_wandb_version", "max_parallel",
        "max_gpu_processes", "gpu_memory_per_job_mib", "gpu_memory_reserve_mib",
        "host_memory_per_job_mib", "host_memory_reserve_mib",
        "max_admission_utilization", "poll_seconds"]}
    plan.update(jobs=jobs, controls={"umaze": dict(env="maze2d-umaze-v1")},
                seed=seed, code_revision=revision, project=str(project),
                reference=str(reference), seed_audit=reservation,
                attribution=SETTINGS, predictor_device="cpu",
                success_timing_available=False,
                protocol="timeout fragments / official retained transitions / totals v1",
                budget_rationale="Unmodified official Maze2D YAML: 1M updates, "
                "5k interval, 100 episodes. Prior AntMaze peaks at 350k-900k; "
                "do not shorten the official budget.",
                information="Predictor reads state/action, boundaries and scalar "
                "fragment totals only. It never reads per-step rewards.",
                conservation="Undiscounted retained-fragment returns only; gamma=.99 "
                "means discounted returns and optimal policies need not be invariant.",
                selection="maximum observed evaluation; retain all 200 evaluations",
                reward_normalization=False, frozen_sha256={})
    for folder in ["upstream", "dependencies", "datasets", "source"]:
        for path in (root / folder).rglob("*"):
            if path.is_file():
                plan["frozen_sha256"][str(path.relative_to(root))] = digest(path)
    write(root / "plan.json", plan)
    write(root / "queue/status.json", dict(stage="prepared", jobs={
        job["id"]: dict(state="queued") for job in jobs}))
    print(root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "reference", "project", "inputs", "audit"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    prepare(*(getattr(args, name).resolve() for name in
              ["root", "reference", "project", "inputs", "audit"]),
            args.revision, args.seed)
