"""Prepare eight isolated, paired timeout-fragment AntMaze reward controls."""

import argparse
import copy
import hashlib
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

from algorithms.offline.shapley_redistribution import folds
from scripts.cql_antmaze_5090.data import prepare_arrays
from scripts.cql_delayed.common import digest, read, verify, write

SEED = 21
UPDATES = 300000
CAMPAIGN = "cql-antmaze-timeout-four-rewards-seed21-300k-5090-20261009"
PREDECESSOR = "cql-uniform-hcm-seeds1-11-12-500k-5090-20261008"
ARMS = ("delayed", "shapley", "uniform", "dense")
TASKS = dict(umaze=dict(env="antmaze-umaze-v2", horizon=701, episodes=1426,
                       filename="Ant_maze_u-maze_noisy_multistart_False_"
                       "multigoal_False_sparse_fixed.hdf5"),
             umaze_diverse=dict(env="antmaze-umaze-diverse-v2", horizon=1001,
                               episodes=999, filename="Ant_maze_u-maze_noisy_"
                               "multistart_True_multigoal_True_sparse_fixed.hdf5"))
ATTRIBUTION = dict(segments=20, permutations=128, outer_folds=5, width=64,
                   learning_rate=1e-3, weight_decay=1e-4, batch=32, epochs=300,
                   patience=30, validation_masks=8, attribution_seed_base=3000)


def policy_config(official, env, arm, group):
    if (official["env"] != env or not official["normalize_reward"]
            or official["reward_scale"] != 10 or official["reward_bias"] != -5):
        raise ValueError("Unexpected official AntMaze CQL config")
    config = copy.deepcopy(official)
    label = "shapley20x128" if arm == "shapley" else arm
    config.update(seed=SEED, max_timesteps=UPDATES, eval_freq=10000, group=group,
                  project="CORL-DDR", name=f"CQL-CORL-{env}-{label}-"
                  f"timeout-v1-seed{SEED}-300k-5090")
    return config


def populate_run(root, job, source_hashes, audit, config, revision):
    work = root / job["directory"]
    (work / job["arm"]).mkdir(parents=True)
    (work / "source").symlink_to("../source", target_is_directory=True)
    spec = dict(env=job["env"], seed=SEED, updates=UPDATES, eval_every=10000,
                eval_episodes=100, config=job["config"],
                reward_mode=f"{job['arm']}-timeout-v1",
                wandb_name=config["name"], wandb_id=job["wandb_id"])
    protocol = dict(method="CORL-CQL-AntMaze-timeout-controls-v1",
                    runs={job["arm"]: spec}, source_sha256=source_hashes,
                    code_revision=revision,
                    wandb_entity="2820402607-shandong-university",
                    wandb_project="CORL-DDR", wandb_group=root.name,
                    retained={key: config[key] for key in [
                        "discount", "normalize_reward", "reward_scale", "reward_bias",
                        "normalize", "batch_size", "cql_alpha", "cql_lagrange",
                        "policy_lr", "qf_lr", "buffer_size", "q_n_hidden_layers"]},
                    migration=dict(target_host="5090", policy_and_eval_seed=SEED,
                                   cql_algorithm_changed=False,
                                   boundary_adaptation="User chose timeouts only"),
                    changes=["Timeout fragments; discard incomplete tail in all arms",
                             "Success terminals excluded from data and predictor",
                             "Retain timeout sample and block bootstrap at timeout",
                             "CQL updates/networks unchanged; eight independent runs",
                             "Official reward scale/bias applied after redistribution",
                             "300k updates; 100 evaluation episodes every 10k updates",
                             "20 near-equal segments padded for 701/1001 lengths"],
                    input_audit=audit)
    validation = work / "validation"
    (validation / job["arm"]).mkdir(parents=True)
    (validation / "source").symlink_to("../../source", target_is_directory=True)
    for name in ["dataset.npz", "audit.json"]:
        (validation / job["arm"] / name).symlink_to(f"../../{job['arm']}/{name}")
    write(work / "protocol.json", protocol)
    write(validation / "protocol.json", protocol)


def finish_data(root, job, rewards, audit, gate=None):
    work = root / job["directory"]
    protocol = verify(work)
    dataset = root / "datasets" / job["dataset"]
    with np.load(dataset / "base_transitions.npz") as base:
        np.savez(work / job["arm"] / "dataset.npz", **dict(base), rewards=rewards)
    with np.load(dataset / "predictor/input.npz") as inputs:
        totals = inputs["returns"]
    length = read(dataset / "audit.json")["trajectory_length"]
    errors = rewards.reshape(-1, length).astype(np.float64).sum(1) - totals
    if not np.isfinite(rewards).all() or np.max(np.abs(errors)) >= 1e-3:
        raise ValueError("Prepared reward totals differ")
    audit = dict(audit, reward_mode=f"{job['arm']}-timeout-v1",
                 max_return_error=float(np.max(np.abs(errors))),
                 matched_nonreward_fields=True, policy_reward_transform="10*r - 5",
                 reward_transform_stage="after redistribution; same in all four arms")
    if gate is not None:
        if not gate["passed"]:
            raise ValueError("Refuse failed Shapley prediction gate")
        protocol["shapley_gate"] = gate
    write(work / job["arm"] / "audit.json", audit)
    protocol["runs"][job["arm"]]["data_sha256"] = {
        name: digest(work / job["arm"] / name) for name in ["dataset.npz", "audit.json"]}
    write(work / "protocol.json", protocol)
    write(work / "validation/protocol.json", protocol)
    job["protocol_sha256"] = digest(work / "protocol.json")


def materialize_shapley(root, job):
    dataset = root / "datasets" / job["dataset"]
    gate = read(dataset / "predictor/fit/gate.json")
    if not gate["passed"]:
        return False
    with np.load(dataset / "predictor/fit/attribution.npz") as stored:
        rewards = stored["rewards"].reshape(-1)
    finish_data(root, job, rewards, read(dataset / "predictor/fit/audit.json"), gate)
    return True


def prepare(project, inputs, revision):
    repository = Path(__file__).resolve().parents[2]
    root = project / "results" / CAMPAIGN
    if root.exists():
        raise FileExistsError("Preserve previous AntMaze campaign")
    for path in list((project / "results").glob("*/protocol.json")) + list(
            (project / "results").glob("*/*/protocol.json")):
        for spec in read(path).get("runs", {}).values():
            if spec.get("env") in {v["env"] for v in TASKS.values()} and spec.get(
                    "seed") == SEED:
                raise ValueError(f"AntMaze seed already reserved: {path}")
    predecessor = project / "results" / PREDECESSOR
    read(predecessor / "plan.json")
    raw_paths = {key: inputs / task["filename"] for key, task in TASKS.items()}
    for path in raw_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    root.mkdir(parents=True)
    source = root / "source"
    names = ["algorithms/offline/cql.py", "algorithms/offline/shapley_redistribution.py",
             "scripts/cql_delayed/common.py", "scripts/cql_delayed/trainer.py"]
    names += [str(path.relative_to(repository)) for path in
              (repository / "scripts/cql_antmaze_5090").glob("*.py")]
    names += [f"configs/offline/cql/antmaze/{key}_v2.yaml" for key in TASKS]
    for name in names:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / name, destination)
    jobs, configurations = [], {}
    for key, task in TASKS.items():
        official = yaml.safe_load((source / f"configs/offline/cql/antmaze/{key}_v2.yaml")
                                  .read_text())
        for arm in ARMS:
            directory = f"{key}-{arm}-seed{SEED}"
            config = policy_config(official, task["env"], arm, root.name)
            name = f"configs/antmaze_timeout/{key}_{arm}.yaml"
            (source / name).parent.mkdir(exist_ok=True)
            (source / name).write_text(yaml.safe_dump(config, sort_keys=False))
            names.append(name)
            configurations[directory] = config
            jobs.append(dict(id=directory, directory=directory, dataset=key, arm=arm,
                             seed=SEED, env=task["env"], config=name,
                             wandb_name=config["name"], wandb_id=hashlib.sha256(
                                 f"{root.name}/{directory}".encode()).hexdigest()[:12]))
    hashes = {name: digest(source / name) for name in names}
    for key, task in TASKS.items():
        dataset = root / "datasets" / key
        (dataset / "predictor").mkdir(parents=True)
        (dataset / "source").symlink_to("../../source", target_is_directory=True)
        with h5py.File(raw_paths[key], "r") as raw:
            # Never read success terminals: they encode the dense success signal.
            values = {field: raw[field][:] for field in [
                "observations", "actions", "rewards", "timeouts", "next_observations"]
                      if field in raw}
        base, predictor, rewards, audit = prepare_arrays(values, task["horizon"])
        if audit["episodes"] != task["episodes"]:
            raise ValueError("Unexpected official AntMaze population")
        audit.update(original_sha256=digest(raw_paths[key]),
                     source_file=str(raw_paths[key]),
                     source_url="https://rail.eecs.berkeley.edu/datasets/offline_rl/"
                     f"ant_maze_v2/{task['filename']}")
        write(dataset / "audit.json", audit)
        np.savez(dataset / "base_transitions.npz", **base)
        np.savez(dataset / "predictor/input.npz", **predictor)
        write(dataset / "predictor/folds.json", list(folds(audit["episodes"])))
        write(dataset / "protocol.json", dict(
            source_sha256=hashes, code_revision=revision, attribution=ATTRIBUTION,
            preparation_sha256={name: digest(dataset / name) for name in [
                "base_transitions.npz", "predictor/input.npz", "predictor/folds.json"]}))
        for job in (j for j in jobs if j["dataset"] == key):
            populate_run(root, job, hashes, audit, configurations[job["id"]], revision)
            if job["arm"] != "shapley":
                finish_data(root, job, rewards[job["arm"]], audit)
        del base, predictor, rewards, values
    plan = dict(jobs=jobs, seed=SEED, updates=UPDATES, code_revision=revision,
                datasets=list(TASKS), max_parallel=2, max_gpu_processes=8,
                gpu_memory_per_job_mib=5120, gpu_memory_reserve_mib=4096,
                host_memory_per_job_mib=6144, host_memory_reserve_mib=8192,
                max_admission_utilization=85, poll_seconds=30,
                predecessor=dict(root=str(predecessor),
                                 plan_sha256=digest(predecessor / "plan.json")),
                prediction_gate="Cross-fitted full-return MSE below training-mean MSE",
                seed_reservations="Slurm no queued jobs; neither host has AntMaze "
                "protocols; policy seed21 avoids prior CQL 0/1/11/12",
                primary_metric="Final score at 300k; also best<=300k and last10 mean",
                preparation_order="Two sequential GPU predictor fits, then CQL queue",
                scheduling="At most two own GPU jobs; all external PIDs count")
    write(root / "initial_plan.json", plan)
    write(root / "plan.json", plan)
    print(root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    prepare(args.project.resolve(), args.inputs.resolve(), args.code_revision)
