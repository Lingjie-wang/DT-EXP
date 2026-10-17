"""Prepare original-reward CQL controls matched to the delayed/Shapley pilots."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

from algorithms.offline.cql_delayed_data import terminal_return_dataset
from scripts.cql_delayed.common import digest, read, verify, write

SEEDS = (1, 11, 12)
UPDATES = 300000
CAMPAIGN = "cql-dense-seeds1-11-12-300k-5090-20261007"
PREDECESSOR = "cql-repeats-seeds11-12-300k-5090-20261007"
REWARD_MODE = "original-dense-matched-v1"


def original_rewards(raw, baseline):
    """Verify the entire historical dataset before replacing only its rewards."""
    delayed, audit = terminal_return_dataset(raw)
    if set(delayed) != set(baseline):
        raise ValueError("Baseline dataset fields differ")
    for key, value in delayed.items():
        np.testing.assert_array_equal(value, baseline[key], err_msg=key)
    rewards = np.asarray(raw["rewards"], dtype=np.float32).reshape(-1).copy()
    data = dict(delayed, rewards=rewards)
    ends = np.flatnonzero(data["terminals"])
    starts = np.r_[0, ends[:-1] + 1]
    totals = np.array([rewards[a:b + 1].astype(np.float64).sum()
                       for a, b in zip(starts, ends)])
    audit.update(
        reward_mode=REWARD_MODE, matched_nonreward_fields=True,
        rewards_equal_original_float32=True,
        nonzero_rewards=int(np.count_nonzero(rewards)),
        nonterminal_nonzero_rewards=int(np.count_nonzero(
            rewards[data["terminals"] == 0])),
        max_episode_return_error_vs_delayed_float32=float(
            np.abs(totals - delayed["rewards"][ends]).max()))
    return data, audit


def clone_run(previous, root, arm, seed, group, prepared, revision):
    if seed not in SEEDS or arm not in {"medium", "medium_replay"}:
        raise ValueError("Only the requested three-seed, two-dataset dense controls")
    p = verify(previous)
    old = p["runs"][arm]
    if old["seed"] != 1 or old["updates"] != 1000000:
        raise ValueError("Expected the original 5090 pilot")
    root.mkdir(parents=True, exist_ok=False)
    for name in p["source_sha256"]:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous / "source" / name, target)
    (root / arm).mkdir()
    for name in ("dataset.npz", "audit.json"):
        shutil.copy2(prepared / name, root / arm / name)
    config_path = root / "source" / old["config"]
    config = yaml.safe_load(config_path.read_text())
    if (config["seed"] != 1 or int(config["max_timesteps"]) != 1000000
            or config["normalize_reward"] or config["eval_freq"] != 5000
            or config["n_episodes"] != 10):
        raise ValueError("Unexpected pilot configuration")
    tag = "HCM" if arm == "medium" else "HCMR"
    name = f"CQL-CORL-{tag}-dense-matched-v1-seed{seed}-300k-5090"
    config.update(seed=seed, max_timesteps=UPDATES, name=name, group=group)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    spec = dict(env=old["env"], seed=seed, updates=UPDATES, eval_every=5000,
                eval_episodes=10, config=old["config"], reward_mode=REWARD_MODE,
                wandb_name=name,
                wandb_id=hashlib.sha256(f"{group}/{root.name}".encode()).hexdigest()[:12],
                reference_wandb_id=old["wandb_id"],
                data_sha256={n: digest(root / arm / n)
                             for n in ("dataset.npz", "audit.json")})
    source_hashes = {n: digest(root / "source" / n) for n in p["source_sha256"]}
    for path, expected in p["source_sha256"].items():
        if path != old["config"] and source_hashes[path] != expected:
            raise ValueError(f"Frozen source changed: {path}")
    protocol = dict(
        method="CORL-CQL-original-dense-matched-v1", reward_mode=REWARD_MODE,
        runs={arm: spec}, source_sha256=source_hashes,
        preparation_code_revision=revision, retained=copy.deepcopy(p["retained"]),
        migration={"reference_root": str(previous.resolve()),
                   "reference_protocol_sha256": digest(previous / "protocol.json"),
                   "target_host": "5090", "policy_and_eval_seed": seed,
                   "source_python_changed": False,
                   "rewards": "Original HDF5 per-transition rewards, float32"},
        changes=["Original step rewards replace terminal totals",
                 "Paired policy/eval seed; fixed 300k updates; new logging IDs",
                 "Retain episodic timeout/terminal handling of comparison arms"],
        wandb_entity=p["wandb_entity"], wandb_project=p["wandb_project"],
        wandb_group=group)
    write(root / "protocol.json", protocol)
    validation = root / "validation"
    (validation / arm).mkdir(parents=True)
    (validation / "source").symlink_to("../source", target_is_directory=True)
    for name in spec["data_sha256"]:
        (validation / arm / name).symlink_to(f"../../{arm}/{name}")
    write(validation / "protocol.json", protocol)
    return spec


def prepare(project, root, raw_paths, revision):
    previous = project / "results/cql-corl-delayed-hc-seed1-5090-20261007"
    baseline = verify(previous)
    predecessor = project / "results" / PREDECESSOR
    predecessor_plan = read(predecessor / "plan.json")
    if len(predecessor_plan["jobs"]) != 8:
        raise ValueError("Expected the existing eight-run priority queue")
    for path in (project / "results").rglob("protocol.json"):
        if "source" in path.parts or "validation" in path.parts:
            continue
        p = read(path)
        for spec in p.get("runs", {}).values():
            if (spec.get("seed") in SEEDS and spec.get("reward_mode",
                    p.get("reward_mode")) == REWARD_MODE):
                raise ValueError(f"Original-reward seed already reserved: {path}")
    root.mkdir(parents=True, exist_ok=False)
    for arm, original in raw_paths.items():
        old_audit = read(previous / arm / "audit.json")
        if digest(original) != old_audit["original_sha256"]:
            raise ValueError(f"Raw HDF5 does not match baseline: {arm}")
        for name, expected in baseline["runs"][arm]["data_sha256"].items():
            if digest(previous / arm / name) != expected:
                raise ValueError(f"Historical data changed: {arm}/{name}")
        with h5py.File(original, "r") as source:
            raw = {k: source[k][:] for k in ["observations", "actions", "rewards",
                                           "terminals", "timeouts", "next_observations"]
                   if k in source}
        with np.load(previous / arm / "dataset.npz") as stored:
            data, audit = original_rewards(raw, dict(stored))
        audit.update(original_sha256=digest(original), original_path=str(original))
        prepared = root / "prepared" / arm
        prepared.mkdir(parents=True)
        np.savez(prepared / "dataset.npz", **data)
        write(prepared / "audit.json", audit)
        del raw, data
    jobs = []
    for seed in SEEDS:
        for arm in ("medium", "medium_replay"):
            directory = f"seed{seed}-{arm}-dense"
            spec = clone_run(previous, root / directory, arm, seed, root.name,
                             root / "prepared" / arm, revision)
            jobs.append(dict(
                id=directory, directory=directory, arm=arm, runner="cql_delayed",
                seed=seed, pair=f"seed{seed}-dense", wandb_name=spec["wandb_name"],
                wandb_id=spec["wandb_id"],
                protocol_sha256=digest(root / directory / "protocol.json")))
    plan = dict(jobs=jobs, seeds=list(SEEDS), updates=UPDATES, max_gpu_processes=4,
                gpu_memory_per_job_mib=3072, gpu_memory_reserve_mib=2048,
                host_memory_reserve_mib=8192, poll_seconds=30,
                predecessor=str(predecessor), predecessor_jobs=[
                    j["id"] for j in predecessor_plan["jobs"]],
                predecessor_plan_sha256=digest(predecessor / "plan.json"),
                reward_mode=REWARD_MODE,
                primary_metric="mean of last 10 evaluations at the common 300k budget",
                secondary_metrics=["score at 300k", "best score within first 300k"])
    write(root / "plan.json", plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--medium-replay", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    prepare(args.project.resolve(), args.project / "results" / CAMPAIGN,
            dict(medium=args.medium, medium_replay=args.medium_replay),
            args.code_revision)
