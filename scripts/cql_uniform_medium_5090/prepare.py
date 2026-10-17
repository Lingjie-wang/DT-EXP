"""Prepare three HCM uniform-return controls, paired with existing CQL seeds."""

import argparse
import copy
import hashlib
import shutil
from pathlib import Path

import numpy as np
import yaml

from scripts.cql_delayed.common import digest, read, verify, write
from scripts.cql_uniform_5090.prepare import uniform_rewards

SEEDS = (1, 11, 12)
UPDATES = 500000
ARM = "medium"
CAMPAIGN = "cql-uniform-hcm-seeds1-11-12-500k-5090-20261008"
REWARD_MODE = "uniform-total-matched-v1"
BASELINE = "cql-corl-delayed-hc-seed1-5090-20261007"
SHAPLEY = "cql-shapley-hcm-v1-seed1-5090-20261007"
PREDECESSOR = "cql-uniform-hcmr-seeds1-11-12-100k-5090-20261008"


def clone_run(previous, root, seed, group, prepared, revision, shapley_id):
    p = copy.deepcopy(verify(previous))
    old = p["runs"][ARM]
    if seed not in SEEDS or old["seed"] != 1 or old["updates"] != 1000000:
        raise ValueError("Expected a paired seed and the frozen seed-1 baseline")
    root.mkdir(parents=True, exist_ok=False)
    for name in p["source_sha256"]:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous / "source" / name, target)
    work = root / ARM
    work.mkdir()
    for name in ("dataset.npz", "audit.json"):
        shutil.copy2(prepared / name, work / name)
    name = f"CQL-CORL-HCM-uniform-matched-v1-seed{seed}-500k-5090"
    config_path = root / "source" / old["config"]
    config = yaml.safe_load(config_path.read_text())
    if (config["env"] != "halfcheetah-medium-v2" or config["seed"] != 1
            or int(config["max_timesteps"]) != 1000000 or config["normalize_reward"]
            or config["eval_freq"] != 5000 or config["n_episodes"] != 10):
        raise ValueError("Unexpected historical CQL configuration")
    config.update(seed=seed, max_timesteps=UPDATES, name=name, group=group)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    spec = dict(old, seed=seed, updates=UPDATES, reward_mode=REWARD_MODE,
                wandb_name=name, paired_shapley_wandb_id=shapley_id,
                wandb_id=hashlib.sha256(f"{group}/{root.name}".encode()).hexdigest()[:12],
                data_sha256={n: digest(work / n) for n in ("dataset.npz", "audit.json")})
    p.update(method="CORL-CQL-uniform-total-matched-v1", reward_mode=REWARD_MODE,
             runs={ARM: spec}, wandb_group=group, preparation_code_revision=revision,
             migration=dict(reference_root=str(previous), target_host="5090",
                            source_python_changed=False, policy_and_eval_seed=seed,
                            training_budget="500k requested; paired comparison at 100k",
                            reference_protocol_sha256=digest(
                                previous / "protocol.json")),
             changes=["Uniformly distribute the same trajectory totals as Shapley",
                      "Only reward, policy/evaluation seed, budget and logging change",
                      "Preserve complete transitions, timeout masks and frozen CQL code",
                      "New campaign; historical seed-0 controls stay cancelled"])
    old_hashes = p["source_sha256"]
    p["source_sha256"] = {n: digest(root / "source" / n) for n in old_hashes}
    for path, expected in old_hashes.items():
        if path != old["config"] and p["source_sha256"][path] != expected:
            raise ValueError(f"Unexpected scientific source change: {path}")
    write(root / "protocol.json", p)
    validation = root / "validation"
    (validation / ARM).mkdir(parents=True)
    (validation / "source").symlink_to("../source", target_is_directory=True)
    for filename in spec["data_sha256"]:
        (validation / ARM / filename).symlink_to(f"../../{ARM}/{filename}")
    write(validation / "protocol.json", p)
    return spec


def prepare(project, root, predictor_input, revision):
    predecessor = project / "results" / PREDECESSOR
    prior_plan = read(predecessor / "plan.json")
    if (prior_plan["seeds"] != list(SEEDS) or prior_plan["updates"] != 100000
            or len(prior_plan["jobs"]) != 3
            or any(job["arm"] != "medium_replay" for job in prior_plan["jobs"])):
        raise ValueError("Expected the existing three-seed HCMR uniform queue")
    previous = project / "results" / BASELINE
    baseline = verify(previous)
    shapley_root = project / "results" / SHAPLEY
    shapley = verify(shapley_root)
    if not shapley.get("shapley_gate", {}).get("passed"):
        raise ValueError("Expected the existing validated Shapley comparison")
    expected = shapley["preparation_sha256"]["predictor/input.npz"]
    if digest(predictor_input) != expected:
        raise ValueError("Predictor labels differ from the actual Shapley inputs")
    for path in (project / "results").rglob("protocol.json"):
        if "source" in path.parts or "validation" in path.parts:
            continue
        for spec in read(path).get("runs", {}).values():
            if (spec.get("env") == "halfcheetah-medium-v2"
                    and spec.get("seed") in SEEDS
                    and "uniform" in str(spec.get("reward_mode", ""))):
                raise ValueError(f"Uniform seed already reserved: {path}")
    for name, expected_hash in baseline["runs"][ARM]["data_sha256"].items():
        if digest(previous / ARM / name) != expected_hash:
            raise ValueError("Frozen baseline data changed")
    with np.load(previous / ARM / "dataset.npz") as stored:
        original = dict(stored)
    with np.load(predictor_input) as stored:
        inputs = dict(stored)
    data, audit = uniform_rewards(original, inputs)
    if audit["episodes"] != 1000:
        raise ValueError("The matched HCM dataset must contain 1000 full trajectories")
    paired = {1: shapley["runs"]["shapley"]["wandb_id"]}
    for seed in (11, 12):
        ref = (project / "results/cql-repeats-seeds11-12-100k-5090-20261007"
               / f"seed{seed}-medium-shapley")
        spec = verify(ref)["runs"]["shapley"]
        if spec["seed"] != seed or spec["updates"] != 100000:
            raise ValueError("Missing matched 100k Shapley seed")
        paired[seed] = spec["wandb_id"]
    root.mkdir(parents=True, exist_ok=False)
    prepared = root / "prepared"
    prepared.mkdir()
    np.savez(prepared / "dataset.npz", **data)
    np.savez(prepared / "totals.npz", returns=inputs["returns"],
             starts=inputs["starts"], ends=inputs["ends"])
    audit.update(predictor_input_sha256=expected,
                 reference_shapley_protocol_sha256=digest(
                     shapley_root / "protocol.json"))
    write(prepared / "audit.json", audit)
    jobs = []
    for seed in SEEDS:
        directory = f"seed{seed}-medium-uniform"
        spec = clone_run(previous, root / directory, seed, root.name,
                         prepared, revision, paired[seed])
        jobs.append(dict(id=directory, directory=directory, seed=seed, arm=ARM,
                         runner="cql_delayed", wandb_name=spec["wandb_name"],
                         wandb_id=spec["wandb_id"],
                         protocol_sha256=digest(root / directory / "protocol.json")))
    write(root / "plan.json", dict(
        jobs=jobs, seeds=list(SEEDS), updates=UPDATES, reward_mode=REWARD_MODE,
        predecessor=dict(root=str(predecessor.resolve()),
                         plan_sha256=digest(predecessor / "plan.json"),
                         require_wandb_verified=True),
        max_parallel=3, max_gpu_processes=9, gpu_memory_per_job_mib=3072,
        gpu_memory_reserve_mib=4096, host_memory_reserve_mib=8192,
        host_memory_per_job_mib=4096, max_admission_utilization=85, poll_seconds=30,
        scheduling_reason="Wait for HCMR training and W&B verification, then admit "
        "one at a time with fresh VRAM, host memory, utilization and PID checks",
        reserved_elsewhere={"slurm_cql_seeds": [0],
                            "paired_5090_comparison_seeds": list(SEEDS)},
        primary_metric="Mean of the last 10 evaluations, 455k through 500k",
        secondary_metrics=["Score at 500k", "Best score within first 500k"],
        matched_comparison="All three seeds: compare only the first 100k with "
        "existing Shapley/delayed/dense runs; use 55k..100k mean, final and best",
        seed1_comparison="Seed 1 delayed/Shapley also support a 500k comparison",
        code_revision=revision))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--predictor-input", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    prepare(project, project / "results" / CAMPAIGN, args.predictor_input.resolve(),
            args.code_revision)
