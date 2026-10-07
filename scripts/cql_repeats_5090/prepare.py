"""Prepare paired policy-seed repeats without changing frozen CQL or rewards."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from scripts.cql_delayed.common import digest, read, verify, write

SEEDS = (11, 12)
UPDATES = 300000
CAMPAIGN = "cql-repeats-seeds11-12-300k-5090-20261007"
REFERENCES = {
    "medium": "cql-corl-delayed-hc-seed1-5090-20261007",
    "medium_replay": "cql-corl-delayed-hc-seed1-5090-20261007",
    "medium_shapley": "cql-shapley-hcm-v1-seed1-5090-20261007",
    "medium_replay_shapley": "cql-shapley-hcmr-v1-seed1-5090-20261007",
}


def clone_run(previous, root, arm, seed, group):
    if seed not in SEEDS:
        raise ValueError("This campaign reserves new policy/evaluation seeds 11 and 12")
    p = copy.deepcopy(verify(previous))
    if arm not in {"medium", "medium_replay", "shapley"}:
        raise ValueError("Only currently active reward arms may be repeated")
    old = p["runs"][arm]
    if old["seed"] != 1 or old["updates"] != 1000000:
        raise ValueError("Expected the frozen 5090 seed-1 pilot")
    if (previous / arm / "user_cancellation.json").exists():
        raise ValueError("Refuse a cancelled arm")
    if arm == "shapley" and not p.get("shapley_gate", {}).get("passed"):
        raise ValueError("Shapley prediction gate must have passed")
    for name, expected in old["data_sha256"].items():
        if digest(previous / arm / name) != expected:
            raise ValueError(f"Historical data changed: {arm}/{name}")
    root.mkdir(parents=True, exist_ok=False)
    for name in p["source_sha256"]:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous / "source" / name, target)
    (root / arm).mkdir()
    for name in old["data_sha256"]:
        shutil.copy2(previous / arm / name, root / arm / name)
    spec = copy.deepcopy(old)
    name = old["wandb_name"].replace("seed1-1M", f"seed{seed}-300k")
    spec.update(seed=seed, updates=UPDATES, wandb_name=name,
                wandb_id=hashlib.sha256(f"{group}/{root.name}".encode()).hexdigest()[:12],
                reference_wandb_id=old["wandb_id"])
    config_path = root / "source" / spec["config"]
    config = yaml.safe_load(config_path.read_text())
    if (config["seed"] != 1 or int(config["max_timesteps"]) != 1000000
            or config["eval_freq"] != 5000 or config["n_episodes"] != 10):
        raise ValueError("Unexpected pilot training/evaluation configuration")
    config.update(seed=seed, max_timesteps=UPDATES, name=name, group=group)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    p.update(runs={arm: spec}, wandb_group=group,
             migration={"reference_root": str(previous.resolve()),
                        "reference_protocol_sha256": digest(previous / "protocol.json"),
                        "target_host": "5090", "policy_and_eval_seed": seed,
                        "source_python_changed": False,
                        "attribution": "Reuse frozen rewards; no new attribution fit",
                        "training_budget": "300000 updates fixed from seed-1 pilot"})
    p["changes"].append(f"Policy/eval seed 1->{seed}; 1M->300k; new logging IDs")
    p["source_sha256"] = {name: digest(root / "source" / name)
                          for name in p["source_sha256"]}
    for name, expected in read(previous / "protocol.json")["source_sha256"].items():
        if name != spec["config"] and p["source_sha256"][name] != expected:
            raise ValueError(f"Unexpected source change: {name}")
    write(root / "protocol.json", p)
    validation = root / "validation"
    (validation / arm).mkdir(parents=True)
    (validation / "source").symlink_to("../source", target_is_directory=True)
    for name in spec["data_sha256"]:
        (validation / arm / name).symlink_to(f"../../{arm}/{name}")
    write(validation / "protocol.json", p)
    return spec


def prepare(project, root):
    # Also catches queued/repeated seeds on this server, including nested campaigns.
    for path in (project / "results").rglob("protocol.json"):
        if "source" in path.parts or "validation" in path.parts:
            continue
        for spec in read(path).get("runs", {}).values():
            if (spec.get("env", "").startswith("halfcheetah")
                    and spec.get("seed") in SEEDS):
                raise ValueError(f"Seed already reserved: {path}")
    root.mkdir(parents=True, exist_ok=False)
    jobs = []
    for seed in SEEDS:
        for dataset in ("medium", "medium_replay"):
            for mode in ("delayed", "shapley"):
                key = dataset if mode == "delayed" else dataset + "_shapley"
                arm = dataset if mode == "delayed" else "shapley"
                directory = f"seed{seed}-{dataset}-{mode}"
                spec = clone_run(project / "results" / REFERENCES[key],
                                 root / directory, arm, seed, root.name)
                jobs.append({"id": directory, "directory": directory, "arm": arm,
                             "runner": "cql_delayed" if mode == "delayed"
                             else "cql_shapley", "seed": seed,
                             "pair": f"seed{seed}-{dataset}",
                             "wandb_name": spec["wandb_name"],
                             "wandb_id": spec["wandb_id"],
                             "protocol_sha256": digest(root / directory
                                                       / "protocol.json")})
    plan = dict(jobs=jobs, seeds=list(SEEDS), updates=UPDATES, max_gpu_processes=4,
                gpu_memory_per_job_mib=3072, gpu_memory_reserve_mib=2048,
                host_memory_reserve_mib=8192, poll_seconds=30,
                reserved_elsewhere={"slurm_cql": [0], "5090_pilot_cql": [1]},
                primary_metric="mean of last 10 evaluations at the common 300k budget",
                secondary_metrics=["score at 300k", "best score within first 300k"],
                attribution="Fixed prepared rewards; independent CQL policy/eval seeds")
    write(root / "plan.json", plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    prepare(args.project.resolve(), args.root or args.project / "results" / CAMPAIGN)
