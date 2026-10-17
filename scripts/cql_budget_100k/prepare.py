"""Replace stopped, entirely unstarted 300k queues with independent 100k queues."""

import argparse
import copy
import fcntl
import hashlib
import shutil
from contextlib import ExitStack
from pathlib import Path

import yaml

from scripts.cql_delayed.common import digest, read, verify, write

OLD_NAMES = (
    "cql-repeats-seeds11-12-300k-5090-20261007",
    "cql-dense-seeds1-11-12-300k-5090-20261007",
)


def require_pending(root, kind):
    plan = read(root / "plan.json")
    status = read(root / ("queue/status.json" if kind == "repeats"
                          else "dependency/status.json"))
    if (plan["updates"] != 300000
            or set(status["jobs"]) != {j["id"] for j in plan["jobs"]}
            or not all(s["state"] == "queued" for s in status["jobs"].values())):
        raise ValueError("Only wholly queued 300k campaigns can be replaced")
    for job in plan["jobs"]:
        run = root / job["directory"]
        for path in [run / job["arm"] / "training",
                     run / "validation" / job["arm"] / "preflight",
                     run / "wandb_sync"]:
            if path.exists():
                raise ValueError(f"An experiment already started: {path}")
    return plan


def clone_run(previous, root, job, group, revision):
    p = copy.deepcopy(verify(previous))
    if digest(previous / "protocol.json") != job["protocol_sha256"]:
        raise ValueError("Historical queue protocol changed")
    arm = job["arm"]
    spec = p["runs"][arm]
    old_id = spec["wandb_id"]
    if spec["updates"] != 300000 or spec["seed"] != job["seed"]:
        raise ValueError("Expected the queued 300k seed")
    for name, expected in spec["data_sha256"].items():
        if digest(previous / arm / name) != expected:
            raise ValueError("Historical reward data changed")
    root.mkdir(parents=True, exist_ok=False)
    for name in p["source_sha256"]:
        destination = root / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous / "source" / name, destination)
    (root / arm).mkdir()
    for name in spec["data_sha256"]:
        shutil.copy2(previous / arm / name, root / arm / name)
    config_path = root / "source" / spec["config"]
    config = yaml.safe_load(config_path.read_text())
    if config["max_timesteps"] != 300000:
        raise ValueError("Expected a 300k training configuration")
    spec.update(updates=100000,
                wandb_name=spec["wandb_name"].replace("-300k-", "-100k-"),
                wandb_id=hashlib.sha256(f"{group}/{root.name}".encode()).hexdigest()[:12])
    config.update(max_timesteps=100000, name=spec["wandb_name"], group=group)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    old_hashes = p["source_sha256"]
    p["source_sha256"] = {n: digest(root / "source" / n) for n in old_hashes}
    for name, expected in old_hashes.items():
        if name != spec["config"] and p["source_sha256"][name] != expected:
            raise ValueError(f"Unrequested source change: {name}")
    p.update(wandb_group=group, budget_revision={
        "reason": "User explicitly requested all queued runs use 100k updates",
        "previous_root": str(previous), "previous_wandb_id": old_id,
        "previous_protocol_sha256": digest(previous / "protocol.json"),
        "preparation_code_revision": revision})
    p["migration"]["training_budget"] = "100000 updates, user-requested before start"
    p["changes"].append("Queued budget 300k->100k; independent outputs and logging IDs")
    write(root / "protocol.json", p)
    validation = root / "validation"
    (validation / arm).mkdir(parents=True)
    (validation / "source").symlink_to("../source", target_is_directory=True)
    for name in spec["data_sha256"]:
        (validation / arm / name).symlink_to(f"../../{arm}/{name}")
    write(validation / "protocol.json", p)
    return dict(job, wandb_name=spec["wandb_name"], wandb_id=spec["wandb_id"],
                protocol_sha256=digest(root / "protocol.json"))


def prepare(project, revision):
    results = project / "results"
    old_roots = [results / name for name in OLD_NAMES]
    new_roots = [results / name.replace("-300k-", "-100k-") for name in OLD_NAMES]
    with ExitStack() as stack:
        # Both old managers must already have been stopped after checking that
        # they own no training children. Locks stay held throughout replacement.
        for root, relative in zip(old_roots, ["queue/lock", "dependency/lock"]):
            lock = stack.enter_context((root / relative).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plans = [require_pending(root, kind)
                 for root, kind in zip(old_roots, ["repeats", "dense"])]
        if [len(p["jobs"]) for p in plans] != [8, 6]:
            raise ValueError("Expected the two authorized 8-run and 6-run queues")
        if any(root.exists() for root in new_roots):
            raise FileExistsError("Preserve any previous 100k preparation attempt")
        for old_root, new_root, original in zip(old_roots, new_roots, plans):
            new_root.mkdir()
            plan = copy.deepcopy(original)
            plan["jobs"] = [clone_run(old_root / j["directory"],
                                      new_root / j["directory"], j, new_root.name,
                                      revision) for j in original["jobs"]]
            plan.update(updates=100000,
                        primary_metric="mean of last 10 evaluations, 55k through 100k",
                        secondary_metrics=["score at 100k", "best within first 100k"],
                        budget_revision=dict(previous_root=str(old_root),
                                             previous_plan_sha256=digest(
                                                 old_root / "plan.json"),
                                             code_revision=revision))
            if "predecessor" in plan:
                plan.update(predecessor=str(new_roots[0]),
                            predecessor_plan_sha256=digest(new_roots[0] / "plan.json"))
            write(new_root / "plan.json", plan)
        for old_root, new_root in zip(old_roots, new_roots):
            # Keep historical plans and status files intact; add an explicit receipt.
            receipt = old_root / "superseded_by_100k.json"
            if receipt.exists():
                raise FileExistsError("Old queue already superseded")
            write(receipt, dict(replacement=str(new_root),
                                replacement_plan_sha256=digest(new_root / "plan.json"),
                                code_revision=revision,
                                reason="User-requested 100k budget before any training"))
            print(new_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    prepare(args.project.resolve(), args.code_revision)
