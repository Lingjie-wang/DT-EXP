"""Preserve the failed launch; reuse frozen rewards in an independent retry."""

import argparse
import copy
import fcntl
import hashlib
import shutil
from pathlib import Path

import numpy as np
import yaml

from scripts.cql_antmaze_5090.data import conserved_rewards
from scripts.cql_delayed.common import digest, read, verify, write
from scripts.cql_repeats_5090.queue import validate_job

def shapley_arrays(previous, job):
    dataset = previous / "datasets" / job["dataset"]
    protocol = verify(dataset)
    for name, expected in protocol["preparation_sha256"].items():
        if digest(dataset / name) != expected:
            raise ValueError("Predictor input changed")
    gate = read(dataset / "predictor/fit/gate.json")
    with np.load(dataset / "predictor/input.npz") as inputs:
        totals = inputs["returns"]
        horizon = inputs["features"].shape[1]
    with np.load(dataset / "predictor/fit/attribution.npz") as attribution:
        np.testing.assert_array_equal(totals, attribution["returns"])
        rewards = attribution["rewards"]
        regenerated = np.stack([conserved_rewards(phi, total, horizon)[0]
                                for phi, total in zip(attribution["contributions"],
                                                      totals)])
        np.testing.assert_array_equal(rewards, regenerated)
    with np.load(dataset / "base_transitions.npz") as stored:
        arrays = dict(stored)
    arrays["rewards"] = rewards.reshape(-1)
    audit = dict(read(dataset / "audit.json"),
                 **read(dataset / "predictor/fit/audit.json"))
    audit.update(matched_nonreward_fields=True, policy_reward_transform="10*r - 5",
                 frozen_attribution_sha256=digest(
                     dataset / "predictor/fit/attribution.npz"),
                 prediction_gate_policy="report-only", gate=gate)
    return arrays, audit, gate


def prepare(previous, root, revision, gate_policy):
    if root.exists():
        raise FileExistsError("Preserve every prior attempt")
    with (previous / "queue/lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(previous / "queue/status.json")
        if state["stage"] != "finished_with_failures":
            raise ValueError("Only recover an inactive, failed initial launch")
        plan = read(previous / "plan.json")
        for job in plan["jobs"]:
            if (previous / job["directory"] / job["arm"] / "training").exists():
                raise ValueError("Do not duplicate any started policy training")
        reference = verify(previous / plan["jobs"][0]["directory"])
        root.mkdir(parents=True)
        source = root / "source"
        names = list(reference["source_sha256"])
        for name in names:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous / "source" / name, target)
        repository = Path(__file__).resolve().parents[2]
        for file in (repository / "scripts/cql_antmaze_retry_5090").glob("*.py"):
            name = str(file.relative_to(repository))
            (source / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, source / name)
            names.append(name)
        jobs, excluded, protocols = [], [], {}
        for original in plan["jobs"]:
            job = copy.deepcopy(original)
            old = previous / job["directory"]
            protocol = copy.deepcopy(verify(old))
            gate = None
            if job["arm"] == "shapley":
                arrays, audit, gate = shapley_arrays(previous, job)
                if not gate["passed"] and gate_policy == "enforce":
                    excluded.append(dict(id=job["id"], reason="prediction_gate_failed",
                                         gate=gate))
                    continue
            else:
                validate_job(previous, job)
                with np.load(old / job["arm"] / "dataset.npz") as data:
                    arrays = dict(data)
                audit = read(old / job["arm"] / "audit.json")
            work = root / job["directory"]
            (work / job["arm"]).mkdir(parents=True)
            (work / "source").symlink_to("../source", target_is_directory=True)
            np.savez(work / job["arm"] / "dataset.npz", **arrays)
            write(work / job["arm"] / "audit.json", audit)
            spec = protocol["runs"][job["arm"]]
            config = yaml.safe_load((old / "source" / spec["config"]).read_text())
            config.update(name=spec["wandb_name"] + "-retry1", group=root.name)
            name = f"configs/antmaze_retry/{job['dataset']}_{job['arm']}.yaml"
            (source / name).parent.mkdir(exist_ok=True)
            (source / name).write_text(yaml.safe_dump(config, sort_keys=False))
            names.append(name)
            identity = hashlib.sha256(
                f"{root.name}/{job['id']}".encode()).hexdigest()[:12]
            spec.update(config=name, wandb_name=config["name"], wandb_id=identity,
                        data_sha256={filename: digest(work / job["arm"] / filename)
                                     for filename in ["dataset.npz", "audit.json"]})
            job.update(config=name, wandb_name=config["name"], wandb_id=identity)
            protocol.update(code_revision=revision, wandb_group=root.name,
                            prediction_gate_policy=gate_policy,
                            retry_of=dict(root=str(old),
                                          protocol_sha256=digest(old / "protocol.json")))
            if gate is not None:
                protocol["shapley_gate"] = gate
                protocol["changes"].append(
                    "Run original frozen Shapley rewards; failed prediction gate "
                    "is reported without refitting or replacing rewards")
            protocol["changes"].append("Module invocation avoids stdlib queue shadowing")
            protocols[job["id"]] = protocol
            jobs.append(job)
        hashes = {name: digest(source / name) for name in names}
        for job in jobs:
            work = root / job["directory"]
            protocol = protocols[job["id"]]
            protocol["source_sha256"] = hashes
            write(work / "protocol.json", protocol)
            validation = work / "validation"
            (validation / job["arm"]).mkdir(parents=True)
            (validation / "source").symlink_to("../../source", target_is_directory=True)
            for name in ["dataset.npz", "audit.json"]:
                target = validation / job["arm"] / name
                target.symlink_to(f"../../{job['arm']}/{name}")
            write(validation / "protocol.json", protocol)
            job["protocol_sha256"] = digest(work / "protocol.json")
        plan.update(jobs=jobs, excluded_jobs=excluded, code_revision=revision,
                    retry_of=str(previous), prediction_gate_policy=gate_policy,
                    preparation_order=(
                        "Reuse already fitted predictors and frozen rewards"))
        write(root / "initial_plan.json", plan)
        write(root / "plan.json", plan)
        print(root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--prediction-gate", choices=["enforce", "report-only"],
                        required=True)
    args = parser.parse_args()
    prepare(args.previous.resolve(), args.root.resolve(), args.code_revision,
            args.prediction_gate)
