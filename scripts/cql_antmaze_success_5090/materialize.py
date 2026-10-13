"""Create independent HDF5 reward variants, without modifying any original file."""

import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

from scripts.cql_antmaze_official_5090.common import (
    admissible,
    digest,
    resources,
    verify,
    write,
)
from scripts.cql_antmaze_success_5090.data import partition, uniform

def load_raw(path):
    with h5py.File(path, "r") as handle:
        return {key: handle[key][:] for key in ["observations", "actions", "rewards",
                                                "terminals", "timeouts"]}


def materialize(root, plan):
    verify(root, plan)
    for name in plan["controls"]:
        job = next(j for j in plan["jobs"] if j["dataset_key"] == name)
        original = root / "datasets" / job["dataset"]
        raw = load_raw(original)
        part = partition(raw)
        if not np.array_equal(raw["rewards"], raw["terminals"]):
            raise ValueError("Dataset differs from the audited binary-success setting")
        if not np.isin(part["returns"], [0, 1]).all():
            raise ValueError("Success fragments must have return zero or one")
        work = root / "predictor" / name
        work.mkdir()
        np.savez_compressed(work / "partition.npz", **part)
        values = {"uniform": uniform(part)}
        del raw
        while True:
            resource = resources()
            if admissible(plan, resource, []):
                break
            write(root / "pipeline_status.json", dict(stage="waiting_for_predictor_slot",
                                                      dataset=name, resources=resource))
            time.sleep(plan["poll_seconds"])
        write(root / "pipeline_status.json", dict(stage="fitting_predictor",
                                                  dataset=name,
                                                  resources=resource))
        subprocess.run([sys.executable, "-m", "scripts.cql_antmaze_success_5090.fit",
                        "--root", str(root), "--dataset", name], check=True)
        with np.load(work / "fit/attribution.npz") as stored:
            values["shapley"] = stored["rewards"]
        for arm, reward in values.items():
            path = root / "reward_datasets" / arm / job["dataset"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            shutil.copy2(original, path)
            with h5py.File(path, "r+") as handle:
                result = handle["rewards"][:]
                result[part["index"]] = reward
                handle["rewards"][:] = result
            np.save(work / f"{arm}_rewards.npy", reward)
            plan["frozen_sha256"][str(path.relative_to(root))] = digest(path)
        summary = dict(fragments=len(part["returns"]), transitions=len(part["index"]),
                       return_one_fragments=int(part["returns"].sum()),
                       one_step_fragments=int((np.diff(part["offsets"]) == 1).sum()),
                       parent_timeout_groups=len(np.unique(part["groups"])),
                       skipped_empty_fragments=part["skipped_empty_fragments"])
        write(work / "partition_audit.json", summary)
        for path in work.rglob("*"):
            if path.is_file():
                plan["frozen_sha256"][str(path.relative_to(root))] = digest(path)
        write(root / "plan.json", plan)
        print(name, summary, flush=True)
    plan["rewards_frozen"] = True
    write(root / "plan.json", plan)
    verify(root, plan)
