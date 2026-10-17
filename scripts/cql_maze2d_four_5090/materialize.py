"""Keep official transitions; delay their totals or redistribute those same totals."""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

from scripts.cql_antmaze_official_5090.common import digest, verify, write
from scripts.cql_antmaze_success_5090.data import partition, uniform
from scripts.cql_antmaze_success_5090.materialize import load_raw

def delayed(part):
    result = np.zeros(len(part["index"]), dtype=np.float32)
    result[part["offsets"][1:] - 1] = part["returns"]
    return result


def predictor_input(raw, part):
    # No raw reward array, reward positions, or success labels cross this boundary.
    return dict(features=np.concatenate([raw["observations"][part["index"]],
                                         raw["actions"][part["index"]]], 1
                                        ).astype(np.float32),
                offsets=part["offsets"], returns=part["returns"], groups=part["groups"])


def materialize(root, plan):
    verify(root, plan)
    job = plan["jobs"][0]
    original = root / "datasets" / job["dataset"]
    raw = load_raw(original)
    if np.any(raw["terminals"]):
        raise ValueError("Expected all-false Maze2D terminals")
    part = partition(raw)
    work = root / "predictor/umaze"
    work.mkdir()
    np.savez_compressed(work / "partition.npz", **part)
    np.savez_compressed(work / "inputs.npz", **predictor_input(raw, part))
    values = dict(original=raw["rewards"][part["index"]].copy(),
                  delayed=delayed(part), uniform=uniform(part))
    summary = dict(fragments=len(part["returns"]), transitions=len(part["index"]),
                   positive_fragments=int((part["returns"] > 0).sum()),
                   zero_fragments=int((part["returns"] == 0).sum()),
                   total_reward=float(part["returns"].sum()),
                   length_min=int(np.diff(part["offsets"]).min()),
                   length_max=int(np.diff(part["offsets"]).max()),
                   mean_length=float(np.diff(part["offsets"]).mean()),
                   skipped_empty_fragments=int(part["skipped_empty_fragments"]))
    write(work / "partition_audit.json", summary)
    del raw
    write(root / "pipeline_status.json", dict(stage="fitting_predictor_cpu",
                                              dataset="umaze", partition=summary))
    # CPU-only fitting can progress while the shared GPU is occupied.
    subprocess.run([sys.executable, "-m", "scripts.cql_maze2d_four_5090.fit",
                    "--root", str(root)],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=True)
    with np.load(work / "fit/attribution.npz") as stored:
        values["shapley"] = stored["rewards"]
    for arm, rewards in values.items():
        totals = np.add.reduceat(rewards.astype(np.float64), part["offsets"][:-1])
        np.testing.assert_allclose(totals, part["returns"], atol=1e-4, rtol=0)
        np.save(work / f"{arm}_rewards.npy", rewards)
        if arm == "original":
            continue
        path = root / "reward_datasets" / arm / job["dataset"]
        path.parent.mkdir(parents=True, exist_ok=False)
        shutil.copy2(original, path)
        with h5py.File(path, "r+") as handle:
            result = handle["rewards"][:]
            result[part["index"]] = rewards
            handle["rewards"][:] = result
        plan["frozen_sha256"][str(path.relative_to(root))] = digest(path)
    for path in work.rglob("*"):
        if path.is_file():
            plan["frozen_sha256"][str(path.relative_to(root))] = digest(path)
    plan["rewards_frozen"] = True
    write(root / "plan.json", plan)
    verify(root, plan)
    print("REWARDS_FROZEN", summary, flush=True)
