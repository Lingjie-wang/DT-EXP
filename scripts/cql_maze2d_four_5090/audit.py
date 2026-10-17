"""Assert equality of every non-reward HDF5 field and the real official loader."""

import argparse
from pathlib import Path

import d4rl
import gym
import h5py
import numpy as np

from scripts.cql_antmaze_official_5090.common import read, verify, write
from scripts.cql_antmaze_success_5090.data import partition
from scripts.cql_antmaze_success_5090.materialize import load_raw

def audit(root, job_id):
    plan = read(root / "plan.json")
    if not plan.get("rewards_frozen"):
        raise ValueError("Rewards have not been frozen")
    verify(root, plan)
    job = next(j for j in plan["jobs"] if j["id"] == job_id)
    package = root / "dependencies/d4rl/d4rl/__init__.py"
    if Path(d4rl.__file__).resolve() != package:
        raise ValueError("Require the pinned D4RL source")
    original = root / "datasets" / job["dataset"]
    modified = root / job["dataset_dir"] / job["dataset"]
    with h5py.File(original, "r") as before, h5py.File(modified, "r") as after:
        names = []
        before.visititems(lambda name, obj: names.append(name)
                          if isinstance(obj, h5py.Dataset) else None)
        for key in names:
            if key != "rewards":
                np.testing.assert_array_equal(before[key][()], after[key][()])
    raw = load_raw(original)
    part = partition(raw)
    env = gym.make(job["env"])
    actual = d4rl.qlearning_dataset(env)
    index = part["index"]
    for key in ["observations", "actions", "terminals"]:
        np.testing.assert_array_equal(actual[key], raw[key][index])
    np.testing.assert_array_equal(actual["next_observations"],
                                  raw["observations"][index + 1])
    rewards = (raw["rewards"][index] if job["arm"] == "original" else np.load(
        root / "predictor" / job["dataset_key"] / f"{job['arm']}_rewards.npy"))
    np.testing.assert_array_equal(actual["rewards"], rewards)
    totals = np.add.reduceat(rewards.astype(np.float64), part["offsets"][:-1])
    np.testing.assert_allclose(totals, part["returns"], atol=1e-4, rtol=0)
    if np.any(actual["terminals"]):
        raise ValueError("Maze2D must retain its all-false terminal flags")
    if job["arm"] == "delayed":
        nonlast = np.ones(len(rewards), dtype=bool)
        nonlast[part["offsets"][1:] - 1] = False
        np.testing.assert_array_equal(rewards[nonlast], 0)
    result = dict(full_array_comparison_passed=True, nonreward_hdf5_equal=True,
                  transitions=len(index),
                  success_terminals=int(actual["terminals"].sum()),
                  fragments=len(totals), max_return_error=float(np.abs(
                      totals - part["returns"]).max()),
                  reward_min=float(rewards.min()), reward_max=float(rewards.max()),
                  negative_reward_fraction=float((rewards < 0).mean()),
                  reward_normalization=False)
    write(root / job_id / "data_audit.json", result)
    env.close()
    print(job_id, result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.job)
