"""Prepare medium Shapley inputs without altering the medium-replay pilot."""

import argparse
import copy
import hashlib
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

from algorithms.offline.cql_delayed_data import terminal_return_dataset
from algorithms.offline.shapley_redistribution import folds
from scripts.cql_delayed.common import digest, read, verify, write

def predictor_input(raw, previous):
    data, audit = terminal_return_dataset(raw)
    for key in data:
        np.testing.assert_array_equal(data[key], previous[key])
    ends = np.flatnonzero(data["terminals"])
    starts = np.r_[0, ends[:-1] + 1]
    if len(ends) != 1000 or not np.all(ends - starts + 1 == 1000):
        raise ValueError("Medium requires 1000 complete 1000-step trajectories")
    totals = np.array([raw["rewards"][a:b + 1].astype(np.float64).sum()
                       for a, b in zip(starts, ends)])
    inputs = dict(features=np.concatenate(
        [data["observations"], data["actions"]], axis=1).reshape(1000, 1000, -1),
        returns=totals, starts=starts, ends=ends)
    return inputs, {key: value for key, value in data.items() if key != "rewards"}, audit


def medium_config(baseline_config, group):
    if (baseline_config["env"] != "halfcheetah-medium-v2"
            or baseline_config["seed"] != 1):
        raise ValueError("Expected the medium seed 1 delayed baseline")
    config = copy.deepcopy(baseline_config)
    config.update(name="CQL-CORL-HCM-shapley20x128-v1-seed1-1M-5090", group=group)
    return config


def prepare(root, baseline, reference, hdf5):
    if root.exists():
        raise FileExistsError("Preserve existing experiment directory")
    base, ref = verify(baseline), verify(reference)
    spec = base["runs"]["medium"]
    for name, expected in spec["data_sha256"].items():
        if digest(baseline / "medium" / name) != expected:
            raise ValueError(f"Medium baseline data changed: {name}")
    audit = read(baseline / "medium/audit.json")
    if digest(hdf5) != audit["original_sha256"]:
        raise ValueError("Raw HDF5 differs from medium delayed baseline")
    algorithm = "algorithms/offline/cql.py"
    if base["source_sha256"][algorithm] != ref["source_sha256"][algorithm]:
        raise ValueError("Baseline and reference CQL algorithms differ")
    config = medium_config(yaml.safe_load(
        (baseline / "source" / spec["config"]).read_text()), root.name)
    with h5py.File(hdf5, "r") as source:
        raw = {k: source[k][:] for k in ["observations", "actions", "rewards",
                                       "terminals", "timeouts", "next_observations"]
               if k in source}
    with np.load(baseline / "medium/dataset.npz") as previous:
        inputs, transitions, matched_audit = predictor_input(raw, previous)
    root.mkdir(parents=True)
    (root / "predictor").mkdir()
    (root / "shapley").mkdir()
    np.savez(root / "predictor/input.npz", **inputs)
    write(root / "predictor/folds.json", list(folds(1000)))
    np.savez(root / "base_transitions.npz", **transitions)
    write(root / "baseline_match.json", matched_audit)
    # The generic fitting and training computations are copied byte for byte.
    sources = [name for name in ref["source_sha256"] if name.endswith(".py")]
    for name in sources:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference / "source" / name, target)
    entry = "scripts/cql_shapley_medium_5090/prepare.py"
    destination = root / "source" / entry
    destination.parent.mkdir(parents=True)
    shutil.copy2(__file__, destination)
    sources.append(entry)
    config_path = "configs/shapley.yaml"
    (root / "source/configs").mkdir(exist_ok=True)
    (root / "source" / config_path).write_text(yaml.safe_dump(config))
    sources.append(config_path)
    run = {key: spec[key] for key in
           ["env", "seed", "updates", "eval_every", "eval_episodes"]}
    run.update(config=config_path, reward_mode="shapley20x128-v1",
               wandb_name=config["name"], wandb_id=hashlib.sha256(
                   f"{root.name}/shapley".encode()).hexdigest()[:12])
    p = dict(
        runs={"shapley": run}, method="CORL-CQL-predictive-Shapley-v1",
        source_sha256={name: digest(root / "source" / name) for name in sources},
        preparation_sha256={name: digest(root / name) for name in
                            ["predictor/input.npz", "predictor/folds.json",
                             "base_transitions.npz"]},
        original_hdf5_sha256=audit["original_sha256"],
        delayed_baseline=dict(root=str(baseline), **spec),
        attribution=copy.deepcopy(ref["attribution"]), retained=base["retained"],
        changes=["medium dataset: 1000 complete trajectories; replay had 202",
                 "new medium cross-fit; unchanged segment/permutation/model settings",
                 "seed 1 CQL; only logging names differ from medium delayed config"],
        migration=dict(reference_root=str(reference), target_host="5090",
                       reference_protocol_sha256=digest(reference / "protocol.json"),
                       baseline_protocol_sha256=digest(baseline / "protocol.json"),
                       policy_and_eval_seed=1, scientific_python_changed=False,
                       attribution="Refit medium; same fold and attribution seeds"),
        wandb_entity=base["wandb_entity"], wandb_project=base["wandb_project"],
        wandb_group=root.name)
    write(root / "protocol.json", p)
    print(f"Prepared {root}: 1000 medium trajectories, seed 1", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "baseline", "reference", "hdf5"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(*(getattr(args, key).resolve()
              for key in ["root", "baseline", "reference", "hdf5"]))
