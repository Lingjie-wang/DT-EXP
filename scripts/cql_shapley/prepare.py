"""Freeze a matched CQL pilot; keep per-step oracle rewards out of predictor input."""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import yaml
from common import digest, read, write

from algorithms.offline.cql_delayed_data import terminal_return_dataset
from algorithms.offline.shapley_redistribution import conserved_rewards, folds

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repository = Path(__file__).resolve().parents[2]
    baseline = read(args.baseline / "protocol.json")
    old_audit = read(args.baseline / "medium_replay/audit.json")
    if digest(args.hdf5) != old_audit["original_sha256"]:
        raise ValueError("Raw dataset does not match the delayed baseline")
    root.mkdir(parents=True, exist_ok=False)
    sources = ["algorithms/offline/cql.py", "algorithms/offline/cql_delayed_data.py",
               "algorithms/offline/shapley_redistribution.py",
               "configs/offline/cql/halfcheetah/medium_replay_v2.yaml"]
    sources += [str(path.relative_to(repository)) for path in
                (repository / "scripts/cql_shapley").iterdir() if path.is_file()]
    for relative in sources:
        target = root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, target)
    for relative in ["algorithms/offline/cql.py",
                     "configs/offline/cql/halfcheetah/medium_replay_v2.yaml"]:
        if digest(root / "source" / relative) != baseline["source_sha256"][relative]:
            raise ValueError(f"CQL scientific source differs from baseline: {relative}")
    with h5py.File(args.hdf5, "r") as source:
        raw = {k: source[k][:] for k in ["observations", "actions", "rewards",
                                       "terminals", "timeouts", "next_observations"]
               if k in source}
    data, audit = terminal_return_dataset(raw)
    with np.load(args.baseline / "medium_replay/dataset.npz") as previous:
        for key in data:
            np.testing.assert_array_equal(data[key], previous[key])
    ends = np.flatnonzero(data["terminals"])
    starts = np.r_[0, ends[:-1] + 1]
    lengths = ends - starts + 1
    if len(ends) != 202 or not np.all(lengths == 1000):
        raise ValueError("Pilot is specified for 202 complete 1000-step trajectories")
    totals = np.array([raw["rewards"][a:b + 1].astype(np.float64).sum()
                       for a, b in zip(starts, ends)])
    predictor = root / "predictor"
    predictor.mkdir()
    # No dense rewards or next observations are present in this artifact.
    np.savez(predictor / "input.npz",
             features=np.concatenate([data["observations"], data["actions"]], axis=1)
             .reshape(202, 1000, -1), returns=totals,
             starts=starts, ends=ends)
    write(predictor / "folds.json", list(folds(202)))
    np.savez(root / "base_transitions.npz",
             **{k: v for k, v in data.items() if k != "rewards"})
    runs = {}
    labels = dict(uniform="uniform-v1", shapley="shapley20x128-v1",
                  dense="dense-matched-v1")
    for arm, label in labels.items():
        work = root / arm
        work.mkdir()
        config_path = f"configs/{arm}.yaml"
        configuration = yaml.safe_load((root / "source" / sources[3]).read_text())
        name = f"CQL-CORL-HCMR-{label}-seed0-1M"
        configuration.update(project="CORL-DDR", group=root.name, name=name)
        destination = root / "source" / config_path
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(yaml.safe_dump(configuration))
        sources.append(config_path)
        runs[arm] = dict(env=configuration["env"], seed=0, updates=1000000,
                         eval_every=5000, eval_episodes=10, reward_mode=label,
                         config=config_path, wandb_name=name,
                         wandb_id=hashlib.sha256(f"{root.name}/{arm}".encode())
                         .hexdigest()[:12])
        if arm == "shapley":
            continue
        rewards = (raw["rewards"].astype(np.float32).reshape(-1) if arm == "dense"
                   else np.concatenate([conserved_rewards(np.zeros(20), r)[0]
                                        for r in totals]))
        np.savez(work / "dataset.npz", **dict(data, rewards=rewards))
        errors = rewards.reshape(202, 1000).astype(np.float64).sum(1) - totals
        assert np.max(np.abs(errors)) < 1e-3
        write(work / "audit.json", dict(
            audit, reward_mode=label, original_sha256=digest(args.hdf5),
            max_return_error=float(np.abs(errors).max()),
            nonzero_rewards=int(np.count_nonzero(rewards)),
            nonterminal_nonzero_rewards=int(np.count_nonzero(
                rewards[data["terminals"] == 0])),
            matched_nonreward_fields=True,
        ))
        runs[arm]["data_sha256"] = {n: digest(work / n)
                                   for n in ["dataset.npz", "audit.json"]}
    write(root / "protocol.json", dict(
        runs=runs, method="CORL-CQL-predictive-Shapley-v1",
        repository_base=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        source_sha256={name: digest(root / "source" / name) for name in sources},
        preparation_sha256={name: digest(root / name) for name in
                            ["predictor/input.npz", "predictor/folds.json",
                             "base_transitions.npz"]},
        original_hdf5_sha256=digest(args.hdf5),
        delayed_baseline=dict(root=str(args.baseline.resolve()),
                              **baseline["runs"]["medium_replay"]),
        retained=baseline["retained"],
        changes=["offline predictive segment Shapley with uniform residual correction",
                 "matched terminal handling across dense, uniform, delayed, Shapley",
                 "gamma=0.99 retained; no discounted objective invariance claim"],
        attribution=dict(segments=20, segment_length=50, permutations=128,
                         outer_folds=5, width=64, learning_rate=1e-3,
                         weight_decay=1e-4, batch=32, epochs=300, patience=30,
                         validation_masks=8, attribution_seed_base=3000),
        wandb_entity=baseline["wandb_entity"], wandb_project=baseline["wandb_project"],
        wandb_group=root.name,
    ))
    print(f"Prepared {root}: 202 trajectories, matching baseline transitions")


if __name__ == "__main__":
    main()
