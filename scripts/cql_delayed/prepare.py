"""Prepare immutable terminal-return datasets and source snapshots for two tasks."""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from common import digest, write

from algorithms.offline.cql_delayed_data import terminal_return_dataset

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--medium-replay", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = ["algorithms/offline/cql.py", "algorithms/offline/cql_delayed_data.py"]
    sources += [str(p.relative_to(repository)) for p in
                (repository / "scripts/cql_delayed").iterdir() if p.is_file()]
    for arm in ["medium", "medium_replay"]:
        sources += [f"configs/offline/cql/halfcheetah/{arm}_delayed_v2.yaml",
                    f"configs/offline/cql/halfcheetah/{arm}_v2.yaml"]
    for relative in sources:
        target = root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, target)
    runs = {}
    for arm, original in [
        ("medium", args.medium), ("medium_replay", args.medium_replay)
    ]:
        work = root / arm
        work.mkdir()
        with h5py.File(original, "r") as f:
            raw = {k: f[k][:] for k in ["observations", "actions", "rewards",
                                      "terminals", "timeouts", "next_observations"]
                   if k in f}
        data, audit = terminal_return_dataset(raw)
        np.savez(work / "dataset.npz", **data)
        audit.update(original_path=str(original.resolve()),
                     original_sha256=digest(original))
        write(work / "audit.json", audit)
        del raw, data
        tag = "HCM" if arm == "medium" else "HCMR"
        name = f"CQL-CORL-{tag}-delayed-seed0-1M"
        identity = hashlib.sha256(f"{root.name}/{arm}".encode()).hexdigest()[:12]
        runs[arm] = dict(
            env=f"halfcheetah-{arm.replace('_', '-')}-v2", seed=0,
            updates=1000000, eval_every=5000, eval_episodes=10,
            config=f"configs/offline/cql/halfcheetah/{arm}_delayed_v2.yaml",
            wandb_name=name, wandb_id=identity,
            data_sha256={n: digest(work / n) for n in ["dataset.npz", "audit.json"]},
        )
        print(arm, audit, flush=True)
    write(root / "protocol.json", dict(
        method="CORL-CQL", reward_mode="terminal-total", runs=runs,
        repository_base=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        source_sha256={name: digest(root / "source" / name) for name in sources},
        wandb_entity="2820402607-shandong-university", wandb_project="CORL-DDR",
        wandb_group=root.name,
        changes=["move undiscounted episode reward sum to final transition",
                 "retain timeout transitions and stop bootstrap at every episode end",
                 "local telemetry and extra checkpoints; CQL updates unchanged"],
        retained=dict(discount=0.99, normalize_reward=False, batch_size=256,
                      cql_alpha=10.0, policy_lr=3e-5, qf_lr=3e-4, buffer_size=10000000),
    ))


if __name__ == "__main__":
    main()
