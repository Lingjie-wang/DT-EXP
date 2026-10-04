"""Freeze unchanged official LPT and prepare the two requested reward datasets."""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
from datasets import Dataset, DatasetDict

COMMIT = "c4e77cb464c6360733a9ce76d8870f076c76d0aa"
ENV_NAME = "halfcheetah-medium-replay-v2"


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def trajectories_from_hdf5(path):
    # This is the HalfCheetah branch of official data/process_data.py: retain
    # every transition and split at terminals OR timeouts. No filtering/windows.
    with h5py.File(path, "r") as f:
        arrays = {
            k: f[k][:] for k in ("observations", "actions", "rewards", "terminals")
        }
        ends = np.flatnonzero(arrays["terminals"] | f["timeouts"][:]) + 1
    assert ends[-1] == len(arrays["rewards"])
    rows = []
    first = 0
    for end in ends:
        rows.append(
            dict(
                observations=arrays["observations"][first:end],
                actions=arrays["actions"][first:end],
                rewards=arrays["rewards"][first:end],
                dones=arrays["terminals"][first:end],
            )
        )
        first = end
    return rows


def terminal_rewards(rewards):
    # Aggregate the observed float32 step values in float64, as the official
    # collator does after Arrow converts its reward lists to Python floats.
    result = np.zeros(len(rewards), dtype=np.float64)
    result[-1] = np.asarray(rewards, dtype=np.float64).sum()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == COMMIT
    assert not subprocess.check_output(
        ["git", "-C", str(args.upstream), "diff", "HEAD", "--"], text=True
    )
    rows = trajectories_from_hdf5(args.hdf5)
    assert len(rows) == 202 and all(len(r["rewards"]) == 1000 for r in rows)
    root.mkdir(parents=True, exist_ok=False)
    packages = root / "runtime-packages.txt"
    packages.write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    source = root / "source"
    source.mkdir()
    upstream = source / "upstream"
    upstream.mkdir()
    archive = subprocess.check_output(
        ["git", "-C", str(args.upstream), "archive", COMMIT]
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(upstream)
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            shutil.copy2(path, source / path.name)
    manifest = {
        str(p.relative_to(upstream)): digest(p)
        for p in upstream.rglob("*")
        if p.is_file()
    }
    for arm in ("dense", "delayed"):
        dataset_rows = [
            {
                **r,
                "rewards": r["rewards"]
                if arm == "dense"
                else terminal_rewards(r["rewards"]),
            }
            for r in rows
        ]
        # DatasetDict and field names match the released processor exactly.
        dataset = DatasetDict(
            train=Dataset.from_dict(
                {key: [r[key] for r in dataset_rows] for key in dataset_rows[0]}
            )
        )
        data_path = root / arm / "dataset" / ENV_NAME
        dataset.save_to_disk(str(data_path))
        # Independent directories preserve the preflight and full-run outputs.
        for phase in ("preflight", "training"):
            checkout = root / arm / phase / "upstream"
            shutil.copytree(upstream, checkout)
            directory = checkout / "data" / "dataset"
            directory.mkdir(exist_ok=True)
            (directory / ENV_NAME).symlink_to(data_path, target_is_directory=True)
    dense = np.array([np.asarray(r["rewards"], dtype=np.float64).sum() for r in rows])
    delayed = np.array([terminal_rewards(r["rewards"]).sum() for r in rows])
    np.testing.assert_array_equal(dense, delayed)
    protocol = dict(
        method="LPT-Official-Unmodified",
        upstream_commit=COMMIT,
        env_name=ENV_NAME,
        eval_env="HalfCheetah-v3",
        model_initialization_seed=0,
        trainer_seed=42,
        data_seed=None,
        epochs=2000,
        total_updates=2000,
        nominal_batch_size=500,
        effective_batch_size=202,
        gradient_accumulation_steps=1,
        hidden_size=128,
        layers=3,
        heads=1,
        window_size=32,
        latent_count=4,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        grad_clip=0.25,
        target_return=6000,
        reward_scale=1000.0,
        train_langevin_steps=3,
        eval_langevin_steps=20,
        eval_updates=[500, 1000, 1500, 2000],
        eval_episodes=10,
        eval_seeding="unchanged official evaluator; seed=None",
        train_entry="scripts/train.py",
        attention="flash-attn 2.3.6 CUDA kernels",
        source_modifications=[],
        algorithm_patches=[],
        task_overrides=[
            "HalfCheetah-medium-replay-v2",
            "model initialization seed 0",
            "separate episode-terminal reward dataset for delayed arm",
        ],
        preserved_behaviors=[
            "official best-latent index and duplicate cache writes",
            "official action history and dropout during evaluation",
            "official Trainer RNG reset and evaluation RNG consumption",
            "official fixed state statistics and 10-episode evaluation",
        ],
        runtime_differences=[
            "Python 3.9 instead of 3.8 to reuse working MuJoCo bindings",
            "Transformers 4.37.0 release instead of unspecified dev build",
            "MuJoCo 2.1/mujoco-py for official Gym-v3 environment",
            "missing datasets, accelerate and FlashAttention installed",
        ],
        trajectories=202,
        transitions=202000,
        total_return_labels_equal=True,
        data_sha256=digest(args.hdf5),
        runtime_packages_sha256=digest(packages),
        upstream_sha256=manifest,
        harness_sha256={p.name: digest(p) for p in source.iterdir() if p.is_file()},
        dataset_sha256={
            str(p.relative_to(root)): digest(p)
            for arm in ("dense", "delayed")
            for p in (root / arm / "dataset").rglob("*")
            if p.is_file()
        },
        wandb_entity="2820402607-shandong-university",
        wandb_project="CORL-DDR",
        wandb_group=root.name,
    )
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                root=str(root),
                source_changes=0,
                trajectories=202,
                total_return_labels_equal=True,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
