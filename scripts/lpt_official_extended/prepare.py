"""Freeze new seed runs from a verified official baseline without changing upstream."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from run import check_source, digest, write

COMMIT = "c4e77cb464c6360733a9ce76d8870f076c76d0aa"
ENV_NAME = "halfcheetah-medium-replay-v2"


def prepare(root, baseline, epochs, seeds, arm):
    root, baseline = root.resolve(), baseline.resolve()
    if epochs <= 2000 or epochs % 500:
        raise ValueError("Extended budget must exceed 2000 and be divisible by 500")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if arm not in ("dense", "delayed"):
        raise ValueError("Reward mode must be dense or delayed")
    original = json.loads((baseline / "protocol.json").read_text())
    if original["upstream_commit"] != COMMIT or original["source_modifications"]:
        raise ValueError("Expected the pinned unmodified official baseline")
    upstream = baseline / "source" / "upstream"
    check_source(upstream, original)
    dataset_hashes = {
        name: expected for name, expected in original["dataset_sha256"].items()
        if name.startswith(f"{arm}/dataset/")
    }
    if not dataset_hashes:
        raise ValueError("No verified dataset for the selected reward mode")
    for name, expected in dataset_hashes.items():
        if digest(baseline / name) != expected:
            raise ValueError(f"Baseline dataset changed: {name}")

    root.mkdir(parents=True, exist_ok=False)
    packages = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    runs = []
    for seed in seeds:
        run_root = root / f"seed{seed}"
        source = run_root / "source"
        source.mkdir(parents=True)
        shutil.copytree(upstream, source / "upstream")
        for path in Path(__file__).parent.iterdir():
            if path.is_file() and path.suffix in (".py", ".sbatch"):
                shutil.copy2(path, source / path.name)
        (run_root / "runtime-packages.txt").write_text(packages)
        data_path = run_root / arm / "dataset" / ENV_NAME
        shutil.copytree(baseline / arm / "dataset" / ENV_NAME, data_path)
        for phase in ("preflight", "training"):
            checkout = run_root / arm / phase / "upstream"
            shutil.copytree(upstream, checkout)
            directory = checkout / "data" / "dataset"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ENV_NAME).symlink_to(data_path, target_is_directory=True)

        name = (
            f"LPT-Official-Unmodified-HCMR-{arm}-"
            f"init{seed}-trainer42-{epochs}updates"
        )
        identity = hashlib.sha256(f"{root.name}/{arm}/{seed}".encode()).hexdigest()[:12]
        protocol = {
            **original,
            "baseline_campaign": str(baseline),
            "campaign": str(root),
            "model_initialization_seed": seed,
            "epochs": epochs,
            "total_updates": epochs,
            "eval_updates": list(range(500, epochs + 1, 500)),
            "reward_mode": arm,
            "start_from_scratch": True,
            "learning_rate_schedule": "official 10% warmup, linear decay to new budget",
            "checkpoint_limitations": "official PMC cache is not saved; no exact resume",
            "task_overrides": [
                ENV_NAME, f"reward mode {arm}", f"model initialization seed {seed}",
                f"training budget {epochs} epochs/updates",
            ],
            "runtime_packages_sha256": digest(run_root / "runtime-packages.txt"),
            "harness_sha256": {
                p.name: digest(p) for p in source.iterdir() if p.is_file()
            },
            "dataset_sha256": dataset_hashes,
            "wandb_group": root.name,
            "wandb_name": name,
            "wandb_run_id": identity,
        }
        write(run_root / "protocol.json", protocol)
        runs.append(dict(seed=seed, root=str(run_root), name=name, run_id=identity))
    campaign = dict(
        root=str(root), baseline=str(baseline), epochs=epochs, seeds=seeds,
        reward_mode=arm, trainer_seed=42, runs=runs,
        upstream_commit=COMMIT, source_modifications=[],
    )
    write(root / "campaign.json", campaign)
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--arm", choices=("dense", "delayed"), default="delayed")
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.root, args.baseline, args.epochs, args.seeds, args.arm
    ), indent=2))


if __name__ == "__main__":
    main()
