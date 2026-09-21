"""Run archived original v3-high, without importing modern training code.

Only orchestration and metric recording live here. Training commands come from
the historical sbatch files, and run in a detached, source-verified old worktree.
"""

import argparse
import hashlib
import json
import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
REVISION = "25626cefd9465dadbd0a6ef95756ab943ca7fab6"
CONTROL_REVISION = "e8126627628ebf2e928796e8a5db3f81fbcd3aa3"
CAMPAIGN = "v3-high-legacy345-20260920"
GROUP = "Legacy012-V3High-Seeds345-To100k-HCMR-delayed"
PROFILES = {
    "legacy345": {
        "seeds": (3, 4, 5), "campaign": CAMPAIGN, "group": GROUP,
        "prefix": "Legacy012", "strict_gpu": False,
    },
    "replay012": {
        "seeds": (0, 1, 2), "campaign": "v3-high-replay012-20260921",
        "group": "Replay012-OriginalV3High-To100k-HCMR-delayed",
        "prefix": "Replay012", "strict_gpu": True,
    },
}
TEMPLATES = {
    "prepare": "prepare_dt50k_pair_diagnostic_seed1.sbatch",
    "diagnose": "run_dt_pair_diagnostic_seed1.sbatch",
    "dt": "run_dt50k_control_to75k_delayed_hcmr_seed1.sbatch",
    "v3": "run_single_target_hard_fork_v3_high_delayed_hcmr_seed1.sbatch",
}
FILES = [
    "algorithms/offline/dt.py", "algorithms/offline/sap_dt_one_sided.py",
    "algorithms/offline/hard_fork_dt.py", "algorithms/offline/diagnose_sap_pairs.py",
    "configs/offline/dt/halfcheetah/medium_replay_v2.yaml",
    *["scripts/dt_experiments/" + name for name in TEMPLATES.values()],
]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def source_for(root, stage):
    if stage == "dt":
        return root / "control_source", CONTROL_REVISION
    return root / "source", REVISION


def verify_source(source, revision=REVISION):
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip() != revision:
        raise RuntimeError("Wrong historical revision")
    hashes = {}
    names = FILES
    if revision == CONTROL_REVISION:
        names = FILES[:5] + ["scripts/dt_experiments/" + TEMPLATES["dt"]]
    for name in names:
        original = subprocess.check_output(
            ["git", "show", f"{revision}:{name}"], cwd=PROJECT
        )
        hashes[name] = hashlib.sha256(original).hexdigest()
        if digest(source / name) != hashes[name]:
            raise RuntimeError(f"Historical source was modified: {name}")
    return hashes


def paths(root, seed, smoke=False):
    directory = root / ("smoke" if smoke else "production") / f"seed{seed}"
    return directory, directory / "dt-step50000.pt", directory / "pairs"


def expected_gpu(seed, stage, profile="legacy345"):
    # Replay uses the original seed numbers; legacy345 maps 0/1/2 -> 3/4/5.
    if profile == "replay012":
        seed += 3
    newer = (stage == "dt" and seed == 3) or (stage == "v3" and seed == 5)
    return "RTX 4090" if newer else "RTX 3090"


def hardware_record(gpu, seed, stage, profile="legacy345"):
    if not any(model in gpu for model in ("RTX 3090", "RTX 4090")):
        raise RuntimeError(f"Unsupported GPU for this campaign: {gpu}")
    historical = expected_gpu(seed, stage, profile)
    strict = PROFILES[profile]["strict_gpu"]
    if strict and historical not in gpu:
        raise RuntimeError(f"Historical GPU required: {historical}; got {gpu}")
    return {
        "gpu": gpu, "historical_gpu": historical,
        "matches_historical_gpu": historical in gpu,
        "policy": ("historical-model-matched-single-gpu" if strict else
                   "user-approved-3090-or-4090-single-gpu"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }


def set_flag(argv, key, value):
    key = "--" + key
    if key in argv:
        argv[argv.index(key) + 1] = str(value)
    else:
        argv.extend([key, str(value)])


def command(source, stage, seed, directory, smoke=False, profile="legacy345"):
    """Mechanically reuse seed-1 launcher; leave all unlisted arguments alone."""
    settings = PROFILES[profile]
    if seed not in settings["seeds"]:
        raise ValueError(f"Invalid seed {seed} for profile {profile}")
    script = source / "scripts/dt_experiments" / TEMPLATES[stage]
    body = script.read_text().split("\npython ", 1)[1].replace("\\\n", " ")
    argv = shlex.split(body.replace("${PROJECT}", str(source)))
    argv[0] = str(source / argv[0])
    pairs = directory / "pairs"
    checkpoint = directory / "dt-step50000.pt"
    pair_file = pairs / (
        f"dt_pair_diagnostic_halfcheetah_medium_replay_delayed_seed{seed}.npz"
    )
    name = f"{settings['prefix']}-{stage}-seed{seed}-HCMR-delayed"
    if smoke:
        name = "Smoke-" + name
    if stage == "diagnose":
        overrides = dict(
            checkpoint=checkpoint, seed=seed, result_dir=pairs,
            wandb_mode="disabled", wandb_group=settings["group"], wandb_name=name,
        )
    else:
        overrides = dict(train_seed=seed, group=settings["group"], name=name)
        if stage == "prepare":
            overrides.update(
                preference_pair_seed=seed, reference_checkpoint_path=checkpoint,
            )
        else:
            overrides.update(
                update_steps=100001, pretrained_checkpoint_path=checkpoint,
                checkpoints_path=directory / "checkpoints" / stage,
            )
            if stage == "v3":
                overrides.update(
                    preference_pair_seed=seed, preference_arrays_path=pair_file,
                )
        if smoke:
            overrides.update(
                update_steps=3 if stage == "prepare" else 5,
                preference_start_step=2, eval_every=2, eval_episodes=1,
            )
    for key, value in overrides.items():
        set_flag(argv, key, value)
    return argv


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "item"):
        return plain(value.item())
    return value


def install_observer(wandb, record, hashes, revision=REVISION, hardware=None):
    """No RNG/tensor hooks. Block ONLY binary-artifact upload, not local saving."""
    original_init, original_finish = wandb.init, wandb.finish

    def init(*args, **kwargs):
        run = original_init(*args, **kwargs)
        if hardware is not None:
            for key, value in hardware.items():
                if value is not None:
                    run.summary[f"hardware/{key}"] = value
        bound_log, bound_save = run.log, run.save

        def log(data, *log_args, **log_kwargs):
            step = log_kwargs.get("step", log_args[0] if log_args else None)
            row = {"step": step, **plain(dict(data))}
            with (record / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            return bound_log(data, *log_args, **log_kwargs)

        def save(glob_str=None, *save_args, **save_kwargs):
            if glob_str and Path(glob_str).suffix in {".pt", ".npz", ".npy"}:
                print(f"Keep binary artifact server-local: {glob_str}", flush=True)
                return []
            return bound_save(glob_str, *save_args, **save_kwargs)

        def finish(*finish_args, **finish_kwargs):
            write_json(record / "summary.json", plain(dict(run.summary)))
            return original_finish(*finish_args, **finish_kwargs)

        # init rebinds these module-level APIs: install observers afterwards.
        wandb.log, wandb.save, wandb.finish = log, save, finish
        write_json(record / "run.json", {
            "id": run.id, "url": run.url, "name": run.name,
            "config": plain(dict(run.config)), "historical_revision": revision,
            "source_sha256": hashes, "hardware": hardware,
        })
        return run

    wandb.init = init


def worker(args, root):
    import torch
    import wandb

    source, revision = source_for(root, args.stage)
    hashes = verify_source(source, revision)
    gpu = torch.cuda.get_device_name(0)
    hardware = hardware_record(gpu, args.seed, args.stage, args.profile)
    directory, _, _ = paths(root, args.seed, args.smoke)
    record = directory / "records" / args.stage
    record.mkdir(parents=True, exist_ok=False)
    argv = command(source, args.stage, args.seed, directory, args.smoke, args.profile)
    write_json(record / "command.json", argv)
    write_json(record / "hardware.json", hardware)
    install_observer(wandb, record, hashes, revision, hardware)
    os.chdir(source)
    sys.path.insert(0, str(source / "algorithms/offline"))
    sys.argv = argv
    try:
        runpy.run_path(argv[0], run_name="__main__")
        if not (record / "metrics.jsonl").is_file():
            raise RuntimeError("No metrics journal produced")
        write_json(record / "completed.json", {"success": True})
    except BaseException as error:
        write_json(record / "failed.json", {"error": repr(error)})
        raise


def run_stage(root, stage, seed, smoke, profile="legacy345"):
    argv = [sys.executable, str(Path(__file__).resolve()), "worker",
            "--root", str(root), "--seed", str(seed), "--stage", stage,
            "--profile", profile]
    if smoke:
        argv.append("--smoke")
    subprocess.run(argv, check=True)


def rows(record):
    merged = {}
    for line in (record / "metrics.jsonl").read_text().splitlines():
        item = json.loads(line)
        merged.setdefault(item["step"], {}).update(item)
    return merged


def compare_original(root, seed, checkpoint, pair_file):
    """Read-only CPU comparison AFTER training/mining subprocesses have exited."""
    import numpy as np
    import torch

    original = read_json(root / "original_inputs.json")[str(seed)]
    for key in ("checkpoint", "pairs"):
        if digest(original[key]) != original[key + "_sha256"]:
            raise RuntimeError(f"Original seed-{seed} input changed: {key}")
    old = torch.load(original["checkpoint"], map_location="cpu")
    new = torch.load(checkpoint, map_location="cpu")
    a, b = old["model_state"], new["model_state"]
    if a.keys() != b.keys():
        raise RuntimeError("Old/new model state structures differ")
    different = [key for key in a if not torch.equal(a[key], b[key])]
    maximum = max((a[key].float() - b[key].float()).abs().max().item()
                  for key in a)
    with np.load(original["pairs"]) as pa, np.load(pair_file) as pb:
        pair_keys_equal = set(pa.files) == set(pb.files)
        different_arrays = [key for key in set(pa.files) & set(pb.files)
                            if not np.array_equal(pa[key], pb[key], equal_nan=True)]
        counts = [int(pa["valid_branch"].sum()), int(pb["valid_branch"].sum())]
    return {
        "original": original, "model_tensors_equal": not different,
        "different_model_tensors": different, "max_abs_tensor_difference": maximum,
        "next_steps": [old["next_step"], new["next_step"]],
        "normalization_equal": all(np.array_equal(old[k], new[k])
                                   for k in ("state_mean", "state_std")),
        "pair_keys_equal": pair_keys_equal,
        "pair_arrays_equal": pair_keys_equal and not different_arrays,
        "different_pair_arrays": sorted(different_arrays),
        "valid_pairs_old_new": counts,
        "note": "File hashes include config/path/serialization metadata; compare tensors and arrays instead.",
    }


def original_inputs():
    manifest = {}
    for seed in (0, 1, 2):
        checkpoint = PROJECT / "checkpoints" / (
            f"dt-halfcheetah-medium-replay-v2-delayed-seed{seed}-step50000.pt"
        )
        pairs = PROJECT / "results/pair_diagnostics" / (
            f"dt_pair_diagnostic_halfcheetah_medium_replay_delayed_seed{seed}.npz"
        )
        manifest[str(seed)] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
            "pairs": str(pairs), "pairs_sha256": digest(pairs),
        }
    return manifest


def prepare(root, seed, smoke, profile="legacy345"):
    import numpy as np
    import torch

    source = root / "source"
    verify_source(source)
    directory, checkpoint, pairs = paths(root, seed, smoke)
    directory.mkdir(parents=True, exist_ok=False)
    run_stage(root, "prepare", seed, smoke, profile)
    loaded = torch.load(checkpoint, map_location="cpu")
    start = 2 if smoke else 50000
    if loaded["next_step"] != start or loaded["config"]["train_seed"] != seed:
        raise RuntimeError("Wrong historical warmup checkpoint")
    if "rng_state" in loaded:
        raise RuntimeError("Unexpected modern checkpoint format")
    diagnostic = command(source, "diagnose", seed, directory, smoke, profile)
    subprocess.run([sys.executable, *diagnostic], cwd=source, check=True)
    pair_file = pairs / (
        f"dt_pair_diagnostic_halfcheetah_medium_replay_delayed_seed{seed}.npz"
    )
    with np.load(pair_file) as archive:
        valid = int(archive["valid_branch"].sum())
    if valid <= 0:
        raise RuntimeError("No strict pairs; do not relax thresholds")
    write_json(directory / "prepared.json", {
        "seed": seed, "historical_revision": REVISION,
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "pairs": str(pair_file), "pairs_sha256": digest(pair_file),
        "valid_pair_count": valid, "next_step": start,
        "gpu": torch.cuda.get_device_name(0),
    })
    if profile == "replay012" and not smoke:
        write_json(directory / "original_comparison.json",
                   compare_original(root, seed, checkpoint, pair_file))


def branch(root, seed, arm, smoke, profile="legacy345"):
    if arm not in ("dt", "v3"):
        raise ValueError("A continuation branch must be dt or v3")
    if not smoke:
        gate = read_json(root / "smoke/passed.json")
        if gate.get("control_revision") != CONTROL_REVISION or not gate["passed"]:
            raise RuntimeError("Complete archived-control GPU smoke has not passed")
        if gate.get("profile", "legacy345") != profile:
            raise RuntimeError("Wrong campaign smoke gate")
    directory, _, _ = paths(root, seed, smoke)
    prepared = read_json(directory / "prepared.json")
    for key in ("checkpoint", "pairs"):
        if digest(prepared[key]) != prepared[key + "_sha256"]:
            raise RuntimeError(f"Changed branch input: {key}")
    run_stage(root, arm, seed, smoke, profile)
    record = directory / "records" / arm
    merged = rows(record)
    first, last = (2, 4) if smoke else (50000, 100000)
    train = {s: r for s, r in merged.items() if "train/total_loss" in r}
    if set(train) != set(range(first, last + 1)):
        raise RuntimeError("Missing or extra legacy training updates")
    for row in train.values():
        if arm == "dt":
            if row["train/total_loss"] != row["train_loss"]:
                raise RuntimeError("Control loss is not ordinary DT")
            if row["train/preference_active"] != 0:
                raise RuntimeError("Control auxiliary branch unexpectedly enabled")
        elif row["train/high_target_fraction"] != 1:
            raise RuntimeError("Preference target is not high-only")
    expected = {2, 4} if smoke else set(range(50000, 100001, 5000))
    scores = {}
    for target in (12000.0, 6000.0):
        field = f"eval/{target}_normalized_score_mean"
        values = {s: r[field] for s, r in merged.items() if field in r}
        if set(values) != expected:
            raise RuntimeError("Missing legacy evaluation points")
        scores[str(target)] = values
    write_json(record / "audit.json", {
        "passed": True, "historical_revision": source_for(root, arm)[1],
        "checkpoint_sha256": prepared["checkpoint_sha256"],
        "pairs_sha256": prepared["pairs_sha256"], "scores": scores,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init", "prepare", "branch", "worker", "smoke"))
    parser.add_argument("--profile", choices=tuple(PROFILES), default="legacy345")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage", choices=("prepare", "dt", "v3"), default="v3")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    settings = PROFILES[args.profile]
    if args.seed is None:
        args.seed = settings["seeds"][0]
    if args.seed not in settings["seeds"]:
        parser.error(f"Seed {args.seed} does not belong to {args.profile}")
    if args.root is None:
        args.root = PROJECT / "results" / settings["campaign"]
    root = args.root.resolve()
    if args.mode != "init":
        manifest = root / "protocol.json"
        recorded = read_json(manifest)["profile"] if manifest.exists() else "legacy345"
        if recorded != args.profile:
            raise RuntimeError("Campaign directory belongs to a different protocol")
    if args.mode == "init":
        originals = original_inputs() if args.profile == "replay012" else None
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "protocol.json", {"profile": args.profile, **settings})
        if originals is not None:
            write_json(root / "original_inputs.json", originals)
        subprocess.run(["git", "worktree", "add", "--detach", str(root / "source"),
                        REVISION], cwd=PROJECT, check=True)
        write_json(root / "source_manifest.json", verify_source(root / "source"))
        subprocess.run(["git", "worktree", "add", "--detach",
                        str(root / "control_source"), CONTROL_REVISION],
                       cwd=PROJECT, check=True)
        write_json(root / "control_source_manifest.json", verify_source(
            root / "control_source", CONTROL_REVISION
        ))
    elif args.mode == "worker":
        worker(args, root)
    elif args.mode == "prepare":
        if not read_json(root / "smoke/passed.json")["passed"]:
            raise RuntimeError("Historical GPU smoke has not passed")
        if read_json(root / "smoke/passed.json").get("profile", "legacy345") != args.profile:
            raise RuntimeError("Wrong campaign smoke gate")
        prepare(root, args.seed, False, args.profile)
    elif args.mode == "branch":
        branch(root, args.seed, args.stage, False, args.profile)
    else:
        prepare(root, args.seed, True, args.profile)
        for arm in ("dt", "v3"):
            branch(root, args.seed, arm, True, args.profile)
        write_json(root / "smoke/passed.json", {
            "passed": True, "revision": REVISION,
            "control_revision": CONTROL_REVISION, "seed": args.seed,
            "profile": args.profile,
        })


if __name__ == "__main__":
    main()
