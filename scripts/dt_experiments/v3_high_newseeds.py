"""Fresh seeds 3/4/5: ordinary DT 0-50k, then paired DT/v3-high to 100k.

Orchestration and local journaling only; all original algorithm modules remain
unchanged. A short paired gate runs before any production continuation.
"""

import argparse
import hashlib
import json
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SEEDS = (3, 4, 5)
GROUP = "V3High-FreshSeeds345-Paired50kTo100k-HCMR-delayed"
SOURCE_HASHES = {
    "dt.py": "477a587e099e5e255d16354a27033db7c367cf39e92dae4b4b4511e0ca15ac0d",
    "hard_fork_dt.py":
        "f387e591dd01dc7b3ceca35b6909c3188bb89e68895082fcdcb5bd112c190777",
    "top_return_weighted_dt.py":
        "368ef8b42f5e55a3c6c62b0488161a08465175e2ea29a73b501ffd7d10b32b77",
    "diagnose_sap_pairs.py":
        "c308cd0228a96bb02bb8b8f230976cc1d0bfccdf59f75a1fc3ef09f0d8169021",
    "sap_dt_one_sided.py":
        "e68253fa9874382f526179240d87c877f7fb2541eed74397bd6c933f5fb39e0e",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources():
    for filename, expected in SOURCE_HASHES.items():
        actual = sha256(PROJECT / "algorithms/offline" / filename)
        if actual != expected:
            raise RuntimeError(f"Original algorithm source changed: {filename}")


def plain(value):
    if isinstance(value, dict) or callable(getattr(value, "keys", None)):
        return {str(k): plain(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return plain(value.item())
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def journal_worker(arguments):
    """Observe the existing W&B API without changing tensors, RNG or updates."""
    import wandb

    verify_sources()
    record = Path(arguments.record_dir).resolve()
    record.mkdir(parents=True, exist_ok=False)
    original_init, original_finish = wandb.init, wandb.finish
    active_run_log = None

    def init(*args, **kwargs):
        nonlocal active_run_log
        run = original_init(*args, **kwargs)
        # wandb.init replaces module-level wandb.log with the new run's method.
        # Install our observer AFTER initialization and forward to that bound
        # method, never to the pre-init proxy or our own observer.
        active_run_log = run.log
        wandb.log, wandb.finish = log, finish
        write_json(record / "run.json", {
            "id": run.id, "url": run.url, "name": run.name,
            "config": dict(run.config), "source_sha256": SOURCE_HASHES,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
            ).strip(),
        })
        return run

    def log(data, *args, **kwargs):
        if active_run_log is None:
            raise RuntimeError("Initialize W&B before logging training metrics")
        step = kwargs.get("step", args[0] if args else None)
        row = {"step": step, **plain(dict(data))}
        # fail loudly if any numeric metric becomes NaN/Inf, never silently skip it
        encoded = json.dumps(row, allow_nan=False)
        with open(record / "metrics.jsonl", "a") as stream:
            stream.write(encoded + "\n")
        return active_run_log(data, *args, **kwargs)

    def finish(*args, **kwargs):
        if wandb.run is not None:
            write_json(record / "summary.json", dict(wandb.run.summary))
        return original_finish(*args, **kwargs)

    wandb.init, wandb.log, wandb.finish = init, log, finish
    entry = PROJECT / "algorithms/offline" / arguments.entry
    child_args = arguments.remaining
    if child_args[:1] == ["--"]:
        child_args = child_args[1:]
    sys.argv = [str(entry), *child_args]
    sys.path.insert(0, str(entry.parent))
    try:
        runpy.run_path(str(entry), run_name="__main__")
        metrics_path = record / "metrics.jsonl"
        if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
            raise RuntimeError("Training returned without a local metrics journal")
    except BaseException as error:
        write_json(record / "failed.json", {"error": repr(error)})
        raise
    write_json(record / "completed.json", {"success": True})


def root_path(campaign):
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    if not campaign or any(c not in allowed for c in campaign):
        raise ValueError("Campaign must be a safe lowercase directory component")
    return PROJECT / "results" / campaign


def validate_seed(seed):
    if seed not in SEEDS:
        raise ValueError("Only NEW seeds 3, 4, 5 are allowed; never rerun 0, 1, 2")


def flags(**values):
    result = []
    for key, value in values.items():
        if isinstance(value, (list, tuple, bool)):
            value = json.dumps(value)
        result.extend(["--" + key, str(value)])
    return result


def common_flags(seed, name, output):
    validate_seed(seed)
    return flags(
        config_path=PROJECT / "configs/offline/dt/halfcheetah/medium_replay_v2.yaml",
        project="CORL-DDR", group=GROUP, name=name,
        reward_mode="delayed", train_seed=seed, device="cuda:0",
        checkpoints_path=output,
    )


def branch_flags(seed, arm, checkpoint, pairs, output, start=50000, stop=100000,
                 gate=False):
    if arm not in ("dt", "v3"):
        raise ValueError("Expected dt or v3")
    name = f"V3High-Fresh345-{'DTControl' if arm == 'dt' else 'V3High'}-seed{seed}"
    name += "-To100k-HCMR-delayed"
    if gate:
        name = "SmokeGate-" + name
    return common_flags(seed, name, output) + flags(
        paired_resume=True, pairing_audit_steps=3, update_steps=stop,
        eval_every=1 if gate else 5000, eval_episodes=1 if gate else 100,
        preference_start_step=start, preference_mode="hard_fork",
        preference_weight=0.0 if arm == "dt" else 0.05,
        reference_weight=0.0 if arm == "dt" else 0.1,
        reference_anchor_mode="mse", reference_tolerance=0.0,
        preference_margin=0.05, preference_batch_size=256,
        preference_hardness_temperature=0.05,
        active_only_normalization=True, preference_min_active_pairs=16,
        prioritized_pair_sampling=True, dynamic_priority_mix=0.5,
        dynamic_priority_ema=0.9, target_aligned_preference=True,
        preference_target_mode="high_only", preference_target_return_high=12000.0,
        preference_pair_seed=seed, pretrained_checkpoint_path=checkpoint,
        preference_arrays_path=pairs,
        checkpoint_steps=[stop] if gate else [55000, 60000, 65000, 70000,
                                             75000, 80000, 85000, 90000, 95000,
                                             100000],
    )


def run_training(entry, arguments, record, offline=False):
    environment = os.environ.copy()
    environment["WANDB_MODE"] = "offline" if offline else "online"
    environment["WANDB_DISABLE_CODE"] = "true"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "worker",
         "--entry", entry, "--record-dir", str(record), "--", *arguments],
        cwd=PROJECT, env=environment, check=True,
    )
    return read_json(record / "run.json")


def state_dict_hash(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def metric_rows(record):
    rows = {}
    for line in (record / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows.setdefault(row["step"], {}).update(row)
    return rows


def audit_pair(directory, checkpoint_path, seed, start, stop, gate=True):
    import torch

    names = ("gate_dt", "gate_v3") if gate else ("dt", "v3")
    records = [directory / "records" / name for name in names]
    metadata = [read_json(path / "run.json") for path in records]
    summaries = [read_json(path / "summary.json") for path in records]
    for path in records:
        if not (path / "completed.json").is_file():
            raise AssertionError(f"Incomplete child run: {path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    initial_hash = state_dict_hash(checkpoint["model_state"])
    if checkpoint["config"]["train_seed"] != seed:
        raise AssertionError("Input checkpoint uses another training seed")
    if checkpoint["completed_updates"] != start or checkpoint["next_step"] != start:
        raise AssertionError("Wrong branch start update")
    allowed_difference = {"name", "checkpoints_path", "preference_weight",
                          "reference_weight"}
    left, right = [m["config"] for m in metadata]
    for key in set(left) | set(right):
        if key not in allowed_difference and left.get(key) != right.get(key):
            raise AssertionError(f"Unpaired configuration: {key}")
    for summary in summaries:
        if summary["pairing/initial_model_sha256"] != initial_hash:
            raise AssertionError("Branch initialization differs from source")
        if summary["pairing/restored_scheduler_epoch"] != start:
            raise AssertionError("Warmup/scheduler was not restored")
        if summary["provenance/pretrained_checkpoint_sha256"] != sha256(checkpoint_path):
            raise AssertionError("Checkpoint file hash mismatch")
    for key in ("pairing/restored_torch_rng_sha256", "pairing/restored_optimizer_lr",
                "provenance/preference_arrays_sha256"):
        if summaries[0][key] != summaries[1][key]:
            raise AssertionError(f"Pairing mismatch: {key}")
    for step in range(start + 1, start + 4):
        for field in ("dt_batch_sha256", "torch_rng_sha256"):
            key = f"pairing/update_{step}/{field}"
            if summaries[0][key] != summaries[1][key]:
                raise AssertionError(f"Unpaired ordinary DT stream: {key}")
    key = f"pairing/update_{start + 1}/dt_loss"
    if summaries[0][key] != summaries[1][key]:
        raise AssertionError("First ordinary DT loss differs")
    rows = [metric_rows(path) for path in records]
    for arm_index, arm_rows in enumerate(rows):
        train_rows = [r for r in arm_rows.values() if "train/total_loss" in r]
        if len(train_rows) != stop - start:
            raise AssertionError("Unexpected number of updates")
        for row in train_rows:
            expected = (row["train_loss"] + row["train/weighted_preference_loss"]
                        + row["train/weighted_reference_loss"])
            if not math.isclose(row["train/total_loss"], expected, abs_tol=1e-7):
                raise AssertionError("Loss components do not sum to objective")
            if arm_index == 0 and (
                row["train/weighted_preference_loss"] != 0.0
                or row["train/weighted_reference_loss"] != 0.0
                or row["train/total_loss"] != row["train_loss"]
            ):
                raise AssertionError("DT control has a non-DT objective")
        snapshot = Path(metadata[arm_index]["config"]["checkpoints_path"])
        snapshot = torch.load(snapshot / f"step{stop:06d}.pt", map_location="cpu")
        if snapshot["completed_updates"] != stop:
            raise AssertionError("Snapshot step mismatch")
        if state_dict_hash(snapshot["reference_model_state"]) != initial_hash:
            raise AssertionError("Reference is not frozen at the shared branch start")
    result = {"passed": True, "seed": seed, "start": start, "stop": stop,
              "initial_model_sha256": initial_hash,
              "checkpoint_sha256": sha256(checkpoint_path),
              "first_dt_loss": summaries[0][key],
              "arms": [m["url"] for m in metadata]}
    if not gate:
        result["scores"] = {}
        for arm, arm_rows in zip(("dt", "v3"), rows):
            result["scores"][arm] = {}
            for target in (12000.0, 6000.0):
                field = f"eval/{target}_normalized_score_mean"
                values = [(step, row[field]) for step, row in arm_rows.items()
                          if start < step <= stop and field in row]
                if len(values) != (stop - start) // 5000:
                    raise AssertionError("Missing evaluation in comparison window")
                best = max(values, key=lambda row: row[1])
                result["scores"][arm][str(int(target))] = {
                    "last": sorted(values)[-1][1], "best": best[1],
                    "best_step": best[0],
                    "window_mean": sum(v for _, v in values) / len(values),
                }
    filename = "gate_audit.json" if gate else "final_pair_audit.json"
    write_json(directory / filename, result)
    return result


def prepare(campaign, seed, smoke=False):
    import numpy as np
    import torch

    validate_seed(seed)
    verify_sources()
    root = root_path(campaign)
    if not smoke:
        gate = read_json(root / "smoke" / "all_seeds_passed.json")
        if (not gate["passed"] or gate["source_sha256"] != SOURCE_HASHES
                or gate["seeds"] != list(SEEDS)):
            raise RuntimeError("Global GPU smoke gate has not passed for this source")
    directory = root / ("smoke" if smoke else "production") / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    output = PROJECT / "checkpoints" / campaign
    output = output / ("smoke" if smoke else "production") / f"seed{seed}"
    start = 2 if smoke else 50000
    name = f"V3High-Fresh345-SharedDTWarmup50k-seed{seed}-HCMR-delayed"
    if smoke:
        name = "Smoke-" + name
    pretrain = common_flags(seed, name, output / "pretrain") + flags(
        top_fraction=0.05, top_weight=1.0, update_steps=start,
        eval_every=2 if smoke else 5000, eval_episodes=1 if smoke else 100,
        checkpoint_steps=[start] if smoke else [10000, 20000, 50000],
        log_every=1 if smoke else 100,
    )
    write_json(directory / "status.json", {"stage": "pretraining", "seed": seed})
    metadata = run_training("top_return_weighted_dt.py", pretrain,
                            directory / "records/pretrain", offline=smoke)
    checkpoint = Path(metadata["config"]["checkpoints_path"]) / f"step{start:06d}.pt"
    loaded = torch.load(checkpoint, map_location="cpu")
    if loaded["config"]["top_weight"] != 1.0:
        raise AssertionError("Warmup is not ordinary DT")
    pair_dir = directory / "pairs"
    write_json(directory / "status.json", {"stage": "mining_pairs", "seed": seed})
    diagnostic = flags(
        checkpoint=checkpoint, env_name="halfcheetah-medium-replay-v2",
        reward_mode="delayed", device="cuda:0", batch_size=512, seed=seed,
        num_pairs=100000, num_candidates=64, state_keep_fraction=0.25,
        good_quantile=0.70, bad_quantile=0.30, timestep_bucket=25,
        prefix_length=5, min_prefix_steps=5, prefix_state_threshold=0.75,
        prefix_action_threshold=0.50, max_timestep_gap=5,
        action_difference_threshold=0.25, min_return_gap=2000,
        preference_margin=0.05, result_dir=pair_dir, wandb_mode="disabled",
    )
    diagnostic_entry = PROJECT / "algorithms/offline/diagnose_sap_pairs.py"
    subprocess.run([sys.executable, str(diagnostic_entry), *diagnostic],
                   cwd=PROJECT, check=True)
    pair_name = f"dt_pair_diagnostic_halfcheetah_medium_replay_delayed_seed{seed}.npz"
    pairs = pair_dir / pair_name
    with np.load(pairs) as archive:
        valid_count = int(archive["valid_branch"].sum())
    if not valid_count:
        raise RuntimeError("No strict valid pairs; do not auto-relax thresholds")
    write_json(directory / "status.json", {"stage": "paired_gate", "seed": seed})
    for arm in ("dt", "v3"):
        argv = branch_flags(seed, arm, checkpoint, pairs, output / f"gate_{arm}",
                            start=start, stop=start + 3, gate=True)
        run_training("hard_fork_dt.py", argv, directory / "records" / f"gate_{arm}",
                     offline=True)
    audit = audit_pair(directory, checkpoint, seed, start, start + 3)
    manifest = {
        "seed": seed, "branch_start": start, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "pairs": str(pairs),
        "initial_model_sha256": loaded["initial_model_sha256"],
        "checkpoint_model_sha256": state_dict_hash(loaded["model_state"]),
        "pairs_sha256": sha256(pairs), "valid_pairs": valid_count,
        "source_sha256": SOURCE_HASHES, "gate_passed": audit["passed"],
        "gpu": torch.cuda.get_device_name(0), "warmup_run": metadata["url"],
    }
    write_json(directory / "prepared.json", manifest)
    write_json(directory / "status.json", {"stage": "prepared", "seed": seed})
    return manifest


def branch(campaign, seed, arm):
    import torch

    validate_seed(seed)
    verify_sources()
    directory = root_path(campaign) / "production" / f"seed{seed}"
    manifest = read_json(directory / "prepared.json")
    if manifest["seed"] != seed or manifest["branch_start"] != 50000:
        raise AssertionError("Wrong seed or checkpoint age")
    if not manifest["gate_passed"] or manifest["source_sha256"] != SOURCE_HASHES:
        raise AssertionError("Pairing/source audit has not passed")
    if torch.cuda.get_device_name(0) != manifest["gpu"]:
        raise AssertionError("Warmup and continuations must use the same GPU model")
    for key in ("checkpoint", "pairs"):
        if sha256(manifest[key]) != manifest[key + "_sha256"]:
            raise AssertionError(f"Input artifact changed: {key}")
    output = PROJECT / "checkpoints" / campaign / "production" / f"seed{seed}" / arm
    arguments = branch_flags(seed, arm, manifest["checkpoint"],
                             manifest["pairs"], output)
    run_training("hard_fork_dt.py", arguments, directory / "records" / arm)
    if all((directory / "records" / key / "completed.json").exists()
           for key in ("dt", "v3")):
        audit_pair(directory, manifest["checkpoint"], seed, 50000, 100000, gate=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "prepare", "branch", "worker"))
    parser.add_argument("--campaign", default="v3-high-fresh345-20260919")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=("dt", "v3"))
    parser.add_argument("--record-dir")
    parser.add_argument("--entry", choices=("top_return_weighted_dt.py",
                                           "hard_fork_dt.py"))
    arguments, remaining = parser.parse_known_args()
    arguments.remaining = remaining
    os.chdir(PROJECT)
    if arguments.phase == "worker":
        journal_worker(arguments)
        return
    if remaining:
        parser.error(f"Unknown arguments: {remaining}")
    verify_sources()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This workflow requires a Slurm-allocated GPU")
    if arguments.phase != "smoke" and "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Production requires the preregistered RTX 4090 model")
    if arguments.phase == "smoke":
        manifests = [prepare(arguments.campaign, seed, smoke=True) for seed in SEEDS]
        for key in ("initial_model_sha256", "checkpoint_model_sha256"):
            if len({m[key] for m in manifests}) != 3:
                raise AssertionError(f"New seeds must have distinct weights: {key}")
        write_json(root_path(arguments.campaign) / "smoke/all_seeds_passed.json",
                   {"passed": True, "seeds": list(SEEDS), "manifests": manifests,
                    "source_sha256": SOURCE_HASHES})
    elif arguments.phase == "prepare":
        prepare(arguments.campaign, arguments.seed)
    else:
        branch(arguments.campaign, arguments.seed, arguments.arm)


if __name__ == "__main__":
    main()
