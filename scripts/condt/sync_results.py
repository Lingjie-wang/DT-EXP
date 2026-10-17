"""Read-only W&B bridge for the separately preserved ConDT author-code campaign.

This process observes JSON files and Slurm state. It never launches training or
changes its sampling, optimizer, checkpoint, or evaluation state.

Upload schema ``condt-metrics-v2`` is an explicit allowlist: experiment settings,
public source/dataset identifiers, aggregate data statistics, package versions,
training metrics, per-episode returns/lengths, and status. Source files, patches,
installation logs, raw protocols/hash inventories, datasets, and checkpoints stay
local. No file artifact API is used; W&B Tables contain evaluation records only.
"""

import argparse
import fcntl
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

CONFIG_SCALARS = (
    "method", "upstream_url", "upstream_commit", "paper_url", "dataset",
    "dataset_sha256", "hdf5_sha256", "seed", "training_seed_count",
    "eval_environment", "target_return", "reward_mode", "reward_scale",
    "main_updates", "total_updates", "eval_every", "eval_episodes_per_seed",
    "batch_size", "context_length", "hidden_size", "layers", "heads", "dropout",
    "activation", "learning_rate", "weight_decay", "warmup_steps", "optimizer",
    "gradient_clip", "normalized_score_min", "normalized_score_max", "primary_result",
)
ARM_SCALARS = (
    "model_type", "pretrain", "pretrain_updates", "total_updates", "beta",
    "learning_rate", "weight_decay", "warmup_steps", "optimizer", "batch_size",
    "num_samples_simclr", "compression_dim", "temperature",
)
DATA_SCALARS = (
    "dataset", "source_url_in_d4rl", "download_url", "source_sha256", "source_bytes",
    "author_commit", "conversion_reference", "conversion_rule",
    "conversion_pickle_protocol", "pickle_sha256", "pickle_bytes", "raw_transitions",
    "raw_terminal_count", "raw_timeout_count", "raw_terminal_and_timeout_count",
    "converted_trajectories", "converted_transitions",
    "discarded_incomplete_tail_transitions",
)
PACKAGES = (
    "python", "torch", "gym", "numpy", "transformers", "wandb", "h5py", "mujoco-py",
    "d4rl", "ray", "pytorch-metric-learning", "pandas", "protobuf",
)


def scalar_fields(source, names):
    """Select named scalar values without copying nested objects or unknown fields."""
    if not isinstance(source, dict):
        return {}
    return {
        key: source[key]
        for key in names
        if key in source and isinstance(source[key], (str, bool, int, float))
    }


def sanitized_config(protocol, arm):
    """Build the entire allowed configuration payload; never forward raw protocol."""
    config = scalar_fields(protocol, CONFIG_SCALARS)
    config.update(arm=arm, observation_schema="condt-metrics-v2")
    for key in ("evaluation_seeds", "eval_updates"):
        if key in protocol:
            config[key] = [int(value) for value in protocol[key]]
    for key in ("preserved_author_behaviors", "limitations", "source_changes"):
        if key in protocol:
            config[key] = [value for value in protocol[key] if isinstance(value, str)]
    config["arms"] = {
        name: scalar_fields(protocol.get("arms", {}).get(name, {}), ARM_SCALARS)
        for name in ("dt", "condt")
    }
    audit = protocol.get("data_audit", {})
    data = scalar_fields(audit, DATA_SCALARS)
    for key in (
        "raw_reward_statistics", "trajectory_length_statistics",
        "trajectory_return_statistics",
    ):
        if key in audit:
            data[key] = {
                name: value
                for name, value in scalar_fields(
                    audit[key], ("mean", "std", "min", "max")
                ).items()
                if isinstance(value, (int, float))
            }
    if "author_prep_data_filter" in audit:
        data["author_prep_data_filter"] = scalar_fields(
            audit["author_prep_data_filter"],
            (
                "rule", "note", "dropped_trajectories", "dropped_transitions",
                "retained_trajectories", "retained_transitions",
            ),
        )
    config["data_statistics"] = data
    differences = protocol.get("runtime_differences", {})
    config["runtime_differences"] = {
        name: scalar_fields(differences[name], ("author", "runtime"))
        for name in PACKAGES if name in differences
    }
    return config


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rows(path):
    """Ignore an incomplete final JSONL record while its writer is active."""
    if not path.exists():
        return []
    raw = path.read_text()
    lines = raw.splitlines()
    if raw and not raw.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def run_identity(protocol, arm):
    key = "/".join(
        str(protocol[field])
        for field in ("wandb_entity", "wandb_project", "wandb_group")
    )
    return hashlib.sha256((key + "/" + arm).encode()).hexdigest()[:16]


def expected_points(protocol):
    if "eval_updates" in protocol:
        points = [int(value) for value in protocol["eval_updates"]]
        if not points or points[0] != 0 or points != sorted(set(points)):
            raise ValueError("eval_updates must start at zero and increase strictly")
        return points
    total = int(protocol.get("main_updates", protocol.get("total_updates", 100000)))
    interval = int(protocol.get("eval_every", 10000))
    if total < 0 or interval <= 0 or total % interval:
        raise ValueError("Expected main updates must be divisible by eval_every")
    return list(range(0, total + 1, interval))


def evaluation_spec(protocol):
    seeds = protocol.get("eval_seeds", protocol.get("evaluation_seeds", [1, 5, 10]))
    episodes = protocol.get("eval_episodes", protocol.get("eval_episodes_per_seed", 100))
    return [int(seed) for seed in seeds], int(episodes)


def validate_evaluation(evaluation, protocol):
    seeds, episodes = evaluation_spec(protocol)
    outputs = evaluation["outputs"]
    if {int(seed) for seed in outputs} != set(seeds):
        raise ValueError("Evaluation file does not have the planned environment seeds")
    if any(len(output["returns"]) != episodes for output in outputs.values()):
        raise ValueError("Evaluation file does not have the planned episode count")


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def evaluation_metrics(evaluation, protocol):
    """Keep episode dispersion distinct from dispersion across eval-seed means.

    These seeds re-seed the environment for one trained model. Their standard
    error is not a training-seed uncertainty estimate.
    """
    minimum = float(protocol["normalized_score_min"])
    maximum = float(protocol["normalized_score_max"])
    if not math.isfinite(maximum - minimum) or maximum <= minimum:
        raise ValueError("Invalid D4RL normalization reference range")
    scale = 100.0 / (maximum - minimum)
    main_updates = int(evaluation["main_updates"])
    pretrain_updates = int(evaluation["pretrain_updates"])
    metrics = {
        "main_updates": main_updates,
        "pretrain_updates": pretrain_updates,
        "total_updates": main_updates + pretrain_updates,
        "eval/phase": evaluation["phase"],
        "eval/wall_seconds": float(evaluation["wall_seconds"]),
    }
    table = []
    all_returns = []
    seed_means = []
    outputs = evaluation["outputs"]
    if not outputs:
        raise ValueError("An evaluation must contain environment seeds")
    for seed, output in sorted(outputs.items(), key=lambda pair: int(pair[0])):
        returns = [float(value) for value in output["returns"]]
        lengths = output["lengths"]
        if not returns or len(returns) != len(lengths):
            raise ValueError(f"Mismatched or empty evaluation arrays for seed {seed}")
        if not all(math.isfinite(value) for value in returns):
            raise ValueError(f"Non-finite evaluation return for seed {seed}")
        mean = statistics.mean(returns)
        seed_means.append(mean)
        all_returns.extend(returns)
        prefix = f"eval/env_seed_{seed}/"
        metrics[prefix + "raw_return_mean"] = mean
        metrics[prefix + "raw_return_episode_std"] = sample_std(returns)
        metrics[prefix + "normalized_score_mean"] = (mean - minimum) * scale
        metrics[prefix + "episode_count"] = len(returns)
        for episode_index, (raw_return, length) in enumerate(zip(returns, lengths)):
            table.append(
                [
                    main_updates,
                    int(seed),
                    episode_index,
                    raw_return,
                    (raw_return - minimum) * scale,
                    int(length),
                ]
            )
    episode_mean = statistics.mean(all_returns)
    seed_mean = statistics.mean(seed_means)
    seed_std = sample_std(seed_means)
    metrics.update(
        {
            "eval/raw_return_mean": episode_mean,
            "eval/normalized_score_mean": (episode_mean - minimum) * scale,
            "eval/raw_return_episode_std": sample_std(all_returns),
            "eval/normalized_score_episode_std": sample_std(all_returns) * scale,
            "eval/raw_return_eval_seed_mean": seed_mean,
            "eval/raw_return_eval_seed_std": seed_std,
            "eval/raw_return_eval_seed_se": seed_std / math.sqrt(len(seed_means)),
            "eval/normalized_score_eval_seed_mean": (seed_mean - minimum) * scale,
            "eval/normalized_score_eval_seed_std": seed_std * scale,
            "eval/normalized_score_eval_seed_se": (
                seed_std * scale / math.sqrt(len(seed_means))
            ),
            "eval/episode_count": len(all_returns),
            "eval/evaluation_seed_count": len(seed_means),
        }
    )
    return metrics, table


def training_metrics(row):
    phase = row["phase"]
    if phase not in ("pretrain", "main"):
        raise ValueError(f"Unknown training phase: {phase}")
    main_updates = int(row["main_updates"])
    total_updates = int(row["total_updates"])
    metrics = {
        "main_updates": main_updates,
        "total_updates": total_updates,
        "pretrain_updates": total_updates - main_updates,
        "train/phase": phase,
        "train/elapsed_seconds": float(row["elapsed_seconds"]),
    }
    for key in ("train_loss", "contrastive_loss"):
        if row.get(key) is not None:
            metrics[f"train/{key}"] = float(row[key])
            metrics[f"{phase}/{key}"] = float(row[key])
    for index, value in enumerate(row["learning_rates"]):
        metrics[f"train/learning_rate_{index}"] = float(value)
        metrics[f"{phase}/learning_rate_{index}"] = float(value)
    return metrics


def pending_events(work, cursor):
    """Merge training and evaluation records into optimizer-update order."""
    events = []
    records = rows(work / "metrics.jsonl")
    if len(records) < cursor["metrics_rows"]:
        raise RuntimeError("Training log was truncated; refusing to lose cursor state")
    previous = -1
    for index, row in enumerate(records):
        update = int(row["total_updates"])
        if update < previous:
            raise ValueError("Training JSONL is not ordered by total_updates")
        previous = update
        if index >= cursor["metrics_rows"]:
            events.append((update, 0, index, row))
    evaluations = []
    for path in sorted((work / "evaluations").glob("eval_main_*.json")):
        evaluation = read(path)
        step = int(evaluation["main_updates"])
        evaluations.append(evaluation)
        if step not in cursor["evaluated"]:
            total = step + int(evaluation["pretrain_updates"])
            events.append((total, 1, step, evaluation))
    return sorted(events, key=lambda event: event[:3]), evaluations


def sync_pending(run, sdk, work, cursor, protocol):
    events, evaluations = pending_events(work, cursor)
    for _, kind, identifier, record in events:
        if kind == 0:
            run.log(training_metrics(record))
            cursor["metrics_rows"] = identifier + 1
            cursor["latest_main_updates"] = int(record["main_updates"])
        else:
            validate_evaluation(record, protocol)
            metrics, table = evaluation_metrics(record, protocol)
            metrics[f"episodes/main_{identifier:06d}"] = sdk.Table(
                columns=[
                    "main_updates",
                    "evaluation_seed",
                    "episode_index",
                    "raw_return",
                    "normalized_score",
                    "length",
                ],
                data=table,
            )
            run.log(metrics)
            cursor["evaluated"].append(identifier)
    return evaluations


def slurm(job):
    result = subprocess.run(
        ["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]
    result = subprocess.run(
        ["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) > 1 and fields[0] == job:
            return fields[1]
    return "UNKNOWN"


def verify_remote(sdk, remote_path, arm, protocol, evaluations=None):
    """Read back config initially; after finishing verify every evaluation point."""
    remote = sdk.Api(timeout=45).run(remote_path)
    if remote.config.get("arm") != arm:
        raise RuntimeError("W&B arm configuration has not reached the API")
    if remote.config.get("upstream_commit") != protocol["upstream_commit"]:
        raise RuntimeError("W&B upstream revision readback mismatch")
    if evaluations is None:
        return
    key = "eval/normalized_score_mean"
    records = remote.scan_history(keys=["main_updates", key])
    values = {int(row["main_updates"]): float(row[key]) for row in records}
    for evaluation in evaluations:
        metrics, _ = evaluation_metrics(evaluation, protocol)
        step = int(evaluation["main_updates"])
        if step not in values or not math.isclose(
            values[step], metrics[key], rel_tol=1e-10, abs_tol=1e-8
        ):
            raise RuntimeError(f"W&B evaluation readback mismatch at update {step}")
    final = max(evaluations, key=lambda item: int(item["main_updates"]))
    metrics, _ = evaluation_metrics(final, protocol)
    actual = remote.summary.get("result/final_normalized_score")
    if actual is None or not math.isclose(actual, metrics[key], abs_tol=1e-8):
        raise RuntimeError("W&B final score summary readback mismatch")


def retry_verification(sdk, remote_path, arm, protocol, evaluations=None):
    last_error = None
    for _ in range(12):
        try:
            verify_remote(sdk, remote_path, arm, protocol, evaluations)
            return
        except Exception as error:
            last_error = error
            time.sleep(5)
    raise RuntimeError("W&B readback failed after retries") from last_error


def initialize_run(sdk, directory, **kwargs):
    """Retry transient initialization failures without changing the stable run ID."""
    last_error = None
    for attempt in range(6):
        try:
            return sdk.init(**kwargs)
        except Exception as error:
            last_error = error
            write(
                directory / "initialization_retry.json",
                dict(attempt=attempt + 1, error=repr(error), time=time.time()),
            )
            if getattr(sdk, "run", None) is not None:
                sdk.finish(exit_code=1)
            if attempt < 5:
                time.sleep(min(10 * (attempt + 1), 60))
    raise RuntimeError("W&B initialization failed after six attempts") from last_error


def monitor(args, sdk, directory):
    protocol = read(args.root / "protocol.json")
    identity = run_identity(protocol, args.arm)
    work = args.root / args.arm
    config = sanitized_config(protocol, args.arm)
    write(
        directory / "upload_scope.json",
        dict(
            schema="condt-metrics-v2",
            config=config,
            metrics="Training losses, learning rates, update counts, elapsed time",
            evaluations="Per-episode returns/lengths and aggregate scores/statistics",
            status="Job state and update counts; raw errors remain local",
            file_artifacts=False,
        ),
    )
    run = initialize_run(
        sdk,
        directory,
        entity=protocol["wandb_entity"],
        project=protocol["wandb_project"],
        group=protocol["wandb_group"],
        id=identity,
        name=f"ConDT-Author-Code-{args.arm}-seed0-Hopper-medium",
        resume="allow",
        mode="online",
        dir=str(directory),
        config=config,
        settings=sdk.Settings(
            _disable_stats=True,
            _disable_meta=True,
            disable_code=True,
            disable_git=True,
            console="off",
            start_method="thread",
        ),
    )
    finished = False
    try:
        for metric in ("main_updates", "pretrain_updates", "total_updates"):
            run.define_metric(metric)
        for prefix, axis in (
            ("train/*", "total_updates"),
            ("pretrain/*", "pretrain_updates"),
            ("main/*", "main_updates"),
            ("eval/*", "main_updates"),
        ):
            run.define_metric(prefix, step_metric=axis)
        cursor_path = directory / "cursor.json"
        cursor = (
            read(cursor_path)
            if cursor_path.exists()
            else dict(
                metrics_rows=0,
                evaluated=[],
                latest_main_updates=0,
                started_at=time.time(),
            )
        )
        run.summary.update(
            dict(
                status="queued",
                slurm_job_id=args.job,
                experiment_config=config,
                training_seed_count=1,
                training_seed=0,
                evaluation_seeds=evaluation_spec(protocol)[0],
                evaluation_episodes_per_seed=evaluation_spec(protocol)[1],
                uncertainty_scope=(
                    "Sample std (ddof=1) and SE across three environment-seed means "
                    "for one trained model, not across training seeds. Episode std "
                    "describes individual episode returns; it is reported separately."
                ),
                evaluation_zero_semantics=(
                    "After contrastive pretraining, before main training"
                    if args.arm == "condt"
                    else "Initialized model, before main training"
                ),
            )
        )
        write(
            directory / "manifest.json", dict(url=run.url, run_id=identity, job=args.job)
        )
        remote_path = (
            f"{protocol['wandb_entity']}/{protocol['wandb_project']}/{identity}"
        )
        expected = expected_points(protocol)
        startup_verified = False
        deadline = cursor["started_at"] + args.max_days * 24 * 60 * 60
        while time.time() < deadline:
            evaluations = sync_pending(run, sdk, work, cursor, protocol)
            write(cursor_path, cursor)
            state = slurm(args.job)
            path = work / "status.json"
            local = read(path) if path.exists() else {}
            status = local.get("status", "training" if state == "RUNNING" else "queued")
            failed = state.startswith(
                (
                    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                    "NODE_FAIL", "PREEMPTED",
                )
            )
            if failed and status != "completed":
                status = "job_failed"
            current = dict(
                status=status,
                slurm_state=state,
                main_updates=local.get("main_updates", cursor["latest_main_updates"]),
                pretrain_updates=local.get("pretrain_updates", 0),
                evaluation_points=len(cursor["evaluated"]),
            )
            if local.get("error"):
                current["error_recorded_locally"] = True
            run.summary.update(current)
            write(directory / "status.json", current)
            print(json.dumps(current), flush=True)
            if not startup_verified:
                retry_verification(sdk, remote_path, args.arm, protocol)
                write(
                    directory / "startup_verification.json",
                    dict(verified=True, url=run.url, run_id=identity),
                )
                startup_verified = True
            if status == "completed":
                # The job may have completed after this poll read the logs. Re-read
                # after the terminal marker so its final files cannot be missed.
                evaluations = sync_pending(run, sdk, work, cursor, protocol)
                write(cursor_path, cursor)
                if sorted(cursor["evaluated"]) != expected:
                    raise RuntimeError("Completed job has missing/extra eval points")
                if cursor["latest_main_updates"] != expected[-1]:
                    raise RuntimeError("Completed job has no final training metric")
                final = max(evaluations, key=lambda item: int(item["main_updates"]))
                metrics, _ = evaluation_metrics(final, protocol)
                run.summary.update(
                    {
                        "evaluation_points": len(cursor["evaluated"]),
                        "result/final_main_updates": expected[-1],
                        "result/final_raw_return": metrics["eval/raw_return_mean"],
                        "result/final_normalized_score": metrics[
                            "eval/normalized_score_mean"
                        ],
                        "result/final_normalized_score_eval_seed_std": metrics[
                            "eval/normalized_score_eval_seed_std"
                        ],
                        "result/final_normalized_score_eval_seed_se": metrics[
                            "eval/normalized_score_eval_seed_se"
                        ],
                    }
                )
                run.finish()
                finished = True
                retry_verification(sdk, remote_path, args.arm, protocol, evaluations)
                write(
                    directory / "completion_verification.json",
                    dict(verified=True, evaluation_points=expected, url=run.url),
                )
                return
            if status in ("failed", "job_failed") or state.startswith("COMPLETED"):
                raise RuntimeError(f"Job ended without full results: {current}")
            time.sleep(args.poll_seconds)
        raise TimeoutError("W&B bridge monitoring window expired (maximum seven days)")
    except BaseException as error:
        failure = dict(status="sync_failed", error=repr(error))
        write(directory / "status.json", failure)
        if not finished:
            run.summary.update(dict(status="sync_failed", error_recorded_locally=True))
            run.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("dt", "condt"), required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--max-days", type=float, default=7)
    args = parser.parse_args()
    if not 0 < args.max_days <= 7 or not 0 < args.poll_seconds <= 60:
        parser.error("max-days must be in (0, 7]; poll-seconds must be in (0, 60]")
    args.root = args.root.resolve()
    directory = args.root / "wandb_sync" / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            import wandb

            monitor(args, wandb, directory)
        except BaseException as error:
            write(directory / "error.json", dict(error=repr(error), time=time.time()))
            raise


if __name__ == "__main__":
    main()
