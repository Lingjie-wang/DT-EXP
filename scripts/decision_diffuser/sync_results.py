"""One-shot results-only W&B bridge for one authorized Decision Diffuser job.

No job submission, code capture, directory artifacts, datasets or checkpoints.
"""
import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import wandb

def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rows(path):
    if not path.exists():
        return []
    raw = path.read_text()
    lines = raw.splitlines()
    if raw and not raw.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def slurm(job):
    result = subprocess.run(
        ["squeue", "-h", "-j", job, "-o", "%T"], text=True, capture_output=True
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]
    result = subprocess.run(
        ["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State"],
        text=True,
        capture_output=True,
    )
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if fields[0] == job:
            return fields[1]
    return "UNKNOWN"


def evaluation_schedule(root, arm, protocol):
    path = root / "evaluation_schedule.json"
    if not path.exists():
        return protocol["eval_updates"], {}
    schedule = read(path)
    expected = schedule["arms"][arm]["expected_eval_updates"]
    metadata = {
        "requested_eval_updates": schedule["requested_eval_updates"],
        "unavailable_past_eval_updates": schedule["arms"][arm]["unavailable_past"],
        "eval_interval": schedule["interval"],
    }
    return expected, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["dense", "delayed"], required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    p = read(args.root / "protocol.json")
    directory = args.root / "wandb_sync" / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    lock = open(directory / "lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    allowed = [
        "method",
        "recipe",
        "env_name",
        "seed",
        "horizon",
        "n_diffusion_steps",
        "dim",
        "dim_mults",
        "inverse_hidden_dim",
        "condition_dropout",
        "guidance",
        "discount",
        "returns_scale",
        "test_return",
        "normalizer",
        "learning_rate",
        "batch_size",
        "gradient_accumulate_every",
        "total_updates",
        "ema_decay",
        "ema_every",
        "ema_start",
        "eval_updates",
        "eval_episodes",
        "eval_seed_start",
        "eval_torch_seed",
    ]
    config = {key: p[key] for key in allowed}
    config.update(reward_mode=args.arm, upstream_commit=p["upstream_commit"])
    identity = hashlib.sha256((p["wandb_group"] + args.arm).encode()).hexdigest()[:12]
    run = wandb.init(
        entity=p["wandb_entity"],
        project=p["wandb_project"],
        group=p["wandb_group"],
        id=identity,
        name=f"DecisionDiffuser-HCMR-{args.arm}-seed0-1M-official-code",
        resume="allow",
        mode="online",
        dir=str(directory),
        config=config,
        settings=wandb.Settings(
            _disable_stats=True,
            _disable_meta=True,
            disable_code=True,
            disable_git=True,
            console="off",
            start_method="thread",
        ),
    )
    run.define_metric("completed_updates")
    for prefix in ["train/*", "eval/*"]:
        run.define_metric(prefix, step_metric="completed_updates")
    marker = directory / "cursor.json"
    cursor = read(marker) if marker.exists() else dict(completed_updates=0, evaluated=[])
    work = args.root / args.arm
    expected_evaluations, schedule_metadata = evaluation_schedule(args.root, args.arm, p)
    run.config.update(
        dict(eval_updates=expected_evaluations, **schedule_metadata),
        allow_val_change=True,
    )
    startup_status = (
        read(work / "status.json")["status"]
        if (work / "status.json").exists()
        else "queued"
    )
    run.summary.update(
        dict(
            status=startup_status,
            completed_updates=cursor["completed_updates"],
            slurm_job_id=args.job,
            interpretation=(
                "Released defaults transferred to HalfCheetah; "
                "single seed; gamma=.99 in both reward settings"
            ),
        )
    )
    write(
        directory / "manifest.json", dict(url=run.url, run_id=identity, job_id=args.job)
    )
    # A bounded startup readback verifies identity/config/summary before waiting.
    for attempt in range(6):
        time.sleep(5)
        remote = wandb.Api(timeout=45).run(
            f"{p['wandb_entity']}/{p['wandb_project']}/{identity}"
        )
        if (
            remote.config.get("reward_mode") == args.arm
            and remote.summary.get("status") == startup_status
            and remote.config.get("eval_updates") == expected_evaluations
        ):
            write(
                directory / "startup_verification.json",
                dict(
                    verified=True,
                    url=run.url,
                    reward_mode=remote.config["reward_mode"],
                    total_updates=remote.config["total_updates"],
                    eval_updates=remote.config["eval_updates"],
                ),
            )
            break
    else:
        raise RuntimeError("W&B startup readback failed")
    print(json.dumps(read(directory / "startup_verification.json")), flush=True)
    previous_status = None
    try:
        for _ in range(15 * 24 * 60):
            history = rows(work / "metrics.jsonl")
            for row in history:
                if row["completed_updates"] <= cursor["completed_updates"]:
                    continue
                run.log(
                    {
                        "completed_updates": row["completed_updates"],
                        "train/loss": row["loss"],
                        "train/diffusion_loss": row["diffusion_loss"],
                        "train/inverse_loss": 2 * row["loss"] - row["diffusion_loss"],
                        "train/elapsed_seconds": row["elapsed_seconds"],
                        "train/cuda_peak_mib": row["cuda_peak_mib"],
                    }
                )
                cursor["completed_updates"] = row["completed_updates"]
            evaluations = [read(path) for path in sorted(work.glob("eval_*.json"))]
            for value in evaluations:
                step = value["completed_updates"]
                if step in cursor["evaluated"]:
                    continue
                run.log(
                    {
                        "completed_updates": step,
                        "eval/normalized_score_mean": value["mean_score"],
                        "eval/normalized_score_std": value["std_score"],
                        "eval/episodes": len(value["episodes"]),
                        "eval/test_return": value["test_return"],
                    }
                )
                table = wandb.Table(
                    columns=["episode_seed", "raw_return", "normalized_score", "length"]
                )
                for episode in value["episodes"]:
                    table.add_data(*[episode[key] for key in table.columns])
                run.log({f"tables/episodes_{step}": table})
                artifact = wandb.Artifact(
                    f"dd-{args.arm}-{identity}-{step}", type="evaluation-results"
                )
                # Only this numeric episode-result file is permitted.
                artifact.add_file(
                    str(work / f"eval_{step:07d}.json"), name="evaluation.json"
                )
                run.log_artifact(artifact)
                cursor["evaluated"].append(step)
            write(marker, cursor)
            state = slurm(args.job)
            local = read(work / "status.json") if (work / "status.json").exists() else {}
            status = local.get("status", "starting" if state == "RUNNING" else "queued")
            failed = any(
                state.startswith(x)
                for x in [
                    "FAILED",
                    "CANCELLED",
                    "TIMEOUT",
                    "OUT_OF_MEMORY",
                    "NODE_FAIL",
                    "PREEMPTED",
                    "BOOT_FAIL",
                ]
            )
            if failed and status != "completed":
                status = "job_failed"
            supplement_status = work / "supplemental/status.json"
            if supplement_status.exists():
                supplemental = read(supplement_status)
                if supplemental["status"] == "failed":
                    raise RuntimeError(supplemental["error"])
            if status == "completed" and (
                sorted(cursor["evaluated"]) != expected_evaluations
            ):
                status = "waiting_for_evaluation"
            run.summary.update(
                dict(
                    status=status,
                    completed_updates=cursor["completed_updates"],
                    slurm_state=state,
                )
            )
            write(
                directory / "status.json",
                dict(
                    status=status,
                    slurm_state=state,
                    completed_updates=cursor["completed_updates"],
                    evaluation_points=len(evaluations),
                    updated_unix=time.time(),
                ),
            )
            if status != previous_status:
                print(json.dumps(read(directory / "status.json")), flush=True)
                previous_status = status
            if status == "completed":
                assert cursor["completed_updates"] == p["total_updates"]
                assert sorted(cursor["evaluated"]) == expected_evaluations
                run.summary["result/final_normalized_score"] = evaluations[-1][
                    "mean_score"
                ]
                url = run.url
                run.finish()
                remote = wandb.Api(timeout=45).run(
                    f"{p['wandb_entity']}/{p['wandb_project']}/{identity}"
                )
                assert math.isclose(
                    remote.summary["result/final_normalized_score"],
                    evaluations[-1]["mean_score"],
                    abs_tol=1e-8,
                )
                for step in expected_evaluations:
                    assert (
                        remote.summary[f"tables/episodes_{step}"]["nrows"]
                        == p["eval_episodes"]
                    )
                records = list(
                    remote.scan_history(
                        keys=["completed_updates", "eval/normalized_score_mean"]
                    )
                )
                mapped = {
                    r["completed_updates"]: r["eval/normalized_score_mean"]
                    for r in records
                }
                for value in evaluations:
                    assert math.isclose(
                        mapped[value["completed_updates"]],
                        value["mean_score"],
                        abs_tol=1e-8,
                    )
                write(
                    directory / "completion_verification.json",
                    dict(
                        verified=True,
                        url=url,
                        completed_updates=cursor["completed_updates"],
                        evaluation_points=len(evaluations),
                    ),
                )
                return
            if failed or (
                state.startswith("COMPLETED") and status != "waiting_for_evaluation"
            ):
                raise RuntimeError(f"Job ended {state} without complete result")
            time.sleep(60)
        raise TimeoutError("One-shot bridge window expired; no training was modified")
    except BaseException as error:
        write(directory / "error.json", dict(error=repr(error)))
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
