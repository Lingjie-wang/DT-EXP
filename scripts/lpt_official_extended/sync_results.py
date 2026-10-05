"""Publish official stdout results from the login node; never alter training."""

import argparse
import fcntl
import json
import math
import subprocess
import time
from pathlib import Path

import wandb

def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


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
        if fields[0] == job:
            return fields[1]
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("dense", "delayed"), required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    p = read(args.root / "protocol.json")
    directory = args.root / "wandb_sync" / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    lock = open(directory / "lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    excluded = {"upstream_sha256", "harness_sha256", "dataset_sha256"}
    config = {key: value for key, value in p.items() if key not in excluded}
    config["reward_mode"] = args.arm
    identity = p["wandb_run_id"]
    run = wandb.init(
        entity=p["wandb_entity"],
        project=p["wandb_project"],
        group=p["wandb_group"],
        id=identity,
        name=p["wandb_name"],
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
    for prefix in ("train/*", "eval/*"):
        run.define_metric(prefix, step_metric="completed_updates")
    cursor_path = directory / "cursor.json"
    cursor = read(cursor_path) if cursor_path.exists() else dict(updates=0, evaluated=[])
    run.summary.update(dict(status="queued", slurm_job_id=args.job))
    write(
        directory / "manifest.json", dict(url=run.url, run_id=identity, job_id=args.job)
    )
    remote_path = f"{p['wandb_entity']}/{p['wandb_project']}/{identity}"
    work = args.root / args.arm
    try:
        for _ in range(6):
            time.sleep(5)
            remote = wandb.Api(timeout=45).run(remote_path)
            if remote.config.get("reward_mode") == args.arm:
                assert remote.config["effective_batch_size"] == 202
                assert remote.config["source_modifications"] == []
                assert remote.config["model_initialization_seed"] == p[
                    "model_initialization_seed"
                ]
                assert remote.config["epochs"] == p["epochs"]
                assert remote.name == p["wandb_name"]
                write(directory / "startup_verification.json", dict(verified=True))
                break
        else:
            raise RuntimeError("W&B startup readback failed")
        # Read-only monitor bounded to this campaign. No Slurm mutations.
        for _ in range(15 * 24 * 60):
            for row in rows(work / "metrics.jsonl"):
                if row["completed_updates"] <= cursor["updates"]:
                    continue
                run.log(
                    dict(
                        completed_updates=row["completed_updates"],
                        **{
                            f"train/{key}": value
                            for key, value in row.items()
                            if key != "completed_updates"
                            and isinstance(value, (int, float))
                        },
                    )
                )
                cursor["updates"] = row["completed_updates"]
            evaluations = [read(path) for path in sorted(work.glob("eval_*.json"))]
            for evaluation in evaluations:
                step = evaluation["completed_updates"]
                if step in cursor["evaluated"]:
                    continue
                metrics = {
                    key: value for key, value in evaluation.items() if key != "episodes"
                }
                table = wandb.Table(columns=["episode", "raw_return", "length"])
                for episode in evaluation["episodes"]:
                    table.add_data(*[episode[key] for key in table.columns])
                metrics[f"tables/episodes_{step}"] = table
                run.log(metrics)
                cursor["evaluated"].append(step)
            write(cursor_path, cursor)
            state = slurm(args.job)
            local = read(work / "status.json") if (work / "status.json").exists() else {}
            status = local.get("status", "starting" if state == "RUNNING" else "queued")
            failed = any(
                state.startswith(s)
                for s in (
                    "FAILED",
                    "CANCELLED",
                    "TIMEOUT",
                    "OUT_OF_MEMORY",
                    "NODE_FAIL",
                    "PREEMPTED",
                )
            )
            if failed and status != "completed":
                status = "job_failed"
            current = dict(
                status=status,
                slurm_state=state,
                completed_updates=local.get("completed_updates", 0),
                evaluation_points=len(evaluations),
            )
            run.summary.update(current)
            write(directory / "status.json", current)
            print(json.dumps(current), flush=True)
            if status == "completed":
                assert sorted(cursor["evaluated"]) == p["eval_updates"]
                assert cursor["updates"] == p["total_updates"]
                run.summary["result/final_normalized_score"] = evaluations[-1][
                    "eval/norm_score"
                ]
                run.finish()
                remote = wandb.Api(timeout=45).run(remote_path)
                assert math.isclose(
                    remote.summary["result/final_normalized_score"],
                    evaluations[-1]["eval/norm_score"],
                    abs_tol=1e-8,
                )
                records = list(
                    remote.scan_history(keys=["completed_updates", "eval/norm_score"])
                )
                values = {
                    row["completed_updates"]: row["eval/norm_score"] for row in records
                }
                for evaluation in evaluations:
                    assert math.isclose(
                        values[evaluation["completed_updates"]],
                        evaluation["eval/norm_score"],
                        abs_tol=1e-8,
                    )
                write(directory / "completion_verification.json", dict(verified=True))
                return
            if status in ("failed", "job_failed") or state.startswith("COMPLETED"):
                raise RuntimeError(f"Job ended without full results: {state}; {status}")
            time.sleep(60)
        raise TimeoutError("W&B bridge monitoring window expired")
    except BaseException as error:
        write(directory / "error.json", dict(error=repr(error)))
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
