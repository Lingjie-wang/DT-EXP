"""Read-only, bounded login-node W&B bridge for an original DT Slurm job."""

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
        ["squeue", "-h", "-j", job, "-o", "%T"],
        capture_output=True,
        text=True,
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
    config = {
        key: p[key]
        for key in (
            "method",
            "recipe",
            "upstream_commit",
            "env_name",
            "eval_env",
            "seed",
            "batch_size",
            "context_length",
            "hidden_size",
            "num_layers",
            "num_heads",
            "activation",
            "dropout",
            "learning_rate",
            "optimizer",
            "weight_decay",
            "warmup_steps",
            "grad_clip",
            "reward_scale",
            "total_updates",
            "eval_every",
            "eval_episodes",
            "eval_seed_start",
            "target_returns",
            "sampling",
            "loss",
            "pct_traj",
            "main_result",
            "limitations",
        )
    }
    config["reward_mode"] = args.arm
    identity = hashlib.sha256((p["wandb_group"] + args.arm).encode()).hexdigest()[:12]
    run = wandb.init(
        entity=p["wandb_entity"],
        project=p["wandb_project"],
        group=p["wandb_group"],
        id=identity,
        name=f"OriginalDT-HCMR-{args.arm}-seed{p['seed']}-100k-b64",
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
    for _ in range(6):
        time.sleep(5)
        remote = wandb.Api(timeout=45).run(remote_path)
        if remote.config.get("reward_mode") == args.arm:
            assert remote.config["batch_size"] == 64
            write(
                directory / "startup_verification.json", dict(verified=True, url=run.url)
            )
            break
    else:
        raise RuntimeError("W&B startup readback failed")
    work = args.root / args.arm
    expected = list(range(p["eval_every"], p["total_updates"] + 1, p["eval_every"]))
    try:
        # Bounded to the two authorized experiments; never submits or changes jobs.
        for _ in range(15 * 24 * 60):
            for row in rows(work / "metrics.jsonl"):
                if row["completed_updates"] <= cursor["updates"]:
                    continue
                run.log(
                    dict(
                        completed_updates=row["completed_updates"],
                        **{
                            f"train/{key}": row[key]
                            for key in ("loss", "learning_rate", "elapsed_seconds")
                        },
                    )
                )
                cursor["updates"] = row["completed_updates"]
            evaluations = [read(path) for path in sorted(work.glob("eval_*.json"))]
            for evaluation in evaluations:
                step = evaluation["completed_updates"]
                if step in cursor["evaluated"]:
                    continue
                metrics = dict(completed_updates=step)
                for target in evaluation["targets"]:
                    prefix = f"eval/{target['target_return']}.0"
                    metrics[prefix + "_normalized_score_mean"] = target["mean_score"]
                    metrics[prefix + "_normalized_score_std"] = target["std_score"]
                    table = wandb.Table(
                        columns=[
                            "episode_seed",
                            "raw_return",
                            "normalized_score",
                            "length",
                        ]
                    )
                    for episode in target["episodes"]:
                        table.add_data(*[episode[key] for key in table.columns])
                    metrics[f"tables/episodes_{step}_{target['target_return']}"] = table
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
                completed_updates=local.get("completed_updates", cursor["updates"]),
                evaluation_points=len(evaluations),
            )
            run.summary.update(current)
            write(directory / "status.json", current)
            print(json.dumps(current), flush=True)
            if status == "completed":
                assert sorted(cursor["evaluated"]) == expected
                assert cursor["updates"] == p["total_updates"]
                final = evaluations[-1]
                for target in final["targets"]:
                    run.summary[f"result/final_{target['target_return']}"] = target[
                        "mean_score"
                    ]
                run.finish()
                remote = wandb.Api(timeout=45).run(remote_path)
                for target in final["targets"]:
                    assert math.isclose(
                        remote.summary[f"result/final_{target['target_return']}"],
                        target["mean_score"],
                        abs_tol=1e-8,
                    )
                for target in p["target_returns"]:
                    key = f"eval/{target}.0_normalized_score_mean"
                    records = list(remote.scan_history(keys=["completed_updates", key]))
                    values = {r["completed_updates"]: r[key] for r in records}
                    for evaluation in evaluations:
                        value = next(
                            t
                            for t in evaluation["targets"]
                            if t["target_return"] == target
                        )
                        assert math.isclose(
                            values[evaluation["completed_updates"]],
                            value["mean_score"],
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
