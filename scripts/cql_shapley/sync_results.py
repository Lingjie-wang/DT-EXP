"""Upload local CQL metrics from the login node without changing training."""

import argparse
import fcntl
import math
import subprocess
import time
from pathlib import Path

import wandb
from common import read, write

def slurm(job):
    r = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip().splitlines()[0]
    r = subprocess.run(["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        fields = line.split("|")
        if fields[0] == job:
            return fields[1]
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["uniform", "shapley", "dense"], required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    p = read(args.root / "protocol.json")
    spec = p["runs"][args.arm]
    directory = args.root / "wandb_sync" / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    lock = open(directory / "lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {k: v for k, v in spec.items() if k != "data_sha256"}
    config.update(p["retained"], reward_mode=spec["reward_mode"], changes=p["changes"],
                  dataset_audit=read(args.root / args.arm / "audit.json"))
    run = wandb.init(
        entity=p["wandb_entity"], project=p["wandb_project"], group=p["wandb_group"],
        id=spec["wandb_id"], name=spec["wandb_name"], config=config,
        resume="allow", mode="online", dir=str(directory),
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True,
                               disable_code=True, disable_git=True,
                               console="off", start_method="thread"),
    )
    run.define_metric("completed_updates")
    for metric in ["train/*", "eval/*", "d4rl_normalized_score"]:
        run.define_metric(metric, step_metric="completed_updates")
    run.summary.update(dict(status="queued", slurm_job_id=args.job))
    write(directory / "manifest.json", dict(
        run_id=run.id, name=run.name, url=run.url, job_id=args.job,
    ))
    remote_path = f"{p['wandb_entity']}/{p['wandb_project']}/{run.id}"
    cursor_file = directory / "cursor.json"
    cursor = (read(cursor_file) if cursor_file.exists()
              else dict(updates=0, evaluations=[]))
    work = args.root / args.arm / "training"
    try:
        for _ in range(6):
            time.sleep(5)
            remote = wandb.Api(timeout=40).run(remote_path)
            if remote.name == spec["wandb_name"]:
                assert remote.config["updates"] == 1000000
                assert remote.config["reward_mode"] == spec["reward_mode"]
                write(directory / "startup_verification.json", dict(verified=True))
                break
        else:
            raise RuntimeError("W&B startup readback failed")
        for _ in range(15 * 24 * 60):
            metrics = work / "metrics.jsonl"
            if metrics.exists():
                import json

                for line in metrics.read_text().splitlines(keepends=True):
                    if not line.endswith("\n"):
                        continue
                    row = json.loads(line)
                    step = row.pop("completed_updates")
                    if step > cursor["updates"]:
                        run.log(dict(completed_updates=step,
                                     **{f"train/{k}": v for k, v in row.items()}))
                        cursor["updates"] = step
            for path in sorted(work.glob("eval_*.json")):
                row = read(path)
                step = row["completed_updates"]
                if step not in cursor["evaluations"]:
                    run.log(dict(completed_updates=step,
                                 d4rl_normalized_score=row["normalized_score"],
                                 **{"eval/normalized_score": row["normalized_score"],
                                    "eval/raw_return_mean": row["raw_return_mean"]}))
                    cursor["evaluations"].append(step)
            write(cursor_file, cursor)
            state = slurm(args.job)
            local = read(work / "status.json") if (work / "status.json").exists() else {}
            status = local.get("status", "preflight" if state == "RUNNING" else "queued")
            if any(state.startswith(v) for v in ["FAILED", "CANCELLED", "TIMEOUT",
                                                "OUT_OF_MEMORY", "NODE_FAIL"]):
                status = "job_failed"
            current = dict(status=status, slurm_state=state,
                           completed_updates=local.get("completed_updates", 0))
            run.summary.update(current)
            write(directory / "status.json", current)
            if status == "completed":
                assert cursor["updates"] == spec["updates"]
                assert cursor["evaluations"] == list(range(5000, 1000001, 5000))
                score = local["final_normalized_score"]
                run.summary["result/final_normalized_score"] = score
                final_ten = [read(path)["normalized_score"]
                             for path in sorted(work.glob("eval_*.json"))[-10:]]
                run.summary["result/last_ten_eval_mean"] = sum(final_ten) / 10
                run.finish()
                remote = wandb.Api(timeout=40).run(remote_path)
                assert math.isclose(
                    remote.summary["result/final_normalized_score"], score)
                write(directory / "completion_verification.json", dict(verified=True))
                return
            if status in ["failed", "job_failed"] or state.startswith("COMPLETED"):
                raise RuntimeError(f"Job ended without full training: {state}, {local}")
            time.sleep(60)
        raise TimeoutError("Observer monitoring window expired")
    except BaseException as error:
        write(directory / "error.json", dict(error=repr(error)))
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
