"""Retryable W&B bridge for file-based CQL telemetry, independent of training."""

import argparse
import fcntl
import json
import math
import time
from pathlib import Path

from scripts.cql_delayed.common import read, write

def collect(work, cursor):
    rows = {}
    metrics = work / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                continue
            row = json.loads(line)
            step = row.pop("completed_updates")
            if step > cursor["updates"]:
                rows.setdefault(step, {}).update(
                    {f"train/{k}": v for k, v in row.items()})
    for path in sorted(work.glob("eval_*.json")):
        row = read(path)
        step = row["completed_updates"]
        if step not in cursor["evaluations"]:
            rows.setdefault(step, {}).update({
                "eval/normalized_score": row["normalized_score"],
                "eval/raw_return_mean": row["raw_return_mean"],
                "d4rl_normalized_score": row["normalized_score"],
            })
    return rows


def observe(root, arm):
    import wandb

    p = read(root / "protocol.json")
    spec = p["runs"][arm]
    work = root / arm / "training"
    directory = root / "wandb_sync" / arm
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            run = None
            try:
                config = {**spec, **p["retained"], "migration": p["migration"],
                          "changes": p["changes"],
                          "reward_mode": spec.get("reward_mode", p.get("reward_mode")),
                          "dataset_audit": read(root / arm / "audit.json")}
                run = wandb.init(
                    entity=p["wandb_entity"], project=p["wandb_project"],
                    group=p["wandb_group"], id=spec["wandb_id"],
                    name=spec["wandb_name"], config=config, resume="allow",
                    mode="online", dir=str(directory),
                    settings=wandb.Settings(init_timeout=60, console="off"),
                )
                run.define_metric("completed_updates")
                run.define_metric("*", step_metric="completed_updates")
                write(directory / "manifest.json", {
                    "id": run.id, "name": run.name, "url": run.url})
                cursor_path = directory / "cursor.json"
                cursor = read(cursor_path) if cursor_path.exists() else {
                    "updates": 0, "evaluations": [], "checkpoints": []}
                while True:
                    for step, values in sorted(collect(work, cursor).items()):
                        run.log({"completed_updates": step, **values})
                        if any(key.startswith("train/") for key in values):
                            cursor["updates"] = max(cursor["updates"], step)
                        if "eval/normalized_score" in values:
                            cursor["evaluations"].append(step)
                    local = {"status": "starting", "completed_updates": 0}
                    if (work / "status.json").exists():
                        local = read(work / "status.json")
                    for checkpoint in sorted(work.glob("checkpoint_*.pt")):
                        step = int(checkpoint.stem.split("_")[-1])
                        # Saving happens after status(step); next progress or successful
                        # completion is proof that the checkpoint write has finished.
                        stable = local["completed_updates"] > step or (
                            local["status"] == "completed")
                        if stable and checkpoint.name not in cursor["checkpoints"]:
                            run.save(str(checkpoint), base_path=str(root), policy="now")
                            cursor["checkpoints"].append(checkpoint.name)
                    write(cursor_path, cursor)
                    run.summary.update(local)
                    write(directory / "status.json", {**local, "sync": "online"})
                    if local["status"] == "completed":
                        expected = list(range(
                            spec["eval_every"], spec["updates"] + 1, spec["eval_every"]))
                        if (cursor["updates"] != spec["updates"]
                                or sorted(cursor["evaluations"]) != expected):
                            raise ValueError("Incomplete final training telemetry")
                        scores = [read(path)["normalized_score"]
                                  for path in sorted(work.glob("eval_*.json"))]
                        run.summary.update({
                            "result/final_normalized_score": scores[-1],
                            "result/best_normalized_score": max(scores),
                            "result/last10_mean": sum(scores[-10:]) / 10})
                        run.finish()
                        run = None
                        remote_path = (f"{p['wandb_entity']}/{p['wandb_project']}/"
                                       f"{spec['wandb_id']}")
                        remote = wandb.Api(timeout=40).run(remote_path)
                        if not math.isclose(
                                remote.summary["result/final_normalized_score"],
                                scores[-1]):
                            raise ValueError("W&B final score readback differs")
                        write(directory / "completion_verification.json",
                              {"verified": True})
                        return
                    if local["status"] == "failed":
                        run.finish(exit_code=1)
                        return
                    time.sleep(30)
            except Exception as error:
                write(directory / "sync_error.json", {"error": repr(error),
                                                       "retry_after_seconds": 30})
                if run is not None:
                    try:
                        run.finish(exit_code=1)
                    except Exception:
                        pass
                time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["medium", "medium_replay", "shapley"],
                        required=True)
    args = parser.parse_args()
    observe(args.root.resolve(), args.arm)
