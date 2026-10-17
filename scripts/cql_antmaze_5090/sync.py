"""Finish uniform-run uploads with a readback of history and the final checkpoint."""

import argparse
import fcntl
import math
import time
from pathlib import Path

from scripts.cql_5090.observe import collect, observe
from scripts.cql_delayed.common import read, write

def remote_history(remote):
    train = {int(row["completed_updates"]): row["train/alpha"]
             for row in remote.scan_history(
                 keys=["completed_updates", "train/alpha"], page_size=2000)
             if row.get("completed_updates") is not None
             and row.get("train/alpha") is not None}
    evaluations = {int(row["completed_updates"]): row["eval/normalized_score"]
                   for row in remote.scan_history(
                       keys=["completed_updates", "eval/normalized_score"],
                       page_size=1000)
                   if row.get("completed_updates") is not None
                   and row.get("eval/normalized_score") is not None}
    return train, evaluations


def missing_rows(local, train, evaluations):
    missing = {}
    for step, row in local.items():
        need_train = "train/alpha" in row and (
            step not in train or not math.isclose(row["train/alpha"], train[step]))
        need_eval = "eval/normalized_score" in row and (
            step not in evaluations
            or not math.isclose(row["eval/normalized_score"], evaluations[step]))
        values = {key: value for key, value in row.items()
                  if (key.startswith("train/") and need_train)
                  or (not key.startswith("train/") and need_eval)}
        if values:
            missing[step] = values
    return missing


def finalize(root, arm):
    import wandb

    protocol = read(root / "protocol.json")
    spec = protocol["runs"][arm]
    work, directory = root / arm / "training", root / "wandb_sync" / arm
    local_status = read(work / "status.json")
    if (local_status["status"] != "completed"
            or local_status["completed_updates"] != spec["updates"]):
        raise ValueError("Only finalize a successfully completed training run")
    local = collect(work, dict(updates=0, evaluations=[]))
    scores = [read(work / f"eval_{step:07d}.json")["normalized_score"] for step in
              range(spec["eval_every"], spec["updates"] + 1, spec["eval_every"])]
    expected_train = set(range(100, spec["updates"] + 1, 100))
    if ({s for s, row in local.items() if "train/alpha" in row} != expected_train
            or not all(math.isfinite(s) for s in scores)):
        raise ValueError("Incomplete local training history")
    checkpoint = work / f"checkpoint_{spec['updates']:07d}.pt"
    path = f"{protocol['wandb_entity']}/{protocol['wandb_project']}/{spec['wandb_id']}"
    remote = wandb.Api(timeout=40).run(path)
    missing = missing_rows(local, *remote_history(remote))
    filename = f"{arm}/training/{checkpoint.name}"
    files = {f.name: f.size for f in remote.files()}
    correct_file = files.get(filename) == checkpoint.stat().st_size
    summary = dict(local_status, **{
        "result/final_normalized_score": scores[-1],
        "result/best_normalized_score": max(scores),
        "result/last10_mean": sum(scores[-10:]) / len(scores[-10:])})
    def summary_matches(remote):
        return (remote.summary.get("completed_updates") == spec["updates"]
                and remote.summary.get("status") == "completed"
                and all(isinstance(remote.summary.get(key), (int, float))
                        and math.isclose(remote.summary[key], value)
                        for key, value in summary.items() if key.startswith("result/")))

    correct_summary = summary_matches(remote)
    if missing or not correct_file or not correct_summary or remote.state != "finished":
        run = wandb.init(entity=protocol["wandb_entity"],
                         project=protocol["wandb_project"],
                         id=spec["wandb_id"], resume="must", mode="online",
                         dir=str(directory), settings=wandb.Settings(console="off"))
        try:
            run.define_metric("completed_updates")
            run.define_metric("*", step_metric="completed_updates")
            for step, values in sorted(missing.items()):
                run.log(dict(completed_updates=step, **values))
            if not correct_file:
                run.save(str(checkpoint), base_path=str(root), policy="now")
            run.summary.update(summary)
        except BaseException:
            run.finish(exit_code=1)
            raise
        run.finish()
        remote = wandb.Api(timeout=40).run(path)
        files = {f.name: f.size for f in remote.files()}
    if (missing_rows(local, *remote_history(remote)) or remote.state != "finished"
            or files.get(filename) != checkpoint.stat().st_size
            or not summary_matches(remote)):
        raise ValueError("W&B final readback incomplete; retry from original local data")
    write(directory / "full_completion_verification.json", dict(
        verified=True, id=remote.id, url=remote.url, state=remote.state,
        updates=spec["updates"], evaluations=len(scores),
        training_records=len(expected_train),
        checkpoint_bytes=checkpoint.stat().st_size, verified_unix=time.time()))


def execute(root, arm):
    observe(root, arm)
    if read(root / arm / "training/status.json")["status"] != "completed":
        return
    directory = root / "wandb_sync" / arm
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                finalize(root, arm)
                return
            except Exception as error:
                write(directory / "finalization_error.json", dict(
                    error=repr(error), retry_after_seconds=30))
                time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["delayed", "shapley", "uniform", "dense"],
                        required=True)
    args = parser.parse_args()
    execute(args.root.resolve(), args.arm)
