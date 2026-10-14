"""One W&B comparison run: original, uniform and Shapley curves on each env."""

import argparse
import fcntl
import hashlib
import json
import time
import uuid
from pathlib import Path

import wandb

from scripts.cql_antmaze_official_5090.common import evaluations, read, write

ARMS = ["original", "uniform", "shapley"]
LABELS = ["Original (official)", "Uniform (success boundary)",
          "Shapley (success boundary)"]
COLUMNS = ["environment", "arm", "updates", "normalized_score",
           "best_so_far", "source_url"]


def snapshot(root, plan):
    rows, best_rows, curves = [], [], {}
    for name, control in plan["controls"].items():
        series = {}
        for arm in ARMS:
            job = next(j for j in plan["jobs"] if j["dataset_key"] == name
                       and j["arm"] == arm)
            work = root / job["id"] / "training"
            records = (evaluations((work / "console.log").read_text(errors="replace"))
                       if (work / "console.log").exists() else [])
            url = (read(work / "wandb_manifest.json")["url"]
                   if (work / "wandb_manifest.json").exists() else "")
            series[arm] = records, url
        curves[name] = []
        for arm in ARMS:
            records, url = series[arm]
            steps, scores, running_best = [], [], None
            for record in records:
                step, score = record["completed_updates"], record["normalized_score"]
                if steps and step <= steps[-1]:
                    raise ValueError("Non-increasing official evaluation steps")
                if record["episodes"] != 100:
                    raise ValueError("Exclude preflights from comparison")
                running_best = (score if running_best is None
                                else max(running_best, score))
                steps.append(step)
                scores.append(score)
                rows.append([control["env"], arm, step, score, running_best, url])
            curves[name].append((steps, scores))
            best_step = steps[scores.index(max(scores))] if scores else None
            best_rows.append([control["env"], arm, max(scores) if scores else None,
                              best_step, len(scores), url])
    state = (read(root / "queue/status.json")
             if (root / "queue/status.json").exists() else {})
    pipeline = read(root / "pipeline_status.json") if (
        root / "pipeline_status.json").exists() else {}
    return dict(rows=rows, best_rows=best_rows, curves=curves,
                queue_stage=state.get("stage", "not_started"),
                job_states={k: v["state"] for k, v in state.get("jobs", {}).items()},
                pipeline_stage=pipeline.get("stage", "prepared"))


def observe(root):
    plan = read(root / "plan.json")
    out = root / "comparison"
    with (out / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / "manifest.json"
        identity = read(path)["id"] if path.exists() else str(uuid.uuid4())
        url = f"https://wandb.ai/{plan['wandb_entity']}/{plan['wandb_project']}/runs/{identity}"
        write(path, dict(id=identity, url=url))
        run, previous, uploads = None, None, 0
        while True:
            try:
                if run is None:
                    run = wandb.init(entity=plan["wandb_entity"],
                                     project=plan["wandb_project"],
                                     id=identity, resume="allow", dir=str(out),
                                     group=root.name + "-comparison",
                                     name="Medium AntMaze: Original / Uniform / Shapley",
                                     job_type="comparison",
                                     config=dict(seed=plan["seed"], updates=1000000,
                                     evaluation_episodes=100, score_selection="best",
                                     success_timing_available=True,
                                     controls=plan["controls"], campaign=str(root)))
                data = snapshot(root, plan)
                encoded = json.dumps(data, sort_keys=True).encode()
                signature = hashlib.sha256(encoded).hexdigest()
                if signature != previous:
                    payload = {"comparison/evaluations": wandb.Table(columns=COLUMNS,
                                                                        data=data["rows"]),
                               "comparison/best_scores": wandb.Table(columns=[
                                   "environment", "arm", "best_score",
                                   "first_best_updates",
                                   "evaluations", "source_url"], data=data["best_rows"])}
                    for name, series in data["curves"].items():
                        if not any(s[0] for s in series):
                            continue
                        payload[f"curves/{name}"] = wandb.plot.line_series(
                            xs=[s[0] for s in series], ys=[s[1] for s in series],
                            keys=LABELS, title=f"{name}: original / uniform / Shapley",
                            xname="Gradient updates")
                    run.log(payload)
                    run.summary.update({k: data[k] for k in ["queue_stage", "job_states",
                                                            "pipeline_stage"]})
                    run.summary["evaluation_rows"] = len(data["rows"])
                    run.summary["last_observed_unix"] = time.time()
                    write(out / "latest.json", data)
                    uploads += 1
                    write(out / "status.json", dict(stage="uploaded", url=url,
                          evaluation_rows=len(data["rows"]), uploads=uploads,
                          observed_unix=time.time()))
                    previous = signature
                if (data["queue_stage"] in {"completed", "finished_with_failures"}
                        or data["pipeline_stage"] == "failed"):
                    run.finish(exit_code=0 if data["queue_stage"] == "completed" else 1)
                    return
            except Exception as error:
                write(out / "status.json", dict(stage="retrying", error=repr(error),
                                                url=url, observed_unix=time.time()))
                print("comparison retry", repr(error), flush=True)
                if run is not None:
                    try:
                        run.finish(exit_code=1)
                    except Exception:
                        pass
                    run = None
                previous = None
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    observe(parser.parse_args().root.resolve())
