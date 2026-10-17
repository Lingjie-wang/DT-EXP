"""Wait for complete predecessor artifacts and exited processes, not W&B exit codes."""

from pathlib import Path

from scripts.cql_antmaze_official_5090.common import completed, digest, read

def process_running(pid, root):
    if not pid:
        return False
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return str(root).encode() in command


def predecessor_status(plan):
    root = Path(plan["predecessor"])
    if digest(root / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("Predecessor plan changed; inspect before admitting jobs")
    old = read(root / "plan.json")
    state = read(root / "queue/status.json")
    results = {}
    for job in old["jobs"]:
        key = job["id"]
        previous = state.get("jobs", {}).get(key, {})
        if previous.get("state") not in {"completed", "failed"}:
            results[key] = dict(ready=False, reason="predecessor_not_terminal",
                               state=previous.get("state"))
            continue
        if process_running(previous.get("pid"), root):
            results[key] = dict(ready=False, reason="predecessor_process_alive")
            continue
        try:
            proof = completed(root / key / "training", job["updates"],
                              job["eval_every"], job["eval_episodes"])
            results[key] = dict(ready=True, completed_updates=proof["completed_updates"],
                               recorded_exit=previous.get("exit"),
                               best_score=proof["best_normalized_score"])
        except (OSError, ValueError) as error:
            results[key] = dict(ready=False, reason="incomplete_training_artifacts",
                               error=str(error))
    return dict(ready=bool(results) and all(r["ready"] for r in results.values()),
                jobs=results)


def predecessor_ready(plan):
    return predecessor_status(plan)["ready"]
