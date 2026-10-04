"""Append single-step reward C after the entire existing dense campaign."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
from scripts.dense_preference.queue import process_token, read_json, write_json

def dependency_ready(root, expected_commit):
    """Require both formal runs and the old queue's final successful comparison."""
    manifest = read_json(root / "queue_manifest.json")
    if manifest["git_commit"] != expected_commit:
        raise ValueError("Unexpected dependency source revision")
    state = read_json(root / "queue_status.json")["state"]
    if state == "failed":
        raise RuntimeError("Prior campaign failed; do not start the appended experiment")
    if state in {"waiting", "smoke", "training"}:
        return False
    if state != "completed":
        raise ValueError(f"Unexpected dependency queue state: {state}")
    read_json(root / "comparison.json")
    for label, variant in (("c-seed0", "c"), ("dt-seed0", "dt")):
        trial = root / label
        status = read_json(trial / "status.json")
        config = read_json(trial / "config.json")
        summary = read_json(trial / "summary.json")
        provenance = read_json(trial / "provenance.json")
        evaluation = read_json(trial / "evaluations/step100000.json")
        if (status["state"] != "completed" or status["completed_updates"] != 100000
                or summary["completed_updates"] != 100000
                or config["update_steps"] != 100000 or config["variant"] != variant
                or config["reward_mode"] != "original"
                or config["preference_weight"] != (0.05 if variant == "c" else 0)
                or provenance["git_commit"] != expected_commit
                or evaluation["step"] != 100000
                or len(evaluation["returns"]) != config["eval_episodes"]):
            raise ValueError(f"Incomplete or mismatched prior trial: {label}")
    return True


def verify_control(control, trial, smoke=False):
    """Reuse the old DT only when settings, initialization and shared code match."""
    reference = read_json(control / "config.json")
    candidate = read_json(trial / "config.json")
    ignored = {"name", "group", "project", "wandb_entity", "wandb_mode", "variant",
               "preference_weight", "output_dir", "checkpoints_path",
               "resume_checkpoint"}
    if smoke:
        ignored |= {"update_steps", "eval_every", "eval_episodes", "log_every"}
    for key, value in reference.items():
        if key not in ignored and candidate.get(key) != value:
            raise ValueError(f"Control training setting differs: {key}")
    left = read_json(control / "provenance.json")
    right = read_json(trial / "provenance.json")
    for key in ("initial_model_sha256", "dataset_sha256", "packages",
                "attention_backend"):
        if left[key] != right[key]:
            raise ValueError(f"Control provenance differs: {key}")
    for filename in ("dt.py", "state_only_preference.py", "dense_action_preference.py",
                     "top_return_weighted_dt.py"):
        key = "algorithms/offline/" + filename
        if left["source_sha256"][key] != right["source_sha256"][key]:
            raise ValueError(f"Shared training source differs: {filename}")


def run_trial(root, label, smoke=False):
    output = root / label
    if output.exists():
        raise RuntimeError(f"Output already exists; do not silently rerun: {output}")
    command = ["bash", str(SOURCE / "scripts/step_reward_preference/run.sh"),
               "--output_dir", str(output), "--name", f"StepRewardPreference-{label}"]
    if smoke:
        command += ["--update_steps", "3", "--eval_every", "3", "--eval_episodes", "1",
                    "--log_every", "1", "--wandb_mode", "offline"]
    with (root / f"{label}.log").open("a", buffering=1) as log:
        subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    if read_json(output / "status.json")["state"] != "completed":
        raise RuntimeError(f"Trial failed: {label}")


def run_queue(args):
    root = args.output_root.resolve()
    dependency = args.dependency.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_file = root / "queue_manifest.json"
        if manifest_file.exists():
            raise RuntimeError("Queue already installed; inspect it before restarting")
        parent_status = read_json(dependency / "queue_status.json")
        if parent_status["queue_pid"] != args.dependency_queue_pid:
            raise ValueError("Dependency queue PID differs")
        token = process_token(args.dependency_queue_pid)
        manifest = {
            "source": str(SOURCE),
            "git_commit": subprocess.check_output(
                ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
            "dependency": str(dependency), "dependency_commit": args.dependency_commit,
            "dependency_queue_pid": args.dependency_queue_pid,
            "dependency_process_token": token,
            "order": ["smoke-c", "c-seed0"],
            "reused_control": str(dependency / "dt-seed0"),
        }
        write_json(manifest_file, manifest)

        def report(state, **extra):
            value = {"state": state, "queue_pid": os.getpid(),
                     "updated_at_unix": time.time(), **extra}
            write_json(root / "queue_status.json", value)
            print(json.dumps(value), flush=True)

        try:
            report("waiting", dependency=str(dependency))
            while True:
                ready = dependency_ready(dependency, args.dependency_commit)
                alive = (token is not None
                         and process_token(args.dependency_queue_pid) == token)
                if ready and not alive:
                    break
                if not ready and not alive:
                    raise RuntimeError("Prior queue exited before successful completion")
                time.sleep(args.poll_seconds)
            report("smoke", trial="smoke-c")
            run_trial(root, "smoke-c", smoke=True)
            verify_control(dependency / "dt-seed0", root / "smoke-c", smoke=True)
            report("training", trial="c-seed0")
            run_trial(root, "c-seed0")
            verify_control(dependency / "dt-seed0", root / "c-seed0")
            comparisons = {
                "step_reward_c": read_json(root / "c-seed0/summary.json"),
                "rtg_c": read_json(dependency / "c-seed0/summary.json"),
                "dt": read_json(dependency / "dt-seed0/summary.json"),
            }
            comparisons["final_score_difference_step_reward_c_minus_dt"] = (
                comparisons["step_reward_c"]["last_score"]
                - comparisons["dt"]["last_score"])
            write_json(root / "comparison.json", comparisons)
            report("completed", comparison=comparisons)
        except BaseException as error:
            report("failed", error_type=type(error).__name__, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency", required=True, type=Path)
    parser.add_argument("--dependency-commit", required=True)
    parser.add_argument("--dependency-queue-pid", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if min(args.poll_seconds, args.dependency_queue_pid) <= 0:
        parser.error("Poll interval and PID must be positive")
    run_queue(args)


if __name__ == "__main__":
    main()
