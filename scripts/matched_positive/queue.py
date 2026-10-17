"""Append matched positive-imitation B after the frozen MC-value campaign."""

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
from scripts.mc_value_preference.queue import dependency_ready as step_campaign_ready
from scripts.step_reward_preference.queue import verify_control

def dependency_ready(root, expected_commit):
    """Require completed MC-value C and every earlier campaign."""
    manifest = read_json(root / "queue_manifest.json")
    if manifest["git_commit"] != expected_commit:
        raise ValueError("Unexpected dependency source revision")
    state = read_json(root / "queue_status.json")["state"]
    if state == "failed":
        raise RuntimeError("Prior campaign failed; do not start matched B")
    if state in {"waiting", "smoke", "training"}:
        return False
    if state != "completed":
        raise ValueError(f"Unexpected dependency queue state: {state}")
    read_json(root / "comparison.json")
    trial = root / "c-seed0"
    status = read_json(trial / "status.json")
    config = read_json(trial / "config.json")
    summary = read_json(trial / "summary.json")
    provenance = read_json(trial / "provenance.json")
    evaluation = read_json(trial / "evaluations/step100000.json")
    if (status["state"] != "completed" or status["completed_updates"] != 100000
            or summary["completed_updates"] != 100000
            or config["update_steps"] != 100000 or config["variant"] != "c"
            or config["reward_mode"] != "original"
            or config["preference_label"] != "mc_value_advantage"
            or config["preference_weight"] != 0.05
            or provenance["git_commit"] != expected_commit
            or evaluation["step"] != 100000
            or len(evaluation["returns"]) != config["eval_episodes"]):
        raise ValueError("Incomplete or mismatched prior MC-value C")
    if not step_campaign_ready(Path(manifest["dependency"]),
                                manifest["dependency_commit"]):
        raise ValueError("Earlier single-step/dense campaign is incomplete")
    return True


def run_trial(root, label, reference, smoke=False):
    output = root / label
    if output.exists():
        raise RuntimeError(f"Output already exists; do not silently rerun: {output}")
    command = ["bash", str(SOURCE / "scripts/matched_positive/run.sh"),
               "--output_dir", str(output), "--name", f"MatchedPositive-{label}",
               "--reference_dir", str(reference)]
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
        parent_manifest = read_json(dependency / "queue_manifest.json")
        control = Path(parent_manifest["reused_control"])
        reference = control.parent / "c-seed0"
        manifest = {
            "source": str(SOURCE),
            "git_commit": subprocess.check_output(
                ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
            "dependency": str(dependency), "dependency_commit": args.dependency_commit,
            "dependency_queue_pid": args.dependency_queue_pid,
            "dependency_process_token": token,
            "order": ["smoke-b", "b-seed0"],
            "reused_rtg_c": str(reference),
            "reused_control": str(control),
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
            report("smoke", trial="smoke-b")
            run_trial(root, "smoke-b", reference.parent / "smoke-c", smoke=True)
            verify_control(control, root / "smoke-b", smoke=True)
            report("training", trial="b-seed0")
            run_trial(root, "b-seed0", reference)
            verify_control(control, root / "b-seed0")
            if (read_json(root / "b-seed0/provenance.json")["pairs_sha256"]
                    != read_json(reference / "provenance.json")["pairs_sha256"]):
                raise ValueError("B pair pool differs from RTG C")
            comparisons = {
                "positive_b": read_json(root / "b-seed0/summary.json"),
                "rtg_c": read_json(reference / "summary.json"),
                "dt": read_json(control / "summary.json"),
            }
            for endpoint in ("last_score", "late_60_80_100k_mean"):
                comparisons[endpoint + "_b_minus_dt"] = (
                    comparisons["positive_b"][endpoint] - comparisons["dt"][endpoint])
                comparisons[endpoint + "_c_minus_b"] = (
                    comparisons["rtg_c"][endpoint] - comparisons["positive_b"][endpoint])
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
