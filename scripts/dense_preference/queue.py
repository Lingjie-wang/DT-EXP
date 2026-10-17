"""Run dense C and a matched DT control only after a successful delayed run.

Run in detached tmux. Status files are atomic; an exclusive campaign lock prevents
duplicate launches. Failures stop the queue, never silently restart experiments.
"""

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_token(pid):
    """PID plus Linux start ticks distinguishes a reused process identifier."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rindex(")") + 2:].split()
    if fields[0] == "Z":
        return None
    return f"{pid}:{fields[19]}"


def dependency_ready(root, required_steps, expected_run_id):
    """Failure is different from completion. Require final evaluation on disk."""
    run = read_json(root / "wandb_run.json")
    if run["id"] != expected_run_id:
        raise ValueError("Dependency W&B run ID differs from the queued experiment")
    status = read_json(root / "status.json")
    if status["state"] == "failed":
        raise RuntimeError("Dependency failed; queue must not proceed")
    if status["state"] != "completed":
        return False
    summary = read_json(root / "summary.json")
    config = read_json(root / "config.json")
    evaluation = read_json(root / "evaluations" / f"step{required_steps:06d}.json")
    if (status["completed_updates"] != required_steps
            or summary["completed_updates"] != required_steps
            or config["update_steps"] != required_steps
            or evaluation["step"] != required_steps
            or len(evaluation["returns"]) != config["eval_episodes"]):
        raise ValueError("Dependency final update/evaluation is incomplete")
    return True


def run_trial(source, campaign, label, variant, smoke=False):
    output = campaign / label
    if output.exists():
        status_file = output / "status.json"
        if status_file.exists() and read_json(status_file)["state"] == "completed":
            config = read_json(output / "config.json")
            provenance = read_json(output / "provenance.json")
            revision = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            if (config["variant"] == variant and config["reward_mode"] == "original"
                    and config["update_steps"] == (3 if smoke else 100000)
                    and provenance["git_commit"] == revision):
                return
        raise RuntimeError(f"Existing incomplete or mismatched output: {output}")
    command = [
        "bash", str(source / "scripts/dense_preference/run.sh"),
        "--output_dir", str(output), "--variant", variant,
        "--preference_weight", "0.05" if variant == "c" else "0.0",
        "--name", f"DensePreference-{label}",
    ]
    if smoke:
        command += ["--update_steps", "3", "--eval_every", "3",
                    "--eval_episodes", "1", "--log_every", "1",
                    "--wandb_mode", "offline"]
    with (campaign / f"{label}.log").open("a", buffering=1) as log:
        subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    status = read_json(output / "status.json")
    if status["state"] != "completed":
        raise RuntimeError(f"Trial did not complete: {label}")


def verify_matched_arms(campaign, first, second):
    left = read_json(campaign / first / "provenance.json")
    right = read_json(campaign / second / "provenance.json")
    for key in ("initial_model_sha256", "dataset_sha256", "pairs_sha256",
                "git_commit", "source_sha256", "attention_backend"):
        if left[key] != right[key]:
            raise ValueError(f"Unmatched comparison: {key}")


def run_queue(args):
    campaign = args.output_root.resolve()
    source = Path(__file__).resolve().parents[2]
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_file = campaign / "queue_status.json"
        manifest_file = campaign / "queue_manifest.json"
        manifest = {
            "dependency": str(args.dependency.resolve()),
            "dependency_run_id": args.dependency_run_id,
            "required_steps": args.required_steps,
            "dependency_pid": args.dependency_pid,
            "source": str(source),
            "git_commit": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "order": ["smoke-c", "smoke-dt", "c-seed0", "dt-seed0"],
        }
        if manifest_file.exists():
            old = read_json(manifest_file)
            if any(old.get(key) != value for key, value in manifest.items()):
                raise ValueError("Existing queue manifest differs; use a new campaign")
            dependency_token = old["dependency_process_token"]
        else:
            dependency_token = process_token(args.dependency_pid)
            manifest["dependency_process_token"] = dependency_token
            write_json(manifest_file, manifest)

        def report(state, **extra):
            value = {"state": state, "queue_pid": os.getpid(),
                     "updated_at_unix": time.time(), **extra}
            write_json(status_file, value)
            print(json.dumps(value), flush=True)

        try:
            report("waiting", dependency=manifest["dependency"],
                   required_steps=args.required_steps)
            while True:
                ready = dependency_ready(args.dependency, args.required_steps,
                                         args.dependency_run_id)
                alive = (dependency_token is not None
                         and process_token(args.dependency_pid) == dependency_token)
                if ready and not alive:
                    break
                if not ready and not alive:
                    raise RuntimeError("Dependency process exited without completion")
                time.sleep(args.poll_seconds)
            for label, variant in (("smoke-c", "c"), ("smoke-dt", "dt")):
                report("smoke", trial=label)
                run_trial(source, campaign, label, variant, smoke=True)
            verify_matched_arms(campaign, "smoke-c", "smoke-dt")
            for label, variant in (("c-seed0", "c"), ("dt-seed0", "dt")):
                report("training", trial=label)
                run_trial(source, campaign, label, variant)
            verify_matched_arms(campaign, "c-seed0", "dt-seed0")
            summaries = {arm: read_json(campaign / arm / "summary.json")
                         for arm in ("c-seed0", "dt-seed0")}
            summaries["final_score_difference_c_minus_dt"] = (
                summaries["c-seed0"]["last_score"]
                - summaries["dt-seed0"]["last_score"])
            write_json(campaign / "comparison.json", summaries)
            report("completed", comparison=summaries)
        except BaseException as error:
            report("failed", error_type=type(error).__name__, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency", required=True, type=Path)
    parser.add_argument("--dependency-run-id", required=True)
    parser.add_argument("--dependency-pid", required=True, type=int)
    parser.add_argument("--required-steps", type=int, default=100000)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if min(args.required_steps, args.poll_seconds, args.dependency_pid) <= 0:
        parser.error("Steps, PID and poll interval must be positive")
    run_queue(args)


if __name__ == "__main__":
    main()
