"""Run an independent extended-budget seed with the unchanged official CLI."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from observe import parse_line

def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with open(path, "a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_source(checkout, protocol):
    for name, expected in protocol["upstream_sha256"].items():
        if digest(checkout / name) != expected:
            raise RuntimeError(f"Official source changed: {name}")


def official_command(protocol, phase):
    return [
        sys.executable, "-u", "train.py",
        "--env_name", "halfcheetah", "--eval_dataset", "medium-replay",
        "--seed", str(protocol["model_initialization_seed"]),
        "--epochs", "1" if phase == "preflight" else str(protocol["epochs"]),
    ]


def launch(root, arm, phase, protocol):
    work = root / arm / phase
    checkout = work / "upstream"
    check_source(checkout, protocol)
    if (work / "console.log").exists():
        raise RuntimeError("Refusing to overwrite a previous attempt")
    command = official_command(protocol, phase)
    write(work / "command.json", dict(argv=command, cwd=str(checkout / "scripts")))
    completed, episodes, evaluations = 0, [], []
    write(
        root / arm / "status.json",
        dict(status=phase, completed_updates=0, job_id=os.getenv("SLURM_JOB_ID")),
    )
    with open(work / "console.log", "w", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=checkout / "scripts",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                log.write(line)
                parsed = parse_line(line, protocol["total_updates"])
                if not parsed or phase != "training":
                    continue
                kind, row = parsed
                if kind == "episode":
                    episodes.append(row)
                elif kind == "evaluation":
                    completed = row["completed_updates"]
                    assert [r["episode"] for r in episodes] == list(range(1, 11))
                    row["episodes"] = episodes
                    write(root / arm / f"eval_{completed:06d}.json", row)
                    episodes = []
                    evaluations.append(completed)
                    print(json.dumps(row), flush=True)
                elif kind == "training":
                    append(root / arm / "metrics.jsonl", row)
                if "completed_updates" in row:
                    completed = max(completed, row["completed_updates"])
                    write(
                        root / arm / "status.json",
                        dict(status="training", completed_updates=completed),
                    )
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
    check_source(checkout, protocol)
    if returncode:
        raise RuntimeError(
            f"Official {phase} exited {returncode}; see {work / 'console.log'}"
        )
    if phase == "training":
        assert evaluations == protocol["eval_updates"]
        assert completed == protocol["total_updates"]
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("dense", "delayed"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    status = root / args.arm / "status.json"
    if status.exists():
        raise RuntimeError("Refusing to replace an earlier run")
    try:
        for name, expected in protocol["harness_sha256"].items():
            assert digest(root / "source" / name) == expected
        for name, expected in protocol["dataset_sha256"].items():
            assert digest(root / name) == expected
        # Kernel and full-batch preflight run in separate processes, so their
        # random draws cannot change the subsequent official training process.
        subprocess.run(
            [
                sys.executable,
                str(root / "source" / "verify_runtime.py"),
                "--output",
                str(root / args.arm / "runtime.json"),
            ],
            check=True,
        )
        launch(root, args.arm, "preflight", protocol)
        completed = launch(root, args.arm, "training", protocol)
        write(
            status,
            dict(
                status="completed",
                completed_updates=completed,
                upstream_files_unchanged=True,
            ),
        )
    except BaseException as error:
        previous = json.loads(status.read_text()) if status.exists() else {}
        write(
            status,
            dict(
                status="failed",
                completed_updates=previous.get("completed_updates", 0),
                error=repr(error),
            ),
        )
        raise


if __name__ == "__main__":
    main()
