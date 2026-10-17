"""Submit each validated arm once, preserving existing jobs and source snapshots."""

import argparse
import fcntl
import os
import subprocess
from pathlib import Path

from common import digest, read, verify, write

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bridge-python", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = open(root / "submission.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = verify(root)
    for arm in ["uniform", "dense", "shapley"]:
        if arm == "shapley" and not p.get("shapley_gate", {}).get("passed", False):
            print("Shapley withheld: held-out prediction gate not passed", flush=True)
            continue
        for name, expected in p["runs"][arm]["data_sha256"].items():
            if digest(root / arm / name) != expected:
                raise ValueError(f"Dataset checksum mismatch: {arm}/{name}")
        manifest = root / arm / "submission.json"
        if manifest.exists():
            job = read(manifest)["job_id"]
        else:
            # Record intent before contacting Slurm. An ambiguous interrupted
            # submission must be reconciled with the scheduler, never retried blindly.
            intent = root / arm / "submission_intent.json"
            if intent.exists():
                raise RuntimeError(f"Reconcile unresolved Slurm submission: {intent}")
            command = ["sbatch", "--parsable", "--job-name", f"cql-shp-{arm}-s0",
                       str(root / "source/scripts/cql_shapley/run.sbatch"),
                       str(root), arm]
            write(intent, dict(command=command))
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            write(manifest, dict(job_id=job, command=command))
        directory = root / "wandb_sync" / arm
        directory.mkdir(parents=True, exist_ok=True)
        pid_file = directory / "pid"
        if pid_file.exists():
            # Avoid duplicate or unexpectedly restarted observers on repeated calls.
            print(arm, "existing job", job, "observer", pid_file.read_text().strip())
            continue
        env = os.environ.copy()
        env.pop("WANDB_MODE", None)
        env["PYTHONUNBUFFERED"] = "1"
        with (directory / "bridge.log").open("a") as stream:
            process = subprocess.Popen(
                [str(args.bridge_python),
                 str(root / "source/scripts/cql_shapley/sync_results.py"),
                 "--root", str(root), "--arm", arm, "--job", job],
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, env=env, cwd=root,
            )
        pid_file.write_text(str(process.pid) + "\n")
        print(arm, "submitted", job, "bridge", process.pid, flush=True)


if __name__ == "__main__":
    main()
