"""Queue ordinary DT on non-final data after the existing final-data run succeeds."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def predecessor_complete(status):
    if status["status"] in {"starting", "running"}:
        return False
    if status["status"] != "completed":
        raise RuntimeError(f"Predecessor did not succeed: {status}")
    if (status.get("completed_updates") != 100000
            or status.get("upstream_python_unchanged") is not True):
        raise RuntimeError("Predecessor completion lacks 100k/source verification")
    return True


def execute(project, poll_seconds):
    code = project / "scripts/auctionnet_dt"
    own = project / "scripts/auctionnet_dt_nonfinal"
    data = project / ".runtime/auctionnet-dt-nonfinal-data-20261006"
    upstream = project / ".runtime/auctionnet-upstream-20261005"
    previous = project / "results/prgs-dt-auctionnet-5090-20261005"
    output = project / "results/prgs-dt-auctionnet-nonfinal-5090-20261006"
    data.mkdir(parents=True, exist_ok=True)
    lock = (data / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if output.exists():
        raise FileExistsError(f"Preserve existing run: {output}")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    status_path = data / "queue_status.json"

    def update(stage, **extra):
        write(status_path, {"stage": stage, "pid": os.getpid(),
                            "predecessor": str(previous), "output": str(output),
                            "updated_unix": time.time(), **extra})

    def command(script, *arguments):
        subprocess.run([sys.executable, str(script), *map(str, arguments)],
                       cwd=project, env=environment, check=True)

    downloader = None
    try:
        prior = json.loads((previous / "status.json").read_text())
        ready = predecessor_complete(prior)
        with (data / "download.log").open("a", buffering=1) as download_log:
            downloader = subprocess.Popen(
                [sys.executable, str(own / "download_data.py"), "--root", str(data)],
                cwd=project, env=environment, stdout=download_log,
                stderr=subprocess.STDOUT)
            while not ready:
                if downloader.poll() not in (None, 0):
                    raise RuntimeError("Dataset download failed; see download.log")
                update("waiting_for_predecessor", predecessor_status=prior,
                       download_pid=downloader.pid,
                       download_complete=downloader.poll() == 0)
                time.sleep(poll_seconds)
                prior = json.loads((previous / "status.json").read_text())
                ready = predecessor_complete(prior)
            update("waiting_for_download", predecessor_status=prior,
                   download_pid=downloader.pid)
            if downloader.wait():
                raise RuntimeError("Dataset download failed; see download.log")
        # A successful predecessor includes its completed evaluations and source check.
        if not predecessor_complete(json.loads((previous / "status.json").read_text())):
            raise RuntimeError("Predecessor changed before preprocessing")
        update("preprocessing", predecessor_status=prior)
        for period in range(7, 14):
            command(code / "prepare_data.py", "period", "--raw",
                    data / f"raw/period-{period}.csv", "--generator",
                    upstream / "auctionnet/train_data_generator.py", "--output",
                    data / "processed")
        command(code / "prepare_data.py", "assemble", "--inputs", data / "processed",
                "--output", data / "prepared-full")
        audit_path = data / "prepared-full/audit.json"
        audit = json.loads(audit_path.read_text())
        audit["source_dataset"] = "AuctionNet non-final/general (CPA 6-12 version)"
        audit["download_manifest"] = json.loads((data / "download_complete.json").read_text())
        write(audit_path, audit)
        command(code / "prepare_run.py", "--upstream", upstream / "prgs/AuctionNet",
                "--data", data / "prepared-full", "--csvs",
                *[data / f"raw/period-{p}.csv" for p in range(14, 21)], "--root", output)
        # Correct provenance metadata only; no upstream source/config patching.
        protocol_path = output / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["changes"]["data"] = (
            "Reconstructed from public NON-FINAL/general raw CSV, P7-P13 training; "
            "P14-P20 evaluation. PRGS private pickle equivalence unverified.")
        protocol["dataset_version"] = "non-final/general"
        protocol["predecessor"] = {"root": str(previous), "completion": prior}
        assert protocol["config"]["is_stitch"] is False
        assert protocol["config"]["model_type"] == "dt"
        write(protocol_path, protocol)
        update("running", predecessor_status=prior)
        command(code / "run.py", "--root", output, "--wandb-mode", "online")
        update("completed", run_status=json.loads((output / "status.json").read_text()))
    except BaseException as error:
        update("failed", error=repr(error))
        raise
    finally:
        if downloader and downloader.poll() is None:
            downloader.terminate()
            try:
                downloader.wait(timeout=20)
            except subprocess.TimeoutExpired:
                downloader.kill()
                downloader.wait()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    execute(args.project.resolve(), args.poll_seconds)


if __name__ == "__main__":
    main()
