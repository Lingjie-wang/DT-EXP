"""Observe the explicit delayed-reward variant without changing its training loop."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify(root, protocol):
    for name, expected in protocol["runtime_sha256"].items():
        actual = hashlib.sha256((root / "upstream" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen upstream file changed: {name}")


class Observer:
    def __init__(self, root, protocol, mode):
        self.root, self.protocol = root, protocol
        self.accumulators, self.seen, self.saved = {}, set(), set()
        self.completed = 0
        self.run = None
        if mode != "disabled":
            import wandb

            self.run = wandb.init(
                project="CORL-DDR", entity="2820402607-shandong-university",
                group="AuctionNet-DT-PRGS-5090-20261005", name=root.name,
                dir=str(root), config=protocol, mode=mode,
                settings=wandb.Settings(init_timeout=60),
            )
            self.run.define_metric("completed_updates")
            self.run.define_metric("*", step_metric="completed_updates")
            write(root / "wandb_run.json", {"id": self.run.id, "url": self.run.url})

    def sync(self):
        rows = {}
        stride = 1 if self.protocol["smoke"] else 100
        steps_per_iter = self.protocol["config"]["num_steps_per_iter"]
        for path in (self.root / "upstream/log").rglob("events.out.tfevents.*"):
            if str(path) not in self.accumulators:
                self.accumulators[str(path)] = EventAccumulator(
                    str(path), size_guidance={"scalars": 0}
                )
            accumulator = self.accumulators[str(path)]
            accumulator.Reload()
            for tag in accumulator.Tags()["scalars"]:
                for event in accumulator.Scalars(tag):
                    key = (str(path), tag, event.step, event.wall_time)
                    if key in self.seen:
                        continue
                    self.seen.add(key)
                    if not math.isfinite(event.value):
                        raise ValueError(f"Nonfinite official metric: {tag}")
                    update = event.step + 1 if tag == "train_loss" else (
                        event.step + 1) * steps_per_iter
                    self.completed = max(self.completed, update)
                    if tag == "train_loss" and update % stride:
                        continue
                    rows.setdefault(update, {})[tag] = event.value
        for update, metrics in sorted(rows.items()):
            metrics["completed_updates"] = update
            with (self.root / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            if self.run:
                self.run.log(metrics)
        if self.run:
            for path in (self.root / "upstream/model").rglob("*.pkl"):
                name = str(path)
                if name not in self.saved:
                    self.run.save(name, base_path=str(self.root), policy="now")
                    self.saved.add(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="online")
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if (root / "status.json").exists():
        raise FileExistsError("Preserve prior attempt; prepare another root")
    verify(root, protocol)
    command = [sys.executable, "-u", "main.py", "--algo", "dt", "--env", "AuctionNet"]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    write(root / "command.json", {"argv": command, "cwd": str(root / "upstream"),
                                 "thread_limits": 4})
    write(root / "status.json", {"status": "starting", "completed_updates": 0})
    observer = None
    process = None
    started = time.time()
    try:
        observer = Observer(root, protocol, args.wandb_mode)
        process = subprocess.Popen(command, cwd=root / "upstream", env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        last_sync = 0.0
        evaluations = []
        iteration = 0
        with (root / "console.log").open("x", buffering=1) as console:
            for line in process.stdout:
                console.write(line)
                match = re.fullmatch(r"Iteration (\d+)\s*", line)
                if match:
                    iteration = int(match[1])
                score = re.match(r"evaluation/(target_[\d.]+_return_\w+): (.*)", line)
                if score:
                    evaluations.append({"iteration": iteration, "metric": score[1],
                                        "value": float(score[2])})
                    write(root / "evaluations.json", evaluations)
                    print(line.strip(), flush=True)
                if time.time() - last_sync >= 60:
                    observer.sync()
                    write(root / "status.json", {
                        "status": "running", "pid": process.pid,
                        "completed_updates": observer.completed,
                        "elapsed_seconds": time.time() - started,
                    })
                    last_sync = time.time()
        returncode = process.wait()
        observer.sync()
        verify(root, protocol)
        if returncode:
            raise RuntimeError(f"Official CLI exited {returncode}; see console.log")
        expected = protocol["config"]["max_iters"] * protocol["config"][
            "num_steps_per_iter"]
        if observer.completed != expected:
            raise RuntimeError(f"Expected {expected} updates, saw {observer.completed}")
        write(root / "status.json", {"status": "completed",
                                    "completed_updates": observer.completed,
                                    "elapsed_seconds": time.time() - started,
                                    "upstream_python_unchanged": False,
                                    "runtime_source_unchanged": True})
    except BaseException as error:
        write(root / "status.json", {"status": "failed", "error": repr(error),
                                    "completed_updates": observer.completed
                                    if observer else 0})
        raise
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if observer and observer.run:
            observer.run.finish()


if __name__ == "__main__":
    main()
