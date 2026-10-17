"""Release only verified completed CQL runs stuck in the old W&B SDK shutdown."""

import argparse
import ast
import fcntl
import os
import signal
import sys
import time
from pathlib import Path

import torch

from scripts.cql_antmaze_official_5090.common import (
    completed,
    digest,
    read,
    verify,
    write,
)

def process(pid):
    try:
        base = Path("/proc") / str(pid)
        fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(pid=pid, parent=int(fields[1]), start=fields[19],
                    command=(base / "cmdline").read_bytes().decode().replace("\0", " "))
    except FileNotFoundError:
        return None


def same_process(identity):
    current = process(identity["pid"])
    return current is not None and all(current[k] == identity[k]
                                       for k in ["pid", "start", "command"])


def holds_native(pid, native):
    directory = Path("/proc") / str(pid) / "fd"
    try:
        return any(path.resolve() == native for path in directory.iterdir())
    except FileNotFoundError:
        return False


def uploaded(cursor, native, updates):
    stat = native.stat()
    return (cursor.get("path") == str(native)
            and cursor.get("device") == stat.st_dev
            and cursor.get("inode") == stat.st_ino
            and cursor.get("offset") == stat.st_size
            and cursor.get("history_lines") == updates
            and cursor.get("last_history_step") == updates
            and cursor.get("summary_step") == updates
            and cursor.get("exit_code") == 0)


def checkpoint_proof(root, path, updates):
    # The official CLI pickles Scalar as __main__.Scalar. Load its exact saved
    # class solely in this inspection process; never import or patch a learner.
    source = root / "upstream/algorithms/offline/cql.py"
    tree = ast.parse(source.read_text())
    scalar = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name == "Scalar")
    scope = dict(torch=torch, nn=torch.nn, __name__="__main__")
    exec(compile(ast.Module(body=[scalar], type_ignores=[]), str(source), "exec"), scope)
    setattr(sys.modules["__main__"], "Scalar", scope["Scalar"])
    model = torch.load(path, map_location="cpu", weights_only=False)
    if model["total_it"] != updates or not all(
            torch.isfinite(t).all().item()
            for key in ["actor", "critic1", "critic2",
                        "critic1_target", "critic2_target"]
            for t in model[key].values()):
        raise ValueError("Checkpoint is incomplete or contains non-finite weights")
    return dict(sha256=digest(path), total_it=updates, finite_weights=True)


def candidate(root, plan, job, state, relay, grace):
    if state.get("state") != "running" or not state.get("pid"):
        return None
    work = root / job["id"] / "training"
    try:
        proof = completed(work, job["updates"], job["eval_every"], job["eval_episodes"])
    except (OSError, ValueError):
        return None
    checkpoint = Path(proof["final_checkpoint"])
    if time.time() - checkpoint.stat().st_mtime < grace:
        return None
    if "Waiting for W&B process to finish... (success)." not in (
            work / "console.log").read_text(errors="replace"):
        return None
    cursor = relay.get(job["id"], {})
    if cursor.get("exit_code") != 0:
        return None
    parent = process(state["pid"])
    if (parent is None or str(root) not in parent["command"]
            or str(work / "checkpoints") not in parent["command"]):
        return None
    manifest = read(work / "wandb_manifest.json")
    native = Path(cursor["path"])
    if (native.parent.parent != work / "wandb"
            or cursor.get("run_id") != manifest["id"]
            or native.name != f"run-{manifest['id']}.wandb"):
        raise ValueError("Native record does not belong to this training run")
    children = Path("/proc") / str(parent["pid"]) / "task" / str(parent["pid"])
    identities = [process(int(pid))
                  for pid in (children / "children").read_text().split()]
    backends = [p for p in identities if p and p["parent"] == parent["pid"]
                and holds_native(p["pid"], native)]
    if len(backends) != 1:
        return None  # Never guess a PID when writer ownership cannot be proved.
    verify(root, plan)
    proof["checkpoint_verification"] = checkpoint_proof(root, checkpoint, job["updates"])
    return dict(stage="verified", parent=parent, backend=backends[0], native=str(native),
                training=proof, relay_before=cursor, signals=[],
                verified_unix=time.time())


def send(identity, sig, record, path):
    # Linux pidfds prevent a reused numeric PID receiving our signal.
    if not same_process(identity):
        return
    fd = os.pidfd_open(identity["pid"])
    try:
        if not same_process(identity):
            return
        signal.pidfd_send_signal(fd, sig)
        record["signals"].append(dict(pid=identity["pid"], signal=sig.name,
                                      unix=time.time()))
        write(path, record)
    finally:
        os.close(fd)


def advance(record, path, cursor, updates):
    backend, parent = record["backend"], record["parent"]
    native = Path(record["native"])
    if record["stage"] == "verified":
        send(backend, signal.SIGINT, record, path)
        time.sleep(2)
        send(backend, signal.SIGINT, record, path)
        record["stage"] = "waiting_for_flush_and_upload"
        write(path, record)
    if record["stage"] == "waiting_for_flush_and_upload":
        log = native.parent / "logs/debug-internal.log"
        if (holds_native(backend["pid"], native)
                or f"close: {native}" not in log.read_text(errors="replace")
                or not uploaded(cursor, native, updates)):
            return
        record["relay_after_flush"] = cursor
        write(path, record)
        send(parent, signal.SIGINT, record, path)
        record["stage"] = "releasing_completed_cli"
        record["released_unix"] = time.time()
        write(path, record)
    if record["stage"] == "releasing_completed_cli":
        if time.time() - record["released_unix"] < 5:
            return
        # Both writer closure and full remote acknowledgement are already saved.
        send(backend, signal.SIGTERM, record, path)
        if same_process(parent):
            send(parent, signal.SIGTERM, record, path)
        record["stage"] = "released"
        write(path, record)


def run(root, out, grace=600):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            # Materialization extends the frozen manifest; reload it each poll.
            plan = read(root / "plan.json")
            queue = read(root / "queue/status.json")
            relay_path = root / "telemetry_relay/cursors.json"
            relay = read(relay_path) if relay_path.exists() else {}
            errors, stages = {}, {}
            for job in plan["jobs"]:
                key = job["id"]
                path = out / f"{key}.json"
                try:
                    record = read(path) if path.exists() else candidate(
                        root, plan, job, queue.get("jobs", {}).get(key, {}),
                        relay, grace)
                    if record:
                        write(path, record)
                        advance(record, path, relay.get(key, {}), job["updates"])
                        stages[key] = record["stage"]
                except Exception as error:
                    errors[key] = repr(error)
            write(out / "status.json", dict(pid=os.getpid(), updated_unix=time.time(),
                  grace_seconds=grace, queue_stage=queue["stage"], jobs=stages,
                  errors=errors))
            if (queue["stage"] in {"completed", "finished_with_failures"}
                    and not errors and all(s == "released" for s in stages.values())):
                return
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grace", type=int, default=600)
    args = parser.parse_args()
    if args.grace < 600:
        parser.error("Retain at least ten minutes for normal SDK shutdown")
    run(args.root.resolve(), args.out.resolve(), args.grace)
