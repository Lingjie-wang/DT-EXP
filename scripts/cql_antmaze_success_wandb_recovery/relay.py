"""Replay saved W&B history at original line offsets without touching training."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import requests
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore
from wandb.sdk.lib import proto_util

from scripts.cql_antmaze_official_5090.common import read, write

MAX_BYTES = 1024 * 1024


def scan(path, offset=None, max_bytes=MAX_BYTES, max_records=20000):
    """Read only complete records present at the beginning of this scan.

    A partial last record is retried at the same byte offset on the next scan.
    Corruption away from the live tail is an error, never silently skipped.
    """
    size = path.stat().st_size
    ds = DataStore()
    ds.open_for_scan(str(path))
    try:
        if offset is not None:
            ds._fp.seek(offset)
            ds._index = offset
        start = ds._index
        cursor, lines, byte_count, exit_code = start, [], 0, None
        for _ in range(max_records):
            before = ds._index
            try:
                data = ds.scan_data()
            except (AssertionError, IndexError):
                if ds._fp.tell() >= size:
                    break
                raise
            if data is None or ds._index > size:
                break
            pb = wandb_internal_pb2.Record()
            pb.ParseFromString(data)
            kind = pb.WhichOneof("record_type")
            if kind == "history":
                # Exactly the conversion used by SendManager.send_history/_save_history.
                line = json.dumps(proto_util.dict_from_proto_list(pb.history.item))
                if lines and byte_count + len(line.encode()) > max_bytes:
                    cursor = before
                    break
                if len(line.encode()) > max_bytes:
                    raise ValueError("Single history row exceeds the relay batch limit")
                lines.append(line)
                byte_count += len(line.encode())
            elif kind == "exit":
                exit_code = pb.exit.exit_code
            cursor = ds._index
        return dict(offset=cursor, histories=lines, exit_code=exit_code,
                    caught_up=cursor >= size, scanned_bytes=cursor - start)
    finally:
        ds._fp.close()


def safe_summary(path):
    try:
        value = read(path)
        if not isinstance(value, dict):
            raise ValueError("Expected a native W&B summary object")
        return value
    except (FileNotFoundError, json.JSONDecodeError):
        # The SDK rewrites its summary in place while the trainer is running.
        return None


def send(session, endpoint, state, batch, summary, complete=False):
    files = {}
    if batch["histories"]:
        files["wandb-history.jsonl"] = dict(offset=state.get("history_lines", 0),
                                            content=batch["histories"])
    if summary is not None:
        files["wandb-summary.json"] = dict(offset=0, content=[json.dumps(summary)])
    payload = dict(files=files, complete=complete, failed=False, dropped=0)
    if complete:
        payload["exitcode"] = state.get("exit_code", 0)
    # Unlike the old SDK's Session.timeout attribute, this is an actual timeout.
    response = session.post(endpoint, json=payload, timeout=(10, 30))
    response.raise_for_status()
    # Only advance after acknowledgement. Replaying an unknown-outcome POST uses
    # the same explicit line offsets and contents, so it cannot append duplicates.
    result = dict(state, offset=batch["offset"],
                  history_lines=state.get("history_lines", 0) + len(batch["histories"]),
                  acknowledged_unix=time.time(), complete=complete)
    if batch["exit_code"] is not None:
        result["exit_code"] = batch["exit_code"]
    if summary is not None:
        result["summary_step"] = summary.get("_step")
        result["summary_timestamp"] = summary.get("_timestamp")
    if batch["histories"]:
        result["last_history_step"] = json.loads(batch["histories"][-1]).get("_step")
    return result


def discover(root, plan):
    entries = [("comparison", root / "comparison", root / "comparison/manifest.json")]
    entries += [(j["id"], root / j["id"] / "training",
                 root / j["id"] / "training/wandb_manifest.json") for j in plan["jobs"]]
    result = []
    for name, work, manifest in entries:
        if not manifest.exists():
            continue
        identity = read(manifest)["id"]
        files = list((work / "wandb").glob(f"run-*/run-{identity}.wandb"))
        if len(files) == 0:
            continue
        if len(files) != 1:
            raise ValueError(f"Inspect multiple native sessions before relaying {name}")
        path = files[0]
        result.append(dict(name=name, id=identity, path=path,
                           summary=path.parent / "files/wandb-summary.json"))
    return result


def relay(root, out, once=False):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(root / "plan.json")
        session = requests.Session()
        session.auth = ("api", os.environ["WANDB_API_KEY"])
        session.headers["User-Agent"] = "DT-EXP-native-history-recovery/1"
        states_path = out / "cursors.json"
        states = read(states_path) if states_path.exists() else {}
        while True:
            errors, all_caught_up = {}, True
            queue = read(root / "queue/status.json")
            for entry in discover(root, plan):
                name, path = entry["name"], entry["path"]
                stat = path.stat()
                signature = dict(path=str(path), device=stat.st_dev, inode=stat.st_ino,
                                 run_id=entry["id"])
                state = states.get(name, dict(**signature, history_lines=0))
                if any(state.get(k) != v for k, v in signature.items()):
                    raise ValueError("Native source changed; inspect before recovery")
                endpoint = (f"https://api.wandb.ai/files/{plan['wandb_entity']}/"
                            f"{plan['wandb_project']}/{entry['id']}/file_stream")
                try:
                    # Bounded batches prevent a large run starving the comparison.
                    for _ in range(8):
                        batch = scan(path, state.get("offset"))
                        summary = safe_summary(entry["summary"])
                        job_state = queue.get("jobs", {}).get(name, {})
                        terminal = (queue.get("stage") in {
                            "completed", "finished_with_failures"}
                            if name == "comparison" else job_state.get("state")
                            in {"completed", "failed"})
                        code = batch["exit_code"]
                        if code is None:
                            code = state.get("exit_code", job_state.get("exit"))
                        if code is not None:
                            state = dict(state, exit_code=code)
                        complete = terminal and batch["caught_up"] and code is not None
                        state = send(session, endpoint, state, batch, summary, complete)
                        states[name] = state
                        write(states_path, states)
                        if batch["caught_up"] or batch["scanned_bytes"] == 0:
                            break
                    all_caught_up &= batch["caught_up"]
                except Exception as error:
                    errors[name] = dict(type=type(error).__name__, message=str(error))
                    all_caught_up = False
                    # Drop this HTTP connection pool after a bounded network failure.
                    session.close()
                    session = requests.Session()
                    session.auth = ("api", os.environ["WANDB_API_KEY"])
                write(out / "status.json", dict(pid=os.getpid(),
                      updated_unix=time.time(),
                      errors=errors, runs=states, caught_up=all_caught_up))
            if once or (states and all(s.get("complete") for s in states.values())
                        and len(states) == len(plan["jobs"]) + 1):
                return
            time.sleep(30 if all_caught_up else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    relay(args.root.resolve(), args.out.resolve(), args.once)
