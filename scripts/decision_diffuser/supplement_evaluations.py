"""Add exact-checkpoint evaluations to a running frozen campaign.

The original trainer and its protocol are never modified. A capture thread retains
atomic rolling checkpoints before they are replaced; a separate process evaluates
their EMA weights under the original seeds/conditions on the existing allocation.
"""
import argparse
import fcntl
import importlib.util
import json
import os
import pickle
import threading
import time
from pathlib import Path

import torch

def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def schedule_arm(requested, original, completed, evaluated):
    future = [step for step in requested if step > completed]
    expected = sorted(set(evaluated) | set(original) | set(future))
    return {
        "completed_at_amendment": completed,
        "expected_eval_updates": expected,
        "supplemental_eval_updates": [step for step in future if step not in original],
        "unavailable_past": [step for step in requested if step not in expected],
    }


def amend(root):
    destination = root / "evaluation_schedule.json"
    if destination.exists():
        raise FileExistsError("An evaluation amendment already exists")
    p = read(root / "protocol.json")
    requested = sorted(
        set(range(100000, p["total_updates"] + 1, 100000)) | {p["total_updates"]}
    )
    result = dict(
        interval=100000,
        requested_eval_updates=requested,
        original_eval_updates=p["eval_updates"],
        created_unix=time.time(),
        reason="User requested evaluation every 100k completed updates",
        training_protocol_unchanged=True,
        arms={},
    )
    for arm in p["arms"]:
        completed = read(root / arm / "status.json")["completed_updates"]
        evaluated = [
            read(path)["completed_updates"] for path in (root / arm).glob("eval_*.json")
        ]
        result["arms"][arm] = schedule_arm(
            requested, p["eval_updates"], completed, evaluated
        )
    write(destination, result)
    print(json.dumps(result, indent=2), flush=True)


def retain_exact(source, destination, target, protocol):
    """Hard-link one immutable inode; validate before assigning a step label."""
    if destination.exists():
        return True
    temporary = destination.with_suffix(".capture")
    # The caller holds the per-arm supplement lock, so this is only its own file.
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        state = torch.load(temporary, map_location="cpu")
        observed = state["completed_updates"]
        if state["protocol"] != protocol:
            raise ValueError("Checkpoint protocol mismatch")
        del state
        if observed < target:
            return False
        if observed > target:
            raise RuntimeError(f"Missed checkpoint {target}; latest is {observed}")
        temporary.replace(destination)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_retained(root, arm, device):
    p = read(root / "protocol.json")
    plan = read(root / "evaluation_schedule.json")["arms"][arm]
    targets = plan["supplemental_eval_updates"]
    directory = root / arm / "supplemental"
    directory.mkdir(parents=True, exist_ok=True)
    lock = open(directory / "lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location(
        "dd_frozen", root / "source/decision_diffuser_repro.py"
    )
    dd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dd)
    dd.verify(root, p)
    torch.set_num_threads(1)
    stop = threading.Event()
    errors = []

    def capture():
        try:
            for target in targets:
                destination = directory / f"step{target:07d}.pt"
                while not destination.exists() and not stop.is_set():
                    status = read(root / arm / "status.json")
                    if status["completed_updates"] >= target:
                        if retain_exact(
                            root / arm / "latest.pt", destination, target, p
                        ):
                            write(
                                directory / f"capture_{target:07d}.json",
                                {
                                    "completed_updates": target,
                                    "checkpoint_sha256": dd.sha(destination),
                                    "retained_unix": time.time(),
                                },
                            )
                            print(f"RETAINED exact checkpoint {target}", flush=True)
                            break
                    if status["status"] in ["completed", "interrupted"]:
                        raise RuntimeError(f"Training ended before capture {target}")
                    stop.wait(5)
        except BaseException as error:
            errors.append(error)
            write(directory / "capture_error.json", {"error": repr(error)})

    thread = threading.Thread(target=capture, daemon=True)
    thread.start()
    completed = []
    try:
        write(
            directory / "status.json",
            {
                "status": "waiting_for_checkpoint",
                "targets": targets,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )
        model = None
        for target in targets:
            result_path = root / arm / f"eval_{target:07d}.json"
            if result_path.exists():
                completed.append(target)
                continue
            checkpoint = directory / f"step{target:07d}.pt"
            while not checkpoint.exists():
                if errors:
                    raise errors[0]
                time.sleep(5)
            if model is None:
                model = dd.build_model(p, root / "source/diffuser").to(device)
                with open(root / "normalizers.pkl", "rb") as stream:
                    normalizers = pickle.load(stream)
            state = torch.load(checkpoint, map_location="cpu")
            assert state["completed_updates"] == target and state["protocol"] == p
            model.load_state_dict(state["ema"])
            del state
            model.eval()
            before = dd.state_hash(model)
            write(
                directory / "status.json",
                {
                    "status": "evaluating",
                    "completed_updates": target,
                    "finished_evaluations": completed,
                    "targets": targets,
                },
            )
            # Same EMA policy, 200 diffusion steps, fixed RTG and episode seeds.
            dd.evaluate(model, normalizers, p, device, target, root / arm)
            assert dd.state_hash(model) == before
            completed.append(target)
            write(
                directory / f"audit_{target:07d}.json",
                {
                    "completed_updates": target,
                    "ema_unchanged": True,
                    "separate_process_from_training": True,
                    "eval_episodes": p["eval_episodes"],
                },
            )
        if errors:
            raise errors[0]
        write(
            directory / "status.json",
            {
                "status": "completed",
                "finished_evaluations": completed,
            },
        )
    except BaseException as error:
        write(
            directory / "status.json",
            {
                "status": "failed",
                "error": repr(error),
                "finished_evaluations": completed,
            },
        )
        raise
    finally:
        stop.set()
        thread.join(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["amend", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["dense", "delayed"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "amend":
        amend(args.root.resolve())
    else:
        if not args.arm:
            parser.error("--arm is required for run")
        evaluate_retained(args.root.resolve(), args.arm, args.device)


if __name__ == "__main__":
    main()
