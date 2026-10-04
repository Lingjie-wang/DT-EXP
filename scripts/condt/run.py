"""Run the author's instrumented CLI in fresh, isolated subprocesses."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def launch(root, arm, phase, protocol, device):
    work = root / arm / phase
    checkout = work / "upstream"
    record_dir = work / "records" if phase == "preflight" else root / arm
    if (work / "console.log").exists():
        raise RuntimeError("Refusing to overwrite a previous attempt")
    for name, expected in protocol["execution_sha256"].items():
        assert digest(checkout / name) == expected, name
    command = [
        sys.executable,
        "-u",
        "experiment_clean.py",
        "--env",
        "hopper",
        "--dataset",
        "medium",
        "--model_type",
        protocol["arms"][arm]["model_type"],
        "--device",
        device,
        "--name",
        f"{root.name}-{arm}",
    ]
    if protocol["arms"][arm]["pretrain"]:
        command.append("--pretrain")
    if phase == "preflight":
        command += [
            "--max_iters",
            "1",
            "--num_steps_per_iter",
            "2",
            "--num_eval_episodes",
            "1",
        ]
    environment = os.environ.copy()
    environment.update(
        CONDT_RECORD_DIR=str(record_dir),
        CONDT_TRAIN_SEED=str(protocol["seed"]),
        WANDB_MODE="disabled",
        PYTHONUNBUFFERED="1",
    )
    environment.pop("CONDT_PREFLIGHT_PRETRAIN_STEPS", None)
    if phase == "preflight":
        environment["CONDT_PREFLIGHT_PRETRAIN_STEPS"] = "2"
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    write(work / "command.json", dict(argv=command, cwd=str(checkout / "gym")))
    with open(work / "console.log", "w", buffering=1) as stream:
        result = subprocess.run(
            command,
            cwd=checkout / "gym",
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(
            f"Author {phase} exited {result.returncode}: {work / 'console.log'}"
        )
    expected = [0, 2] if phase == "preflight" else protocol["eval_updates"]
    actual = [
        json.loads(path.read_text())["main_updates"]
        for path in sorted((record_dir / "evaluations").glob("eval_*.json"))
    ]
    assert actual == expected, (actual, expected)
    for path in (record_dir / "evaluations").glob("eval_*.json"):
        outputs = json.loads(path.read_text())["outputs"]
        assert set(outputs) == {str(s) for s in protocol["evaluation_seeds"]}
        episodes = 1 if phase == "preflight" else protocol["eval_episodes_per_seed"]
        assert all(len(value["returns"]) == episodes for value in outputs.values())
    for name, expected_hash in protocol["execution_sha256"].items():
        assert digest(checkout / name) == expected_hash, name
    return record_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("dt", "condt"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    status_path = root / args.arm / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("status") not in ("queued", "preflight_passed"):
            raise RuntimeError(
                "Existing attempt must be preserved; prepare another campaign"
            )
    try:
        for name, expected in protocol["harness_sha256"].items():
            assert digest(root / "source" / name) == expected, name
        assert digest(Path(protocol["dataset_path"])) == protocol["dataset_sha256"]
        preflight_marker = root / args.arm / "preflight_passed.json"
        if preflight_marker.exists():
            assert json.loads(preflight_marker.read_text())["device"] == args.device
        if not preflight_marker.exists():
            write(
                status_path, dict(status="preflight", main_updates=0, pretrain_updates=0)
            )
            record_dir = launch(root, args.arm, "preflight", protocol, args.device)
            write(
                preflight_marker,
                dict(passed=True, records=str(record_dir), device=args.device),
            )
        if args.preflight_only:
            write(
                status_path,
                dict(status="preflight_passed", main_updates=0, pretrain_updates=0),
            )
            return
        launch(root, args.arm, "training", protocol, args.device)
        write(
            status_path,
            dict(
                status="completed",
                main_updates=protocol["main_updates"],
                pretrain_updates=protocol["arms"][args.arm]["pretrain_updates"],
            ),
        )
    except BaseException as error:
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        write(
            status_path,
            dict(
                status="failed",
                main_updates=previous.get("main_updates", 0),
                pretrain_updates=previous.get("pretrain_updates", 0),
                error=repr(error),
            ),
        )
        raise


if __name__ == "__main__":
    main()
