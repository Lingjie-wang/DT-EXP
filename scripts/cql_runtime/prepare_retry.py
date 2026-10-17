"""Clone selected failed CQL runs, preserving frozen data and historical attempts."""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def prepare(previous, root, arms, runner):
    repository = Path(__file__).resolve().parents[2]
    p = json.loads((previous / "protocol.json").read_text())
    if runner not in ["cql_delayed", "cql_shapley"]:
        raise ValueError("Unknown CQL runner")
    if not arms or any(arm not in p["runs"] for arm in arms):
        raise ValueError("Explicit existing arms are required")
    if any((previous / arm / "user_cancellation.json").exists() for arm in arms):
        raise ValueError("Refusing to restart a user-cancelled arm")
    replacement = f"scripts/{runner}/train.py"
    for name, expected in p["source_sha256"].items():
        if digest(previous / "source" / name) != expected:
            raise ValueError(f"Historical source changed: {name}")
    for arm in arms:
        for name, expected in p["runs"][arm]["data_sha256"].items():
            if digest(previous / arm / name) != expected:
                raise ValueError(f"Historical data changed: {arm}/{name}")
    root.mkdir(parents=True, exist_ok=False)
    for name in p["source_sha256"]:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = (repository / name if name == replacement else
                  previous / "source" / name)
        shutil.copy2(source, target)
    for name in ["prepare_retry.py", "validate.sbatch"]:
        target = root / "source/scripts/cql_runtime" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / "scripts/cql_runtime" / name, target)
    runs = {}
    for arm in arms:
        (root / arm).mkdir()
        for name in p["runs"][arm]["data_sha256"]:
            shutil.copy2(previous / arm / name, root / arm / name)
        spec = copy.deepcopy(p["runs"][arm])
        spec["retry_of_wandb_id"] = spec["wandb_id"]
        spec["wandb_id"] = hashlib.sha256(f"{root.name}/{arm}".encode()).hexdigest()[:12]
        spec["wandb_name"] += "-retry1-env"
        runs[arm] = spec
    p["runs"] = runs
    p["retry_of"] = str(previous.resolve())
    p["retry_reason"] = (
        "D4RL optional Adroit import failed on missing mjrl, skipping gym_mujoco; "
        "explicitly import official d4rl.gym_mujoco before gym.make"
    )
    p["original_repository_base"] = p["repository_base"]
    p["repository_base"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    p["wandb_group"] = root.name
    p["source_sha256"] = {
        str(path.relative_to(root / "source")): digest(path)
        for path in (root / "source").rglob("*") if path.is_file()
    }
    p["attribution_provenance_root"] = str(previous.resolve())
    if "preparation_sha256" in p:
        p["original_preparation_sha256"] = p.pop("preparation_sha256")
    write(root / "protocol.json", p)
    # Separate smoke outputs; formal jobs still run a fresh preflight process.
    validation = root / "validation"
    validation.mkdir()
    (validation / "source").symlink_to(root / "source", target_is_directory=True)
    for arm in arms:
        (validation / arm).mkdir()
        for name in runs[arm]["data_sha256"]:
            (validation / arm / name).symlink_to(root / arm / name)
    write(validation / "protocol.json", p)
    print(json.dumps(dict(root=str(root), arms=arms, retry_of=str(previous))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--runner", choices=["cql_delayed", "cql_shapley"],
                        required=True)
    args = parser.parse_args()
    prepare(args.previous.resolve(), args.root.resolve(), args.arms, args.runner)
