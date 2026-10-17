"""Make portable seed-1 campaigns from validated Slurm CQL data and source."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from scripts.cql_delayed.common import digest, read, verify, write

def prepare(previous, root, arms, seed=1):
    if seed != 1:
        raise ValueError("This campaign reserves policy/evaluation seed 1 for 5090")
    p = copy.deepcopy(verify(previous))
    allowed = {"medium", "medium_replay", "shapley"}
    if not arms or set(arms) - allowed or any(a not in p["runs"] for a in arms):
        raise ValueError("Only the three requested active CQL arms are allowed")
    for arm in arms:
        if (previous / arm / "user_cancellation.json").exists():
            raise ValueError("Refuse a cancelled arm")
        if p["runs"][arm]["seed"] != 0:
            raise ValueError("Expected Slurm seed 0 reference")
        if arm == "shapley" and not p.get("shapley_gate", {}).get("passed"):
            raise ValueError("Shapley prediction gate did not pass")
        for name, expected in p["runs"][arm]["data_sha256"].items():
            if digest(previous / arm / name) != expected:
                raise ValueError(f"Historical data changed: {arm}/{name}")
    root.mkdir(parents=True, exist_ok=False)
    for name in p["source_sha256"]:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous / "source" / name, target)
    runs = {}
    for arm in arms:
        (root / arm).mkdir()
        spec = copy.deepcopy(p["runs"][arm])
        for name in spec["data_sha256"]:
            shutil.copy2(previous / arm / name, root / arm / name)
        name = spec["wandb_name"].removesuffix("-retry1-env")
        spec.update(seed=seed, wandb_name=name.replace("seed0", "seed1") + "-5090",
                    wandb_id=hashlib.sha256(f"{root.name}/{arm}".encode()).hexdigest()[:12],
                    slurm_reference_wandb_id=spec["wandb_id"])
        spec.pop("retry_of_wandb_id", None)
        config_path = root / "source" / spec["config"]
        config = yaml.safe_load(config_path.read_text())
        config.update(seed=seed, name=spec["wandb_name"], group=root.name)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        runs[arm] = spec
    p.update(runs=runs, wandb_group=root.name,
             migration={"reference_root": str(previous.resolve()),
                        "reference_protocol_sha256": digest(previous / "protocol.json"),
                        "target_host": "5090", "policy_and_eval_seed": seed,
                        "source_python_changed": False,
                        "attribution": "Reuse frozen rewards and attribution seeds",
                        "runtime": "Existing isolated 5090 environment, read-only; "
                                   "Torch 2.7.1+cu128, Python3.10"})
    p["changes"].append("5090: policy/evaluation seed 0->1; logging identifiers changed")
    p["source_sha256"] = {name: digest(root / "source" / name)
                           for name in p["source_sha256"]}
    for name, expected in read(previous / "protocol.json")["source_sha256"].items():
        if name.endswith(".py") and p["source_sha256"][name] != expected:
            raise ValueError(f"Python source changed: {name}")
    write(root / "protocol.json", p)
    # Relative links remain valid when the campaign is transferred to another host.
    validation = root / "validation"
    validation.mkdir()
    (validation / "source").symlink_to("../source", target_is_directory=True)
    for arm in arms:
        (validation / arm).mkdir()
        for name in runs[arm]["data_sha256"]:
            (validation / arm / name).symlink_to(f"../../{arm}/{name}")
    write(validation / "protocol.json", p)
    print(json.dumps({"root": str(root), "arms": arms, "seed": seed}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    prepare(args.previous.resolve(), args.root.resolve(), args.arms)
