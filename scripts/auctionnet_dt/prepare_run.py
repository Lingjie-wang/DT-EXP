"""Freeze an independent run of the unchanged PRGS AuctionNet DT Python source."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

PRGS_COMMIT = "33bad7bc3e3f7cbfbefb68294b8e61069a6e58d4"
AUCTIONNET_COMMIT = "c20de631a40e24669d1d6a8e2acefd8f492f172d"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(upstream, data, csvs, root, smoke):
    if root.exists():
        raise FileExistsError(f"Preserve previous attempts: {root}")
    audit = json.loads((data / "audit.json").read_text())
    if not smoke and audit["training_periods"] != list(range(7, 14)):
        raise ValueError("Full run requires training periods P7-P13")
    expected_csvs = [f"period-{p}.csv" for p in range(14, 21)]
    if not smoke and [p.name for p in csvs] != expected_csvs:
        raise ValueError("Full run requires all held-out periods P14-P20 in order")
    for path in csvs:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, expected in audit["sha256"].items():
        if digest(data / name) != expected:
            raise ValueError(f"Prepared data changed: {name}")
    shutil.copytree(upstream, root / "upstream",
                    ignore=shutil.ignore_patterns("__pycache__", ".git", "data", "log",
                                                  "model"))
    checkout = root / "upstream"
    original = {str(p.relative_to(upstream)): digest(p)
                for p in upstream.rglob("*") if p.is_file()
                and p.suffix in {".py", ".yaml"}}
    pinned = json.loads(Path(__file__).with_name("upstream_manifest.json").read_text())
    if original != pinned["files"]:
        raise ValueError("Official PRGS source does not match the pinned revision")
    env_path = checkout / "config/env/AuctionNet.yaml"
    config = yaml.safe_load(env_path.read_text())
    config["dataset_path_small"] = str(data / "training_data_small.pkl")
    config["normalize_dict_path"] = str(data / "normalize_dict.pkl")
    config["test_csv_list_small"] = [str(p) for p in csvs]
    if smoke:
        config["env_targets"] = [1.0]
    env_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if smoke:
        default_path = checkout / "config/default.yaml"
        default = yaml.safe_load(default_path.read_text())
        default.update(num_steps_per_iter=10, max_iters=1)
        default_path.write_text(yaml.safe_dump(default, sort_keys=False))
    effective = {}
    for name in ("default.yaml", "env/AuctionNet.yaml", "algo/dt.yaml"):
        effective.update(yaml.safe_load((checkout / "config" / name).read_text()))
    assert effective["is_stitch"] is False and effective["model_type"] == "dt"
    protocol = {
        "upstream": "https://github.com/deligentfool/PRGS", "commit": PRGS_COMMIT,
        "data_generator_repo": "https://github.com/alimama-tech/AuctionNet",
        "data_generator_commit": AUCTIONNET_COMMIT,
        "upstream_original_sha256": original,
        "runtime_sha256": {name: digest(checkout / name) for name in original},
        "config": effective, "data_audit": audit, "smoke": smoke,
        "python_source_modified": False,
        "changes": {
            "data": "Reconstructed from public final/general raw CSV; unpublished "
                    "PRGS pickle equivalence unverified",
            "paths": "Only dataset_path_small, normalize_dict_path, test_csv_list_small",
            "runtime": "Python 3.10 / Torch 2.7.1+cu128 for RTX 5090",
            "smoke": "10 updates, 1 target, P7-P8 training and one P14 advertiser"
                     if smoke else None,
        },
        "rng": "Official main.py does not seed RNG; no seed injection or patch",
        "evaluation": "Official evaluator, targets and max_ep_len=96 unchanged; "
                      "public data generator uses 48 steps; do not silently reconcile",
        "primary_result": "final 100000-update target_1.0 score; no target selection",
    }
    for name in original:
        if name.endswith(".py"):
            assert protocol["runtime_sha256"][name] == original[name]
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"root": str(root), "smoke": smoke,
                      "total_updates": effective["max_iters"]
                      * effective["num_steps_per_iter"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare(args.upstream.resolve(), args.data.resolve(),
            [p.resolve() for p in args.csvs], args.root.resolve(), args.smoke)


if __name__ == "__main__":
    main()
