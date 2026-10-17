"""Bridge unchanged AuctionNet preprocessing to the PRGS trajectory pickle format.

The PRGS authors did not publish their processed data. This adapter reconstructs
it from public raw data; equivalence to their private pickle is not claimed.
"""

import argparse
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def convert_frame(frame):
    """Use the official DT loader's done splitting and singleton/tail exclusion."""
    trajectories = []
    discarded = 0
    keys = ["deliveryPeriodIndex", "advertiserNumber"]
    for _, group in frame.groupby(keys, sort=True):
        pending = []
        for row in group.sort_values("timeStepIndex").itertuples(index=False):
            pending.append(row)
            if not row.done:
                continue
            if len(pending) > 1:
                trajectories.append({
                    "observations": np.asarray([r.state for r in pending]),
                    "actions": np.asarray([r.action for r in pending]).reshape(-1, 1),
                    "rewards": np.asarray([r.reward for r in pending]),
                    "dones": np.asarray([r.done for r in pending]),
                })
            else:
                discarded += len(pending)
            pending = []
        discarded += len(pending)
    if not trajectories:
        raise ValueError("No complete multi-step training trajectories")
    for trajectory in trajectories:
        assert trajectory["observations"].shape[1:] == (16,)
        assert trajectory["dones"][-1] == 1
        for values in trajectory.values():
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite training data")
    return trajectories, discarded


def prepare_period(raw, generator, output):
    pinned = json.loads(Path(__file__).with_name("upstream_manifest.json").read_text())
    if sha256(generator) != pinned["generator_sha256"]:
        raise ValueError("Official preprocessing source changed")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{raw.stem}.pkl"
    if destination.exists():
        raise FileExistsError(destination)
    spec = importlib.util.spec_from_file_location("official_data_generator", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"READ {raw}", flush=True)
    frame = pd.read_csv(raw)
    periods = sorted(int(x) for x in frame.deliveryPeriodIndex.unique())
    if len(periods) != 1 or periods[0] not in range(7, 14):
        raise ValueError(f"Training input must be a single period in P7-P13: {periods}")
    raw_rows = len(frame)
    print(f"OFFICIAL_PREPROCESS {raw_rows} rows", flush=True)
    processed = module.TrainDataGenerator()._generate_train_data(frame)
    del frame
    trajectories, discarded = convert_frame(processed)
    with destination.open("xb") as stream:
        pickle.dump(trajectories, stream, protocol=4)
    processed.to_csv(output / f"{raw.stem}-rlData.csv", index=False)
    report = {
        "raw": str(raw), "raw_sha256": sha256(raw), "raw_rows": raw_rows,
        "periods": periods, "generator_sha256": sha256(generator),
        "rl_rows": len(processed), "trajectories": len(trajectories),
        "discarded_singleton_or_incomplete_rows": discarded,
        "transitions": sum(len(t["rewards"]) for t in trajectories),
        "pickle_sha256": sha256(destination),
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def assemble(inputs, output, periods):
    if not periods or len(set(periods)) != len(periods):
        raise ValueError("Training periods must be nonempty and unique")
    if any(period not in range(7, 14) for period in periods):
        raise ValueError("P14-P20 are held out and cannot enter the training set")
    output.mkdir(parents=True, exist_ok=False)
    trajectories, audits = [], []
    for period in sorted(periods):
        path = inputs / f"period-{period}.pkl"
        audit = json.loads(path.with_suffix(".json").read_text())
        if audit["pickle_sha256"] != sha256(path) or audit["periods"] != [period]:
            raise ValueError(f"Training input provenance mismatch: {path}")
        with path.open("rb") as stream:
            trajectories.extend(pickle.load(stream))
        audits.append(audit)
    states = np.concatenate([t["observations"] for t in trajectories])
    normalization = {"state_mean": states.mean(axis=0),
                     "state_std": states.std(axis=0) + 1e-6}
    for name, value in (("training_data_small.pkl", trajectories),
                        ("normalize_dict.pkl", normalization)):
        with (output / name).open("xb") as stream:
            pickle.dump(value, stream, protocol=4)
    report = {
        "training_periods": sorted(periods), "test_periods": list(range(14, 21)),
        "source_audits": audits, "trajectories": len(trajectories),
        "transitions": len(states), "state_dim": 16, "action_dim": 1,
        "reward": "official exposed conversion count; original step rewards",
        "normalization": "training states only; std + 1e-6",
        "processed_data_equivalence_to_prgs_authors": "unverified",
        "sha256": {p.name: sha256(p) for p in output.glob("*.pkl")},
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_audits"}))


def smoke_csv(raw, output):
    """Keep all rows of one P14 advertiser for an explicitly limited smoke test."""
    if output.exists():
        raise FileExistsError(output)
    advertiser = None
    count = 0
    for frame in pd.read_csv(raw, chunksize=250000, float_precision="round_trip"):
        if set(frame.deliveryPeriodIndex.unique()) != {14}:
            raise ValueError("Smoke evaluation must use held-out P14")
        if advertiser is None:
            advertiser = int(frame.advertiserNumber.min())
        selected = frame[frame.advertiserNumber == advertiser]
        if len(selected):
            selected.to_csv(output, mode="a", header=count == 0, index=False)
            count += len(selected)
    if not count:
        raise ValueError("No smoke evaluation rows")
    output.with_suffix(".json").write_text(json.dumps({
        "source": str(raw), "source_sha256": sha256(raw), "period": 14,
        "advertiser": advertiser, "rows": count, "sha256": sha256(output),
        "purpose": "execution smoke only, not a benchmark score",
    }, indent=2) + "\n")
    print(f"SMOKE_CSV advertiser={advertiser} rows={count}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    period = sub.add_parser("period")
    period.add_argument("--raw", type=Path, required=True)
    period.add_argument("--generator", type=Path, required=True)
    period.add_argument("--output", type=Path, required=True)
    joined = sub.add_parser("assemble")
    joined.add_argument("--inputs", type=Path, required=True)
    joined.add_argument("--output", type=Path, required=True)
    joined.add_argument("--periods", type=int, nargs="+", default=list(range(7, 14)))
    smoke = sub.add_parser("smoke-csv")
    smoke.add_argument("--raw", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "period":
        prepare_period(args.raw, args.generator, args.output)
    elif args.command == "assemble":
        assemble(args.inputs, args.output, args.periods)
    else:
        smoke_csv(args.raw, args.output)


if __name__ == "__main__":
    main()
