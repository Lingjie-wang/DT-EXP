"""Report three unchanged-code runs without selecting targets or checkpoints."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from scripts.auctionnet_dt_nonfinal.pipeline import predecessor_complete, write

TARGETS = [1.0, 0.8, 0.6, 0.4, 0.2]
PERIODS = list(range(14, 21))
PAPER_DT = [282.0, 290.9, 270.2, 285.3, 216.5, 285.4, 241.0]


def validate_protocol(protocol):
    config = protocol["config"]
    if (protocol.get("dataset_version") != "non-final/general"
            or protocol["python_source_modified"]
            or config["delayed_reward"] or config["is_stitch"]
            or config["model_type"] != "dt"):
        raise ValueError("Require ordinary non-final/general DT, unchanged Python")
    for name, sha in protocol["upstream_original_sha256"].items():
        if name.endswith(".py") and protocol["runtime_sha256"][name] != sha:
            raise ValueError(f"Changed official Python: {name}")
    if (config["env_targets"] != TARGETS or config["num_eval_episodes"] != 1
            or config["max_iters"] != 10 or config["num_steps_per_iter"] != 10000):
        raise ValueError("Unexpected official evaluation/training configuration")


def final_scores(console):
    # Upstream prints all target evaluations BEFORE each Iteration header.
    headers = list(re.finditer(r"^Iteration (\d+)\s*$", console, re.MULTILINE))
    if [int(m[1]) for m in headers] != list(range(1, 11)):
        raise ValueError("Missing or repeated completed iterations")
    block = console[headers[-2].end():headers[-1].start()]
    rows = re.findall(r"\[EVAL\]period-(\d+)\.csv Period Score: (\S+)", block)
    means = re.findall(r"\[EVAL\] Overall mean score across 7 CSVs: (\S+)", block)
    if len(rows) != 35 or len(means) != 5:
        raise ValueError("Incomplete final per-period evaluation")
    result = {}
    for index, target in enumerate(TARGETS):
        batch = rows[index * 7:(index + 1) * 7]
        if [int(period) for period, _ in batch] != PERIODS:
            raise ValueError("Unexpected period order")
        values = [float(score) for _, score in batch]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite evaluation")
        if not math.isclose(statistics.mean(values), float(means[index]), abs_tol=2e-6):
            raise ValueError("Period scores do not match overall mean")
        result[str(target)] = dict(zip(map(str, PERIODS), values))
    return result


def summarize(roots):
    if len(roots) != 3 or len({r.resolve() for r in roots}) != 3:
        raise ValueError("Require three distinct run directories")
    protocols, scores = [], []
    for root in roots:
        status = json.loads((root / "status.json").read_text())
        if not predecessor_complete(status):
            raise ValueError("Run has not finished evaluation")
        protocol = json.loads((root / "protocol.json").read_text())
        validate_protocol(protocol)
        if protocols:
            for key in ("config", "commit", "runtime_sha256"):
                if protocol[key] != protocols[0][key]:
                    raise ValueError(f"Runs differ in {key}")
            if protocol["data_audit"] != protocols[0]["data_audit"]:
                raise ValueError("Runs differ in training data provenance")
        protocols.append(protocol)
        scores.append(final_scores((root / "console.log").read_text()))
    targets = {}
    for target in map(str, TARGETS):
        rows = {}
        for period, paper in zip(map(str, PERIODS), PAPER_DT):
            values = [score[target][period] for score in scores]
            rows[period] = {"values": values, "mean": statistics.mean(values),
                            "std_sample": statistics.stdev(values),
                            "std_population": statistics.pstdev(values),
                            "paper_dt_mean": paper}
        run_means = [statistics.mean(score[target].values()) for score in scores]
        targets[target] = {"periods": rows, "mean": statistics.mean(run_means),
                           "run_means": run_means,
                           "std_sample": statistics.stdev(run_means)}
    return {
        "runs": [str(r) for r in roots], "independent_runs": 3,
        "seed_values": None,
        "rng": "Official main.py does not set seeds; three separate fresh processes. "
               "Not a reproduction of the authors' undisclosed seed values.",
        "primary": "Final 100k updates, target 1.0; fixed before additional runs",
        "std": "Report sample std (ddof=1); also retain population std because "
               "the paper does not specify its std convention.",
        "limitations": ["Author processed-data equivalence unverified",
                        "Paper Table 5 says 32 eval episodes; official AuctionNet "
                        "overrides to 1, preserved here",
                        "Paper Table 3 target/checkpoint selection not specified",
                        "RTX 5090/Torch 2.7.1 environment differs from paper RTX 3090"],
        "targets": targets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write(args.output, summarize(args.runs))


if __name__ == "__main__":
    main()
