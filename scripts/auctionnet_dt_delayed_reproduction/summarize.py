"""Summarize three matching delayed-reward DT runs, without score selection."""

import json
import statistics

from scripts.auctionnet_dt_reproduction.summarize import final_scores, PERIODS, TARGETS

def complete(status):
    if status["status"] in {"starting", "running"}:
        return False
    if (status["status"] != "completed" or status.get("completed_updates") != 100000
            or status.get("runtime_source_unchanged") is not True
            or status.get("upstream_python_unchanged") is not False):
        raise ValueError("Delayed run lacks successful 100k/source verification")
    return True


def validate_protocol(protocol):
    config = protocol["config"]
    if (protocol.get("dataset_version") != "non-final/general"
            or protocol["python_source_modified"] is not True
            or config["delayed_reward"] is not True or config["is_stitch"]
            or config["model_type"] != "dt"):
        raise ValueError("Require the explicit delayed-reward non-final DT variant")
    changed = {name for name, sha in protocol["upstream_original_sha256"].items()
               if name.endswith(".py") and protocol["runtime_sha256"][name] != sha}
    if changed != {"evaluation_bidding.py"} or not protocol.get("reward_protocol"):
        raise ValueError("Unexpected delayed evaluator provenance")
    if (config["env_targets"] != TARGETS or config["num_eval_episodes"] != 1
            or config["max_iters"] != 10 or config["num_steps_per_iter"] != 10000):
        raise ValueError("Unexpected delayed training/evaluation configuration")


def summarize(roots):
    if len(roots) != 3 or len({p.resolve() for p in roots}) != 3:
        raise ValueError("Require three distinct delayed run directories")
    protocols, scores = [], []
    for root in roots:
        if not complete(json.loads((root / "status.json").read_text())):
            raise ValueError("Delayed evaluation has not completed")
        protocol = json.loads((root / "protocol.json").read_text())
        validate_protocol(protocol)
        if protocols:
            for key in ("config", "commit", "runtime_sha256", "data_audit",
                        "reward_protocol"):
                if protocol[key] != protocols[0][key]:
                    raise ValueError(f"Delayed runs differ in {key}")
        protocols.append(protocol)
        scores.append(final_scores((root / "console.log").read_text()))
    targets = {}
    for target in map(str, TARGETS):
        periods = {}
        for period in map(str, PERIODS):
            values = [score[target][period] for score in scores]
            periods[period] = {"values": values, "mean": statistics.mean(values),
                               "std_sample": statistics.stdev(values),
                               "std_population": statistics.pstdev(values)}
        means = [statistics.mean(score[target].values()) for score in scores]
        targets[target] = {"periods": periods, "run_means": means,
                           "mean": statistics.mean(means),
                           "std_sample": statistics.stdev(means),
                           "std_population": statistics.pstdev(means)}
    return {
        "runs": [str(p) for p in roots], "independent_runs": 3, "seed_values": None,
        "rng": "Fresh independent processes, no explicitly fixed seed values; "
               "not paired-seed experiments with ordinary DT",
        "primary": "Final 100k checkpoint, target 1.0; retain all five targets",
        "std": "Sample std ddof=1, also retain population std ddof=0",
        "reward_protocol": protocols[0]["reward_protocol"],
        "task": "User-requested delayed-reward extension, not paper Table 3 baseline",
        "targets": targets,
    }
