"""Read-only cross-arm audit of local configs, paired data/RNG and loss terms."""

import argparse
import json
import math
from pathlib import Path

ARMS = ("dt", "recent_ref", "adaptive_pref")


def read(path):
    with open(path) as stream:
        return json.load(stream)


def audit(root, require_complete=False):
    directories = []
    for arm in ARMS:
        matches = list((root / arm).glob("*/config.json"))
        if len(matches) != 1:
            raise AssertionError(f"Expected one run for {arm}, got {len(matches)}")
        directories.append(matches[0].parent)
    configs = [read(path / "config.json") for path in directories]
    allowed = {"arm", "name", "checkpoints_path", "wandb_id", "wandb_url",
               "slurm_job_id"}
    keys = set().union(*(set(config) for config in configs))
    for key in keys - allowed:
        if any(config.get(key) != configs[0].get(key) for config in configs[1:]):
            raise AssertionError(f"Configuration/provenance mismatch: {key}")
    if tuple(config["arm"] for config in configs) != ARMS:
        raise AssertionError("Wrong experiment arms")
    rows = [{row["step"]: row for row in read(path / "pairing_audit.json")}
            for path in directories]
    common = sorted(set.intersection(*(set(arm_rows) for arm_rows in rows)))
    if not common:
        raise AssertionError("No shared audited steps")
    equal_fields = ("dt_batch_sha256", "torch_rng_sha256", "dt_sample_stream_sha256",
                    "reference_sample_stream_sha256")
    for step in common:
        fields = equal_fields
        if step <= configs[0]["auxiliary_start"] + 1:
            fields += ("model_sha256", "dt_loss", "total_loss")
        for key in fields:
            if len({arm[step][key] for arm in rows}) != 1:
                raise AssertionError(f"Pairing mismatch at {step}: {key}")
    for arm, path in zip(ARMS, directories):
        with open(path / "metrics.jsonl") as stream:
            for line in stream:
                row = json.loads(line)
                if "train_loss" not in row:
                    continue
                pref, ref = (row["train/weighted_preference_loss"],
                             row["train/weighted_reference_loss"])
                if (arm != "adaptive_pref" and pref != 0) or (arm == "dt" and ref != 0):
                    raise AssertionError(f"Unexpected enabled loss in {arm}")
                if not math.isclose(row["train/total_loss"],
                                    row["train_loss"] + pref + ref,
                                    abs_tol=1e-6, rel_tol=1e-5):
                    raise AssertionError(f"Incorrect loss sum in {arm}")
    complete = all((path / "summary.json").exists() for path in directories)
    result = {"pairing_passed": True, "all_complete": complete,
              "audited_steps": common, "gpu": configs[0]["gpu"],
              "initial_model_sha256": configs[0]["initial_model_sha256"],
              "strict_pair_count": configs[0]["strict_pair_count"],
              "wandb_urls": dict(zip(ARMS, [config["wandb_url"] for config in configs]))}
    if complete:
        summaries = [read(path / "summary.json") for path in directories]
        for key in ("completed_updates", "dt_sample_stream_sha256",
                    "reference_sample_stream_sha256", "last_five_steps"):
            if len({json.dumps(summary[key]) for summary in summaries}) != 1:
                raise AssertionError(f"Final mismatch: {key}")
        result["completed_updates"] = summaries[0]["completed_updates"]
        result["results"] = dict(zip(ARMS, summaries))
    elif require_complete:
        raise AssertionError("Not all arms are complete")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.root, arguments.require_complete), indent=2))
