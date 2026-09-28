"""Read-only pairing audit and score comparison for one two-arm PREFORL job."""

import argparse
import json
from pathlib import Path

def read_json(path):
    with open(path) as source:
        return json.load(source)


def audit(root, require_complete=False):
    paths = [root / "noise-0", root / "noise-0.01"]
    configs = [read_json(path / "config.json") for path in paths]
    allowed = {
        "shadow_noise", "output", "wandb_id", "wandb_url", "slurm_job_id",
        "slurm_array_task_id",
    }
    differences = {
        key: [configs[0].get(key), configs[1].get(key)]
        for key in set(configs[0]) | set(configs[1])
        if key not in allowed and configs[0].get(key) != configs[1].get(key)
    }
    if differences:
        raise AssertionError(f"Unexpected config mismatch: {differences}")
    if [config["shadow_noise"] for config in configs] != [0.0, 0.01]:
        raise AssertionError("Wrong arm noise amplitudes")
    records = [read_json(path / "pairing_audit.json") for path in paths]
    by_step = [{row["step"]: row for row in rows} for rows in records]
    common = sorted(set(by_step[0]) & set(by_step[1]))
    if not common:
        raise AssertionError("No common audited batches yet")
    for step in common:
        if by_step[0][step] != by_step[1][step]:
            raise AssertionError(f"Batch/RNG pairing diverged at step {step}")
    result = {
        "pairing_passed": True,
        "shared_initial_model": configs[0]["initial_model_sha256"],
        "shared_selected_data": configs[0]["selected_data_sha256"],
        "audited_common_steps": common,
        "structural_stream_verified_through": common[-1],
        "gpu": configs[0]["gpu"],
        "source_sha256": configs[0]["source_sha256"],
        "run_names": [f"PREFORL-Noise{config['shadow_noise']:g}-Paired-HCMR-"
                      "delayed-seed0" for config in configs],
        "wandb_urls": [config.get("wandb_url") for config in configs],
    }
    complete = all((path / "summary.json").exists() for path in paths)
    result["both_complete"] = complete
    if complete:
        summaries = [read_json(path / "summary.json") for path in paths]
        for key in ("completed_steps", "cumulative_structural_sha256",
                    "final_sampler_state_sha256", "last_five_steps"):
            if summaries[0][key] != summaries[1][key]:
                raise AssertionError(f"Final pairing mismatch: {key}")
        result["completed_steps"] = summaries[0]["completed_steps"]
        result["last_five_steps"] = summaries[0]["last_five_steps"]
        result["scores"] = {
            key: {"noise_0": summaries[0][key], "noise_0.01": summaries[1][key],
                  "noise_minus_zero": summaries[1][key] - summaries[0][key]}
            for key in ("last_normalized_score", "last_five_mean_normalized_score",
                        "best_normalized_score")
        }
    elif require_complete:
        raise AssertionError("One or both runs have not finished")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.require_complete), indent=2))
