"""Prepare the requested delayed-reward task as an explicit independent variant."""

import argparse
import difflib
import json
from pathlib import Path

import yaml

from scripts.auctionnet_dt.prepare_run import digest, prepare

REWARD_CALL = "                                pre_reward=pre_reward\n"
DELAYED_CALL = "                                pre_reward=0.0\n"


def configure_delayed(root):
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    checkout = root / "upstream"
    for name, expected in protocol["runtime_sha256"].items():
        if digest(checkout / name) != expected:
            raise ValueError(f"Prepared source changed before delayed setup: {name}")
    config_path = checkout / "config/env/AuctionNet.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["delayed_reward"]:
        raise ValueError("Refuse to reconfigure an existing delayed run")
    evaluator = checkout / "evaluation_bidding.py"
    original = evaluator.read_text()
    if original.count(REWARD_CALL) != 1:
        raise ValueError("Unexpected upstream evaluator reward call")
    modified = original.replace(REWARD_CALL, DELAYED_CALL)
    config["delayed_reward"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    evaluator.write_text(modified)
    patch = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), modified.splitlines(keepends=True),
        fromfile="official/evaluation_bidding.py",
        tofile="delayed/evaluation_bidding.py"))
    (root / "delayed_reward.patch").write_text(patch)
    protocol["config"]["delayed_reward"] = True
    protocol["dataset_version"] = "non-final/general"
    protocol["python_source_modified"] = True
    protocol["changes"]["data"] = (
        "Same non-final/general P7-P13 trajectories and normalization as the ordinary "
        "DT predecessor; independent snapshot, unchanged original rewards on disk.")
    protocol["changes"]["reward"] = (
        "User-requested terminal-only rewards: official delayed_reward=True loader "
        "moves each trajectory's reward sum to its last transition.")
    protocol["changes"]["python"] = {
        "evaluation_bidding.py": "Only the take_actions pre_reward argument becomes "
                                 "0.0, so no intermediate reward decrements RTG."}
    protocol["reward_protocol"] = {
        "training": "r[0:T-1]=0; r[T-1]=sum(original rewards); no cross-episode sums",
        "evaluation": "All pre-action reward feedback is zero; RTG stays constant. "
                      "Terminal conversions/CPA score are reported after the episode.",
        "observations": "Unchanged, including historical conversion statistics. "
                        "This changes explicit rewards, not the observation space.",
        "metric": "Original total-conversion score with CPA penalty, unchanged",
    }
    protocol["evaluation"] = (
        "Independent delayed-reward evaluator: explicit feedback to DT is zero; "
        "auction simulation, observations, targets, score and max_ep_len=96 unchanged.")
    protocol["runtime_sha256"] = {
        name: digest(checkout / name) for name in protocol["runtime_sha256"]}
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.upstream.resolve(), args.data.resolve(),
            [p.resolve() for p in args.csvs], args.root.resolve(), False)
    configure_delayed(args.root.resolve())


if __name__ == "__main__":
    main()
