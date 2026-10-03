"""Gate the 100k job on real terminal targets, finite weights and full evaluation."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

def verify(directory):
    root = Path(directory)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["complete"] and summary["completed_updates"] == 20
    assert summary["terminal_reward_correction"] and summary["eta"] == 0.01
    metrics = [
        json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()
    ]
    last = metrics[-1]
    before = last["audit/nonzero_rewards_before_cumulative"]
    assert before > 0, "Smoke must actually sample a nonzero terminal reward"
    assert last["audit/nonzero_rewards_after_cumulative"] == before
    assert last["audit/terminal_reward_targets_cumulative"] == before
    assert last["audit/episode_end_samples_cumulative"] >= before
    assert last["audit/terminal_target_abs_error_max"] == 0
    assert all(np.isfinite(value) for record in metrics for value in record.values())
    protocols = (
        "qt_strict_delayed", "actor_12000", "actor_6000", "qt_upstream_feedback"
    )
    for protocol in protocols:
        assert summary["last"][f"eval/{protocol}/episodes"] == 1
        assert np.isfinite(summary["last"][f"eval/{protocol}/normalized_score"])
    for step in (0, 20):
        snapshot = torch.load(root / f"checkpoint_{step}.pt", map_location="cpu")
        assert snapshot["completed_updates"] == step and snapshot["eta"] == 0.01
        for network in ("actor", "critic", "critic_target", "ema_model"):
            assert all(torch.isfinite(t).all() for t in snapshot[network].values())
    result = {"passed": True, "terminal_rewards_preserved_and_supervised": before}
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    verify(sys.argv[1])
