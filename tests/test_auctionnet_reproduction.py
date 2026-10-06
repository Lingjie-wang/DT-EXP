"""Validate selection, replicate consistency and between-run statistics."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.auctionnet_dt_reproduction.summarize import (
    final_scores,
    summarize,
    validate_protocol,
)

def protocol():
    return {
        "dataset_version": "non-final/general", "python_source_modified": False,
        "commit": "pinned", "upstream_original_sha256": {"main.py": "sha"},
        "runtime_sha256": {"main.py": "sha"}, "data_audit": {"sha256": "data"},
        "config": {"delayed_reward": False, "is_stitch": False, "model_type": "dt",
                   "env_targets": [1., .8, .6, .4, .2], "num_eval_episodes": 1,
                   "max_iters": 10, "num_steps_per_iter": 10000},
    }


def console(offset):
    lines = []
    for iteration in range(1, 11):
        for target in range(5):
            # Earlier checkpoints score higher: never select their maximum.
            base = offset + target + (1000 if iteration < 10 else 0)
            for period in range(14, 21):
                lines.append(f"[EVAL]period-{period}.csv Period Score: {base + period}")
            lines.append(f"[EVAL] Overall mean score across 7 CSVs: {base + 17}")
        lines.extend(["=" * 20, f"Iteration {iteration}", "train_loss: 0.1"])
    return "\n".join(lines) + "\n"


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.roots = [Path(self.temp.name) / str(i) for i in range(3)]
        for i, root in enumerate(self.roots):
            root.mkdir()
            (root / "protocol.json").write_text(json.dumps(protocol()))
            (root / "status.json").write_text(json.dumps({
                "status": "completed", "completed_updates": 100000,
                "upstream_python_unchanged": True}))
            (root / "console.log").write_text(console(i * 2))

    def test_final_checkpoint_fixed_targets_and_between_run_std(self):
        result = summarize(self.roots)
        row = result["targets"]["1.0"]["periods"]["14"]
        self.assertEqual(row["values"], [14., 16., 18.])
        self.assertEqual(row["mean"], 16.)
        self.assertEqual(row["std_sample"], 2.)
        self.assertAlmostEqual(row["std_population"], (8 / 3) ** .5)
        self.assertEqual(result["targets"]["1.0"]["mean"], 19.)
        self.assertEqual(result["targets"]["0.2"]["mean"], 23.)
        self.assertIsNone(result["seed_values"])

    def test_incomplete_or_repeated_iterations_rejected(self):
        for text in (console(0).replace("Iteration 10", "Iteration 9"),
                     console(0).replace("[EVAL]period-20.csv", "missing"),
                     console(0).replace("period-14.csv", "period-15.csv")):
            with self.assertRaises(ValueError):
                final_scores(text)

    def test_mean_crosscheck(self):
        with self.assertRaisesRegex(ValueError, "overall mean"):
            final_scores(console(0).replace("Period Score: 14\n", "Period Score: 99\n"))

    def test_distinct_completed_runs_required(self):
        with self.assertRaises(ValueError):
            summarize([self.roots[0]] * 3)
        (self.roots[2] / "status.json").write_text(json.dumps({
            "status": "running", "completed_updates": 100000}))
        with self.assertRaisesRegex(ValueError, "not finished"):
            summarize(self.roots)

    def test_data_or_training_changes_rejected(self):
        for key, value in (("data_audit", {"sha256": "other"}), ("commit", "other")):
            changed = protocol()
            changed[key] = value
            (self.roots[2] / "protocol.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "differ"):
                summarize(self.roots)

    def test_delayed_or_modified_source_is_not_a_baseline(self):
        baseline = protocol()
        validate_protocol(baseline)
        for key, value in (("delayed_reward", True), ("num_eval_episodes", 32)):
            changed = copy.deepcopy(baseline)
            changed["config"][key] = value
            with self.assertRaises(ValueError):
                validate_protocol(changed)
        baseline["runtime_sha256"]["main.py"] = "modified"
        with self.assertRaisesRegex(ValueError, "Changed official Python"):
            validate_protocol(baseline)


if __name__ == "__main__":
    unittest.main()
