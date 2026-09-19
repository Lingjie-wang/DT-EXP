"""Protocol tests; GPU smoke also checks actual checkpoint/data/RNG pairing."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import v3_high_newseeds as workflow
from v3_high_newseeds import (
    branch_flags,
    plain,
    root_path,
    SEEDS,
    validate_seed,
    verify_sources,
)

def mapping(argv):
    return {key.removeprefix("--"): value for key, value in zip(argv[::2], argv[1::2])}


class FreshSeedTests(unittest.TestCase):
    def run_journal(self, root, send_metrics=True, nonfinite=False):
        fake = types.ModuleType("wandb")
        run = types.SimpleNamespace(
            id="test-run", url=None, name="test-run", config={"train_seed": 3},
            summary={"nested": {"value": 1.0}}, log=mock.Mock(),
        )
        fake.run = None
        fake.log = mock.Mock(side_effect=AssertionError("Pre-init logger called"))
        fake.finish = mock.Mock()

        def initialize():
            fake.run = run
            fake.log = run.log  # Reproduce W&B's actual initialization behavior.
            return run

        fake.init = initialize

        def entry(*args, **kwargs):
            fake.init()
            if send_metrics:
                fake.log({"train_loss": float("nan") if nonfinite else 0.25}, step=3)
                fake.log({"eval/score": 42.0}, step=3)
            fake.finish()

        record = Path(root) / "records"
        arguments = types.SimpleNamespace(
            record_dir=str(record), entry="hard_fork_dt.py", remaining=[],
        )
        with mock.patch.dict(sys.modules, {"wandb": fake}), \
                mock.patch.object(workflow.runpy, "run_path", side_effect=entry), \
                mock.patch.object(sys, "argv", ["test"]), \
                mock.patch.object(sys, "path", list(sys.path)):
            workflow.journal_worker(arguments)
        return record, run

    def test_journal_survives_wandb_init_rebinding_and_forwards_once(self):
        with tempfile.TemporaryDirectory() as root:
            record, run = self.run_journal(root)
            rows = [json.loads(line) for line in
                    (record / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(rows, [
                {"step": 3, "train_loss": 0.25}, {"step": 3, "eval/score": 42.0},
            ])
            self.assertEqual(run.log.call_count, 2)
            self.assertEqual(workflow.metric_rows(record)[3]["train_loss"], 0.25)
            self.assertEqual(workflow.metric_rows(record)[3]["eval/score"], 42.0)
            self.assertTrue((record / "completed.json").is_file())
            self.assertEqual(workflow.read_json(record / "summary.json"), run.summary)

    def test_missing_metrics_fails_before_completion_marker(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "without a local metrics"):
                self.run_journal(root, send_metrics=False)
            record = Path(root) / "records"
            self.assertTrue((record / "failed.json").is_file())
            self.assertFalse((record / "completed.json").exists())

    def test_nonfinite_metrics_are_not_forwarded_or_marked_complete(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                self.run_journal(root, nonfinite=True)
            record = Path(root) / "records"
            self.assertTrue((record / "failed.json").is_file())
            self.assertFalse((record / "completed.json").exists())

    def test_nested_wandb_summary_mapping_can_be_journaled(self):
        class SummaryLike:
            def keys(self):
                return ["loss"]

            def __getitem__(self, key):
                return 0.5

        self.assertEqual(plain({"nested": SummaryLike()}), {"nested": {"loss": 0.5}})

    def test_old_seeds_are_rejected(self):
        self.assertEqual(SEEDS, (3, 4, 5))
        for seed in (0, 1, 2, 6, None):
            with self.assertRaises(ValueError):
                validate_seed(seed)
        for seed in SEEDS:
            validate_seed(seed)

    def test_original_sources_are_pinned_and_unchanged(self):
        verify_sources()

    def test_only_objective_weights_and_names_differ_within_seed(self):
        for seed in SEEDS:
            options = [mapping(branch_flags(seed, arm, "/tmp/shared.pt",
                       "/tmp/pairs.npz", Path("/tmp") / arm)) for arm in ("dt", "v3")]
            different = {key for key in options[0] if options[0][key] != options[1][key]}
            self.assertEqual(different, {"name", "checkpoints_path",
                                         "preference_weight", "reference_weight"})
            self.assertEqual(options[0]["train_seed"], str(seed))
            self.assertEqual(options[0]["preference_pair_seed"], str(seed))
            self.assertEqual(options[0]["preference_weight"], "0.0")
            self.assertEqual(options[1]["preference_weight"], "0.05")
            self.assertEqual(options[1]["reference_weight"], "0.1")

    def test_original_v3_high_recipe_and_budget(self):
        config = mapping(branch_flags(3, "v3", "shared.pt", "pairs.npz", "output"))
        for key, expected in {
            "paired_resume": "true", "preference_start_step": "50000",
            "update_steps": "100000", "eval_every": "5000", "eval_episodes": "100",
            "preference_mode": "hard_fork", "preference_target_mode": "high_only",
            "reference_anchor_mode": "mse", "dynamic_priority_mix": "0.5",
            "dynamic_priority_ema": "0.9", "preference_margin": "0.05",
            "preference_batch_size": "256", "preference_min_active_pairs": "16",
        }.items():
            self.assertEqual(config[key], expected)

    def test_smoke_has_three_paired_updates_with_intervening_evaluation(self):
        config = mapping(branch_flags(4, "dt", "shared.pt", "pairs.npz", "output",
                                      start=2, stop=5, gate=True))
        self.assertEqual(config["preference_start_step"], "2")
        self.assertEqual(config["update_steps"], "5")
        self.assertEqual(config["eval_every"], "1")
        self.assertEqual(config["eval_episodes"], "1")

    def test_campaign_cannot_escape_experiment_directory(self):
        with self.assertRaises(ValueError):
            root_path("../../another-project")


if __name__ == "__main__":
    unittest.main()
