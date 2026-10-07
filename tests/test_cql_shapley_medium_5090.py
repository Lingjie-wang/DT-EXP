"""Medium preparation preserves matched transitions and excludes dense labels."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from algorithms.offline.cql_delayed_data import terminal_return_dataset
from algorithms.offline.shapley_redistribution import folds
from scripts.cql_delayed.common import write
from scripts.cql_shapley_medium_5090.prepare import medium_config, predictor_input
from scripts.cql_shapley_medium_5090.run import validation

class MediumTests(unittest.TestCase):
    def raw(self, episodes=1000):
        count = episodes * 1000
        timeouts = np.zeros(count, dtype=bool)
        timeouts[999::1000] = True
        return dict(observations=np.arange(count, dtype=np.float32).reshape(-1, 1),
                    actions=np.ones((count, 1), dtype=np.float32),
                    rewards=np.full(count, 0.1234567, dtype=np.float32),
                    terminals=np.zeros(count, dtype=bool), timeouts=timeouts)

    def test_medium_input_has_totals_only_and_exact_matched_transitions(self):
        raw = self.raw()
        previous, _ = terminal_return_dataset(raw)
        inputs, transitions, audit = predictor_input(raw, previous)
        self.assertEqual(set(inputs), {"features", "returns", "starts", "ends"})
        self.assertEqual(inputs["features"].shape, (1000, 1000, 2))
        self.assertNotIn("rewards", transitions)
        np.testing.assert_array_equal(inputs["returns"],
                                      raw["rewards"].reshape(1000, 1000)
                                      .astype(np.float64).sum(axis=1))
        for key in transitions:
            np.testing.assert_array_equal(transitions[key], previous[key])
        self.assertEqual(audit["episodes"], 1000)
        previous["actions"][0] = -1
        with self.assertRaises(AssertionError):
            predictor_input(raw, previous)

    def test_wrong_trajectory_count_rejected(self):
        raw = self.raw(202)
        previous, _ = terminal_return_dataset(raw)
        with self.assertRaisesRegex(ValueError, "1000 complete"):
            predictor_input(raw, previous)

    def test_policy_configuration_changes_only_logging(self):
        baseline = dict(env="halfcheetah-medium-v2", seed=1, name="delayed",
                        group="old", discount=0.99, normalize_reward=False,
                        batch_size=256, eval_freq=5000, max_timesteps=1000000)
        config = medium_config(baseline, "new")
        self.assertEqual({k for k in baseline if config[k] != baseline[k]},
                         {"name", "group"})
        self.assertEqual(baseline["name"], "delayed")
        with self.assertRaises(ValueError):
            medium_config(dict(baseline, env="halfcheetah-medium-replay-v2"), "new")

    def test_full_medium_fold_coverage_and_no_leakage(self):
        held_all = []
        for split in folds(1000):
            train, valid, held = [set(split[key]) for key in
                                  ["train", "validation", "held"]]
            self.assertEqual((len(train), len(valid), len(held)), (640, 160, 200))
            self.assertFalse(train & valid or train & held or valid & held)
            held_all.extend(held)
        self.assertEqual(sorted(held_all), list(range(1000)))

    def test_failed_prediction_gate_cannot_reach_gpu_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "protocol.json", dict(source_sha256={},
                                                shapley_gate=dict(passed=False)))
            with self.assertRaisesRegex(ValueError, "gate failed"):
                validation(root)
            self.assertFalse((root / "validation").exists())


if __name__ == "__main__":
    unittest.main()
