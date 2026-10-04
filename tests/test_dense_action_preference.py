"""Check dense labels, causal target budgets, and safe queue dependency gates."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithms/offline"))
from dense_action_preference import dense_preference_batch, mine_dense_pairs

spec = importlib.util.spec_from_file_location(
    "dense_queue", ROOT / "scripts/dense_preference/queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def trajectory(rewards, states=None):
    rewards = np.asarray(rewards, dtype=np.float32)
    if states is None:
        states = np.zeros(len(rewards))
    return {"observations": np.asarray(states, dtype=np.float32).reshape(-1, 1),
            "actions": np.ones((len(rewards), 1), dtype=np.float32),
            "rewards": rewards,
            "returns": rewards[::-1].cumsum()[::-1].copy()}


class DensePreferenceTests(unittest.TestCase):
    def setUp(self):
        self.mean, self.std = np.zeros((1, 1)), np.ones((1, 1))
        self.data = [trajectory([90, 10]), trajectory([20, 60])]

    def test_remaining_return_reverses_whole_trajectory_ranking(self):
        pairs, _, stats = mine_dense_pairs(self.data, self.mean, self.std)
        np.testing.assert_array_equal(pairs, [[0, 0, 1, 0], [1, 1, 0, 1]])
        self.assertEqual(stats["total_return_order_disagreement_fraction"], 0.5)
        self.assertEqual(stats["rtg_gap_min"], 20)

    def test_past_rewards_do_not_change_later_labels(self):
        before, _, _ = mine_dense_pairs(self.data, self.mean, self.std)
        self.data[1] = trajectory([500, 60])
        after, _, _ = mine_dense_pairs(self.data, self.mean, self.std)
        np.testing.assert_array_equal(before[before[:, 1] == 1],
                                      after[after[:, 1] == 1])

    def test_nearest_matching_cutoff_and_prefix_independence(self):
        data = [trajectory([10, 10], [1000, 0]),
                trajectory([0, 0], [0, 0.4]),
                trajectory([-1, -1], [0, 0.1]),
                trajectory([5, 5], [0, 5])]
        pairs, distances, _ = mine_dense_pairs(data, self.mean, self.std, 0.2)
        np.testing.assert_array_equal(pairs, [[0, 1, 2, 1]])
        np.testing.assert_allclose(distances, [0.1])
        with self.assertRaisesRegex(ValueError, "No supported"):
            mine_dense_pairs(data, self.mean, self.std, 0.01)

    def test_ties_and_unequal_horizons_are_not_false_preferences(self):
        with self.assertRaisesRegex(ValueError, "No supported"):
            mine_dense_pairs([trajectory([1, 1])] * 2, self.mean, self.std)
        with self.assertRaisesRegex(ValueError, "equal"):
            mine_dense_pairs([trajectory([1]), trajectory([0, 0])],
                             self.mean, self.std)

    def test_dense_auxiliary_rtg_subtracts_only_observed_prefix(self):
        data = [trajectory([10, 20, 30, 40]), trajectory([0, 0, 0, 0])]
        ds = SimpleNamespace(dataset=data, seq_len=2, state_mean=self.mean,
                             state_std=self.std, reward_scale=0.001)
        pairs = np.array([[0, 2, 1, 2], [0, 0, 1, 0]])
        batch = dense_preference_batch(ds, pairs, [0, 1], 100)
        np.testing.assert_allclose(batch[2], [[0.09, 0.07], [0.1, 0]])
        np.testing.assert_array_equal(batch[3], [[1, 2], [0, 1]])
        np.testing.assert_array_equal(batch[4], [[1, 1], [1, 0]])
        data[0]["rewards"][2:] = -10000
        changed = dense_preference_batch(ds, pairs, [0, 1], 100)
        np.testing.assert_array_equal(batch[2], changed[2])

    def test_negative_budget_is_not_silently_clipped(self):
        ds = SimpleNamespace(dataset=self.data, seq_len=2, state_mean=self.mean,
                             state_std=self.std, reward_scale=1)
        batch = dense_preference_batch(ds, np.array([[0, 1, 1, 1]]), [0], 50)
        np.testing.assert_array_equal(batch[2], [[50, -40]])


class QueueGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "evaluations").mkdir()
        self.write("wandb_run.json", {"id": "expected"})
        self.write("config.json", {"update_steps": 100000, "eval_episodes": 2})

    def write(self, filename, value):
        (self.root / filename).write_text(json.dumps(value))

    def ready(self):
        return queue.dependency_ready(self.root, 100000, "expected")

    def test_training_or_failed_dependency_cannot_launch(self):
        self.write("status.json", {"state": "training", "completed_updates": 99000})
        self.assertFalse(self.ready())
        self.write("status.json", {"state": "failed", "completed_updates": 99000})
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.ready()

    def test_complete_requires_final_step_and_all_evaluation_episodes(self):
        self.write("status.json", {"state": "completed", "completed_updates": 100000})
        self.write("summary.json", {"completed_updates": 100000})
        self.write("evaluations/step100000.json", {"step": 100000, "returns": [1]})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.ready()
        self.write("evaluations/step100000.json", {"step": 100000, "returns": [1, 2]})
        self.assertTrue(self.ready())

    def test_wrong_dependency_run_is_rejected(self):
        self.write("wandb_run.json", {"id": "another-run"})
        with self.assertRaisesRegex(ValueError, "run ID"):
            self.ready()


if __name__ == "__main__":
    unittest.main()
