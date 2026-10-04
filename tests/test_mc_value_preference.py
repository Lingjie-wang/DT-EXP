"""MC supervision, finite-horizon advantages, RNG isolation and queue ordering."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithms/offline"))
from mc_value_preference import (
    fit_frozen_value,
    mine_value_pairs,
    one_step_scores,
    StateTimeValue,
    trajectory_split,
    value_data,
)

spec = importlib.util.spec_from_file_location(
    "mc_value_queue", ROOT / "scripts/mc_value_preference/queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def trajectory(rewards, states=None):
    r = np.asarray(rewards, dtype=np.float32)
    if states is None:
        states = np.zeros(len(r))
    return {"observations": np.asarray(states, dtype=np.float32).reshape(-1, 1),
            "actions": np.zeros((len(r), 1), dtype=np.float32), "rewards": r,
            "returns": np.full(len(r), -9999, dtype=np.float32)}


class MCValueTests(unittest.TestCase):
    def setUp(self):
        self.mean, self.std = np.zeros((1, 1)), np.ones((1, 1))
        self.data = [trajectory([1, 2, 3]), trajectory([-1, 4, 2]),
                     trajectory([2, 1, 0]), trajectory([0, 3, 1]),
                     trajectory([1, 1, 1])]

    def test_mc_targets_use_observed_rewards_and_current_state_time(self):
        inputs, targets, _ = value_data(self.data, self.mean, self.std)
        np.testing.assert_array_equal(targets[0], [6, 5, 3])
        np.testing.assert_array_equal(targets[1], [5, 6, 2])
        np.testing.assert_allclose(inputs[0, :, -1], [0, 1 / 3, 2 / 3])
        self.assertEqual(inputs.shape, (5, 3, 2))

    def test_held_out_split_keeps_complete_trajectories_separate(self):
        train, val = trajectory_split(202, 1729)
        self.assertEqual((len(train), len(val)), (161, 41))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(202)))
        np.testing.assert_array_equal(val, trajectory_split(202, 1729)[1])

    def test_terminal_value_zero_and_no_cross_episode_bootstrap(self):
        scores = one_step_scores([[1, 2], [3, 4]], [[10, 20], [100, 200]])
        np.testing.assert_array_equal(scores, [[11, -18], [103, -196]])

    def test_true_sample_returns_telescope_to_zero_without_off_by_one(self):
        _, returns, rewards = value_data(self.data, self.mean, self.std)
        np.testing.assert_array_equal(one_step_scores(rewards, returns),
                                      np.zeros_like(rewards))

    def test_scores_determine_labels_not_reward_or_recorded_rtg(self):
        data = [trajectory([100, 100], [0, 0]), trajectory([-100, -100], [0.1, 0.1])]
        pairs, distances, _ = mine_value_pairs(
            data, [[-1, 3], [2, -2]], self.mean, self.std, 0.2)
        np.testing.assert_array_equal(pairs, [[1, 0, 0, 0], [0, 1, 1, 1]])
        np.testing.assert_allclose(distances, [0.1, 0.1])
        with self.assertRaisesRegex(ValueError, "No supported"):
            mine_value_pairs(data, [[-1, 3], [2, -2]], self.mean, self.std, 0.01)

    def test_value_fit_isolated_rng_and_saved_predictions_match_scores(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        scores, values, report, history, checkpoint = fit_frozen_value(
            self.data, self.mean, self.std, epochs=3, seed=12)
        torch.testing.assert_close(torch.get_rng_state(), before)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        self.assertEqual(np.random.get_state()[2:], numpy_before[2:])
        self.assertEqual(report["best_epoch"],
                         min(history, key=lambda r: r["validation_mse_scaled"])["epoch"])
        model = StateTimeValue(1)
        model.load_state_dict(checkpoint["model_state"])
        model.eval().requires_grad_(False)
        inputs, _, rewards = value_data(self.data, self.mean, self.std)
        predicted = model(torch.from_numpy(inputs)).numpy() / 0.001
        np.testing.assert_allclose(predicted, values, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(
            scores, one_step_scores(rewards, predicted), atol=1e-4)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_bad_values_data_and_unsupported_pairs_fail(self):
        with self.assertRaises(ValueError):
            one_step_scores([[1]], [[np.nan]])
        with self.assertRaises(ValueError):
            value_data([trajectory([1]), trajectory([0, 0])],
                       self.mean, self.std)
        with self.assertRaisesRegex(ValueError, "No supported"):
            mine_value_pairs(self.data, np.zeros((5, 3)), self.mean, self.std)


class MCQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("queue_manifest.json", {"git_commit": "step-commit",
                   "dependency": "/old-dense", "dependency_commit": "dense-commit"})

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def complete_parent(self):
        self.write("queue_status.json", {"state": "completed"})
        self.write("comparison.json", {})
        self.write("c-seed0/status.json",
                   {"state": "completed", "completed_updates": 100000})
        self.write("c-seed0/summary.json", {"completed_updates": 100000})
        self.write("c-seed0/config.json", {"update_steps": 100000, "variant": "c",
                   "reward_mode": "original", "preference_label": "step_reward",
                   "preference_weight": 0.05, "eval_episodes": 2})
        self.write("c-seed0/provenance.json", {"git_commit": "step-commit"})
        self.write("c-seed0/evaluations/step100000.json",
                   {"step": 100000, "returns": [1, 2]})

    def test_waits_for_step_reward_and_stops_on_failure(self):
        for state in ("waiting", "smoke", "training"):
            self.write("queue_status.json", {"state": state})
            self.assertFalse(queue.dependency_ready(self.root, "step-commit"))
        self.write("queue_status.json", {"state": "failed"})
        with self.assertRaisesRegex(RuntimeError, "failed"):
            queue.dependency_ready(self.root, "step-commit")

    def test_complete_checks_earlier_dense_campaign_too(self):
        self.complete_parent()
        with mock.patch.object(queue, "dense_campaign_ready",
                               return_value=True) as earlier:
            self.assertTrue(queue.dependency_ready(self.root, "step-commit"))
            earlier.assert_called_once_with(Path("/old-dense"), "dense-commit")
        with mock.patch.object(queue, "dense_campaign_ready", return_value=False):
            with self.assertRaisesRegex(ValueError, "Earlier"):
                queue.dependency_ready(self.root, "step-commit")

    def test_wrong_commit_or_missing_final_episodes_cannot_launch(self):
        with self.assertRaisesRegex(ValueError, "revision"):
            queue.dependency_ready(self.root, "wrong")
        self.complete_parent()
        self.write("c-seed0/evaluations/step100000.json",
                   {"step": 100000, "returns": [1]})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            queue.dependency_ready(self.root, "step-commit")


if __name__ == "__main__":
    unittest.main()
