"""Guard C's mining semantics and detached-negative, whole-batch gradients."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithms/offline"))
from state_only_preference import (
    mine_pairs,
    preference_batch,
    single_sided_loss,
    validate_resume_config,
)

def trajectory(states, actions, total):
    return {"observations": np.array(states, dtype=np.float32).reshape(-1, 1),
            "actions": np.array(actions, dtype=np.float32).reshape(-1, 1),
            "returns": np.full(len(states), total, dtype=np.float32)}


class StateOnlyPreferenceTests(unittest.TestCase):
    def test_resume_allows_cadence_but_rejects_method_changes(self):
        saved = {"eval_every": 20000, "preference_weight": 0.05,
                 "batch_size": 4096, "update_steps": 100000, "betas": [0.9, 0.999]}
        current = {**saved, "eval_every": 10000, "betas": (0.9, 0.999)}
        validate_resume_config(current, saved, 5000)
        for key, value in (("preference_weight", 0.01), ("batch_size", 1024)):
            with self.assertRaisesRegex(ValueError, key):
                validate_resume_config({**current, key: value}, saved, 5000)

    def test_resume_rejects_completed_or_invalid_steps(self):
        config = {"update_steps": 100000}
        for completed in (-1, 100000, 100001):
            with self.assertRaisesRegex(ValueError, "Resume step"):
                validate_resume_config(config, config, completed)

    def setUp(self):
        self.data = [trajectory([0, 0.1, 0.2], [1, 1, 1], 10),
                     trajectory([0.1, 0.2, 0.3], [-1, -1, -1], 0)]
        self.mean, self.std = np.zeros((1, 1)), np.ones((1, 1))

    def test_same_time_nearest_and_cutoff(self):
        pairs, distances, stats = mine_pairs(self.data, self.mean, self.std, 0.11)
        np.testing.assert_array_equal(pairs, [[0, 0, 1, 0], [0, 1, 1, 1], [0, 2, 1, 2]])
        np.testing.assert_allclose(distances, 0.1, atol=1e-7)
        self.assertEqual(stats["coverage"], 1)
        with self.assertRaisesRegex(ValueError, "No supported"):
            mine_pairs(self.data, self.mean, self.std, 0.01)

    def test_prefixes_and_action_differences_do_not_filter_current_pair(self):
        self.data[1]["observations"][0] = 1000
        self.data[1]["actions"][:] = self.data[0]["actions"]
        pairs, _, _ = mine_pairs(self.data, self.mean, self.std, 0.11)
        self.assertIn([0, 2, 1, 2], pairs.tolist())

    def test_exact_nearest_low_trajectory(self):
        data = [trajectory([0], [0], 10), trajectory([0.4], [1], 0),
                trajectory([0.1], [-1], -1), trajectory([5], [0], 5)]
        pairs, _, _ = mine_pairs(data, self.mean, self.std)
        np.testing.assert_array_equal(pairs, [[0, 0, 2, 0]])

    def test_degenerate_returns_are_rejected(self):
        self.data[1]["returns"][:] = 10
        with self.assertRaisesRegex(ValueError, "disjoint"):
            mine_pairs(self.data, self.mean, self.std)

    def test_positive_context_right_padding_and_fixed_rtg(self):
        ds = SimpleNamespace(dataset=self.data, seq_len=3, state_mean=self.mean,
                             state_std=self.std, reward_scale=0.001)
        batch = preference_batch(ds, np.array([[0, 1, 1, 1]]), [0], 12000)
        states, actions, returns, times, mask, index, negative = batch
        np.testing.assert_allclose(states[0, :, 0], [0, 0.1, 0])
        np.testing.assert_array_equal(actions[0, :, 0], [1, 1, 0])
        np.testing.assert_array_equal(returns[0], [12, 12, 0])
        np.testing.assert_array_equal(times[0], [0, 1, 2])
        np.testing.assert_array_equal(mask[0], [1, 1, 0])
        self.assertEqual(index.item(), 1)
        self.assertEqual(negative.item(), -1)

    def test_detached_negative_and_whole_batch_denominator(self):
        prediction = torch.tensor([[0.0], [1.0]], requires_grad=True)
        positive = torch.tensor([[1.0], [1.0]], requires_grad=True)
        negative = torch.tensor([[0.0], [-1.0]], requires_grad=True)
        loss, _, _, active = single_sided_loss(prediction, positive, negative)
        self.assertAlmostEqual(loss.item(), 1.05 / 2, places=6)
        self.assertEqual(active.tolist(), [True, False])
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[-1.0], [0.0]]))
        self.assertIsNone(positive.grad)
        self.assertIsNone(negative.grad)


if __name__ == "__main__":
    unittest.main()
