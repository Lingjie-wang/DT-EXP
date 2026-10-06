"""Scientific invariants: attribution, non-leakage, conservation and determinism."""

import itertools
import unittest

import numpy as np
import torch

from algorithms.offline.shapley_redistribution import (
    conserved_rewards,
    folds,
    permutation_shapley,
    random_masks,
    segment_inputs,
    SegmentReturnModel,
    standardize,
)

class ShapleyTests(unittest.TestCase):
    def test_exact_additive_negative_and_interaction(self):
        def game(masks):
            return 7 + masks @ np.array([2, -3, 0]) + 6 * masks[:, 0] * masks[:, 2]

        samples, empty, full = permutation_shapley(
            game, 3, list(itertools.permutations(range(3))))
        np.testing.assert_allclose(samples.mean(0), [5, -3, 3])
        np.testing.assert_allclose(samples.sum(1), full - empty)
        self.assertEqual(empty, 7)
        self.assertEqual(full, 12)
        with self.assertRaises(ValueError):
            permutation_shapley(game, 3, [[0, 0, 1]])

    def test_reward_conservation_with_negative_and_zero_returns(self):
        for total in [5000.123456, -638.4321, 0.0]:
            phi = np.linspace(-500, 400, 20)
            rewards, correction, rounding = conserved_rewards(phi, total)
            self.assertLess(abs(rewards.astype(np.float64).sum() - total), 1e-3)
            self.assertTrue(np.any(rewards < 0))
            expected = np.repeat(phi / 50, 50) + correction / 1000
            np.testing.assert_allclose(rewards[:-1], expected[:-1], atol=1e-6, rtol=1e-6)
            self.assertLess(abs(rounding), 0.01)
        uniform, _, _ = conserved_rewards(np.zeros(20), 1000)
        np.testing.assert_array_equal(uniform, np.ones(1000))

    def test_outer_and_inner_isolation(self):
        splits = list(folds(202))
        held_all = []
        for split in splits:
            train, valid, held = [set(split[k]) for k in ["train", "validation", "held"]]
            self.assertFalse(train & valid or train & held or valid & held)
            self.assertEqual(train | valid | held, set(range(202)))
            held_all.extend(held)
        self.assertEqual(sorted(held_all), list(range(202)))
        self.assertEqual(splits, list(folds(202)))

    def test_scaler_uses_only_inner_training(self):
        features = np.arange(6 * 100 * 2, dtype=np.float32).reshape(6, 100, 2)
        returns = np.arange(6, dtype=np.float64)
        _, first = standardize(features, returns, [0, 1, 2])
        features[3:] = 1e9
        returns[3:] = -1e9
        _, second = standardize(features, returns, [0, 1, 2])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_masked_segments_cannot_affect_value_or_gradient(self):
        torch.manual_seed(42)
        model = SegmentReturnModel(11, segments=4)
        x = torch.randn(3, 4, 11, requires_grad=True)
        mask = torch.tensor([[1., 0., 1., 0.]]).repeat(3, 1)
        result = model(x, mask)
        changed = x.detach().clone()
        changed[:, [1, 3]] = 1e6
        torch.testing.assert_close(result, model(changed, mask))
        result.sum().backward()
        self.assertEqual(float(x.grad[:, [1, 3]].abs().sum()), 0)
        torch.testing.assert_close(model(x, torch.zeros_like(mask)), torch.zeros(3))

    def test_masks_and_segment_order_are_reproducible(self):
        first = random_masks(np.random.default_rng(0), 200, 20)
        second = random_masks(np.random.default_rng(0), 200, 20)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all((first.sum(1) >= 1) & (first.sum(1) <= 19)))
        x = np.arange(2000, dtype=np.float32).reshape(1, 1000, 2)
        segmented = segment_inputs(x)
        self.assertEqual(segmented.shape, (1, 20, 101))
        np.testing.assert_array_equal(segmented[0, 1, :-1], x[0, 50:100].reshape(-1))


if __name__ == "__main__":
    unittest.main()
