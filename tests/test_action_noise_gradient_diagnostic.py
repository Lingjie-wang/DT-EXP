"""Objective and noise-statistics tests; no MuJoCo or dataset required."""

import unittest

import numpy as np
import torch
from diagnose_action_noise_gradients import (
    compare_gradients,
    gradient_vector,
    objectives,
    sample_negative,
    scores,
)

class ActionNoiseDiagnosticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.prediction = torch.randn(3, 4, 2, requires_grad=True)
        self.actions = torch.randn(3, 4, 2)
        self.mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]).double()

    def test_zero_noise_loss_and_gradient_match(self):
        first, second = objectives(
            self.prediction, self.actions, self.actions, self.mask
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        difference = gradient_vector(second - first, (self.prediction,))
        np.testing.assert_array_equal(difference, np.zeros_like(difference))

    def test_padding_does_not_change_scores_or_receive_gradient(self):
        score = scores(self.prediction, self.actions, self.mask, 0.1)
        gradient = torch.autograd.grad(score.sum(), self.prediction)[0]
        self.assertEqual(float(gradient[~self.mask.bool()].abs().max()), 0)
        modified = self.actions.clone()
        modified[~self.mask.bool()] = 1e6
        torch.testing.assert_close(
            score, scores(self.prediction, modified, self.mask, 0.1)
        )

    def test_gaussian_constant_is_retained(self):
        score = scores(self.prediction, self.prediction.detach(), self.mask, 0.1)
        expected = -np.log(0.1 * np.sqrt(2 * np.pi))
        torch.testing.assert_close(score, torch.full_like(score, expected))

    def test_noise_mask_bounds_repeatability_and_no_clipping(self):
        actions = torch.ones(100, 20, 6)
        mask = torch.ones(100, 20)
        first, audit = sample_negative(actions, mask, np.random.default_rng(17))
        second, _ = sample_negative(actions, mask, np.random.default_rng(17))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        changed = first != actions
        self.assertTrue(bool((changed == changed[..., :1]).all()))
        self.assertLessEqual(audit["maximum_absolute_perturbation"], 0.01)
        self.assertTrue(0.35 < audit["selected_valid_timestep_fraction"] < 0.45)
        self.assertGreater(float(first.max()), 1)

    def test_differential_gradient_matches_direct_gradients(self):
        negative, _ = sample_negative(self.actions, self.mask, np.random.default_rng(1))
        first, second = objectives(self.prediction, self.actions, negative, self.mask)
        base = gradient_vector(first, (self.prediction,), retain_graph=True)
        altered = gradient_vector(second, (self.prediction,), retain_graph=True)
        delta = gradient_vector(second - first, (self.prediction,))
        np.testing.assert_allclose(delta, altered - base, rtol=1e-3, atol=1e-6)

    def test_statistics_distinguish_mean_signal_from_noise(self):
        base = np.array([1., 0.])
        dt = np.array([2., 0.])
        deltas = np.array([[0., 0.01], [0., -0.01]])
        stats = compare_gradients(base, dt, deltas)
        self.assertEqual(stats["mean_relative_difference"], 0)
        self.assertEqual(stats["mean_gradient_cosine"], 1)
        self.assertGreater(stats["mc_mean_error_norm_relative"], 0)
        self.assertLess(stats["noise_corrected_squared_signal_relative"], 0)
        self.assertEqual(stats["split_half_difference_cosine"], -1)


if __name__ == "__main__":
    unittest.main()
