"""CPU mechanism tests, followed by real MuJoCo three-arm GPU smoke checks."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from adaptive_hard_fork_dt import (
    combined_loss,
    evaluating,
    frozen_copy,
    PairPool,
    preference_loss,
    ramp,
    refresh_due,
    sampling_probabilities,
)

class AdaptiveHardForkTests(unittest.TestCase):
    def test_ramp_and_refresh_use_completed_updates(self):
        self.assertEqual([ramp(x, 10, 20) for x in (0, 10, 15, 20, 30)],
                         [0, 0, 0.5, 1, 1])
        self.assertEqual([x for x in range(26) if refresh_due(x, 10, 5)],
                         [10, 15, 20, 25])

    def test_sampler_coverage_fallback_and_cap(self):
        confidence = np.array([0.2, 0.4, 0.8])
        base = confidence / confidence.sum()
        np.testing.assert_allclose(sampling_probabilities(
            confidence, np.zeros(3), 0.05, 0.2, 5), base)
        probabilities = sampling_probabilities(confidence, [0, 0.1, 100], 0.05, 0.2, 5)
        self.assertAlmostEqual(probabilities.sum(), 1)
        self.assertTrue(np.all(probabilities >= 0.2 * base))
        np.testing.assert_allclose(probabilities, sampling_probabilities(
            confidence, [0, 0.1, 0.25], 0.05, 0.2, 5))

    def test_negative_distance_has_no_repulsion_gradient(self):
        prediction = torch.tensor([[0.5]], requires_grad=True)
        loss, metrics = preference_loss(prediction, torch.zeros(1, 1),
                                        torch.ones(1, 1), 0.05, 1)
        self.assertEqual(metrics["preference_active_count"], 1)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.ones(1, 1))

    def test_active_normalization_and_fallback(self):
        prediction = torch.tensor([[0.5], [0.0]])
        positive, negative = torch.zeros(2, 1), torch.ones(2, 1)
        active_loss, _ = preference_loss(prediction, positive, negative, 0.05, 1)
        fallback, _ = preference_loss(prediction, positive, negative, 0.05, 2)
        self.assertAlmostEqual(active_loss.item(), 0.05, places=6)
        self.assertAlmostEqual(fallback.item(), 0.025, places=6)

    def test_reference_frozen_deterministic_and_student_differentiable(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(0.8))
        reference = frozen_copy(model)
        inputs = torch.ones(8, 3)
        self.assertTrue(model.training)
        with evaluating(model):
            self.assertFalse(model.training)
            torch.testing.assert_close(model(inputs), reference(inputs), rtol=0, atol=0)
            with torch.no_grad():
                model[0].weight.add_(0.01)
            loss = (model(inputs) - reference(inputs)).square().mean()
            loss.backward()
        self.assertTrue(model.training)
        self.assertGreater(model[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad
                            for p in reference.parameters()))

    def test_control_is_exact_dt_objective_and_ref_only_has_no_pref_gradient(self):
        prediction = torch.tensor([0.3], requires_grad=True)
        config = SimpleNamespace(reference_weight=0.1, preference_weight=0.05)
        dt_loss = prediction.square().mean()
        pref = (prediction - 1).square().mean()
        ref = (prediction + 1).square().mean()
        self.assertIs(combined_loss(dt_loss, pref, ref, "dt", 1, config), dt_loss)
        actual = combined_loss(dt_loss, pref, ref, "recent_ref", 1, config)
        expected = dt_loss + 0.1 * ref
        torch.testing.assert_close(torch.autograd.grad(actual, prediction,
                                                      retain_graph=True)[0],
                                   torch.autograd.grad(expected, prediction)[0])

    def test_auxiliary_fork_does_not_advance_dt_rng(self):
        torch.manual_seed(12)
        before = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.rand(100)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_pool_never_requires_old_checkpoint_scores(self):
        trajectory = {
            "observations": np.zeros((6, 2), dtype=np.float32),
            "actions": np.ones((6, 1), dtype=np.float32),
            "returns": np.ones(6, dtype=np.float32) * 3000,
        }
        dataset = SimpleNamespace(dataset=[trajectory, trajectory], seq_len=4,
                                  state_mean=np.zeros((1, 2), dtype=np.float32),
                                  state_std=np.ones((1, 2), dtype=np.float32),
                                  reward_scale=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            np.savez(path, pairs=np.array([[0, 4, 1, 4], [0, 5, 1, 5]]),
                     valid_branch=np.array([1, 0]), pair_confidence=np.array([0.4, 0.3]))
            pool = PairPool(dataset, path, 12000)
        self.assertEqual(len(pool.pairs), 1)
        self.assertTrue(torch.all(pool.tensors[2] == 12))
        np.testing.assert_array_equal(pool.base, [1.0])


if __name__ == "__main__":
    unittest.main()
