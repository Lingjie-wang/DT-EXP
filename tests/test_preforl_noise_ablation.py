"""Run in the existing MuJoCo environment; load the unchanged server port."""

import os
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from preforl_noise_ablation import (
    load_source,
    losses,
    model_hash,
    PairedSampler,
    pairing_signature,
)

class PreforlNoiseAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        default = Path(__file__).resolve().parents[1] / (
            "third_party/PREFORL/train_mujoco_delayed.py"
        )
        cls.source = load_source(os.environ.get("PREFORL_SOURCE", str(default)))

    def setUp(self):
        rng = np.random.default_rng(44)
        self.episodes = [{
            "observations": rng.normal(size=(40, 3)).astype(np.float32),
            "actions": rng.uniform(-1, 1, size=(40, 2)).astype(np.float32),
        } for _ in range(8)]
        self.low = -np.ones(2, dtype=np.float32)
        self.high = np.ones(2, dtype=np.float32)

    def sample(self, sampler, noise):
        return sampler.sample(
            self.source, self.episodes, 4, 10, noise, 0.4, self.low, self.high
        )

    def test_pairing_survives_noise_switch_and_external_rng_consumption(self):
        zero, noisy = PairedSampler(0), PairedSampler(0)
        for _ in range(5):
            first = self.sample(zero, 0.0)
            np.random.randn(71)
            random.random()
            second = self.sample(noisy, 0.01)
            self.assertEqual(pairing_signature(first), pairing_signature(second))
            self.assertEqual(zero.state_hash(), noisy.state_hash())
            self.assertLessEqual(float(np.abs(first[1] - second[1]).max()), 0.010001)
            self.assertLessEqual(float(np.abs(first[3] - second[3]).max()), 0.010001)

    def test_sampler_preserves_surrounding_global_rng(self):
        np.random.seed(77)
        random.seed(88)
        expected_numpy = np.random.get_state()
        expected_python = random.getstate()
        self.sample(PairedSampler(0), 0.01)
        current_numpy = np.random.get_state()
        for actual, expected in zip(current_numpy, expected_numpy):
            np.testing.assert_equal(actual, expected)
        self.assertEqual(random.getstate(), expected_python)

    def test_identical_model_initialization_and_learnable_std(self):
        self.source.set_seed(0)
        first = self.source.GaussianPolicy(3, 2, 16, 2)
        self.source.set_seed(0)
        second = self.source.GaussianPolicy(3, 2, 16, 2)
        self.assertEqual(model_hash(first), model_hash(second))
        self.assertTrue(first.log_std.requires_grad)

    def test_loss_matches_original_port_expression(self):
        policy = self.source.GaussianPolicy(3, 2, 16, 2)
        arrays = self.sample(PairedSampler(0), 0.01)
        actual, _, _ = losses(self.source, policy, arrays, "cpu", 0.1, 0.5, 0.5)
        o1, a1, o2, a2, bo, ba, labels = [torch.from_numpy(v) for v in arrays]
        s1 = 0.1 * policy.log_prob(o1.reshape(-1, 3), a1.reshape(-1, 2))
        s2 = 0.1 * policy.log_prob(o2.reshape(-1, 3), a2.reshape(-1, 2))
        expected = self.source.biased_preference_loss(
            s1.reshape(4, 10).sum(1), s2.reshape(4, 10).sum(1), labels, 0.5
        ) - 0.5 * policy.log_prob(bo.reshape(-1, 3), ba.reshape(-1, 2)).mean()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        self.assertTrue(bool(torch.isfinite(policy.log_std.grad).all()))
        self.assertGreater(float(policy.log_std.grad.norm()), 0)

    def test_both_arms_keep_first_segment_bc_convention(self):
        for noise in (0., 0.01):
            batch = self.sample(PairedSampler(0), noise)
            np.testing.assert_array_equal(batch[4], batch[0])
            np.testing.assert_array_equal(batch[5], batch[1])


if __name__ == "__main__":
    unittest.main()
