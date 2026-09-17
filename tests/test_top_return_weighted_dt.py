"""Run with PYTHONPATH=algorithms/offline python -m unittest discover -s tests
-p test_top_return_weighted_dt.py in the existing adt-delayed environment.
"""

import random
import unittest

import numpy as np
import torch
from dt import SequenceDataset
from top_return_weighted_dt import (
    select_top_trajectories,
    TopReturnDataset,
    weighted_action_loss,
)

class TopReturnTest(unittest.TestCase):
    def test_selection_counts_trajectories_and_handles_ties(self):
        mask = select_top_trajectories(np.arange(202), 0.05)
        self.assertEqual(mask.sum(), 11)
        self.assertEqual(np.flatnonzero(mask)[0], 191)
        np.testing.assert_array_equal(
            select_top_trajectories([3, 3, 1, 0], 0.25), [True, False, False, False]
        )

    def test_unit_weights_match_corl_loss_and_gradient_with_padding(self):
        prediction = torch.randn(3, 4, 2, requires_grad=True)
        actions = torch.randn_like(prediction)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]).double()
        loss, _, _, _ = weighted_action_loss(
            prediction, actions, mask, torch.tensor([1., 0., 0.]), 1.0
        )
        expected = ((prediction - actions).square() * mask.unsqueeze(-1)).mean()
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, prediction)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_doubled_relative_gradient_and_mean_one_weights(self):
        prediction = torch.ones(2, 1, 1, requires_grad=True)
        loss, ordinary, _, normalizer = weighted_action_loss(
            prediction, torch.zeros_like(prediction), torch.ones(2, 1),
            torch.tensor([1., 0.]), 2.0,
        )
        torch.testing.assert_close(normalizer, torch.tensor(1.5))
        torch.testing.assert_close(loss, ordinary)
        grad = torch.autograd.grad(loss, prediction)[0].flatten()
        torch.testing.assert_close(grad[0], 2 * grad[1])

    def test_sampler_is_identical_to_original_and_keeps_recorded_rtg(self):
        weighted = object.__new__(TopReturnDataset)
        weighted.dataset = [
            {
                "observations": np.ones((4, 2)) * index,
                "actions": np.ones((4, 1)) * index,
                "returns": np.ones(4) * (index + 1),
                "rewards": np.array([0., 0., 0., index + 1]),
            } for index in range(2)
        ]
        weighted.top_selected = np.array([False, True])
        weighted.sample_prob = np.array([0.5, 0.5])
        weighted.seq_len = 3
        weighted.state_mean = 0.0
        weighted.state_std = 1.0
        weighted.reward_scale = 0.001
        plain = object.__new__(SequenceDataset)
        plain.__dict__.update(weighted.__dict__)
        np.random.seed(7)
        random.seed(7)
        weighted_iterator = iter(weighted)
        weighted_samples = [next(weighted_iterator) for _ in range(10)]
        np.random.seed(7)
        random.seed(7)
        plain_iterator = iter(plain)
        for weighted_sample in weighted_samples:
            for actual, expected in zip(weighted_sample[:5], next(plain_iterator)):
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
