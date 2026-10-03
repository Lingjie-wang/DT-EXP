"""Hand-computed Bellman targets and actual upstream sampler boundary tests."""

import ast
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from qt_terminal_correction import (
    episode_end_flags,
    patch_episode_ends,
    terminal_multistep_targets,
)

SOURCE = Path(os.environ.get("QT_SOURCE", "third_party/QT")).resolve()


class TerminalTargetTests(unittest.TestCase):
    def targets(self, terminal, reward=8.0, bootstrap=10.0):
        rewards = torch.tensor([[[0.], [0.], [1.], [2.], [reward]]])
        dones = torch.tensor([[[1], [1], [0], [0], [int(terminal)]]])
        mask = torch.tensor([[0, 0, 1, 1, 1]])
        original = rewards.clone()
        result = terminal_multistep_targets(
            rewards, dones, mask, torch.tensor([[bootstrap]]), 0.5
        )
        torch.testing.assert_close(rewards, original, rtol=0, atol=0)
        return result

    def test_terminal_rewards_discount_back_to_all_valid_positions(self):
        targets, mask = self.targets(True)
        torch.testing.assert_close(targets[0, 2:, 0], torch.tensor([4., 6., 8.]))
        self.assertEqual(mask.tolist(), [[False, False, True, True, True]])

    def test_terminal_target_does_not_depend_on_bootstrap_value(self):
        for reward in (0.0, 8.0, -8.0):
            first = self.targets(True, reward, 10.0)
            second = self.targets(True, reward, 999.0)
            torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)

    def test_nonterminal_window_bootstraps_and_excludes_last_reward(self):
        targets, mask = self.targets(False)
        torch.testing.assert_close(targets[0, 2:, 0], torch.tensor([4.5, 7., 10.]))
        self.assertEqual(mask.tolist(), [[False, False, True, True, False]])
        torch.testing.assert_close(targets, self.targets(False, 999.0)[0])

    def test_single_valid_terminal_token_is_supervised(self):
        rewards = torch.tensor([[[0.], [0.], [7.]]])
        targets, mask = terminal_multistep_targets(
            rewards, torch.ones(1, 3, 1).long(), torch.tensor([[0, 0, 1]]),
            torch.tensor([[999.]]), 0.99,
        )
        self.assertEqual(targets[mask].item(), 7.0)
        self.assertEqual(mask.sum().item(), 1)

    def test_boundary_flags_do_not_mutate_dataset_and_handle_zero_return(self):
        original = np.zeros((1, 5, 1), dtype=np.int64)
        self.assertEqual(episode_end_flags(original, 0, 5, 10).sum(), 0)
        self.assertEqual(episode_end_flags(original, 5, 5, 10).sum(), 1)
        self.assertEqual(original.sum(), 0)


@unittest.skipUnless((SOURCE / "experiment.py").exists(), "QT checkout required")
class UpstreamSamplerTests(unittest.TestCase):
    def sample(self, start):
        tree = patch_episode_ends(ast.parse((SOURCE / "experiment.py").read_text()))
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        function = next(n for n in functions if n.name == "get_batch")
        tree = ast.Module(body=[function], type_ignores=[])
        trajectory = {
            "observations": np.ones((10, 3)), "actions": np.zeros((10, 2)),
            "rewards": np.zeros(10), "terminals": np.zeros(10),
        }
        namespace = {
            "np": np, "torch": torch, "random": random,
            "episode_end_flags": episode_end_flags,
            "batch_size": 2, "K": 5, "num_trajectories": 1,
            "p_sample": [1.0], "sorted_inds": [0], "trajectories": [trajectory],
            "gym_name": "halfcheetah-medium-replay-v2", "state_dim": 3, "act_dim": 2,
            "max_ep_len": 1000, "variant": {"reward_tune": "no"},
            "state_mean": np.zeros(3), "state_std": np.ones(3),
            "scale": 1000.0, "device": "cpu",
            "discount_cumsum": lambda values, gamma: np.cumsum(values[::-1])[::-1],
        }
        exec(compile(tree, str(SOURCE / "experiment.py"), "exec"), namespace)
        with patch.object(random, "randint", return_value=start):
            batch = namespace["get_batch"](2)
        self.assertEqual(trajectory["terminals"].sum(), 0)
        return batch

    def test_actual_sampler_marks_full_terminal_window(self):
        batch = self.sample(5)
        self.assertEqual(batch[4][:, -1, 0].tolist(), [1, 1])
        self.assertEqual(batch[7].sum().item(), 10)

    def test_actual_sampler_marks_left_padded_terminal_window(self):
        batch = self.sample(9)
        self.assertEqual(batch[4][:, -1, 0].tolist(), [1, 1])
        self.assertEqual(batch[7].sum().item(), 2)

    def test_actual_sampler_keeps_nonterminal_windows_open(self):
        batch = self.sample(0)
        self.assertEqual(batch[4].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
