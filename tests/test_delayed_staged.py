"""Reject dense leakage and preserve matched delayed-stage comparisons."""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'algorithms/offline'))
from delayed_staged import (
    audit_batch_rtg,
    audit_delayed_dataset,
    stage_objective,
    TerminalReward,
    validate_delayed_parent,
)
from dt import eval_rollout, transform_trajectory_rewards
from state_only_preference import single_sided_loss

class ThreeStepEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        self.t = 0
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        self.t += 1
        return np.array([self.t], dtype=np.float32), float(self.t), self.t == 3, {}


class RecordingPolicy:
    episode_len, seq_len, state_dim, action_dim = 3, 2, 1, 1

    def __init__(self):
        self.rtgs = []

    def __call__(self, states, actions, returns, times):
        self.rtgs.append(returns[0, -1].item())
        return torch.zeros_like(actions)


class DelayedStudyTests(unittest.TestCase):
    def test_environment_exposes_only_terminal_return_and_resets_accumulator(self):
        env = TerminalReward(ThreeStepEnv())
        for _ in range(2):
            env.reset()
            self.assertEqual([env.step(np.zeros(1))[1] for _ in range(3)], [0, 0, 6])

    def test_policy_receives_constant_rtg_without_dense_feedback(self):
        model = RecordingPolicy()
        total, length = eval_rollout(
            model, TerminalReward(ThreeStepEnv()), 12.0, 'cpu', 'delayed')
        self.assertEqual(model.rtgs, [12, 12, 12])
        self.assertEqual((total, length), (6, 3))

    def dataset(self):
        rewards = transform_trajectory_rewards(np.array([1., 2., 3.]), 'delayed')
        returns = np.full_like(rewards, rewards[-1])
        return SimpleNamespace(reward_mode='delayed', dataset=[{
            'rewards': rewards, 'returns': returns}])

    def test_training_rewards_and_rtgs_are_checked_not_just_config(self):
        d = self.dataset()
        audit = audit_delayed_dataset(d)
        self.assertTrue(audit['all_nonterminal_rewards_zero'])
        self.assertTrue(audit['all_training_rtgs_constant'])
        d.reward_mode = 'original'
        with self.assertRaisesRegex(ValueError, 'mandatory'):
            audit_delayed_dataset(d)
        d = self.dataset()
        d.dataset[0]['rewards'][0] = 1
        with self.assertRaisesRegex(ValueError, 'Nonterminal'):
            audit_delayed_dataset(d)
        d = self.dataset()
        d.dataset[0]['returns'][1] = 5
        with self.assertRaisesRegex(ValueError, 'RTG'):
            audit_delayed_dataset(d)

    def test_batch_audit_rejects_prefix_subtraction_and_ignores_padding(self):
        r = torch.tensor([[6., 6., 0.], [4., 4., 4.]])
        m = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
        a = torch.tensor([[12., 12., 0.], [12., 12., 12.]])
        audit_batch_rtg(r, m, a, m, 12)
        a[0, 1] = 11
        with self.assertRaisesRegex(ValueError, 'Auxiliary'):
            audit_batch_rtg(r, m, a, m, 12)
        a[0, 1] = 12
        r[0, 1] = 5
        with self.assertRaisesRegex(ValueError, 'Ordinary'):
            audit_batch_rtg(r, m, a, m, 12)

    def test_rejects_dense_or_already_preference_trained_parent(self):
        parent = {'update_steps': 100000, 'variant': 'parent_dt',
                  'reward_mode': 'delayed', 'preference_weight': 0.,
                  'resume_checkpoint': '', 'batch_size': 4096, 'learning_rate': .0008}
        child = {**parent, 'update_steps': 5000, 'variant': 'c',
                 'preference_weight': .05, 'learning_rate': .0001}
        validate_delayed_parent(child, parent, 100000)
        for key, value in [('reward_mode', 'original'), ('variant', 'c'),
                           ('preference_weight', .05), ('update_steps', 50000)]:
            bad = copy.deepcopy(parent)
            bad[key] = value
            with self.assertRaisesRegex(ValueError, 'ordinary delayed'):
                validate_delayed_parent(child, bad, 100000)
        with self.assertRaisesRegex(ValueError, 'reward_mode'):
            validate_delayed_parent({**child, 'reward_mode': 'original'}, parent, 100000)
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            validate_delayed_parent({**child, 'batch_size': 64}, parent, 100000)

    def test_c_only_excludes_dt_gradient_while_dt_control_excludes_preference(self):
        grads = {}
        for arm in ('dt', 'c', 'c_only'):
            x = torch.tensor([[1.]], requires_grad=True)
            dt = (x - 3).square().mean()
            pref, _, _, _ = single_sided_loss(x, torch.zeros_like(x), x.detach(), .05)
            if arm == 'c_only':
                dt = dt.detach()
            stage_objective(arm, dt, pref).backward()
            grads[arm] = x.grad.item()
        self.assertAlmostEqual(grads['dt'], -4.)
        self.assertAlmostEqual(grads['c'], -3.9, places=5)
        self.assertAlmostEqual(grads['c_only'], .1, places=5)
        with self.assertRaisesRegex(ValueError, 'autograd'):
            stage_objective('c_only', torch.tensor(1., requires_grad=True),
                            torch.tensor(1., requires_grad=True))


if __name__ == '__main__':
    unittest.main()
