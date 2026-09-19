"""Mechanism checks against the actual pinned upstream trainer, no long training."""

import os
import random
import sys
import unittest
from pathlib import Path

import gym
import numpy as np
import torch
from qt_official_runner import isolated_evaluation, TerminalReward

SOURCE = Path(os.environ.get("QT_SOURCE", "third_party/QT")).resolve()


class FakeEnv(gym.Env):
    def reset(self):
        self.n = 0
        return np.zeros(3)

    def step(self, action):
        self.n += 1
        return np.zeros(3), float(self.n), self.n == 3, {}


class AdapterTests(unittest.TestCase):
    def test_terminal_wrapper_hides_intermediate_rewards(self):
        env = TerminalReward(FakeEnv())
        env.reset()
        self.assertEqual([env.step(None)[1] for _ in range(3)], [0.0, 0.0, 6.0])
        self.assertEqual(env.episode_return, 6.0)
        env.reset()
        self.assertEqual(env.step(None)[1], 0.0)

    def test_evaluation_preserves_training_rng(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        with isolated_evaluation(42):
            random.random()
            np.random.rand(33)
            torch.rand(31)
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


@unittest.skipUnless((SOURCE / "experiment.py").exists(), "QT checkout required")
class UpstreamBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SOURCE))
        from decision_transformer.models.ql_DT import Critic, DecisionTransformer
        from decision_transformer.training.ql_trainer import Trainer

        cls.critic_class = Critic
        cls.model_class = DecisionTransformer
        cls.trainer_class = Trainer
        torch.set_num_threads(2)

    def model(self):
        return self.model_class(
            state_dim=3, act_dim=2, hidden_size=16, max_length=5, max_ep_len=1000,
            n_layer=1, n_head=1, n_inner=64, activation_function="relu",
            n_positions=1024, resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0,
            scale=1000.0,
        )

    def one_update(self, reward_position):
        torch.manual_seed(7)
        np.random.seed(7)
        actor, critic = self.model(), self.critic_class(3, 2, hidden_dim=16)
        states = torch.randn(4, 5, 3)
        actions = torch.tanh(torch.randn(4, 5, 2))
        rewards = torch.zeros(4, 5, 1)
        if reward_position is not None:
            rewards[:, reward_position] = 1000.0
        batch = (
            states, actions, rewards, actions.clone(), torch.zeros(4, 5, 1).long(),
            torch.ones(4, 6, 1), torch.arange(5).repeat(4, 1), torch.ones(4, 5),
        )
        trainer = self.trainer_class(
            model=actor, critic=critic, batch_size=4, tau=0.005, discount=0.99,
            get_batch=lambda _: batch, loss_fn=None, eta=5.0, eta2=1.0,
            grad_norm=15.0, scale=1000.0, k_rewards=True, use_discount=True,
        )
        metrics = {key: [] for key in (
            "bc_loss", "ql_loss", "actor_loss", "critic_loss", "target_q_mean"
        )}
        result = trainer.train_step(loss_metric=metrics)
        weights = torch.cat([p.detach().flatten() for p in critic.parameters()])
        actor_weights = torch.cat([p.detach().flatten() for p in actor.parameters()])
        return result, weights, actor_weights, rewards

    def test_upstream_terminal_reward_is_ignored_not_silently_fixed(self):
        zero = self.one_update(None)
        terminal = self.one_update(-1)
        self.assertEqual(zero[0], terminal[0])
        torch.testing.assert_close(zero[1], terminal[1], rtol=0, atol=0)
        torch.testing.assert_close(zero[2], terminal[2], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(terminal[3]).item(), 0)

    def test_reward_at_an_earlier_position_does_affect_critic(self):
        zero = self.one_update(None)
        earlier = self.one_update(-2)
        self.assertNotEqual(zero[0]["critic_loss"], earlier[0]["critic_loss"])
        self.assertFalse(torch.equal(zero[1], earlier[1]))

    def test_direct_fixed_rtg_forward_matches_upstream_first_candidate(self):
        torch.manual_seed(3)
        model = self.model().eval()
        model.infer_no_q = True
        critic = self.critic_class(3, 2, hidden_dim=16).eval()
        states, actions = torch.randn(3, 3), torch.randn(3, 2)
        rewards, rtg = torch.zeros(1, 3), torch.ones(1, 3) * 12
        times = torch.arange(3).reshape(1, 3)
        with torch.no_grad():
            original = model.get_action(critic, states, actions, rewards, rtg, times)
            s = torch.cat([torch.zeros(2, 3), states]).unsqueeze(0)
            a = torch.cat([torch.zeros(2, 2), actions]).unsqueeze(0)
            r = torch.zeros(1, 5, 1)
            g = torch.tensor([0., 0., 12., 12., 12.]).reshape(1, 5, 1)
            t = torch.tensor([[0, 0, 0, 1, 2]])
            mask = torch.tensor([[0, 0, 1, 1, 1]])
            direct = model(s, a, r, None, g, t, attention_mask=mask)[1][:, -1]
        torch.testing.assert_close(original, direct[0], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
