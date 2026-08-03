import unittest
from unittest import mock

import numpy as np
import torch

from algorithms.offline import dt

class DatasetEnv:
    def get_dataset(self):
        return {
            "observations": np.arange(10, dtype=np.float32).reshape(5, 2),
            "actions": np.arange(5, dtype=np.float32).reshape(5, 1),
            "rewards": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "terminals": np.array([False, True, False, False, True]),
            "timeouts": np.zeros(5, dtype=bool),
        }


class RecordingModel:
    episode_len = 2
    state_dim = 1
    action_dim = 1
    seq_len = 2

    def __init__(self):
        self.seen_returns = []

    def __call__(self, states, actions, returns, time_steps):
        self.seen_returns.append(returns.cpu().clone())
        return torch.zeros((*states.shape[:2], self.action_dim), device=states.device)


class RolloutEnv:
    def __init__(self):
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        rewards = (1.0, 2.0)
        reward = rewards[self.step_index]
        self.step_index += 1
        done = self.step_index == len(rewards)
        return np.zeros(1, dtype=np.float32), reward, done, {}


class RewardModeTest(unittest.TestCase):
    def test_delayed_rewards_move_episode_return_to_final_step(self):
        with mock.patch.object(dt.gym, "make", return_value=DatasetEnv()):
            trajectories, info = dt.load_d4rl_trajectories(
                "fake-env", reward_mode="delayed"
            )

        np.testing.assert_array_equal(trajectories[0]["rewards"], [0.0, 3.0])
        np.testing.assert_array_equal(trajectories[0]["returns"], [3.0, 3.0])
        np.testing.assert_array_equal(trajectories[1]["rewards"], [0.0, 0.0, 12.0])
        np.testing.assert_array_equal(trajectories[1]["returns"], [12.0, 12.0, 12.0])
        self.assertEqual(info["num_trajectories"], 2)
        self.assertEqual(info["num_transitions"], 5)
        self.assertEqual(info["nonzero_reward_fraction"], 2 / 5)
        self.assertEqual(info["original_nonzero_reward_fraction"], 1.0)

    def test_original_rewards_are_unchanged(self):
        rewards = np.array([1.0, -2.0, 4.0], dtype=np.float32)
        transformed = dt.transform_trajectory_rewards(rewards, "original")
        np.testing.assert_array_equal(transformed, rewards)
        self.assertIsNot(transformed, rewards)

    def test_delayed_rollout_keeps_rtg_constant(self):
        original_model = RecordingModel()
        delayed_model = RecordingModel()

        original_return, _ = dt.eval_rollout(
            original_model, RolloutEnv(), target_return=10.0, reward_mode="original"
        )
        delayed_return, _ = dt.eval_rollout(
            delayed_model, RolloutEnv(), target_return=10.0, reward_mode="delayed"
        )

        self.assertEqual(original_return, delayed_return)
        self.assertEqual(original_return, 3.0)
        np.testing.assert_array_equal(original_model.seen_returns[1], [[10.0, 9.0]])
        np.testing.assert_array_equal(delayed_model.seen_returns[1], [[10.0, 10.0]])


if __name__ == "__main__":
    unittest.main()
