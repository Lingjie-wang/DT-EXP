"""Check that delayed rewards survive timeouts without cross-episode targets."""

import unittest

import numpy as np

from algorithms.offline.cql_delayed_data import terminal_return_dataset

class TerminalReturnTests(unittest.TestCase):
    def raw(self):
        return dict(observations=np.arange(12).reshape(6, 2),
                    actions=np.arange(6).reshape(6, 1),
                    rewards=np.array([1, 2, 3, -4, 2, 2], dtype=np.float32),
                    terminals=np.array([False, True, False, False, False, False]),
                    timeouts=np.array([False, False, True, False, False, True]))

    def test_termination_timeout_zero_return_and_final_transition(self):
        raw = self.raw()
        data, audit = terminal_return_dataset(raw)
        np.testing.assert_array_equal(data["rewards"], [0, 3, 3, 0, 0, 0])
        np.testing.assert_array_equal(data["terminals"], [0, 1, 1, 0, 0, 1])
        np.testing.assert_array_equal(raw["rewards"], [1, 2, 3, -4, 2, 2])
        np.testing.assert_array_equal(data["observations"], raw["observations"])
        np.testing.assert_array_equal(data["actions"], raw["actions"])
        np.testing.assert_array_equal(
            data["next_observations"][[1, 2, 5]], np.zeros((3, 2)))
        np.testing.assert_array_equal(data["next_observations"][[0, 3, 4]],
                                      raw["observations"][[1, 4, 5]])
        self.assertEqual(audit["transitions"], 6)
        self.assertEqual(audit["episodes"], 3)
        self.assertEqual(audit["nonterminal_nonzero_rewards"], 0)
        # At a delayed terminal reward, Bellman targets must equal the reward
        # regardless of the value predicted for a reset/placeholder next state.
        target = data["rewards"] + .99 * (1 - data["terminals"]) * 123456
        np.testing.assert_array_equal(target[[1, 2, 5]], [3, 3, 0])

    def test_recorded_successors_and_incomplete_tail(self):
        raw = self.raw()
        raw["next_observations"] = raw["observations"] + 123
        data, _ = terminal_return_dataset(raw)
        np.testing.assert_array_equal(data["next_observations"][[0, 3, 4]],
                                      raw["next_observations"][[0, 3, 4]])
        raw["timeouts"][-1] = False
        with self.assertRaisesRegex(ValueError, "Incomplete final trajectory"):
            terminal_return_dataset(raw)


if __name__ == "__main__":
    unittest.main()
