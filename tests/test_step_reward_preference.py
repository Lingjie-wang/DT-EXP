"""Guard single-step labels and ordering behind the complete dense campaign."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithms/offline"))
from dense_action_preference import mine_dense_pairs
from step_reward_preference import mine_step_reward_pairs

spec = importlib.util.spec_from_file_location(
    "step_reward_queue", ROOT / "scripts/step_reward_preference/queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def trajectory(rewards, states=None):
    r = np.asarray(rewards, dtype=np.float32)
    if states is None:
        states = np.zeros(len(r))
    return {"observations": np.asarray(states, dtype=np.float32).reshape(-1, 1),
            "actions": np.zeros((len(r), 1), dtype=np.float32), "rewards": r,
            "returns": r[::-1].cumsum()[::-1].copy()}


class StepRewardTests(unittest.TestCase):
    def setUp(self):
        self.mean, self.std = np.zeros((1, 1)), np.ones((1, 1))

    def mine(self, data, cutoff=0.5):
        return mine_step_reward_pairs(data, self.mean, self.std, cutoff)

    def test_immediate_reward_can_disagree_with_remaining_return(self):
        data = [trajectory([10, -100]), trajectory([1, 100])]
        immediate, _, stats = self.mine(data)
        remaining, _, _ = mine_dense_pairs(data, self.mean, self.std)
        np.testing.assert_array_equal(immediate[0], [0, 0, 1, 0])
        np.testing.assert_array_equal(remaining[0], [1, 0, 0, 0])
        self.assertEqual(stats["remaining_return_order_disagreement_fraction"], 0.5)

    def test_other_timesteps_and_rtg_fields_do_not_change_current_labels(self):
        data = [trajectory([1, 10, 1]), trajectory([0, 1, 0])]
        before, _, _ = self.mine(data)
        data[0]["rewards"][[0, 2]] = -1000
        data[1]["rewards"][[0, 2]] = 1000
        data[0]["returns"][:] = np.nan
        data[1]["returns"][:] = np.nan
        after, _, _ = self.mine(data)
        np.testing.assert_array_equal(before[before[:, 1] == 1],
                                      after[after[:, 1] == 1])

    def test_nearest_state_cutoff_prefix_and_action_independence(self):
        data = [trajectory([10, 10], [1000, 0]),
                trajectory([0, 0], [0, 0.4]),
                trajectory([-1, -1], [0, 0.1]),
                trajectory([5, 5], [0, 5])]
        pairs, distances, _ = self.mine(data, 0.2)
        np.testing.assert_array_equal(pairs, [[0, 1, 2, 1]])
        np.testing.assert_allclose(distances, [0.1])
        data[2]["actions"][:] = 1
        np.testing.assert_array_equal(self.mine(data, 0.2)[0], pairs)
        with self.assertRaisesRegex(ValueError, "No supported"):
            self.mine(data, 0.01)

    def test_negative_rewards_are_ranked_relatively_and_ties_skipped(self):
        pairs, _, stats = self.mine([trajectory([-1, 0]), trajectory([-10, 0])])
        np.testing.assert_array_equal(pairs, [[0, 0, 1, 0]])
        self.assertEqual(stats["skipped_tied_timesteps"], 1)
        self.assertEqual(stats["reward_gap_min"], 9)

    def test_no_support_bad_rewards_and_unequal_horizons_fail(self):
        for data in ([trajectory([0])] * 2,
                     [trajectory([np.nan]), trajectory([0])],
                     [trajectory([1]), trajectory([0, 0])]):
            with self.assertRaises(ValueError):
                self.mine(data)


class AppendedQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("queue_manifest.json", {"git_commit": "expected"})

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def complete_trial(self, label, variant):
        self.write(label + "/status.json", {"state": "completed",
                                           "completed_updates": 100000})
        self.write(label + "/summary.json", {"completed_updates": 100000})
        self.write(label + "/config.json", {"variant": variant,
                   "reward_mode": "original", "update_steps": 100000,
                   "preference_weight": 0.05 if variant == "c" else 0,
                   "eval_episodes": 2, "train_seed": 0})
        self.write(label + "/provenance.json", {"git_commit": "expected"})
        self.write(label + "/evaluations/step100000.json",
                   {"step": 100000, "returns": [1, 2]})

    def test_waits_for_entire_campaign_and_stops_on_failure(self):
        for state in ("waiting", "smoke", "training"):
            self.write("queue_status.json", {"state": state})
            self.assertFalse(queue.dependency_ready(self.root, "expected"))
        self.write("queue_status.json", {"state": "failed"})
        with self.assertRaisesRegex(RuntimeError, "failed"):
            queue.dependency_ready(self.root, "expected")

    def test_completed_queue_requires_both_finished_trials(self):
        self.write("queue_status.json", {"state": "completed"})
        self.write("comparison.json", {})
        self.complete_trial("c-seed0", "c")
        with self.assertRaises(FileNotFoundError):
            queue.dependency_ready(self.root, "expected")
        self.complete_trial("dt-seed0", "dt")
        self.assertTrue(queue.dependency_ready(self.root, "expected"))
        self.write("dt-seed0/evaluations/step100000.json",
                   {"step": 100000, "returns": [1]})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            queue.dependency_ready(self.root, "expected")

    def test_different_dependency_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "revision"):
            queue.dependency_ready(self.root, "unexpected")

    def test_reused_baseline_must_match_training_and_shared_code(self):
        shared = {"algorithms/offline/" + f: "hash" for f in
                  ("dt.py", "state_only_preference.py", "dense_action_preference.py",
                   "top_return_weighted_dt.py")}
        provenance = {"initial_model_sha256": "init", "dataset_sha256": "data",
                      "packages": {"torch": "2.7.1"}, "attention_backend": "math",
                      "source_sha256": shared}
        for arm in ("control", "trial"):
            self.write(arm + "/config.json", {"train_seed": 0, "batch_size": 4096})
            self.write(arm + "/provenance.json", provenance)
        queue.verify_control(self.root / "control", self.root / "trial")
        self.write("trial/config.json", {"train_seed": 1, "batch_size": 4096})
        with self.assertRaisesRegex(ValueError, "train_seed"):
            queue.verify_control(self.root / "control", self.root / "trial")
        self.write("trial/config.json", {"train_seed": 0, "batch_size": 4096})
        shared["algorithms/offline/dt.py"] = "changed"
        self.write("trial/provenance.json", provenance)
        with self.assertRaisesRegex(ValueError, "Shared"):
            queue.verify_control(self.root / "control", self.root / "trial")


if __name__ == "__main__":
    unittest.main()
