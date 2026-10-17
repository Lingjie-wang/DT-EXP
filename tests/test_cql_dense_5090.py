"""Validate original reward restoration, matched controls and queue priority."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import yaml

from algorithms.offline.cql_delayed_data import terminal_return_dataset
from scripts.cql_delayed.common import digest, read, verify, write
from scripts.cql_dense_5090.prepare import (
    CAMPAIGN,
    original_rewards,
    PREDECESSOR,
    prepare,
    REWARD_MODE,
)
from scripts.cql_dense_5090.queue import admitted

def raw_data():
    return dict(observations=np.arange(12, dtype=np.float32).reshape(6, 2),
                actions=np.ones((6, 1), dtype=np.float32),
                rewards=np.array([2, -1, 3, 0.25, -0.5, 1.25], dtype=np.float32),
                terminals=np.array([0, 0, 1, 0, 0, 0], dtype=bool),
                timeouts=np.array([0, 0, 0, 0, 0, 1], dtype=bool))


class RewardTests(unittest.TestCase):
    def test_exact_step_rewards_and_unchanged_boundary_transitions(self):
        raw = raw_data()
        baseline, _ = terminal_return_dataset(raw)
        original = {k: v.copy() for k, v in baseline.items()}
        data, audit = original_rewards(raw, baseline)
        np.testing.assert_array_equal(data["rewards"], raw["rewards"])
        for key in data:
            np.testing.assert_array_equal(baseline[key], original[key])
            if key != "rewards":
                np.testing.assert_array_equal(data[key], baseline[key])
        self.assertEqual(audit["nonterminal_nonzero_rewards"], 4)
        self.assertEqual(audit["bootstrap_stops"], 2)
        self.assertTrue(audit["matched_nonreward_fields"])

    def test_reject_mismatched_transition_or_terminal_total(self):
        for key in ("actions", "observations", "rewards", "terminals"):
            with self.subTest(key=key):
                raw = raw_data()
                baseline, _ = terminal_return_dataset(raw)
                baseline[key].flat[0] += 1
                with self.assertRaises(AssertionError):
                    original_rewards(raw, baseline)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.previous = (self.project
                         / "results/cql-corl-delayed-hc-seed1-5090-20261007")
        source = self.previous / "source"
        source.mkdir(parents=True)
        (source / "cql.py").write_text("# frozen training algorithm\n")
        self.paths = {}
        runs = {}
        for arm in ("medium", "medium_replay"):
            path = self.project / f"{arm}.hdf5"
            self.paths[arm] = path
            with h5py.File(path, "w") as f:
                for key, value in raw_data().items():
                    f.create_dataset(key, data=value)
            data, audit = terminal_return_dataset(raw_data())
            work = self.previous / arm
            work.mkdir()
            np.savez(work / "dataset.npz", **data)
            write(work / "audit.json", dict(audit, original_sha256=digest(path)))
            config = dict(seed=1, max_timesteps=1000000, normalize_reward=False,
                          eval_freq=5000, n_episodes=10, name="pilot", group="pilot",
                          batch_size=256, discount=0.99)
            (source / f"{arm}.yaml").write_text(yaml.safe_dump(config))
            runs[arm] = dict(seed=1, updates=1000000, config=f"{arm}.yaml",
                             env=f"halfcheetah-{arm}-v2", wandb_id=f"pilot-{arm}",
                             data_sha256={n: digest(work / n)
                                          for n in ("dataset.npz", "audit.json")})
        write(self.previous / "protocol.json", dict(
            runs=runs, source_sha256={p.name: digest(p) for p in source.iterdir()},
            retained=dict(batch_size=256), wandb_entity="test", wandb_project="test"))
        predecessor = self.project / "results" / PREDECESSOR
        predecessor.mkdir()
        write(predecessor / "plan.json", {"jobs": [{"id": str(i)} for i in range(8)]})
        self.root = self.project / "results" / CAMPAIGN

    def test_all_six_independent_configs_with_original_rewards(self):
        historical = {str(p): digest(p) for p in self.previous.rglob("*")
                      if p.is_file()}
        prepare(self.project, self.root, self.paths, "published-test-commit")
        plan = read(self.root / "plan.json")
        self.assertEqual(len(plan["jobs"]), 6)
        self.assertEqual({j["seed"] for j in plan["jobs"]}, {1, 11, 12})
        self.assertEqual(len({j["wandb_id"] for j in plan["jobs"]}), 6)
        for job in plan["jobs"]:
            campaign = self.root / job["directory"]
            p = verify(campaign)
            verify(campaign / "validation")
            self.assertEqual(p["reward_mode"], REWARD_MODE)
            self.assertEqual(p["source_sha256"]["cql.py"],
                             digest(self.previous / "source/cql.py"))
            spec = p["runs"][job["arm"]]
            config = yaml.safe_load((campaign / "source" / spec["config"]).read_text())
            self.assertEqual(config["seed"], job["seed"])
            self.assertEqual(config["max_timesteps"], 300000)
            self.assertFalse(config["normalize_reward"])
            with np.load(campaign / job["arm"] / "dataset.npz") as data:
                np.testing.assert_array_equal(data["rewards"], raw_data()["rewards"])
        self.assertEqual(historical, {str(p): digest(p)
                                    for p in self.previous.rglob("*") if p.is_file()})
        with self.assertRaisesRegex(ValueError, "already reserved"):
            prepare(self.project, self.root, self.paths, "test")

    def test_wrong_original_file_rejected(self):
        self.paths["medium"].write_bytes(b"wrong dataset")
        with self.assertRaisesRegex(ValueError, "does not match"):
            prepare(self.project, self.root, self.paths, "test")
        self.assertFalse((self.root / "plan.json").exists())


class DependencyTests(unittest.TestCase):
    def status(self, **changes):
        return dict(dict(stage="running", updated_unix=100,
                         jobs={"a": {"state": "completed"},
                               "b": {"state": "running", "pid": 123}}), **changes)

    def test_waits_for_all_predecessor_jobs_and_cuda_initialization(self):
        self.assertTrue(admitted(self.status(), ["a", "b"], [123], 110))
        self.assertFalse(admitted(self.status(), ["a", "b"], [], 110))
        status = self.status()
        status["jobs"]["a"] = {"state": "queued"}
        self.assertFalse(admitted(status, ["a", "b"], [123], 110))
        self.assertFalse(admitted(self.status(), ["a", "b", "c"], [123], 110))

    def test_stale_failed_manager_and_completed_dependency(self):
        self.assertFalse(admitted(self.status(), ["a", "b"], [123], 1000))
        self.assertFalse(admitted(self.status(stage="manager_failed"),
                                  ["a", "b"], [123], 110))
        status = self.status(stage="training_completed")
        self.assertFalse(admitted(status, ["a", "b"], [], 110))
        status["jobs"]["b"] = {"state": "completed"}
        self.assertTrue(admitted(status, ["a", "b"], [], 1000))


if __name__ == "__main__":
    unittest.main()
