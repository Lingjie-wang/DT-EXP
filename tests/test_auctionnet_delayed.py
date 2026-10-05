"""Queue safety and terminal-reward behavior against the pinned official source."""

import hashlib
import importlib.util
import json
import os
import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.auctionnet_dt.prepare_run import prepare
from scripts.auctionnet_dt_delayed.pipeline import dependency_ready, snapshot_data
from scripts.auctionnet_dt_delayed.prepare_run import configure_delayed

def write(path, value):
    path.write_text(json.dumps(value))


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QueueTests(unittest.TestCase):
    def test_wait_failure_and_verified_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous, queue = root / "run", root / "queue"
            previous.mkdir()
            queue.mkdir()
            self.assertFalse(dependency_ready(previous, queue))
            for state in ({"status": "starting"},
                          {"status": "running", "completed_updates": 100000}):
                write(previous / "status.json", state)
                self.assertFalse(dependency_ready(previous, queue))
            for state in ({"status": "failed"},
                          {"status": "completed", "completed_updates": 90000,
                           "upstream_python_unchanged": True},
                          {"status": "completed", "completed_updates": 100000}):
                write(previous / "status.json", state)
                with self.assertRaises(RuntimeError):
                    dependency_ready(previous, queue)
            write(previous / "status.json", {
                "status": "completed", "completed_updates": 100000,
                "upstream_python_unchanged": True})
            self.assertTrue(dependency_ready(previous, queue))
            write(queue / "queue_status.json", {"stage": "failed"})
            with self.assertRaisesRegex(RuntimeError, "queue failed"):
                dependency_ready(previous, queue)

    def test_snapshot_verifies_provenance_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            data = b"immutable test data"
            (source / "training_data_small.pkl").write_bytes(data)
            audit = {"sha256": {"training_data_small.pkl":
                                hashlib.sha256(data).hexdigest()}}
            write(source / "audit.json", audit)
            protocol = {"dataset_version": "non-final/general", "data_audit": audit,
                        "config": {"delayed_reward": False, "is_stitch": False,
                                   "model_type": "dt"}}
            for key, value in (("delayed_reward", True), ("is_stitch", True),
                               ("model_type", "other")):
                bad = {**protocol, "config": {**protocol["config"], key: value}}
                with self.assertRaises(ValueError):
                    snapshot_data(source, root / "bad", bad)
            with self.assertRaises(ValueError):
                snapshot_data(source, root / "bad", {**protocol,
                                                     "dataset_version": "final/general"})
            snapshot_data(source, root / "copy", protocol)
            (root / "copy/training_data_small.pkl").write_bytes(b"independent")
            self.assertEqual((source / "training_data_small.pkl").read_bytes(), data)
            (source / "training_data_small.pkl").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "data changed"):
                snapshot_data(source, root / "bad", protocol)


@unittest.skipUnless(os.environ.get("PRGS_AUCTIONNET_SOURCE"),
                     "Set PRGS_AUCTIONNET_SOURCE to the pinned source")
class OfficialBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(os.environ["PRGS_AUCTIONNET_SOURCE"])
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_actual_loader_delays_each_episode_without_changing_pickle(self):
        loader = module_from(self.source / "utils.py", "official_utils")
        paths = [{"observations": np.arange(6).reshape(3, 2) + offset,
                  "actions": np.arange(3).reshape(3, 1),
                  "rewards": np.array([1., 2., 4.]) + offset,
                  "terminals": np.array([0, 0, 1])} for offset in (0, 10)]
        dataset = self.root / "data.pkl"
        original = pickle.dumps(paths)
        dataset.write_bytes(original)
        config = {"env_name": "AuctionNet", "state_dim": 2, "act_dim": 1,
                  "dataset_path": str(dataset), "device": "cpu", "max_ep_len": 96,
                  "scale": 2000, "reward_scale": 1, "K": 3, "delayed_reward": True}
        delayed = loader.SequenceDataset(config)
        ordinary = loader.SequenceDataset({**config, "delayed_reward": False})
        for i, (before, after) in enumerate(zip(paths, delayed.trajectories)):
            np.testing.assert_array_equal(
                after["rewards"], [0, 0, before["rewards"].sum()])
            for key in ("observations", "actions", "terminals"):
                np.testing.assert_array_equal(after[key], before[key])
            np.testing.assert_allclose(delayed.rtg[i * 3:(i + 1) * 3],
                                       before["rewards"].sum() / 2000)
        np.testing.assert_array_equal(delayed.returns, ordinary.returns)
        np.testing.assert_array_equal(delayed.state_mean, ordinary.state_mean)
        self.assertEqual(dataset.read_bytes(), original)

    def test_preparation_and_actual_evaluator_only_change_reward_feedback(self):
        data = self.root / "data"
        data.mkdir()
        for name in ("training_data_small.pkl", "normalize_dict.pkl"):
            (data / name).write_bytes(pickle.dumps({}))
        audit = {"training_periods": list(range(7, 14)), "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}}
        write(data / "audit.json", audit)
        csvs = [self.root / f"period-{p}.csv" for p in range(14, 21)]
        for path in csvs:
            path.touch()
        run = self.root / "run"
        prepare(self.source, data, csvs, run, False)
        before = json.loads((run / "protocol.json").read_text())
        configure_delayed(run)
        after = json.loads((run / "protocol.json").read_text())
        changed = {name for name, sha in before["runtime_sha256"].items()
                   if after["runtime_sha256"][name] != sha}
        self.assertEqual(changed,
                         {"evaluation_bidding.py", "config/env/AuctionNet.yaml"})
        self.assertEqual(
            {k: v for k, v in after["config"].items() if k != "delayed_reward"},
            {k: v for k, v in before["config"].items() if k != "delayed_reward"})
        self.assertTrue(after["config"]["delayed_reward"])
        with self.assertRaises(ValueError):
            configure_delayed(run)

        class Loader:
            def __init__(self, file_path):
                self.keys = [0, 1]

            def mock_data(self, key):
                values = [np.array([1.])] * 3
                return 3, values, values, values, 100., 10., 0

        class Environment:
            min_remaining_budget = 0

            def simulate_ad_bidding(self, *args):
                return tuple(np.array([v]) for v in (1., 1., 1., 2.))

        class Model:
            def __init__(self):
                self.feedback, self.states = [], []

            def eval(self):
                pass

            def init_eval(self):
                pass

            def take_actions(self, state, target_return, pre_reward):
                self.feedback.append(pre_reward)
                self.states.append(state.copy())
                return 1.

        mocks = {}
        for name, attr, cls in (
                ("offline_eval.test_dataloader", "TestDataLoader", Loader),
                ("offline_eval.offline_env", "OfflineEnv", Environment)):
            mocks[name] = types.ModuleType(name)
            setattr(mocks[name], attr, cls)
        with patch.dict("sys.modules", mocks):
            ordinary_eval = module_from(
                self.source / "evaluation_bidding.py", "ordinary_eval")
            delayed_eval = module_from(
                run / "upstream/evaluation_bidding.py", "delayed_eval")
        ordinary_model, delayed_model = Model(), Model()
        config = {"test_csv_list": ["fixture.csv"], "max_ep_len": 96, "is_stitch": False}
        score = ordinary_eval.Evaluation(config, None, None)._run_once(
            ordinary_model, 1.)
        delayed_score = delayed_eval.Evaluation(config, None, None)._run_once(
            delayed_model, 1.)
        self.assertEqual(ordinary_model.feedback, [None, 2., 2.] * 2)
        self.assertEqual(delayed_model.feedback, [0.] * 6)
        np.testing.assert_array_equal(ordinary_model.states, delayed_model.states)
        self.assertEqual(score, (6., 0))
        self.assertEqual(delayed_score, score)


if __name__ == "__main__":
    unittest.main()
