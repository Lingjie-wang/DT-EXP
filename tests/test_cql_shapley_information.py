"""Counterfactual tests for reward information and held-fold isolation."""

import importlib.util
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from algorithms.offline.cql_delayed_data import terminal_return_dataset
from algorithms.offline.shapley_redistribution import folds
from scripts.cql_delayed.common import digest, read, write
from scripts.cql_shapley_medium_5090.prepare import predictor_input

REPO = Path(__file__).resolve().parents[1]

class InformationTests(unittest.TestCase):
    def raw(self, episodes):
        count = episodes * 1000
        timeouts = np.zeros(count, dtype=bool)
        timeouts[999::1000] = True
        return dict(observations=np.arange(count, dtype=np.float32).reshape(-1, 1),
                    actions=np.ones((count, 1), dtype=np.float32),
                    rewards=((np.arange(count) % 9) - 4).astype(np.float32) / 8,
                    terminals=np.zeros(count, dtype=bool), timeouts=timeouts)

    def terminal_only(self, raw):
        # Retain exact totals; remove the entire within-trajectory reward pattern.
        rewards = np.zeros(len(raw["rewards"]), dtype=np.float64)
        rewards[999::1000] = raw["rewards"].reshape(-1, 1000).astype(np.float64).sum(1)
        return dict(raw, rewards=rewards)

    def test_medium_predictor_input_unchanged_without_dense_rewards(self):
        raw = self.raw(1000)
        baseline, _ = terminal_return_dataset(raw)
        dense = predictor_input(raw, baseline)
        delayed = predictor_input(self.terminal_only(raw), baseline)
        for left, right in zip(dense[:2], delayed[:2]):
            self.assertEqual(set(left), set(right))
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])

    def test_actual_replay_preparer_input_unchanged_without_dense_rewards(self):
        raw = self.raw(202)
        data, _ = terminal_return_dataset(raw)
        scientific = ["algorithms/offline/cql.py",
                      "configs/offline/cql/halfcheetah/medium_replay_v2.yaml"]
        with tempfile.TemporaryDirectory() as tmp:
            temporary = Path(tmp)
            roots = []
            for name, values in [("dense", raw), ("delayed", self.terminal_only(raw))]:
                hdf5 = temporary / f"{name}.hdf5"
                with h5py.File(hdf5, "w") as stream:
                    for key, value in values.items():
                        stream[key] = value
                baseline = temporary / f"baseline_{name}"
                (baseline / "medium_replay").mkdir(parents=True)
                np.savez(baseline / "medium_replay/dataset.npz", **data)
                write(baseline / "medium_replay/audit.json",
                      dict(original_sha256=digest(hdf5)))
                write(baseline / "protocol.json", dict(
                    source_sha256={p: digest(REPO / p) for p in scientific},
                    runs={"medium_replay": {}}, retained={},
                    wandb_entity="test", wandb_project="test"))
                output = temporary / f"prepared_{name}"
                argv = ["prepare.py", "--root", str(output), "--baseline", str(baseline),
                        "--hdf5", str(hdf5)]
                # Only revision metadata is stubbed for CI exports without .git.
                with patch.object(sys, "argv", argv), patch(
                        "subprocess.check_output", return_value="audit-fixture\n"):
                    runpy.run_path(str(REPO / "scripts/cql_shapley/prepare.py"),
                                   run_name="__main__")
                roots.append(output)
            for name in ["predictor/input.npz", "base_transitions.npz"]:
                with np.load(roots[0] / name) as a, np.load(roots[1] / name) as b:
                    self.assertEqual(set(a.files), set(b.files))
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(read(roots[0] / "predictor/folds.json"),
                             read(roots[1] / "predictor/folds.json"))

    def test_held_features_and_returns_cannot_change_fitted_weights(self):
        rng = np.random.default_rng(19)
        features = rng.normal(size=(20, 1000, 2)).astype(np.float32)
        returns = rng.normal(size=20)
        split = list(folds(20))[0]
        settings = dict(learning_rate=.001, weight_decay=.0001, batch=8,
                        validation_masks=2, epochs=4, patience=4)
        spec = importlib.util.spec_from_file_location(
            "audited_fit", REPO / "scripts/cql_shapley/fit_redistribute.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            outputs = []
            for index in range(2):
                out = Path(tmp) / str(index)
                out.mkdir()
                if index:
                    features[split["held"]] = 1e5
                    returns[split["held"]] = -1e6
                model, _, _ = module.fit_one(features, returns, split, settings,
                                             out, "cpu")
                outputs.append({k: v.clone() for k, v in model.state_dict().items()})
            for key in outputs[0]:
                self.assertTrue(torch.equal(outputs[0][key], outputs[1][key]), key)


# The unchanged standalone entrypoints resolve their sibling common module here.
sys.path.insert(0, str(REPO / "scripts/cql_shapley"))
torch.set_num_threads(2)

if __name__ == "__main__":
    unittest.main()
