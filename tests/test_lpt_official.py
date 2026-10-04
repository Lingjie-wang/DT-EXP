"""Validate observation and dataset preparation against the released collator."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from datasets import Dataset
from observe import parse_line
from prepare import terminal_rewards, trajectories_from_hdf5
from run import check_source, digest

SOURCE = Path("third_party/Latent-Plan-Transformer")


class ObserverTests(unittest.TestCase):
    def test_official_stdout(self):
        kind, value = parse_line("Episode 3: Return = -1.25e+02, Length = 1000\n")
        self.assertEqual(kind, "episode")
        self.assertEqual(value, dict(episode=3, raw_return=-125.0, length=1000))
        kind, value = parse_line(
            "Evaluation metrics at step 500: {'eval/avg_reward': 4000.0, "
            "'eval/norm_score': 34.0}\n"
        )
        self.assertEqual(kind, "evaluation")
        self.assertEqual(value["completed_updates"], 500)
        self.assertEqual(value["eval/norm_score"], 34.0)
        kind, value = parse_line(
            "{'loss': 0.123, 'learning_rate': 1e-4, 'epoch': 500.0}"
        )
        self.assertEqual(kind, "training")
        self.assertEqual(value["completed_updates"], 500)
        self.assertEqual(
            parse_line(" 25%|██ | 500/2000 [00:30<02:00]"),
            ("progress", dict(completed_updates=500)),
        )
        self.assertIsNone(parse_line("iter: 0 Reward Pred: 0.4"))

    def test_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.py"
            path.write_text("original")
            protocol = dict(upstream_sha256={"train.py": digest(path)})
            check_source(Path(directory), protocol)
            path.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "Official source changed"):
                check_source(Path(directory), protocol)


class DatasetTests(unittest.TestCase):
    def test_terminal_or_timeout_boundaries_preserve_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.hdf5"
            with h5py.File(path, "w") as f:
                f["observations"] = np.arange(12).reshape(6, 2)
                f["actions"] = np.arange(6).reshape(6, 1)
                f["rewards"] = np.array([1, 2, -3, 4, 5, 6], dtype=np.float32)
                f["terminals"] = [False, True, False, False, False, False]
                f["timeouts"] = [False, False, False, False, False, True]
            rows = trajectories_from_hdf5(path)
            self.assertEqual([len(r["rewards"]) for r in rows], [2, 4])
            np.testing.assert_array_equal(
                np.concatenate([r["actions"] for r in rows]), np.arange(6).reshape(6, 1)
            )
            self.assertFalse(rows[-1]["dones"][-1])

    @unittest.skipUnless(
        (SOURCE / "model/utils_train.py").exists(), "Official source required"
    )
    def test_actual_official_collator_consumes_identical_labels(self):
        spec = importlib.util.spec_from_file_location(
            "official_collator", SOURCE / "model/utils_train.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.RandomState(17)
        data = dict(
            observations=rng.randn(8, 20, 17).astype(np.float32),
            actions=rng.randn(8, 20, 6).astype(np.float32),
            rewards=rng.randn(8, 20).astype(np.float32),
            dones=np.zeros((8, 20), dtype=bool),
        )
        dense = Dataset.from_dict(data)
        delayed = Dataset.from_dict(
            {**data, "rewards": [terminal_rewards(r) for r in data["rewards"]]}
        )
        batches = []
        for dataset in (dense, delayed):
            collator = module.LPTGymDataCollator(
                dataset, max_len=20, scale=1000.0, sample_by_length=True
            )
            np.random.seed(42)
            batches.append(collator([{}] * 8))
        for key in (
            "states",
            "actions",
            "rewards",
            "timesteps",
            "attention_mask",
            "batch_inds",
        ):
            torch.testing.assert_close(batches[0][key], batches[1][key], rtol=0, atol=0)
        self.assertFalse(
            torch.equal(batches[0]["returns_to_go"], batches[1]["returns_to_go"])
        )
        self.assertEqual(batches[0]["states"].shape, (8, 20, 17))


if __name__ == "__main__":
    unittest.main()
