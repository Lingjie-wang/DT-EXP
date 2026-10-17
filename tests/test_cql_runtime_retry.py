"""Retry preparation must preserve results and reject cancelled/corrupted inputs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.cql_runtime.prepare_retry import digest, prepare, write

class RetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.previous = Path(self.tmp.name) / "old"
        self.root = Path(self.tmp.name) / "new"
        self.entry = "scripts/cql_shapley/train.py"
        original = self.previous / "source" / self.entry
        original.parent.mkdir(parents=True)
        original.write_text("# original training entry\n")
        runs = {}
        for arm in ["shapley", "uniform", "dense"]:
            work = self.previous / arm
            work.mkdir()
            (work / "dataset.npz").write_bytes(b"frozen-test-data")
            (work / "audit.json").write_text('{}\n')
            runs[arm] = dict(wandb_id=f"old-{arm}", wandb_name=f"test-{arm}",
                             data_sha256={name: digest(work / name) for name in
                                          ["dataset.npz", "audit.json"]})
        write(self.previous / "protocol.json", dict(
            runs=runs, source_sha256={self.entry: digest(original)},
            repository_base="old-commit", shapley_gate=dict(passed=True),
            preparation_sha256={"predictor/input.npz": "old-input-hash"},
        ))
        write(self.previous / "uniform/user_cancellation.json", dict(cancelled=True))
        write(self.previous / "dense/user_cancellation.json", dict(cancelled=True))

    def test_selected_retry_isolated_and_bit_identical(self):
        protocol_bytes = (self.previous / "protocol.json").read_bytes()
        # Source exports intentionally have no .git; mock only revision metadata.
        with patch("scripts.cql_runtime.prepare_retry.subprocess.check_output",
                   return_value="fixture-revision\n"):
            prepare(self.previous, self.root, ["shapley"], "cql_shapley")
        p = json.loads((self.root / "protocol.json").read_text())
        self.assertEqual(list(p["runs"]), ["shapley"])
        self.assertFalse((self.root / "uniform").exists())
        self.assertFalse((self.root / "dense").exists())
        self.assertEqual((self.previous / "protocol.json").read_bytes(), protocol_bytes)
        for name in ["dataset.npz", "audit.json"]:
            self.assertEqual(digest(self.root / "shapley" / name),
                             digest(self.previous / "shapley" / name))
            self.assertEqual(digest(self.root / "validation/shapley" / name),
                             digest(self.previous / "shapley" / name))
        self.assertTrue(p["shapley_gate"]["passed"])
        self.assertNotEqual(p["runs"]["shapley"]["wandb_id"], "old-shapley")
        self.assertEqual(p["runs"]["shapley"]["retry_of_wandb_id"], "old-shapley")
        self.assertNotIn("preparation_sha256", p)
        self.assertIn("original_preparation_sha256", p)
        for name, expected in p["source_sha256"].items():
            self.assertEqual(digest(self.root / "source" / name), expected)

    def test_cancelled_control_cannot_be_restarted(self):
        with self.assertRaisesRegex(ValueError, "user-cancelled"):
            prepare(self.previous, self.root, ["uniform"], "cql_shapley")
        self.assertFalse(self.root.exists())

    def test_corrupted_data_rejected_before_creating_retry(self):
        (self.previous / "shapley/dataset.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Historical data changed"):
            prepare(self.previous, self.root, ["shapley"], "cql_shapley")
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
