"""Protect seed separation, frozen data/source, and complete CUDA preflights."""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.cql_5090.observe import collect
from scripts.cql_5090.prepare import prepare
from scripts.cql_5090.run import require_preflight
from scripts.cql_delayed.common import digest, read, verify, write

class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.previous = Path(temporary.name) / "slurm"
        self.target = Path(temporary.name) / "5090"
        source = self.previous / "source"
        source.mkdir(parents=True)
        (source / "train.py").write_text("# unchanged official computation\n")
        config = dict(seed=0, name="old", group="old", batch_size=256,
                      max_timesteps=1000000, discount=0.99)
        (source / "config.yaml").write_text(yaml.safe_dump(config))
        data = self.previous / "shapley"
        data.mkdir()
        (data / "dataset.npz").write_bytes(b"exact frozen attribution output")
        write(data / "audit.json", dict(episodes=10))
        self.protocol = dict(
            source_sha256={p.name: digest(p) for p in source.iterdir()},
            runs=dict(shapley=dict(
                seed=0, wandb_name="CQL-shapley-seed0-retry1-env", wandb_id="old-id",
                config="config.yaml", data_sha256={
                    p.name: digest(p) for p in data.iterdir()})),
            changes=[], shapley_gate=dict(passed=True))
        write(self.previous / "protocol.json", self.protocol)

    def test_frozen_inputs_seed_change_and_portable_validation(self):
        original = {p.relative_to(self.previous): p.read_bytes()
                    for p in self.previous.rglob("*") if p.is_file()}
        prepare(self.previous, self.target, ["shapley"])
        p = verify(self.target)
        verify(self.target / "validation")
        self.assertEqual(p["runs"]["shapley"]["seed"], 1)
        self.assertNotEqual(p["runs"]["shapley"]["wandb_id"], "old-id")
        self.assertEqual(p["source_sha256"]["train.py"],
                         self.protocol["source_sha256"]["train.py"])
        old = yaml.safe_load((self.previous / "source/config.yaml").read_text())
        new = yaml.safe_load((self.target / "source/config.yaml").read_text())
        self.assertEqual({key for key in old if old[key] != new[key]},
                         {"seed", "name", "group"})
        for name, expected in p["runs"]["shapley"]["data_sha256"].items():
            self.assertEqual(digest(self.target / "shapley" / name), expected)
            self.assertEqual(digest(self.target / "validation/shapley" / name), expected)
        self.assertEqual(original, {path.relative_to(self.previous): path.read_bytes()
                                    for path in self.previous.rglob("*")
                                    if path.is_file()})
        with self.assertRaises(FileExistsError):
            prepare(self.previous, self.target, ["shapley"])

    def test_reject_inactive_arms_and_wrong_seed(self):
        for arms in [["dense"], ["uniform"], ["medium"], []]:
            with self.assertRaises(ValueError):
                prepare(self.previous, self.target, arms)
        with self.assertRaises(ValueError):
            prepare(self.previous, self.target, ["shapley"], seed=0)
        self.assertFalse(self.target.exists())

    def test_failed_gate_and_cancellation(self):
        self.protocol["shapley_gate"]["passed"] = False
        write(self.previous / "protocol.json", self.protocol)
        with self.assertRaisesRegex(ValueError, "gate"):
            prepare(self.previous, self.target, ["shapley"])
        self.protocol["shapley_gate"]["passed"] = True
        write(self.previous / "protocol.json", self.protocol)
        write(self.previous / "shapley/user_cancellation.json", dict(cancelled=True))
        with self.assertRaisesRegex(ValueError, "cancelled"):
            prepare(self.previous, self.target, ["shapley"])
        self.assertFalse(self.target.exists())

    def test_corrupted_data_rejected_before_write(self):
        (self.previous / "shapley/dataset.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Historical data changed"):
            prepare(self.previous, self.target, ["shapley"])
        self.assertFalse(self.target.exists())


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def test_preflight_requires_gpu_complete_updates_and_finite_eval(self):
        good = dict(status="completed", completed_updates=100, final_normalized_score=2)
        write(self.work / "status.json", good)
        write(self.work / "runtime.json", dict(device="cuda:0"))
        require_preflight(self.work)
        for change in [dict(status="failed"), dict(completed_updates=99),
                       dict(final_normalized_score=float("nan"))]:
            (self.work / "status.json").write_text(json.dumps({**good, **change}))
            with self.assertRaises(ValueError):
                require_preflight(self.work)
        write(self.work / "status.json", good)
        write(self.work / "runtime.json", dict(device="cpu"))
        with self.assertRaises(ValueError):
            require_preflight(self.work)

    def test_partial_lines_and_late_evaluation_are_not_lost(self):
        metrics = self.work / "metrics.jsonl"
        metrics.write_text('{"completed_updates": 100, "loss": 3}\n'
                           '{"completed_updates": 200')
        cursor = dict(updates=0, evaluations=[])
        self.assertEqual(collect(self.work, cursor), {100: {"train/loss": 3}})
        cursor["updates"] = 100
        write(self.work / "eval_0000100.json", dict(
            completed_updates=100, normalized_score=42, raw_return_mean=1000))
        self.assertEqual(collect(self.work, cursor)[100]["eval/normalized_score"], 42)
        cursor["evaluations"].append(100)
        self.assertEqual(collect(self.work, cursor), {})
        self.assertEqual(read(self.work / "eval_0000100.json")["normalized_score"], 42)


if __name__ == "__main__":
    unittest.main()
