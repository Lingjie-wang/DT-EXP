"""Delayed repeats must wait for both ordinary runs and preserve their own task."""

import json
import tempfile
import unittest
from pathlib import Path

from test_auctionnet_reproduction import console, protocol

from scripts.auctionnet_dt_delayed_reproduction.pipeline import dependency_ready
from scripts.auctionnet_dt_delayed_reproduction.summarize import (
    complete,
    summarize,
    validate_protocol,
)

def ordinary_done():
    return {"status": "completed", "completed_updates": 100000,
            "upstream_python_unchanged": True}


def delayed_done():
    return {**ordinary_done(), "upstream_python_unchanged": False,
            "runtime_source_unchanged": True}


def delayed_protocol():
    result = protocol()
    result["python_source_modified"] = True
    result["config"]["delayed_reward"] = True
    result["upstream_original_sha256"]["evaluation_bidding.py"] = "original"
    result["runtime_sha256"]["evaluation_bidding.py"] = "delayed"
    result["reward_protocol"] = {"training": "terminal only", "evaluation": "zero"}
    return result


class QueueTests(unittest.TestCase):
    def test_waits_for_both_training_and_final_evaluation(self):
        self.assertFalse(dependency_ready(None, [None] * 3))
        self.assertFalse(dependency_ready({"stage": "running"}, [
            ordinary_done(), ordinary_done(),
            {"status": "running", "completed_updates": 100000}]))
        self.assertFalse(dependency_ready({"stage": "summarizing"},
                                          [ordinary_done()] * 3))
        self.assertTrue(dependency_ready({"stage": "completed"},
                                         [ordinary_done()] * 3))

    def test_failure_or_inconsistent_completion_blocks_queue(self):
        cases = [({"stage": "failed"}, [None] * 3),
                 ({"stage": "running"}, [ordinary_done(), {"status": "failed"}, None]),
                 ({"stage": "completed"}, [ordinary_done()] * 2 + [None]),
                 ({"stage": "completed"}, [ordinary_done()] * 2),
                 ({"stage": "completed"}, [ordinary_done()] * 2 + [
                     {"status": "completed", "completed_updates": 100000}])]
        for state, statuses in cases:
            with self.subTest(state=state, statuses=statuses):
                with self.assertRaises(RuntimeError):
                    dependency_ready(state, statuses)


class DelayedStatisticsTests(unittest.TestCase):
    def test_ordinary_completion_not_accepted_as_delayed(self):
        self.assertTrue(complete(delayed_done()))
        self.assertFalse(complete({"status": "running", "completed_updates": 100000}))
        with self.assertRaises(ValueError):
            complete(ordinary_done())

    def test_only_existing_delayed_evaluator_change_allowed(self):
        validate_protocol(delayed_protocol())
        with self.assertRaises(ValueError):
            validate_protocol(protocol())
        changed = delayed_protocol()
        changed["runtime_sha256"]["main.py"] = "modified"
        with self.assertRaises(ValueError):
            validate_protocol(changed)

    def test_statistics_preserve_task_and_reject_mixed_rewards(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = [Path(folder) / str(i) for i in range(3)]
            for i, root in enumerate(roots):
                root.mkdir()
                (root / "status.json").write_text(json.dumps(delayed_done()))
                (root / "protocol.json").write_text(json.dumps(delayed_protocol()))
                (root / "console.log").write_text(console(i * 2))
            result = summarize(roots)
            self.assertEqual(result["targets"]["1.0"]["mean"], 19.)
            self.assertEqual(result["targets"]["1.0"]["std_sample"], 2.)
            self.assertEqual(result["targets"]["0.2"]["mean"], 23.)
            self.assertIsNone(result["seed_values"])
            self.assertNotIn("paper_dt_mean", result["targets"]["1.0"]["periods"]["14"])
            with self.assertRaises(ValueError):
                summarize([roots[0]] * 3)
            changed = delayed_protocol()
            changed["reward_protocol"]["evaluation"] = "step reward"
            (roots[2] / "protocol.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "reward_protocol"):
                summarize(roots)
            (roots[2] / "protocol.json").write_text(json.dumps(protocol()))
            with self.assertRaises(ValueError):
                summarize(roots)


if __name__ == "__main__":
    unittest.main()
