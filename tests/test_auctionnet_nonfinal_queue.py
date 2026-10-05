"""The second dataset must never start training before the first run succeeds."""

import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "auctionnet_nonfinal_pipeline",
    Path(__file__).resolve().parents[1] / "scripts/auctionnet_dt_nonfinal/pipeline.py",
)
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class PredecessorTests(unittest.TestCase):
    def test_running_at_100k_still_waits_for_evaluation(self):
        self.assertFalse(PIPELINE.predecessor_complete(
            {"status": "running", "completed_updates": 100000}))

    def test_starting_waits(self):
        self.assertFalse(PIPELINE.predecessor_complete({"status": "starting"}))

    def test_failed_predecessor_blocks_the_next_run(self):
        with self.assertRaisesRegex(RuntimeError, "did not succeed"):
            PIPELINE.predecessor_complete({"status": "failed"})

    def test_partial_completion_is_not_enough(self):
        with self.assertRaisesRegex(RuntimeError, "100k"):
            PIPELINE.predecessor_complete({"status": "completed",
                                           "completed_updates": 90000,
                                           "upstream_python_unchanged": True})

    def test_source_verification_is_required(self):
        with self.assertRaisesRegex(RuntimeError, "verification"):
            PIPELINE.predecessor_complete({"status": "completed",
                                           "completed_updates": 100000})

    def test_verified_full_completion_releases_queue(self):
        self.assertTrue(PIPELINE.predecessor_complete({"status": "completed",
                                                     "completed_updates": 100000,
                                                     "upstream_python_unchanged": True}))


if __name__ == "__main__":
    unittest.main()
