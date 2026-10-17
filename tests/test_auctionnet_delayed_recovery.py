"""A logging outage may be retried only before any model training started."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.auctionnet_dt_delayed_reproduction.recover_repeat3 import (
    initialization_failure,
)

class RecoveryTests(unittest.TestCase):
    def test_missing_unrelated_or_started_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(initialization_failure(root))
            for status in [
                {"status": "running", "completed_updates": 0},
                {"status": "failed", "completed_updates": 0, "error": "CUDA OOM"},
                {"status": "failed", "completed_updates": 1,
                 "error": "wandb timed out initializing run"},
            ]:
                (root / "status.json").write_text(json.dumps(status))
                self.assertFalse(initialization_failure(root))

    def test_wandb_init_failure_requires_no_training_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            status = {"status": "failed", "completed_updates": 0,
                      "error": "Timed out initializing run: api.wandb.ai/graphql"}
            (root / "status.json").write_text(json.dumps(status))
            self.assertTrue(initialization_failure(root))
            (root / "console.log").touch()
            self.assertFalse(initialization_failure(root))
            (root / "console.log").unlink()
            checkpoint = root / "upstream/model/dt/0.pkl"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            self.assertFalse(initialization_failure(root))


if __name__ == "__main__":
    unittest.main()
