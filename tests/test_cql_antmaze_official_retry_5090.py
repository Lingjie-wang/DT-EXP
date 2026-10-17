"""A recovery must preserve and positively identify a pre-training failure."""

import tempfile
import unittest
from pathlib import Path

from scripts.cql_antmaze_official_5090.common import digest, write
from scripts.cql_antmaze_official_retry_5090.prepare import require_unstarted_failure

class OfficialRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "umaze/training"
        self.work.mkdir(parents=True)
        (self.root / "queue").mkdir()
        self.log = self.work / "console.log"
        self.log.write_text(
            "TypeError: Run.save() missing 1 required positional argument")
        self.source = self.root / "cql.py"
        self.source.write_text("unchanged official source")
        write(self.root / "plan.json", dict(jobs=[dict(id="umaze")],
              frozen_sha256={"cql.py": digest(self.source)}))
        self.state = dict(stage="finished_with_failures",
                          jobs={"umaze": dict(state="failed", completed_updates=0)})
        self.save_state()

    def save_state(self):
        write(self.root / "queue/status.json", self.state)

    def test_known_failure_passes_without_modifying_records(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(require_unstarted_failure(self.root)["jobs"],
                         [dict(id="umaze")])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_running_manager_and_other_failure_are_rejected(self):
        self.state["stage"] = "running"
        self.save_state()
        with self.assertRaisesRegex(ValueError, "finished with failures"):
            require_unstarted_failure(self.root)
        self.state["stage"] = "finished_with_failures"
        self.save_state()
        self.log.write_text("CUDA out of memory")
        with self.assertRaisesRegex(ValueError, "known pre-training"):
            require_unstarted_failure(self.root)

    def test_training_evidence_prevents_fresh_start(self):
        checkpoint = self.work / "checkpoints/official/checkpoint_49999.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"previous progress")
        with self.assertRaisesRegex(ValueError, "known pre-training"):
            require_unstarted_failure(self.root)
        checkpoint.unlink()
        self.log.write_text(self.log.read_text() + "\nTime steps: 50000\n")
        with self.assertRaisesRegex(ValueError, "known pre-training"):
            require_unstarted_failure(self.root)

    def test_changed_historical_source_is_rejected(self):
        self.source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            require_unstarted_failure(self.root)


if __name__ == "__main__":
    unittest.main()
