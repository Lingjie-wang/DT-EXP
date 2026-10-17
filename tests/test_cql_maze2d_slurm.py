import tempfile
import unittest
from pathlib import Path

from scripts.cql_antmaze_official_5090.common import digest, read, write
from scripts.cql_maze2d_slurm.migrate import migrate
from scripts.cql_maze2d_slurm.observe import aggregate, parse_accounting

class SlurmMigrationTest(unittest.TestCase):
    def test_migration_preserves_frozen_inputs_and_requires_explicit_fresh_run(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source/scripts").mkdir(parents=True)
            (root / "input.bin").write_bytes(b"fixed rewards and transitions")
            (root / "original").mkdir()
            write(root / "original/data_audit.json", dict(passed=True))
            original = dict(jobs=[dict(id="original")], seed=23,
                            frozen_sha256={"input.bin": digest(root / "input.bin")})
            write(root / "plan.json", original)
            write(root / "migration_stop.json", dict(formal_training_started=False,
                                                     remaining_processes=[]))
            migrate(root, project, Path("/runtime/bin/python"))
            self.assertEqual(read(root / "provenance_5090/plan.json"), original)
            migrated = read(root / "plan.json")
            self.assertEqual(migrated["jobs"], original["jobs"])
            self.assertEqual(migrated["seed"], 23)
            self.assertEqual(digest(root / "input.bin"),
                             migrated["frozen_sha256"]["input.bin"])
            self.assertFalse((root / "original/data_audit.json").exists())
            with self.assertRaises(FileExistsError):
                migrate(root, project, Path("/runtime/bin/python"))

    def test_migration_rejects_source_with_training_or_live_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "plan.json", {})
            for training, processes in [(True, []), (False, [123])]:
                write(root / "migration_stop.json", dict(
                    formal_training_started=training, remaining_processes=processes))
                with self.assertRaises(ValueError):
                    migrate(root, root, Path("/runtime/bin/python"))

    def test_recovery_uses_only_selected_attempt_for_each_arm(self):
        jobs = [dict(id=name) for name in ["original", "delayed", "uniform", "shapley"]]
        submitted = [dict(id=j["id"], slurm_id=task) for j, task in zip(
            jobs, ["20_0", "20_1", "10_2", "10_3"])]
        report = ("10_0|NODE_FAIL|\n10_1|NODE_FAIL|\n10_2|RUNNING|\n"
                  "10_3|COMPLETED|\n20_0|PENDING|\n20_1|RUNNING|\n"
                  "20_1.batch|RUNNING|\n")
        self.assertEqual(parse_accounting(report, jobs, submitted),
                         {0: "PENDING", 1: "RUNNING", 2: "RUNNING", 3: "COMPLETED"})

    def test_scheduler_cancellation_cannot_leave_observer_waiting_forever(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(jobs=[dict(id="original"), dict(id="delayed")])
            (root / "original").mkdir()
            write(root / "original/slurm_status.json", dict(state="completed"))
            mixed = aggregate(root, plan, {0: "COMPLETED", 1: "PENDING"})
            self.assertEqual(mixed["stage"], "slurm_queue")
            final = aggregate(root, plan, {0: "COMPLETED", 1: "CANCELLED"})
            self.assertEqual(final["stage"], "finished_with_failures")
            self.assertEqual(final["jobs"]["original"]["state"], "completed")
            self.assertEqual(final["jobs"]["delayed"]["state"], "failed")


if __name__ == "__main__":
    unittest.main()
