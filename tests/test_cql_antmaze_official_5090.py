"""Queue resource safety, immutable inputs and honest completion detection."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.cql_antmaze_official_5090.common import (
    admissible,
    completed,
    digest,
    evaluations,
    predecessor_ready,
    verify,
    write,
)
from scripts.cql_antmaze_official_5090.queue import command, execute

class OfficialBaselineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_predecessor_requires_every_job_and_an_unchanged_plan(self):
        previous = self.root / "previous"
        (previous / "queue").mkdir(parents=True)
        write(previous / "plan.json", dict(jobs=[dict(id="first"), dict(id="second")]))
        plan = dict(predecessor=str(previous),
                    predecessor_plan_sha256=digest(previous / "plan.json"))
        write(previous / "queue/status.json", dict(jobs={
            "first": dict(state="completed"), "second": dict(state="queued")}))
        self.assertFalse(predecessor_ready(plan))
        write(previous / "queue/status.json", dict(jobs={
            "first": dict(state="completed"), "second": dict(state="completed")}))
        self.assertTrue(predecessor_ready(plan))
        write(previous / "plan.json", dict(jobs=[]))
        with self.assertRaisesRegex(ValueError, "Predecessor plan changed"):
            predecessor_ready(plan)

    def test_manager_does_not_probe_or_launch_ahead_of_predecessor(self):
        (self.root / "queue").mkdir()
        job = dict(id="umaze-official-seed21")
        write(self.root / "plan.json", dict(jobs=[job], frozen_sha256={},
                                           predecessor="earlier", poll_seconds=30))
        write(self.root / "data_audit.json", dict(jobs={job["id"]: dict(
            full_array_comparison_passed=True)}))
        prefix = "scripts.cql_antmaze_official_5090.queue."
        with patch(prefix + "predecessor_ready", return_value=False), \
                patch(prefix + "time.sleep", side_effect=KeyboardInterrupt), \
                patch(prefix + "resources") as probe, \
                patch(prefix + "subprocess.Popen") as spawn:
            with self.assertRaises(KeyboardInterrupt):
                execute(self.root)
            probe.assert_not_called()
            spawn.assert_not_called()

    def test_invisible_starting_process_reserves_memory_and_process_slot(self):
        plan = dict(max_parallel=2, max_gpu_processes=8, gpu_memory_reserve_mib=4096,
                    gpu_memory_per_job_mib=5120, host_memory_reserve_mib=8192,
                    host_memory_per_job_mib=6144, max_admission_utilization=85)
        resource = dict(gpu_pids=list(range(6)), free_gpu_mib=10000,
                        free_host_mib=30000, utilization=50)
        self.assertTrue(admissible(plan, resource, []))
        self.assertFalse(admissible(plan, resource, [100]))
        resource["free_gpu_mib"] = 16000
        self.assertTrue(admissible(plan, resource, [100]))
        self.assertFalse(admissible(plan, resource, [100, 101]))
        resource["utilization"] = 95
        self.assertFalse(admissible(plan, resource, []))

    def test_announced_step_is_not_a_completed_evaluation(self):
        log = ("Time steps: 50000\n"
               "Evaluation over 100 episodes: 0.300 , D4RL score: 30.000\n")
        log += "Time steps: 100000\n"
        self.assertEqual([r["completed_updates"] for r in evaluations(log)], [50000])
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            evaluations("Time steps: 1\n"
                        "Evaluation over 1 episodes: nan , D4RL score: nan")

    def test_completion_requires_evaluations_and_zero_index_checkpoint(self):
        log = ("Time steps: 50000\n"
               "Evaluation over 100 episodes: 0.300 , D4RL score: 30.000\n")
        (self.root / "console.log").write_text(log)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            completed(self.root, 100000, 50000, 100)
        log += ("Time steps: 100000\n"
                "Evaluation over 100 episodes: 0.400 , D4RL score: 40.000\n")
        (self.root / "console.log").write_text(log)
        with self.assertRaisesRegex(ValueError, "Missing"):
            completed(self.root, 100000, 50000, 100)
        checkpoint = self.root / "checkpoints/official/checkpoint_99999.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"test checkpoint")
        self.assertEqual(completed(self.root, 100000, 50000, 100)[
            "final_normalized_score"], 40.0)

    def test_mutated_source_is_rejected(self):
        source = self.root / "cql.py"
        source.write_text("original")
        plan = dict(frozen_sha256={"cql.py": digest(source)})
        verify(self.root, plan)
        source.write_text("edited")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            verify(self.root, plan)

    def test_training_invokes_original_cli_without_algorithm_overrides(self):
        job = dict(id="umaze-official-seed21", seed=21,
                   config="configs/offline/cql/antmaze/umaze_v2.yaml")
        argv = command(self.root, dict(wandb_project="CORL-DDR"), job, "training")
        self.assertEqual(argv[2], str(self.root / "upstream/algorithms/offline/cql.py"))
        for flag in ["--max_timesteps", "--eval_freq", "--n_episodes", "--load_model",
                     "--reward_scale", "--reward_bias", "--discount"]:
            self.assertNotIn(flag, argv)
        self.assertEqual(argv[argv.index("--seed") + 1], "21")


if __name__ == "__main__":
    unittest.main()
