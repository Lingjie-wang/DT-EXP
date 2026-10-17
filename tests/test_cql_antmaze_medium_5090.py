import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.cql_antmaze_medium_5090 import pipeline
from scripts.cql_antmaze_medium_5090.compare import snapshot
from scripts.cql_antmaze_medium_5090.dependency import predecessor_status
from scripts.cql_antmaze_medium_5090.prepare import job_matrix
from scripts.cql_antmaze_medium_5090.queue import command, environment
from scripts.cql_antmaze_official_5090.common import digest, read, write

class MediumCampaignTest(unittest.TestCase):
    def test_six_paired_jobs_keep_official_scientific_arguments(self):
        jobs = job_matrix(22)
        self.assertEqual(len(jobs), 6)
        self.assertEqual(len({j["id"] for j in jobs}), 6)
        self.assertEqual({j["seed"] for j in jobs}, {22})
        root = Path("/tmp/medium-campaign")
        plan = dict(jobs=jobs, wandb_project="project", wandb_entity="entity")
        for name in {j["dataset_key"] for j in jobs}:
            arms = [j for j in jobs if j["dataset_key"] == name]
            self.assertEqual({j["arm"] for j in arms},
                             {"original", "uniform", "shapley"})
            self.assertEqual(len({j["config"] for j in arms}), 1)
            for job in arms:
                argv = command(root, plan, job, "training")
                for option in ["--max_timesteps", "--eval_freq", "--n_episodes",
                               "--discount", "--cql_alpha", "--normalize_reward"]:
                    self.assertNotIn(option, argv)
                self.assertEqual(argv[argv.index("--seed") + 1], "22")
                env = environment(root, plan, root / job["id"] / "training", "training")
                self.assertEqual(env["D4RL_DATASET_DIR"],
                                 str(root / job["dataset_dir"]))
                self.assertEqual(env["WANDB_MODE"], "online")

    def predecessor(self, root, state="failed", complete=True):
        job = dict(id="old", updates=100, eval_every=50, eval_episodes=100)
        write(root / "plan.json", dict(jobs=[job]))
        (root / "queue").mkdir()
        write(root / "queue/status.json",
              dict(jobs={"old": dict(state=state, exit=255)}))
        work = root / "old/training"
        (work / "checkpoints/model").mkdir(parents=True)
        (work / "checkpoints/model/checkpoint_99.pt").write_bytes(b"checkpoint")
        lines = [f"Time steps: {step}\nEvaluation over 100 episodes: 0.4 , "
                 "D4RL score: 40.0\n" for step in ([50, 100] if complete else [50])]
        (work / "console.log").write_text("".join(lines))
        return dict(predecessor=str(root), predecessor_plan_sha256=digest(
                    root / "plan.json"))

    def test_nonzero_exit_only_admitted_with_full_training_artifacts(self):
        for complete in [False, True]:
            with tempfile.TemporaryDirectory() as directory:
                plan = self.predecessor(Path(directory), complete=complete)
                self.assertEqual(predecessor_status(plan)["ready"], complete)

    def test_live_or_not_terminal_predecessor_blocks_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.predecessor(root, state="running")
            self.assertFalse(predecessor_status(plan)["ready"])
            write(root / "queue/status.json", dict(jobs={
                "old": dict(state="completed", pid=123)}))
            with patch("scripts.cql_antmaze_medium_5090.dependency.process_running",
                       return_value=True):
                self.assertFalse(predecessor_status(plan)["ready"])
            (root / "old/training/checkpoints/model/checkpoint_99.pt").unlink()
            self.assertFalse(predecessor_status(plan)["ready"])

    def test_changed_predecessor_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.predecessor(root)
            write(root / "plan.json", dict(jobs=[]))
            with self.assertRaisesRegex(ValueError, "changed"):
                predecessor_status(plan)

    def test_wait_precedes_predictor_and_every_training_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "queue").mkdir()
            write(root / "queue/status.json", dict(stage="prepared", jobs={}))
            write(root / "plan.json", dict(poll_seconds=30))
            with patch.object(pipeline, "validate_runtime"), \
                    patch.object(pipeline, "predecessor_status",
                                 return_value=dict(ready=False, jobs={})), \
                    patch.object(pipeline.time, "sleep", side_effect=InterruptedError), \
                    patch.object(pipeline, "materialize") as materialize, \
                    patch.object(pipeline, "check") as preflight, \
                    patch.object(pipeline, "execute") as execute:
                with self.assertRaises(InterruptedError):
                    pipeline.run(root)
                materialize.assert_not_called()
                preflight.assert_not_called()
                execute.assert_not_called()
            self.assertEqual(read(root / "queue/status.json")["stage"],
                             "waiting_for_predecessor")

    def test_comparison_includes_new_original_and_never_fabricates_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = job_matrix(22)
            plan = dict(jobs=jobs, controls={j["dataset_key"]: dict(env=j["env"])
                                           for j in jobs})
            empty = snapshot(root, plan)
            self.assertEqual(empty["rows"], [])
            self.assertEqual(len(empty["best_rows"]), 6)
            self.assertTrue(all(row[2] is None for row in empty["best_rows"]))
            work = root / jobs[0]["id"] / "training"
            work.mkdir(parents=True)
            (work / "console.log").write_text(
                "Time steps: 50000\nEvaluation over 100 episodes: 0.4 , "
                "D4RL score: 40.0\nTime steps: 100000\n"
                "Evaluation over 100 episodes: 0.2 , D4RL score: 20.0\n")
            result = snapshot(root, plan)
            self.assertEqual(len(result["rows"]), 2)
            self.assertEqual(result["best_rows"][0][1:4], ["original", 40.0, 50000])


if __name__ == "__main__":
    unittest.main()
