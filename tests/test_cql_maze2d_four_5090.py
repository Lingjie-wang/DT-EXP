import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.cql_antmaze_official_5090.common import digest, read
from scripts.cql_antmaze_success_5090.data import partition, uniform
from scripts.cql_antmaze_success_5090.fit import fit, SETTINGS
from scripts.cql_maze2d_four_5090.compare import snapshot
from scripts.cql_maze2d_four_5090.materialize import delayed, predictor_input
from scripts.cql_maze2d_four_5090.online_check import score_matches
from scripts.cql_maze2d_four_5090.prepare import CONFIG, CONFIG_SHA256, job_matrix
from scripts.cql_maze2d_four_5090.queue import command, environment

class Maze2DFourTest(unittest.TestCase):
    def raw(self):
        return dict(rewards=np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1.]),
                    terminals=np.zeros(12),
                    timeouts=np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]),
                    observations=np.arange(48).reshape(12, 4),
                    actions=np.arange(24).reshape(12, 2))

    def test_delay_keeps_all_official_rows_and_only_moves_retained_returns(self):
        raw = self.raw()
        part = partition(raw)
        np.testing.assert_array_equal(part["index"], [0, 1, 2, 5, 6, 7, 8, 10])
        np.testing.assert_array_equal(part["offsets"], [0, 3, 7, 8])
        np.testing.assert_array_equal(part["returns"], [2, 0, 1])
        self.assertEqual(part["skipped_empty_fragments"], 1)
        np.testing.assert_array_equal(delayed(part), [0, 0, 2, 0, 0, 0, 0, 1])
        for rewards in [delayed(part), uniform(part)]:
            np.testing.assert_allclose(np.add.reduceat(rewards.astype(float),
                                       part["offsets"][:-1]), [2, 0, 1], atol=1e-4)
        np.testing.assert_array_equal(raw["terminals"], 0)

    def test_predictor_input_cannot_reveal_original_reward_positions(self):
        raw = self.raw()
        part = partition(raw)
        first = predictor_input(raw, part)
        # Move rewards within a fragment, retaining its return and all other fields.
        raw["rewards"][:3] = [0, 1, 1]
        other = predictor_input(raw, partition(raw))
        self.assertEqual(set(first), {"features", "offsets", "returns", "groups"})
        for key in first:
            np.testing.assert_array_equal(first[key], other[key])

    def test_all_four_arms_preserve_official_yaml_and_cli_scientific_arguments(self):
        source = Path(__file__).resolve().parents[1]
        self.assertEqual(digest(source / CONFIG), CONFIG_SHA256)
        jobs = job_matrix(23)
        self.assertEqual({j["arm"] for j in jobs},
                         {"original", "delayed", "uniform", "shapley"})
        self.assertEqual({(j["seed"], j["updates"], j["eval_every"], j["eval_episodes"])
                          for j in jobs}, {(23, 1000000, 5000, 100)})
        root = Path("/tmp/maze2d-campaign")
        plan = dict(jobs=jobs, wandb_project="project", wandb_entity="entity")
        for job in jobs:
            argv = command(root, plan, job, "training")
            keys = {arg for arg in argv if arg.startswith("--")}
            self.assertEqual(keys, {"--config_path", "--seed", "--device", "--project",
                                    "--group", "--name", "--checkpoints_path"})
            env = environment(root, plan, root / job["id"] / "training", "training")
            self.assertEqual(env["D4RL_DATASET_DIR"], str(root / job["dataset_dir"]))
            self.assertEqual(env["WANDB_MODE"], "online")

    def test_shapley_crossfit_conserves_multireward_and_zero_return_fragments(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(23)
        lengths = np.tile([21, 43, 22], 10)
        offsets = np.r_[0, np.cumsum(lengths)]
        part = dict(offsets=offsets, returns=np.tile([0., 8., 53.], 10),
                    groups=np.arange(30))
        features = rng.normal(size=(offsets[-1], 6)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "fit"
            settings = dict(SETTINGS, epochs=2, patience=1, permutations=8, batch=8)
            rewards = fit(features, part, out, "cpu", settings)
            totals = np.add.reduceat(rewards.astype(float), offsets[:-1])
            np.testing.assert_allclose(totals, part["returns"], atol=1e-4)
            self.assertEqual(read(out / "status.json")["stage"], "completed")

    def test_comparison_has_four_arms_and_only_completed_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = job_matrix(23)
            plan = dict(jobs=jobs, controls={"umaze": dict(env="maze2d-umaze-v1")})
            empty = snapshot(root, plan)
            self.assertEqual(empty["rows"], [])
            self.assertEqual(len(empty["best_rows"]), 4)
            work = root / jobs[1]["id"] / "training"
            work.mkdir(parents=True)
            (work / "console.log").write_text(
                "Time steps: 5000\nEvaluation over 100 episodes: 1.000 , "
                "D4RL score: -16.557\nTime steps: 10000\n")
            result = snapshot(root, plan)
            self.assertEqual(len(result["rows"]), 1)
            self.assertEqual(result["best_rows"][1][1:4], ["delayed", -16.557, 5000])
            self.assertEqual(result["curves"]["umaze"][0], ([], []))

    def test_remote_full_precision_matches_official_three_decimal_print(self):
        self.assertTrue(score_matches(-16.5568, -16.557))
        self.assertFalse(score_matches(-16.55, -16.557))
        self.assertFalse(score_matches(None, 0))
        self.assertFalse(score_matches(float("nan"), 0))


if __name__ == "__main__":
    unittest.main()
