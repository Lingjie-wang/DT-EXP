import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from algorithms.offline.shapley_redistribution import SegmentReturnModel
from scripts.cql_antmaze_official_5090.common import admissible, read, write
from scripts.cql_antmaze_success_5090.compare import snapshot
from scripts.cql_antmaze_success_5090.data import (
    conserved,
    grouped_folds,
    inputs,
    lengths,
    masks,
    partition,
    uniform,
)
from scripts.cql_antmaze_success_5090.fit import fit, SETTINGS
from scripts.cql_antmaze_success_5090.queue import command, environment

class SuccessBoundaryTest(unittest.TestCase):
    def test_official_filter_success_timeout_overlap_and_tail(self):
        raw = dict(rewards=np.array([0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=float),
                   terminals=np.array([0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=float),
                   timeouts=np.array([0, 0, 0, 1, 0, 1, 0, 0, 0], dtype=float))
        p = partition(raw)
        np.testing.assert_array_equal(p["index"], [0, 1, 2, 4, 6, 7])
        np.testing.assert_array_equal(p["offsets"], [0, 2, 3, 4, 6])
        np.testing.assert_array_equal(p["groups"], [0, 0, 1, 2])
        np.testing.assert_array_equal(p["returns"], [1, 1, 0, 0])
        self.assertEqual(p["skipped_empty_fragments"], 1)
        np.testing.assert_allclose(uniform(p), [.5, .5, 1, 0, 0, 0])

    def test_randomized_official_transition_coverage(self):
        rng = np.random.default_rng(21)
        for _ in range(50):
            terminals, timeouts = rng.random((2, 100)) < .2
            p = partition(dict(rewards=terminals.astype(float), terminals=terminals,
                               timeouts=timeouts))
            expected = np.flatnonzero(~timeouts[:-1])
            np.testing.assert_array_equal(p["index"], expected)
            self.assertTrue(np.all(np.diff(p["offsets"]) > 0))
            for a, b, raw_bound in zip(p["offsets"][:-1], p["offsets"][1:],
                                      p["raw_bounds"]):
                selected = expected[a:b]
                self.assertTrue(np.all((selected >= raw_bound[0])
                                       & (selected <= raw_bound[1])))
            sums = np.add.reduceat(uniform(p).astype(float), p["offsets"][:-1])
            np.testing.assert_allclose(sums, p["returns"], atol=1e-4)

    def test_variable_length_conservation_and_singleton(self):
        rng = np.random.default_rng(8)
        for horizon in [1, 2, 19, 20, 21, 137, 700, 1000]:
            for total in [0, 1]:
                phi = rng.normal(size=len(lengths(horizon)))
                rewards, correction, rounding = conserved(phi, total, horizon)
                self.assertLess(abs(rewards.astype(float).sum() - total), 1e-4)
                expected = np.repeat(phi / lengths(horizon), lengths(horizon))
                expected += correction / horizon
                np.testing.assert_allclose(rewards[:-1], expected[:-1], atol=1e-6)
                self.assertLess(abs(rounding), 1e-4)
                if horizon == 1:
                    self.assertEqual(float(rewards[0]), total)

    def test_group_folds_do_not_leak_parent_rollouts(self):
        groups = np.repeat(np.arange(30), np.arange(30) % 4 + 1)
        held = []
        for split in grouped_folds(groups):
            sets = [set(groups[split[key]]) for key in ["train", "validation", "held"]]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertEqual(set.union(*sets), set(groups))
            held += split["held"]
        self.assertEqual(sorted(held), list(range(len(groups))))

    def test_padding_masks_and_training_only_statistics(self):
        offsets = np.array([0, 1, 4, 29])
        features = np.arange(58, dtype=np.float32).reshape(-1, 2)
        returns = np.array([1., 0., 1.])
        x, valid, stats = inputs(features, offsets, returns, [0, 1])
        changed = features.copy()
        changed[4:] = 1e6
        y, _, other = inputs(changed, offsets, returns * [1, 1, 10000], [0, 1])
        np.testing.assert_array_equal(x[:2], y[:2])
        np.testing.assert_array_equal(stats["mean"], other["mean"])
        self.assertEqual(stats["target_mean"], other["target_mean"])
        self.assertTrue(np.all(x[valid == 0] == 0))
        for _ in range(10):
            mask = masks(np.random.default_rng(_), valid)
            self.assertTrue(np.all(mask <= valid))
            self.assertEqual(mask[0].sum(), 1)
            self.assertTrue(np.all(mask.sum(1) > 0))
        model = SegmentReturnModel(x.shape[-1])
        altered = x.copy()
        altered[valid == 0] = 1000
        with torch.no_grad():
            a = model(torch.from_numpy(x), torch.from_numpy(valid))
            b = model(torch.from_numpy(altered), torch.from_numpy(valid))
        torch.testing.assert_close(a, b)

    def test_small_cross_fit_covers_every_fragment_and_conserves_returns(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(4)
        lens = np.tile([1, 3, 23], 10)
        offsets = np.r_[0, np.cumsum(lens)]
        part = dict(offsets=offsets, returns=np.tile([1., 0., 1.], 10),
                    groups=np.repeat(np.arange(10), 3))
        features = rng.normal(size=(offsets[-1], 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "fit"
            settings = dict(SETTINGS, epochs=2, patience=1, permutations=8, batch=8)
            result = fit(features, part, out, "cpu", settings)
            totals = np.add.reduceat(result.astype(float), offsets[:-1])
            np.testing.assert_allclose(totals, part["returns"], atol=1e-4)
            self.assertEqual(read(out / "status.json")["stage"], "completed")
            self.assertEqual(len(read(out / "diagnostics.json")), len(lens))
            self.assertEqual(read(out / "gate.json")["policy"], "report_only")

    def test_official_cli_does_not_override_training_hyperparameters(self):
        root = Path("/tmp/campaign")
        job = dict(id="umaze-uniform-seed21", seed=21, config="config.yaml",
                   dataset_dir="reward_datasets/uniform")
        plan = dict(wandb_project="CORL-DDR", wandb_entity="entity", jobs=[job])
        argv = command(root, plan, job, "training")
        for option in ["--max_timesteps", "--eval_freq", "--n_episodes", "--batch_size",
                       "--normalize_reward", "--discount"]:
            self.assertNotIn(option, argv)
        env = environment(root, plan, root / job["id"] / "training", "training")
        self.assertEqual(env["D4RL_DATASET_DIR"],
                         "/tmp/campaign/reward_datasets/uniform")
        self.assertEqual(env["WANDB_MODE"], "online")

    def test_shared_gpu_admission_counts_other_jobs(self):
        plan = dict(max_parallel=2, max_gpu_processes=8, gpu_memory_reserve_mib=4096,
                    gpu_memory_per_job_mib=5120, host_memory_reserve_mib=8192,
                    host_memory_per_job_mib=6144, max_admission_utilization=85)
        resource = dict(gpu_pids=list(range(6)), free_gpu_mib=22000,
                        free_host_mib=35000, utilization=74)
        self.assertTrue(admissible(plan, resource, []))
        self.assertTrue(admissible(plan, resource, [10]))
        self.assertFalse(admissible(plan, resource, [10, 11]))
        self.assertFalse(admissible(plan, dict(resource, gpu_pids=list(range(8))), []))
        self.assertFalse(admissible(plan, dict(resource, utilization=90), []))

    def test_comparison_pending_arms_have_no_invented_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "baseline.json", [dict(completed_updates=50000, episodes=100,
                                                normalized_score=25)])
            plan = dict(controls={"umaze": dict(env="antmaze-umaze-v2",
                        evaluations="baseline.json", url="https://example.org/original")},
                        jobs=[dict(id=arm, arm=arm, dataset_key="umaze")
                              for arm in ["uniform", "shapley"]])
            result = snapshot(root, plan)
            self.assertEqual(len(result["rows"]), 1)
            self.assertEqual(result["curves"]["umaze"][1:], [([], []), ([], [])])
            self.assertEqual(result["best_rows"][0][2], 25)
            self.assertIsNone(result["best_rows"][1][2])


if __name__ == "__main__":
    unittest.main()
