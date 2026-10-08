"""Matched rewards, historical preservation, shared-GPU admission and upload repair."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from scripts.cql_delayed.common import digest, read, verify, write
from scripts.cql_repeats_5090.queue import validate_job
from scripts.cql_uniform_5090.prepare import (
    ARM,
    BASELINE,
    CAMPAIGN,
    prepare,
    SHAPLEY,
    uniform_rewards,
)
from scripts.cql_uniform_5090.queue import admissible, execute
from scripts.cql_uniform_5090.sync import missing_rows, remote_history

def dataset(n=3):
    total = np.resize(np.array([-123.456789, 0, 4321.1234567]), n)
    states = np.arange(n * 1000 * 2, dtype=np.float32).reshape(n * 1000, 2)
    actions = np.ones((n * 1000, 1), dtype=np.float32)
    ends = np.arange(999, n * 1000, 1000)
    terminals = np.zeros(n * 1000, dtype=np.float32)
    terminals[ends] = 1
    rewards = np.zeros(n * 1000, dtype=np.float32)
    rewards[ends] = total.astype(np.float32)
    data = dict(observations=states, actions=actions, next_observations=states.copy(),
                rewards=rewards, terminals=terminals)
    predictor = dict(features=np.concatenate([states, actions], axis=1)
                     .reshape(n, 1000, 3),
                     returns=total, starts=ends - 999, ends=ends)
    return data, predictor


class RewardTests(unittest.TestCase):
    def test_uniform_signed_returns_and_only_float_rounding_at_end(self):
        baseline, predictor = dataset()
        before = {k: v.copy() for k, v in baseline.items()}
        data, audit = uniform_rewards(baseline, predictor)
        rewards = data["rewards"].reshape(3, 1000)
        for i, total in enumerate(predictor["returns"]):
            np.testing.assert_array_equal(rewards[i, :-1],
                                          np.full(999, total / 1000, dtype=np.float32))
            self.assertAlmostEqual(rewards[i].astype(np.float64).sum(), total, places=5)
        self.assertLess(rewards[0].max(), 0)
        self.assertEqual(np.count_nonzero(rewards[1]), 0)
        for key in baseline:
            np.testing.assert_array_equal(baseline[key], before[key])
            if key != "rewards":
                np.testing.assert_array_equal(data[key], baseline[key])
        self.assertFalse(audit["original_dense_rewards_used"])
        self.assertFalse(audit["predictor_fitted"])

    def test_wrong_labels_features_boundaries_or_nonterminal_rewards_rejected(self):
        for kind in ["label", "feature", "boundary", "dense", "nan", "incomplete"]:
            with self.subTest(kind=kind):
                baseline, predictor = dataset()
                if kind == "label":
                    predictor["returns"][0] += 1
                elif kind == "feature":
                    predictor["features"][0, 0, 0] += 1
                elif kind == "boundary":
                    predictor["starts"][0] += 1
                elif kind == "dense":
                    baseline["rewards"][0] = 1
                elif kind == "nan":
                    predictor["returns"][0] = float("nan")
                else:
                    baseline["terminals"][-1] = 0
                with self.assertRaises((ValueError, AssertionError)):
                    uniform_rewards(baseline, predictor)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.previous = self.project / "results" / BASELINE
        source = self.previous / "source"
        source.mkdir(parents=True)
        (source / "frozen.py").write_text("# unchanged CQL algorithm and trainer\n")
        self.config = dict(env="halfcheetah-medium-replay-v2", seed=1,
                           max_timesteps=1000000, normalize_reward=False,
                           eval_freq=5000, n_episodes=10, batch_size=256, discount=0.99,
                           name="old", group="old")
        (source / "config.yaml").write_text(yaml.safe_dump(self.config))
        data, predictor = dataset(202)
        (self.previous / ARM).mkdir()
        np.savez(self.previous / ARM / "dataset.npz", **data)
        write(self.previous / ARM / "audit.json", dict(episodes=202))
        spec = dict(env=self.config["env"], seed=1, updates=1000000,
                    config="config.yaml", eval_every=5000, eval_episodes=10,
                    wandb_id="old", data_sha256={
                        name: digest(self.previous / ARM / name)
                        for name in ["dataset.npz", "audit.json"]})
        write(self.previous / "protocol.json", dict(
            runs={ARM: spec}, retained=dict(batch_size=256),
            source_sha256={name: digest(source / name)
                           for name in ["frozen.py", "config.yaml"]},
            wandb_entity="test", wandb_project="test"))
        self.input = self.project / "input.npz"
        np.savez(self.input, **predictor)
        shapley = self.project / "results" / SHAPLEY
        shapley.mkdir()
        write(shapley / "protocol.json", dict(
            source_sha256={}, shapley_gate=dict(passed=True),
            original_preparation_sha256={"predictor/input.npz": digest(self.input)},
            runs=dict(shapley=dict(wandb_id="shapley-1"))))
        for seed in (11, 12):
            root = (self.project / "results/cql-repeats-seeds11-12-100k-5090-20261007"
                    / f"seed{seed}-medium_replay-shapley")
            root.mkdir(parents=True)
            write(root / "protocol.json", dict(source_sha256={}, runs=dict(
                shapley=dict(seed=seed, updates=100000, wandb_id=f"shapley-{seed}"))))
        self.root = self.project / "results" / CAMPAIGN

    def test_matched_three_seeds_preserve_historical_scientific_code_and_data(self):
        before = {str(p): digest(p) for p in self.project.rglob("*") if p.is_file()}
        prepare(self.project, self.root, self.input, "test-revision")
        plan = read(self.root / "plan.json")
        self.assertEqual(plan["seeds"], [1, 11, 12])
        self.assertEqual(plan["updates"], 100000)
        self.assertEqual(len({job["wandb_id"] for job in plan["jobs"]}), 3)
        reward_hashes = set()
        for job in plan["jobs"]:
            root = validate_job(self.root, job)
            protocol = verify(root)
            verify(root / "validation")
            config = yaml.safe_load((root / "source/config.yaml").read_text())
            changed = {k for k in config if config[k] != self.config[k]}
            self.assertTrue(changed <= {"seed", "max_timesteps", "name", "group"})
            self.assertEqual(config["seed"], job["seed"])
            self.assertEqual(config["max_timesteps"], 100000)
            self.assertEqual(digest(root / "source/frozen.py"),
                             digest(self.previous / "source/frozen.py"))
            spec = protocol["runs"][ARM]
            self.assertEqual(spec["paired_shapley_wandb_id"], f"shapley-{job['seed']}")
            reward_hashes.add(spec["data_sha256"]["dataset.npz"])
        self.assertEqual(len(reward_hashes), 1)
        for path, expected in before.items():
            self.assertEqual(digest(path), expected)
        with self.assertRaisesRegex(ValueError, "reserved"):
            prepare(self.project, self.root, self.input, "test-revision")

    def test_wrong_input_or_corrupted_baseline_rejected_before_new_campaign(self):
        self.input.write_bytes(b"wrong predictor labels")
        with self.assertRaisesRegex(ValueError, "labels differ"):
            prepare(self.project, self.root, self.input, "test-revision")
        self.assertFalse(self.root.exists())

    def test_queue_waits_then_launches_once_and_waits_for_restarted_observer(self):
        prepare(self.project, self.root, self.input, "test-revision")
        training, observers, preflights = [], [], []
        ticks = []

        def smoke(command, **kwargs):
            root = Path(command[command.index("--root") + 1])
            work = root / ARM / "preflight"
            work.mkdir()
            write(work / "status.json", dict(status="completed", completed_updates=100,
                                             final_normalized_score=1.))
            write(work / "runtime.json", dict(device="cuda:0"))
            preflights.append(root)

        def spawn(command, **kwargs):
            root = Path(command[command.index("--root") + 1])
            process = SimpleNamespace(pid=1000 + len(training) + len(observers),
                                      returncode=None, root=root)
            process.poll = lambda: process.returncode
            if "-m" in command:
                observers.append(process)
                (root / "wandb_sync" / ARM).mkdir(parents=True, exist_ok=True)
            else:
                training.append(process)
                work = root / ARM / "training"
                work.mkdir()
                write(work / "status.json", dict(status="training", completed_updates=0))
            return process

        def sleep(seconds):
            ticks.append(seconds)
            self.assertLess(len(ticks), 20, "Queue stopped making progress")
            if len(ticks) == 2:
                observers[0].returncode = 1
            if len(training) == 3:
                for process in training:
                    process.returncode = 0
                    write(process.root / ARM / "training/status.json",
                          dict(status="completed", completed_updates=100000))
                if len(ticks) >= 5:
                    for process in observers:
                        if process.returncode is None:
                            process.returncode = 0
                            write(process.root / "wandb_sync" / ARM
                                  / "full_completion_verification.json",
                                  dict(verified=True))

        def probe():
            return dict(gpu_pids=list(range(6)), free_gpu_mib=15000,
                        free_host_mib=32000, utilization=99 if not ticks else 60)

        target = "scripts.cql_uniform_5090.queue."
        with patch(target + "probe", side_effect=probe), \
                patch(target + "time.sleep", side_effect=sleep), \
                patch(target + "subprocess.run", side_effect=smoke), \
                patch(target + "subprocess.Popen", side_effect=spawn):
            execute(self.root)
        self.assertEqual(len(training), 3)
        self.assertEqual(len(preflights), 3)
        self.assertEqual(len(observers), 4)
        status = read(self.root / "queue/status.json")
        self.assertEqual(status["stage"], "completed")
        self.assertEqual({job["state"] for job in status["jobs"].values()},
                         {"completed"})


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(max_parallel=3, max_gpu_processes=9,
                         gpu_memory_reserve_mib=4096, gpu_memory_per_job_mib=3072,
                         host_memory_reserve_mib=8192, host_memory_per_job_mib=4096,
                         max_admission_utilization=85)
        self.resource = dict(gpu_pids=list(range(6)), free_gpu_mib=15000,
                             free_host_mib=32000, utilization=60)

    def test_shared_gpu_admission_and_limits(self):
        self.assertTrue(admissible(self.plan, self.resource, []))
        self.assertTrue(admissible(self.plan, self.resource, [7, 8]))
        self.assertFalse(admissible(self.plan, self.resource, [7, 8, 9]))
        for changes in [dict(gpu_pids=list(range(9))), dict(free_gpu_mib=7000),
                        dict(free_host_mib=12000), dict(utilization=86)]:
            self.assertFalse(admissible(self.plan, dict(self.resource, **changes), []))

    def test_pending_cuda_initialization_reserves_memory(self):
        resource = dict(self.resource, free_gpu_mib=9000)
        self.assertTrue(admissible(self.plan, resource, []))
        self.assertFalse(admissible(self.plan, resource, [7]))
        resource["gpu_pids"].append(7)
        self.assertTrue(admissible(self.plan, resource, [7]))


class SyncTests(unittest.TestCase):
    def test_null_api_rows_and_lost_final_upload_repaired_selectively(self):
        class Remote:
            def scan_history(self, keys, page_size):
                column = keys[-1]
                return [dict(completed_updates=99900, **{column: None}),
                        dict(completed_updates=99800, **{column: 3.0})]

        train, evaluations = remote_history(Remote())
        self.assertEqual(train, {99800: 3.0})
        local = {99800: {"train/alpha": 3.0},
                 99900: {"train/alpha": 2.0},
                 100000: {"train/alpha": 1.0, "eval/normalized_score": 44.0,
                          "d4rl_normalized_score": 44.0}}
        before = copy.deepcopy(local)
        missing = missing_rows(local, train, evaluations)
        self.assertEqual(set(missing), {99900, 100000})
        self.assertEqual(local, before)
        self.assertEqual(missing_rows(local, {99800: 3., 99900: 2., 100000: 1.},
                                      {100000: 44.}), {})


if __name__ == "__main__":
    unittest.main()
