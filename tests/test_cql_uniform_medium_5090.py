"""HCM population, 500k budgets, predecessor gating and matched-budget summaries."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from scripts.cql_delayed.common import digest, read, verify, write
from scripts.cql_repeats_5090.queue import validate_job
from scripts.cql_uniform_medium_5090.prepare import (
    ARM,
    BASELINE,
    CAMPAIGN,
    PREDECESSOR,
    prepare,
    SHAPLEY,
)
from scripts.cql_uniform_medium_5090.queue import execute, predecessor_ready
from scripts.cql_uniform_medium_5090.sync import budget_summaries

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


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.previous = self.project / "results" / BASELINE
        source = self.previous / "source"
        source.mkdir(parents=True)
        (source / "frozen.py").write_text("# unchanged CQL algorithm and trainer\n")
        self.config = dict(env="halfcheetah-medium-v2", seed=1,
                           max_timesteps=1000000, normalize_reward=False,
                           eval_freq=5000, n_episodes=10, batch_size=256, discount=0.99,
                           name="old", group="old")
        (source / "config.yaml").write_text(yaml.safe_dump(self.config))
        data, predictor = dataset(1000)
        (self.previous / ARM).mkdir()
        np.savez(self.previous / ARM / "dataset.npz", **data)
        write(self.previous / ARM / "audit.json", dict(episodes=1000))
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
            preparation_sha256={"predictor/input.npz": digest(self.input)},
            runs=dict(shapley=dict(wandb_id="shapley-1"))))
        for seed in (11, 12):
            root = (self.project / "results/cql-repeats-seeds11-12-100k-5090-20261007"
                    / f"seed{seed}-medium-shapley")
            root.mkdir(parents=True)
            write(root / "protocol.json", dict(source_sha256={}, runs=dict(
                shapley=dict(seed=seed, updates=100000, wandb_id=f"shapley-{seed}"))))
        predecessor = self.project / "results" / PREDECESSOR
        predecessor.mkdir()
        write(predecessor / "plan.json", dict(
            seeds=[1, 11, 12], updates=100000,
            jobs=[dict(arm="medium_replay") for _ in range(3)]))
        self.root = self.project / "results" / CAMPAIGN

    def test_matched_three_seeds_preserve_historical_scientific_code_and_data(self):
        before = {str(p): digest(p) for p in self.project.rglob("*") if p.is_file()}
        prepare(self.project, self.root, self.input, "test-revision")
        plan = read(self.root / "plan.json")
        self.assertEqual(plan["seeds"], [1, 11, 12])
        self.assertEqual(plan["updates"], 500000)
        self.assertEqual(plan["predecessor"]["plan_sha256"], digest(
            self.project / "results" / PREDECESSOR / "plan.json"))
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
            self.assertEqual(config["max_timesteps"], 500000)
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

    def test_wrong_hcm_population_rejected_before_new_campaign(self):
        data, predictor = dataset(202)
        np.savez(self.previous / ARM / "dataset.npz", **data)
        np.savez(self.input, **predictor)
        protocol = read(self.previous / "protocol.json")
        protocol["runs"][ARM]["data_sha256"]["dataset.npz"] = digest(
            self.previous / ARM / "dataset.npz")
        write(self.previous / "protocol.json", protocol)
        path = self.project / "results" / SHAPLEY / "protocol.json"
        protocol = read(path)
        protocol["preparation_sha256"]["predictor/input.npz"] = digest(self.input)
        write(path, protocol)
        with self.assertRaisesRegex(ValueError, "1000 full trajectories"):
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
                self.assertIn("scripts.cql_uniform_medium_5090.sync", command)
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
            if len(ticks) <= 2:
                self.assertFalse(training)
                self.assertFalse(preflights)
            if len(ticks) == 4:
                observers[0].returncode = 1
            if len(training) == 3:
                for process in training:
                    process.returncode = 0
                    write(process.root / ARM / "training/status.json",
                          dict(status="completed", completed_updates=500000))
                if len(ticks) >= 7:
                    for process in observers:
                        if process.returncode is None:
                            process.returncode = 0
                            write(process.root / "wandb_sync" / ARM
                                  / "full_completion_verification.json",
                                  dict(verified=True))

        def probe():
            return dict(gpu_pids=list(range(6)), free_gpu_mib=15000,
                        free_host_mib=32000, utilization=99 if len(ticks) == 2 else 60)

        target = "scripts.cql_uniform_medium_5090.queue."
        with patch(target + "predecessor_ready", side_effect=[False, False, True]), \
                patch(target + "probe", side_effect=probe), \
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

class PredecessorTests(unittest.TestCase):
    def test_requires_success_and_matching_wandb_receipt_and_frozen_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = root / "seed1"
            (job / "medium_replay/training").mkdir(parents=True)
            (job / "wandb_sync/medium_replay").mkdir(parents=True)
            protocol = job / "protocol.json"
            write(protocol, dict(source_sha256={}, runs=dict(
                medium_replay=dict(wandb_id="prior-1"))))
            write(root / "plan.json", dict(updates=100000, jobs=[dict(
                id="seed1", directory="seed1", arm="medium_replay",
                protocol_sha256=digest(protocol))]))
            plan = dict(predecessor=dict(root=str(root),
                                        plan_sha256=digest(root / "plan.json")))
            (root / "queue").mkdir()
            self.assertFalse(predecessor_ready(plan))
            for stage in ("running", "waiting_for_wandb", "finished_with_failures"):
                write(root / "queue/status.json", dict(stage=stage))
                self.assertFalse(predecessor_ready(plan))
            write(root / "queue/status.json", dict(stage="completed", jobs=dict(
                seed1=dict(state="completed", exit=0))))
            write(job / "medium_replay/training/status.json",
                  dict(status="completed", completed_updates=100000))
            self.assertFalse(predecessor_ready(plan))
            receipt = job / "wandb_sync/medium_replay/full_completion_verification.json"
            for values in [dict(id="wrong"), dict(updates=50000), dict(verified=False),
                           dict(state="running")]:
                write(receipt, dict(dict(verified=True, id="prior-1", state="finished",
                                         updates=100000), **values))
                self.assertFalse(predecessor_ready(plan))
            write(receipt, dict(verified=True, id="prior-1", state="finished",
                                updates=100000))
            self.assertTrue(predecessor_ready(plan))
            write(root / "plan.json", dict(updates=500000, jobs=[]))
            with self.assertRaisesRegex(ValueError, "plan changed"):
                predecessor_ready(plan)


class BudgetSummaryTests(unittest.TestCase):
    def test_later_scores_do_not_change_100k_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            for step in range(5000, 500001, 5000):
                score = 90 if step == 5000 else step / 5000
                write(work / f"eval_{step:07d}.json", dict(normalized_score=score))
            summaries = budget_summaries(work, 5000, 500000)
            self.assertEqual(summaries["result/100k/final_normalized_score"], 20)
            self.assertEqual(summaries["result/100k/best_normalized_score"], 90)
            self.assertEqual(summaries["result/100k/last10_mean"], 15.5)
            self.assertEqual(summaries["result/500k/final_normalized_score"], 100)
            self.assertEqual(summaries["result/500k/best_normalized_score"], 100)
            self.assertEqual(summaries["result/500k/last10_mean"], 95.5)
            (work / "eval_0100000.json").unlink()
            with self.assertRaises(FileNotFoundError):
                budget_summaries(work, 5000, 500000)


if __name__ == "__main__":
    unittest.main()
