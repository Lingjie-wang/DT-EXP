"""Protect matched repeat inputs and scheduling around existing GPU users."""

import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts.cql_delayed.common import digest, read, verify, write
from scripts.cql_repeats_5090.prepare import clone_run, prepare
from scripts.cql_repeats_5090.queue import capacity, execute

class RepeatTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.previous = self.base / "pilot"
        self.root = self.base / "repeat"
        (self.previous / "source").mkdir(parents=True)
        (self.previous / "source/algorithm.py").write_text("# frozen CQL\n")
        config = dict(seed=1, max_timesteps=1000000, eval_freq=5000, n_episodes=10,
                      name="old", group="old", batch_size=256, discount=0.99)
        (self.previous / "source/config.yaml").write_text(yaml.safe_dump(config))
        data = self.previous / "shapley"
        data.mkdir()
        (data / "dataset.npz").write_bytes(b"unchanged predictive rewards")
        write(data / "audit.json", {"episodes": 1000})
        self.protocol = dict(
            source_sha256={p.name: digest(p)
                           for p in (self.previous / "source").iterdir()},
            runs=dict(shapley=dict(
                seed=1, updates=1000000, config="config.yaml", wandb_id="pilot",
                wandb_name="CQL-CORL-HCM-shapley20x128-v1-seed1-1M-5090",
                data_sha256={p.name: digest(p) for p in data.iterdir()})),
            changes=[], shapley_gate=dict(passed=True))
        write(self.previous / "protocol.json", self.protocol)

    def test_only_seed_budget_logging_change_and_originals_remain(self):
        original = {str(p): p.read_bytes() for p in self.previous.rglob("*")
                    if p.is_file()}
        spec = clone_run(self.previous, self.root, "shapley", 11, "repeats")
        verify(self.root)
        verify(self.root / "validation")
        old = yaml.safe_load((self.previous / "source/config.yaml").read_text())
        new = yaml.safe_load((self.root / "source/config.yaml").read_text())
        self.assertEqual({k for k in old if old[k] != new[k]},
                         {"seed", "max_timesteps", "name", "group"})
        self.assertEqual(spec["updates"], 300000)
        self.assertEqual(spec["seed"], 11)
        for name, expected in spec["data_sha256"].items():
            self.assertEqual(digest(self.root / "shapley" / name), expected)
            self.assertEqual(digest(self.root / "validation/shapley" / name), expected)
        self.assertEqual(original, {str(p): p.read_bytes()
                                    for p in self.previous.rglob("*") if p.is_file()})
        with self.assertRaises(FileExistsError):
            clone_run(self.previous, self.root, "shapley", 11, "repeats")

    def test_seeds_distinct_ids_and_forbidden_historical_seeds(self):
        a = clone_run(self.previous, self.root, "shapley", 11, "repeats")
        b = clone_run(self.previous, self.base / "repeat12", "shapley", 12, "repeats")
        self.assertNotEqual(a["wandb_id"], b["wandb_id"])
        for seed in (0, 1, 2, 42):
            with self.assertRaises(ValueError):
                clone_run(self.previous, self.base / str(seed), "shapley", seed, "g")

    def test_corrupt_rewards_failed_gate_and_cancelled_arm_refused(self):
        self.protocol["shapley_gate"]["passed"] = False
        write(self.previous / "protocol.json", self.protocol)
        with self.assertRaisesRegex(ValueError, "gate"):
            clone_run(self.previous, self.root, "shapley", 11, "g")
        self.protocol["shapley_gate"]["passed"] = True
        write(self.previous / "protocol.json", self.protocol)
        marker = self.previous / "shapley/user_cancellation.json"
        write(marker, {})
        with self.assertRaisesRegex(ValueError, "cancelled"):
            clone_run(self.previous, self.root, "shapley", 11, "g")
        marker.unlink()
        (self.previous / "shapley/dataset.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "data changed"):
            clone_run(self.previous, self.root, "shapley", 11, "g")
        self.assertFalse(self.root.exists())

    def test_existing_seed_reservation_prevents_campaign_creation(self):
        reserved = self.base / "results/other/seed11"
        reserved.mkdir(parents=True)
        write(reserved / "protocol.json", dict(runs={
            "medium": dict(seed=11, env="halfcheetah-medium-v2")}))
        with self.assertRaisesRegex(ValueError, "already reserved"):
            prepare(self.base, self.root)
        self.assertFalse(self.root.exists())


PLAN = dict(max_gpu_processes=4, gpu_memory_per_job_mib=3072,
            gpu_memory_reserve_mib=2048, host_memory_reserve_mib=8192,
            poll_seconds=0.01, updates=300000)


class QueueTests(unittest.TestCase):
    def test_counts_external_jobs_even_with_plenty_of_free_vram(self):
        self.assertFalse(capacity(PLAN, [1, 2, 3, 4], [], 22000, 100000))
        self.assertFalse(capacity(PLAN, [1, 2, 3], [], 22000, 100000))
        self.assertTrue(capacity(PLAN, [1, 2], [], 22000, 100000))

    def test_pending_cuda_initialization_and_memory_limits(self):
        self.assertFalse(capacity(PLAN, [1, 2], [3], 22000, 100000))
        self.assertTrue(capacity(PLAN, [1, 2], [1, 2], 22000, 100000))
        self.assertFalse(capacity(PLAN, [], [], 8000, 100000))
        self.assertFalse(capacity(PLAN, [], [], 22000, 8000))

    def test_queue_waits_then_preflights_and_runs_pair_without_reruns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = []
            for arm in ("medium", "shapley"):
                directory = root / arm
                script = directory / "source/scripts/fake/train.py"
                script.parent.mkdir(parents=True)
                script.write_text(
                    "import argparse,json\nfrom pathlib import Path\n"
                    "p=argparse.ArgumentParser()\np.add_argument('--root')\n"
                    "p.add_argument('--arm')\n"
                    "p.add_argument('--smoke',action='store_true')\na=p.parse_args()\n"
                    "w=Path(a.root)/a.arm/('preflight' if a.smoke else 'training')\n"
                    "w.mkdir(parents=True,exist_ok=False)\n"
                    "(w/'runtime.json').write_text(json.dumps({'device':'cuda:0'}))\n"
                    "(w/'status.json').write_text(json.dumps({'status':'completed',"
                    "'completed_updates':100 if a.smoke else 300000,"
                    "'final_normalized_score':1.0}))\n")
                (directory / "validation").mkdir()
                jobs.append(dict(id=arm, directory=arm, arm=arm, runner="fake",
                                 pair="paired", seed=11))
            write(root / "plan.json", dict(PLAN, jobs=jobs))
            real_popen = subprocess.Popen
            launched = []
            probe_calls = []

            def probe():
                probe_calls.append(True)
                count = len(probe_calls)
                pids = {1: [1, 2, 3, 4], 3: [1, 2, 3]}.get(count, [1, 2])
                return dict(gpu_pids=pids,
                            free_gpu_mib=22000, free_host_mib=100000)

            def popen(command, **kwargs):
                if "scripts.cql_5090.observe" in command:
                    return mock.Mock(pid=999, poll=lambda: None)
                self.assertGreater(len(probe_calls), 1)
                launched.append(command)
                return real_popen(command, **kwargs)

            with mock.patch("scripts.cql_repeats_5090.queue.resources",
                            side_effect=probe), \
                    mock.patch("scripts.cql_repeats_5090.queue.validate_job",
                               side_effect=lambda r, j: r / j["directory"]), \
                    mock.patch("scripts.cql_repeats_5090.queue.subprocess.Popen",
                               side_effect=popen):
                started = time.monotonic()
                execute(root)
                self.assertLess(time.monotonic() - started, 10)
            self.assertEqual(len(launched), 4)  # two smoke runs, two formal runs
            self.assertEqual(read(root / "queue/status.json")["stage"],
                             "training_completed")
            with self.assertRaises(FileExistsError):
                execute(root)


if __name__ == "__main__":
    unittest.main()
