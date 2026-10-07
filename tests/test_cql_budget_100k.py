"""Protect historical runs, both dependency plans and actual training budgets."""

import fcntl
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.cql_budget_100k.prepare import OLD_NAMES, prepare, require_pending
from scripts.cql_delayed.common import digest, read, verify, write
from scripts.cql_repeats_5090.queue import validate_job

class BudgetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.old_roots = [self.project / "results" / name for name in OLD_NAMES]
        for number, (root, count) in enumerate(zip(self.old_roots, [8, 6])):
            jobs = []
            for index in range(count):
                directory = f"run{index}"
                run = root / directory
                source = run / "source"
                source.mkdir(parents=True)
                (source / "train.py").write_text("# frozen CQL training\n")
                (source / "config.yaml").write_text(yaml.safe_dump(dict(
                    seed=11, max_timesteps=300000, name="CQL-300k-test", group="old",
                    eval_freq=5000, n_episodes=10, batch_size=256, discount=0.99)))
                data = run / "medium"
                data.mkdir()
                (data / "dataset.npz").write_bytes(b"unchanged frozen rewards")
                write(data / "audit.json", dict(matched=True))
                spec = dict(seed=11, updates=300000, config="config.yaml",
                            wandb_name="CQL-300k-test", wandb_id=f"old-{number}-{index}",
                            data_sha256={p.name: digest(p) for p in data.iterdir()})
                write(run / "protocol.json", dict(
                    runs={"medium": spec}, changes=[], migration={},
                    source_sha256={p.name: digest(p) for p in source.iterdir()}))
                jobs.append(dict(id=directory, directory=directory,
                                 arm="medium", seed=11,
                                 protocol_sha256=digest(run / "protocol.json")))
            plan = dict(updates=300000, jobs=jobs)
            if number:
                plan.update(predecessor=str(self.old_roots[0]),
                            predecessor_plan_sha256=digest(
                                self.old_roots[0] / "plan.json"))
            write(root / "plan.json", plan)
            status_dir = root / ("dependency" if number else "queue")
            status_dir.mkdir()
            write(status_dir / "status.json", dict(jobs={
                j["id"]: dict(state="queued") for j in jobs}))

    def test_replaces_all_14_without_changing_old_files_rewards_or_python(self):
        old_files = {p: p.read_bytes() for root in self.old_roots
                     for p in root.rglob("*") if p.is_file()}
        prepare(self.project, "published-commit")
        new_roots = [p.with_name(p.name.replace("-300k-", "-100k-"))
                     for p in self.old_roots]
        ids = []
        for root in new_roots:
            plan = read(root / "plan.json")
            self.assertEqual(plan["updates"], 100000)
            for job in plan["jobs"]:
                run = validate_job(root, job)
                p = verify(run / "validation")
                config = yaml.safe_load((run / "source/config.yaml").read_text())
                self.assertEqual(config["max_timesteps"], 100000)
                self.assertEqual(config["seed"], 11)
                self.assertEqual(config["eval_freq"], 5000)
                self.assertIn("-100k-", job["wandb_name"])
                ids.append(job["wandb_id"])
                for name, expected in p["runs"]["medium"]["data_sha256"].items():
                    self.assertEqual(digest(run / "medium" / name), expected)
                    self.assertEqual(digest(run / "validation/medium" / name), expected)
                self.assertEqual((run / "source/train.py").read_text(),
                                 "# frozen CQL training\n")
        self.assertEqual(len(set(ids)), 14)
        dense = read(new_roots[1] / "plan.json")
        self.assertEqual(dense["predecessor"], str(new_roots[0]))
        self.assertEqual(dense["predecessor_plan_sha256"],
                         digest(new_roots[0] / "plan.json"))
        for path, original in old_files.items():
            self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            prepare(self.project, "another-attempt")

    def test_rejects_live_manager_lock(self):
        with (self.old_roots[0] / "queue/lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                prepare(self.project, "test")

    def test_rejects_started_job_or_preflight_even_if_status_still_queued(self):
        root = self.old_roots[0]
        path = root / "queue/status.json"
        status = read(path)
        status["jobs"]["run0"]["state"] = "running"
        write(path, status)
        with self.assertRaises(ValueError):
            require_pending(root, "repeats")
        status["jobs"]["run0"]["state"] = "queued"
        write(path, status)
        (root / "run0/validation/medium/preflight").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already started"):
            prepare(self.project, "test")

    def test_validator_rejects_300k_yaml_even_with_matching_protocol_hashes(self):
        prepare(self.project, "test")
        root = self.old_roots[0].with_name(OLD_NAMES[0].replace("-300k-", "-100k-"))
        plan = read(root / "plan.json")
        job = plan["jobs"][0]
        run = root / job["directory"]
        config = yaml.safe_load((run / "source/config.yaml").read_text())
        config["max_timesteps"] = 300000
        (run / "source/config.yaml").write_text(yaml.safe_dump(config))
        protocol = read(run / "protocol.json")
        protocol["source_sha256"]["config.yaml"] = digest(run / "source/config.yaml")
        write(run / "protocol.json", protocol)
        job["protocol_sha256"] = digest(run / "protocol.json")
        with self.assertRaisesRegex(ValueError, "budget differs"):
            validate_job(root, job)


if __name__ == "__main__":
    unittest.main()
