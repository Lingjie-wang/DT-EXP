"""Exercise real process stopping and incomplete-save guards without a GPU."""

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.cql_delayed.common import digest, read, write
from scripts.cql_stop_500k.run import (
    bounded_rows,
    checkpoint,
    evaluation,
    finish_wandb,
    identity,
    result_summary,
    send,
    terminate,
    watch,
)

class StopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "medium/training"
        self.work.mkdir(parents=True)
        self.spool = self.root / "guard"
        self.spool.mkdir()
        self.target = 500000
        self.spec = dict(updates=1000000, eval_every=500000, eval_episodes=10)

    def sleeper(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def cleanup():
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)

        self.addCleanup(cleanup)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            found = identity(child.pid)
            if found and found["argv"][1:] == ["-c", "import time; time.sleep(60)"]:
                return child
            time.sleep(0.01)
        self.fail("Temporary test process did not finish starting")

    def saved(self, target=None):
        target = self.target if target is None else target
        trainer = {key: {} for key in [
            "actor", "critic1", "critic2", "critic1_target", "critic2_target",
            "critic_1_optimizer", "critic_2_optimizer", "actor_optim", "sac_log_alpha",
            "sac_log_alpha_optim", "cql_log_alpha", "cql_log_alpha_optim"]}
        trainer["total_it"] = target
        return dict(trainer=trainer, completed_updates=target,
                    state_mean=torch.zeros(1), state_std=torch.ones(1),
                    numpy_rng=None, python_rng=None, torch_rng=torch.get_rng_state(),
                    cuda_rng=[])

    def eval(self, target=None, score=42, episodes=10):
        target = self.target if target is None else target
        path = self.work / f"eval_{target:07d}.json"
        write(path, dict(completed_updates=target, normalized_score=score,
                         raw_return_mean=100, episodes=[100] * episodes))
        return path

    @unittest.skipUnless(hasattr(os, "pidfd_open"), "Requires the 5090 Python runtime")
    def test_reused_pid_identity_is_never_signaled(self):
        child = self.sleeper()
        expected = identity(child.pid)
        for change in [dict(start_ticks="0"), dict(boot_id="other-boot"),
                       dict(argv=["different", "training"] )]:
            self.assertFalse(send({**expected, **change}, signal.SIGTERM))
            self.assertIsNone(child.poll())

    @unittest.skipUnless(hasattr(os, "pidfd_open"), "Requires the 5090 Python runtime")
    def test_targeted_termination_does_not_signal_other_processes(self):
        child, other = self.sleeper(), self.sleeper()
        terminate(identity(child.pid), grace=1)
        self.assertEqual(child.wait(timeout=5), -signal.SIGTERM)
        self.assertIsNone(other.poll())

    def test_corrupt_and_wrong_step_checkpoints_rejected(self):
        path = self.work / "checkpoint_0500000.pt"
        torch.save(self.saved(), path)
        checkpoint(path, self.target)
        complete = path.read_bytes()
        path.write_bytes(complete[:len(complete) // 2])
        with self.assertRaises(Exception):
            checkpoint(path, self.target)
        torch.save(self.saved(self.target - 1), path)
        with self.assertRaises(ValueError):
            checkpoint(path, self.target)
        saved = self.saved()
        saved["trainer"]["total_it"] -= 1
        torch.save(saved, path)
        with self.assertRaises(ValueError):
            checkpoint(path, self.target)

    def test_incomplete_or_nonfinite_evaluations_rejected(self):
        path = self.eval(episodes=9)
        with self.assertRaises(ValueError):
            evaluation(path, self.target, 10)
        path = self.eval()
        path.write_text(path.read_text().replace('"normalized_score": 42',
                                                 '"normalized_score": NaN'))
        with self.assertRaises(ValueError):
            evaluation(path, self.target, 10)

    def test_summaries_and_replay_exclude_overshoot(self):
        self.eval()
        self.eval(505000, score=999)
        (self.work / "metrics.jsonl").write_text(
            '{"completed_updates":500000,"loss":1}\n'
            '{"completed_updates":500100,"loss":2}\n'
            '{"completed_updates":500200')
        result = result_summary(self.work, self.target, self.spec)
        self.assertEqual(result["result/best_normalized_score"], 42)
        self.assertEqual(result["result/final_normalized_score"], 42)
        self.assertEqual(set(bounded_rows(self.work, self.target)), {500000})
        self.assertEqual(read(self.work / "eval_0505000.json")["normalized_score"], 999)

    def test_wandb_finalizer_keeps_id_and_verifies_bounded_result(self):
        self.eval()
        self.eval(505000, score=999)
        torch.save(self.saved(), self.work / "checkpoint_0500000.pt")
        (self.root / "wandb_sync/medium").mkdir(parents=True)
        write(self.root / "protocol.json", dict(wandb_entity="e", wandb_project="p"))
        write(self.spool / "stop_receipt.json", dict(
            last_recorded_updates=500100, tail_note="discarded", checkpoint_sha256="h"))
        spec = dict(self.spec, wandb_name="CQL-seed1-1M-5090", wandb_id="same-id")
        calls = dict(rows=[], files=[], config={})
        run = SimpleNamespace(
            name="", id="same-id", state="finished", url="url", summary={},
            config=SimpleNamespace(
                update=lambda values, **kw: calls["config"].update(values)),
            define_metric=lambda *a, **kw: None,
            log=lambda row: calls["rows"].append(row),
            save=lambda path, **kw: calls["files"].append(path),
            finish=lambda **kw: calls.update(finished=True),
            file=lambda name: SimpleNamespace(size=123))

        def init(**kw):
            calls["init"] = kw
            return run

        fake = SimpleNamespace(init=init, Settings=lambda **kw: kw,
                               Api=lambda **kw: SimpleNamespace(run=lambda path: run))
        with patch.dict(sys.modules, wandb=fake):
            finish_wandb(dict(root=str(self.root), arm="medium", spec=spec),
                         self.spool, self.target)
        self.assertEqual(calls["init"]["id"], "same-id")
        self.assertEqual(calls["init"]["resume"], "must")
        self.assertEqual(run.name, "CQL-seed1-500k-5090")
        self.assertEqual(calls["config"]["updates"], 500000)
        self.assertEqual(calls["config"]["original_max_timesteps"], 1000000)
        self.assertEqual([row["completed_updates"] for row in calls["rows"]], [500000])
        self.assertEqual(run.summary["result/best_normalized_score"], 42)
        self.assertTrue(calls["finished"])
        self.assertTrue(read(self.spool / "wandb_verified.json")["verified"])

    @unittest.skipUnless(hasattr(os, "pidfd_open"), "Requires the 5090 Python runtime")
    def test_live_guard_waits_for_save_then_stops_and_records_result(self):
        child, observer, unrelated = self.sleeper(), self.sleeper(), self.sleeper()
        write(self.root / "protocol.json", dict(source_sha256={}))
        write(self.work / "status.json",
              dict(status="training", completed_updates=500000))
        torch.save(self.saved(100000), self.work / "checkpoint_0100000.pt")
        self.eval()
        path = self.work / "checkpoint_0500000.pt"
        path.write_bytes(b"incomplete-save")
        job = dict(root=str(self.root), arm="medium", spec=self.spec,
                   train=identity(child.pid), observer=identity(observer.pid),
                   protocol_sha256=digest(self.root / "protocol.json"))
        plan = self.spool / "plan.json"
        write(plan, dict(jobs=[job], target_updates=self.target, reason="test stop"))
        errors = []

        def run():
            try:
                watch(plan, 0)
            except BaseException as error:
                errors.append(error)

        def finish(job, spool, target):
            self.assertIsNotNone(child.poll())
            self.assertIsNotNone(observer.poll())
            write(spool / "wandb_verified.json", dict(verified=True))

        with patch("scripts.cql_stop_500k.run.finish_wandb", side_effect=finish):
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            status_path = self.spool / "run0/status.json"
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if (status_path.exists() and read(status_path)["stage"]
                        == "waiting_for_complete_500k_save"):
                    break
                time.sleep(0.05)
            self.assertIsNone(child.poll())
            self.assertEqual(read(status_path)["stage"],
                             "waiting_for_complete_500k_save")
            torch.save(self.saved(), path)
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(read(status_path)["stage"], "completed")
        self.assertEqual(read(self.work / "status.json")["status"], "stopped_by_user")
        receipt = read(self.spool / "run0/stop_receipt.json")
        self.assertEqual(receipt["checkpoint_sha256"], digest(path))
        self.assertEqual(receipt["target_updates"], 500000)
        self.assertIsNone(unrelated.poll())
        # An operator can restart the watcher after completion without re-signaling.
        with patch("scripts.cql_stop_500k.run.send") as mocked:
            watch(plan, 0)
            self.assertFalse(any(call.args[0] == job["train"]
                                 for call in mocked.call_args_list))


if __name__ == "__main__":
    unittest.main()
