import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.cql_antmaze_finish_guard_5090 import guard

class FinishGuardTests(unittest.TestCase):
    def fixture(self, root):
        native = root / "run/test.wandb"
        native.parent.mkdir()
        native.write_bytes(b"complete native file")
        (native.parent / "logs").mkdir()
        (native.parent / "logs/debug-internal.log").write_text(f"close: {native}")
        stat = native.stat()
        cursor = dict(path=str(native), device=stat.st_dev, inode=stat.st_ino,
                      offset=stat.st_size, history_lines=100, last_history_step=100,
                      summary_step=100, exit_code=0)
        identity = dict(pid=123, start="42", command="python cql.py", parent=1)
        record = dict(stage="waiting_for_flush_and_upload", native=str(native),
                      parent=identity, backend=dict(identity, pid=456), signals=[])
        return native, cursor, record

    def test_upload_proof_requires_exact_file_identity_and_full_history(self):
        with tempfile.TemporaryDirectory() as temp:
            native, cursor, _ = self.fixture(Path(temp))
            self.assertTrue(guard.uploaded(cursor, native, 100))
            for key, value in [("offset", 1), ("history_lines", 99),
                               ("last_history_step", 99), ("summary_step", 99),
                               ("exit_code", 255), ("inode", -1), ("device", -1),
                               ("path", "/another/run.wandb")]:
                with self.subTest(key=key):
                    self.assertFalse(guard.uploaded(dict(cursor, **{key: value}),
                                                   native, 100))

    def test_no_release_until_writer_closed_and_upload_acknowledged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native, cursor, record = self.fixture(root)
            for open_writer, acknowledged in [(True, True), (False, False)]:
                current = cursor if acknowledged else dict(cursor, history_lines=99)
                with patch.object(guard, "holds_native", return_value=open_writer), \
                        patch.object(guard, "send") as send:
                    guard.advance(record, root / "state.json", current, 100)
                    send.assert_not_called()
            (native.parent / "logs/debug-internal.log").write_text("still writing")
            with patch.object(guard, "holds_native", return_value=False), \
                    patch.object(guard, "send") as send:
                guard.advance(record, root / "state.json", cursor, 100)
                send.assert_not_called()

    def test_complete_proofs_release_only_the_recorded_trainer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, cursor, record = self.fixture(root)
            with patch.object(guard, "holds_native", return_value=False), \
                    patch.object(guard, "send") as send:
                guard.advance(record, root / "state.json", cursor, 100)
                send.assert_called_once_with(record["parent"], guard.signal.SIGINT,
                                             record, root / "state.json")
                self.assertEqual(record["stage"], "releasing_completed_cli")

    def test_pid_reuse_never_receives_signal(self):
        identity = dict(pid=123, start="42", command="python cql.py")
        with patch.object(guard, "process", return_value=dict(identity, start="43")), \
                patch.object(guard.os, "pidfd_open") as pidfd:
            guard.send(identity, guard.signal.SIGINT, {}, Path("unused"))
            pidfd.assert_not_called()

    def test_queued_incomplete_recent_and_non_shutdown_jobs_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = dict(id="job", updates=100, eval_every=50, eval_episodes=100)
            self.assertIsNone(guard.candidate(root, {}, job,
                                             {"state": "queued"}, {}, 600))
            state = dict(state="running", pid=123)
            with patch.object(guard, "completed", side_effect=ValueError("incomplete")):
                self.assertIsNone(guard.candidate(root, {}, job, state, {}, 600))
            checkpoint = root / "checkpoint.pt"
            checkpoint.touch()
            proof = dict(final_checkpoint=str(checkpoint))
            with patch.object(guard, "completed", return_value=proof):
                self.assertIsNone(guard.candidate(root, {}, job, state, {}, 600))
                work = root / "job/training"
                work.mkdir(parents=True)
                (work / "console.log").write_text("still evaluating")
                with patch.object(guard.time, "time", return_value=time.time() + 700):
                    self.assertIsNone(guard.candidate(root, {}, job, state, {}, 600))
                    (work / "console.log").write_text(
                        "Waiting for W&B process to finish... (success).")
                    self.assertIsNone(guard.candidate(root, {}, job, state, {}, 600))

    def test_checkpoint_contents_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "upstream/algorithms/offline/cql.py"
            source.parent.mkdir(parents=True)
            source.write_text("class Scalar(nn.Module):\n    pass\n")
            path = root / "checkpoint.pt"
            model = {k: dict(weight=torch.tensor([1.0])) for k in [
                "actor", "critic1", "critic2", "critic1_target", "critic2_target"]}
            for steps, value, valid in [(100, 1.0, True), (99, 1.0, False),
                                        (100, float("nan"), False)]:
                model.update(total_it=steps, actor=dict(weight=torch.tensor([value])))
                torch.save(model, path)
                if valid:
                    self.assertEqual(guard.checkpoint_proof(root, path, 100)["total_it"],
                                     100)
                else:
                    with self.assertRaises(ValueError):
                        guard.checkpoint_proof(root, path, 100)


if __name__ == "__main__":
    unittest.main()
