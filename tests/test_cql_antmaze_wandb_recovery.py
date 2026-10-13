import json
import tempfile
import unittest
from pathlib import Path

import requests
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

from scripts.cql_antmaze_success_wandb_recovery.relay import safe_summary, scan, send

def record(step, extra=None):
    pb = wandb_internal_pb2.Record()
    for key, value in dict(_step=step, loss=step / 10, **(extra or {})).items():
        item = pb.history.item.add()
        item.key, item.value_json = key, json.dumps(value)
    return pb.SerializeToString()


def native_file(path, records):
    writer = DataStore()
    writer.open_for_write(str(path))
    try:
        for data in records:
            pb = wandb_internal_pb2.Record()
            pb.ParseFromString(data)
            writer.write(pb)
    finally:
        writer._fp.close()


class FakeSession:
    def __init__(self, fail_after_accept=False):
        self.rows = []
        self.payloads = []
        self.fail_after_accept = fail_after_accept

    def post(self, endpoint, json, timeout):
        self.payloads.append(json)
        assert timeout == (10, 30)
        history = json["files"].get("wandb-history.jsonl")
        if history:
            offset, lines = history["offset"], history["content"]
            self.rows[offset:offset + len(lines)] = lines
        if self.fail_after_accept:
            self.fail_after_accept = False
            raise requests.ReadTimeout("unknown outcome")
        result = requests.Response()
        result.status_code = 200
        return result


class RecoveryTest(unittest.TestCase):
    def test_batches_preserve_original_history_and_nested_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.wandb"
            values = [record(i, {"nested": {"value": i}}) for i in range(100)]
            native_file(path, values)
            offset, histories = None, []
            for _ in range(100):
                batch = scan(path, offset, max_bytes=200)
                self.assertNotEqual(offset, batch["offset"])
                offset = batch["offset"]
                histories += [json.loads(s) for s in batch["histories"]]
                if batch["caught_up"]:
                    break
            self.assertEqual([row["_step"] for row in histories], list(range(100)))
            self.assertEqual(histories[77]["nested"], {"value": 77})
            self.assertEqual(offset, path.stat().st_size)

    def test_partial_tail_is_retried_without_skipping_or_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.wandb"
            native_file(path, [record(1), record(2, {"long": "a" * 40000})])
            full = path.read_bytes()
            path.write_bytes(full[:-13])
            first = scan(path)
            self.assertEqual([json.loads(s)["_step"] for s in first["histories"]], [1])
            self.assertFalse(first["caught_up"])
            path.write_bytes(full)
            second = scan(path, first["offset"])
            self.assertEqual([json.loads(s)["_step"] for s in second["histories"]], [2])
            self.assertTrue(second["caught_up"])

    def test_middle_corruption_is_never_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.wandb"
            native_file(path, [record(i) for i in range(10)])
            content = bytearray(path.read_bytes())
            content[16] ^= 1
            path.write_bytes(content)
            with self.assertRaises(AssertionError):
                scan(path)

    def test_network_unknown_outcome_replays_same_offsets(self):
        session = FakeSession(fail_after_accept=True)
        state = {"history_lines": 2, "offset": 100}
        session.rows = ["old0", "old1"]
        batch = dict(histories=['{"_step": 3}', '{"_step": 4}'], offset=200,
                     exit_code=None)
        with self.assertRaises(requests.ReadTimeout):
            send(session, "https://example.test", state, batch, {"_step": 4})
        self.assertEqual(state, {"history_lines": 2, "offset": 100})
        updated = send(session, "https://example.test", state, batch, {"_step": 4})
        self.assertEqual(len(session.rows), 4)
        self.assertEqual(updated["history_lines"], 4)
        self.assertEqual(updated["last_history_step"], 4)
        self.assertFalse(updated["complete"])
        self.assertEqual(session.payloads[0], session.payloads[1])

    def test_summary_and_heartbeat_do_not_create_history(self):
        session = FakeSession()
        empty = dict(histories=[], offset=700, exit_code=None)
        updated = send(session, "https://example.test", {"history_lines": 10},
                       empty, {"_step": 20, "d4rl_normalized_score": 88})
        self.assertEqual(updated["history_lines"], 10)
        self.assertEqual(session.rows, [])
        self.assertFalse(session.payloads[0]["complete"])
        self.assertNotIn("exitcode", session.payloads[0])

    def test_partial_summary_and_exit_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text('{"unfinished":')
            self.assertIsNone(safe_summary(path))
            path.write_text('{"_step": 300000}')
            self.assertEqual(safe_summary(path)["_step"], 300000)
            pb = wandb_internal_pb2.Record()
            pb.exit.exit_code = 1
            data = Path(directory) / "run.wandb"
            native_file(data, [record(1), pb.SerializeToString()])
            batch = scan(data)
            self.assertEqual(batch["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
