import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/decision_diffuser" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


supplement = load("dd_supplement", "supplement_evaluations.py")
sync = load("dd_sync", "sync_results.py")


class EvaluationScheduleTests(unittest.TestCase):
    def test_amendment_preserves_original_and_marks_unavailable_points(self):
        requested = list(range(100000, 1000001, 100000))
        original = [100000, 500000, 1000000]
        dense = supplement.schedule_arm(requested, original, 668500, [100000, 500000])
        delayed = supplement.schedule_arm(requested, original, 912500, [100000, 500000])
        self.assertEqual(dense["supplemental_eval_updates"], [700000, 800000, 900000])
        self.assertEqual(dense["unavailable_past"], [200000, 300000, 400000, 600000])
        self.assertEqual(delayed["supplemental_eval_updates"], [])
        self.assertEqual(delayed["expected_eval_updates"], original)
        # An original evaluation that is in progress must remain required.
        in_progress = supplement.schedule_arm(requested, original, 100000, [])
        self.assertIn(100000, in_progress["expected_eval_updates"])
        self.assertNotIn(100000, in_progress["unavailable_past"])

    def test_atomic_checkpoint_replacement_does_not_change_retained_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, retained = root / "latest.pt", root / "step.pt"
            protocol = {"seed": 0}
            torch.save(
                {
                    "completed_updates": 700000,
                    "protocol": protocol,
                    "ema": {"weight": torch.tensor([1.0])},
                },
                latest,
            )
            self.assertTrue(supplement.retain_exact(latest, retained, 700000, protocol))
            temporary = root / "latest.tmp"
            torch.save(
                {
                    "completed_updates": 710000,
                    "protocol": protocol,
                    "ema": {"weight": torch.tensor([2.0])},
                },
                temporary,
            )
            temporary.replace(latest)
            preserved = torch.load(retained)
            self.assertEqual(preserved["completed_updates"], 700000)
            self.assertEqual(float(preserved["ema"]["weight"]), 1.0)

    def test_checkpoint_never_receives_an_incorrect_step_or_protocol_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest, retained = root / "latest.pt", root / "step.pt"
            protocol = {"seed": 0}
            torch.save({"completed_updates": 690000, "protocol": protocol}, latest)
            self.assertFalse(supplement.retain_exact(latest, retained, 700000, protocol))
            self.assertFalse(retained.exists())
            torch.save({"completed_updates": 710000, "protocol": protocol}, latest)
            with self.assertRaisesRegex(RuntimeError, "Missed checkpoint"):
                supplement.retain_exact(latest, retained, 700000, protocol)
            torch.save({"completed_updates": 700000, "protocol": {"seed": 1}}, latest)
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                supplement.retain_exact(latest, retained, 700000, protocol)
            self.assertFalse(retained.exists())
            self.assertFalse(retained.with_suffix(".capture").exists())

    def test_wandb_uses_arm_specific_schedule_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"eval_updates": [100000, 500000, 1000000]}
            self.assertEqual(
                sync.evaluation_schedule(root, "dense", protocol),
                (protocol["eval_updates"], {}),
            )
            value = {
                "interval": 100000,
                "requested_eval_updates": list(range(100000, 1000001, 100000)),
                "arms": {
                    "dense": {
                        "expected_eval_updates": [
                            100000,
                            500000,
                            700000,
                            800000,
                            900000,
                            1000000,
                        ],
                        "unavailable_past": [200000, 300000, 400000, 600000],
                    }
                },
            }
            (root / "evaluation_schedule.json").write_text(json.dumps(value))
            expected, metadata = sync.evaluation_schedule(root, "dense", protocol)
            self.assertEqual(expected, value["arms"]["dense"]["expected_eval_updates"])
            self.assertEqual(
                metadata["unavailable_past_eval_updates"],
                value["arms"]["dense"]["unavailable_past"],
            )


if __name__ == "__main__":
    unittest.main()
