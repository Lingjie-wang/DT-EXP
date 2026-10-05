"""Verify ungated gradients, matched controls and safe predecessor completion."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithms/offline"))
from matched_positive import positive_imitation_loss, verify_reference
from state_only_preference import single_sided_loss

spec = importlib.util.spec_from_file_location(
    "matched_positive_queue", ROOT / "scripts/matched_positive/queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class PositiveObjectiveTests(unittest.TestCase):
    def test_all_positives_contribute_even_when_c_gate_is_closed(self):
        prediction = torch.tensor([[1.0], [2.0]], requires_grad=True)
        positive = torch.zeros_like(prediction, requires_grad=True)
        negative = torch.full_like(prediction, -10, requires_grad=True)
        c, _, _, active_c = single_sided_loss(prediction, positive, negative)
        self.assertEqual(c.item(), 0)
        self.assertFalse(active_c.any())
        b, _, _, active_b = positive_imitation_loss(prediction, positive, negative)
        self.assertEqual(b.item(), 2.5)
        self.assertTrue(active_b.all())
        b.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[1.0], [2.0]]))
        self.assertIsNone(positive.grad)
        self.assertIsNone(negative.grad)

    def test_negative_values_and_margin_cannot_change_b_gradient(self):
        grads, losses = [], []
        for negative, margin in ((100.0, 0.05), (-10.0, 1000.0)):
            prediction = torch.tensor([[1.0, -2.0]], requires_grad=True)
            loss, _, _, _ = positive_imitation_loss(
                prediction, torch.zeros_like(prediction),
                torch.full_like(prediction, negative), margin)
            loss.backward()
            grads.append(prediction.grad)
            losses.append(loss.item())
        torch.testing.assert_close(grads[0], grads[1])
        self.assertEqual(losses[0], losses[1])

    def test_b_equals_c_gradient_when_all_c_gates_are_open(self):
        gradients = []
        for objective in (single_sided_loss, positive_imitation_loss):
            prediction = torch.tensor([[1.0], [2.0]], requires_grad=True)
            loss, _, _, active = objective(
                prediction, torch.zeros_like(prediction), prediction.detach())
            self.assertTrue(active.all())
            (0.05 * loss).backward()
            gradients.append(prediction.grad)
        torch.testing.assert_close(*gradients)


class MatchedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"variant": "b", "preference_weight": 0.05, "batch_size": 8,
                       "train_seed": 0, "update_steps": 2, "target_returns": (12000,),
                       "group": "b-group", "output_dir": "/new"}
        previous_config = {**self.config, "variant": "c", "group": "old",
                           "output_dir": "/old", "target_returns": [12000]}
        self.provenance = {key: "same" for key in (
            "initial_model_sha256", "dataset_sha256", "pairs_sha256",
            "packages", "attention_backend")}
        self.provenance.update({"git_commit": "new-commit", "source_sha256": {
            "algorithms/offline/" + filename: "unchanged" for filename in
            ("dt.py", "state_only_preference.py", "dense_action_preference.py",
             "top_return_weighted_dt.py")}})
        for name, data in {
            "config": previous_config, "provenance": self.provenance,
            "status": {"state": "completed", "completed_updates": 2},
            "wandb_run": {"id": "reference-c", "url": "example"},
        }.items():
            (self.root / (name + ".json")).write_text(json.dumps(data))

    def test_allows_only_declared_arm_and_operational_changes(self):
        result = verify_reference(self.config, self.provenance, self.root)
        self.assertTrue(result["training_settings_match"])
        self.assertEqual(result["reference_run"]["id"], "reference-c")
        for key in ("batch_size", "train_seed", "update_steps"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                verify_reference({**self.config, key: 999}, self.provenance, self.root)

    def test_changed_pair_pool_initialization_or_shared_code_is_rejected(self):
        for key in ("pairs_sha256", "initial_model_sha256", "dataset_sha256"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                verify_reference(self.config, {**self.provenance, key: "changed"},
                                 self.root)
        changed = copy.deepcopy(self.provenance)
        changed["source_sha256"]["algorithms/offline/dt.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "shared source"):
            verify_reference(self.config, changed, self.root)

    def test_unfinished_reference_is_rejected(self):
        (self.root / "status.json").write_text(json.dumps(
            {"state": "training", "completed_updates": 1}))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify_reference(self.config, self.provenance, self.root)


class MatchedQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("queue_manifest.json", {"git_commit": "mc-commit",
                   "dependency": "/step", "dependency_commit": "step-commit"})

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def complete_parent(self):
        self.write("queue_status.json", {"state": "completed"})
        self.write("comparison.json", {})
        self.write("c-seed0/status.json",
                   {"state": "completed", "completed_updates": 100000})
        self.write("c-seed0/summary.json", {"completed_updates": 100000})
        self.write("c-seed0/config.json", {"update_steps": 100000, "variant": "c",
                   "reward_mode": "original", "preference_label": "mc_value_advantage",
                   "preference_weight": 0.05, "eval_episodes": 2})
        self.write("c-seed0/provenance.json", {"git_commit": "mc-commit"})
        self.write("c-seed0/evaluations/step100000.json",
                   {"step": 100000, "returns": [1, 2]})

    def test_waits_for_mc_training_and_stops_on_failure(self):
        self.write("queue_status.json", {"state": "training"})
        self.assertFalse(queue.dependency_ready(self.root, "mc-commit"))
        self.write("queue_status.json", {"state": "failed"})
        with self.assertRaisesRegex(RuntimeError, "failed"):
            queue.dependency_ready(self.root, "mc-commit")

    def test_complete_requires_entire_preceding_chain(self):
        self.complete_parent()
        with mock.patch.object(queue, "step_campaign_ready", return_value=True) as prev:
            self.assertTrue(queue.dependency_ready(self.root, "mc-commit"))
            prev.assert_called_once_with(Path("/step"), "step-commit")
        with mock.patch.object(queue, "step_campaign_ready", return_value=False):
            with self.assertRaisesRegex(ValueError, "Earlier"):
                queue.dependency_ready(self.root, "mc-commit")

    def test_wrong_commit_or_missing_final_episodes_cannot_launch(self):
        with self.assertRaisesRegex(ValueError, "revision"):
            queue.dependency_ready(self.root, "wrong")
        self.complete_parent()
        self.write("c-seed0/evaluations/step100000.json",
                   {"step": 100000, "returns": [1]})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            queue.dependency_ready(self.root, "mc-commit")


if __name__ == "__main__":
    unittest.main()
