"""Protect optimizer continuity, matched branches and frozen parent provenance."""

import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithms/offline"))
from late_preference import (
    auxiliary_loss,
    fingerprint,
    restore_parent,
    validate_parent_config,
    verify_parent_provenance,
)

class ParentTransferTests(unittest.TestCase):
    def make_state(self):
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0008)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
        model(torch.ones(2, 2)).sum().backward()
        opt.step()
        sched.step()
        return model, opt, sched

    def test_restores_all_moments_and_keeps_lower_lr_after_scheduler_step(self):
        model, opt, sched = self.make_state()
        saved = copy.deepcopy({"model_state": model.state_dict(),
                               "optimizer_state": opt.state_dict(),
                               "scheduler_state": sched.state_dict()})
        fresh, new_opt, new_sched = self.make_state()
        audit = restore_parent(fresh, new_opt, new_sched, saved, 1e-4)
        self.assertEqual(audit["restored_optimizer_sha256"],
                         fingerprint(saved["optimizer_state"]))
        self.assertEqual(fingerprint(new_opt.state_dict()["state"]),
                         fingerprint(saved["optimizer_state"]["state"]))
        self.assertEqual(fingerprint(fresh.state_dict()),
                         fingerprint(saved["model_state"]))
        new_opt.step()
        new_sched.step()
        self.assertEqual(new_opt.param_groups[0]["lr"], 1e-4)
        self.assertEqual(new_sched.last_epoch, sched.last_epoch + 1)
        self.assertEqual(saved["optimizer_state"]["param_groups"][0]["lr"], 0.0008)

    def test_parent_c_and_changed_pair_settings_are_rejected(self):
        saved = {"variant": "dt", "preference_weight": 0, "reward_mode": "original",
                 "update_steps": 100000, "resume_checkpoint": "", "seq_len": 20,
                 "batch_size": 4096, "learning_rate": 0.0008}
        current = {**saved, "variant": "c", "preference_weight": 0.05,
                   "update_steps": 5000, "learning_rate": 1e-4}
        validate_parent_config(current, saved, 100000)
        for changed, count in (({**saved, "variant": "c"}, 100000),
                               (saved, 99999)):
            with self.assertRaises(ValueError):
                validate_parent_config(current, changed, count)
        for key in ("seq_len", "batch_size"):
            with self.assertRaisesRegex(ValueError, key):
                validate_parent_config({**current, key: 1}, saved, 100000)

    def test_provenance_requires_same_data_pairs_runtime_and_shared_sources(self):
        parent = {k: k for k in ("dataset_sha256", "pairs_sha256", "packages",
                                "attention_backend")}
        parent["source_sha256"] = {"algorithms/offline/" + name: name for name in
                                  ("dt.py", "dense_action_preference.py",
                                   "state_only_preference.py",
                                   "top_return_weighted_dt.py")}
        verify_parent_provenance(parent, copy.deepcopy(parent))
        for key in ("pairs_sha256", "packages"):
            with self.assertRaisesRegex(ValueError, key):
                verify_parent_provenance(parent, {**parent, key: "wrong"})
        child = copy.deepcopy(parent)
        child["source_sha256"]["algorithms/offline/dt.py"] = "different"
        with self.assertRaisesRegex(ValueError, "dt.py"):
            verify_parent_provenance(parent, child)

    def test_control_gradient_equals_dt_and_b_keeps_closed_gate_positives(self):
        grads = {}
        for arm, weight in (("dt", 0), ("b", .05), ("c", .05)):
            prediction = torch.tensor([[1.], [2.]], requires_grad=True)
            dt = prediction.square().mean()
            pref, _, _, _ = auxiliary_loss(
                arm, prediction, torch.zeros_like(prediction),
                torch.full_like(prediction, -10), .05)
            (dt + weight * pref).backward()
            grads[arm] = prediction.grad
        torch.testing.assert_close(grads["dt"], torch.tensor([[1.], [2.]]))
        torch.testing.assert_close(grads["c"], grads["dt"])
        torch.testing.assert_close(grads["b"], 1.05 * grads["dt"])


if __name__ == "__main__":
    unittest.main()
