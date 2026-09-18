"""CPU unit tests; real full-batch pairing is additionally checked by smoke jobs."""

import ast
import hashlib
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import hard_fork_dt
import numpy as np
import torch
from hard_fork_dt import continuation_steps, SAPTrainConfig
from top_return_weighted_dt import restore_rng, rng_state

class PairedResumeTest(unittest.TestCase):
    def test_legacy_defaults_and_exact_completed_update_counts(self):
        self.assertFalse(SAPTrainConfig().paired_resume)
        legacy = continuation_steps(50000, 75001, False)
        paired = continuation_steps(50000, 75000, True)
        self.assertEqual((legacy[0], legacy[-1], len(legacy)), (50000, 75000, 25001))
        self.assertEqual((paired[0], paired[-1], len(paired)), (50001, 75000, 25000))

    def test_rng_roundtrip_covers_python_numpy_and_torch(self):
        state = rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(4))
        restore_rng(state)
        actual = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_production_zero_weight_objective_has_exact_dt_gradient(self):
        # Evaluate the actual trainer expression, not a loss reimplementation.
        tree = ast.parse(Path(hard_fork_dt.__file__).read_text())
        assignments = [
            node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "total_loss" for t in node.targets
            )
            and "preference_weight" in ast.dump(node.value)
        ]
        self.assertEqual(len(assignments), 1)
        expression = compile(ast.Expression(assignments[0].value), "loss", "eval")
        prediction = torch.tensor([0.2, -0.4, 0.1], requires_grad=True)
        dt_loss = prediction.square().mean()
        preference = (prediction - 0.7).square().mean()
        reference = (prediction + 0.1).square().mean()
        actual = eval(expression, {}, {
            "dt_loss": dt_loss, "preference_loss": preference,
            "reference_loss": reference,
            "config": SimpleNamespace(preference_weight=0.0, reference_weight=0.0),
        })
        actual_grad = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(dt_loss, prediction)[0]
        torch.testing.assert_close(actual, dt_loss, rtol=0, atol=0)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_original_v3_auxiliary_branch_is_unchanged(self):
        tree = ast.parse(Path(hard_fork_dt.__file__).read_text())
        branches = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.If) and len(n.body) > 5
            and "GtE" in ast.dump(n.test)
            and "preference_start_step" in ast.dump(n.test)
        ]
        self.assertEqual(len(branches), 1)
        # AST digest of the entire branch at commit 1e8a9c1, excluding locations.
        actual = hashlib.sha256(ast.dump(branches[0]).encode()).hexdigest()
        self.assertEqual(
            actual, "5e394f646cd4fbf92cd83a14e64c39380c8cbbf8e32a818ab46f996861f0cada"
        )


if __name__ == "__main__":
    unittest.main()
