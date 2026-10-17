"""Check that ConDT observation preserves training state and phase accounting."""

import ast
import copy
import json
import os
import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from scripts.condt import telemetry
from scripts.condt.prepare import observed_source

def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


class RemoveObservation(ast.NodeTransformer):
    """Restore the permitted additions, exposing any other AST changes."""

    def visit_Import(self, node):
        if any(alias.name == "condt_observe" for alias in node.names):
            return None
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            function = node.value.func
            if (isinstance(function, ast.Attribute)
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "observer"):
                return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == "observed_lrs"
               for target in node.targets):
            return None
        return self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg == "num_steps" and isinstance(node.value, ast.Call):
            expected = ast.parse(
                "int(os.environ.get('CONDT_PREFLIGHT_PRETRAIN_STEPS', '100000'))",
                mode="eval",
            ).body
            if ast.dump(node.value) == ast.dump(expected):
                node.value = ast.Constant(value=100000)
        return self.generic_visit(node)


class ObservationPatchTests(unittest.TestCase):
    def test_patch_preserves_algorithm_statements_and_formal_pretrain_default(self):
        sources = {
            "gym/experiment_clean.py": """import json
import os
def experiment(exp_prefix, variant):
    data_class, env = prep_data(variant)
    optimizer = make_optimizer(model)
    trainer = make_trainer(model, optimizer)
    print(f'Starting iter')
    if variant['pretrain']:
        seeded_outputs = trainer.train_iteration(num_steps=100000, iter_num = 0, print_logs=True)
        optimizer.param_groups[0]['lr'] = variant['learning_rate']
    else:
        seeded_outputs = trainer.eval_method()
    for iteration in range(variant['max_iters']):
        trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iteration+1)
""",  # noqa: E501
            "gym/decision_transformer/training/trainer_clean.py": """import itertools
class Trainer:
    def train_iteration(self, num_steps, iter_num):
        for ix in range(num_steps):
            if iter_num == 0:
                train_loss = self.pretrain()
            else:
                train_loss = self.train_step(ix + (iter_num - 1)*num_steps)
            if self.scheduler is not None and iter_num != 0:
                self.scheduler.step()
        json_outputs = self.eval_method(logs)
        return json_outputs
""",
        }
        for path, before in sources.items():
            with self.subTest(path=path):
                after = observed_source(path, before)
                restored = RemoveObservation().visit(ast.parse(after))
                self.assertEqual(ast.dump(restored), ast.dump(ast.parse(before)))
        unrelated = "loss = mse + beta * contrastive_loss\n"
        self.assertEqual(observed_source("gym/algorithm.py", unrelated), unrelated)

    def test_unrecognized_upstream_fails_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "one exact upstream block"):
            observed_source("gym/experiment_clean.py", "import json\n")


class ConDTObservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        telemetry._state.clear()
        telemetry._state.update(
            directory=self.directory, start=time.monotonic(),
            pretrain_updates=0, main_updates=0, seed=0,
        )
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
        model(torch.tensor([[1.0, 2.0]])).sum().backward()
        optimizer.step()
        scheduler.step()
        self.trainer = SimpleNamespace(
            model=model, optimizer=optimizer, scheduler=scheduler,
            data_class=SimpleNamespace(
                state_mean=np.array([1.0, 2.0]),
                state_std=np.array([2.0, 3.0]), traj_lens=np.array([20, 30]),
            ),
        )
        self.outputs = {
            seed: dict(returns=[float(seed), 2.5], lengths=[10, 20])
            for seed in (1, 5, 10)
        }

    def tearDown(self):
        telemetry._state.clear()
        self.temporary.cleanup()

    def assert_rng_equal(self, before):
        after = rng_state()
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1][0], after[1][0])
        np.testing.assert_array_equal(before[1][1], after[1][1])
        self.assertEqual(before[1][2:], after[1][2:])
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_phase_counts_lrs_and_checkpoint_preserve_rng_and_model(self):
        trainer = self.trainer
        optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
        model_before = copy.deepcopy(trainer.model.state_dict())
        scheduler_before = copy.deepcopy(trainer.scheduler.state_dict())
        before = rng_state()
        telemetry.ready(trainer)
        for index in range(3):
            telemetry.training_update(
                0, index, 3, {"train_loss": 0, "contrastive_loss": 1.5},
                [0.02], trainer.optimizer,
            )
        telemetry.evaluation(0, 3, self.outputs, trainer, {})
        for iteration in (1, 2):
            for index in range(2):
                telemetry.training_update(
                    iteration, index, 2, {"train_loss": 0.5}, [0.03],
                    trainer.optimizer,
                )
            telemetry.evaluation(iteration, 2, self.outputs, trainer, {"loss": 0.5})
        self.assert_rng_equal(before)
        records = [json.loads(line) for line in
                   (self.directory / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["total_updates"] for row in records], [1, 3, 4, 5, 6, 7])
        self.assertEqual(records[-1]["main_updates"], 4)
        self.assertEqual(records[-1]["phase_updates"], 4)
        self.assertEqual(records[-1]["learning_rates"], [0.03])
        self.assertEqual(records[-1]["next_learning_rates"], [0.01])
        initial = json.loads(
            (self.directory / "evaluations/eval_main_000000.json").read_text()
        )
        self.assertEqual(initial["phase"], "post_pretrain")
        self.assertEqual(initial["pretrain_updates"], 3)
        checkpoint = torch.load(self.directory / "checkpoints/main_000004.pt")
        self.assertEqual(checkpoint["main_updates"], 4)
        self.assertEqual(checkpoint["pretrain_updates"], 3)
        self.assertEqual(checkpoint["scheduler"], scheduler_before)
        self.assertEqual(checkpoint["optimizer"]["param_groups"],
                         optimizer_before["param_groups"])
        for name, value in model_before.items():
            self.assertTrue(torch.equal(checkpoint["model"][name], value))
            self.assertTrue(torch.equal(trainer.model.state_dict()[name], value))
        for index, values in optimizer_before["state"].items():
            for name, value in values.items():
                saved = checkpoint["optimizer"]["state"][index][name]
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(saved, value))
                else:
                    self.assertEqual(saved, value)
        self.assertEqual(checkpoint["python_rng"], before[0])
        np.testing.assert_array_equal(checkpoint["numpy_rng"][1], before[1][1])
        self.assertTrue(torch.equal(checkpoint["torch_rng"], before[2]))
        with self.assertRaisesRegex(RuntimeError, "replace evaluation"):
            telemetry.evaluation(2, 2, self.outputs, trainer, {})

    def test_initial_dt_evaluation_and_repeated_initialization_guard(self):
        with mock.patch.dict(os.environ, {
            "CONDT_RECORD_DIR": str(self.directory), "CONDT_TRAIN_SEED": "5",
        }), mock.patch.object(
            telemetry.importlib.metadata, "version", return_value="test"
        ):
            telemetry.initialize({"model_type": "dt"})
            before = rng_state()
            telemetry.evaluation(0, 0, self.outputs, self.trainer, {})
            self.assert_rng_equal(before)
            result = json.loads(
                (self.directory / "evaluations/eval_main_000000.json").read_text()
            )
            self.assertEqual(result["phase"], "initial")
            self.assertEqual(result["pretrain_updates"], 0)
            with self.assertRaisesRegex(RuntimeError, "existing ConDT attempt"):
                telemetry.initialize({"model_type": "dt"})


if __name__ == "__main__":
    unittest.main()
