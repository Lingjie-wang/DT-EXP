import ast
import copy
import importlib.util
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "dd_repro", ROOT / "algorithms/offline/decision_diffuser_repro.py"
)
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)


class DecisionDiffuserTests(unittest.TestCase):
    def test_delayed_has_only_terminal_information(self):
        rewards = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 3.0, -2.0, 1.0]])
        altered = rewards.copy()
        altered[:, 0] += 10
        altered[:, 2] -= 10
        delayed = dd.reward_transform(rewards, "delayed")
        np.testing.assert_array_equal(delayed[:, :-1], 0)
        np.testing.assert_array_equal(delayed.sum(1), rewards.sum(1))
        np.testing.assert_array_equal(delayed, dd.reward_transform(altered, "delayed"))
        np.testing.assert_array_equal(dd.reward_transform(rewards, "dense"), rewards)
        expected = rewards.sum(1)[:, None] * 0.99 ** np.arange(3, -1, -1)[None]
        np.testing.assert_allclose(
            dd.discounted_returns(delayed, 0.99), expected, atol=1e-14
        )

    def test_original_dataset_getitem_parity(self):
        # Execute the upstream dataset's actual accessor/indices functions without
        # its MuJoCo/environment loader. The numerical reference is upstream code.
        source = (
            ROOT / "third_party/decision-diffuser/code/diffuser/datasets/sequence.py"
        )
        tree = ast.parse(source.read_text())
        klass = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "SequenceDataset"
        )
        methods = [
            n
            for n in klass.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ["__getitem__", "get_conditions", "make_indices"]
        ]
        namespace = {
            "np": np,
            "RewardBatch": namedtuple("Batch", "trajectories conditions returns"),
        }
        exec(
            compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
            namespace,
        )
        reference = type("Reference", (), {n.name: namespace[n.name] for n in methods})()
        rng = np.random.default_rng(3)
        raw_rewards = rng.normal(size=(3, 1000)).astype(np.float32)
        obs = rng.normal(size=(3, 1000, 17)).astype(np.float32)
        actions = rng.normal(size=(3, 1000, 6)).astype(np.float32)
        reference.max_path_length = 1000
        reference.horizon, reference.use_padding, reference.include_returns = (
            100,
            True,
            True,
        )
        reference.returns_scale = 400
        reference.discounts = 0.99 ** np.arange(1000)[:, None]
        reference.indices = reference.make_indices([1000] * 3, 100)
        self.assertEqual(len(reference.indices), 2700)
        np.testing.assert_array_equal(reference.indices[899], [0, 899, 999])
        p = {"valid_starts_per_episode": 900, "horizon": 100}
        for arm in ["dense", "delayed"]:
            rewards = dd.reward_transform(raw_rewards, arm)
            data = {
                "observations": obs,
                "actions": actions,
                arm: (dd.discounted_returns(rewards, 0.99) / 400).astype(np.float32),
            }
            reference.fields = SimpleNamespace(
                normed_observations=obs,
                normed_actions=actions,
                rewards=rewards[:, :, None],
            )
            for index in [0, 899, 900, 1702, 2699]:
                original = reference[index]
                actual = dd.batch(data, np.array([index]), arm, p, "cpu")
                np.testing.assert_array_equal(actual[0][0], original.trajectories)
                np.testing.assert_array_equal(actual[1][0][0], original.conditions[0])
                np.testing.assert_allclose(
                    actual[2][0], original.returns, rtol=1e-6, atol=1e-9
                )

    def test_shuffle_resume_and_pairing(self):
        stream = dd.WindowStream(99, 0)
        initial = [stream.next(32) for _ in range(4)]
        self.assertEqual([len(x) for x in initial], [32, 32, 32, 3])
        np.testing.assert_array_equal(np.sort(np.concatenate(initial)), np.arange(99))
        state = copy.deepcopy(stream.state())
        expected = [stream.next(32) for _ in range(7)]
        recovered = dd.WindowStream(99, 444)
        recovered.restore(state)
        for ids in expected:
            np.testing.assert_array_equal(recovered.next(32), ids)
        self.assertEqual(stream.chain, recovered.chain)

    def test_official_loss_rng_resume_and_conditioning(self):
        torch.set_num_threads(2)
        p = dict(
            horizon=100,
            dim=8,
            dim_mults=[1, 2, 4],
            inverse_hidden_dim=16,
            n_diffusion_steps=4,
            condition_dropout=0.25,
            guidance=1.2,
        )
        dd.seed_all(5)
        model = dd.build_model(p, ROOT / "third_party/decision-diffuser/code/diffuser")
        x = torch.randn(2, 100, 23)
        cond, ret = {0: x[:, 0, 6:].clone()}, torch.full((2, 1), 0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        loss, _ = model.loss(x, cond, ret)
        loss.backward()
        optimizer.step()
        state = dict(
            model=copy.deepcopy(model.state_dict()),
            optimizer=copy.deepcopy(optimizer.state_dict()),
            rng=dd.rng_state(),
        )

        def update():
            optimizer.zero_grad()
            loss, _ = model.loss(x, cond, ret)
            loss.backward()
            optimizer.step()
            return float(loss), dd.state_hash(model)

        expected = update()
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        dd.restore_rng(state["rng"])
        self.assertEqual(expected, update())
        sample = model.conditional_sample(cond, returns=ret, verbose=False)
        torch.testing.assert_close(sample[:, 0], cond[0], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(sample).all())
        self.assertEqual(tuple(sample.shape), (2, 100, 17))


if __name__ == "__main__":
    unittest.main()
