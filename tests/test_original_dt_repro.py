"""Check the actual pinned original DT's reward conditioning and masked loss."""

import copy
import os
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from original_dt_repro import isolated_evaluation, load_upstream, seed_all

SOURCE = Path(
    os.environ.get("ORIGINAL_DT_SOURCE", "third_party/decision-transformer/gym")
)


@unittest.skipUnless((SOURCE / "experiment.py").exists(), "Original DT source required")
class OriginalDTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = load_upstream(SOURCE)
        torch.set_num_threads(2)

    def test_evaluation_preserves_rng(self):
        seed_all(9)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        seed_all(9)
        with isolated_evaluation(42):
            random.random()
            np.random.rand(4)
            torch.rand(13)
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_actual_rollout_uses_constant_rtg_only_when_delayed(self):
        class FakeEnv:
            def reset(self):
                self.step_number = 0
                return np.zeros(3)

            def step(self, action):
                self.step_number += 1
                return np.zeros(3), 100.0, self.step_number == 3, {}

        class SpyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.targets = []

            def get_action(self, states, actions, rewards, rtg, times):
                self.targets.append(rtg[0, -1].item())
                return torch.zeros(2)

        for mode, expected in [("normal", [12.0, 11.9, 11.8]), ("delayed", [12.0] * 3)]:
            model = SpyModel()
            value, length = self.upstream.evaluate_episode_rtg(
                FakeEnv(),
                3,
                2,
                model,
                device="cpu",
                target_return=12.0,
                mode=mode,
                state_mean=np.zeros(3),
                state_std=np.ones(3),
                scale=1000.0,
            )
            np.testing.assert_allclose(model.targets, expected, atol=1e-5)
            self.assertEqual((value, length), (300.0, 3))

    def test_upstream_mask_excludes_padded_targets_from_gradient(self):
        seed_all(1)
        model = self.upstream.DecisionTransformer(
            state_dim=3,
            act_dim=2,
            hidden_size=16,
            max_length=5,
            max_ep_len=20,
            n_layer=1,
            n_head=1,
            n_inner=64,
            activation_function="relu",
            n_positions=1024,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            embd_pdrop=0.0,
        )
        batch = [
            torch.randn(2, 5, 3),
            torch.randn(2, 5, 2),
            torch.zeros(2, 5, 1),
            torch.zeros(2, 5),
            torch.ones(2, 6, 1),
            torch.arange(5).repeat(2, 1),
            torch.tensor([[0, 0, 1, 1, 1]] * 2),
        ]
        results = []
        for padding in (-10.0, 1000.0):
            copy_model = copy.deepcopy(model)
            sample = [x.clone() for x in batch]
            sample[1][:, :2] = padding
            optimizer = torch.optim.AdamW(copy_model.parameters(), lr=1e-4)
            trainer = self.upstream.SequenceTrainer(
                model=copy_model,
                optimizer=optimizer,
                batch_size=2,
                get_batch=lambda _: sample,
                loss_fn=lambda _s, prediction, _r, _st, action, _rt: torch.mean(
                    (prediction - action) ** 2
                ),
            )
            loss = trainer.train_step()
            results.append(
                (
                    loss,
                    torch.cat([p.detach().flatten() for p in copy_model.parameters()]),
                )
            )
        self.assertAlmostEqual(results[0][0], results[1][0], places=6)
        torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
