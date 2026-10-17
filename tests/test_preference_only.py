"""Guard the missing DT gradient and exact matched-reference comparison."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'algorithms/offline'))
from preference_only import preference_only_objective, verify_initial, verify_reference
from state_only_preference import single_sided_loss

class PreferenceOnlyTests(unittest.TestCase):
    def test_only_open_gate_positive_has_gradient(self):
        ordinary = torch.tensor([100.0], requires_grad=True)
        prediction = torch.tensor([[1.0], [1.0]], requires_grad=True)
        positive = torch.zeros_like(prediction, requires_grad=True)
        negative = torch.tensor([[1.0], [-10.0]], requires_grad=True)
        with torch.no_grad():
            diagnostic = ordinary.square().mean()
        preference, _, _, active = single_sided_loss(
            prediction, positive, negative, .05)
        preference_only_objective(diagnostic, preference, .05).backward()
        self.assertEqual(active.tolist(), [True, False])
        torch.testing.assert_close(prediction.grad, torch.tensor([[.05], [0.0]]))
        self.assertIsNone(ordinary.grad)
        self.assertIsNone(positive.grad)
        self.assertIsNone(negative.grad)

    def test_rejects_accidental_dt_graph_or_changed_coefficient(self):
        loss = torch.tensor(1.0, requires_grad=True)
        with self.assertRaisesRegex(ValueError, 'without autograd'):
            preference_only_objective(loss, loss, .05)
        with self.assertRaisesRegex(ValueError, 'coefficient'):
            preference_only_objective(loss.detach(), loss, 1.0)

    def reference(self, root):
        config = {'variant': 'c', 'preference_weight': .05, 'update_steps': 5000,
                  'learning_rate': .0001, 'eval_episodes': 100, 'name': 'ref'}
        provenance = {key: key for key in ('dataset_sha256', 'pairs_sha256',
                                         'initial_model_sha256', 'packages',
                                         'attention_backend', 'fine_tuning')}
        provenance.update(git_commit='reference', source_sha256={
            'algorithms/offline/late_preference_dt.py': 'old-runner',
            'algorithms/offline/state_only_preference.py': 'shared-helper'})
        files = {'config': config, 'status': {'state': 'completed',
                                             'completed_updates': 5000},
                 'provenance': provenance, 'wandb_run': {'id': 'reference-run'}}
        for name, value in files.items():
            (root / (name + '.json')).write_text(json.dumps(value))
        return {**config, 'variant': 'c_only', 'name': 'new'}, provenance

    def test_requires_same_optimizer_data_lr_and_shared_preference_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, provenance = self.reference(root)
            self.assertTrue(verify_reference(config, provenance, root)[
                'same_parent_model_optimizer_scheduler'])
            for key in ('learning_rate', 'preference_weight', 'eval_episodes'):
                with self.assertRaisesRegex(ValueError, key):
                    verify_reference({**config, key: 99}, provenance, root)
            for key in ('fine_tuning', 'pairs_sha256'):
                with self.assertRaisesRegex(ValueError, key):
                    verify_reference(config, {**provenance, key: 'changed'}, root)
            bad = copy.deepcopy(provenance)
            bad['source_sha256']['algorithms/offline/state_only_preference.py'] = 'new'
            with self.assertRaisesRegex(ValueError, 'helper'):
                verify_reference(config, bad, root)

    def test_rejects_incomplete_reference_and_changed_first_dropout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, provenance = self.reference(root)
            (root / 'status.json').write_text(json.dumps({
                'state': 'training', 'completed_updates': 4900}))
            with self.assertRaisesRegex(ValueError, 'completed'):
                verify_reference(config, provenance, root)
            child = root / 'child'
            child.mkdir()
            for base in (root, child):
                (base / 'parent_verified.json').write_text('{"model":"same"}')
                (base / 'first_update.json').write_text('{"dropout":"same"}')
            verify_initial(child, root)
            (child / 'first_update.json').write_text('{"dropout":"changed"}')
            with self.assertRaisesRegex(ValueError, 'first_update'):
                verify_initial(child, root)


if __name__ == '__main__':
    unittest.main()
