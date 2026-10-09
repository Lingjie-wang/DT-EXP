"""AntMaze timeout isolation, reward conservation, source fidelity and queue gates."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
import yaml

from algorithms.offline.shapley_redistribution import (
    conserved_rewards as historical_conservation,
    folds,
)
from scripts.cql_antmaze_5090.data import (
    conserved_rewards,
    prepare_arrays,
    segment_inputs,
)
from scripts.cql_antmaze_5090.fit import fit_one
from scripts.cql_antmaze_5090.prepare import (
    ARMS,
    CAMPAIGN,
    materialize_shapley,
    policy_config,
    PREDECESSOR,
    prepare,
    TASKS,
)
from scripts.cql_antmaze_5090.queue import execute
from scripts.cql_delayed.common import digest, read, verify, write
from scripts.cql_repeats_5090.queue import validate_job

REPO = Path(__file__).resolve().parents[1]

def raw_data(episodes=10, horizon=41, tail=3):
    n = episodes * horizon + tail
    rng = np.random.default_rng(19)
    timeouts = np.zeros(n, dtype=bool)
    timeouts[horizon-1:episodes*horizon:horizon] = True
    return dict(observations=rng.normal(size=(n, 3)).astype(np.float32),
                actions=rng.normal(size=(n, 2)).astype(np.float32),
                rewards=rng.integers(0, 2, n).astype(np.float32),
                terminals=rng.integers(0, 2, n).astype(bool), timeouts=timeouts)


class DataTests(unittest.TestCase):
    def test_all_controls_share_timeout_boundaries_and_exact_totals(self):
        raw = raw_data()
        base, inputs, rewards, audit = prepare_arrays(raw, 41)
        self.assertEqual(audit['dropped_incomplete_tail'], 3)
        self.assertEqual(audit['episodes'], 10)
        np.testing.assert_array_equal(base['terminals'], raw['timeouts'][:410])
        live = np.flatnonzero(base['terminals'] == 0)
        np.testing.assert_array_equal(base['next_observations'][live],
                                      raw['observations'][live + 1])
        self.assertFalse(np.count_nonzero(base['next_observations'][~np.isin(
            np.arange(410), live)]))
        self.assertFalse(np.count_nonzero(rewards['delayed'][live]))
        np.testing.assert_array_equal(rewards['dense'], raw['rewards'][:410])
        for values in rewards.values():
            np.testing.assert_allclose(values.reshape(10, 41).sum(1), inputs['returns'],
                                       atol=1e-4)
            np.testing.assert_allclose((values * 10 - 5).reshape(10, 41).sum(1),
                                       inputs['returns'] * 10 - 5 * 41, atol=1e-3)
        np.testing.assert_allclose(rewards['uniform'].reshape(10, 41)[:, :-1],
            np.repeat((inputs['returns'] / 41)[:, None], 40, 1), atol=1e-6)

    def test_success_flags_and_dense_timing_do_not_reach_predictor_or_other_arms(self):
        raw = raw_data()
        first = prepare_arrays(raw, 41)
        changed = copy.deepcopy(raw)
        changed['terminals'] = ~raw['terminals']
        changed['rewards'][:410] = 0
        changed['rewards'][40:410:41] = first[1]['returns']
        second = prepare_arrays(changed, 41)
        for a, b in zip(first[:2], second[:2]):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        for arm in ['delayed', 'uniform']:
            np.testing.assert_array_equal(first[2][arm], second[2][arm])
        del changed['terminals']
        prepare_arrays(changed, 41)

    def test_padding_and_unequal_segments_preserve_order_and_conservation(self):
        for horizon in [701, 1001]:
            values = np.arange(horizon, dtype=np.float32).reshape(1, horizon, 1)
            inputs = segment_inputs(values)
            width = int(np.ceil(horizon / 20))
            restored = []
            for segment in inputs[0]:
                mask = segment[width:2*width].astype(bool)
                restored.extend(segment[:width][mask])
                self.assertFalse(np.count_nonzero(segment[:width][~mask]))
            np.testing.assert_array_equal(restored, values.reshape(-1))
            for total in [-45.123, 0, 150.]:
                rewards, _, _ = conserved_rewards(np.arange(20)-10, total, horizon)
                self.assertAlmostEqual(float(rewards.astype(np.float64).sum()),
                                       total, places=4)
        phi = np.arange(20) / 3
        np.testing.assert_array_equal(conserved_rewards(phi, 32., 1000)[0],
                                      historical_conservation(phi, 32.)[0])

    def test_held_data_cannot_change_fitted_predictor_weights(self):
        rng = np.random.default_rng(41)
        features = rng.normal(size=(20, 41, 3)).astype(np.float32)
        returns = rng.normal(size=20)
        split = list(folds(20))[0]
        settings = dict(learning_rate=.001, weight_decay=.0001, batch=8,
                        validation_masks=2, epochs=3, patience=3)
        with tempfile.TemporaryDirectory() as temporary:
            states = []
            for index in range(2):
                path = Path(temporary) / str(index)
                path.mkdir()
                if index:
                    features[split['held']] = 1e5
                    returns[split['held']] = -1e6
                model, _, _ = fit_one(features, returns, split, settings, path, 'cpu')
                states.append({k: v.clone() for k, v in model.state_dict().items()})
            for key in states[0]:
                self.assertTrue(torch.equal(states[0][key], states[1][key]), key)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        (self.project / 'results' / PREDECESSOR).mkdir(parents=True)
        write(self.project / 'results' / PREDECESSOR / 'plan.json', {})
        self.inputs = self.project / 'inputs'
        self.inputs.mkdir()
        self.tasks = {key: dict(task, horizon=41, episodes=10)
                      for key, task in TASKS.items()}
        for task in self.tasks.values():
            with h5py.File(self.inputs / task['filename'], 'w') as data:
                for key, value in raw_data().items():
                    data[key] = value
        with patch('scripts.cql_antmaze_5090.prepare.TASKS', self.tasks):
            prepare(self.project, self.inputs, 'test-revision')
        self.root = self.project / 'results' / CAMPAIGN

    def fake_fit(self, dataset, passed=True):
        work = self.root / 'datasets' / dataset / 'predictor/fit'
        work.mkdir()
        with np.load(work.parent / 'input.npz') as values:
            rewards = np.stack([conserved_rewards(np.zeros(20), r, 41)[0]
                                for r in values['returns']])
        np.savez(work / 'attribution.npz', rewards=rewards)
        write(work / 'gate.json', dict(passed=passed))
        write(work / 'audit.json', dict(predictor_fitted=True))

    def test_eight_configs_preserve_official_algorithm_and_pair_inputs(self):
        plan = read(self.root / 'plan.json')
        self.assertEqual(len(plan['jobs']), 8)
        self.assertEqual(plan['updates'], 300000)
        self.assertEqual(len({j['wandb_id'] for j in plan['jobs']}), 8)
        for dataset in self.tasks:
            self.fake_fit(dataset)
        for job in plan['jobs']:
            if job['arm'] == 'shapley':
                self.assertTrue(materialize_shapley(self.root, job))
            campaign = validate_job(self.root, job)
            verify(campaign / 'validation')
            algorithm = 'algorithms/offline/cql.py'
            self.assertEqual(digest(campaign / 'source' / algorithm),
                             digest(REPO / algorithm))
            config = yaml.safe_load((campaign / 'source' / job['config']).read_text())
            original = yaml.safe_load((REPO / 'configs/offline/cql/antmaze'
                                       / f"{job['dataset']}_v2.yaml").read_text())
            changed = {key for key in config if config[key] != original.get(key)}
            self.assertTrue(changed <= {'seed', 'max_timesteps', 'eval_freq',
                                        'name', 'group', 'project'})
            self.assertEqual(config['seed'], 21)
            self.assertEqual(config['n_episodes'], 100)
            self.assertEqual(config['max_timesteps'], 300000)
        for dataset in self.tasks:
            arrays = []
            for job in (j for j in plan['jobs'] if j['dataset'] == dataset):
                with np.load(self.root / job['directory'] / job['arm']
                             / 'dataset.npz') as data:
                    arrays.append({k: data[k] for k in data.files if k != 'rewards'})
            for other in arrays[1:]:
                for key in arrays[0]:
                    np.testing.assert_array_equal(arrays[0][key], other[key])

    def test_failed_prediction_gate_cannot_materialize_shapley_data(self):
        plan = read(self.root / 'plan.json')
        for dataset in self.tasks:
            self.fake_fit(dataset, passed=False)
        for job in plan['jobs']:
            if job['arm'] == 'shapley':
                self.assertFalse(materialize_shapley(self.root, job))
                self.assertFalse((self.root / job['directory'] / job['arm']
                                  / 'dataset.npz').exists())

    def test_queue_fits_then_preflights_and_runs_all_eight_once(self):
        plan = read(self.root / 'plan.json')
        training, fitters, observers, preflights, ticks = [], [], [], [], []
        def spawn(command, **kwargs):
            root = Path(command[command.index('--root')+1])
            process = SimpleNamespace(pid=100 + len(training) + len(fitters)
                                      + len(observers), returncode=None, root=root)
            process.poll = lambda: process.returncode
            if 'scripts.cql_antmaze_5090.fit' in command:
                self.fake_fit(root.name)
                process.returncode = 0
                fitters.append(process)
            else:
                process.arm = command[command.index('--arm')+1]
                if '-m' in command:
                    self.assertIn('scripts.cql_antmaze_5090.sync', command)
                    observers.append(process)
                else:
                    training.append(process)
                    (root / process.arm / 'training').mkdir()
            return process
        def smoke(command, **kwargs):
            root = Path(command[command.index('--root')+1])
            arm = command[command.index('--arm')+1]
            preflights.append(root)
            (root / arm / 'preflight').mkdir()
            write(root / arm / 'preflight/status.json', dict(status='completed',
                  completed_updates=100, final_normalized_score=1.))
            write(root / arm / 'preflight/runtime.json', dict(device='cuda:0'))
        def sleep(seconds):
            ticks.append(seconds)
            self.assertLess(len(ticks), 25)
            for process in training:
                process.returncode = 0
                write(process.root / process.arm / 'training/status.json',
                      dict(status='completed', completed_updates=300000))
            for process in observers:
                process.returncode = 0
                directory = process.root / 'wandb_sync' / process.arm
                directory.mkdir(parents=True, exist_ok=True)
                write(directory / 'full_completion_verification.json',
                      dict(verified=True))
        prefix = 'scripts.cql_antmaze_5090.queue.'
        resource = dict(gpu_pids=list(range(6)), free_gpu_mib=16000,
                        free_host_mib=32000, utilization=50)
        with patch(prefix+'predecessor_ready', side_effect=[False, True]), \
                patch(prefix+'probe', return_value=resource), \
                patch(prefix+'subprocess.Popen', side_effect=spawn), \
                patch(prefix+'subprocess.run', side_effect=smoke), \
                patch(prefix+'time.sleep', side_effect=sleep):
            execute(self.root)
        self.assertEqual(len(fitters), 2)
        self.assertEqual(len(training), 8)
        self.assertEqual(len(preflights), 8)
        self.assertEqual(len(observers), 8)
        self.assertEqual(read(self.root/'queue/status.json')['stage'], 'completed')
        self.assertEqual(plan, read(self.root/'initial_plan.json'))
        with self.assertRaises(FileExistsError):
            execute(self.root)

    def test_official_config_guard(self):
        with self.assertRaises(ValueError):
            policy_config(dict(env='wrong'), 'antmaze-umaze-v2', ARMS[0], 'group')


torch.set_num_threads(2)
if __name__ == '__main__':
    unittest.main()
