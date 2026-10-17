"""Preserve the original failure, report weak predictors and fix module startup."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_cql_antmaze_5090 as fixtures

from scripts.cql_antmaze_retry_5090.prepare import prepare
from scripts.cql_antmaze_retry_5090.queue import execute
from scripts.cql_delayed.common import digest, read, verify, write
from scripts.cql_repeats_5090.queue import validate_job

class RetryTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.CampaignTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.previous = fixture.root
        for dataset in fixture.tasks:
            fixture.fake_fit(dataset, passed=False)
            work = self.previous / 'datasets' / dataset / 'predictor'
            with np.load(work/'input.npz') as stored:
                totals = stored['returns']
            with np.load(work/'fit/attribution.npz') as stored:
                rewards = stored['rewards']
            np.savez(work/'fit/attribution.npz', rewards=rewards, returns=totals,
                     contributions=np.zeros((len(totals), 20)))
        (self.previous/'queue').mkdir()
        (self.previous/'queue/lock').touch()
        write(self.previous/'queue/status.json', dict(stage='finished_with_failures'))
        self.root = self.previous.with_name('retry')

    def test_report_only_keeps_failed_gate_and_all_original_files(self):
        before = {p: digest(p) for p in self.previous.rglob('*') if p.is_file()}
        prepare(self.previous, self.root, 'retry-revision', 'report-only')
        plan = read(self.root/'plan.json')
        self.assertEqual(len(plan['jobs']), 8)
        for job in plan['jobs']:
            work = validate_job(self.root, job)
            protocol = verify(work)
            if job['arm'] == 'shapley':
                self.assertFalse(protocol['shapley_gate']['passed'])
                self.assertEqual(protocol['prediction_gate_policy'], 'report-only')
            else:
                with np.load(work/job['arm']/'dataset.npz') as new, np.load(
                        self.previous/job['directory']/job['arm']/'dataset.npz') as old:
                    for key in old:
                        np.testing.assert_array_equal(new[key], old[key])
        for path, original in before.items():
            self.assertEqual(digest(path), original)

    def test_enforcement_excludes_failed_shapley_and_started_runs_cannot_replay(self):
        prepare(self.previous, self.root, 'retry-revision', 'enforce')
        plan = read(self.root/'plan.json')
        self.assertEqual(len(plan['jobs']), 6)
        self.assertEqual(len(plan['excluded_jobs']), 2)
        first = plan['jobs'][0]
        (self.previous/first['directory']/first['arm']/'training').mkdir()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'started policy'):
                prepare(self.previous, Path(tmp)/'new', 'revision', 'report-only')

    def test_module_commands_preflight_and_train_eight_runs(self):
        prepare(self.previous, self.root, 'retry-revision', 'report-only')
        processes, observers, smoke_calls = [], [], []
        def spawn(command, **kwargs):
            module = command[command.index('-m') + 1]
            root = Path(command[command.index('--root') + 1])
            arm = command[command.index('--arm') + 1]
            process = SimpleNamespace(pid=100+len(processes)+len(observers),
                                      root=root, arm=arm, returncode=None)
            process.poll = lambda: process.returncode
            if module == 'scripts.cql_antmaze_retry_5090.train':
                processes.append(process)
                (root/arm/'training').mkdir()
            else:
                self.assertEqual(module, 'scripts.cql_antmaze_5090.sync')
                observers.append(process)
            return process
        def smoke(command, **kwargs):
            self.assertEqual(command[1:3],
                             ['-m', 'scripts.cql_antmaze_retry_5090.train'])
            root = Path(command[command.index('--root') + 1])
            arm = command[command.index('--arm') + 1]
            (root/arm/'preflight').mkdir()
            write(root/arm/'preflight/status.json', dict(status='completed',
                  completed_updates=100, final_normalized_score=0.))
            write(root/arm/'preflight/runtime.json', dict(device='cuda:0'))
            smoke_calls.append(command)
        def sleep(seconds):
            for process in processes:
                process.returncode = 0
                write(process.root/process.arm/'training/status.json',
                      dict(status='completed', completed_updates=300000))
            for observer in observers:
                observer.returncode = 0
                work = observer.root/'wandb_sync'/observer.arm
                work.mkdir(parents=True, exist_ok=True)
                write(work/'full_completion_verification.json', dict(verified=True))
        prefix = 'scripts.cql_antmaze_retry_5090.queue.'
        with patch(prefix+'probe', return_value=dict(gpu_pids=list(range(6)),
                   free_gpu_mib=16000, free_host_mib=32000, utilization=50)), \
                patch(prefix+'subprocess.Popen', side_effect=spawn), \
                patch(prefix+'subprocess.run', side_effect=smoke), \
                patch(prefix+'time.sleep', side_effect=sleep):
            execute(self.root)
        self.assertEqual(len(processes), 8)
        self.assertEqual(len(smoke_calls), 8)
        self.assertEqual(read(self.root/'queue/status.json')['stage'], 'completed')


if __name__ == '__main__':
    unittest.main()
