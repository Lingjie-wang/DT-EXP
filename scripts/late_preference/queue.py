"""Run audited GPU smokes, then three independent short fine-tuning children."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def verify_children(roots):
    """Require identical starting state, first sampled data/dropout and baseline."""
    reference = roots[0]
    first = read_json(reference / 'first_update.json')
    initial = read_json(reference / 'parent_verified.json')
    baseline = read_json(reference / 'evaluations/step000000.json')
    config = read_json(reference / 'config.json')
    for root in roots:
        current_config = read_json(root / 'config.json')
        for key, expected in config.items():
            if key not in {'name', 'variant', 'preference_weight', 'output_dir'}:
                if current_config[key] != expected:
                    raise ValueError(f'Child training setting differs: {key}')
        if read_json(root / 'parent_verified.json') != initial:
            raise ValueError('Initial checkpoint/optimizer/scheduler states differ')
        current = read_json(root / 'first_update.json')
        for key in ('ordinary_batch_sha256', 'pair_indices_sha256',
                    'before_forward_rng_sha256', 'dt_loss', 'positive_mse',
                    'negative_mse'):
            if current[key] != first[key]:
                raise ValueError(f'First update differs: {key}')
        if read_json(root / 'evaluations/step000000.json') != baseline:
            raise ValueError('Child baseline evaluations differ')
        status = read_json(root / 'status.json')
        if (status['state'] != 'completed'
                or status['completed_updates'] != current_config['update_steps']):
            raise ValueError('Child incomplete')
        for step in current_config['eval_steps']:
            evaluation = read_json(root / 'evaluations' / f'step{step:06d}.json')
            if len(evaluation['returns']) != current_config['eval_episodes']:
                raise ValueError('Incomplete evaluation')
    return {'same_parent_states': True, 'same_initial_batch_and_dropout': True,
            'same_baseline_returns': True, 'complete_evaluations': True}


def run_queue(args):
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / 'queue_manifest.json'
        if manifest_path.exists():
            raise RuntimeError('Existing campaign: inspect; never silently restart')
        commit = subprocess.check_output(
            ['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        if subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'],
                                   text=True).strip():
            raise RuntimeError('Freeze and publish a clean source before launch')
        gpu = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            text=True).strip()
        if gpu:
            raise RuntimeError('GPU already occupied; do not interfere')
        write_json(manifest_path, {
            'git_commit': commit, 'source': str(SOURCE),
            'parent_checkpoint': str(args.parent_checkpoint.resolve()),
            'order': ['smoke-dt', 'smoke-b', 'smoke-c',
                      'dt-seed0', 'b-seed0', 'c-seed0'],
            'primary_endpoint': 'normalized score at 5000 additional updates',
            'learning_rate': 0.0001,
        })

        def report(state, **extra):
            value = {'state': state, 'queue_pid': os.getpid(),
                     'updated_at_unix': time.time(), **extra}
            write_json(root / 'queue_status.json', value)
            print(json.dumps(value), flush=True)

        try:
            for smoke in (True, False):
                completed = []
                for arm in ('dt', 'b', 'c'):
                    label = ('smoke-' + arm) if smoke else (arm + '-seed0')
                    output = root / label
                    if output.exists():
                        raise RuntimeError('Refuse to overwrite ' + str(output))
                    report('smoke' if smoke else 'training', trial=label)
                    command = ['bash', str(SOURCE / 'scripts/late_preference/run.sh'),
                               '--output_dir', str(output), '--variant', arm,
                               '--preference_weight', '0' if arm == 'dt' else '0.05',
                               '--parent_checkpoint',
                               str(args.parent_checkpoint.resolve()),
                               '--name', 'LatePreference-' + label]
                    if smoke:
                        command += ['--validation_mode', 'true', '--update_steps', '3',
                                    '--eval_steps', '[0,3]', '--eval_episodes', '1',
                                    '--log_every', '1', '--wandb_mode', 'offline',
                                    '--group', 'LatePreference-Validation-20261005']
                    with (root / (label + '.log')).open('w', buffering=1) as log:
                        subprocess.run(command, cwd=SOURCE, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                    completed.append(output)
                    audit = verify_children(completed)
                    write_json(root / ('smoke_verified.json' if smoke
                                       else 'children_verified.json'), audit)
            summaries = {arm: read_json(root / (arm + '-seed0') / 'summary.json')
                         for arm in ('dt', 'b', 'c')}
            comparison = {
                'summaries': summaries,
                'final_b_minus_dt': (
                    summaries['b']['last_score'] - summaries['dt']['last_score']),
                'final_c_minus_dt': (
                    summaries['c']['last_score'] - summaries['dt']['last_score']),
                'final_c_minus_b': (
                    summaries['c']['last_score'] - summaries['b']['last_score']),
                'single_training_seed': True,
                'interpretation': 'Benefits after a 100k DT. Does not isolate activation '
                                  'timing against from-scratch C at equal total budget.',
            }
            write_json(root / 'comparison.json', comparison)
            report('completed', comparison=comparison)
        except BaseException as error:
            report('failed', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    run_queue(parser.parse_args())
