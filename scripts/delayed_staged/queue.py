"""Audited delayed-only parent followed by four matched short continuations."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
ARMS = ('dt', 'b', 'c', 'c_only')


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def verify_complete(root, parent=False):
    config, status = read_json(root / 'config.json'), read_json(root / 'status.json')
    if (config['reward_mode'] != 'delayed' or status['state'] != 'completed'
            or status['completed_updates'] != config['update_steps']):
        raise ValueError('Incomplete or non-delayed run: ' + str(root))
    audit = read_json(root / 'delayed_reward_audit.json')
    if not (audit['all_nonterminal_rewards_zero']
            and audit['all_training_rtgs_constant']):
        raise ValueError('Delayed reward audit is missing')
    batch = read_json(root / 'batch_rtg_verified.json')
    if not (batch['ordinary_valid_rtgs_constant']
            and batch['auxiliary_valid_rtgs_constant']):
        raise ValueError('Invalid delayed batch RTGs')
    if parent:
        if config['variant'] != 'parent_dt' or config['preference_weight'] != 0:
            raise ValueError('Parent must be ordinary DT without preference training')
        steps = list(range(config['eval_every'], config['update_steps'] + 1,
                           config['eval_every']))
        if config['update_steps'] not in steps:
            steps.append(config['update_steps'])
    else:
        steps = config['eval_steps']
    for step in steps:
        row = read_json(root / 'evaluations' / f'step{step:06d}.json')
        if len(row['returns']) != config['eval_episodes']:
            raise ValueError('Incomplete evaluation')
    if not (root / 'checkpoints' / f"step{config['update_steps']:06d}.pt").is_file():
        raise ValueError('Missing final checkpoint')


def verify_children(roots):
    first = roots[0]
    config = read_json(first / 'config.json')
    initial = read_json(first / 'first_update.json')
    for root in roots:
        verify_complete(root)
        current = read_json(root / 'config.json')
        for key, expected in config.items():
            if key not in {'name', 'variant', 'preference_weight', 'output_dir'}:
                if current[key] != expected:
                    raise ValueError('Child setting differs: ' + key)
        for name in ('parent_verified.json', 'delayed_reward_audit.json',
                     'batch_rtg_verified.json', 'evaluations/step000000.json'):
            if read_json(root / name) != read_json(first / name):
                raise ValueError('Child equivalence failed: ' + name)
        now = read_json(root / 'first_update.json')
        for key in ('ordinary_batch_sha256', 'pair_indices_sha256',
                    'before_forward_rng_sha256', 'dt_loss', 'positive_mse',
                    'negative_mse'):
            if now[key] != initial[key]:
                raise ValueError('First update differs: ' + key)
    return {'delayed_only': True, 'same_parent_states': True,
            'same_initial_batch_pairs_dropout': True, 'same_baseline_returns': True,
            'constant_delayed_rtgs_verified': True, 'complete_evaluations': True}


def run_queue(args):
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'queue_manifest.json').exists():
            raise RuntimeError('Existing campaign; do not silently restart')
        commit = subprocess.check_output(
            ['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        if subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'],
                                   text=True).strip():
            raise RuntimeError('Publish and freeze clean source first')
        if subprocess.check_output([
                'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                text=True).strip():
            raise RuntimeError('GPU occupied; do not interfere with other jobs')
        write_json(root / 'queue_manifest.json', {
            'git_commit': commit, 'source': str(SOURCE), 'reward_mode': 'delayed',
            'parent_updates': 100000, 'child_updates': 5000,
            'child_order': list(ARMS), 'training_seed': 0,
            'primary_endpoint': 'final additional update 5000',
            'correction': 'Previous LatePreference/PreferenceOnly campaigns used '
                          'original rewards and are outside the delayed-reward scope.',
        })

        def report(state, **extra):
            value = {'state': state, 'queue_pid': os.getpid(),
                     'updated_at_unix': time.time(), 'reward_mode': 'delayed', **extra}
            write_json(root / 'queue_status.json', value)
            print(json.dumps(value), flush=True)

        def execute(label, command):
            if (root / label).exists():
                raise RuntimeError('Refuse to overwrite ' + label)
            with (root / (label + '.log')).open('w', buffering=1) as log:
                subprocess.run(command, cwd=SOURCE, stdout=log,
                               stderr=subprocess.STDOUT, check=True)

        try:
            for smoke in (True, False):
                parent_name = 'smoke-parent' if smoke else 'parent-dt-seed0'
                parent = root / parent_name
                command = ['bash', str(SOURCE / 'scripts/delayed_staged/run_parent.sh'),
                           '--output_dir', str(parent),
                           '--name', 'DelayedStage-' + parent_name]
                if smoke:
                    command += ['--validation_mode', 'true', '--update_steps', '3',
                                '--eval_every', '3', '--eval_episodes', '1',
                                '--warmup_steps', '1', '--log_every', '1',
                                '--wandb_mode', 'offline',
                                '--group', 'DelayedStage-Validation-20261005']
                report('smoke_parent' if smoke else 'training_parent', trial=parent_name)
                execute(parent_name, command)
                verify_complete(parent, parent=True)
                completed = []
                parent_steps = 3 if smoke else 100000
                for arm in ARMS:
                    label = ('smoke-' + arm) if smoke else (arm + '-seed0')
                    output = root / label
                    report('smoke_child' if smoke else 'training_child', trial=label)
                    command = ['bash', str(SOURCE / 'scripts/delayed_staged/run.sh'),
                               '--output_dir', str(output), '--variant', arm,
                               '--preference_weight', '0' if arm == 'dt' else '0.05',
                               '--parent_checkpoint', str(parent / 'checkpoints' /
                                                          f'step{parent_steps:06d}.pt'),
                               '--name', 'DelayedStage-' + label]
                    if smoke:
                        command += ['--validation_mode', 'true', '--update_steps', '3',
                                    '--eval_steps', '[0,3]', '--eval_episodes', '1',
                                    '--warmup_steps', '1', '--log_every', '1',
                                    '--wandb_mode', 'offline',
                                    '--group', 'DelayedStage-Validation-20261005']
                    execute(label, command)
                    completed.append(output)
                    audit_name = ('smoke_verified.json' if smoke else
                                  'children_verified.json')
                    write_json(root / audit_name, verify_children(completed))
                if smoke and args.smoke_only:
                    report('smoke_completed')
                    return
            summaries = {arm: read_json(root / (arm + '-seed0') / 'summary.json')
                         for arm in ARMS}
            comparison = {
                'reward_mode': 'delayed',
                'parent': read_json(root / 'parent-dt-seed0/summary.json'),
                'summaries': summaries,
                'final_b_minus_dt': (summaries['b']['last_score']
                                     - summaries['dt']['last_score']),
                'final_c_minus_dt': (summaries['c']['last_score']
                                     - summaries['dt']['last_score']),
                'final_c_only_minus_dt': (summaries['c_only']['last_score']
                                         - summaries['dt']['last_score']),
                'final_c_only_minus_c': (summaries['c_only']['last_score']
                                        - summaries['c']['last_score']),
                'single_training_seed': True,
                'interpretation': 'Delayed-reward staged-preference ablation; '
                                  'does not establish overfitting as a causal '
                                  'mechanism.',
            }
            write_json(root / 'comparison.json', comparison)
            report('completed', comparison=comparison)
        except BaseException as error:
            report('failed', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    run_queue(parser.parse_args())
