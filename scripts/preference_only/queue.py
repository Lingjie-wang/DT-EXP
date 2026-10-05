"""Validate one C-only child and compare it with the completed matched campaign."""

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


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify_completed(root):
    config = read_json(root / 'config.json')
    status = read_json(root / 'status.json')
    if (status['state'] != 'completed'
            or status['completed_updates'] != config['update_steps']):
        raise ValueError('Incomplete run: ' + str(root))
    for step in config['eval_steps']:
        row = read_json(root / 'evaluations' / f'step{step:06d}.json')
        if len(row['returns']) != config['eval_episodes']:
            raise ValueError('Incomplete evaluation: ' + str(root))


def run_queue(args):
    root = args.output_root.resolve()
    reference = args.reference_campaign.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'queue_manifest.json').exists():
            raise RuntimeError('Existing campaign; inspect instead of restarting')
        commit = subprocess.check_output(
            ['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        if subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'],
                                   text=True).strip():
            raise RuntimeError('Publish and freeze clean source first')
        if subprocess.check_output([
                'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                text=True).strip():
            raise RuntimeError('GPU occupied; do not interfere with another job')
        if read_json(reference / 'queue_status.json')['state'] != 'completed':
            raise ValueError('Reference campaign is incomplete')
        for name in ('smoke-c', 'dt-seed0', 'b-seed0', 'c-seed0'):
            verify_completed(reference / name)
        write_json(root / 'queue_manifest.json', {
            'git_commit': commit, 'source': str(SOURCE),
            'parent_checkpoint': str(args.parent_checkpoint.resolve()),
            'reference_campaign': str(reference),
            'order': ['smoke-c-only', 'c-only-seed0'],
            'primary_endpoint': 'final score at 5000 additional updates',
            'objective': '0.05 * C preference loss; no new DT-loss gradient',
            'optimizer': 'preserve parent moments and step counters',
        })

        def report(state, **extra):
            value = {'state': state, 'queue_pid': os.getpid(),
                     'updated_at_unix': time.time(), **extra}
            write_json(root / 'queue_status.json', value)
            print(json.dumps(value), flush=True)

        try:
            for smoke in (True, False):
                label = 'smoke-c-only' if smoke else 'c-only-seed0'
                output = root / label
                old = reference / ('smoke-c' if smoke else 'c-seed0')
                if output.exists():
                    raise RuntimeError('Refusing to overwrite ' + str(output))
                report('smoke' if smoke else 'training', trial=label)
                command = ['bash', str(SOURCE / 'scripts/preference_only/run.sh'),
                           '--parent_checkpoint', str(args.parent_checkpoint.resolve()),
                           '--reference_dir', str(old), '--output_dir', str(output),
                           '--name', 'PreferenceOnly-' + label]
                if smoke:
                    command += ['--validation_mode', 'true', '--update_steps', '3',
                                '--eval_steps', '[0,3]', '--eval_episodes', '1',
                                '--log_every', '1', '--wandb_mode', 'offline',
                                '--group', 'PreferenceOnly-Validation-20261005']
                with (root / (label + '.log')).open('w', buffering=1) as log:
                    subprocess.run(command, cwd=SOURCE, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                verify_completed(output)
                for name in ('first_update.json', 'parent_verified.json',
                             'evaluations/step000000.json'):
                    if read_json(output / name) != read_json(old / name):
                        raise ValueError('Reference equivalence failed: ' + name)
                metrics = [json.loads(line) for line in
                           (output / 'metrics.jsonl').read_text().splitlines()]
                for row in metrics:
                    if 'train/total_loss' in row:
                        if (row['train/dt_loss_weight'] != 0
                                or row['train/total_loss']
                                != row['train/weighted_preference_loss']):
                            # Float32 multiplication may differ from Python double.
                            if (row['train/dt_loss_weight'] != 0 or abs(
                                    row['train/total_loss']
                                    - row['train/weighted_preference_loss']) > 1e-8):
                                raise ValueError('Unexpected training objective')
                write_json(root / ('smoke_verified.json' if smoke else
                                   'formal_verified.json'), {
                    'same_parent_model_optimizer_scheduler': True,
                    'same_first_batch_pairs_dropout_losses': True,
                    'same_baseline_evaluation': True,
                    'no_new_dt_loss_gradient': True,
                    'all_evaluations_completed': True,
                })
            summaries = {name: read_json(reference / (name + '-seed0') / 'summary.json')
                         for name in ('dt', 'b', 'c')}
            summaries['c_only'] = read_json(root / 'c-only-seed0/summary.json')
            final = summaries['c_only']['last_score']
            comparison = {
                'summaries': summaries,
                'c_only_minus_dt': final - summaries['dt']['last_score'],
                'c_only_minus_c': final - summaries['c']['last_score'],
                'c_only_minus_parent': summaries['c_only']['gain_from_parent'],
                'single_training_seed': True,
                'interpretation': 'Ablates the ordinary DT loss during fine-tuning. '
                                  'Does not establish overfitting as a causal '
                                  'mechanism.',
            }
            write_json(root / 'comparison.json', comparison)
            report('completed', comparison=comparison)
        except BaseException as error:
            report('failed', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-checkpoint', type=Path, required=True)
    parser.add_argument('--reference-campaign', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    run_queue(parser.parse_args())
