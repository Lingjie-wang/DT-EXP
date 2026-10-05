"""Remove the ordinary DT gradient while preserving the audited C reference."""

import json
from pathlib import Path

def read_json(path):
    return json.loads(Path(path).read_text())


def preference_only_objective(diagnostic_dt_loss, preference_loss, weight):
    """Keep ordinary loss out of autograd; retain the matched preference scale."""
    if diagnostic_dt_loss.requires_grad:
        raise ValueError("DT diagnostics must be computed without autograd")
    if weight != 0.05:
        raise ValueError("Keep the matched reference preference coefficient")
    return weight * preference_loss


def verify_reference(config, provenance, reference_dir):
    root = Path(reference_dir)
    ref = read_json(root / 'config.json')
    status = read_json(root / 'status.json')
    old = read_json(root / 'provenance.json')
    if (ref['variant'] != 'c' or ref['preference_weight'] != 0.05
            or status['state'] != 'completed'
            or status['completed_updates'] != ref['update_steps']):
        raise ValueError('Expected a completed matched late-preference C')
    operational = {'name', 'group', 'variant', 'output_dir'}
    for key, expected in ref.items():
        actual = config.get(key)
        if isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
            expected, actual = tuple(expected), tuple(actual)
        if key not in operational and expected != actual:
            raise ValueError('Reference training setting differs: ' + key)
    for key in ('dataset_sha256', 'pairs_sha256', 'initial_model_sha256',
                'packages', 'attention_backend', 'fine_tuning'):
        if provenance[key] != old[key]:
            raise ValueError('Reference provenance differs: ' + key)
    for filename, expected in old['source_sha256'].items():
        if filename.endswith('/late_preference_dt.py'):
            continue
        if provenance['source_sha256'].get(filename) != expected:
            raise ValueError('Reference helper differs: ' + filename)
    return {
        'reference_run': read_json(root / 'wandb_run.json'),
        'reference_dir': str(root.resolve()),
        'reference_git_commit': old['git_commit'],
        'same_parent_model_optimizer_scheduler': True,
        'same_training_configuration_except_dt_objective': True,
        'dt_loss_weight': 0.0,
        'preference_loss_weight': 0.05,
        'optimizer_note': 'Parent AdamW moments are retained for the matched '
                          'ablation. No new DT-loss gradient; historical DT '
                          'momentum and ordinary AdamW weight decay remain.',
    }


def verify_initial(root, reference):
    for name in ('parent_verified.json', 'first_update.json'):
        if read_json(Path(root) / name) != read_json(Path(reference) / name):
            raise ValueError('Initial equivalence failed: ' + name)
    return {'same_restored_state': True, 'same_first_batch_pairs_dropout_losses': True}
